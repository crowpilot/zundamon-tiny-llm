"""llama2.c 形式（legacy v0）の読み書き。numpy だけで動く。

レイアウト: header(7 x int32) → token_embedding → rms_att(全層) → wq → wk → wv → wo
            → rms_ffn → w1 → w2 → w3 → rms_final → freq_cis_real → freq_cis_imag → [wcls]
vocab_size が負なら分類層(wcls)を別に持つ。Core2 の llm_core.c と同じ並び。
"""
import struct

import numpy as np

HEADER = ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len")
LAYER_KEYS = ("rms_att", "wq", "wk", "wv", "wo", "rms_ffn", "w1", "w2", "w3")


def rope_tables(seq_len, head_size, theta=10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, head_size, 2)[: head_size // 2].astype(np.float64) / head_size))
    t = np.arange(seq_len, dtype=np.float64)
    ang = np.outer(t, freqs)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def write_llama2c(path, cfg, w):
    """cfg: dict(HEADER), w: dict of numpy arrays.
    w["tok_emb"]  (vocab, dim)
    w["rms_att"]  list[n_layers] of (dim,)   ... 他の LAYER_KEYS も同様に層ごとのリスト
    w["rms_final"] (dim,)
    w["wcls"] (vocab, dim) or None (= tok_emb と共有)
    重みは nn.Linear.weight の形 (out, in) のまま渡す。"""
    shared = w.get("wcls") is None
    head_size = cfg["dim"] // cfg["n_heads"]
    vals = [cfg[k] for k in HEADER]
    if not shared:
        vals[5] = -vals[5]
    with open(path, "wb") as f:
        f.write(struct.pack("7i", *vals))

        def put(a):
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())

        put(w["tok_emb"])
        for k in LAYER_KEYS:
            for layer in w[k]:
                put(layer)
        put(w["rms_final"])
        cos, sin = rope_tables(cfg["seq_len"], head_size)
        put(cos)
        put(sin)
        if not shared:
            put(w["wcls"])


def read_llama2c(path):
    """write_llama2c の逆。検証用。"""
    data = open(path, "rb").read()
    vals = list(struct.unpack("7i", data[:28]))
    cfg = dict(zip(HEADER, vals))
    shared = cfg["vocab_size"] > 0
    cfg["vocab_size"] = abs(cfg["vocab_size"])
    arr = np.frombuffer(data[28:], dtype=np.float32)
    d, hd, L, H, KV, V, S = (cfg[k] for k in HEADER)
    head_size = d // H
    kv_dim = KV * head_size
    shapes = {
        "rms_att": (d,), "wq": (d, d), "wk": (kv_dim, d), "wv": (kv_dim, d), "wo": (d, d),
        "rms_ffn": (d,), "w1": (hd, d), "w2": (d, hd), "w3": (hd, d),
    }
    off = 0

    def take(shape):
        nonlocal off
        n = int(np.prod(shape))
        out = arr[off:off + n].reshape(shape)
        off += n
        return out

    w = {"tok_emb": take((V, d))}
    for k in LAYER_KEYS:
        w[k] = [take(shapes[k]) for _ in range(L)]
    w["rms_final"] = take((d,))
    take((S, head_size // 2))
    take((S, head_size // 2))
    w["wcls"] = None if shared else take((V, d))
    return cfg, w


def write_tokenizer(path, pieces, scores):
    """llama2.c の tokenizer.bin 形式。pieces は表示用の文字列（▁→空白置換済み）。"""
    bs = [p.encode("utf-8") for p in pieces]
    max_len = max(len(b) for b in bs)
    with open(path, "wb") as f:
        f.write(struct.pack("I", max_len))
        for b, s in zip(bs, scores):
            f.write(struct.pack("fI", float(s), len(b)))
            f.write(b)


def numpy_forward(cfg, w, tokens, act_gs=None):
    """書き出したファイルの検証用の参照実装（Core2 の C 実装と同じ計算）。logits (T, vocab) を返す。
    act_gs を与えると、行列積の入力を int8 に丸める Q8 推論の誤差も再現する。"""
    d, H, KV = cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"]
    hs = d // H
    T = len(tokens)
    cos, sin = rope_tables(cfg["seq_len"], hs)
    aq = (lambda a: fake_quant_act(a, act_gs)) if act_gs else (lambda a: a)

    def rms(x, g):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * g

    def rope(t):
        tr = t.reshape(T, t.shape[1], hs // 2, 2)
        c, s = cos[:T][:, None, :], sin[:T][:, None, :]
        r = tr[..., 0] * c - tr[..., 1] * s
        i = tr[..., 0] * s + tr[..., 1] * c
        return np.stack([r, i], -1).reshape(t.shape)

    x = w["tok_emb"][np.asarray(tokens)].astype(np.float64)
    mask = np.triu(np.full((T, T), -1e9), 1)[None]
    for l in range(cfg["n_layers"]):
        h = aq(rms(x, w["rms_att"][l]))
        q = rope((h @ w["wq"][l].T).reshape(T, H, hs))
        k = rope((h @ w["wk"][l].T).reshape(T, KV, hs))
        v = (h @ w["wv"][l].T).reshape(T, KV, hs)
        k = np.repeat(k, H // KV, axis=1)
        v = np.repeat(v, H // KV, axis=1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(hs) + mask
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        x = x + aq(np.einsum("hts,shd->thd", att, v).reshape(T, d)) @ w["wo"][l].T
        h = aq(rms(x, w["rms_ffn"][l]))
        a1, a3 = h @ w["w1"][l].T, h @ w["w3"][l].T
        x = x + aq((a1 / (1 + np.exp(-a1))) * a3) @ w["w2"][l].T
    x = aq(rms(x, w["rms_final"]))
    wcls = w["tok_emb"] if w["wcls"] is None else w["wcls"]
    return x @ wcls.T


# ------------------------------------------------------------------ int8 (Q8) 形式
# 重みを gs 個ずつのグループに分け「int8 + グループごとの倍率(float)」で持つ。Core2 の llm_core.c と同じ並び:
#   int32 magic 0x38515454('TTQ8') , int32 version=1, 7 x int32 config, int32 gs
#   fp32: rms_att(全層) rms_ffn(全層) rms_final
#   Q8:   tok_emb（分類層と共有）、続いて wq wk wv wo w1 w2 w3 の順に、各々 全層ぶん
#   Q8 1テンソル = int8[n] のあとに float[n/gs]
Q8_MAGIC = 0x38515454
Q8_KEYS = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")


def quantize_q8(w, gs):
    a = np.asarray(w, dtype=np.float32).reshape(-1, gs)
    scale = np.abs(a).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.round(a / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q.reshape(-1), scale.astype(np.float32)


def dequantize_q8(q, s, shape, gs):
    return (q.reshape(-1, gs).astype(np.float32) * s[:, None]).reshape(shape)


def write_q8(path, cfg, w, gs=32):
    if w.get("wcls") is not None:
        raise ValueError("Q8 形式は分類層共有モデルのみ対応")
    for k in ("dim", "hidden_dim"):
        if cfg[k] % gs:
            raise ValueError(f"{k}={cfg[k]} が gs={gs} で割り切れません（--multiple_of を {gs} の倍数に）")
    with open(path, "wb") as f:
        f.write(struct.pack("ii", Q8_MAGIC, 1))
        f.write(struct.pack("7i", *[cfg[k] for k in HEADER]))
        f.write(struct.pack("i", gs))

        def put(a):
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())

        for layer in w["rms_att"]:
            put(layer)
        for layer in w["rms_ffn"]:
            put(layer)
        put(w["rms_final"])

        def putq(a):
            q, s = quantize_q8(a, gs)
            f.write(q.tobytes())
            f.write(s.tobytes())

        putq(w["tok_emb"])
        for k in Q8_KEYS:
            for layer in w[k]:
                putq(layer)


def read_q8(path):
    """Q8 ファイルを読み、量子化後の値に戻した（dequantize した）重みを返す。検証用。"""
    data = open(path, "rb").read()
    magic, ver = struct.unpack("ii", data[:8])
    assert magic == Q8_MAGIC and ver == 1
    cfg = dict(zip(HEADER, struct.unpack("7i", data[8:36])))
    gs = struct.unpack("i", data[36:40])[0]
    off = 40
    d, hd, L, H, KV, V = (cfg[k] for k in HEADER[:6])
    kv_dim = KV * (d // H)

    def take_f(n):
        nonlocal off
        a = np.frombuffer(data, np.float32, n, off)
        off += 4 * n
        return a

    def take_q(shape):
        nonlocal off
        n = int(np.prod(shape))
        q = np.frombuffer(data, np.int8, n, off)
        off += n
        s = take_f(n // gs)
        return dequantize_q8(q, s, shape, gs)

    w = {"rms_att": [take_f(d) for _ in range(L)], "rms_ffn": [take_f(d) for _ in range(L)], "rms_final": take_f(d)}
    w["tok_emb"] = take_q((V, d))
    shapes = {"wq": (d, d), "wk": (kv_dim, d), "wv": (kv_dim, d), "wo": (d, d), "w1": (hd, d), "w2": (d, hd), "w3": (hd, d)}
    for k in Q8_KEYS:
        w[k] = [take_q(shapes[k]) for _ in range(L)]
    w["wcls"] = None
    assert off == len(data), (off, len(data))
    return cfg, w, gs


def fake_quant_act(x, gs):
    """行列積の入力（活性）を gs 個ずつ int8 に丸める（Core2 の計算と同じ誤差を再現）。"""
    shp = x.shape
    a = x.reshape(-1, gs)
    s = np.abs(a).max(axis=1) / 127.0
    s[s == 0] = 1.0
    return (np.round(a / s[:, None]).clip(-127, 127) * s[:, None]).reshape(shp)
