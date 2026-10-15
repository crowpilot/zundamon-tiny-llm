"""学習済みモデルを量子化して、品質（val loss）とサイズを比べる。numpy のみ（Core2 の C 実装と同じ計算）。

  python quant_eval.py out/mix-tiny            → out/mix-tiny/model_q8.bin と quant.json

比べるもの
  fp32 : そのまま（llama2.c 形式）
  int8 : 重みも行列積の入力も 32 個ずつ int8 に丸める（Core2 の Q8 推論そのもの）
  int4 : 重みだけ 32 個ずつ 4bit に丸める（参考。C 実装はまだ無い）
同じ問いかけの greedy 生成も fp32 / int8 で並べて保存する。
"""
import argparse
import json
import os

import numpy as np
import sentencepiece as spm

from llama2c_format import numpy_forward, read_llama2c, read_q8, write_q8


def quant_weights_bits(w, bits, gs):
    """重みだけ bits ビットに丸めた（戻した）コピー。"""
    qmax = 2 ** (bits - 1) - 1

    def q(a):
        a = np.asarray(a, np.float32)
        g = a.reshape(-1, gs)
        s = np.abs(g).max(1) / qmax
        s[s == 0] = 1
        return (np.round(g / s[:, None]).clip(-qmax, qmax) * s[:, None]).reshape(a.shape)

    out = dict(w)
    out["tok_emb"] = q(w["tok_emb"])
    for k in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        out[k] = [q(x) for x in w[k]]
    return out


def val_loss(cfg, w, windows, act_gs=None):
    tot, n = 0.0, 0
    for win in windows:
        logits = numpy_forward(cfg, w, win[:-1], act_gs=act_gs)
        logits = logits - logits.max(-1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        tot += -logp[np.arange(len(win) - 1), win[1:]].sum()
        n += len(win) - 1
    return float(tot / n)


def greedy(cfg, w, ids, n, stop, act_gs=None):
    ids = list(ids)
    start = len(ids)
    for _ in range(n):
        nxt = int(numpy_forward(cfg, w, ids[-cfg["seq_len"]:], act_gs=act_gs)[-1].argmax())
        if nxt in stop:
            break
        ids.append(nxt)
    return ids[start:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", default=None, help="val.bin のあるディレクトリ（既定: run.json の data）")
    ap.add_argument("--windows", type=int, default=24, help="評価に使う val の区間数")
    ap.add_argument("--win_len", type=int, default=128)
    ap.add_argument("--gs", type=int, default=32)
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--gen", type=int, default=40)
    args = ap.parse_args()

    run = json.load(open(os.path.join(args.run_dir, "run.json")))
    data_dir = args.data or run["args"]["data"]
    cfg, w = read_llama2c(os.path.join(args.run_dir, "model.bin"))
    q8_path = os.path.join(args.run_dir, "model_q8.bin")
    write_q8(q8_path, cfg, w, args.gs)
    cq, wq, gs = read_q8(q8_path)

    val = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16).astype(np.int64)
    if len(val) < args.win_len * 2:
        val = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16).astype(np.int64)[-200000:]
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(val) - args.win_len - 1, args.windows)
    windows = [val[s:s + args.win_len + 1] for s in starts]

    res = {"run": run["run"], "params": run["params"], "config": run["config"],
           "size_fp32": os.path.getsize(os.path.join(args.run_dir, "model.bin")), "size_q8": os.path.getsize(q8_path)}
    res["loss_fp32"] = val_loss(cfg, w, windows)
    res["loss_int8"] = val_loss(cq, wq, windows, act_gs=gs)
    res["loss_int4w"] = val_loss(cfg, quant_weights_bits(w, 4, args.gs), windows)
    print(f"{run['run']}: fp32 {res['loss_fp32']:.4f} / int8 {res['loss_int8']:.4f} / int4(重みのみ) {res['loss_int4w']:.4f}"
          f"  size {res['size_fp32']/1e6:.2f}MB → {res['size_q8']/1e6:.2f}MB")

    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.run_dir, "tok.model"))
    stop = {sp.bos_id(), sp.eos_id()}
    gens = []
    for line in open(args.prompts, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = "" if line.strip() == "<BOS>" else line
        ids = [sp.bos_id()] + sp.encode(p)
        g32 = sp.decode(greedy(cfg, w, ids, args.gen, stop))
        g8 = sp.decode(greedy(cq, wq, ids, args.gen, stop, act_gs=gs))
        gens.append({"prompt": p, "fp32": g32, "int8": g8})
    res["generations"] = gens
    json.dump(res, open(os.path.join(args.run_dir, "quant.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
