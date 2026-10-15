"""極小日本語LLMの学習 ＋「育っていく様子」の記録（途中保存・再開・区間実行に対応）

  python train.py --preset tiny --device cpu                  # 最初から最後まで
  python train.py --preset tiny --smoke --device cpu          # 動作確認（数百ステップ）
  python train.py ... --segment_seconds 600                   # 10分で区切って保存し exit 10
  python train.py ... --resume                                # 確定 checkpoint から続行
  python train.py ... --stop_file /path/STOP --events ev.jsonl  # runner からの協調停止・数値イベント

終了コード: 0=完了 10=区間終了 11=停止要求で保存して終了 2=checkpoint/設定の不整合 1=その他の失敗
  完了(0)は、最終 step の成長記録・最終 bin・model.bin・last.pt がそろい、completed checkpoint を確定したときだけ。
  区間の期限（--segment_seconds, 単調時計）と停止要求は、学習更新の境界だけでなく評価・生成（1トークンごと）でも確認し、
  途中なら記録を書かずに保存して終了する（その step の成長記録は再開後に作り直す）。

チェックポイントは対数間隔（序盤ほど細かく）で保存し、毎回同じ問いかけで生成した結果を
out/<run>/growth.jsonl に追記する。report.py でタイムラインHTMLにできる。
各チェックポイントは llama2.c 形式（Core2 にそのまま焼ける）で out/<run>/ckpt/ に出る。
"""
import argparse
import json
import math
import os
import shutil
import signal
import sys
import time

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ckpt import (CKPT_FORMAT, CheckpointError, hash_obj, latest, load_checkpoint, save_checkpoint,
                  sha256_file, verify_data, write_json_atomic)
from llama2c_format import numpy_forward, read_llama2c, rope_tables, write_llama2c

PRESETS = {
    "nano":  dict(dim=48,  n_layers=4, n_heads=6, n_kv_heads=3),   # 約10万（語彙1024時）
    "tiny":  dict(dim=64,  n_layers=5, n_heads=8, n_kv_heads=4),   # stories260K と同じ骨格
    "small": dict(dim=96,  n_layers=6, n_heads=6, n_kv_heads=3),   # 約60万
    "base":  dict(dim=128, n_layers=8, n_heads=8, n_kv_heads=4),   # 約150万
}


# ------------------------------------------------------------------ model
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def apply_rope(x, cos, sin):
    # x: (B,T,h,hs)。隣り合う2要素を1組として回転（llama2.c / Core2 実装と同じ規約）
    xr = x.float().reshape(*x.shape[:-1], -1, 2)
    a, b = xr[..., 0], xr[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, dim, hidden, n_heads, n_kv_heads):
        super().__init__()
        self.H, self.KV, self.hs = n_heads, n_kv_heads, dim // n_heads
        self.attention_norm = RMSNorm(dim)
        self.wq = nn.Linear(dim, n_heads * self.hs, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.hs, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.hs, bias=False)
        self.wo = nn.Linear(n_heads * self.hs, dim, bias=False)
        self.ffn_norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        h = self.attention_norm(x)
        q = apply_rope(self.wq(h).view(B, T, self.H, self.hs), cos, sin)
        k = apply_rope(self.wk(h).view(B, T, self.KV, self.hs), cos, sin)
        v = self.wv(h).view(B, T, self.KV, self.hs)
        rep = self.H // self.KV
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.wo(y.transpose(1, 2).reshape(B, T, -1))
        h = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class TinyLlama(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["dim"]
        self.tok_embeddings = nn.Embedding(cfg["vocab_size"], d)
        self.layers = nn.ModuleList(Block(d, cfg["hidden_dim"], cfg["n_heads"], cfg["n_kv_heads"])
                                    for _ in range(cfg["n_layers"]))
        self.norm = RMSNorm(d)
        self.output = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.output.weight = self.tok_embeddings.weight   # 重み共有
        cos, sin = rope_tables(cfg["seq_len"], d // cfg["n_heads"])
        self.register_buffer("cos", torch.from_numpy(cos), persistent=False)
        self.register_buffer("sin", torch.from_numpy(sin), persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("w3.weight") or n.endswith("wo.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg["n_layers"]))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, self.cos[:T], self.sin[:T])
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss


def hidden_dim_for(dim, multiple_of=4):
    h = int(2 * (4 * dim) / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def export_numpy(model):
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    L = model.cfg["n_layers"]

    def per_layer(name):
        return [sd[f"layers.{i}.{name}"] for i in range(L)]

    return {
        "tok_emb": sd["tok_embeddings.weight"],
        "rms_att": per_layer("attention_norm.weight"),
        "wq": per_layer("wq.weight"), "wk": per_layer("wk.weight"),
        "wv": per_layer("wv.weight"), "wo": per_layer("wo.weight"),
        "rms_ffn": per_layer("ffn_norm.weight"),
        "w1": per_layer("w1.weight"), "w2": per_layer("w2.weight"), "w3": per_layer("w3.weight"),
        "rms_final": sd["norm.weight"], "wcls": None,
    }


def header_cfg(cfg):
    return {k: cfg[k] for k in ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len")}


# ------------------------------------------------------------------ growth log
def piece_text(sp, i):
    if i == sp.bos_id():
        return "<BOS>"
    if i == sp.eos_id():
        return "<EOS>"
    return sp.id_to_piece(i).replace("▁", " ")


class Interrupted(Exception):
    """評価・生成の途中で停止要求や区間期限が来たときに投げる（その step の成長記録は書かない）"""


@torch.no_grad()
def generate(model, sp, prompt_ids, max_new, temperature, gen, n_cand=0, cand_steps=10, check=None):
    ids = list(prompt_ids)
    cands = []
    for step in range(max_new):
        if check is not None and check():   # 1トークンごとに停止要求・区間期限を確認
            raise Interrupted
        x = torch.tensor([ids[-model.cfg["seq_len"]:]], device=model.cos.device)
        logits = model(x)[0][0, -1].float()
        probs = F.softmax(logits, -1)
        if n_cand and step < cand_steps:
            p, ix = probs.topk(n_cand)
            cands.append([[piece_text(sp, int(j)), round(float(q), 4)] for q, j in zip(p, ix)])
        if temperature == 0:
            nxt = int(probs.argmax())
        else:
            nxt = int(torch.multinomial(F.softmax(logits / temperature, -1).cpu(), 1, generator=gen))
        if nxt in (sp.bos_id(), sp.eos_id()):
            break
        ids.append(nxt)
    return sp.decode(ids[len(prompt_ids):]), cands


def load_prompts(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            continue
        if line.strip() == "<BOS>":
            out.append("")
        elif line.strip():
            out.append(line)
    return out


def checkpoint_steps(max_iters, per_decade):
    s = {0, max_iters}
    k = 0
    while True:
        v = int(round(10 ** (k / per_decade)))
        if v >= max_iters:
            break
        s.add(v)
        k += 1
    return sorted(s)


# ------------------------------------------------------------------ run state helpers
EXIT_COMPLETED, EXIT_ERROR, EXIT_SEGMENT_END, EXIT_STOP = 0, 1, 10, 11

# 検証用: 成長記録の途中で停止要求／区間期限が来た状況を再現する（例 TINYJA_YIELD_IN_GROWTH=deadline:16）
_HOOK = os.environ.get("TINYJA_YIELD_IN_GROWTH", "")
HOOK_KIND, HOOK_STEP = (_HOOK.split(":")[0], int(_HOOK.split(":")[1])) if _HOOK else (None, None)
_FAULT = os.environ.get("TINYJA_FAULT", "")


class Events:
    """runner 向けの機械可読イベント（JSON Lines）。数値と識別子だけを書き、プロンプト・生成文は書かない。"""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8") if path else None

    def emit(self, typ, **kw):
        if not self.f:
            return
        self.f.write(json.dumps({"type": typ, "time": round(time.time(), 3), **kw}, ensure_ascii=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())


class StopFlag:
    """協調停止要求: SIGTERM/SIGINT か、--stop_file の出現。次の安全な境界で保存して終了する。"""

    def __init__(self, stop_file):
        self.stop_file = stop_file
        self.signalled = None
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, frame):
        self.signalled = signal.Signals(signum).name

    def requested(self):
        if self.signalled:
            return self.signalled
        if self.stop_file and os.path.exists(self.stop_file):
            return "stop_file"
        return None


def reconcile_history(path, keep):
    """再開時: 確定 step より先の履歴行を取り除き、取り除いた行は退避ファイルに残す（追跡用）。"""
    if not os.path.exists(path):
        return 0
    kept, dropped = [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:   # 書きかけの行
            dropped.append(line)
            continue
        (kept if keep(rec) else dropped).append(line if line.endswith("\n") else line + "\n")
    if dropped:
        with open(f"{path}.discarded-{int(time.time())}", "w", encoding="utf-8") as f:
            f.writelines(dropped)
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(kept)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    return len(dropped)


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def recipe_hash(files):
    here = os.path.dirname(os.path.abspath(__file__))
    return hash_obj({f: sha256_file(os.path.join(here, f)) for f in files})


RECIPE_FILES = ("train.py", "ckpt.py", "llama2c_format.py")


# ------------------------------------------------------------------ main
def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--run", default=None, help="実行名（既定: preset名）")
    ap.add_argument("--preset", default="tiny", choices=PRESETS)
    ap.add_argument("--dim", type=int)
    ap.add_argument("--n_layers", type=int)
    ap.add_argument("--n_heads", type=int)
    ap.add_argument("--n_kv_heads", type=int)
    ap.add_argument("--multiple_of", type=int, default=32, help="FFN幅をこの倍数に（int8量子化のグループ32で割り切れるように）")
    ap.add_argument("--seq_len", type=int, default=256, help="Core2側は256トークンまで")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=float, default=10, help="学習データを何周するか（--max_iters 未指定時に使う）")
    ap.add_argument("--max_iters", type=int, default=None, help="指定するとエポック数より優先")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--ckpt_per_decade", type=int, default=5, help="ステップ数が10倍になる間に何回記録するか")
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--n_cand", type=int, default=5, help="「迷い」記録で保存する候補数")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="fp16 混合精度（V100向け。小モデルでは効果小）")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="数百ステップの動作確認")
    # 再開・区間実行・協調停止（外側の runner 用）
    ap.add_argument("--resume", action="store_true", help="<out>/<run>/state の確定 checkpoint から続行（計画が一致しなければ拒否）")
    ap.add_argument("--segment_steps", type=int, default=0, help="この区間で実行する更新回数の上限（0=無制限）")
    ap.add_argument("--segment_seconds", type=float, default=0, help="この区間の実行時間の上限（秒、0=無制限）")
    ap.add_argument("--save_every_steps", type=int, default=0, help="区間内の定期保存（0=区間終了時のみ）")
    ap.add_argument("--stop_file", default=None, help="このファイルが現れたら次の境界で保存して終了（exit 11）")
    ap.add_argument("--events", default=None, help="機械可読イベント(JSON Lines)の出力先")
    ap.add_argument("--keep_checkpoints", type=int, default=2)
    ap.add_argument("--a_weight", type=float, default=1.0,
                    help="Ａ（相手）側トークンの損失の重み。0=ずんだもん（Ｂ）の台詞だけ学ぶ。<data>/train_spk.bin が必要")
    ap.add_argument("--print_samples", action="store_true", help="生成文も stdout に出す（既定は数値のみ）")
    args = ap.parse_args(argv)
    if args.smoke:
        args.max_iters, args.batch_size, args.eval_iters, args.log_interval = 300, 16, 5, 20
        args.warmup = 30
        args.seq_len = min(args.seq_len, 64)
    return args


def make_plan(args, cfg, train_len):
    """学習全体の計画。区間が変わっても変えない値だけを入れる（再開時はこれが一致しなければ拒否）。"""
    tokens_per_iter = args.batch_size * args.seq_len
    max_iters = args.max_iters
    if max_iters is None:
        max_iters = max(100, math.ceil(args.epochs * train_len / tokens_per_iter))
    warmup = min(args.warmup, max(10, max_iters // 10))
    return {"cfg": cfg, "batch_size": args.batch_size, "seq_len": args.seq_len, "max_iters": max_iters,
            "warmup": warmup, "lr": args.lr, "min_lr": args.min_lr, "weight_decay": args.weight_decay,
            "betas": [0.9, 0.99], "grad_clip": 1.0, "dropout": args.dropout, "seed": args.seed, "amp": bool(args.amp),
            "eval_iters": args.eval_iters, "ckpt_per_decade": args.ckpt_per_decade,
            "max_new": args.max_new, "n_cand": args.n_cand,
            **({"a_weight": args.a_weight} if args.a_weight != 1.0 else {})}   # 既定値のときは計画に入れない（既存の計画と同じ hash）


def main(argv=None):
    args = build_args(argv)
    run = args.run or (args.preset + ("-smoke" if args.smoke else ""))
    out = os.path.join(args.out, run)
    state_dir = os.path.join(out, "state")
    os.makedirs(os.path.join(out, "ckpt"), exist_ok=True)
    events = Events(args.events)
    stop = StopFlag(args.stop_file)
    t_proc = time.time()

    # ---- データ（manifest で検証）
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.data, "tok.model"))
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    train_data = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")
    data_hash = verify_data(args.data, train_data, val_data)
    spk_path = lambda n: os.path.join(args.data, n)   # noqa: E731
    train_spk = np.memmap(spk_path("train_spk.bin"), dtype=np.uint8, mode="r") if os.path.exists(spk_path("train_spk.bin")) else None
    val_spk = np.memmap(spk_path("val_spk.bin"), dtype=np.uint8, mode="r") if os.path.exists(spk_path("val_spk.bin")) else None
    if args.a_weight != 1.0 and (train_spk is None or len(train_spk) != len(train_data)):
        raise SystemExit("--a_weight には話者の印 train_spk.bin が必要です（prepare.py で作り直してください）")
    if len(train_data) < args.seq_len * 4:
        raise SystemExit(f"学習データが短すぎます（{len(train_data)}トークン）。--seq_len を下げるかデータを増やしてください")
    if len(val_data) < args.seq_len + 2:
        print("注意: val データが seq_len より短いので train の一部で代用します")
        val_data = train_data

    cfg = dict(PRESETS[args.preset])
    for k in ("dim", "n_layers", "n_heads", "n_kv_heads"):
        if getattr(args, k):
            cfg[k] = getattr(args, k)
    cfg["hidden_dim"] = hidden_dim_for(cfg["dim"], args.multiple_of)
    cfg["vocab_size"] = meta["vocab_size"]
    cfg["seq_len"] = args.seq_len
    assert cfg["dim"] % cfg["n_heads"] == 0 and cfg["n_heads"] % cfg["n_kv_heads"] == 0
    assert (cfg["dim"] // cfg["n_heads"]) % 2 == 0, "head_size は偶数にする（RoPE）"
    plan = make_plan(args, cfg, len(train_data))
    prompts = load_prompts(args.prompts)
    ids = {"plan_hash": hash_obj(plan), "data_hash": data_hash, "recipe_hash": recipe_hash(RECIPE_FILES),
           "prompts_hash": hash_obj(prompts)}
    max_iters, tokens_per_iter = plan["max_iters"], plan["batch_size"] * plan["seq_len"]

    dev = args.device
    torch.manual_seed(plan["seed"])
    model = TinyLlama(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": plan["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=plan["lr"], betas=tuple(plan["betas"]), fused=dev.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=plan["amp"] and dev.startswith("cuda"))
    train_rng = np.random.default_rng([plan["seed"], 1])   # 学習バッチ専用（評価では使わない）
    step, running = 0, None

    growth_path, loss_path = os.path.join(out, "growth.jsonl"), os.path.join(out, "loss.jsonl")
    if args.resume:
        payload, manifest, name = load_checkpoint(state_dir, map_location="cpu")
        bad = [k for k in ("plan_hash", "data_hash", "recipe_hash", "prompts_hash") if manifest.get(k) != ids[k]]
        if bad:
            diff = [k for k in plan if payload["plan"].get(k) != plan.get(k)] if "plan_hash" in bad else []
            raise CheckpointError(f"checkpoint と現在の設定が一致しません: {bad} {('計画の差: ' + str(diff)) if diff else ''}")
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        train_rng.bit_generator.state = payload["rng"]["numpy_train"]
        torch.set_rng_state(payload["rng"]["torch_cpu"])
        if payload["rng"]["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
        step, running = int(payload["step"]), payload["running_loss"]
        n1 = reconcile_history(growth_path, lambda r: r.get("step", -1) <= step)
        n2 = reconcile_history(loss_path, lambda r: r.get("step", -1) < step)
        run_info = json.load(open(os.path.join(out, "run.json"), encoding="utf-8"))
        run_info.setdefault("segments", []).append({"resumed_from": name, "step": step, "time": time.time(),
                                                    "history_discarded": n1 + n2})
        write_json_atomic(os.path.join(out, "run.json"), run_info)
        print(f"resume: {name} step={step}/{max_iters}（履歴から {n1 + n2} 行を退避）")
        events.emit("resume", checkpoint=name, step=step, max_iters=max_iters, history_discarded=n1 + n2)
    else:
        if latest(state_dir) is not None:
            raise CheckpointError(f"{state_dir} に確定 checkpoint があります。続けるなら --resume、やり直すなら run を別名に")
        for p in (growth_path, loss_path):
            open(p, "w").close()
        shutil.copy(os.path.join(args.data, "tok.bin"), os.path.join(out, "tok.bin"))
        shutil.copy(os.path.join(args.data, "tok.model"), os.path.join(out, "tok.model"))
        write_json_atomic(os.path.join(out, "run.json"),
                          {"run": run, "params": n_params, "config": cfg, "plan": plan, "args": vars(args),
                           "data": meta, **ids, "segments": [{"started": time.time(), "step": 0}]})
        events.emit("start", step=0, max_iters=max_iters, params=n_params)

    saved_at = step if args.resume else None   # 確定 checkpoint がある step
    print(f"run={run} device={dev} params={n_params:,} step={step}/{max_iters} config={cfg}")
    print(f"train tokens={len(train_data):,} tokens/iter={tokens_per_iter:,} warmup={plan['warmup']}")
    fwd = torch.compile(model) if args.compile else model
    have_growth = set()
    if os.path.exists(growth_path):
        have_growth = {json.loads(l)["step"] for l in open(growth_path, encoding="utf-8") if l.strip()}

    def get_batch(data, rng, bs, spk=None):
        ix = rng.integers(0, len(data) - plan["seq_len"] - 1, bs)
        x = torch.from_numpy(np.stack([data[i:i + plan["seq_len"]].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + plan["seq_len"]].astype(np.int64) for i in ix]))
        if spk is None:
            return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        s = torch.from_numpy(np.stack([spk[i + 1:i + 1 + plan["seq_len"]].astype(np.int64) for i in ix]))   # 予測先トークンの話者
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), s.to(dev, non_blocking=True)

    a_weight = plan.get("a_weight", 1.0)

    def token_losses(x, y):
        logits, _ = fwd(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), reduction="none")

    def lr_at(it):
        if it < plan["warmup"]:
            return plan["lr"] * (it + 1) / plan["warmup"]
        r = (it - plan["warmup"]) / max(1, max_iters - plan["warmup"])
        return plan["min_lr"] + 0.5 * (1 + math.cos(math.pi * min(1.0, r))) * (plan["lr"] - plan["min_lr"])

    seg_start_step, seg_t0 = step, time.monotonic()
    cur = {"recording": None}

    def yield_reason():
        """区間を譲る理由: ("stop", 理由) / ("segment", 理由) / None。学習更新・評価・生成のどこでも同じ判定を使う。"""
        why = stop.requested()
        if why:
            return ("stop", why)
        if HOOK_KIND and (cur["recording"] == HOOK_STEP or cur.get("hook_fired")):
            cur["hook_fired"] = True   # 一度起きたら、このプロセスの間は譲り続ける（本物の停止要求・期限と同じ振る舞い）
            return ("stop" if HOOK_KIND == "stop" else "segment", "test_hook")
        if args.segment_seconds and time.monotonic() - seg_t0 >= args.segment_seconds:
            return ("segment", "seconds")
        if args.segment_steps and step - seg_start_step >= args.segment_steps:
            return ("segment", "steps")
        return None

    @torch.no_grad()
    def estimate(data, at_step):
        rng = np.random.default_rng([plan["seed"], 2, at_step])   # 評価専用の乱数（学習の乱数列を変えない）
        model.eval()
        losses, b_sum, b_n = [], 0.0, 0
        spk = val_spk if (data is val_data and val_spk is not None and len(val_spk) == len(val_data)) else None
        for _ in range(plan["eval_iters"]):
            if yield_reason():
                model.train()
                raise Interrupted
            batch = get_batch(data, rng, plan["batch_size"], spk)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                if spk is None:
                    losses.append(fwd(batch[0], batch[1])[1].item())
                else:   # 全トークンの平均（従来と同じ値）に加えて、Ｂ（ずんだもん）側だけの平均も出す
                    tl = token_losses(batch[0], batch[1])
                    losses.append(tl.mean().item())
                    m = batch[2].view(-1) == 1
                    b_sum, b_n = b_sum + tl[m].sum().item(), b_n + int(m.sum())
        model.train()
        estimate.last_b = b_sum / b_n if b_n else None
        return float(np.mean(losses))

    def record(at_step):
        """成長記録（評価＋固定プロンプト生成＋配布形式の書き出し）。停止要求・区間期限で中断したら何も書かない。"""
        t0 = time.time()
        cur["recording"] = at_step
        try:
            val_loss = estimate(val_data, at_step)
            model.eval()
            gen = torch.Generator().manual_seed(1234)
            samples = []
            chk = lambda: yield_reason() is not None   # noqa: E731
            try:
                for p in prompts:
                    pid = [sp.bos_id()] + sp.encode(p)
                    greedy, cands = generate(model, sp, pid, plan["max_new"], 0.0, gen, plan["n_cand"], check=chk)
                    sampled, _ = generate(model, sp, pid, plan["max_new"], 0.8, gen, check=chk)
                    samples.append({"prompt": p, "greedy": greedy, "sampled": sampled, "cands": cands})
            finally:
                model.train()
        finally:
            cur["recording"] = None
        path = os.path.join(out, "ckpt", f"step_{at_step:07d}.bin")
        tmp = f"{path}.tmp-{os.getpid()}"
        write_llama2c(tmp, header_cfg(cfg), export_numpy(model))
        os.replace(tmp, path)
        if at_step in (0, max_iters):
            c2, w2 = read_llama2c(path)
            tid = [sp.bos_id()] + sp.encode(prompts[-1] or "Ａ：こんにちは。Ｂ：")
            ref = numpy_forward(c2, w2, tid)
            with torch.no_grad():
                model.eval()
                got = model(torch.tensor([tid], device=dev))[0][0].float().cpu().numpy()
                model.train()
            diff = float(np.abs(ref - got).max())
            print(f"  export check: max|Δlogit|={diff:.2e} {'OK' if diff < 1e-3 else '!! NG !!'}")
        rec = {"step": at_step, "tokens_seen": at_step * tokens_per_iter, "train_loss": running,
               "val_loss": val_loss, "lr": lr_at(at_step), "time": time.time(), "samples": samples,
               "ckpt": os.path.basename(path)}
        if getattr(estimate, "last_b", None) is not None:
            rec["val_loss_b"] = estimate.last_b
        append_jsonl(growth_path, rec)
        have_growth.add(at_step)
        dt = time.time() - t0
        vb = f" val_B={rec['val_loss_b']:.3f}" if "val_loss_b" in rec else ""
        print(f"[ckpt {at_step}] val={val_loss:.3f}{vb} ({dt:.1f}s)")
        if args.print_samples and len(samples) > 1:
            print(f"   「{samples[1]['prompt']}」→ {samples[1]['greedy'][:60]!r}")
        events.emit("growth", step=at_step, val_loss=round(val_loss, 5), seconds=round(dt, 2))

    def save(reason):
        nonlocal saved_at
        t0 = time.time()
        payload = {"format": CKPT_FORMAT, "step": step, "model": model.state_dict(), "optimizer": opt.state_dict(),
                   "scaler": scaler.state_dict(), "running_loss": running, "plan": plan,
                   "rng": {"numpy_train": train_rng.bit_generator.state, "torch_cpu": torch.get_rng_state(),
                           "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        name, digest, size = save_checkpoint(state_dir, payload, {**ids, "reason": reason, "time": time.time()},
                                             keep=args.keep_checkpoints)
        dt = time.time() - t0
        saved_at = step
        events.emit("checkpoint", step=step, name=name, sha256=digest, bytes=size, seconds=round(dt, 3), reason=reason)
        print(f"  checkpoint {name} ({size/1e6:.2f}MB, {dt:.2f}s, {reason})")
        return {"name": name, "sha256": digest}

    def finish(code, reason, ck):
        events.emit("exit", code=code, reason=reason, next_step=step, max_iters=max_iters, checkpoint=ck,
                    seconds=round(time.time() - t_proc, 2))
        print(f"exit {code}: {reason} next_step={step}/{max_iters}")
        return code

    ckpts = set(checkpoint_steps(max_iters, plan["ckpt_per_decade"]))
    final_bin = os.path.join(out, "ckpt", f"step_{max_iters:07d}.bin")

    def finalize():
        """最終 step の成果物（成長記録・最終 bin・model.bin・last.pt）がそろってから completed にする。
        そろっていなければ None を返し、ループに戻って作り直す。"""
        if max_iters in ckpts and max_iters not in have_growth:
            return None
        if not os.path.exists(final_bin):   # 成長記録だけ残って bin が無い（想定外）→ 記録を退避して作り直す
            reconcile_history(growth_path, lambda r: r.get("step") != max_iters)
            have_growth.discard(max_iters)
            return None
        if _FAULT == "before_finalize":
            os._exit(99)
        tmp = os.path.join(out, f"model.bin.tmp-{os.getpid()}")
        shutil.copy(final_bin, tmp)
        os.replace(tmp, os.path.join(out, "model.bin"))
        tmp = os.path.join(out, f"last.pt.tmp-{os.getpid()}")
        torch.save({"model": model.state_dict(), "cfg": cfg}, tmp)
        os.replace(tmp, os.path.join(out, "last.pt"))
        return finish(EXIT_COMPLETED, "completed", save("completed"))

    model.train()
    try:
        while True:
            # --- 境界（step 回の更新が終わった状態）。まず譲る理由がないか確認する
            yr = yield_reason()
            if yr:
                kind, why = yr
                if kind == "stop":
                    return finish(EXIT_STOP, f"stop_requested:{why}", save(f"stop:{why}"))
                return finish(EXIT_SEGMENT_END, f"segment_end:{why}", save("segment_end"))
            if step >= max_iters and saved_at != step:
                save("reached_end")   # 最終 step の成果物作りを、学習をやり直さずに再開できるように
            if step in ckpts and step not in have_growth:
                try:
                    record(step)
                except Interrupted:
                    print("  growth 記録を中断（この step の記録は書かず、次の区間で記録する）")
                    continue
            if step >= max_iters:
                res = finalize()
                if res is not None:
                    return res
                continue
            done = step - seg_start_step
            if args.save_every_steps and done and step % args.save_every_steps == 0:
                save("periodic")
            # --- 1回の更新
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            if a_weight == 1.0:
                x, y = get_batch(train_data, train_rng, plan["batch_size"])
                with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                    _, loss = fwd(x, y)
            else:   # Ａ側トークンの損失を a_weight 倍にした重み付き平均
                x, y, s = get_batch(train_data, train_rng, plan["batch_size"], train_spk)
                with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                    tl = token_losses(x, y)
                w = torch.where(s.view(-1) == 1, 1.0, a_weight)
                loss = (tl * w).sum() / w.sum().clamp_min(1e-8)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), plan["grad_clip"])
            scaler.step(opt)
            scaler.update()
            lv = loss.item()
            running = lv if running is None else 0.9 * running + 0.1 * lv
            if step % args.log_interval == 0:
                print(f"it {step:6d} loss {lv:.4f} lr {lr_at(step):.2e} {time.time() - t_proc:.0f}s")
                append_jsonl(loss_path, {"step": step, "loss": lv})
                events.emit("progress", step=step + 1, max_iters=max_iters, loss=round(lv, 5), lr=lr_at(step),
                            tokens_seen=(step + 1) * tokens_per_iter)
            step += 1
    except CheckpointError:
        raise
    except Exception as e:  # noqa: BLE001  失敗は理由を残して非0終了（直前の確定 checkpoint は残る）
        events.emit("exit", code=EXIT_ERROR, reason=f"error:{type(e).__name__}", next_step=step, checkpoint=latest(state_dir))
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckpointError as e:
        print(f"checkpoint error: {e}", file=sys.stderr)
        sys.exit(2)
