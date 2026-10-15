"""学習したモデルに、評価用データ（学習に使っていない会話）の相手の第一声を渡し、ずんだもんの返答を1つ生成させる。

  python3 eval_replies.py out/zs-a000 --data data_zunda2 --out out/zs-a000/replies.jsonl

- チェックポイントは growth.jsonl の val_loss_b（Ｂ側の損失）が最小のものを使う（--step で指定も可）
- greedy 生成。「Ａ：」が出た時点（ずんだもんが台詞を終えた）で止める
- 出力は zunda_judge.py でそのまま採点できる形 {"turns": [相手, 返答], "step": ...}
推論は numpy の参照実装（Core2 の C 実装と同じ計算）。
"""
import argparse
import json
import os

import numpy as np
import sentencepiece as spm

from llama2c_format import numpy_forward, read_llama2c


def pick_step(run_dir):
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "growth.jsonl"), encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("val_loss_b") is not None] or recs
    key = "val_loss_b" if "val_loss_b" in recs[0] else "val_loss"
    best = min(recs, key=lambda r: r[key])
    return best["step"], best[key], key


def val_openers(data_dir, sp):
    """val.bin を BOS で区切って会話に戻し、相手（Ａ）の第一声を取り出す"""
    ids = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16).tolist()
    convs, cur = [], []
    for t in ids:
        if t == sp.bos_id():
            if cur:
                convs.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        convs.append(cur)
    out = []
    for c in convs:
        text = sp.decode(c)
        if text.startswith("Ａ："):
            first = text[2:].split("Ｂ：")[0].split("Ａ：")[0].strip()
            if first:
                out.append(first)
    return out


def reply(cfg, w, sp, opener, max_new=48):
    a_id, b_id = sp.piece_to_id("Ａ："), sp.piece_to_id("Ｂ：")
    toks = [sp.bos_id()] + sp.encode("Ａ：" + opener + "Ｂ：")
    out = []
    for _ in range(max_new):
        logits = numpy_forward(cfg, w, toks[-cfg["seq_len"]:])[-1]
        nxt = int(np.argmax(logits))
        if nxt in (a_id, b_id, sp.bos_id(), sp.eos_id()):
            break
        out.append(nxt)
        toks.append(nxt)
    return sp.decode(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--openers", default=None, help="1行1つの話しかけ一覧（省略時は評価用データの第一声）")
    args = ap.parse_args()
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.run_dir, "tok.model"))
    if args.step is None:
        step, val, key = pick_step(args.run_dir)
    else:
        step, val, key = args.step, None, None
    cfg, w = read_llama2c(os.path.join(args.run_dir, "ckpt", f"step_{step:07d}.bin"))
    openers = ([l.strip() for l in open(args.openers, encoding="utf-8") if l.strip()]
               if args.openers else val_openers(args.data, sp))
    with open(args.out, "w", encoding="utf-8") as f:
        for o in openers:
            r = reply(cfg, w, sp, o)
            f.write(json.dumps({"turns": [o, r or "（無言）"], "step": step,
                                "run": os.path.basename(args.run_dir.rstrip("/"))}, ensure_ascii=False) + "\n")
    print(f"{args.run_dir}: step {step} ({key}={val}) で {len(openers)} 本の返答 → {args.out}")


if __name__ == "__main__":
    main()
