"""気持ちタグ付きモデルの評価。

  python3 eval_tags.py out/v100-ps-base-tag/run --data data_pairs_tag \
      --openers eval/inputs_claude_v1.txt eval/inputs_user_v1.txt --out eval/replies_tag_base.jsonl

1) val（学習に使っていない組）で、Gemma が付けたタグと、モデルが選ぶタグ（Ｂ：直後の 1 トークンをタグだけに制限して最大のもの）の一致率
   - 制限しないときに 1 トークン目がタグになる割合も出す（形を覚えているか）
2) 話しかけ一覧に返事を作る（1 トークン目はタグに制限、あとは greedy）。出力 {"turns": [話しかけ, 返事], "emo": タグ}
"""
import argparse
import collections
import json
import os

import numpy as np
import sentencepiece as spm

from eval_replies import pick_step
from llama2c_format import numpy_forward, read_llama2c
from prepare import EMO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--openers", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--tag_end", action="store_true", help="タグが返事の最後にあるモデル（返事のあとにタグを選ばせる）")
    args = ap.parse_args()
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.data, "tok.model"))
    step, val, key = pick_step(args.run_dir)
    cfg, w = read_llama2c(os.path.join(args.run_dir, "ckpt", f"step_{step:07d}.bin"))
    tag_ids = [sp.piece_to_id("〈" + e + "〉") for e in EMO]
    assert all(t != sp.unk_id() for t in tag_ids), "タグがトークナイザにありません"
    a_id, b_id = sp.piece_to_id("Ａ："), sp.piece_to_id("Ｂ：")
    stop = {a_id, b_id, sp.bos_id(), sp.eos_id()}

    def first_logits(text):
        toks = [sp.bos_id()] + sp.encode("Ａ：" + text + "Ｂ：")
        return toks, numpy_forward(cfg, w, toks[-cfg["seq_len"]:])[-1]

    # 1) val でのタグ一致
    ids = np.fromfile(os.path.join(args.data, "val.bin"), dtype=np.uint16).tolist()
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
    n = hit = free_tag = 0
    conf = collections.Counter()
    for c in convs:
        if b_id not in c or c[0] != a_id:
            continue
        k = c.index(b_id)
        if args.tag_end:
            j = next((i for i in range(k + 1, len(c)) if c[i] in tag_ids), None)
            if j is None:
                continue
            gold = EMO[tag_ids.index(c[j])]
            logits = numpy_forward(cfg, w, ([sp.bos_id()] + c[:j])[-cfg["seq_len"]:])[-1]
        else:
            if k + 1 >= len(c) or c[k + 1] not in tag_ids:
                continue
            gold = EMO[tag_ids.index(c[k + 1])]
            logits = numpy_forward(cfg, w, ([sp.bos_id()] + c[:k + 1])[-cfg["seq_len"]:])[-1]
        pred = EMO[int(np.argmax(logits[tag_ids]))]
        free_tag += int(np.argmax(logits)) in tag_ids
        n += 1
        hit += pred == gold
        conf[(gold, pred)] += 1
    print(f"{args.run_dir} step {step} ({key}={val:.4f})")
    where = "返事のあとの1トークンがタグ" if args.tag_end else "1トークン目がタグ"
    print(f"val のタグ一致 {hit}/{n} = {hit / max(1, n):.1%}   制限なしで{where} {free_tag / max(1, n):.1%}")
    gold_c = collections.Counter(g for (g, _), v in conf.items() for _ in range(v))
    for e in EMO:
        row = {p: conf[(e, p)] for p in EMO if conf[(e, p)]}
        if gold_c[e]:
            print(f"  正解 {e:5s} {gold_c[e]:4d} 件 → 当たり {conf[(e, e)] / gold_c[e]:.0%}  {row}")

    # 2) 話しかけ一覧への返事
    if args.openers and args.out:
        dist = collections.Counter()
        with open(args.out, "w", encoding="utf-8") as f:
            for p in args.openers:
                for o in [l.rstrip("\n").split("\t")[-1].strip() for l in open(p, encoding="utf-8") if l.strip()]:
                    toks, logits = first_logits(o)
                    out, tag = [], None
                    if not args.tag_end:
                        tag = tag_ids[int(np.argmax(logits[tag_ids]))]
                        toks.append(tag)
                        logits = numpy_forward(cfg, w, toks[-cfg["seq_len"]:])[-1]
                    for _ in range(args.max_new):
                        nxt = int(np.argmax(logits))
                        if args.tag_end and (nxt in tag_ids or nxt in stop):
                            tag = tag_ids[int(np.argmax(logits[tag_ids]))]   # 返事の終わりで、タグだけに制限して選ぶ
                            break
                        if nxt in stop:
                            break
                        out.append(nxt)
                        toks.append(nxt)
                        logits = numpy_forward(cfg, w, toks[-cfg["seq_len"]:])[-1]
                    if tag is None:
                        tag = tag_ids[int(np.argmax(logits[tag_ids]))]
                    emo = EMO[tag_ids.index(tag)]
                    dist[emo] += 1
                    f.write(json.dumps({"turns": [o, sp.decode(out).strip() or "（無言）"], "emo": emo, "step": step,
                                        "run": os.path.basename(os.path.dirname(args.run_dir.rstrip("/")))},
                                       ensure_ascii=False) + "\n")
        tot = sum(dist.values())
        print(f"話しかけ {tot} 個のタグ: " + ", ".join(f"{e} {dist[e]}({dist[e] / tot:.0%})" for e in EMO))


if __name__ == "__main__":
    main()
