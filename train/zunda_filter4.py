"""返事ペア v4 のふるい分け。

  python3 zunda_filter4.py corpus/v4/pairs.jsonl corpus/v4/judge.jsonl --out corpus/v4/pairs_filtered.jsonl

残す条件（既定値は仕様の案。お試しの結果で変える）:
  - 採点済みで response >= 3、natural >= 3、specific（ならでは）>= 2（10/17 お試しの結果で 3 → 2 に変更）
  - 同じ返事は 3 回まで
  - 同じ書き出し（返事の先頭 4 文字）は全体の 1.5% まで、同じ語尾（記号を除いた末尾 5 文字）は 3% まで
  - 評価用の話しかけ（eval/inputs_*.tsv）と同じ文は除く（話しかけの段階でも除いているが念のため）
"""
import argparse
import collections
import json
import os
import random
import re

from gen_pairs4 import eval_inputs, norm_key


def head(r):
    return r[:4]


def tail(r):
    return re.sub(r"[！？。…!?ー〜]+$", "", r)[-5:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("judge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_response", type=int, default=3)
    ap.add_argument("--min_natural", type=int, default=3)
    ap.add_argument("--min_specific", type=int, default=2)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--head_share", type=float, default=0.015)
    ap.add_argument("--tail_share", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    scores = {tuple(json.loads(l)["turns"]): json.loads(l) for l in open(args.judge, encoding="utf-8")}
    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    banned = eval_inputs(os.path.dirname(os.path.abspath(__file__)))
    # 書き出し・語尾の上限は、採点で残る見込みの件数を基準にする
    passing = [r for r in rows if (s := scores.get(tuple(r["turns"]))) and s["response"] >= args.min_response
               and s["natural"] >= args.min_natural and s.get("specific", 5) >= args.min_specific]
    hcap = max(10, int(len(passing) * args.head_share))
    tcap = max(10, int(len(passing) * args.tail_share))
    why, used, heads, tails, keep = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), []
    for r in rows:
        s = scores.get(tuple(r["turns"]))
        a, b = r["turns"]
        if s is None:
            why["未採点"] += 1
        elif norm_key(a) in banned:
            why["評価用と同じ"] += 1
        elif s["response"] < args.min_response:
            why["応答が低い"] += 1
        elif s["natural"] < args.min_natural:
            why["不自然"] += 1
        elif s.get("specific", 5) < args.min_specific:
            why["ならではが低い"] += 1
        elif used[b] >= args.cap:
            why["同じ返事の使いすぎ"] += 1
        elif heads[head(b)] >= hcap:
            why["書き出しの集中"] += 1
        elif tails[tail(b)] >= tcap:
            why["語尾の集中"] += 1
        else:
            used[b] += 1
            heads[head(b)] += 1
            tails[tail(b)] += 1
            keep.append({**r, "judge": {k: s[k] for k in ("response", "natural", "zunda", "specific") if k in s}})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} 組 → {len(keep)} 組を採用（書き出しの上限 {hcap}、語尾の上限 {tcap}）  除外: "
          + ", ".join(f"{k} {v}" for k, v in why.most_common()))


if __name__ == "__main__":
    main()
