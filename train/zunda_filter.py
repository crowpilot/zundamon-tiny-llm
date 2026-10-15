"""採点結果でずんだもんコーパスをふるう。

  python3 zunda_filter.py corpus/zunda_v41.jsonl corpus/zunda_v41_judge.jsonl --out corpus/zunda_filtered.jsonl

残す条件:
  - 採点済みで response（前の発言に答えているか）>= --min_response、natural >= --min_natural
  - ずんだもんの同じ発言は全体で --cap 回まで（例文の丸写し「ボクにまかせるのだ！」などの偏りを抑える）。
    超える発言を含む会話は落とす（会話の順番はランダムにしてから数える）
"""
import argparse
import collections
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("judge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_response", type=int, default=3)
    ap.add_argument("--min_natural", type=int, default=3)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    scores = {}
    for line in open(args.judge, encoding="utf-8"):
        r = json.loads(line)
        scores[tuple(r["turns"])] = r
    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    why = collections.Counter()
    used = collections.Counter()
    keep = []
    for r in rows:
        s = scores.get(tuple(r["turns"]))
        if s is None:
            why["未採点"] += 1
            continue
        if s["response"] < args.min_response:
            why["応答が低い"] += 1
            continue
        if s["natural"] < args.min_natural:
            why["不自然"] += 1
            continue
        b = r["turns"][1::2]
        if any(used[t] >= args.cap for t in b):
            why["使われすぎの発言"] += 1
            continue
        used.update(b)
        keep.append({**r, "judge": {k: s[k] for k in ("response", "natural", "zunda")}})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} 本 → {len(keep)} 本を採用  除外: " + ", ".join(f"{k} {v}" for k, v in why.most_common()))


if __name__ == "__main__":
    main()
