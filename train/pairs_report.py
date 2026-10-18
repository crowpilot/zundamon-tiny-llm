"""返事ペアのデータの比べ方（v3 と v4 など）。生のペアと採点結果を渡す。

  python3 pairs_report.py v3=corpus/pairs/pairs.jsonl:corpus/pairs/judge.jsonl v4=corpus/v4/pairs.jsonl:corpus/v4/judge.jsonl
"""
import collections
import json
import re
import sys

FOOD = re.compile(r"おなか|お腹|食べ|たべ|おいし|美味し|ずんだ餅|ずんだもち|おやつ")
KANJI = re.compile(r"[一-鿿]")


def report(name, pairs_path, judge_path, limit=None):
    P = [json.loads(l) for l in open(pairs_path, encoding="utf-8") if l.strip()]
    J = {tuple(json.loads(l)["turns"]): json.loads(l) for l in open(judge_path, encoding="utf-8")} if judge_path else {}
    if limit:
        P = P[:limit]
    n = len(P)
    A = [p["turns"][0] for p in P]
    R = [p["turns"][1] for p in P]
    heads = collections.Counter(r[:4] for r in R)
    tails = collections.Counter(re.sub(r"[！？。…!?ー〜]+$", "", r)[-5:] for r in R)
    bigr = collections.Counter(a[i:i + 2] for a in A for i in range(len(a) - 1))
    out = {
        "組": n,
        "話しかけの重複なし": f"{len(set(A)) / n:.1%}",
        "返事の重複なし": f"{len(set(R)) / n:.1%}",
        "話しかけ 2文字の種類/全数": f"{len(bigr) / max(1, sum(bigr.values())):.3f}",
        "話しかけの漢字の割合": f"{sum(len(KANJI.findall(a)) for a in A) / max(1, sum(map(len, A))):.1%}",
        "話しかけの平均文字数": f"{sum(map(len, A)) / n:.1f}",
        "書き出し上位5の合計": f"{sum(v for _, v in heads.most_common(5)) / n:.1%}",
        "語尾上位5の合計": f"{sum(v for _, v in tails.most_common(5)) / n:.1%}",
        "食べ物の返事": f"{sum(bool(FOOD.search(r)) for r in R) / n:.1%}",
        "「ボクも」を含む返事": f"{sum('ボクも' in r for r in R) / n:.1%}",
    }
    js = [J[tuple(p["turns"])] for p in P if tuple(p["turns"]) in J]
    for k in ("response", "natural", "zunda", "specific"):
        v = [j[k] for j in js if k in j]
        if v:
            out[f"採点 {k}"] = f"{sum(v) / len(v):.2f}（2以下 {sum(x <= 2 for x in v) / len(v):.0%}）"
    out["書き出し上位"] = heads.most_common(5)
    out["語尾上位"] = tails.most_common(5)
    print(f"== {name}")
    for k, v in out.items():
        print(f"  {k}: {v}")


def main():
    for arg in sys.argv[1:]:
        name, paths = arg.split("=", 1)
        parts = paths.split(":")
        report(name, parts[0], parts[1] if len(parts) > 1 else None, int(parts[2]) if len(parts) > 2 else None)


if __name__ == "__main__":
    main()
