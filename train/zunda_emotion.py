"""返事ペアに「ずんだもんの気持ち」タグを付ける（推論キュー / Gemma 4）。標準ライブラリのみ。

  python3 zunda_emotion.py corpus/pairs/pairs_filtered.jsonl --out corpus/pairs/emotion.jsonl
  # 一致率を見るとき（同じ返事を別の並びでもう一度分類）
  python3 zunda_emotion.py corpus/pairs/pairs_filtered.jsonl --out corpus/pairs/emotion_b.jsonl --sample 1000 --seed 7

Gemma にさせるのは「返事 20 個それぞれの気持ちを 8 つの中から 1 つ選ぶ」だけ（返事の生成とは別の工程）。
出力: 1行1組 {"turns": [話しかけ, 返事], "emo": タグ}
"""
import argparse
import collections
import json
import os
import random
import re

import gen_corpus as gc

SPEC_VERSION = "zunda-emotion/1"
EMOTIONS = [
    ("ふつう", "落ち着いて答えている・説明している"),
    ("うれしい", "喜んでいる・楽しい・好き・ほめられて照れる"),
    ("ドヤ", "得意げ・自慢・「天才なのだ」「まかせるのだ」"),
    ("びっくり", "驚いている・「ええっ」"),
    ("かなしい", "落ち込む・残念・相手に同情してしょんぼり"),
    ("ぷんぷん", "すねる・怒る・むっとする・言い返す"),
    ("こまった", "不安・あせる・どうしよう・わからなくて戸惑う"),
    ("ねむい", "眠い・疲れた・ぐったり"),
]
LABELS = [e for e, _ in EMOTIONS]

SYSTEM = "あなたは会話文の感情を判定する審査員です。指示どおりのJSONだけを出力します。"
PROMPT = """次の各やり取りで、Ｂ（ずんだもん）の返事にこめられた気持ちを、下の8つから1つずつ選んでください。
話しかけ（Ａ）ではなく、返事（Ｂ）の気持ちを選びます。迷ったら返事の語調にいちばん近いもの。

{labels}

{items}

出力形式（JSONのみ。番号順に気持ちの名前だけ）: ["うれしい", "ふつう", ...]"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_call", type=int, default=20)
    ap.add_argument("--sample", type=int, default=0, help="0 なら全部。数を指定するとランダムにその数だけ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--queue_url", default=os.environ.get("V100_QUEUE_URL", "http://localhost:8081"))
    ap.add_argument("--priority", choices=["auto", "normal"], default="auto")
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--api_key_file", default="~/.config/tiny-ja-llm/api-key")
    ap.add_argument("--model", default="gemma4-26b-qat-q8")
    ap.add_argument("--client_id", default="tiny-ja-llm")
    ap.add_argument("--inflight", type=int, default=1)
    ap.add_argument("--queue_timeout", type=int, default=3600)
    ap.add_argument("--exec_timeout", type=int, default=600)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--max_fail", type=int, default=3)
    ap.add_argument("--polite", action="store_true")
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    if args.sample:
        pool = pool[:args.sample]
    corpus = gc.Corpus(args.out)
    pool = [p for p in pool if gc.Corpus._h(p["turns"]) not in corpus.seen]
    args.n = len(corpus.seen) + len(pool)
    labels = "\n".join(f"- {e}: {d}" for e, d in EMOTIONS)
    stats = collections.Counter()
    retry = []

    def make_req():
        if not pool and not retry:
            return None
        items = retry.pop() if retry else []   # 数が合わなかった組は、別の組み合わせに混ぜてもう一度だけ
        items += [pool.pop() for _ in range(min(args.per_call - len(items), len(pool)))]
        lines = "\n".join(f"{i + 1}. Ａ「{it['turns'][0]}」 Ｂ「{it['turns'][1]}」" for i, it in enumerate(items))
        req = {"model": args.model, "temperature": 0.0, "max_tokens": 600, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": PROMPT.format(labels=labels, items=lines)}]}
        return req, {"items": items}

    def parse_fn(text, meta):
        items = meta["items"]
        m = re.search(r"\[.*\]", text or "", flags=re.S)
        try:
            got = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            got = []
        if len(got) != len(items):
            stats["数が合わない"] += 1
            again = [it for it in items if not it.get("_retried")]
            for it in again:
                it["_retried"] = True
            for i in range(0, len(again), 10):
                retry.append(again[i:i + 10])
            return []
        out = []
        for it, e in zip(items, got):
            e = str(e).strip()
            if e not in LABELS:
                stats["知らないラベル"] += 1
                continue
            stats[e] += 1
            out.append({"turns": it["turns"], "emo": e, "spec": SPEC_VERSION})
        return out

    print(f"既存 {len(corpus.seen)} → 目標 {args.n}  spec={SPEC_VERSION}")
    (gc.run_direct if args.base_url else gc.run_queue)(args, corpus, make_req=make_req, parse_fn=parse_fn)
    tot = sum(v for k, v in stats.items() if k in LABELS)
    print("今回の内訳: " + ", ".join(f"{e} {stats[e]}({stats[e] / max(1, tot):.0%})" for e in LABELS)
          + "".join(f", {k} {v}" for k, v in stats.items() if k not in LABELS))


if __name__ == "__main__":
    main()
