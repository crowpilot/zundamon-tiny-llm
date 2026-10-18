"""生成した会話を LLM に採点させる（試し生成の方式比較用）。推論キュー経由、標準ライブラリのみ。

  python3 zunda_judge.py corpus/pilot_direct.jsonl corpus/pilot_inline.jsonl corpus/pilot_2call.jsonl \
      --out corpus/pilot_judge.jsonl --polite

全ファイルの会話を混ぜてシャッフルし、方式名を伏せて10本ずつ採点させる（方式による先入観を避ける）。
採点（各1〜5）: response=前の発言にちゃんと答えているか / natural=日本語として自然か / zunda=ずんだもんらしいか
最後に方式ごとの平均を表示する。
"""
import argparse
import collections
import json
import os
import random

import gen_corpus as gc

SYSTEM = "あなたは会話文の審査員です。指示どおりのJSONだけを出力します。"
PROMPT = """次の短い会話をそれぞれ採点してください。Ａは相手、Ｂはずんだもん（一人称ボク、語尾「〜のだ」の妖精）です。

採点（各1〜5の整数、5が最良）:
- response: 各発言が直前の発言にちゃんと答えているか。話が急に飛ぶ・つながらない発言があれば下げる。
- natural: 日本語として自然か。文法の崩れ・意味の通らない文があれば下げる。
- zunda: Ｂがずんだもんらしいか（口調・性格）。{extra_rule}

{convs}

出力形式（JSONのみ）:
[{{"i":1,"response":3,"natural":3,"zunda":3{extra_key}}}, ...]"""
EXTRA_RULE = "\n- specific: Ｂの返事が、その発言だから出てくる返事か。どんな発言にも使える決まり文句（「お腹が空いてきたのだ」「とってもいそがしいのだ」など）なら低い。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_call", type=int, default=10)
    ap.add_argument("--queue_url", default=os.environ.get("V100_QUEUE_URL", "http://localhost:8081"))
    ap.add_argument("--priority", choices=["auto", "normal"], default="auto",
                    help="auto: キューが background_priority_v1 を返せば background で投入（既定）。normal: 従来どおり")
    ap.add_argument("--base_url", default=None, help="指定するとキューを使わず OpenAI 互換 API を直接呼ぶ（例: http://localhost:8000/v1）")
    ap.add_argument("--api_key_file", default="~/.config/tiny-ja-llm/api-key")
    ap.add_argument("--model", default="gemma4-26b-qat-q8")
    ap.add_argument("--client_id", default="tiny-ja-llm")
    ap.add_argument("--inflight", type=int, default=1)
    ap.add_argument("--queue_timeout", type=int, default=3600)
    ap.add_argument("--exec_timeout", type=int, default=600)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--max_fail", type=int, default=3)
    ap.add_argument("--polite", action="store_true")
    ap.add_argument("--specific", action="store_true", help="「ならでは」（specific）も採点する（v4）")
    args = ap.parse_args()
    keys = ("response", "natural", "zunda") + (("specific",) if args.specific else ())

    pool = []
    for f in args.files:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                pool.append({"turns": json.loads(line)["turns"], "src": os.path.basename(f)})
    random.Random(0).shuffle(pool)
    corpus = gc.Corpus(args.out)
    pool = [p for p in pool if gc.Corpus._h(p["turns"]) not in corpus.seen]
    args.n = len(corpus.seen) + len(pool)

    def make_req():
        if not pool:
            return None
        items = [pool.pop(0) for _ in range(min(args.per_call, len(pool)))]
        convs = "\n\n".join(f"会話{i + 1}:\n" + "\n".join(("Ｂ：" if j % 2 else "Ａ：") + t for j, t in enumerate(it["turns"]))
                            for i, it in enumerate(items))
        req = {"model": args.model, "temperature": 0.0, "max_tokens": 1500, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": PROMPT.format(
                                convs=convs, extra_rule=EXTRA_RULE if args.specific else "",
                                extra_key=',"specific":3' if args.specific else "")}]}
        return req, {"items": items}

    def parse_fn(text, meta):
        items = meta["items"]
        out = []
        for r in gc_json(text):
            try:
                i = int(r["i"]) - 1
                scores = {k: int(r[k]) for k in keys}
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= i < len(items) and all(1 <= v <= 5 for v in scores.values()):
                out.append({"turns": items[i]["turns"], "src": items[i]["src"], **scores})
        return out

    (gc.run_direct if args.base_url else gc.run_queue)(args, corpus, make_req=make_req, parse_fn=parse_fn)
    summarize(args.out)


def gc_json(text):
    import re
    m = re.search(r"\[.*\]", text or "", flags=re.S)
    try:
        data = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def summarize(path):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        for k in ("response", "natural", "zunda"):
            agg[r["src"]][k].append(r[k])
    print(f"{'ファイル':32s} {'本数':>4s} {'応答':>5s} {'自然さ':>5s} {'ずんだ':>5s}  応答≦2の割合")
    for src, d in sorted(agg.items()):
        n = len(d["response"])
        low = sum(v <= 2 for v in d["response"]) / n
        print(f"{src:32s} {n:4d} {sum(d['response'])/n:5.2f} {sum(d['natural'])/n:5.2f} {sum(d['zunda'])/n:5.2f}  {low:.0%}")


if __name__ == "__main__":
    main()
