"""「入力 → ずんだもんの返事」ペアのコーパスを2段階で作る（推論キュー / Gemma 4）。標準ライブラリのみ。

  1) 入力だけを作る（種類・長さ・種の言葉をばらばらに指定して多様にする）
     python3 gen_pairs.py inputs --n 30000 --out corpus/pair_inputs.jsonl --polite
  2) 入力に返事を付ける（1回の推論で20個ずつ）
     python3 gen_pairs.py replies --inputs corpus/pair_inputs.jsonl --n 30000 --out corpus/pairs.jsonl --polite

返事の方針: 入力の内容にちゃんと反応する／入力の言葉を返事に入れるのは約2割だけ（--echo 0.3 で頼み、頼んでいないのにくり返した返事は除く）／決まり文句に逃げない。
出力: 1行1ペア {"turns": [入力, 返事], "cat": 種類, "len": 長さ区分, "overlap": 入力と共通する2文字以上の語があるか}
キューの扱いは gen_corpus.py と同じ（台帳・同じ request_id での再投入・中断時の取り消し・--polite）。
"""
import argparse
import json
import os
import random
import re

import gen_corpus as gc
from gen_zunda import ZUNDA_OFFICIAL, check, normalize, json_array

SPEC_VERSION = "zunda-pairs/3"

# 入力の種類（名前, 説明, 重み）
CATS = [
    ("あいさつ", "あいさつや声かけ", 8),
    ("質問", "ずんだもんへの質問（好き嫌い・予定・気持ち・知っていることなど）", 14),
    ("報告", "自分に起きたことの報告", 12),
    ("誘い", "遊びや買い物などへの誘い", 7),
    ("ほめる", "ほめる・感心する", 6),
    ("からかう", "軽くからかう・つっこむ（いじわるすぎない）", 6),
    ("頼みごと", "手伝ってほしいこと・お願い", 7),
    ("困りごと", "困っていること・なやみ", 8),
    ("感想", "見たもの・食べたもの・天気などへの感想", 9),
    ("一言", "「ねむい」「おなかすいた」のような一言だけ", 8),
    ("単語", "名詞を1つだけ（「りんご」「でんしゃ」など）", 5),
    ("くずれ", "打ち間違い・言いかけ・意味がよく分からない短い文", 4),
    ("長め", "2文くらいの少し長い話", 6),
]
LENS = {"一言": "1〜6文字", "単語": "1〜6文字", "くずれ": "2〜12文字", "長め": "25〜40文字"}

SYSTEM = "あなたは日本語の会話データを作る作家です。指示どおりのJSONだけを出力します。"

INPUT_PROMPT = """チャットで「ずんだもん」（ずんだの妖精のキャラクター）に話しかける人の発言を{n}個つくってください。

種類: {cat}（{desc}）
長さ: {length}
話題の種の言葉（そのまま使わなくてよい。ばらばらの話題にするヒント）: {seeds}

条件:
- 人がチャット欄にふつうに打つような文。ていねい語・くだけた言い方・子どもっぽい言い方をまぜる。
- ひらがな多め。漢字は少なめ。記号は「。」「！」「？」「…」くらい。絵文字は使わない。
- {n}個とも別の内容・別の言い回しにする。同じ書き出しをくりかえさない。
- 発言だけを書く。名前や説明は書かない。

出力形式（JSONのみ）: ["発言1", "発言2", ...]"""

REPLY_PROMPT = """次の各発言に、ずんだもんとして1回ずつ返事をしてください。

ずんだもんの話し方（必ず守る）:
{zunda}

返事の条件:
- 発言の中身にちゃんと反応する。質問には答え、報告には感想を言い、誘いには乗るか断る。
- 発言の後ろの【】の指示に従う。
  【自分の言葉で】… 発言の言葉をくり返さず、言いかえや自分の気持ち・行動で返す（例: 「りんごたべた」→「いいなあ、ボクもおなかすいたのだ！」）。
  【言葉を入れて】… 発言の中の言葉を1つ、返事の途中に自然に入れる。返事の最初でくり返さない（例: 「りんごたべた」→「いいなあ、ボクもりんごたべたいのだ！」）。
- 意味が分からない発言には、ずんだもんらしく聞き返す。
- 1つの返事は30文字以内。ひらがな多め。「ボクにまかせるのだ」「ボクは天才なのだ」などの決まり文句ばかりにしない。
- 返事どうしで同じ言い回しをくりかえさない。

発言:
{inputs}

出力形式（JSONのみ。発言の番号順に返事だけ）: ["返事1", "返事2", ...]"""


def pick(rng, table):
    return rng.choices(table, weights=[t[2] for t in table])[0]


def overlap(a, b):
    """入力の言葉が返事に入っているか: 入力と共通する「ひらがな4文字以上」または「漢字・カタカナを含む2文字以上」の並びがあるか
    （「そうな」「ている」のような短いひらがなの一致は数えない）"""
    a = re.sub(r"[^\u3041-\u30ff\u4e00-\u9fff]", "", a)
    for n in (4, 2):
        for i in range(len(a) - n + 1):
            s = a[i:i + n]
            if (n == 4 or re.search(r"[\u30a1-\u30ff\u4e00-\u9fff]", s)) and s in b:
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["inputs", "replies"])
    ap.add_argument("--inputs", help="replies で使う入力ファイル")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_inputs", type=int, default=30)
    ap.add_argument("--per_replies", type=int, default=20)
    ap.add_argument("--queue_url", default=os.environ.get("V100_QUEUE_URL", "http://localhost:8081"))
    ap.add_argument("--priority", choices=["auto", "normal"], default="auto",
                    help="auto: キューが background_priority_v1 を返せば background で投入（既定）。normal: 従来どおり")
    ap.add_argument("--base_url", default=None, help="指定するとキューを使わず OpenAI 互換 API を直接呼ぶ（例: http://localhost:8000/v1）")
    ap.add_argument("--api_key_file", default="~/.config/tiny-ja-llm/api-key")
    ap.add_argument("--model", default="gemma4-26b-qat-q8")
    ap.add_argument("--client_id", default="tiny-ja-llm")
    ap.add_argument("--inflight", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--queue_timeout", type=int, default=3600)
    ap.add_argument("--exec_timeout", type=int, default=600)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--max_fail", type=int, default=3)
    ap.add_argument("--polite", action="store_true")
    ap.add_argument("--echo", type=float, default=0.3,
                    help="返事に入力の言葉を入れるよう指示する割合（指示しなかった分でくり返したものは除くので、残る割合は約2割）")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = random.Random(args.seed)
    corpus = gc.Corpus(args.out)
    stats = {}

    def bump(k):
        stats[k] = stats.get(k, 0) + 1

    def request(prompt, temp):
        return {"model": args.model, "temperature": temp, "max_tokens": args.max_tokens, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}

    if args.stage == "inputs":
        def make_req():
            name, desc, _ = pick(rng, CATS)
            length = LENS.get(name, rng.choice(["5〜12文字", "8〜20文字", "12〜25文字"]))
            seeds = "、".join(rng.sample(gc.WORDS + gc.TOPICS, 4))
            p = INPUT_PROMPT.format(n=args.per_inputs, cat=name, desc=desc, length=length, seeds=seeds)
            return request(p, args.temperature), {"cat": name, "len": length, "spec": SPEC_VERSION}

        def parse_fn(text, meta):
            out = []
            for s in json_array(text):
                if not isinstance(s, str):
                    continue
                s = normalize(s)
                if not s or len(s) > 45 or re.search(r"[ＡＢ][：:]|ずんだもん[：:]", s):
                    bump("除外")
                    continue
                out.append({"turns": [s], "cat": meta["cat"], "len": meta["len"], "spec": SPEC_VERSION})
                bump("ok")
            print("  入力: " + ", ".join(f"{k} {v}" for k, v in stats.items()), flush=True)
            return out
    else:
        done = {json.loads(l)["turns"][0] for l in open(args.out, encoding="utf-8")} if os.path.exists(args.out) else set()
        pool = [json.loads(l) for l in open(args.inputs, encoding="utf-8") if l.strip()]
        pool = [p for p in pool if p["turns"][0] not in done]
        rng.shuffle(pool)
        args.n = min(args.n, len(corpus.seen) + len(pool))
        rej_path = args.out + ".rejects.jsonl"

        def make_req():
            if not pool:
                return None
            items = [dict(pool.pop(), echo=rng.random() < args.echo) for _ in range(min(args.per_replies, len(pool)))]
            lines = "\n".join(f"{i + 1}. {it['turns'][0]}　【{'言葉を入れて' if it['echo'] else '自分の言葉で'}】"
                               for i, it in enumerate(items))
            p = REPLY_PROMPT.format(zunda=ZUNDA_OFFICIAL, inputs=lines)
            return request(p, 0.9), {"items": items}

        def parse_fn(text, meta):
            items, out = meta["items"], []
            replies = json_array(text)
            if len(replies) != len(items):
                bump("数が合わない")
                return out
            with open(rej_path, "a", encoding="utf-8") as rej:
                for it, r in zip(items, replies):
                    if not isinstance(r, str) or "【" in r:
                        bump("形式")
                        continue
                    pair = [it["turns"][0], normalize(r)]
                    ok, why = check(pair, partner_noda_ok=True)   # 入力側の「のだ」は人の打ち方なので許す
                    if not ok:
                        bump(why)
                        rej.write(json.dumps({"why": why, "turns": pair, "cat": it.get("cat")}, ensure_ascii=False) + "\n")
                        continue
                    ov = overlap(pair[0], pair[1])
                    if ov and not it["echo"]:   # 「自分の言葉で」と頼んだのに入力の言葉をくり返した返事は使わない
                        bump("くり返し")
                        rej.write(json.dumps({"why": "くり返し", "turns": pair, "cat": it.get("cat")}, ensure_ascii=False) + "\n")
                        continue
                    bump("ok")
                    out.append({"turns": pair, "cat": it.get("cat"), "len": it.get("len"), "echo_req": it["echo"],
                                "overlap": ov, "spec": SPEC_VERSION})
            print("  返事: " + ", ".join(f"{k} {v}" for k, v in stats.items()), flush=True)
            return out

    print(f"既存 {len(corpus.seen)} → 目標 {args.n}  stage={args.stage} spec={SPEC_VERSION}")
    (gc.run_direct if args.base_url else gc.run_queue)(args, corpus, make_req=lambda: make_req(), parse_fn=parse_fn)


if __name__ == "__main__":
    main()
