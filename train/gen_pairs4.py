"""返事ペア v4（シーンカード方式）。標準ライブラリのみ。

  1) 話しかけ: シーンカード 1 枚 → 話しかけ 8 個（意図 4 種類 × 2 個）
     python3 gen_pairs4.py inputs --n 90000 --out corpus/v4/inputs.jsonl
  2) 返事: 話しかけ 20 個ずつ（シーンは見せない。話しかけの文だけを見て答える）
     python3 gen_pairs4.py replies --inputs corpus/v4/inputs.jsonl --n 90000 --out corpus/v4/pairs.jsonl

どちらも途中から再開でき、--n を増やして同じ出力に追記できる（6 万 → 10 万など）。
評価用の話しかけ（eval/inputs_*.tsv）と同じ文は、話しかけの段階で除く。
キューの扱いは gen_corpus.py と同じ（台帳・同じ request_id での再投入・background の同時枠）。
"""
import argparse
import glob
import json
import os
import random
import re

import gen_corpus as gc
from gen_pairs import overlap
from gen_zunda import ZUNDA_OFFICIAL, check, normalize, json_array

SPEC_VERSION = "zunda-pairs/4"

# ---------------------------------------------------------------- シーンカード
PLACES = ["公園のベンチ", "台所", "駅のホーム", "雨の日の窓辺", "図書館", "海辺", "ゲームセンター", "病院の待合室", "田んぼ道",
          "コンビニ", "教室", "布団の中", "お風呂", "バスの中", "屋上", "お祭りの屋台", "遊園地", "キャンプ場", "雪の積もった道",
          "商店街", "職場の休憩室", "電車の中", "スーパーのレジ", "公民館", "川べり", "動物園", "水族館", "神社", "おばあちゃんの家",
          "自分の部屋", "ファミレス", "カフェ", "高速道路のサービスエリア", "旅館", "空港", "体育館", "畑", "山道", "本屋", "駐車場"]
TIMES = ["朝", "昼", "夕方", "夜", "真夜中", "休みの日の朝"]
WEATHERS = ["春の晴れ", "梅雨の雨", "真夏の暑い日", "台風の前", "秋の夕焼け", "冬の雪", "くもり", "風の強い日"]
# (話しかける人, 表記の指示)
PERSONS = [
    ("小学生", "ひらがな多め。子どもらしい言い方"),
    ("中学生", "ひらがなと簡単な漢字。くだけた言い方"),
    ("高校生", "漢字まじり。友だちに話すくだけた言い方や若者ことばもよい"),
    ("大学生", "漢字まじり。くだけた言い方"),
    ("会社員", "漢字まじり。仕事の疲れや大人の話題もよい。「飯」「マジ」などの口語もよい"),
    ("おばあちゃん", "やさしい言い方。ひらがな多め"),
    ("お店の人", "ていねいな言い方。漢字まじり"),
    ("ずんだもんの友だち", "気安い言い方。漢字まじり"),
    ("旅行中の人", "漢字まじり。旅先の話題"),
    ("夜ふかしの人", "漢字まじり。ゆるい言い方、「エモい」などの口語もよい"),
]
MOODS = ["うれしい", "ねむい", "いらいら", "不安", "ひま", "わくわく", "さみしい", "つかれた"]
# (意図, 説明, 長さ, 重み)
INTENTS = [
    ("あいさつ", "あいさつや声かけ", "3〜15文字", 7),
    ("質問", "ずんだもんへの質問（好き嫌い・予定・気持ち・知っていることなど）", "5〜25文字", 12),
    ("報告", "自分に起きたことの報告", "6〜30文字", 12),
    ("誘い", "遊びや買い物などへの誘い", "6〜25文字", 6),
    ("ほめる", "ずんだもんをほめる・感心する", "4〜20文字", 5),
    ("からかう", "ずんだもんを軽くからかう・つっこむ（いじわるすぎない）", "5〜25文字", 6),
    ("頼みごと", "手伝ってほしいこと・お願い", "6〜25文字", 6),
    ("困りごと", "困っていること・なやみ", "6〜30文字", 7),
    ("感想", "見たもの・食べたもの・天気などへの感想", "5〜25文字", 8),
    ("一言", "「ねむい」「おなかすいた」のような一言だけ", "1〜8文字", 7),
    ("単語", "名詞を1つだけ", "1〜6文字", 3),
    ("くずれ", "打ち間違い・言いかけ・意味がよく分からない短い文", "2〜12文字", 4),
    ("長め", "2文くらいの少し長い話", "25〜50文字", 6),
    ("気持ち", "自分の気持ちを打ち明ける", "4〜25文字", 6),
    ("反応", "相づち・聞き返し・ツッコミ", "2〜12文字", 5),
]

SYSTEM = "あなたは日本語の会話データを作る作家です。指示どおりのJSONだけを出力します。"

INPUT_PROMPT = """次の場面にいる人が、チャットで「ずんだもん」（ずんだの妖精のキャラクター）に話しかけます。

場面: {place}／{time}／{weather}
話しかける人: {person}（気分: {mood}）
さっきあったこと: 「{seed}」に関係する小さな出来事（あなたが考える。発言の中で自然に触れてもよい）
表記: {style}

この人がチャットに打ちそうな発言を8つ書いてください。
①②は{i1}（{d1}、{l1}）
③④は{i2}（{d2}、{l2}）
⑤⑥は{i3}（{d3}、{l3}）
⑦⑧は{i4}（{d4}、{l4}）

条件:
- 人がチャット欄にふつうに打つような文。8つとも別の内容・別の言い回しにする。
- 記号は「。」「！」「？」「…」「ー」「〜」くらい。絵文字は使わない。
- 発言だけを書く。名前・番号・説明は書かない。

出力形式（JSONのみ、8個）: ["発言1", "発言2", ...]"""

REPLY_PROMPT = """次の各発言に、ずんだもんとして1回ずつ返事をしてください。

ずんだもんの話し方（必ず守る）:
{zunda}

返事の条件:
- 発言の中身にちゃんと反応する。質問には答え、報告には感想を言い、誘いには乗るか断る。
- 発言に書いていないことは、知っているふりをしない。
- 発言の後ろの【】の指示に従う。
  【自分の言葉で】… 発言の言葉をくり返さず、言いかえや自分の気持ち・行動で返す（例: {ex_own}）。
  【言葉を入れて】… 発言の中の言葉を1つ、返事の途中に自然に入れる。返事の最初でくり返さない（例: {ex_echo}）。
- 意味が分からない発言には、ずんだもんらしく聞き返す。
- 1つの返事は30文字以内。ひらがな多め。決まり文句ばかりにしない。どの発言にも使えるような返事にしない。
- 返事どうしで同じ言い回しをくりかえさない。

発言:
{inputs}

出力形式（JSONのみ。発言の番号順に返事だけ）: ["返事1", "返事2", ...]"""

EX_OWN = ["「テストで100点だった」→「すごいのだ！ボクも鼻が高いのだ！」",
          "「雨やまないね」→「おうちで遊ぶしかないのだ…」",
          "「ねむい」→「ボクもまぶたが重いのだ…」",
          "「新しい自転車買った！」→「かっこいいのだ！今度乗せてほしいのだ！」",
          "「上司に怒られた」→「それはつらかったのだ…ボクがなぐさめるのだ」",
          "「ずんだもんって何歳？」→「ひみつなのだ！でも気持ちはいつも元気なのだ」"]
EX_ECHO = ["「ねこ拾った」→「えっ、そのねこ、名前はもう決めたのだ？」",
           "「花火見に行こう」→「行くのだ！花火、近くで見たいのだ！」",
           "「明日テストなんだ」→「テスト、がんばるのだ！ボクが応援するのだ」",
           "「海に行ってきた」→「海、いいなあ！貝がらは拾ったのだ？」",
           "「スマホ落とした」→「ええっ、スマホ割れてないのだ？」",
           "「おやすみ」→「おやすみなのだ！いい夢見るのだ」"]


def kanji_ratio(s):
    return len(re.findall(r"[一-鿿]", s)) / max(1, len(s))


def norm_key(s):
    return re.sub(r"[\s、。！？!?…ー〜~・]", "", s)


def eval_inputs(root):
    """評価用の話しかけ（学習に入れない）"""
    out = set()
    for p in glob.glob(os.path.join(root, "eval", "inputs_*.tsv")):
        for line in open(p, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                out.add(norm_key(parts[1]))
    return out


def len_ok(s, spec):
    m = re.match(r"(\d+)〜(\d+)", spec)
    lo, hi = (int(m.group(1)), int(m.group(2))) if m else (1, 50)
    return max(1, lo // 2) <= len(s) <= hi + 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["inputs", "replies"])
    ap.add_argument("--inputs", help="replies で使う入力ファイル")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_replies", type=int, default=20)
    ap.add_argument("--queue_url", default=os.environ.get("V100_QUEUE_URL", "http://localhost:8081"))
    ap.add_argument("--priority", choices=["auto", "normal"], default="auto")
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--api_key_file", default="~/.config/tiny-ja-llm/api-key")
    ap.add_argument("--model", default="gemma4-26b-qat-q8")
    ap.add_argument("--client_id", default="tiny-ja-llm")
    ap.add_argument("--inflight", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2000)
    ap.add_argument("--queue_timeout", type=int, default=3600)
    ap.add_argument("--exec_timeout", type=int, default=600)
    ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--max_fail", type=int, default=5)
    ap.add_argument("--polite", action="store_true")
    ap.add_argument("--echo", type=float, default=0.3)
    ap.add_argument("--max_kanji", type=float, default=0.30, help="返事の漢字の割合の上限（v3 は会話全体で 0.15）")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = random.Random(args.seed)
    corpus = gc.Corpus(args.out)
    stats = {}
    here = os.path.dirname(os.path.abspath(__file__))

    def bump(k):
        stats[k] = stats.get(k, 0) + 1

    def request(prompt, temp):
        return {"model": args.model, "temperature": temp, "max_tokens": args.max_tokens, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}

    if args.stage == "inputs":
        banned = eval_inputs(here)
        print(f"評価用の話しかけ {len(banned)} 個は使わない")
        words = gc.WORDS + gc.TOPICS

        def make_req():
            person, style = rng.choice(PERSONS)
            intents = []
            while len(intents) < 4:
                it = rng.choices(INTENTS, weights=[x[3] for x in INTENTS])[0]
                if it not in intents:
                    intents.append(it)
            card = {"place": rng.choice(PLACES), "time": rng.choice(TIMES), "weather": rng.choice(WEATHERS),
                    "person": person, "mood": rng.choice(MOODS), "seed": rng.choice(words)}
            kw = dict(card, style=style)
            for k, it in enumerate(intents, 1):
                kw[f"i{k}"], kw[f"d{k}"], kw[f"l{k}"] = it[0], it[1], it[2]
            return request(INPUT_PROMPT.format(**kw), args.temperature), {"card": card, "intents": [x[0] for x in intents],
                                                                          "lens": [x[2] for x in intents]}

        def parse_fn(text, meta):
            out = []
            arr = json_array(text)
            if len(arr) != 8:
                bump("数が合わない")
            for i, s in enumerate(arr[:8]):
                if not isinstance(s, str):
                    bump("形式")
                    continue
                s = normalize(s)
                cat, ln = meta["intents"][i // 2], meta["lens"][i // 2]
                if not s or len(s) > 60 or re.search(r"[ＡＢ][：:]|ずんだもん[：:]|[①-⑧]", s):
                    bump("除外")
                    continue
                if not len_ok(s, ln):
                    bump("長さ")
                    continue
                if norm_key(s) in banned:
                    bump("評価用と同じ")
                    continue
                out.append({"turns": [s], "cat": cat, "card": meta["card"], "spec": SPEC_VERSION})
                bump("ok")
            bump("_calls")
            if stats["_calls"] % 50 == 0:
                print("  入力: " + ", ".join(f"{k} {v}" for k, v in stats.items() if k != "_calls"), flush=True)
            return out
    else:
        done = {json.loads(l)["turns"][0] for l in open(args.out, encoding="utf-8")} if os.path.exists(args.out) else set()
        rej_path = args.out + ".rejects.jsonl"
        if os.path.exists(rej_path):
            done |= {json.loads(l)["turns"][0] for l in open(rej_path, encoding="utf-8")}
        pool = [json.loads(l) for l in open(args.inputs, encoding="utf-8") if l.strip()]
        pool = [p for p in pool if p["turns"][0] not in done]
        rng.shuffle(pool)
        args.n = min(args.n, len(corpus.seen) + len(pool))

        def make_req():
            if not pool:
                return None
            items = [dict(pool.pop(), echo=rng.random() < args.echo) for _ in range(min(args.per_replies, len(pool)))]
            lines = "\n".join(f"{i + 1}. {it['turns'][0]}　【{'言葉を入れて' if it['echo'] else '自分の言葉で'}】"
                               for i, it in enumerate(items))
            p = REPLY_PROMPT.format(zunda=ZUNDA_OFFICIAL, inputs=lines, ex_own=rng.choice(EX_OWN), ex_echo=rng.choice(EX_ECHO))
            return request(p, 0.9), {"items": items}

        def parse_fn(text, meta):
            items, out = meta["items"], []
            replies = json_array(text)
            if len(replies) != len(items):
                bump("数が合わない")
                for it in items:   # 1 回だけ、別の組み合わせでやり直す
                    if not it.get("_retried"):
                        it["_retried"] = True
                        pool.append(it)
                return out
            with open(rej_path, "a", encoding="utf-8") as rej:
                for it, r in zip(items, replies):
                    if not isinstance(r, str) or "【" in r:
                        bump("形式")
                        continue
                    pair = [it["turns"][0], normalize(r)]
                    ok, why = check(["あ", pair[1]], partner_noda_ok=True)   # 返事だけを検査（話しかけは漢字まじりでもよい）
                    if why == "漢字が多い" and kanji_ratio(pair[1]) <= args.max_kanji:
                        ok, why = True, ""   # check() の 0.15 は v3 のひらがな前提。v4 は返事だけで --max_kanji まで許す
                    if not ok:
                        bump(why)
                        rej.write(json.dumps({"why": why, "turns": pair, "cat": it.get("cat")}, ensure_ascii=False) + "\n")
                        continue
                    ov = overlap(pair[0], pair[1])
                    if ov and not it["echo"]:
                        bump("くり返し")
                        rej.write(json.dumps({"why": "くり返し", "turns": pair, "cat": it.get("cat")}, ensure_ascii=False) + "\n")
                        continue
                    bump("ok")
                    out.append({"turns": pair, "cat": it.get("cat"), "card": it.get("card"), "echo_req": it["echo"],
                                "overlap": ov, "spec": SPEC_VERSION})
            bump("_calls")
            if stats["_calls"] % 20 == 0:
                print("  返事: " + ", ".join(f"{k} {v}" for k, v in stats.items() if k != "_calls"), flush=True)
            return out

    print(f"既存 {len(corpus.seen)} → 目標 {args.n}  stage={args.stage} spec={SPEC_VERSION}")
    (gc.run_direct if args.base_url else gc.run_queue)(args, corpus, make_req=make_req, parse_fn=parse_fn)
    print("内訳: " + ", ".join(f"{k} {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
