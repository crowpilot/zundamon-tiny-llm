"""ずんだもんコーパスを 推論キュー（Gemma 4）で作る。標準ライブラリのみ。

生成方式（--mode）:
  direct       1回の推論で8本の台詞だけを書く（最速）
  inline_plan  1回の推論で8本それぞれ「種3単語 → 短い構成 → 台詞」を書く（構成→本文の流れを1回に圧縮）
  plan_only    2回方式の1回目: 8本分の構成だけを書き、--out に構成を保存する
  from_plans   2回方式の2回目: --plans の構成から台詞を書く

  python3 gen_zunda.py --mode inline_plan --n 6000 --out corpus/zunda.jsonl --polite
  python3 gen_zunda.py --mode plan_only --n 40 --out corpus/plans.jsonl --polite
  python3 gen_zunda.py --mode from_plans --plans corpus/plans.jsonl --n 40 --out corpus/zunda_2call.jsonl --polite
  --persona persona/cast.json を付けると 人物設定 JSONでキャラクター説明・相手役を置き換える。

キューの扱い（台帳・同じ request_id での再投入・中断時の取り消し・ログに本文を出さない）は gen_corpus.py と同じ。
--polite: キューに他の利用者の待ちジョブがある間は新しく投入しない。

出力: 1行1会話 {"turns": [相手, ずんだもん, ...], "speakers": [...], "seeds": [...], "plan": {...}, ...}
条件に合わなかった会話は <out>.rejects.jsonl に理由つきで残す。
"""
import argparse
import json
import os
import random
import re

import gen_corpus as gc

SPEC_VERSION = "zunda-corpus/4.1"

PARTNERS = [   # (名前, 説明, 重み, のだを使ってよいか)
    ("ともだち", "ずんだもんと仲よしの人。ふつうの話し方", 40, False),
    ("はじめて会った人", "ずんだもんのことを知らない人", 10, False),
    ("近所の子ども", "元気な小さい子", 10, False),
    ("お店の人", "町のお店の人。ていねいな話し方", 8, False),
    ("東北ずん子", "ずんだもんの仲間の女の子。やさしくて、しっかり者", 12, False),
    ("ねこ", "しゃべるねこ。のんびりしている", 6, False),
    ("からす", "しゃべるからす。ものしり", 6, False),
    ("おばあちゃん", "近所のおばあちゃん。おだやか", 8, False),
]

FACETS = [
    ("いばる", "自分を大きく見せようとして、えらそうにする。でもすぐボロが出る", 10),
    ("ツイてない", "ちょっとツイてないことが起きて、しょんぼりする。でも立ち直る", 10),
    ("おさんぽ", "その辺をふらふら歩いていて見つけたものの話をする", 9),
    ("ずんだ", "ずんだ餅が大好きで、つい話したくなる", 10),
    ("すなお", "うれしいことがあって、すなおに喜ぶ", 9),
    ("こわがり", "こわいものにびくびくするけど、強がる", 7),
    ("ちょうし", "ほめられて調子に乗る", 7),
    ("はんせい", "失敗して、ちゃんとあやまる", 6),
    ("しりたがり", "知らないことを聞いて、知ったかぶりしそうになる", 8),
    ("てつだい", "だれかを手伝おうとはりきる", 8),
    ("なやみ", "小さなことでなやんで、相談する", 7),
    ("あそび", "遊びにさそう、またはさそわれる", 9),
]

ZUNDA_OFFICIAL = """- ずんだの妖精。一人称は「ボク」。たまに自分を「ずんだもん」とよぶ。
- 文のおわりは「〜のだ」「〜なのだ」が基本。たずねるときは「〜のだ？」「〜なのか？」、さそうときは「〜するのだ！」。
- 「です」「ます」は使わない。「わたし」「おれ」は使わない。
- すなおで、少しいばりんぼ。ちょっとツイていないことが多いけど、すぐ元気になる。"""

EXAMPLES = """ずんだもんの発言の例（そのまま使わない）:
「ボクにまかせるのだ！」
「えっ、ボクのおやつがないのだ……」
「それ、ボクも知ってたのだ。たぶん。」"""

# 台本生成の経験から、短い会話に効くものを抜き出したもの
WRITING = """書き方:
- 種の言葉から、その場で起きている一つの身近な出来事を思いつく。種を全部使わなくてよい。
- 相手の言葉や動作に、具体的に反応する。次の行動には、直前の出来事から分かる理由がある。関係のない話を「そういえば」でつながない。
- 途中に小さな変化（発見・勘ちがい・失敗など）を一つ入れ、その変化を受けて自然に終わる。最後に新しい話題を始めない。
- キャラクターの設定は全部使わない。この場面に関係するものだけ、さりげなく出す。性格を説明するセリフは書かない。
- 意味のない相づちや、同じ言いまわしのくりかえしで埋めない。"""

COMMON_RULES = """共通の条件:
- 発言は 相手 → ずんだもん → 相手 → ずんだもん … の順に交互。最初は相手。
- 「のだ」を使うのはずんだもんだけ。相手は「のだ」を使わず、その人らしい話し方をする。{partner_noda}
- 1回の発言は30文字以内。
- 漢字はほとんど使わず、ひらがなとカタカナで書く（小さい子が読める文）。
- 話者名（「ずんだもん：」など）・記号・説明文・ト書きは書かない。発言の文字だけ。"""

SYSTEM = "あなたはキャラクター会話の脚本家です。指示どおりのJSONだけを出力します。"


# ------------------------------------------------------------------ キャラクター（公式プロフィール or 人物設定 JSON）
class Cast:
    def __init__(self, persona_path=None, extra_weight=12):
        self.zunda = ZUNDA_OFFICIAL
        self.partners = list(PARTNERS)
        self.source = "official"
        if not persona_path:
            return
        data = json.load(open(persona_path, encoding="utf-8"))
        self.source = f"{data.get('format')} {data.get('exportedAt', '')}"
        people = {c["name"]: c for c in data["cast"]}
        z = people.get("ずんだもん")
        if z:
            p = z["profile"]
            selfs = "、".join(v["value"] for v in z.get("self", [])) or "ボク"
            extra = "\n".join(f"- {k}: {p[k]}" for k in ("summary", "voice", "behavior") if p.get(k))
            self.zunda = ZUNDA_OFFICIAL + f"\n- 一人称の候補: {selfs}\n制作設定（参考。全部を台詞に出さない）:\n{extra}"
        for name, c in people.items():
            if name == "ずんだもん":
                continue
            p = c["profile"]
            desc = " / ".join(p[k] for k in ("summary", "voice") if p.get(k))[:300]
            calls = [v["value"] for o in (z or {}).get("others", []) if o["target"] == name for v in o["variants"]]
            if calls:
                desc += f"（ずんだもんからの呼び方: {'、'.join(calls)}）"
            uses_noda = "のだ" in p.get("voice", "")
            self.partners.append((name, desc, extra_weight, uses_noda))
        cues = data.get("publicCues") or []
        self.cues = {tuple(c["between"]): c["publicCues"] for c in cues}

    def pick_partner(self, rng):
        return rng.choices(self.partners, weights=[w for _, _, w, _ in self.partners])[0]

    def noda_ok(self, name):
        return next((n for nm, _, _, n in self.partners if nm == name), False)


def _pick(rng, table):
    return rng.choices(table, weights=[t[2] for t in table])[0]


def make_spec(rng, cast):
    pname, pdesc, _, _ = cast.pick_partner(rng)
    fname, fdesc, _ = _pick(rng, FACETS)
    seeds = rng.sample(gc.WORDS + gc.TOPICS, 3)
    return {"partner": pname, "partner_desc": pdesc, "facet": fname, "facet_desc": fdesc,
            "mood": rng.choice(gc.MOODS), "seeds": seeds, "n_turns": rng.choice([4, 4, 6, 6, 8])}


def spec_line(i, s, plan=None):
    t = (f"{i + 1}本目: 相手「{s['partner']}（{s['partner_desc']}）」/ ずんだもんの様子「{s['facet_desc']}」"
         f" / ふんいき「{s['mood']}」/ 種「{'・'.join(s['seeds'])}」/ 発言は全部で{s['n_turns']}回")
    if s["facet"] != "ずんだ":
        t += " / ずんだ餅の話は出さない"
    if plan:
        t += f"\n   構成: 相手のしたいこと「{plan.get('partner_goal', '')}」/ ずんだもんのしたいこと「{plan.get('zunda_goal', '')}」" \
             f" / 途中の変化「{plan.get('turn', '')}」/ 終わり方「{plan.get('ending', '')}」"
    return t


PLAN_FIELDS = '{"partner_goal":"相手が今したいこと","zunda_goal":"ずんだもんが今したいこと","turn":"途中の小さな変化","ending":"その変化を受けた終わり方"}'


def build_prompt(mode, cast, specs, plans=None):
    noda = "（設定でのだを使う相手は例外）" if any(cast.noda_ok(s["partner"]) for s in specs) else ""
    head = f"ずんだもんと相手の、短い会話を{len(specs)}本つくってください。本ごとに、まったく別の場面・別の話にします。"
    body = [head, "", "ずんだもんの話し方（必ず守る）:", cast.zunda, "", EXAMPLES, "", WRITING, "",
            "\n".join(spec_line(i, s, plans[i] if plans else None) for i, s in enumerate(specs)), "",
            COMMON_RULES.format(partner_noda=noda), ""]
    if mode == "direct":
        body.append('出力形式（JSONのみ。本の順番は設定の番号どおり）:\n[["相手の発言", "ずんだもんの発言", ...], ...]')
    elif mode == "from_plans":
        body.append("各本の「構成」にそって台詞を書く。構成の文章をそのまま台詞にしない。")
        body.append('出力形式（JSONのみ。本の順番は設定の番号どおり）:\n[["相手の発言", "ずんだもんの発言", ...], ...]')
    elif mode == "inline_plan":
        body.append("各本ごとに、先に短い構成を決めてから台詞を書く。構成は「相手のしたいこと／ずんだもんのしたいこと／途中の変化／終わり方」"
                    "を／で区切った1行（全体で60文字以内）。構成の文章をそのまま台詞にしない。")
        body.append('出力形式（JSONのみ。本の順番は設定の番号どおり）:\n'
                    '[{"plan":"相手のしたいこと／ずんだもんのしたいこと／途中の変化／終わり方","lines":["相手の発言","ずんだもんの発言", ...]}, ...]')
    elif mode == "plan_only":
        body = [f"ずんだもんと相手の短い会話を{len(specs)}本つくるための、構成だけを決めてください。台詞はまだ書きません。",
                "", "ずんだもんの設定:", cast.zunda, "", WRITING, "",
                "\n".join(spec_line(i, s) for i, s in enumerate(specs)), "",
                "各項目は20文字以内。性格の紹介を目的にしない。",
                f'出力形式（JSONのみ。本の順番は設定の番号どおり）:\n[{PLAN_FIELDS}, ...]']
    return "\n".join(body)


def make_request(args, cast, rng, plan_pool=None):
    if args.mode == "from_plans":
        if not plan_pool:
            return None
        items = [plan_pool.pop(0) for _ in range(min(args.per_call, len(plan_pool)))]
        specs, plans = [it["spec"] for it in items], [it["plan"] for it in items]
    else:
        specs, plans = [make_spec(rng, cast) for _ in range(args.per_call)], None
    prompt = build_prompt(args.mode, cast, specs, plans)
    req = {"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens, "stream": False,
           "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    if not args.think:
        req["chat_template_kwargs"] = {"enable_thinking": False}
    drop = () if args.mode == "plan_only" else ("partner_desc", "facet_desc")   # 構成ファイルには2回目用に説明も残す
    metas = [{k: v for k, v in s.items() if k not in drop} for s in specs]
    if plans:
        for m, p in zip(metas, plans):
            m["plan"] = p
    return req, {"specs": metas, "spec_version": SPEC_VERSION, "mode": args.mode, "cast_source": cast.source}


# ------------------------------------------------------------------ 出力の解析と口調チェック
def json_array(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return salvage(m.group(0))


def salvage(text):
    """括弧の閉じ違いなどで全体が JSON として読めないとき、"lines" の配列を1本ずつ拾い直す。
    plan は直前の "plan" の文字列（または plan オブジェクト内の文字列）から取る。"""
    out = []
    for chunk in re.split(r'(?=\{\s*"plan")', text):
        lm = re.search(r'"lines"\s*:\s*(\[[^\[\]]*\])', chunk, flags=re.S)
        if not lm:
            continue
        try:
            lines = json.loads(lm.group(1))
        except json.JSONDecodeError:
            continue
        pm = re.search(r'"plan"\s*:\s*"([^"]*)"', chunk)
        plan = pm.group(1) if pm else "／".join(re.findall(r'"(?:partner_goal|zunda_goal|turn|ending)"\s*:\s*"([^"]*)"', chunk))
        out.append({"plan": plan, "lines": lines, "salvaged": True})
    return out


END = r"[。！？!?…ー〜～\s]*$"
NODA = re.compile(r"のだ|なのか[？?]|のか[？?]")
DESU = re.compile(r"(です|ます|でした|ました|でしょう|ません)" + END)
KANJI = re.compile(r"[一-鿿]")
BAD_FIRST = re.compile(r"わたし|私|あたし|おれ|俺|オレ")
A_NODA = re.compile(r"のだ")
SPEAKER = re.compile(r"^[^：:「」！？、。]{1,10}[：:]\s*")


def normalize(t):
    t = SPEAKER.sub("", str(t).strip())
    t = t.replace("僕", "ボク").replace("ぼく", "ボク")
    return re.sub(r"\s+", "", t)


def check(turns, partner_noda_ok=False):
    b, a = turns[1::2], turns[0::2]
    if len(turns) < 2 or not b:
        return False, "短すぎる"
    if any(len(t) > 40 for t in turns) or any(not t for t in turns):
        return False, "長すぎる発言"
    if any(DESU.search(t) for t in b):
        return False, "ずんだもんがです・ます"
    if any(BAD_FIRST.search(t) for t in b):
        return False, "一人称が違う"
    if sum(bool(NODA.search(t)) for t in b) < max(1, len(b) / 2):
        return False, "のだが少ない"
    if not partner_noda_ok and any(A_NODA.search(t) for t in a):
        return False, "相手がのだ口調"
    kanji = sum(len(KANJI.findall(t)) for t in turns) / max(1, sum(map(len, turns)))
    if kanji > 0.15:
        return False, "漢字が多い"
    return True, ""


class Filter:
    def __init__(self, out, cast, mode):
        self.rej_path = out + ".rejects.jsonl"
        self.cast, self.mode = cast, mode
        self.stats = {}

    def _bump(self, k):
        self.stats[k] = self.stats.get(k, 0) + 1

    def __call__(self, text, meta):
        items = json_array(text)
        specs = (meta or {}).get("specs") or []
        if self.mode == "plan_only":
            keep = []
            for i, p in enumerate(items):
                if isinstance(p, dict) and i < len(specs) and all(isinstance(p.get(k), str) for k in
                                                                 ("partner_goal", "zunda_goal", "turn", "ending")):
                    keep.append({"turns": [json.dumps(p, ensure_ascii=False)], "spec": specs[i], "plan": p})
                    self._bump("ok")
                else:
                    self._bump("構成の形式")
            return keep
        keep = []
        with open(self.rej_path, "a", encoding="utf-8") as rej:
            for i, it in enumerate(items):
                m = dict(specs[i]) if i < len(specs) else {}
                if isinstance(it, dict):
                    if isinstance(it.get("plan"), (dict, str)):
                        m["plan"] = it["plan"]
                    if it.get("salvaged"):
                        self._bump("拾い直し")
                    it = it.get("lines")
                if not isinstance(it, list) or not all(isinstance(t, str) for t in it):
                    self._bump("形式")
                    continue
                turns = [normalize(t) for t in it]
                noda_ok = self.cast.noda_ok(m.get("partner"))
                ok, why = check(turns, noda_ok)
                if not ok and why == "相手がのだ口調":
                    # 相手の口調が途中でうつった会話は、うつる直前（ずんだもんの発言で終わる位置）までを使う
                    cut = next(i for i in range(0, len(turns), 2) if A_NODA.search(turns[i]))
                    if cut >= 2 and check(turns[:cut], noda_ok)[0]:
                        turns, ok, why = turns[:cut], True, ""
                        self._bump("途中まで採用")
                self._bump(why or "ok")
                if ok:
                    sp = [m.get("partner", "相手"), "ずんだもん"] * len(turns)
                    keep.append({"turns": turns, **m, "speakers": sp[:len(turns)], "spec": SPEC_VERSION,
                                 "mode": self.mode})
                else:
                    rej.write(json.dumps({"why": why, "turns": turns, **m}, ensure_ascii=False) + "\n")
        print("  検査: " + ", ".join(f"{k} {v}" for k, v in sorted(self.stats.items(), key=lambda x: -x[1])), flush=True)
        return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["direct", "inline_plan", "plan_only", "from_plans"], default="inline_plan")
    ap.add_argument("--plans", help="from_plans で使う構成ファイル（plan_only の出力）")
    ap.add_argument("--persona", help="人物設定 JSON（cast: 名前・profile・呼び方）")
    ap.add_argument("--persona_weight", type=int, default=12, help="人物設定 JSON から読んだ相手役1人あたりの重み")
    ap.add_argument("--queue_url", default=os.environ.get("V100_QUEUE_URL", "http://localhost:8081"))
    ap.add_argument("--priority", choices=["auto", "normal"], default="auto",
                    help="auto: キューが background_priority_v1 を返せば background で投入（既定）。normal: 従来どおり")
    ap.add_argument("--base_url", default=None, help="指定するとキューを使わず OpenAI 互換 API を直接呼ぶ（例: http://localhost:8000/v1）")
    ap.add_argument("--api_key_file", default="~/.config/tiny-ja-llm/api-key")
    ap.add_argument("--model", default="gemma4-26b-qat-q8")
    ap.add_argument("--client_id", default="tiny-ja-llm")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--per_call", type=int, default=8)
    ap.add_argument("--inflight", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--queue_timeout", type=int, default=1800)
    ap.add_argument("--exec_timeout", type=int, default=600)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--max_fail", type=int, default=3)
    ap.add_argument("--polite", action="store_true", help="キューに他の待ちジョブがある間は新規投入しない")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="corpus/zunda.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cast = Cast(args.persona, args.persona_weight)
    rng = random.Random(args.seed)
    pool = None
    if args.mode == "from_plans":
        pool = [json.loads(l) for l in open(args.plans, encoding="utf-8") if l.strip()]
        args.n = min(args.n, len(pool))
    corpus = gc.Corpus(args.out)
    print(f"既存 {len(corpus.seen)} 本 → 目標 {args.n} 本  mode={args.mode} model={args.model} "
          f"cast={cast.source} spec={SPEC_VERSION}")
    (gc.run_direct if args.base_url else gc.run_queue)(args, corpus, make_req=lambda: make_request(args, cast, rng, pool),
                                                     parse_fn=Filter(args.out, cast, args.mode))


if __name__ == "__main__":
    main()
