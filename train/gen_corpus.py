"""ローカルLLM（Gemma 4 など）で、やさしい日本語の日常会話コーパスを量産する。標準ライブラリのみ。

■ 非同期ジョブ方式の推論キューで使う  ← 既定
  python3 gen_corpus.py --queue_url http://localhost:8081 --api_key_file ~/.config/tiny-ja-llm/api-key \
      --model gemma4-26b-qat-q8 --n 3000 --out corpus/daily.jsonl

  キュー契約に従う:
  - 1件ずつ投入（同時に抱えるジョブは --inflight 件まで。既定1 = 他の利用者と順番に交互）
  - request_id は投入前に台帳(<out>.jobs.jsonl)へ書き、通信が曖昧なときは「同じ request_id・同じ中身」で再投入
    （キュー側が同じジョブを返す）。失敗・中断したジョブを新しいIDで生成し直すことはしない
  - 待ち時間と推論時間は別扱い。中断(Ctrl+C)時は抱えているジョブを DELETE で取り消す
  - キー・プロンプト・出力はログに出さない
  - 再実行すると台帳から未完了ジョブを拾い直す（再送はしない）

■ 普通の OpenAI 互換 API（Mac の omlx など）で使う
  python3 gen_corpus.py --base_url http://localhost:8000/v1 --model ... --n 3000

出力: 1行1会話 {"turns": [...], "topic": ..., "words": [...]}
多様性のため、話題・関係・場所・ふんいき・必須単語・長さを毎回ランダムに組み合わせる（TinyStories と同じ発想）。
"""
import argparse
import hashlib
import json
import os
import random
import re
import signal
import sys
import time
import urllib.error
import urllib.request
import uuid

TOPICS = ["ごはん", "天気", "買い物", "学校", "しごと", "どうぶつ", "あそび", "でんしゃ", "ねむい", "びょういん",
          "おまつり", "そうじ", "りょうり", "ほん", "おんがく", "うみ", "やま", "ゆき", "誕生日", "ともだち",
          "しゅくだい", "スポーツ", "ゲーム", "おかし", "ひっこし", "まいご", "かぜ", "あさ", "よる", "やすみの日",
          "じてんしゃ", "うちゅう", "ロボット", "はな", "むし", "さかな", "おふろ", "てがみ", "でんわ", "しゃしん"]
RELATIONS = ["ともだちどうし", "おやとこ", "きょうだい", "せんせいとせいと", "となりの人", "店員とおきゃく",
             "おじいちゃんとまご", "どうりょう", "はじめて会った人", "ペットにはなしかける人と、こたえるペット"]
PLACES = ["いえ", "こうえん", "えき", "がっこう", "スーパー", "バスのなか", "かいしゃ", "びょういん", "としょかん", "どこか"]
MOODS = ["たのしい", "こまっている", "おどろいている", "ねむそう", "いそいでいる", "うれしい", "すこしかなしい", "ふつう"]
WORDS = ["りんご", "かさ", "ねこ", "いぬ", "くつ", "ぼうし", "でんしゃ", "パン", "ぎゅうにゅう", "かばん", "えんぴつ", "とけい",
         "まど", "そら", "ほし", "つき", "かぜ", "あめ", "みず", "おちゃ", "たまご", "さくら", "かみ", "はさみ", "いす",
         "つくえ", "テレビ", "ラジオ", "カメラ", "ボール", "なわとび", "すいか", "みかん", "おにぎり", "カレー", "うどん",
         "はし", "コップ", "まくら", "ふとん", "かぎ", "さいふ", "ちず", "きっぷ", "バス", "ふね", "ひこうき", "やさい",
         "はやい", "おそい", "おおきい", "ちいさい", "あつい", "さむい", "あまい", "からい", "しずか", "にぎやか",
         "はしる", "あるく", "ねる", "おきる", "たべる", "のむ", "よむ", "かく", "うたう", "わらう", "なく", "さがす"]

SYSTEM = "あなたは子ども向け教材の作家です。指示どおりのJSONだけを出力します。"
PROMPT = """二人の短い日常会話を{n}本つくってください。本ごとに、下の設定にそって、まったく別の場面・別の話にします。

{specs}

共通の条件:
- 1回の発言は30文字以内。
- ひらがな中心。漢字は小学1〜2年生でならうものだけ。カタカナ語は少なめ。
- かんたんな言葉だけでも、会話としてちゃんと意味が通るようにする。
- 話者名・記号・説明文・ト書きは書かない。発言の文字だけ。
- 同じ言いまわしをくりかえさない。

出力形式（JSONのみ。本の順番は設定の番号どおり）:
[["Aの発言", "Bの発言", "Aの発言", ...], ...]"""

TERMINAL = {"succeeded", "failed", "cancelled", "expired", "interrupted"}


def make_request(model, per_call, temperature, max_tokens, think=False):
    rng = random.Random()
    specs, metas = [], []
    for i in range(per_call):
        topic, words = rng.choice(TOPICS), rng.sample(WORDS, 2)
        specs.append(f"{i + 1}本目: 話題「{topic}」/ 関係「{rng.choice(RELATIONS)}」/ 場所「{rng.choice(PLACES)}」/ "
                     f"ふんいき「{rng.choice(MOODS)}」/ 使う言葉「{'、'.join(words)}」/ {rng.choice([2, 4, 4, 6, 6, 8, 10])}回ほどのやりとり")
        metas.append({"topic": topic, "words": words})
    prompt = PROMPT.format(n=per_call, specs="\n".join(specs))
    req = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "stream": False,
           "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    if not think:   # Gemma 4 / Qwen の思考モードを切る（llama.cpp のテンプレート引数）
        req["chat_template_kwargs"] = {"enable_thinking": False}
    return req, {"specs": metas}


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\[\s*\[.*\]\s*\]", text, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for conv in data:
        if not isinstance(conv, list):
            continue
        turns = [re.sub(r"^[ＡＢAB][：:]\s*", "", str(t)).strip() for t in conv]
        turns = [t for t in turns if t]
        if 2 <= len(turns) <= 14 and all(len(t) <= 60 for t in turns):
            out.append(turns)
    return out


class Corpus:
    def __init__(self, path):
        self.path = path
        self.seen = set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    self.seen.add(self._h(json.loads(line)["turns"]))
                except (json.JSONDecodeError, KeyError):
                    pass

    @staticmethod
    def _h(turns):
        return hashlib.md5("\x1f".join(turns).encode()).hexdigest()

    def add(self, convs, meta):
        n = 0
        specs = meta.get("specs") or []
        with open(self.path, "a", encoding="utf-8") as f:
            for i, conv in enumerate(convs):
                if isinstance(conv, dict):   # {"turns": [...], その他の付帯情報} をそのまま書く形
                    turns, m = conv["turns"], {k: v for k, v in conv.items() if k != "turns"}
                else:
                    turns, m = conv, (specs[i] if i < len(specs) else {})
                h = self._h(turns)
                if h in self.seen:
                    continue
                self.seen.add(h)
                f.write(json.dumps({"turns": turns, **m}, ensure_ascii=False) + "\n")
                n += 1
        return n


# ------------------------------------------------------------------ queue mode
class Queue:
    def __init__(self, base, key, timeout=30):
        self.base, self.key, self.timeout = base.rstrip("/"), key, timeout

    def call(self, method, path, body=None, token=None):
        h = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}
        if token:
            h["X-V100-Job-Token"] = token
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}"), r.headers
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read() or b"{}")
            except json.JSONDecodeError:
                payload = {}
            return e.code, payload, e.headers


class Ledger:
    """request_id と投入内容を投入前に残す台帳（曖昧な投入を同じIDで再投入するため）"""

    def __init__(self, path):
        self.path = path
        self.jobs = {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                self.jobs.setdefault(rec["request_id"], {}).update(rec)

    def put(self, rid, **kw):
        self.jobs.setdefault(rid, {"request_id": rid}).update(kw)
        # job_token を含むので本人だけ読めるように
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps({"request_id": rid, **kw}, ensure_ascii=False) + "\n")

    def open_jobs(self):
        return [j for j in self.jobs.values() if j.get("state") not in TERMINAL | {"collected"}]


def queue_busy(q):
    """他の利用者のジョブが待っているか（--polite のとき、待ちがあれば投入を見送る）"""
    try:
        st, s, _ = q.call("GET", "/v1/queue/status")
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return True
    if st != 200 or not s.get("accepting", True):
        return True
    return int(s.get("queued") or 0) > 0


def run_queue(args, corpus, make_req=None, parse_fn=None):
    make_req = make_req or (lambda: make_request(args.model, args.per_call, args.temperature, args.max_tokens, args.think))
    parse_fn = parse_fn or (lambda text, meta: parse(text))
    key = open(os.path.expanduser(args.api_key_file)).read().strip()
    q = Queue(args.queue_url, key)
    st, status, _ = q.call("GET", "/v1/queue/status")
    if st != 200:
        sys.exit(f"queue status HTTP {st}: {status.get('error', status)}")
    caps = status.get("capabilities", {})
    print(f"queue: {status.get('contract_version')} accepting={status.get('accepting')} "
          f"queued={status.get('queued')} running={status.get('running')} caps={sorted(k for k, v in caps.items() if v)}")
    if not status.get("accepting", True):
        sys.exit("キューが受付停止中です")
    # 優先度: キューが background_priority_v1 を返すときだけ background で投入し、サーバー側の待機に任せる。
    # 返さないときは従来どおり（--polite なら他の待ちがある間は投入しない）。通常優先度の大量投入へは自動で切り替えない。
    use_bg = bool(caps.get("background_priority_v1")) and getattr(args, "priority", "auto") == "auto"
    if use_bg:
        print("  background 優先度で投入します（実行期限は 120 秒以内に制限）")
    st, models, _ = q.call("GET", "/v1/models")
    ids = [m.get("id") for m in models.get("data", [])] if st == 200 else []
    if ids and args.model not in ids:
        sys.exit(f"モデル {args.model} がありません。候補: {ids}")

    ledger = Ledger(args.out + ".jobs.jsonl")
    inflight = ledger.open_jobs()
    if inflight:
        print(f"台帳から未完了ジョブ {len(inflight)} 件を引き継ぎます（再送はしません）")
    stop = {"flag": False}

    def on_sigint(*_):
        stop["flag"] = True
        print("\n中断要求: 抱えているジョブを取り消して終了します")
    signal.signal(signal.SIGINT, on_sigint)

    fails_in_row, t0, made = 0, time.time(), 0

    def submit(job):
        # 再投入は、台帳に記録した最初の投入と同じ中身（priority・期限も同じ）で行う
        body = {"client_id": args.client_id, "request_id": job["request_id"], "request": job["request"],
                "queue_timeout_s": job.get("queue_timeout", args.queue_timeout),
                "execution_timeout_s": job.get("exec_timeout", args.exec_timeout)}
        if job.get("priority"):
            body["priority"] = job["priority"]
        for _ in range(20):   # 同じ request_id・同じ中身でのみ再投入（キュー側で重複排除される）
            try:
                st, r, hdr = q.call("POST", "/v1/queue/jobs", body)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print(f"  投入の応答が不明（{type(e).__name__}）→ 同じ request_id で確認し直します")
                time.sleep(5)
                continue
            if st in (200, 202):
                ledger.put(job["request_id"], job_id=r["job_id"], job_token=r["job_token"], state=r["state"])
                job.update(job_id=r["job_id"], job_token=r["job_token"], state=r["state"])
                return True
            if st == 429:
                wait = int((hdr or {}).get("Retry-After", 30))
                print(f"  キュー満杯 → {wait}s 待って同じ request_id で再投入")
                time.sleep(wait)
                continue
            print(f"  投入拒否 HTTP {st}: {r.get('error', {}).get('code', r)}")
            ledger.put(job["request_id"], state="failed", error=f"submit_{st}")
            return False
        ledger.put(job["request_id"], state="failed", error="submit_unresolved")
        return False

    yielded = 0
    while True:
        # 新しいジョブを足す
        waiting = False
        while not stop["flag"] and len(inflight) < args.inflight and len(corpus.seen) < args.n:
            if getattr(args, "polite", False) and not use_bg and queue_busy(q):
                waiting = True   # 他の利用者が待っている間は投入しない
                yielded += 1
                if yielded % 20 == 1:
                    print("  キューに他の待ちがあるので投入を見送り中", flush=True)
                break
            made_req = make_req()
            if made_req is None:   # 材料（構成など）が尽きた
                args.n = len(corpus.seen)
                break
            req, meta = made_req
            job = {"request_id": str(uuid.uuid4()), "request": req, "meta": meta,
                   "queue_timeout": args.queue_timeout, "exec_timeout": min(args.exec_timeout, 120) if use_bg else args.exec_timeout}
            if use_bg:
                job["priority"] = "background"
            ledger.put(job["request_id"], request=req, meta=meta, state="submitting", **{k: job[k] for k in
                       ("queue_timeout", "exec_timeout", "priority") if k in job})
            if submit(job):
                inflight.append(job)
            else:
                fails_in_row += 1
                break
        if not inflight:
            if waiting and not stop["flag"]:
                time.sleep(30)
                continue
            break
        # 状態を見る
        time.sleep(args.poll)
        for job in list(inflight):
            if "job_id" not in job:   # 前回の実行で投入結果が不明のまま終わったもの → 同じIDで投入し直す
                if not submit(job):
                    inflight.remove(job)
                continue
            if stop["flag"] and job.get("state") not in TERMINAL:
                q.call("DELETE", f"/v1/queue/jobs/{job['job_id']}", token=job["job_token"])
            try:
                st, s, _ = q.call("GET", f"/v1/queue/jobs/{job['job_id']}", token=job["job_token"])
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                continue
            if st != 200:
                continue
            job["state"] = s["state"]
            if s["state"] not in TERMINAL:
                continue
            inflight.remove(job)
            if s["state"] == "succeeded":
                body = s.get("response", {}).get("body") or {}
                if isinstance(body, str):
                    body = json.loads(body)
                ch = (body.get("choices") or [{}])[0]
                text = ch.get("message", {}).get("content", "") or ""
                added = corpus.add(parse_fn(text, job.get("meta", {})), job.get("meta", {}))
                if not added:
                    why = "本文が空（思考で上限を使い切った可能性）" if not text.strip() else "JSONとして読めない/条件外"
                    print(f"  取り出し0件: {why}, finish={ch.get('finish_reason')}")
                made += 1
                fails_in_row = 0 if added else fails_in_row + 1
                ledger.put(job["request_id"], state="collected", added=added)
                el = time.time() - t0
                print(f"{len(corpus.seen)}/{args.n} 本  (+{added}, ジョブ{made}件, {el/60:.1f}分, "
                      f"推論 {_dur(s)}, 待ち {_wait(s)})", flush=True)
            else:
                fails_in_row += 1
                err = (s.get("error") or {}).get("code", s["state"])
                ledger.put(job["request_id"], state=s["state"], error=err)
                print(f"  ジョブ {s['state']}: {err}（再送しません）")
        if fails_in_row >= args.max_fail:
            print(f"連続 {fails_in_row} 件うまくいかなかったので止めます")
            stop["flag"] = True
        if stop["flag"] and not any(j.get("state") not in TERMINAL for j in inflight):
            break
        if len(corpus.seen) >= args.n and not inflight:
            break
    print(f"完了: {len(corpus.seen)} 本 → {args.out}")


def _ts(x):
    if not x:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return time.mktime(time.strptime(x[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None


def _dur(s):
    a, b = _ts(s.get("started_at")), _ts(s.get("finished_at"))
    return f"{b - a:.0f}s" if a and b else "?"


def _wait(s):
    a, b = _ts(s.get("submitted_at")), _ts(s.get("started_at"))
    return f"{b - a:.0f}s" if a and b else "?"


# ------------------------------------------------------------------ direct OpenAI mode
def run_direct(args, corpus, make_req=None, parse_fn=None):
    """キューを使わず OpenAI 互換 API（/v1/chat/completions）を直接呼ぶ。1件ずつ順番に。"""
    make_req = make_req or (lambda: make_request(args.model, args.per_call, args.temperature, args.max_tokens, args.think))
    parse_fn = parse_fn or (lambda text, meta: parse(text))
    key = open(os.path.expanduser(args.api_key_file)).read().strip() \
        if args.api_key_file and os.path.exists(os.path.expanduser(args.api_key_file)) else ""
    made = 0
    while len(corpus.seen) < args.n:
        made_req = make_req()
        if made_req is None:
            break
        req, meta = made_req
        h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
        r = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
                                   data=json.dumps(req).encode(), headers=h)
        try:
            with urllib.request.urlopen(r, timeout=args.exec_timeout) as resp:
                text = json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print(f"  失敗: {type(e).__name__}（止めます。再実行で続きから）")
            break
        added = corpus.add(parse_fn(text, meta), meta)
        made += 1
        print(f"{len(corpus.seen)}/{args.n} 本  (+{added})", flush=True)
    print(f"完了: {len(corpus.seen)} 本 → {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue_url", default=os.environ.get("V100_QUEUE_URL", "http://localhost:8081"))
    ap.add_argument("--base_url", default=None, help="指定するとキューを使わず直接 OpenAI 互換APIへ")
    ap.add_argument("--api_key_file", default="~/.config/tiny-ja-llm/api-key")
    ap.add_argument("--model", default="gemma4-26b-qat-q8")
    ap.add_argument("--client_id", default="tiny-ja-llm")
    ap.add_argument("--n", type=int, default=3000, help="目標の会話本数")
    ap.add_argument("--per_call", type=int, default=8, help="1ジョブで作らせる本数")
    ap.add_argument("--inflight", type=int, default=1, help="同時に抱えるジョブ数（キューは直列なので1で十分）")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--think", action="store_true", help="思考モードを切らない（既定は切る）")
    ap.add_argument("--queue_timeout", type=int, default=1800)
    ap.add_argument("--exec_timeout", type=int, default=600)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--max_fail", type=int, default=3)
    ap.add_argument("--out", default="corpus/daily.jsonl")
    ap.add_argument("--polite", action="store_true", help="キューに他の待ちジョブがある間は新規投入しない")
    ap.add_argument("--priority", choices=["auto", "normal"], default="auto")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    corpus = Corpus(args.out)
    print(f"既存 {len(corpus.seen)} 本 → 目標 {args.n} 本  model={args.model}")
    if args.base_url:
        run_direct(args, corpus)
    else:
        run_queue(args, corpus)


if __name__ == "__main__":
    main()
