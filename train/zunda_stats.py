"""ずんだもんコーパス（または学習中モデルの生成文）の口調・偏りを数える。

  python3 zunda_stats.py corpus/zunda.jsonl            # コーパスの集計
  python3 zunda_stats.py out/zunda-tiny/growth.jsonl   # 成長記録: チェックポイントごとの「のだ率」など

見るもの:
  のだ率      … ずんだもんの発言のうち「のだ／なのか？」を含む割合（口調が身についたか）
  ボク率      … 一人称「ボク」を含む割合
  です・ます率 … 口調くずれ
  ずんだ率    … 会話に「ずんだ」が出てくる割合（設定に寄りすぎていないか。目安 15% 前後）
  重複率      … 同じ発言の使い回し（多様性）
"""
import collections
import json
import re
import sys

NODA = re.compile(r"のだ|なのか[？?]|のか[？?]")
DESU = re.compile(r"(です|ます|でした|ました|でしょう|ません)[。！？!?…ー〜～\s]*$")


def b_lines_from_text(text):
    """生成文（Ａ：…Ｂ：… 形式）から Ｂ の発言を取り出す"""
    return [s.strip() for s in re.findall(r"Ｂ：([^Ａ＃\n]*)", text) if s.strip()]


def rates(lines):
    n = len(lines) or 1
    return {"n": len(lines),
            "のだ率": sum(bool(NODA.search(t)) for t in lines) / n,
            "ボク率": sum("ボク" in t for t in lines) / n,
            "です・ます率": sum(bool(DESU.search(t)) for t in lines) / n,
            "重複率": 1 - len(set(lines)) / n}


def corpus_stats(path):
    convs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    b = [t for c in convs for t in c["turns"][1::2]]
    r = rates(b)
    r["会話数"] = len(convs)
    r["平均発言数"] = sum(len(c["turns"]) for c in convs) / max(1, len(convs))
    r["ずんだ率"] = sum(any("ずんだ" in t for t in c["turns"]) for c in convs) / max(1, len(convs))
    r["平均文字数(ずんだもん)"] = sum(map(len, b)) / max(1, len(b))
    alltxt = "".join(t for c in convs for t in c["turns"])
    r["漢字率"] = len(re.findall(r"[\u4e00-\u9fff]", alltxt)) / max(1, len(alltxt))
    r["文字の種類"] = len(set(alltxt))
    for k, v in r.items():
        print(f"{k:16s} {v:.3f}" if isinstance(v, float) else f"{k:16s} {v}")
    for key in ("partner", "facet"):
        c = collections.Counter(x.get(key, "?") for x in convs)
        print(f"{key}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    ends = collections.Counter(re.sub(r"[。！？!?…ー〜～]+$", "", t)[-3:] for t in b)
    print("ずんだもんの語尾(末尾3文字) 上位: " + ", ".join(f"{k}:{v}" for k, v in ends.most_common(12)))
    print("よく出る発言: " + " / ".join(f"{k}({v})" for k, v in collections.Counter(b).most_common(8)))


def growth_stats(path):
    print(f"{'step':>7s} {'のだ率':>6s} {'ボク率':>6s} {'ですます':>6s} {'重複':>6s}  n")
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        texts = []
        for s in rec.get("samples", []):
            if not isinstance(s, dict):
                continue
            head = "" if s.get("prompt", "").startswith("<") else s.get("prompt", "")   # 生成文はプロンプトの続きだけ
            for k in ("greedy", "sampled", "text"):
                v = s.get(k)
                if isinstance(v, str):
                    texts.append(head + v)
        r = rates([t for x in texts for t in b_lines_from_text(x)])
        print(f"{rec.get('step', '?'):>7} {r['のだ率']:6.2f} {r['ボク率']:6.2f} {r['です・ます率']:6.2f} {r['重複率']:6.2f}  {r['n']}")


if __name__ == "__main__":
    p = sys.argv[1]
    (growth_stats if "growth" in p else corpus_stats)(p)
