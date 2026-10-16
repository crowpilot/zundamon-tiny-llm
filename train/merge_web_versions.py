"""make_web_models.py で作った models.js を複数まとめ、ブラウザ版で「版」を切り替えられるようにする。

  python merge_web_versions.py --out ../web/models.js \
      --version models_base.js:最新版:all \
      --version models_first.js:最初の版:best

--version は「models.js:表示名:残す段階」。残す段階は all（全部）か best（最後の「いちばん」だけ）。
最初に書いた版がデフォルト。話しかけ一覧は最初の版のものを使う。
"""
import argparse
import json
import os


def load(path):
    s = open(path, encoding="utf-8").read()
    d = json.loads(s[s.index("{"):s.rindex("}") + 1])
    if "versions" in d:
        raise SystemExit(f"{path} はすでにまとめた形式です（make_web_models.py の出力を指定してください）")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", action="append", required=True, help="models.js:表示名:all|best")
    args = ap.parse_args()
    versions, openers = [], None
    for spec in args.version:
        path, name, keep = spec.rsplit(":", 2)
        d = load(path)
        models = d["models"] if keep == "all" else d["models"][-1:]
        versions.append({"name": name, "run": d["run"], "params": d["params"], "sizeMB": d["sizeMB"],
                         "lossKey": d["lossKey"], "tok": d["tok"], "models": models})
        openers = openers or d["openers"]
        print(f"{name}: {d['run']} {d['params']} 段階 {[m['label'] for m in models]}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("// 自動生成（train/merge_web_versions.py）。版ごとの int8 モデルとトークナイザを base64 で埋め込んだもの\n")
        f.write("window.ZUNDA_MODELS = " + json.dumps({"versions": versions, "openers": openers}, ensure_ascii=False) + ";\n")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
