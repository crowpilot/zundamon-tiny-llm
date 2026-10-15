"""ブラウザ版（web/index.html）用に、成長段階ごとのチェックポイントを int8 にして models.js にまとめる。

  python make_web_models.py out/zs-a000 --steps 100:ことば0日目 251:くちぐせ 398:おしゃべり best:いちばん \
      --openers openers_zunda.txt --out ../web/models.js

best は growth.jsonl で val_loss_b（なければ val_loss）が最小のステップ。
"""
import argparse
import base64
import json
import os
import tempfile

from llama2c_format import read_llama2c, write_q8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--steps", nargs="+", required=True, help="step:表示名（step は数字か best）")
    ap.add_argument("--openers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = {json.loads(l)["step"]: json.loads(l) for l in open(os.path.join(args.run_dir, "growth.jsonl"), encoding="utf-8")}
    key = "val_loss_b" if all("val_loss_b" in r for r in recs.values()) else "val_loss"
    best = min(recs.values(), key=lambda r: r[key])["step"]
    run = json.load(open(os.path.join(args.run_dir, "run.json"), encoding="utf-8"))
    models = []
    for spec in args.steps:
        s, label = spec.split(":", 1)
        step = best if s == "best" else int(s)
        cfg, w = read_llama2c(os.path.join(args.run_dir, "ckpt", f"step_{step:07d}.bin"))
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as t:
            path = t.name
        write_q8(path, cfg, w, gs=32)
        data = open(path, "rb").read()
        os.unlink(path)
        models.append({"step": step, "label": label, "val": recs[step][key],
                       "b64": base64.b64encode(data).decode()})
        size_mb = len(data) / 1e6
        print(f"step {step:5d} {label:10s} {key}={recs[step][key]:.3f}  {size_mb:.2f}MB")
    tok = base64.b64encode(open(os.path.join(args.run_dir, "tok.bin"), "rb").read()).decode()
    openers = [l.strip() for l in open(args.openers, encoding="utf-8") if l.strip()]
    params = run["params"]
    payload = {"run": run["run"], "params": f"{params / 1e4:.0f}万", "sizeMB": f"{size_mb:.2f}",
               "lossKey": key, "tok": tok, "openers": openers, "models": models}
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("// 自動生成（train/make_web_models.py）。int8 のモデルとトークナイザを base64 で埋め込んだもの\n")
        f.write("window.ZUNDA_MODELS = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
