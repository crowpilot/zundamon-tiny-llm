"""再開・区間実行・停止の CPU 検証。

  cd train && OMP_NUM_THREADS=2 .venv/bin/python tests/test_resume.py [--keep]

隔離した小さいデータ（sample_corpus.jsonl）を一時ディレクトリに作り、train.py を子プロセスで実行して確認する:
  1. 連続実行 と 区間で切って再開した実行 で、最終の model / optimizer / 乱数状態 / 次のバッチ / 学習率 が一致
  2. 停止ファイルによる協調停止（exit 11）→ 再開で同じ結果
  3. SIGTERM による協調停止（exit 11）→ 再開で同じ結果
  4. 保存の途中で落ちる（LATEST 更新前）→ 直前の確定 checkpoint が残り、再開で同じ結果、残骸は掃除
  5. checkpoint 破損 → 再開を拒否（exit 2）
  6. 学習計画の不一致（lr 変更）→ 再開を拒否（exit 2）
  7. 確定 checkpoint があるのに --resume なし → 拒否（exit 2）
  8. 履歴（growth.jsonl / loss.jsonl）に重複や確定より先の行が残らない
  9. 最終 step の成長記録の途中で停止要求 → 完了にしない（exit 11）→ 再開で成果物をそろえてから完了
 10. 評価・生成の途中で区間期限 → 追加の更新をせずに保存して exit 10 → 再開で同じ結果
 11. 最終 step で区間期限 → 完了にしない → 再開で完了
 12. 最終処理（model.bin 書き出し前）で落ちる → 再開で完了・成長記録の重複なし
比較するもの（CPU で比べられる範囲）: model / optimizer / GradScaler（CPU では無効状態）/ NumPy 学習乱数 / Torch CPU 乱数 /
  running loss / 成長記録の val_loss・greedy と sampled の生成文・候補 / loss 履歴の値。
  CUDA 乱数・GPU 計算・AMP は対象外（GPU 検証前）。
結果・実測値・対象コードの SHA256・環境を tests/last_result.json に書く。
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"), "PYTHONHASHSEED": "0"}
COMMON = ["--device", "cpu", "--preset", "nano", "--seq_len", "32", "--batch_size", "8", "--max_iters", "60",
          "--eval_iters", "2", "--log_interval", "5", "--max_new", "8", "--n_cand", "2", "--ckpt_per_decade", "3"]
results = {"checks": {}, "measurements": {}}


def check(name, cond, detail=""):
    results["checks"][name] = {"ok": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))


def run(out, name, *extra, env=None, wait=True):
    cmd = [PY, "train.py", "--data", DATA, "--out", out, "--run", name, "--prompts", PROMPTS,
           "--events", os.path.join(out, f"{name}.events.jsonl"), *COMMON, *extra]
    p = subprocess.Popen(cmd, cwd=HERE, env=env or ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait:
        return p
    o, _ = p.communicate(timeout=900)
    return p.returncode, o


def events(out, name):
    p = os.path.join(out, f"{name}.events.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def final_state(out, name):
    st = os.path.join(out, name, "state")
    lt = json.load(open(os.path.join(st, "LATEST")))
    return torch.load(os.path.join(st, lt["name"], "state.pt"), weights_only=False), lt


def same_tensors(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tensors(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tensors(x, y) for x, y in zip(a, b))
    return a == b


def compare(ref, out, name, label):
    a, _ = ref
    b, _ = final_state(out, name)
    check(f"{label}: step", a["step"] == b["step"], f"{a['step']} vs {b['step']}")
    check(f"{label}: model 完全一致", same_tensors(a["model"], b["model"]))
    check(f"{label}: optimizer 完全一致", same_tensors(a["optimizer"], b["optimizer"]))
    check(f"{label}: 学習用乱数(=次のバッチ)一致", a["rng"]["numpy_train"] == b["rng"]["numpy_train"])
    check(f"{label}: running loss 一致", a["running_loss"] == b["running_loss"])
    check(f"{label}: Torch CPU 乱数一致", torch.equal(a["rng"]["torch_cpu"], b["rng"]["torch_cpu"]))
    check(f"{label}: GradScaler 状態一致（CPU では無効状態どうしの比較。AMP の検証ではない）",
          same_tensors(a["scaler"], b["scaler"]), f"{a['scaler']}")
    ga = [json.loads(l) for l in open(os.path.join(REF_OUT, "ref", "growth.jsonl"))]
    gb = [json.loads(l) for l in open(os.path.join(out, name, "growth.jsonl"))]
    sa, sb = [g["step"] for g in ga], [g["step"] for g in gb]
    check(f"{label}: growth の step 列一致・重複なし", sa == sb and len(sb) == len(set(sb)), f"{sb}")
    key = lambda g: (g["val_loss"], g.get("val_loss_b"), [(s["greedy"], s["sampled"], s["cands"]) for s in g["samples"]])  # noqa: E731
    check(f"{label}: growth の val_loss・生成文（greedy/sampled）・候補一致", [key(g) for g in ga] == [key(g) for g in gb])
    la = [(r["step"], r["loss"]) for r in map(json.loads, open(os.path.join(REF_OUT, "ref", "loss.jsonl")))]
    lb = [(r["step"], r["loss"]) for r in map(json.loads, open(os.path.join(out, name, "loss.jsonl")))]
    check(f"{label}: loss 履歴（step と値）一致・重複なし", la == lb, f"{len(la)} vs {len(lb)}")
    for f in ("model.bin", "last.pt"):
        check(f"{label}: {f} がある", os.path.exists(os.path.join(out, name, f)))
    check(f"{label}: model.bin 一致", open(os.path.join(REF_OUT, "ref", "model.bin"), "rb").read() ==
          open(os.path.join(out, name, "model.bin"), "rb").read())


def wait_progress(out, name, min_step, proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ev = [e for e in events(out, name) if e["type"] == "progress"]
        if ev and ev[-1]["step"] >= min_step:
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def main():
    global DATA, PROMPTS, REF_OUT
    keep = "--keep" in sys.argv
    tmp = tempfile.mkdtemp(prefix="tiny-resume-test-", dir=HERE)
    DATA = os.path.join(tmp, "data")
    PROMPTS = os.path.join(tmp, "prompts.txt")
    open(PROMPTS, "w").write("<BOS>\nＡ：こんにちは。Ｂ：\n")
    subprocess.run([PY, "prepare.py", "sample_corpus.jsonl", "--out", DATA, "--vocab", "400"], cwd=HERE, env=ENV,
                   check=True, capture_output=True)
    REF_OUT = os.path.join(tmp, "o")
    O = REF_OUT
    try:
        # 0. 基準: 連続実行
        t0 = time.time()
        code, o = run(O, "ref")
        check("連続実行が完了 (exit 0)", code == 0, o[-300:] if code else f"{time.time() - t0:.1f}s")
        ref = final_state(O, "ref")
        check("連続実行の最終 step = max_iters", ref[0]["step"] == 60)

        # 1. 区間実行（25 → 20 → 残り）
        c1, _ = run(O, "seg", "--segment_steps", "25")
        c2, _ = run(O, "seg", "--resume", "--segment_steps", "20")
        c3, _ = run(O, "seg", "--resume")
        check("区間終了 exit 10 → 10 → 完了 0", (c1, c2, c3) == (10, 10, 0), f"{(c1, c2, c3)}")
        ex = [e for e in events(O, "seg") if e["type"] == "exit"]
        check("exit イベントに next_step と checkpoint", [e["next_step"] for e in ex] == [25, 45, 60]
              and all(e["checkpoint"] and e["checkpoint"]["sha256"] for e in ex), f"{[e['next_step'] for e in ex]}")
        compare(ref, O, "seg", "区間再開")

        # 2. 停止ファイル
        stopf = os.path.join(tmp, "STOP")
        p = run(O, "stopf", "--stop_file", stopf, wait=False)
        wait_progress(O, "stopf", 21, p)
        t_req = time.time()
        open(stopf, "w").close()
        p.communicate(timeout=300)
        lat = time.time() - t_req
        os.remove(stopf)
        ex = [e for e in events(O, "stopf") if e["type"] == "exit"][-1]
        check("停止ファイル → exit 11", p.returncode == 11 and ex["reason"].startswith("stop_requested"), ex["reason"])
        results["measurements"]["stop_file_to_exit_s"] = round(lat, 3)
        c, _ = run(O, "stopf", "--resume")
        check("停止後の再開が完了", c == 0)
        compare(ref, O, "stopf", "停止ファイル→再開")

        # 3. SIGTERM
        p = run(O, "sigterm", wait=False)
        wait_progress(O, "sigterm", 31, p)
        t_req = time.time()
        p.send_signal(signal.SIGTERM)
        p.communicate(timeout=300)
        results["measurements"]["sigterm_to_exit_s"] = round(time.time() - t_req, 3)
        check("SIGTERM → exit 11", p.returncode == 11)
        c, _ = run(O, "sigterm", "--resume")
        check("SIGTERM 後の再開が完了", c == 0)
        compare(ref, O, "sigterm", "SIGTERM→再開")

        # 4. 保存途中のクラッシュ（LATEST 更新前 / state 書き込み直後）
        run(O, "crash", "--segment_steps", "20")
        for fault, seg in (("before_latest", "15"), ("after_state", "10")):
            c, _ = run(O, "crash", "--resume", "--segment_steps", seg, env={**ENV, "TINYJA_FAULT": fault})
            lt = json.load(open(os.path.join(O, "crash", "state", "LATEST")))
            check(f"保存中クラッシュ({fault}) → LATEST は step 20 のまま", c == 99 and lt["step"] == 20, f"exit={c} LATEST={lt['step']}")
        c, _ = run(O, "crash", "--resume")
        left = [d for d in os.listdir(os.path.join(O, "crash", "state")) if d.startswith(".tmp-")]
        check("クラッシュ後の再開が完了・一時ファイル掃除", c == 0 and not left, f"exit={c} tmp={left}")
        compare(ref, O, "crash", "クラッシュ→再開")

        # 5. 破損
        shutil.copytree(os.path.join(O, "seg"), os.path.join(O, "corrupt"))
        st = os.path.join(O, "corrupt", "state")
        lt = json.load(open(os.path.join(st, "LATEST")))
        sp = os.path.join(st, lt["name"], "state.pt")
        b = bytearray(open(sp, "rb").read())
        b[len(b) // 2] ^= 0xFF
        open(sp, "wb").write(b)
        c, o = run(O, "corrupt", "--resume")
        check("破損した checkpoint の再開を拒否 (exit 2)", c == 2 and "sha256" in o, o.strip().splitlines()[-1] if o else "")

        # 6. 計画の不一致
        run(O, "mismatch", "--segment_steps", "10")
        c, o = run(O, "mismatch", "--resume", "--lr", "2e-3")
        check("計画不一致（lr）の再開を拒否 (exit 2)", c == 2 and "plan_hash" in o, o.strip().splitlines()[-1] if o else "")

        # 7. --resume なしで既存 run を上書きしようとする
        c, o = run(O, "mismatch")
        check("確定 checkpoint がある run の新規開始を拒否 (exit 2)", c == 2)

        # 9. 最終 step の成長記録の途中で停止要求
        c, _ = run(O, "finstop", env={**ENV, "TINYJA_YIELD_IN_GROWTH": "stop:60"})
        st = os.path.join(O, "finstop")
        lt = json.load(open(os.path.join(st, "state", "LATEST")))
        g = [json.loads(l)["step"] for l in open(os.path.join(st, "growth.jsonl"))]
        check("最終記録中の停止 → exit 11・完了扱いにしない", c == 11 and lt["step"] == 60 and 60 not in g
              and not os.path.exists(os.path.join(st, "model.bin")), f"exit={c} LATEST={lt['step']} growth末尾={g[-1:]}")
        c, _ = run(O, "finstop", "--resume")
        check("最終記録中の停止 → 再開で完了", c == 0)
        compare(ref, O, "finstop", "最終記録中の停止→再開")

        # 10. 評価・生成の途中で区間期限（step 22 の成長記録中）
        c, _ = run(O, "evaldl", env={**ENV, "TINYJA_YIELD_IN_GROWTH": "deadline:22"})
        ex = [e for e in events(O, "evaldl") if e["type"] == "exit"][-1]
        g = [json.loads(l)["step"] for l in open(os.path.join(O, "evaldl", "growth.jsonl"))]
        check("記録中の区間期限 → 追加更新なしで exit 10", c == 10 and ex["next_step"] == 22 and 22 not in g,
              f"exit={c} next_step={ex['next_step']} reason={ex['reason']}")
        c, _ = run(O, "evaldl", "--resume")
        check("記録中の区間期限 → 再開で完了", c == 0)
        compare(ref, O, "evaldl", "記録中の区間期限→再開")

        # 11. 最終 step で区間期限
        c, _ = run(O, "findl", env={**ENV, "TINYJA_YIELD_IN_GROWTH": "deadline:60"})
        check("最終記録中の区間期限 → exit 10・完了扱いにしない",
              c == 10 and not os.path.exists(os.path.join(O, "findl", "model.bin")), f"exit={c}")
        c, _ = run(O, "findl", "--resume")
        check("最終記録中の区間期限 → 再開で完了", c == 0)
        compare(ref, O, "findl", "最終記録中の区間期限→再開")

        # 12. 最終処理の途中で落ちる
        c, _ = run(O, "finfault", env={**ENV, "TINYJA_FAULT": "before_finalize"})
        lt = json.load(open(os.path.join(O, "finfault", "state", "LATEST")))
        check("最終処理中のクラッシュ → 最終 step の checkpoint は確定済み", c == 99 and lt["step"] == 60
              and not os.path.exists(os.path.join(O, "finfault", "model.bin")), f"exit={c} LATEST={lt['step']}")
        c, _ = run(O, "finfault", "--resume")
        check("最終処理中のクラッシュ → 再開で完了", c == 0)
        compare(ref, O, "finfault", "最終処理中のクラッシュ→再開")

        # 実測: 保存時間・サイズ（tiny/small/base 相当の大きさで1回ずつ）
        sizes = {}
        for preset in ("tiny", "small", "base"):
            args = [a for a in COMMON]
            args[args.index("nano")] = preset
            cmd = [PY, "train.py", "--data", DATA, "--out", O, "--run", f"size-{preset}", "--prompts", PROMPTS,
                   "--events", os.path.join(O, f"size-{preset}.events.jsonl"), *args, "--max_iters", "3",
                   "--seq_len", "32", "--segment_steps", "1"]
            subprocess.run(cmd, cwd=HERE, env=ENV, capture_output=True)
            ck = [e for e in events(O, f"size-{preset}") if e["type"] == "checkpoint"]
            if ck:
                sizes[preset] = {"bytes": ck[-1]["bytes"], "save_s": ck[-1]["seconds"]}
        results["measurements"]["checkpoint_by_preset_vocab400"] = sizes
        steps = [e for e in events(O, "ref") if e["type"] == "checkpoint"]
        results["measurements"]["nano_checkpoint"] = steps[-1] if steps else None
    finally:
        n_ok = sum(v["ok"] for v in results["checks"].values())
        results["summary"] = f"{n_ok}/{len(results['checks'])} passed"
        results["env"] = {"torch": torch.__version__, "numpy": __import__("numpy").__version__,
                          "sentencepiece": __import__("sentencepiece").__version__,
                          "threads": torch.get_num_threads(), "python": sys.version.split()[0], "platform": sys.platform}
        results["code_sha256"] = {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
                                  for f in ("train.py", "ckpt.py", "llama2c_format.py", "prepare.py", "tests/test_resume.py")}
        results["command"] = "cd train && OMP_NUM_THREADS=2 .venv/bin/python tests/test_resume.py"
        json.dump(results, open(os.path.join(HERE, "tests", "last_result.json"), "w"), ensure_ascii=False, indent=1)
        print(results["summary"], json.dumps(results["measurements"], ensure_ascii=False))
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if all(v["ok"] for v in results["checks"].values()) else 1)


if __name__ == "__main__":
    main()
