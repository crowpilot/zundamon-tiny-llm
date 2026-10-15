"""学習の途中保存（再開用 checkpoint）とデータ・計画の識別。

形式: tiny-ja-llm-train-ckpt/1
  <run>/state/
    ckpt-<step7桁>-<rand>/state.pt       torch.save した辞書（下記）
    ckpt-<step7桁>-<rand>/manifest.json  形式・step・state.pt の sha256・計画/データ/レシピの hash
    LATEST                               {"name", "step", "state_sha256"}（最後に確定した checkpoint）

確定（commit）の順序:
  1. 一時ディレクトリ .tmp-* に state.pt を書いて fsync
  2. manifest.json を書いて fsync（state.pt の sha256 を含む）
  3. 一時ディレクトリを ckpt-* に rename（同一FS内で atomic）→ 親ディレクトリを fsync
  4. LATEST.tmp を書いて fsync → LATEST に os.replace → 親ディレクトリを fsync
  LATEST が指す checkpoint だけが「確定」。1〜4 のどこで落ちても、直前の確定 checkpoint は壊れない。
  古い確定 checkpoint は新しいものの確定後に keep 件だけ残して削除する。

state.pt の中身:
  format, step(次に実行する更新番号), model, optimizer, scaler, running_loss,
  rng: {numpy_train(Generator state), torch_cpu, torch_cuda(list or None)}, plan, plan_hash
"""
import hashlib
import json
import os
import secrets
import shutil

import torch

CKPT_FORMAT = "tiny-ja-llm-train-ckpt/1"
_FAULT = os.environ.get("TINYJA_FAULT", "")   # 検証用: 保存の途中でプロセスを落とす


class CheckpointError(Exception):
    pass


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_obj(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path, obj):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(os.path.abspath(path)))


def _fault(point):
    if _FAULT == point:
        os._exit(99)


def save_checkpoint(state_dir, payload, ids, keep=2):
    """payload を確定 checkpoint として保存し、(name, sha256, bytes) を返す。ids: 計画/データ/レシピの hash など"""
    os.makedirs(state_dir, exist_ok=True)
    step = int(payload["step"])
    tmp = os.path.join(state_dir, f".tmp-{step:07d}-{os.getpid()}-{secrets.token_hex(3)}")
    os.makedirs(tmp)
    sp = os.path.join(tmp, "state.pt")
    with open(sp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    _fault("after_state")
    digest = sha256_file(sp)
    size = os.path.getsize(sp)
    manifest = {"format": CKPT_FORMAT, "step": step, "state_sha256": digest, "state_bytes": size, **ids}
    with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(tmp)
    name = f"ckpt-{step:07d}-{secrets.token_hex(3)}"
    os.rename(tmp, os.path.join(state_dir, name))
    _fsync_dir(state_dir)
    _fault("before_latest")
    write_json_atomic(os.path.join(state_dir, "LATEST"), {"name": name, "step": step, "state_sha256": digest})
    _fault("after_latest")
    _prune(state_dir, name, keep)
    return name, digest, size


def _prune(state_dir, current, keep):
    """確定 checkpoint を keep 件残す。LATEST より先の step を持つ孤立ディレクトリ（LATEST 更新前に落ちた保存）と
    一時ディレクトリの残骸は消す。"""
    cur_step = int(current.split("-")[1])
    done = sorted(d for d in os.listdir(state_dir) if d.startswith("ckpt-"))
    for d in done:
        if int(d.split("-")[1]) > cur_step:
            shutil.rmtree(os.path.join(state_dir, d), ignore_errors=True)
    done = [d for d in done if int(d.split("-")[1]) <= cur_step]
    for d in (done[:-keep] if keep > 0 else []):
        if d != current:
            shutil.rmtree(os.path.join(state_dir, d), ignore_errors=True)
    for d in os.listdir(state_dir):
        if d.startswith(".tmp-"):
            shutil.rmtree(os.path.join(state_dir, d), ignore_errors=True)


def latest(state_dir):
    p = os.path.join(state_dir, "LATEST")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(state_dir, name=None, map_location="cpu"):
    """確定 checkpoint を検証して読む。形式・sha256・LATEST との整合が合わなければ CheckpointError。"""
    lt = latest(state_dir)
    if name is None:
        if lt is None:
            raise CheckpointError(f"確定した checkpoint がありません: {state_dir}/LATEST")
        name = lt["name"]
    d = os.path.join(state_dir, name)
    mp = os.path.join(d, "manifest.json")
    sp = os.path.join(d, "state.pt")
    if not (os.path.exists(mp) and os.path.exists(sp)):
        raise CheckpointError(f"checkpoint が不完全です: {d}")
    manifest = json.load(open(mp, encoding="utf-8"))
    if manifest.get("format") != CKPT_FORMAT:
        raise CheckpointError(f"形式が違います: {manifest.get('format')} != {CKPT_FORMAT}")
    digest = sha256_file(sp)
    if digest != manifest["state_sha256"]:
        raise CheckpointError(f"state.pt の sha256 が manifest と一致しません（破損の可能性）: {name}")
    if lt and lt["name"] == name and lt["state_sha256"] != digest:
        raise CheckpointError("LATEST と checkpoint の sha256 が一致しません")
    payload = torch.load(sp, map_location=map_location, weights_only=False)
    if payload.get("format") != CKPT_FORMAT or int(payload["step"]) != int(manifest["step"]):
        raise CheckpointError("state.pt の中身が manifest と一致しません")
    return payload, manifest, name


# ------------------------------------------------------------------ データの識別
DATA_FILES = ("train.bin", "val.bin", "meta.json", "tok.model", "tok.bin")
OPTIONAL_DATA_FILES = ("train_spk.bin", "val_spk.bin")   # 話者の印（あれば manifest に含める）


def data_manifest(data_dir):
    """データ一式の sha256・サイズ・語彙数（prepare.py が data_manifest.json に保存する）"""
    names = list(DATA_FILES) + [f for f in OPTIONAL_DATA_FILES if os.path.exists(os.path.join(data_dir, f))]
    files = {f: {"sha256": sha256_file(os.path.join(data_dir, f)), "bytes": os.path.getsize(os.path.join(data_dir, f))}
             for f in names}
    meta = json.load(open(os.path.join(data_dir, "meta.json"), encoding="utf-8"))
    return {"format": "tiny-ja-llm-data/1", "files": files, "vocab_size": meta["vocab_size"],
            "train_tokens": meta["train_tokens"], "val_tokens": meta["val_tokens"]}


def verify_data(data_dir, train_data, val_data):
    """data_manifest.json と実ファイルの一致、token ID が語彙内であることを確認。manifest の hash を返す。"""
    mp = os.path.join(data_dir, "data_manifest.json")
    if not os.path.exists(mp):
        raise CheckpointError(f"{mp} がありません（prepare.py で作り直すか、make_data_manifest で作成）")
    want = json.load(open(mp, encoding="utf-8"))
    got = data_manifest(data_dir)
    if want != got:
        bad = [f for f in set(want["files"]) | set(got["files"]) if want["files"].get(f) != got["files"].get(f)]
        raise CheckpointError(f"データが data_manifest.json と一致しません: {bad or '語彙数/トークン数'}")
    V = got["vocab_size"]
    for name, arr in (("train.bin", train_data), ("val.bin", val_data)):
        if len(arr) != (got["train_tokens"] if name == "train.bin" else got["val_tokens"]):
            raise CheckpointError(f"{name} のトークン数が meta.json と一致しません")
        if len(arr) and int(arr.max()) >= V:
            raise CheckpointError(f"{name} に語彙数 {V} 以上の token ID があります")
    return hash_obj(want)


if __name__ == "__main__":
    # 既存のデータディレクトリに data_manifest.json を作る / checkpoint を検証する
    #   python ckpt.py manifest data_mix
    #   python ckpt.py verify out/mix-tiny/state
    import sys
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "manifest":
        write_json_atomic(os.path.join(path, "data_manifest.json"), data_manifest(path))
        print(f"wrote {path}/data_manifest.json")
    elif cmd == "verify":
        _, m, name = load_checkpoint(path)
        print(json.dumps({"name": name, **m}, ensure_ascii=False, indent=1))
