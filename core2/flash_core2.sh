#!/bin/bash
# Core2 にモデルを焼いて動かす（Mac で実行）
#   ./flash_core2.sh                          → stories260K（自動ダウンロード）
#   ./flash_core2.sh model.bin tok.bin 名前    → 自作モデル（train/out/ の ckpt_*.bin と tok.bin）
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data

if [ $# -ge 2 ]; then
  MODEL="$1"; TOK="$2"; NAME="${3:-$(basename "$1" .bin)}"
else
  MODEL=data/stories260K.bin; TOK=data/tok512.bin; NAME=stories260K
  HF=https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K
  [ -f "$MODEL" ] || curl -fL -o "$MODEL" "$HF/stories260K.bin"
  [ -f "$TOK" ]   || curl -fL -o "$TOK"   "$HF/tok512.bin"
fi

python3 tools/bin2c.py "$MODEL" "$TOK" src/model_data.h "$NAME"

# PlatformIO（無ければこのフォルダ内の venv に入れる）
if command -v pio >/dev/null 2>&1; then
  PIO=pio
else
  if [ ! -x .venv/bin/pio ]; then
    echo "PlatformIO を .venv に導入します（初回のみ・数分）"
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip platformio
  fi
  PIO=.venv/bin/pio
fi

echo "== USBシリアル候補 =="
ls /dev/cu.usbserial* /dev/cu.wchusbserial* /dev/cu.SLAB* 2>/dev/null || echo "（見つかりません: ケーブル/ドライバを確認）"

$PIO run -t upload
echo "== 書き込み完了。シリアル出力を表示します（Ctrl+C で終了） =="
$PIO device monitor
