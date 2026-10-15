# 幼生ずんだもん

ゼロから学習させた、とても小さな「ずんだもん風」日本語言語モデルです。パラメータは約71万個（int8 で 0.8MB）。ブラウザの中だけで動き、M5Stack Core2（ESP32）でも 1 秒に 6 トークンほどで話します。

**デモ**: https://crowpilot.github.io/zundamon-tiny-llm/

非公式のファン作品です。ずんだもんは[東北ずん子・ずんだもんプロジェクト](https://zunko.jp/)のキャラクターで、[キャラクター利用の手引き](https://zunko.jp/guideline.html)に沿って非商用で公開しています。モデルがとても小さいので、ときどきおかしなことを言います。

## 育っていく様子

学習の途中で保存したモデルに「こんにちは。」と話しかけたときの返事です（デモでは段階を切り替えて話せます）。

| 学習回数 | 返事 |
|---|---|
| 40 | 「、、、、、、」 |
| 63 | 「あ、あ、あ、あ、あ、」 |
| 100 | 「あ、おち、おちい！」 |
| 251 | 「ボクが、おとてつだうのだ！」 |
| 398 | 「おばあちゃん、おなかがすいてきたのだ！」 |
| 631 | 「ボクは、とってもいそがしいのだ！」 |

最初は読点だけ、次に「あ」、語尾の「のだ」と一人称の「ボク」、それから短い受け答え、という順番で身につけていきます。

## どうやって作ったか

1. **会話データを合成** — Gemma 4（Apache 2.0）に、ずんだもんと色々な相手との短い会話を書かせます。1 回の推論で「種の言葉 3 つ → 短い構成 → 台詞」を 8 本ずつ。相手・ずんだもんの様子（いばる、ツイてない、おさんぽ…）・話題・気分を毎回ランダムに組み合わせて多様にします。
2. **ふるい分け** — 口調（語尾「のだ」、一人称「ボク」、です・ます禁止）、役の取り違え、漢字の多さを機械的に検査し、さらに LLM に「前の発言に答えているか」を採点させて低いものを除きます。同じ台詞の使いすぎも制限します。6,004 本 → 3,819 本。
3. **トークナイザ** — SentencePiece BPE、語彙 1,024。
4. **学習** — Llama 型の Transformer（埋め込み 96 次元・6 層・注意ヘッド 6／KV 3・FFN 256）を CPU で約 8 分。
5. **int8 化** — 32 個ずつのグループで量子化。精度はほぼ落ちません。
6. **推論** — llama2.c 互換の C 実装（Core2）と、同じ計算をする JavaScript 実装（ブラウザ）。

| 項目 | 値 |
|---|---|
| パラメータ | 707,808 |
| 学習データ | 約 21 万トークン（34 万文字） |
| 評価 loss（ずんだもんの台詞） | 2.64 |
| モデルサイズ | fp32 2.8MB / int8 0.80MB |
| 速度 | ブラウザ（PC）約 300 トークン/秒、Core2 約 6 トークン/秒 |

## フォルダ

| 場所 | 中身 |
|---|---|
| `docs/` | ブラウザ版（GitHub Pages で公開するページ） |
| `train/` | コーパス生成・ふるい分け・学習・評価のスクリプト（Python） |
| `core2/` | M5Stack Core2 用のプログラム（PlatformIO） |
| `host_test/` | PC で C 実装を試すツール |
| `models/` | 学習済みモデル（int8 と fp32、トークナイザ） |

## 自分で動かす

ブラウザ版は `docs/index.html` を開くだけで動きます（ローカルでもそのまま動きます）。

Core2 に入れる:

```sh
cd core2
./flash_core2.sh ../models/zunda-small-631_q8.bin ../models/tok.bin zunda
```

学習し直す（OpenAI 互換の API で Gemma 4 などを動かしている場合）:

```sh
cd train
pip install -r requirements.txt
python3 gen_zunda.py --mode inline_plan --n 6000 --out corpus/zunda.jsonl --base_url http://localhost:8000/v1 --model gemma-4
python3 zunda_judge.py corpus/zunda.jsonl --out corpus/judge.jsonl --base_url http://localhost:8000/v1 --model gemma-4
python3 zunda_filter.py corpus/zunda.jsonl corpus/judge.jsonl --out corpus/zunda_filtered.jsonl
python3 prepare.py corpus/zunda_filtered.jsonl --out data --vocab 1024 --char_coverage 0.995
python3 train.py --data data --run zunda --preset small --epochs 25 --batch_size 32 --prompts prompts_zunda.txt
python3 quant_eval.py out/zunda --prompts prompts_zunda.txt
```

## ライセンス

- コード: MIT（`LICENSE`）。`core2/src/llm_core.c` は [llama2.c](https://github.com/karpathy/llama2.c)（MIT, Andrej Karpathy）を改変したものです（`THIRD_PARTY_NOTICES.md`）。
- 学習済みモデル: MIT。学習データは Gemma 4（Apache 2.0）で生成しました。
- ずんだもんのキャラクターに関する権利は東北ずん子・ずんだもんプロジェクトにあり、このリポジトリのライセンスの対象外です。利用する場合はキャラクター利用の手引きに従ってください。
