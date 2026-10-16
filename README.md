# 幼生ずんだもん

ゼロから学習させた、とても小さな「ずんだもん風」日本語言語モデルです。話しかけると、ずんだもんとして短く返事をします。パラメータは約161万個（int8 で 1.8MB）で、ブラウザの中だけで動きます。M5Stack Core2（ESP32）では、同じデータで学習した約71万パラメータの小さい版が 1 秒に 6 トークンほどで話します。

**デモ**: https://crowpilot.github.io/zundamon-tiny-llm/

非公式のファン作品です。ずんだもんは[東北ずん子・ずんだもんプロジェクト](https://zunko.jp/)のキャラクターで、[キャラクター利用の手引き](https://zunko.jp/guideline.html)に沿って非商用で公開しています。モデルがとても小さいので、ときどきおかしなことを言います。

## 育っていく様子

学習の途中で保存したモデルに「こんにちは。」と話しかけたときの返事です（デモでは段階を切り替えて話せます。「版」を切り替えると、最初に作った 71 万パラメータの版とも話せます）。

| 学習回数 | 返事 |
|---|---|
| 16 | 「Ｂ：Ｂ：Ｂ：、、、、、、」 |
| 63 | 「ボクもボクもボクもボクも……」 |
| 100 | 「おんがく、ボクも一緒に探すのだ！」 |
| 251 | 「おつきさまが、とってもかわいいのだ！」 |
| 398 | 「にがいのは、ボクは苦手なのだ！」 |
| 631 | 「こんにちはなのだ！ボクはだいじょうぶなのだ！」 |

最初は記号と読点だけ、次に一人称の「ボク」のくり返し、100 回目にはもう語尾「〜のだ！」の文になり、そのあと話しかけに合わせた返事を覚えていきます。

## どうやって作ったか

1. **話しかけを合成** — Gemma 4（Apache 2.0）に、ずんだもんへの話しかけを書かせます。あいさつ・質問・報告・誘い・一言・打ち間違いなど 13 種類と長さ・話題の種を毎回ランダムに指定して、1 回の推論で 30 個ずつ。36,012 個。
2. **返事を合成** — 話しかけ 20 個ずつに、ずんだもんとして返事を付けさせます。話しかけの言葉をそのまま返す「おうむ返し」ばかりにならないよう、言葉を入れてよいのは約 3 割の指示分だけにし、それ以外でくり返した返事は除きます（最終的に約 2 割）。
3. **ふるい分け** — 口調（語尾「のだ」、一人称「ボク」、です・ます禁止）と漢字の多さを機械的に検査し、さらに LLM に「話しかけに答えているか」「日本語として自然か」を採点させて低いものを除きます。同じ返事の使いすぎも制限します。26,547 組 → 24,190 組。
4. **トークナイザ** — SentencePiece BPE、語彙 1,024。
5. **学習** — Llama 型の Transformer（埋め込み 128 次元・8 層・注意ヘッド 8／KV 4・FFN 352）を 10 エポック。小型 PC の CPU で約 10 分です。
6. **int8 化** — 32 個ずつのグループで量子化。精度はほぼ落ちません。
7. **推論** — llama2.c 互換の C 実装（Core2）と、同じ計算をする JavaScript 実装（ブラウザ）。

最初の版は「ずんだもんと相手の短い会話」6,004 本から学習した 71 万パラメータのモデルでした。話しかけと返事の組に作り直してモデルを大きくしたことで、話しかけ 40 個への応答を LLM に 5 段階で採点させた平均が 1.4 から 2.6 に上がりました。

| 項目 | ブラウザ版 | Core2 版 |
|---|---|---|
| ファイル | `zunda-base-631` | `zunda-small-631` |
| パラメータ | 1,607,808 | 707,808 |
| 構成 | 128 次元・8 層 | 96 次元・6 層 |
| 学習データ | 約 53 万トークン（84 万文字） | 同じ |
| 評価 loss（ずんだもんの返事） | 2.35 | 2.59（話しかけ側の重み 0.3 で学習） |
| モデルサイズ | fp32 6.4MB / int8 1.8MB | fp32 2.8MB / int8 0.80MB |
| 速度 | PC のブラウザで 100 トークン/秒以上 | 約 6 トークン/秒 |

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
API="--base_url http://localhost:8000/v1 --model gemma-4"
python3 gen_pairs.py inputs --n 36000 --out corpus/pair_inputs.jsonl $API
python3 gen_pairs.py replies --inputs corpus/pair_inputs.jsonl --n 30000 --out corpus/pairs.jsonl $API
python3 zunda_judge.py corpus/pairs.jsonl --out corpus/pairs_judge.jsonl $API
python3 zunda_filter.py corpus/pairs.jsonl corpus/pairs_judge.jsonl --out corpus/pairs_filtered.jsonl
python3 prepare.py corpus/pairs_filtered.jsonl --out data --vocab 1024 --char_coverage 0.995 --val_frac 0.02
python3 train.py --data data --run zunda-base --preset base --epochs 10 --batch_size 32 --prompts prompts_zunda.txt
python3 quant_eval.py out/zunda-base --prompts prompts_zunda.txt
```

最初の版（会話形式のデータ）は `gen_zunda.py` で作れます。

## ライセンス

- コード: MIT（`LICENSE`）。`core2/src/llm_core.c` は [llama2.c](https://github.com/karpathy/llama2.c)（MIT, Andrej Karpathy）を改変したものです（`THIRD_PARTY_NOTICES.md`）。
- 学習済みモデル: MIT。学習データは Gemma 4（Apache 2.0）で生成しました。
- ずんだもんのキャラクターに関する権利は東北ずん子・ずんだもんプロジェクトにあり、このリポジトリのライセンスの対象外です。利用する場合はキャラクター利用の手引きに従ってください。
