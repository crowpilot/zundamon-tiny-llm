"""コーパス → トークナイザ学習 → トークン列(.bin) 作成

入力（複数可）:
  *.jsonl : 1行1会話 {"turns": ["こんにちは", "やあ、げんき？", ...]}  （gen_corpus.py の出力）
  *.txt   : 1行1サンプルの生テキスト（会話なら「Ａ：…Ｂ：…」形式で書く）

出力 (--out, 既定 data/):
  tok.model / tok.vocab : SentencePiece
  tok.bin               : llama2.c / Core2 用トークナイザ
  train.bin / val.bin   : uint16 トークン列（各サンプル先頭に BOS=1）
  meta.json

例: python prepare.py corpus/*.jsonl --vocab 1024
"""
import argparse
import hashlib
import json
import os
import random

import numpy as np
import sentencepiece as spm

from ckpt import data_manifest, write_json_atomic
from llama2c_format import write_tokenizer

SPK = ["Ａ：", "Ｂ："]   # 話者マーカー（1トークンとして扱う）
# 気持ちタグ（--tags のとき、返事の先頭に 1 トークンとして入れる。zunda_emotion.py の8種類）
EMO = ["ふつう", "うれしい", "ドヤ", "びっくり", "かなしい", "ぷんぷん", "こまった", "ねむい"]
TAGS = ["〈" + e + "〉" for e in EMO]


def load_samples(paths, tags=False, tags_end=False):
    out, seen = [], set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if p.endswith(".jsonl"):
                    try:
                        rec = json.loads(line)
                        turns = rec["turns"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                    turns = [t.strip().replace("\n", "") for t in turns if isinstance(t, str) and t.strip()]
                    if len(turns) < 2:
                        continue
                    if tags:   # 1組（話しかけ＋返事）に気持ちタグ。タグの無い組は使わない
                        if rec.get("emo") not in EMO or len(turns) != 2:
                            continue
                        tg = "〈" + rec["emo"] + "〉"
                        turns = [turns[0], turns[1] + tg] if tags_end else [turns[0], tg + turns[1]]
                    text = "".join(SPK[i % 2] + t for i, t in enumerate(turns))
                else:
                    text = line
                h = hashlib.md5(text.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="data")
    ap.add_argument("--vocab", type=int, default=1024, help="語彙数。512はバイト用に259枠取られるので日本語はかなり厳しい")
    ap.add_argument("--char_coverage", type=float, default=0.9995,
                    help="必ず1文字として語彙に入れる文字の割合。漢字の多い青空文庫を混ぜるなら 0.995 前後に下げる")
    ap.add_argument("--dialogue_repeat", type=int, default=1,
                    help="会話サンプル（Ａ：で始まる行）を学習側で何倍に水増しするか。地の文が多いデータと混ぜるとき用")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tags", action="store_true", help="jsonl の emo を気持ちタグとして返事の先頭に入れる（タグも1トークン）")
    ap.add_argument("--tags_end", action="store_true", help="--tags のタグを返事の先頭ではなく最後に置く")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    samples = load_samples(args.inputs, tags=args.tags or args.tags_end, tags_end=args.tags_end)
    if len(samples) < 10:
        raise SystemExit(f"サンプルが少なすぎます: {len(samples)}")
    random.Random(args.seed).shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_frac))
    val, train = samples[:n_val], samples[n_val:]
    if args.dialogue_repeat > 1:
        dia = [t for t in train if t.startswith(SPK[0])]
        train = train + dia * (args.dialogue_repeat - 1)
        random.Random(args.seed + 1).shuffle(train)
        print(f"会話 {len(dia)} 件を {args.dialogue_repeat} 倍に水増し")
    chars = sum(len(s) for s in samples)
    print(f"samples: {len(samples)} (train {len(train)} / val {len(val)}), chars: {chars:,}")

    raw = os.path.join(args.out, "spm_input.txt")
    with open(raw, "w", encoding="utf-8") as f:
        f.write("\n".join(train))

    prefix = os.path.join(args.out, "tok")
    spm.SentencePieceTrainer.train(
        input=raw, model_prefix=prefix, model_type="bpe", vocab_size=args.vocab,
        character_coverage=args.char_coverage, byte_fallback=True, split_digits=True,
        add_dummy_prefix=False, normalization_rule_name="identity",
        remove_extra_whitespaces=False, user_defined_symbols=SPK + (TAGS if (args.tags or args.tags_end) else []),
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        max_sentence_length=16384, input_sentence_size=2_000_000, shuffle_input_sentence=True,
        hard_vocab_limit=False, num_threads=os.cpu_count() or 4,
    )
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    V = sp.get_piece_size()
    if V != args.vocab:
        print(f"注意: データが少ないため語彙数は {V} になりました")

    pieces, scores = [], []
    for i in range(V):
        p = sp.id_to_piece(i)
        if i == sp.bos_id():
            p = "\n<s>\n"
        elif i == sp.eos_id():
            p = "\n</s>\n"
        p = p.replace("▁", " ")
        pieces.append(p)
        scores.append(sp.get_score(i))
    write_tokenizer(os.path.join(args.out, "tok.bin"), pieces, scores)

    spk_ids = [sp.piece_to_id(m) for m in SPK]

    def encode(texts, path):
        """トークン列と、各トークンが誰の台詞に属するか（0=Ａ/その他, 1=Ｂ）の印を書く。
        話者マーカーは「直前の話者が台詞を終えた」という予測なので、直前の話者の側に数える。"""
        ids, spk = [], []
        for t in texts:
            ids.append(sp.bos_id())
            spk.append(0)
            cur = 0
            for tok in sp.encode(t):
                if tok == spk_ids[0]:
                    spk.append(cur)
                    cur = 0
                elif tok == spk_ids[1]:
                    spk.append(cur)
                    cur = 1
                else:
                    spk.append(cur)
                ids.append(tok)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(path)
        np.array(spk, dtype=np.uint8).tofile(path.replace(".bin", "_spk.bin"))
        return len(arr)

    nt = encode(train, os.path.join(args.out, "train.bin"))
    nv = encode(val, os.path.join(args.out, "val.bin"))
    meta = {"vocab_size": V, "train_tokens": nt, "val_tokens": nv, "samples": len(samples),
            "chars": chars, "speaker_markers": SPK}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    write_json_atomic(os.path.join(args.out, "data_manifest.json"), data_manifest(args.out))
    print(f"tokens: train {nt:,} / val {nv:,}  (平均 {chars/ max(1,nt+nv):.2f} 文字/トークン)")
    print("例:", sp.encode(train[0], out_type=str)[:30])


if __name__ == "__main__":
    main()
