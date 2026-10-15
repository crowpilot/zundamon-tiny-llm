#!/usr/bin/env python3
"""モデル(.bin)とトークナイザ(.bin)を Core2 に埋め込むための C ヘッダに変換する。
usage: python3 bin2c.py model.bin tok.bin out.h [表示名]
"""
import os
import sys


def emit(f, name, data):
    f.write(f"static const uint8_t {name}[] __attribute__((aligned(4))) = {{\n")
    for i in range(0, len(data), 24):
        f.write(",".join(str(b) for b in data[i:i + 24]))
        f.write(",\n")
    f.write("};\n")
    f.write(f"static const size_t {name}_len = {len(data)};\n\n")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    model_path, tok_path, out_path = sys.argv[1:4]
    name = sys.argv[4] if len(sys.argv) > 4 else os.path.splitext(os.path.basename(model_path))[0]
    model = open(model_path, "rb").read()
    tok = open(tok_path, "rb").read()
    import struct
    if struct.unpack("i", model[:4])[0] == 0x38515454:
        cfg = struct.unpack("7i", model[8:36])
        print(f"model {model_path}: {len(model)/1e6:.2f} MB  int8(Q8, gs={struct.unpack('i', model[36:40])[0]})  config={cfg}")
    else:
        cfg = struct.unpack("7i", model[:28])
        print(f"model {model_path}: {len(model)/1e6:.2f} MB  fp32  config={cfg}")
    if len(model) > 5_000_000:
        print("警告: モデルが大きすぎます（アプリ領域 6MB 程度が上限）")
    with open(out_path, "w") as f:
        f.write("// 自動生成ファイル（tools/bin2c.py）。手で編集しないこと\n#pragma once\n#include <stdint.h>\n#include <stddef.h>\n\n")
        f.write(f'#define MODEL_NAME "{name}"\n\n')
        emit(f, "model_bin", model)
        emit(f, "tok_bin", tok)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
