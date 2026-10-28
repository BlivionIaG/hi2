# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""bpw of a GGUF, counted the way ``hi2 size`` counts an HI2 checkpoint: every byte the
loader reads over the quantized weights, divided by the number of weights those bytes hold.

Pure python (the GGUF container is read, not the tensors), so it runs anywhere and needs
no gguf-py. The count is reported for the quantized decoder linears, the way the AWQ and
HI2 numbers in [../docs/results.md](../docs/results.md) are counted, and for the whole file.

    python gguf_bpw.py model.gguf
"""

import argparse
import re
import struct
from collections import defaultdict
from pathlib import Path

# ggml type -> (block size, bytes per block)
TYPES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 31: (256, 54), 32: (256, 66), 34: (32, 17),
}
NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
         9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
         15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
         26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 31: "TQ1_0",
         32: "TQ2_0", 34: "MXFP4"}
# types that are not a quantized weight
UNQUANTIZED = {0, 1, 24, 25, 26, 27, 28, 30}


class R:
    """A reader over the GGUF header."""

    def __init__(self, fh):
        self.f = fh

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def s(self):
        n = self.u64()
        return self.f.read(n).decode("utf-8", "replace")

    def val(self, t):
        if t == 0:
            return struct.unpack("<B", self.f.read(1))[0]
        if t == 1:
            return struct.unpack("<b", self.f.read(1))[0]
        if t == 2:
            return struct.unpack("<H", self.f.read(2))[0]
        if t == 3:
            return struct.unpack("<h", self.f.read(2))[0]
        if t == 4:
            return self.u32()
        if t == 5:
            return struct.unpack("<i", self.f.read(4))[0]
        if t == 6:
            return struct.unpack("<f", self.f.read(4))[0]
        if t == 7:
            return bool(self.u32())
        if t == 8:
            return self.s()
        if t == 9:
            et = self.u32()
            n = self.u64()
            return [self.val(et) for _ in range(n)]
        if t == 10:
            return self.u64()
        if t == 11:
            return struct.unpack("<q", self.f.read(8))[0]
        if t == 12:
            return struct.unpack("<d", self.f.read(8))[0]
        raise ValueError("kv type {}".format(t))


def read_header(path: Path):
    """The metadata and the tensor table of a GGUF (the tensor data is not read)."""
    with open(path, "rb") as fh:
        r = R(fh)
        assert fh.read(4) == b"GGUF", "not a GGUF"
        ver = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        for _ in range(n_kv):
            k = r.s()
            t = r.u32()
            kv[k] = r.val(t)
        tensors = []
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ttype = r.u32()
            off = r.u64()
            tensors.append((name, dims, ttype, off))
    return ver, kv, tensors


def nbytes(dims, ttype):
    blk, sz = TYPES[ttype]
    n = 1
    for d in dims:
        n *= d
    assert n % blk == 0, (dims, ttype)
    return n // blk * sz


def nweights(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    a = ap.parse_args()

    ver, kv, tensors = read_header(a.gguf)
    size = a.gguf.stat().st_size

    by_type = defaultdict(lambda: [0, 0, 0])
    body_b = body_w = 0
    lin_b = lin_w = 0
    dense_b = 0
    for name, dims, ttype, _ in tensors:
        b = nbytes(dims, ttype)
        w = nweights(dims)
        t = by_type[NAMES.get(ttype, ttype)]
        t[0] += b
        t[1] += w
        t[2] += 1
        if name.startswith("blk."):
            lin_b += b
            lin_w += w
            if name.endswith(("_q.weight", "_gate.weight", "_up.weight", "_down.weight")):
                body_b += b
                body_w += w
        else:
            dense_b += b

    print("file: {} ({:.2f} GiB), gguf v{}, {} tensors".format(
        a.gguf.name, size / 2 ** 30, ver, len(tensors)))
    print("quant types:")
    for t in sorted(by_type, key=lambda k: -by_type[k][0]):
        b, w, c = by_type[t]
        print("   {:8s} {:5d} tensors {:10.2f} MiB {:7.3f} bpw".format(
            t, c, b / 2 ** 20, b * 8 / w if w else 0))
    print()
    print("decoder linears (blk.* projections): {:.1f} M weights, {:.1f} MiB, {:.3f} bpw".format(
        body_w / 1e6, body_b / 2 ** 20, body_b * 8 / body_w))
    print("all blk.* tensors:                  {:.1f} M weights, {:.1f} MiB, {:.3f} bpw".format(
        lin_w / 1e6, lin_b / 2 ** 20, lin_b * 8 / lin_w))
    print("everything else (embeddings, head, norms): {:.1f} MiB".format(dense_b / 2 ** 20))
    print("file bpw over the decoder linears: {:.3f}".format(size * 8 / body_w))

    # the same set HI2 and AWQ are counted on: every quantized decoder linear
    qb = qw = 0
    names = []
    for name, dims, ttype, _ in tensors:
        if ttype not in UNQUANTIZED:
            qb += nbytes(dims, ttype)
            qw += nweights(dims)
            names.append(name)
    print()
    print("quantized decoder linears: {} tensors, {:.1f} M weights, {:.1f} MiB, {:.3f} bpw".format(
        len(names), qw / 1e6, qb / 2 ** 20, qb * 8 / qw))
    print("file bpw over them: {:.3f}".format(size * 8 / qw))
    pats = sorted({re.sub(r"\.\d+\.", ".N.", n) for n in names})
    print("name patterns ({}):".format(len(pats)))
    for p in pats:
        print("   ", p)


if __name__ == "__main__":
    main()
