#!/bin/sh
# Downloads MNIST (LeCun, Cortes, Burges) into data/mnist/ (gitignored; MNIST is not
# committed) and unpacks the four IDX files that crates/hekit-bench/src/data.rs reads.
#
# Mirror: the PyTorch/torchvision mirror on S3 (the original yann.lecun.com host is often
# unavailable). The MD5 sums are torchvision's. Usage: scripts/fetch_mnist.sh [DIR]
set -eu

DIR="${1:-$(dirname "$0")/../data/mnist}"
MIRROR="https://ossci-datasets.s3.amazonaws.com/mnist"
mkdir -p "$DIR"
cd "$DIR"

md5_of() {
    if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi
}

while read -r name sum; do
    if [ -f "$name" ]; then
        echo "have $name"
        continue
    fi
    curl -fsSL --retry 3 -o "$name.gz" "$MIRROR/$name.gz"
    got=$(md5_of "$name.gz")
    if [ "$got" != "$sum" ]; then
        echo "MD5 mismatch for $name.gz: $got (want $sum)" >&2
        rm -f "$name.gz"
        exit 1
    fi
    gunzip -f "$name.gz"
    echo "fetched $name"
done <<LIST
train-images-idx3-ubyte f68b3c2dcbeaaa9fbdd348bbdeb94873
train-labels-idx1-ubyte d53e105ee54ea40749a09fcbcd1e9432
t10k-images-idx3-ubyte 9fb629c4189551a2d022fa330f9573f3
t10k-labels-idx1-ubyte ec29112dd5afa0611ce80d1b7f02629c
LIST
