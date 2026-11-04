#!/bin/sh
# Downloads MNIST (LeCun, Cortes, Burges) into data/mnist/ (gitignored; MNIST is not
# committed) and unpacks the four IDX files that crates/hekit-bench/src/data.rs reads.
#
# Mirror: the PyTorch/torchvision mirror on S3 (the original yann.lecun.com host is often
# unavailable). Each .gz is checked against torchvision's MD5, and each unpacked file against
# its SHA-256 (pinned from a verified download); a file already present is re-checked, so a
# truncated or altered file is replaced. Usage: scripts/fetch_mnist.sh [DIR]
set -eu

DIR="${1:-$(dirname "$0")/../data/mnist}"
MIRROR="https://ossci-datasets.s3.amazonaws.com/mnist"
mkdir -p "$DIR"
cd "$DIR"

md5_of() {
    if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi
}

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        sha256sum "$1" | cut -d' ' -f1
    fi
}

while read -r name md5 sha; do
    if [ -f "$name" ]; then
        if [ "$(sha256_of "$name")" = "$sha" ]; then
            echo "have $name"
            continue
        fi
        echo "replacing $name (SHA-256 mismatch)" >&2
        rm -f "$name"
    fi
    curl -fsSL --retry 3 -o "$name.gz" "$MIRROR/$name.gz"
    got=$(md5_of "$name.gz")
    if [ "$got" != "$md5" ]; then
        echo "MD5 mismatch for $name.gz: $got (want $md5)" >&2
        rm -f "$name.gz"
        exit 1
    fi
    gunzip -f "$name.gz"
    got=$(sha256_of "$name")
    if [ "$got" != "$sha" ]; then
        echo "SHA-256 mismatch for $name: $got (want $sha)" >&2
        rm -f "$name"
        exit 1
    fi
    echo "fetched $name"
done <<LIST
train-images-idx3-ubyte f68b3c2dcbeaaa9fbdd348bbdeb94873 ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db
train-labels-idx1-ubyte d53e105ee54ea40749a09fcbcd1e9432 65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5
t10k-images-idx3-ubyte 9fb629c4189551a2d022fa330f9573f3 0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7
t10k-labels-idx1-ubyte ec29112dd5afa0611ce80d1b7f02629c ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2
LIST
