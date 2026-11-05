#!/bin/sh
# Downloads Fashion-MNIST (Xiao, Rasul, Vollgraf 2017; Zalando SE, MIT License) into
# data/fashion_mnist/ (gitignored; the data is not committed) and unpacks the four IDX files,
# which crates/hekit-bench/src/data.rs reads with the MNIST loader (same format).
#
# Source: the official repository, https://github.com/zalandoresearch/fashion-mnist
# (data/fashion/*.gz). Each .gz is checked against the MD5 published in that repository's
# README, and each unpacked file against its SHA-256 (pinned from a verified download on
# 2026-11-06); a file already present is re-checked, so a truncated or altered file is
# replaced. Usage: scripts/fetch_fashion_mnist.sh [DIR]
set -eu

DIR="${1:-$(dirname "$0")/../data/fashion_mnist}"
MIRROR="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
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
train-images-idx3-ubyte 8d4fb7e6c68d591d4c3dfef9ec88bf0d c59f468a2f672dc815687fe0f83887768d799fd8a3f3276145d20f83aa44d888
train-labels-idx1-ubyte 25c81989df183df01b3e8a0aad5dffbe bad3541b69d912435c50bb6ba87bec294ff4f6a2e1246121d8633921760443d9
t10k-images-idx3-ubyte bef4ecab320f06d8554ea6380940ec79 5b4141f0afbad91edebe8549f8fcffe087ea10ca49f1dbef5c9a5cd8815ce37b
t10k-labels-idx1-ubyte bb300cfdad3c16e7a12a480ee83cd310 0402a96d92fd2663957122ceb108a494c5af83dab82d92729df917d7dec38c34
LIST
