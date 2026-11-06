#!/usr/bin/env bash
# Rename the workspace crates `hekit-*` to `<name>-*` (default: ckks-train).
#
#   scripts/rename.sh --dry-run            # print every file and line that would change
#   scripts/rename.sh                      # apply (needs a clean working tree)
#   scripts/rename.sh --to other-name      # another prefix (lower case, digits, '-')
#
# What changes:
#   1. `hekit-{core,ckks,nn,cuda,bench}` -> `<name>-{...}` in every tracked text file in scope:
#      package names, dependency keys, `-p` arguments, paths such as `crates/hekit-ckks/...`;
#   2. `hekit_{core,ckks,nn,cuda,bench}` -> `<name_>_{...}` (the crate identifiers in `use`
#      paths, doc links and qualified paths);
#   3. `git mv crates/hekit-<c> crates/<name>-<c>` for every crate directory.
# Other spellings of "hekit" (the `HekitError` type, `HEKIT_*` environment variables, prose)
# are left alone; the dry run counts the lines that still contain one.
#
# Out of scope (history and data, which must keep the names they were produced with):
# results/, spikes/, kat/, .superpowers/, docs/quadruped-log.md, pinned CUDA kernel sources
# (their comments retain historical crate paths), the historical ReBoot-layout assembler,
# and symbolic links.
#
# After applying: `cargo check --workspace --all-targets`, `cargo test --workspace`, and review
# `git diff` before committing. The script never commits, pushes or publishes.
set -euo pipefail

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }

dry_run=0
new="ckks-train"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run | -n) dry_run=1 ;;
        --to)
            [ $# -ge 2 ] || { echo "rename.sh: --to needs a name" >&2; exit 2; }
            new="$2"
            shift
            ;;
        -h | --help) usage; exit 0 ;;
        *) echo "rename.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ! printf '%s' "$new" | grep -Eq '^[a-z][a-z0-9-]*[a-z0-9]$'; then
    echo "rename.sh: invalid crate prefix: $new" >&2
    exit 2
fi
new_us=$(printf '%s' "$new" | tr '-' '_')
crates="core ckks nn cuda bench"

root=$(git rev-parse --show-toplevel)
cd "$root"

if [ "$dry_run" -eq 0 ] && ! git diff --quiet HEAD --; then
    echo "rename.sh: the working tree has uncommitted changes; commit or stash them first" >&2
    exit 1
fi

# Tracked files in scope: text only (grep -I), no symbolic links, no excluded paths.
files=()
while IFS= read -r f; do
    [ -f "$f" ] || continue
    [ -L "$f" ] && continue
    case "$f" in
        results/* | spikes/* | kat/* | .superpowers/* | docs/quadruped-log.md | tools/reboot_layout/assemble.py | crates/*-cuda/kernels/*.cu) continue ;;
    esac
    grep -Iq . "$f" 2>/dev/null || continue
    files+=("$f")
done < <(git ls-files)

# shellcheck disable=SC2016  # the $ signs belong to perl
perl_prog='
    BEGIN { ($new, $new_us, $dry) = splice(@ARGV, 0, 3); $lines = 0; $hits = 0; $files = 0; $rest = 0; }
    $orig = $_;
    $n = s/\bhekit-(core|ckks|nn|cuda|bench)\b/$new-$1/g;
    $n += s/\bhekit_(core|ckks|nn|cuda|bench)\b/${new_us}_$1/g;
    if ($n) {
        $lines++; $hits += $n;
        $files++ unless $seen{$ARGV}++;
        if ($dry) { print "$ARGV:$.\n  - $orig"; print "  + $_"; }
    }
    $rest++ if /hekit/i;
    print if !$dry;
    if (eof) { close ARGV; }
    END {
        if ($dry) {
            print "\nSUMMARY: $lines lines ($hits occurrences) in $files files would change; ",
                  "$rest lines would still contain \"hekit\" (other spellings, left unchanged)\n";
        }
    }
'

if [ "$dry_run" -eq 1 ]; then
    perl -ne "$perl_prog" "$new" "$new_us" 1 "${files[@]}"
    echo "Directory moves:"
    for c in $crates; do
        [ -d "crates/hekit-$c" ] && echo "  git mv crates/hekit-$c crates/$new-$c"
    done
    echo "(dry run: nothing was changed)"
else
    perl -i -ne "$perl_prog" "$new" "$new_us" 0 "${files[@]}"
    for c in $crates; do
        [ -d "crates/hekit-$c" ] && git mv "crates/hekit-$c" "crates/$new-$c"
    done
    echo "Renamed hekit-* -> $new-*. Next: cargo check --workspace --all-targets && cargo test --workspace; review git diff."
fi
