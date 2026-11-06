#!/usr/bin/env bash
# One-shot experiment runner (Task 54). Implements the documented partial §4.4 matrix
# and writes every result to results/run_all/<mode>/ with the binaries' own provenance.
#
#   scripts/run_all.sh [--quick] [--dry-run] [--only REGEX] [--threads N] [--allow-dirty] [--no-build]
#
#   (default)    full mode: the paper's runs. Encrypted rows use 1 seed (ReBoot's seed-0 initial
#                weights; each costs hours), plaintext twins use 10 seeds (eMLP-2/3: the 3 seeds
#                ReBoot's code releases). Estimated ~83 h on quadruped-gpu, dominated by the two
#                196-input full-length runs (R-41-FULL-1/2, ~58 h).
#   --quick      the same matrix on subsets (1-4 updates), 1 seed, small presets only where the
#                row is a count/check smoke. Target: <= 2 h on quadruped-gpu. Quick numbers are
#                pipeline checks, never paper numbers.
#   --dry-run    print every command, its status (todo / done / missing input) and the
#                estimated hours, then exit. Runs nothing, builds nothing.
#   --only RE    run only the ids matching the extended regex RE (e.g. '^plain', 'cuda').
#   --threads N  RAYON_NUM_THREADS for every run (default 12, the server cap next to SLAM).
#   --allow-dirty  run with tracked/untracked changes (default: refuse).
#   --no-build   require an existing matching build receipt and binary hashes.
#   --tables-only  regenerate canonical archived tables without builds or experiments.
#   --from-run-all full  with --tables-only, use new full outputs instead of archives.
#
# Datasets: data/mnist (scripts/fetch_mnist.sh), data/fashion_mnist
# (scripts/fetch_fashion_mnist.sh), and the gitignored ReBoot 14x14 arrays in
# results/quadruped/data/reboot_emlp1_full/ (R-REB-2; copy them in, their sha256 is checked
# against data_manifest.json by the binary). A run whose inputs are missing is reported as
# `missing-input` and not run.
#
# Idempotent: a run is skipped when results/run_all/<mode>/meta/<id>.run.json says `done` at
# the current HEAD/source identity/binary hash, and binary-written provenance agrees.
# Delete the meta file to force a rerun. Source changes invalidate build receipts too.
#
# Output per run id: <id>.json (the binary's own output, unmodified), logs/<id>.log
# (stdout+stderr), meta/<id>.run.json (this runner's record: command, commit, dirty files,
# host, threads, start/end, exit code, status). The meta file is runner metadata, not a
# result. Every NEW accepted output must carry the binary's own provenance; missing writer
# support is a failed/unmeasured row, never replaced by this runner's metadata.
#
# Baseline (b), official ReBoot (OpenFHE, GPL-3.0), is never invoked here: it runs only on the
# server through gpu-runner (results/quadruped/scripts/reboot_launch.sh, requests.md R-REB-*),
# outside this repository's tree.
#
# Exit status: 0 if every selected run is done or skipped for GPU memory (exit 3 + SKIP line),
# 1 if any run failed or had missing inputs, 2 on a usage error.

set -uo pipefail

MODE=full
DRY=0
ONLY=''
THREADS=12
ALLOW_DIRTY=0
BUILD=1
TABLES_ONLY=0
TABLE_INPUT=''

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --quick) MODE=quick ;;
    --full) MODE=full ;;
    --dry-run) DRY=1 ;;
    --only) shift; ONLY="${1:?--only needs a regex}" ;;
    --threads) shift; THREADS="${1:?--threads needs a value}" ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --no-build) BUILD=0 ;;
    --tables-only) TABLES_ONLY=1 ;;
    --from-run-all) shift; TABLE_INPUT="${1:-}" ;;
    -h|--help) usage 0 ;;
    *) echo "run_all.sh: unknown argument $1" >&2; usage 2 ;;
  esac
  shift
done
case "$THREADS" in ''|*[!0-9]*) echo 'run_all.sh: --threads must be 1..24' >&2; exit 2;; esac
if [ ${#THREADS} -gt 2 ] || [ "$THREADS" -lt 1 ] || [ "$THREADS" -gt 24 ]; then
  echo 'run_all.sh: --threads must be 1..24' >&2; exit 2
fi
if [ -n "$TABLE_INPUT" ] && { [ "$TABLES_ONLY" = 0 ] || [ "$TABLE_INPUT" != full ]; }; then
  echo 'run_all.sh: --from-run-all full requires --tables-only' >&2; exit 2
fi

REPO=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "run_all.sh: not in a git checkout" >&2; exit 2; }
cd "$REPO" || exit 2
if [ "$TABLES_ONLY" = 1 ]; then
  if [ -n "$TABLE_INPUT" ]; then
    exec python3 tools/paper_tables.py --from-run-all "$TABLE_INPUT"
  fi
  exec python3 tools/paper_tables.py
fi
HEAD=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain --untracked-files=all)
TRACKED_DIRTY=$(git status --porcelain --untracked-files=no)
# Custom compiler executables/wrappers require a separate audited recipe. Keep this
# native runner explicit; relevant supported Cargo/Rust flag values enter BUILD_ID.
for runner_override in RUSTC RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER CARGO_BUILD_RUSTC CARGO_BUILD_RUSTC_WRAPPER CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER; do
  if [ -n "${!runner_override:-}" ]; then
    echo "run_all.sh: unsupported compiler override $runner_override; unset it for this runner" >&2
    exit 2
  fi
done
RUSTC_BIN=$(command -v rustc) || exit 1
RUSTC_INFO=$(rustc -vV) || exit 1
CARGO_INFO=$(cargo --version) || exit 1
NATIVE_TARGET=$(printf '%s\n' "$RUSTC_INFO" | awk '$1 == "host:" { print $2 }')
case "$NATIVE_TARGET" in ''|*[!a-zA-Z0-9_.-]*) echo 'run_all.sh: cannot identify native Rust target' >&2; exit 1;; esac
# Bind source bytes, including new untracked source files, not merely runtime HEAD.
source_identity() {
  python3 - <<'PY'
import hashlib, pathlib, subprocess
h = hashlib.sha256()
paths = set(subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard']).split(b'\0'))
for raw in sorted(paths):
    if not raw:
        continue
    name = raw.decode()
    if not (name.startswith(('crates/','scripts/','tools/','.cargo/')) or name in
            ('Cargo.toml','Cargo.lock','rust-toolchain','rust-toolchain.toml','deny.toml','build.rs')):
        continue
    p = pathlib.Path(name)
    h.update(raw + b'\0')
    h.update(p.read_bytes() if p.is_file() else b'<deleted>')
    h.update(b'\0')
# Cargo also reads ignored/untracked workspace/ancestor and Cargo-home configuration.
# Hash both spellings conservatively; contents (which may be private) are never printed.
import os
root = pathlib.Path.cwd()
cargo_home = pathlib.Path(os.environ.get('CARGO_HOME', str(pathlib.Path.home()/'.cargo')))
configs = {d/'.cargo'/name for d in (root, *root.parents) for name in ('config','config.toml')}
configs.update(cargo_home/name for name in ('config','config.toml'))
for p in sorted(configs):
    h.update(str(p).encode()+b'\0')
    h.update(p.read_bytes() if p.is_file() else b'<absent>')
    h.update(b'\0')
print(h.hexdigest())
PY
}
SOURCE_ID=$(source_identity) || exit 1
BUILD_ID=$(python3 - "$SOURCE_ID" "$CARGO_INFO" "$RUSTC_INFO" <<'PY'
import hashlib, json, os, sys
env = {k:v for k,v in os.environ.items() if k.startswith(('CARGO_', 'RUST', 'CC_', 'CXX_'))
       or k in ('PATH','CC','CXX','AR','CFLAGS','CXXFLAGS','LDFLAGS')}
print(hashlib.sha256(json.dumps([sys.argv[1:], env], sort_keys=True).encode()).hexdigest())
PY
) || exit 1
OUT="results/run_all/$MODE"
CPU_TARGET=target
BIN=$CPU_TARGET/$NATIVE_TARGET/release
GPU_TARGET=target/run_all-cuda
GBIN=$GPU_TARGET/$NATIVE_TARGET/release
PLAIN_JOBS=$(( THREADS < 4 ? THREADS : 4 ))

# Shared inputs.
D196=results/quadruped/data/reboot_emlp1_full
RF=results/quadruped/raw/reboot_emlp1_full
RS=results/quadruped/raw/reboot_emlp1_official_subset
R23=results/quadruped/raw/reboot_emlp23_full
MNIST=data/mnist
FMNIST=data/fashion_mnist
NPY196="$D196/train_images_14x14.npy $D196/train_labels.npy $D196/test_images_14x14.npy $D196/test_labels.npy"
IDX_MNIST="$MNIST/train-images-idx3-ubyte $MNIST/train-labels-idx1-ubyte $MNIST/t10k-images-idx3-ubyte $MNIST/t10k-labels-idx1-ubyte"
IDX_FMNIST="$FMNIST/train-images-idx3-ubyte $FMNIST/train-labels-idx1-ubyte $FMNIST/t10k-images-idx3-ubyte $FMNIST/t10k-labels-idx1-ubyte"

# ------------------------------------------------------------------------------------------
# The run matrix. add ID GROUP BACKEND EST_HOURS "INPUTS" CMD...
#   BACKEND: cpu (built with --features parallel) or cuda (built with --features cuda into
#   $GPU_TARGET). EST_HOURS: the estimate printed by --dry-run; its basis is in the comment
#   above each group (server = quadruped-gpu, i9-13900KF, 12 threads unless stated).
#   {OUT} in CMD is replaced by the run's output path.
# ------------------------------------------------------------------------------------------
IDS=(); GROUPS_=(); BACKENDS=(); ESTS=(); NEEDS=(); CMDS=()
add() {
  local id=$1 group=$2 backend=$3 est=$4 needs=$5; shift 5
  IDS+=("$id"); GROUPS_+=("$group"); BACKENDS+=("$backend"); ESTS+=("$est"); NEEDS+=("$needs")
  CMDS+=("$*")
}

# Faithful NI eMLP-1 at 196 inputs, ReBoot's full setting (R-M3-13/14), seed-0 weights.
FULL196="--mode noninteractive --placement when_needed --faithful --dataset-npy $D196 --indices $RF/train_indices.txt --init-weights $RF/init_weights_seed0 --init-weights-manifest $RF/config_values.json --lr-schedule $RF/config_values.json --lr-key lr_per_step --epochs 1 --transpose reuse"
# Faithful NI eMLP-1 at 784 inputs on ReBoot's encrypted-script subset (1008 samples, R-M3-12).
SUB784="--mode noninteractive --faithful --indices $RS/reboot_subsample_indices.txt --init-weights $RS/reboot_init_weights --epochs 1 --transpose reuse --data $MNIST"
# The same, first --subset samples in file order instead of the 1008 indices (quick).
Q784="--mode noninteractive --faithful --init-weights $RS/reboot_init_weights --epochs 1 --transpose reuse --data $MNIST"
# Interactive eMLP-1 (M2 Task 32 setting; interactive mode has no faithful/no-bias path).
INT784="--mode interactive --epochs 1 --batch 32 --hidden 32 --seed 32 --lr 0.001 --momentum 0.9 --params n16_train --transpose reuse --data $MNIST"
# Faithful plaintext twin at 196 inputs, ReBoot's full setting, seed k.
plain196() { # k indices_file
  echo "--mode plain --faithful --batch 48 --dataset-npy $D196 --indices $RF/$2 --init-weights $RF/init_weights_seed$1 --init-weights-manifest $RF/config_values.json --lr-schedule $RF/config_values.json --lr-key lr_per_step --epochs 1"
}
# Block-MLP plaintext twin (Task 52), model m (emlp2|emlp3) with --blocks w, seed k.
plain23() { # m blocks k
  local cmp=''
  [ "$3" = 0 ] && cmp="--compare-steps $R23/twin_steps_$1_seed0.tsv"
  echo "--mode plain --faithful --blocks $2 --hidden 32 --batch 48 --epochs 1 --dataset-npy $D196 --indices $R23/train_indices_$1_seed$3.txt --init-weights $R23/init_weights_$1_seed$3 --init-weights-manifest $R23/config_values.json --lr-schedule $R23/config_values.json --lr-key lr_per_step $cmp"
}

if [ "$MODE" = full ]; then
  # RQ1 headline, 196 inputs, full length (4800 x 10), seed 0. Basis: R-41-FULL-1 projection
  # after 10 updates (70596 s); R-41-FULL-2 block estimate (38.6 h).
  add ni196_wn_b64_full_s0 rq1-196 cpu 19.6 "$NPY196" $BIN/mnist $FULL196 --batch 64 --lr-schedule-batch 48 --out {OUT}
  add ni196_wn_b48_full_s0 rq1-196 cpu 38.6 "$NPY196" $BIN/mnist $FULL196 --batch 48 --out {OUT}
  # RQ2 placements and both B on the 784 subset (supplementary rows), seed 0. Basis: server
  # train_total of R-41-F1/F2 (WN), R-41-1/2 (AU), R-41-5 (AA) + ~0.03 h setup.
  add ni784_wn_b32_s0 rq2-784 cpu 0.87 "$IDX_MNIST" $BIN/mnist $SUB784 --placement when_needed --batch 32 --out {OUT}
  add ni784_wn_b64_s0 rq2-784 cpu 0.48 "$IDX_MNIST" $BIN/mnist $SUB784 --placement when_needed --batch 64 --out {OUT}
  add ni784_au_b32_s0 rq2-784 cpu 1.24 "$IDX_MNIST" $BIN/mnist $SUB784 --placement after_update --batch 32 --out {OUT}
  add ni784_au_b64_s0 rq2-784 cpu 0.65 "$IDX_MNIST" $BIN/mnist $SUB784 --placement after_update --batch 64 --out {OUT}
  add ni784_aa_b32_s0 rq2-784 cpu 1.03 "$IDX_MNIST" $BIN/mnist $SUB784 --placement after_activation --batch 32 --out {OUT}
  # Interactive (client refresh) eMLP-1, 1024 samples. Basis: server, 1 thread, 10934 s
  # (mnist_interactive_b32); 12 threads should be faster, so this is an upper bound.
  add int784_b32_cpu interactive cpu 3.05 "$IDX_MNIST" $BIN/mnist $INT784 --subset 1024 --out {OUT}
  # GPU at level 7 only (keys capped at level 7 fit the shared card) and its same-level CPU row,
  # plus the RQ3 ablation. Basis: R-48-2/3/4 block estimates; the level-7 training rows have no
  # measurement yet (rough: GPU ops 33-200x the CPU ones, M4 §1).
  add int784_b32_l7_cuda gpu-l7 cuda 0.30 "$IDX_MNIST" $GBIN/mnist $INT784 --subset 1024 --backend cuda --top-level 7 --out {OUT}
  add int784_b32_l7_cpu gpu-l7 cpu 1.00 "$IDX_MNIST" $BIN/mnist $INT784 --subset 1024 --backend cpu --top-level 7 --out {OUT}
  add ablation_cuda_l7 rq3 cuda 0.17 "$IDX_MNIST" $GBIN/ablation --backend cuda --params n16_train --top-level 7 --optimizer nag --iters 5 --warmup 1 --data $MNIST --out {OUT}
  add ablation_cpu_l7 rq3 cpu 0.42 "$IDX_MNIST" $BIN/ablation --backend cpu --params n16_train --top-level 7 --optimizer nag --iters 5 --warmup 1 --data $MNIST --out {OUT}
  add ablation_cpu_l31 rq3 cpu 1.50 "$IDX_MNIST" $BIN/ablation --backend cpu --params n16_train --optimizer nag --iters 3 --warmup 1 --data $MNIST --out {OUT}
  # eMLP-2/3 block MLPs, encrypted, ReBoot's first 21 steps (1008 samples), seed 0. Basis:
  # R-52-1/2 estimates. Full-length encrypted eMLP-2/3 are on the user's decision list, not here.
  add ni196_emlp2_b48_sub1008_s0 block-mlp cpu 1.70 "$NPY196" $BIN/mnist --mode noninteractive --placement when_needed --batch 48 --faithful --blocks 64 --hidden 32 --dataset-npy $D196 --indices results/reboot_emlp2_seed0_steps1-21_indices.txt --init-weights $R23/init_weights_emlp2_seed0 --init-weights-manifest $R23/config_values.json --epochs 1 --transpose reuse --out {OUT}
  add ni196_emlp3_b48_sub1008_s0 block-mlp cpu 3.20 "$NPY196" $BIN/mnist --mode noninteractive --placement when_needed --batch 48 --faithful --blocks 128,64 --hidden 32 --dataset-npy $D196 --indices results/reboot_emlp3_seed0_steps1-21_indices.txt --init-weights $R23/init_weights_emlp3_seed0 --init-weights-manifest $R23/config_values.json --epochs 1 --transpose reuse --out {OUT}
  # Plaintext twins, 10 seeds (seconds each on the Mac).
  for k in 0 1 2 3 4 5 6 7 8 9; do
    idx=train_indices_seed$k.txt; [ $k = 0 ] && idx=train_indices.txt
    add plain196_emlp1_s$k plain-twin cpu 0.002 "$NPY196" $BIN/mnist $(plain196 $k $idx) --out {OUT}
  done
  for k in 0 1 2; do
    add plain196_emlp2_s$k plain-twin cpu 0.002 "$NPY196" $BIN/mnist $(plain23 emlp2 64 $k) --out {OUT}
    add plain196_emlp3_s$k plain-twin cpu 0.002 "$NPY196" $BIN/mnist $(plain23 emlp3 128,64 $k) --out {OUT}
  done
  for b in 32 48 64; do
    add plain784_reboot_init_b$b plain-twin cpu 0.002 "$IDX_MNIST" $BIN/mnist --mode plain --faithful --batch $b --indices $RS/reboot_subsample_indices.txt --init-weights $RS/reboot_init_weights --data $MNIST --out {OUT}
  done
  # Task 51 dataset reproduction (plaintext, 10 seeds each). Basis: ~22 s per MNIST-family
  # config on the Mac with -j 4, x2 for the server.
  add plain_mnist_emlp1_paper plain-datasets cpu 0.02 "$IDX_MNIST" $BIN/plain --config crates/ckks-train-bench/configs/mnist_emlp1_paper.json --jobs $PLAIN_JOBS --out {OUT}
  add plain_fmnist_emlp1 plain-datasets cpu 0.02 "$IDX_FMNIST" $BIN/plain --config crates/ckks-train-bench/configs/fmnist_emlp1.json --jobs $PLAIN_JOBS --out {OUT}
  add plain_bc_mlp plain-datasets cpu 0.001 "" $BIN/plain --config crates/ckks-train-bench/configs/bc_mlp.json --jobs $PLAIN_JOBS --out {OUT}
  add plain_bc_mlp_split_per_run plain-datasets cpu 0.001 "" $BIN/plain --config crates/ckks-train-bench/configs/bc_mlp.json --split-seed per_run --jobs $PLAIN_JOBS --out {OUT}
  add plain_bc_mlp_standardize plain-datasets cpu 0.001 "" $BIN/plain --config crates/ckks-train-bench/configs/bc_mlp.json --scaling standardize --jobs $PLAIN_JOBS --out {OUT}
  # Baseline (a): the 2020 method (Mihara et al.) on hekit, Iris 4-10-3, seal_2020, seed 2020.
  # Basis: server iris_20 (5686 s, 1 thread).
  add iris_b20_e400_s2020 baseline-a cpu 1.58 "" $BIN/iris --epochs 400 --batch 20 --seed 2020 --out {OUT}
  add iris_b1_timing baseline-a cpu 0.01 "" $BIN/iris --batch 1 --time-iters 10 --warmup 1 --seed 2020 --out {OUT}
  # Baseline (c): ReBoot's per-sample layout in hekit (Task 33), n16_train, interactive.
  # Basis: Mac 1 thread 10261 s (Lazy, 1024) and 2482 s (PerSample, 256), x2 for the server.
  add reboot_layout_lazy_1024 baseline-c cpu 5.70 "$IDX_MNIST" $BIN/mnist_reboot_layout --mode lazy --subset 1024 --batch 32 --data $MNIST --out {OUT}
  add reboot_layout_persample_256 baseline-c cpu 1.40 "$IDX_MNIST" $BIN/mnist_reboot_layout --mode persample --subset 256 --batch 32 --data $MNIST --out {OUT}
else
  # --quick: every group of the full matrix, on 1-4 updates, 1 seed. Basis: the server's
  # per-update times (NI 196: 68 s / 120 s per update with / without bootstraps, FULL-1 log;
  # NI 784: 65-150 s; setup + keygen ~1.5 min) and the full-mode bases above, scaled.
  add q_ni196_wn_b64_sub128_s0 rq1-196 cpu 0.10 "$NPY196" $BIN/mnist --mode noninteractive --placement when_needed --batch 64 --faithful --dataset-npy $D196 --subset 128 --init-weights $RF/init_weights_seed0 --init-weights-manifest $RF/config_values.json --epochs 1 --transpose reuse --out {OUT}
  add q_ni196_wn_b48_sub96_s0 rq1-196 cpu 0.10 "$NPY196" $BIN/mnist --mode noninteractive --placement when_needed --batch 48 --faithful --dataset-npy $D196 --subset 96 --init-weights $RF/init_weights_seed0 --init-weights-manifest $RF/config_values.json --epochs 1 --transpose reuse --out {OUT}
  add q_ni784_wn_b32_sub64_s0 rq2-784 cpu 0.08 "$IDX_MNIST" $BIN/mnist $Q784 --placement when_needed --batch 32 --subset 64 --out {OUT}
  add q_ni784_wn_b64_sub128_s0 rq2-784 cpu 0.08 "$IDX_MNIST" $BIN/mnist $Q784 --placement when_needed --batch 64 --subset 128 --out {OUT}
  add q_ni784_au_b32_sub32_s0 rq2-784 cpu 0.06 "$IDX_MNIST" $BIN/mnist $Q784 --placement after_update --batch 32 --subset 32 --out {OUT}
  add q_ni784_aa_b32_sub32_s0 rq2-784 cpu 0.06 "$IDX_MNIST" $BIN/mnist $Q784 --placement after_activation --batch 32 --subset 32 --out {OUT}
  add q_int784_b32_sub32_cpu interactive cpu 0.12 "$IDX_MNIST" $BIN/mnist $INT784 --subset 32 --out {OUT}
  add q_int784_b32_l7_cuda_sub64 gpu-l7 cuda 0.05 "$IDX_MNIST" $GBIN/mnist $INT784 --subset 64 --backend cuda --top-level 7 --out {OUT}
  add q_int784_b32_l7_cpu_sub64 gpu-l7 cpu 0.08 "$IDX_MNIST" $BIN/mnist $INT784 --subset 64 --backend cpu --top-level 7 --out {OUT}
  add q_ablation_cuda_l7 rq3 cuda 0.12 "$IDX_MNIST" $GBIN/ablation --backend cuda --params n16_train --top-level 7 --optimizer nag --iters 2 --warmup 1 --data $MNIST --out {OUT}
  add q_ablation_cpu_l7_default rq3 cpu 0.05 "$IDX_MNIST" $BIN/ablation --backend cpu --params n16_train --top-level 7 --optimizer nag --configs default --iters 1 --warmup 1 --data $MNIST --out {OUT}
  # Insecure N = 2^12 preset: all 8 flag combinations, counts and closed-form checks only.
  add q_ablation_cpu_small rq3 cpu 0.01 "$IDX_MNIST" $BIN/ablation --backend cpu --params small --batch 2 --iters 2 --warmup 1 --data $MNIST --out {OUT}
  add q_ni196_emlp2_b48_sub48_s0 block-mlp cpu 0.12 "$NPY196" $BIN/mnist --mode noninteractive --placement when_needed --batch 48 --faithful --blocks 64 --hidden 32 --dataset-npy $D196 --subset 48 --init-weights $R23/init_weights_emlp2_seed0 --init-weights-manifest $R23/config_values.json --epochs 1 --transpose reuse --out {OUT}
  add q_ni196_emlp3_b48_sub48_s0 block-mlp cpu 0.18 "$NPY196" $BIN/mnist --mode noninteractive --placement when_needed --batch 48 --faithful --blocks 128,64 --hidden 32 --dataset-npy $D196 --subset 48 --init-weights $R23/init_weights_emlp3_seed0 --init-weights-manifest $R23/config_values.json --epochs 1 --transpose reuse --out {OUT}
  add q_plain196_emlp1_s0 plain-twin cpu 0.002 "$NPY196" $BIN/mnist $(plain196 0 train_indices.txt) --out {OUT}
  add q_plain196_emlp2_s0 plain-twin cpu 0.002 "$NPY196" $BIN/mnist $(plain23 emlp2 64 0) --out {OUT}
  add q_plain196_emlp3_s0 plain-twin cpu 0.002 "$NPY196" $BIN/mnist $(plain23 emlp3 128,64 0) --out {OUT}
  add q_plain_mnist_emlp1_paper plain-datasets cpu 0.005 "$IDX_MNIST" $BIN/plain --config crates/ckks-train-bench/configs/mnist_emlp1_paper.json --seeds 0 --jobs $PLAIN_JOBS --out {OUT}
  add q_plain_fmnist_emlp1 plain-datasets cpu 0.005 "$IDX_FMNIST" $BIN/plain --config crates/ckks-train-bench/configs/fmnist_emlp1.json --seeds 0 --jobs $PLAIN_JOBS --out {OUT}
  add q_plain_bc_mlp plain-datasets cpu 0.001 "" $BIN/plain --config crates/ckks-train-bench/configs/bc_mlp.json --seeds 0 --jobs $PLAIN_JOBS --out {OUT}
  add q_iris_b20_e2_s2020 baseline-a cpu 0.02 "" $BIN/iris --epochs 2 --batch 20 --seed 2020 --out {OUT}
  add q_iris_b1_timing baseline-a cpu 0.01 "" $BIN/iris --batch 1 --time-iters 2 --warmup 1 --seed 2020 --out {OUT}
  add q_reboot_layout_lazy_32 baseline-c cpu 0.20 "$IDX_MNIST" $BIN/mnist_reboot_layout --mode lazy --subset 32 --batch 32 --data $MNIST --out {OUT}
  add q_reboot_layout_persample_8 baseline-c cpu 0.05 "$IDX_MNIST" $BIN/mnist_reboot_layout --mode persample --subset 8 --batch 8 --data $MNIST --out {OUT}
fi

BUILD_CPU_EST=0.10
BUILD_GPU_EST=0.10

# ------------------------------------------------------------------------------------------
# Status of a run: done (at HEAD), missing-input, or todo.
# ------------------------------------------------------------------------------------------
is_done() { # id effective-command
  local id=$1
  python3 - "$OUT/meta/$id.run.json" "$OUT/$id.json" "$HEAD" "$SOURCE_ID" "$2" "$THREADS" "$BUILD_ID" <<'PY'
import hashlib, json, pathlib, shlex, sys
meta_path, out_path, head, source, command, threads, build = sys.argv[1:8]
binary = shlex.split(command)[0]
try:
    meta = json.load(open(meta_path))
    out = json.load(open(out_path))
    digest = hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest()
except (OSError, ValueError, TypeError):
    sys.exit(1)
if (meta.get('status') != 'done' or meta.get('git_commit') != head or
    meta.get('source_identity') != source or meta.get('binary_sha256') != digest or
    meta.get('command') != command or meta.get('rayon_num_threads') != threads or
    meta.get('build_identity') != build):
    sys.exit(1)
prov = out.get('provenance') if isinstance(out, dict) else None
expected = 'the ' + pathlib.Path(binary).name + ' binary (raw output, not assembled)'
sys.exit(0 if isinstance(prov, dict) and prov.get('git_commit') == head and
         prov.get('written_by') == expected and not prov.get('note') else 1)
PY
}

missing_inputs() { # needs
  local f miss=''
  for f in $1; do [ -e "$f" ] || miss="$miss $f"; done
  echo "${miss# }"
}

selected() { [ -z "$ONLY" ] || [[ "$1" =~ $ONLY ]]; }

# ------------------------------------------------------------------------------------------
# Plan (also the --dry-run output).
# ------------------------------------------------------------------------------------------
n=${#IDS[@]}
TODO=(); total=0; need_cuda=0; any_cpu=0; GROUP_LINES=''; failed=0
echo "run_all.sh: mode $MODE, HEAD $HEAD, output $OUT/, RAYON_NUM_THREADS=$THREADS"
if [ "$MODE" = full ]; then
  echo "seeds: encrypted rows 1 seed (ReBoot's seed-0 initial weights; Iris seed 2020); plaintext twins 10 seeds (eMLP-2/3: seeds 0-2, all ReBoot releases); Task 51 configs 10 seeds each"
else
  echo "seeds: 1 everywhere; subsets of 1-4 updates; quick numbers are pipeline checks, not paper numbers"
fi
[ -n "$DIRTY" ] && echo "warning: modified tracked or untracked files (runner metadata records them):" && echo "$DIRTY" | sed 's/^/  /'
echo "baseline (b), official ReBoot (OpenFHE): not invoked; it runs only on the server via gpu-runner (requests.md R-REB-*)"
echo
printf '%-30s %-14s %-5s %8s  %s\n' id group back est_h status
for ((i = 0; i < n; i++)); do
  id=${IDS[$i]}
  selected "$id" || continue
  miss=$(missing_inputs "${NEEDS[$i]}")
  if is_done "$id" "${CMDS[$i]//\{OUT\}/$OUT/$id.json}"; then st=done
  elif [ -n "$miss" ]; then st="missing-input: $miss"; failed=1
  else st=todo; TODO+=("$i")
    total=$(awk -v a="$total" -v b="${ESTS[$i]}" 'BEGIN { printf "%.3f", a + b }')
    GROUP_LINES="$GROUP_LINES${GROUPS_[$i]} ${ESTS[$i]}
"
    [ "${BACKENDS[$i]}" = cuda ] && need_cuda=1 || any_cpu=1
  fi
  printf '%-30s %-14s %-5s %8s  %s\n' "$id" "${GROUPS_[$i]}" "${BACKENDS[$i]}" "${ESTS[$i]}" "$st"
  [ "$DRY" = 1 ] && [ "$st" = todo ] && echo "    RAYON_NUM_THREADS=$THREADS nice -n 10 ${CMDS[$i]//\{OUT\}/$OUT/$id.json}"
done
build_h=0
if [ "$BUILD" = 1 ] && [ ${#TODO[@]} -gt 0 ]; then
  [ "$any_cpu" = 1 ] && build_h=$BUILD_CPU_EST
  [ "$need_cuda" = 1 ] && build_h=$(awk -v a="$build_h" -v b="$BUILD_GPU_EST" 'BEGIN { printf "%.2f", a + b }')
fi
echo
echo "estimated hours (todo runs only, sequential):"
printf '%s' "$GROUP_LINES" | awk 'NF { h[$1] += $2 } END { for (g in h) printf "  %-16s %8.2f\n", g, h[g] }' | sort
printf '  %-16s %8.2f\n' builds "$build_h"
printf '  %-16s %8.2f\n' TOTAL "$(awk -v a="$total" -v b="$build_h" 'BEGIN { print a + b }')"
if [ "$DRY" = 1 ]; then
  echo
  if [ "$BUILD" = 1 ]; then
    [ "$any_cpu" = 1 ] && echo "build: RUSTC=$RUSTC_BIN nice -n 10 cargo build -j $THREADS --release -p ckks-train-bench --features parallel --bins --target-dir $REPO/$CPU_TARGET --target $NATIVE_TARGET --config 'build.rustc-wrapper=\"\"' --config 'build.rustc-workspace-wrapper=\"\"'"
    [ "$need_cuda" = 1 ] && echo "build: RUSTC=$RUSTC_BIN nice -n 10 cargo build -j $THREADS --release -p ckks-train-bench --features cuda --bins --target-dir $REPO/$GPU_TARGET --target $NATIVE_TARGET --config 'build.rustc-wrapper=\"\"' --config 'build.rustc-workspace-wrapper=\"\"'"
  fi
  exit 0
fi
[ ${#TODO[@]} -eq 0 ] && { echo "nothing to do (missing inputs remain failures)"; exit "$failed"; }

if [ -n "$DIRTY" ] && [ "$ALLOW_DIRTY" = 0 ]; then
  echo "run_all.sh: refusing modified tracked or untracked files (use a clean clone, or --allow-dirty)" >&2
  exit 2
fi

# ------------------------------------------------------------------------------------------
# Build, then run sequentially.
# ------------------------------------------------------------------------------------------
mkdir -p "$OUT/logs" "$OUT/meta"
build_receipt() { # write|check binary_directory features
  python3 - "$1" "$2" "$3" "$HEAD" "$SOURCE_ID" "$BUILD_ID" <<'PY'
import hashlib, json, pathlib, sys
action, directory, features, head, source, build = sys.argv[1:7]
base = pathlib.Path(directory)
path = base / 'run_all-build.json'
try:
    binaries = {name: hashlib.sha256((base/name).read_bytes()).hexdigest()
                for name in ('mnist','plain','iris','ablation','mnist_reboot_layout')}
    expected = {'git_commit':head,'source_identity':source,'build_identity':build,
                'features':features,'binaries':binaries}
    if action == 'write':
        path.write_text(json.dumps(expected, indent=1)+'\n')
    elif json.loads(path.read_text()) != expected:
        raise ValueError('source, commit, feature or binary mismatch')
except (OSError, ValueError) as exc:
    print(f'run_all.sh: build receipt {path}: {exc}', file=sys.stderr)
    sys.exit(1)
PY
}
if [ "$BUILD" = 1 ]; then
  if [ "$any_cpu" = 1 ]; then
    RUSTC="$RUSTC_BIN" nice -n 10 cargo build -j "$THREADS" --release -p ckks-train-bench --features parallel --bins \
      --target-dir "$REPO/$CPU_TARGET" --target "$NATIVE_TARGET" \
      --config 'build.rustc-wrapper=""' --config 'build.rustc-workspace-wrapper=""' > "$OUT/logs/build_cpu.log" 2>&1 ||
      { echo "run_all.sh: CPU build failed, see $OUT/logs/build_cpu.log" >&2; exit 1; }
    [ "$(source_identity)" = "$SOURCE_ID" ] && build_receipt write "$BIN" parallel || exit 1
  fi
  if [ "$need_cuda" = 1 ]; then
    RUSTC="$RUSTC_BIN" nice -n 10 cargo build -j "$THREADS" --release -p ckks-train-bench --features cuda --bins \
      --target-dir "$REPO/$GPU_TARGET" --target "$NATIVE_TARGET" \
      --config 'build.rustc-wrapper=""' --config 'build.rustc-workspace-wrapper=""' > "$OUT/logs/build_cuda.log" 2>&1 ||
      { echo "run_all.sh: CUDA build failed, see $OUT/logs/build_cuda.log" >&2; exit 1; }
    [ "$(source_identity)" = "$SOURCE_ID" ] && build_receipt write "$GBIN" cuda || exit 1
  fi
else
  if [ "$any_cpu" = 1 ]; then build_receipt check "$BIN" parallel || exit 1; fi
  if [ "$need_cuda" = 1 ]; then build_receipt check "$GBIN" cuda || exit 1; fi
fi

TIMEV=()
/usr/bin/time -v true > /dev/null 2>&1 && TIMEV=(/usr/bin/time -v)

write_meta() { # id status rc start end command
  python3 - "$OUT/meta/$1.run.json" "$@" "$HEAD" "$DIRTY" "$THREADS" "$MODE" "$SOURCE_ID" "$TRACKED_DIRTY" "$BUILD_ID" <<'PY'
import hashlib, json, pathlib, platform, shlex, socket, sys
path, run_id, status, rc, start, end, command, head, dirty, threads, mode, source, tracked, build = sys.argv[1:15]
binary = pathlib.Path(shlex.split(command)[0])
meta = {
    'written_by': 'scripts/run_all.sh (runner metadata, not a result)',
    'id': run_id, 'mode': mode, 'status': status, 'exit_code': int(rc),
    'command': command, 'git_commit': head, 'git_modified_tracked_files': tracked or 'none',
    'git_modified_tracked_and_untracked_files': dirty or 'none',
    'source_identity': source, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'build_identity': build,
    'host': socket.gethostname(), 'uname': ' '.join(platform.uname()[:3] + platform.uname()[4:5]),
    'rayon_num_threads': threads, 'nice': 10,
    'started_unix': float(start), 'finished_unix': float(end), 'wall_s': float(end) - float(start),
    'output': f'results/run_all/{mode}/{run_id}.json', 'log': f'results/run_all/{mode}/logs/{run_id}.log',
}
try:
    with open(path, 'w') as f:
        json.dump(meta, f, indent=1)
        f.write('\n')
except OSError as exc:
    print(f'run_all.sh: cannot persist runner metadata {path}: {exc}', file=sys.stderr)
    sys.exit(1)
PY
}

for i in "${TODO[@]}"; do
  id=${IDS[$i]}
  out="$OUT/$id.json"
  cmd=${CMDS[$i]//\{OUT\}/$out}
  log="$OUT/logs/$id.log"
  echo "[$(date '+%F %T')] $id: start (est ${ESTS[$i]} h)"
  rm -f "$out"
  start=$(python3 -c 'import time; print(time.time())')
  # shellcheck disable=SC2086  # cmd is this script's own word list (no spaces in paths)
  RAYON_NUM_THREADS=$THREADS nice -n 10 ${TIMEV[@]+"${TIMEV[@]}"} $cmd > "$log" 2>&1
  rc=$?
  end=$(python3 -c 'import time; print(time.time())')
  if [ $rc -eq 0 ] && [ -s "$out" ]; then
    if python3 - "$out" "$HEAD" "${cmd%% *}" <<'PY'
import json, pathlib, sys
try:
    p = json.load(open(sys.argv[1]))['provenance']
    expected = 'the '+pathlib.Path(sys.argv[3]).name+' binary (raw output, not assembled)'
    if p['written_by'] != expected or p['git_commit'] != sys.argv[2] or p.get('note'):
        raise ValueError('writer, commit or historical annotation mismatch')
    if not p['finished_unix'] > p['started_unix'] > 0:
        raise ValueError('invalid writer timestamps')
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f'run_all.sh: output requires binary-written provenance: {exc}', file=sys.stderr)
    sys.exit(1)
PY
    then st=done; else st=failed; failed=1; fi
  elif [ $rc -eq 3 ] && grep -q 'SKIP' "$log"; then st=skipped-gpu-memory
  else st=failed; failed=1
  fi
  if ! write_meta "$id" "$st" "$rc" "$start" "$end" "$cmd"; then st=failed; failed=1; fi
  echo "[$(date '+%F %T')] $id: $st (exit $rc)"
done

for ((i = 0; i < n; i++)); do
  selected "${IDS[$i]}" || continue
  [ -n "$(missing_inputs "${NEEDS[$i]}")" ] && ! is_done "${IDS[$i]}" "${CMDS[$i]//\{OUT\}/$OUT/${IDS[$i]}.json}" && failed=1
done
# Regenerate only mode-local outputs after experiments. Canonical archived draft tables
# are an explicit --tables-only operation and never consume quick data.
python3 tools/paper_tables.py --from-run-all "$MODE" || failed=1
exit $failed
