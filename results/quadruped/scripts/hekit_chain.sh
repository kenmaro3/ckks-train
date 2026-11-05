#!/bin/bash
# Job 2 (interactive MNIST, Iris), Job 3 (n16_boot bootstrap bench) and requests R-41-1..5, one after another on
# quadruped-gpu (~/hekit-runs/hekit_chain.sh). Each step goes through run_job.sh (provenance, RSS watchdog,
# nice 10). Builds are separate untimed jobs so the timed runs do not include compilation.
set -u
R=$HOME/hekit-runs/run_job.sh
O=$HOME/hekit-runs/out
cd $HOME/hekit
export PATH=$HOME/.cargo/bin:$PATH

# --- Job 2a: interactive MNIST, single-threaded build (no 'parallel' feature)
bash $R hekit_build_release_chain $HOME/hekit 12 0 -- cargo build --release -p hekit-bench --bins
# Task 32 arguments, spelled out (all equal to the binary's defaults).
bash $R mnist_interactive_b32 $HOME/hekit 1 0 -- cargo run --release -p hekit-bench --bin mnist -- \
  --mode interactive --epochs 1 --subset 1024 --test-subset 10000 --batch 32 --hidden 32 --seed 32 \
  --lr 0.001 --momentum 0.9 --params n16_train --transpose reuse --data data/mnist \
  --out $O/mnist_interactive_b32/mnist.json

# --- Job 3: bootstrap bench (bench profile, its own artifacts)
bash $R bootstrap_bench_list $HOME/hekit 12 0 -- cargo bench -p hekit-ckks --features parallel --bench bootstrap -- --list
bash $R bootstrap_bench $HOME/hekit 12 0 -- \
  env HEKIT_BOOT_THREADS=1,12 cargo bench -p hekit-ckks --features parallel --bench bootstrap

# --- Requests R-41-1..4 (requests.md): parallel build, 12 threads
bash $R hekit_build_release_parallel $HOME/hekit 12 0 -- cargo build --release -p hekit-bench --features parallel --bins
NI="--mode noninteractive --subset 1024 --epochs 1 --transpose reuse"
bash $HOME/hekit-runs/r41.sh R-41-1 results/quadruped/mnist_emlp1_ni_after_update_b32.json 32 -- $NI --placement after_update --batch 32
bash $HOME/hekit-runs/r41.sh R-41-2 results/quadruped/mnist_emlp1_ni_after_update_b64.json 16 -- $NI --placement after_update --batch 64
bash $HOME/hekit-runs/r41.sh R-41-3 results/quadruped/mnist_emlp1_ni_when_needed_b32.json 32 -- $NI --placement when_needed --batch 32
bash $HOME/hekit-runs/r41.sh R-41-4 results/quadruped/mnist_emlp1_ni_when_needed_b64.json 16 -- $NI --placement when_needed --batch 64

# --- Job 2b: Iris, single-threaded build (cargo run rebuilds iris without 'parallel')
bash $R iris_20 $HOME/hekit 1 0 -- cargo run --release -p hekit-bench --bin iris -- \
  --epochs 400 --batch 20 --seed 2020 --out $O/iris_20/iris.json

# --- R-41-5 (lower priority): make sure target/release/mnist is the parallel build again
bash $R hekit_build_release_parallel2 $HOME/hekit 12 0 -- cargo build --release -p hekit-bench --features parallel --bins
bash $HOME/hekit-runs/r41.sh R-41-5 results/quadruped/mnist_emlp1_ni_after_activation_b32.json 32 -- $NI --placement after_activation --batch 32
echo CHAIN_DONE
