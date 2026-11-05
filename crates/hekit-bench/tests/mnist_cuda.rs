//! Task 48 Step 1 (GPU, `#[ignore]`, run on quadruped-gpu with `--features cuda`): the `mnist`
//! binary's interactive training on real MNIST (subset 1024, 784 -> 32 -> 10, NAG) with
//! `--backend cuda` gives the same `OpCounter` as `--backend cpu` in **every batch** (rotations,
//! relins, key switches, products, rescales, bootstraps: exactly) and decrypted final weights
//! and biases within 2^-12 of the CPU run's.
//!
//! Three variants, each declaring its GPU memory need (the binary estimates it before any key
//! exists and exits with code 3 and a `SKIP` line when the shared card has less free; the test
//! then prints the skip and passes):
//!
//! - `..._n16_train_full_keys`: n16_train, top level 31, full keys (relin + 31 Galois keys at
//!   288 MiB each, 9.0 GiB, plus the network at level 31: 15560 MiB declared). Does not
//!   fit next to the other tenant's ~20.7 GB; expected to skip until the card frees up.
//! - `..._n16_train_level7`: the same network, data and keys, with the network, the batches and
//!   every client refresh on level 7 (`--top-level 7`) and the device keys uploaded for levels
//!   <= 7 (24 MiB each): about 2.4 GiB declared (2456 MiB). The eMLP-1 step uses exactly 7 levels on a full
//!   batch (subset 1024 at B = 32 has full batches only). Same parameters and security as the
//!   full run; only fewer limbs per ciphertext.
//! - `..._small_insecure`: the `small` preset (N = 2^12, 11 primes, insecure), so d = 1024 and
//!   B = 2 lanes: the real 784 -> 32 -> 10 network and data at a reduced layout, about 0.4 GiB.
//!
//! The MNIST IDX files are read from `HEKIT_MNIST_DIR` (default `<workspace>/data/mnist`); if
//! they are missing the test prints `SKIP` and passes. The CPU run of each variant takes
//! minutes (small) to about an hour (n16_train full, 12 threads); it only starts once the GPU
//! run has finished.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

const SKIP_EXIT_CODE: i32 = hekit_bench::backend::SKIP_EXIT_CODE;

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn mnist_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("HEKIT_MNIST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace().join("data/mnist"));
    dir.join("train-images-idx3-ubyte").exists().then_some(dir)
}

fn out_dir() -> PathBuf {
    let d = std::env::var_os("HEKIT_TASK48_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("hekit-task48-{}", std::process::id()))
        });
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Runs `mnist` on `backend` with `args`; `None` if the binary skipped (GPU memory).
fn run(test: &str, backend: &str, data: &PathBuf, args: &[&str]) -> Option<Value> {
    let out = out_dir().join(format!("{test}_{backend}.json"));
    let status = Command::new(env!("CARGO_BIN_EXE_mnist"))
        .args([
            "--backend",
            backend,
            "--subset",
            "1024",
            "--test-subset",
            "1000",
        ])
        .arg("--dump-weights")
        .args(args)
        .arg("--data")
        .arg(data)
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    if status.code() == Some(SKIP_EXIT_CODE) {
        eprintln!("SKIP {test}: the {backend} run does not fit the free GPU memory (see above)");
        return None;
    }
    assert!(
        status.success(),
        "{test}: mnist --backend {backend} exited with {status}"
    );
    eprintln!("{test}: {backend} result {}", out.display());
    Some(serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap())
}

fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        Value::Object(o) => o.values().flat_map(flat).collect(),
        _ => panic!("unexpected {v}"),
    }
}

fn check(test: &str, batches: usize, args: &[&str]) {
    let Some(data) = mnist_dir() else {
        eprintln!("SKIP {test}: MNIST IDX files not found (set HEKIT_MNIST_DIR)");
        return;
    };
    let Some(gpu) = run(test, "cuda", &data, args) else {
        return;
    };
    let cpu = run(test, "cpu", &data, args).expect("the CPU run never skips");
    let rows = |j: &Value| j["measured"]["per_batch_counters"]["rows"].clone();
    assert_eq!(
        rows(&gpu),
        rows(&cpu),
        "{test}: per-batch OpCounter differs"
    );
    assert_eq!(
        rows(&gpu).as_array().unwrap().len(),
        batches,
        "{test}: batches"
    );
    assert_eq!(gpu["measured"]["total"], cpu["measured"]["total"], "{test}");
    for j in [&gpu, &cpu] {
        assert_eq!(
            j["measured"]["every_batch_equals_prediction"], true,
            "{test}"
        );
    }
    let (wg, wc) = (flat(&gpu["final_weights"]), flat(&cpu["final_weights"]));
    assert_eq!(wg.len(), wc.len());
    let diff = wg
        .iter()
        .zip(&wc)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!(
        "{test}: {} weights, max |W_gpu - W_cpu| = {diff:e} (2^{:.1}); GPU step {} s/batch, CPU \
         step {} s/batch; GPU memory after trims {}",
        wg.len(),
        diff.log2(),
        gpu["time_s"]["per_batch_step"]["mean"],
        cpu["time_s"]["per_batch_step"]["mean"],
        gpu["backend"]["memory_after_trims"]
    );
    assert!(diff <= 2f64.powi(-12), "{test}: weights differ by {diff:e}");
}

#[test]
#[ignore = "GPU (quadruped-gpu): needs ~15.2 GiB of GPU memory and about an hour of CPU"]
fn interactive_cuda_matches_cpu_n16_train_full_keys() {
    check("n16_train_full_keys", 32, &["--params", "n16_train"]);
}

#[test]
#[ignore = "GPU (quadruped-gpu): needs ~2.4 GiB of GPU memory and ~15 min of CPU"]
fn interactive_cuda_matches_cpu_n16_train_level7() {
    check(
        "n16_train_level7",
        32,
        &["--params", "n16_train", "--top-level", "7"],
    );
}

#[test]
#[ignore = "GPU (quadruped-gpu): needs ~0.4 GiB of GPU memory, insecure N = 2^12"]
fn interactive_cuda_matches_cpu_small_insecure() {
    check(
        "small_insecure",
        512,
        &["--params", "small", "--batch", "2"],
    );
}
