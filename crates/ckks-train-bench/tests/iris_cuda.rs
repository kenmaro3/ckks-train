//! Task 48 (GPU, `#[ignore]`, quadruped-gpu with `--features cuda`): the `iris` binary on
//! `--backend cuda` against `--backend cpu`, on configurations that fit the shared card's free
//! memory (the binary checks its need and exits with code 3 and a `SKIP` line otherwise):
//!
//! - interactive, seal_2020 (N = 2^15, 9 levels, the 2020 paper's setting), batch 20, timing
//!   mode with 1 warm-up and 5 timed iterations: the same `OpCounter` per iteration, the same
//!   per-iteration weight errors against the plaintext twin (within 2^-12 of each other; the
//!   GPU is bit-exact, so they are expected equal), and the GPU/CPU server times;
//! - non-interactive, mini_boot (N = 2^10, insecure), 1 epoch, batch 20: level operations on
//!   the GPU and every bootstrap on the CPU (`HostBootstrapper`); the same epoch counter
//!   (bootstraps included) and weight error within 2^-12 of the CPU run's.

#![cfg(feature = "cuda")]

use std::process::Command;

use serde_json::Value;

const SKIP_EXIT_CODE: i32 = ckks_train_bench::backend::SKIP_EXIT_CODE;

fn run(tag: &str, backend: &str, args: &[&str]) -> Option<Value> {
    let dir = std::env::var_os("HEKIT_TASK48_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join(format!("iris_{tag}_{backend}.json"));
    let status = Command::new(env!("CARGO_BIN_EXE_iris"))
        .args(["--backend", backend])
        .args(args)
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    if status.code() == Some(SKIP_EXIT_CODE) {
        eprintln!("SKIP iris {tag}: not enough free GPU memory (see above)");
        return None;
    }
    assert!(status.success(), "iris {tag} --backend {backend}: {status}");
    eprintln!("iris {tag}: {backend} result {}", out.display());
    Some(serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap())
}

#[test]
#[ignore = "GPU (quadruped-gpu): seal_2020 interactive, needs ~1.3 GiB of GPU memory (19 keys of 55 MiB)"]
fn iris_interactive_cuda_matches_cpu_seal_2020() {
    let args = ["--batch", "20", "--time-iters", "5", "--warmup", "1"];
    let Some(gpu) = run("seal_2020_timing", "cuda", &args) else {
        return;
    };
    let cpu = run("seal_2020_timing", "cpu", &args).unwrap();
    assert_eq!(gpu["counter_per_iteration"], cpu["counter_per_iteration"]);
    let errs = |v: &Value, k: &str| -> Vec<f64> {
        v["iterations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i[k].as_f64().unwrap())
            .collect()
    };
    for k in ["weight_err", "bias_err"] {
        let (g, c) = (errs(&gpu, k), errs(&cpu, k));
        let d = g
            .iter()
            .zip(&c)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "iris {k}: GPU {g:?}, CPU {c:?}, max diff {d:e} (exact: {})",
            g == c
        );
        assert!(d <= 2f64.powi(-12));
    }
    eprintln!(
        "iris seal_2020 batch 20: server s/iter GPU {} vs CPU {}; refresh GPU {} vs CPU {}; \
         memory {}",
        gpu["server_seconds"]["mean"],
        cpu["server_seconds"]["mean"],
        gpu["refresh_seconds"]["mean"],
        cpu["refresh_seconds"]["mean"],
        gpu["config"]["memory_after_trims"]
    );
}

#[test]
#[ignore = "GPU (quadruped-gpu): mini_boot non-interactive (CPU bootstrap), ~0.1 GiB of GPU memory"]
fn iris_noninteractive_cuda_matches_cpu_mini_boot() {
    let args = [
        "--mode",
        "noninteractive",
        "--preset",
        "mini_boot",
        "--epochs",
        "1",
        "--batch",
        "20",
    ];
    let Some(gpu) = run("mini_boot_ni", "cuda", &args) else {
        return;
    };
    let cpu = run("mini_boot_ni", "cpu", &args).unwrap();
    let (g, c) = (&gpu["epochs"][0], &cpu["epochs"][0]);
    assert_eq!(g["counter"], c["counter"], "epoch counter incl. bootstraps");
    assert!(g["counter"]["bootstraps"].as_u64().unwrap() > 0);
    for k in ["weight_err", "bias_err"] {
        let (a, b) = (g[k].as_f64().unwrap(), c[k].as_f64().unwrap());
        eprintln!("iris NI {k}: GPU {a:e}, CPU {b:e}");
        assert!((a - b).abs() <= 2f64.powi(-12));
    }
    eprintln!(
        "iris NI mini_boot: epoch s GPU {} vs CPU {}",
        g["seconds"], c["seconds"]
    );
}
