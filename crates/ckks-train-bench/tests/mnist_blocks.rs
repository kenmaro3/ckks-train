//! Task 52: `mnist --blocks` (ReBoot's eMLP-2/3 block structure) end to end on a synthetic
//! MNIST-format data set (8 x 8 images, IDX files).
//!
//! - `--mode plain`: the block twin runs, and `--compare-steps` reads a ReBoot-format step log
//!   (here the run's own per-step values, so every step must match exactly).
//! - `--mode noninteractive` on `mini_boot` (n16_boot's chain at N = 2^10, insecure, real
//!   bootstrapping): one block `64 -> 16` (head `-> 16`) and the output stack `16 -> 8 -> 16`;
//!   every update's counter equals the closed form (the stages plus one hand-off bootstrap per
//!   update) and the weights stay within 2^-10 of the plaintext block twin.

use std::fs;
use std::path::Path;
use std::process::Command;

use ckks_train_bench::data::{Split, write_idx_u8};

/// xorshift64*: deterministic bytes for the synthetic images (test data only).
fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8
        })
        .collect()
}

fn write_split(dir: &Path, split: Split, n: usize, seed: u64) {
    let (fi, fl) = split.files();
    fs::write(dir.join(fi), write_idx_u8(&[n, 8, 8], &bytes(seed, n * 64))).unwrap();
    let labels: Vec<u8> = bytes(seed + 1, n).iter().map(|b| b % 10).collect();
    fs::write(dir.join(fl), write_idx_u8(&[n], &labels)).unwrap();
}

/// The number after `"key": ` (first occurrence).
fn number(json: &str, key: &str) -> f64 {
    let pat = format!("\"{key}\": ");
    let at = json.find(&pat).unwrap_or_else(|| panic!("{key} missing")) + pat.len();
    let end = json[at..].find([',', '\n']).map_or(json.len(), |e| at + e);
    json[at..end]
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{key}: {e}"))
}

/// The numbers of the one-line array after `"key": [`.
fn array(json: &str, key: &str) -> Vec<f64> {
    let pat = format!("\"{key}\": [");
    let at = json.find(&pat).unwrap_or_else(|| panic!("{key} missing")) + pat.len();
    let end = json[at..].find(']').unwrap() + at;
    json[at..end]
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

#[test]
fn mnist_blocks_plain_and_noninteractive() {
    let dir = std::env::temp_dir().join(format!("hekit-mnist-blocks-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    write_split(&dir, Split::Train, 16, 7);
    write_split(&dir, Split::Test, 20, 9);
    let base = |mode: &str| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_mnist"));
        c.args(["--mode", mode, "--blocks", "16", "--hidden", "8"])
            .args(["--subset", "16", "--test-subset", "20", "--batch", "8"])
            .arg("--data")
            .arg(&dir);
        c
    };

    // Plain, then the same run compared with its own step log in ReBoot's format.
    let plain = dir.join("plain.json");
    let st = base("plain")
        .args(["--faithful", "--epochs", "2", "--out"])
        .arg(&plain)
        .status()
        .unwrap();
    assert!(st.success());
    let json = fs::read_to_string(&plain).unwrap();
    let loss = array(&json, "step_batch_loss_sum");
    let acc = array(&json, "step_batch_acc_all_outputs");
    let lr = array(&json, "step_lr_on_batch_mean_gradient");
    assert_eq!(loss.len(), 4);
    let mut tsv =
        String::from("# a ReBoot-format step log\nstep\tepoch\tbatch\tlr\tbatch_loss\tbatch_acc\n");
    for i in 0..4 {
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:e}\t{}\n",
            i + 1,
            i / 2 + 1,
            i % 2 + 1,
            lr[i] / 8.0,
            loss[i],
            acc[i]
        ));
    }
    let steps = dir.join("steps.tsv");
    fs::write(&steps, tsv).unwrap();
    let cmp = dir.join("cmp.json");
    let st = base("plain")
        .args(["--faithful", "--epochs", "2", "--compare-steps"])
        .arg(&steps)
        .arg("--out")
        .arg(&cmp)
        .status()
        .unwrap();
    assert!(st.success());
    let json = fs::read_to_string(&cmp).unwrap();
    assert_eq!(number(&json, "batch_acc_equal"), 4.0);
    assert!(number(&json, "max_rel_batch_loss_diff") < 1e-12);

    // Non-interactive on mini_boot: 2 updates each.
    for (placement, extra) in [
        ("when_needed", vec!["--faithful"]),
        ("after_update", vec!["--lr", "0.05", "--out-width", "16"]),
    ] {
        let out = dir.join(format!("ni_{placement}.json"));
        let st = base("noninteractive")
            .args(["--params", "mini_boot", "--placement", placement])
            .args(extra)
            .arg("--out")
            .arg(&out)
            .status()
            .unwrap();
        assert!(st.success(), "{placement}");
        let json = fs::read_to_string(&out).unwrap();
        eprintln!("{json}");
        assert!(json.contains("\"every_update_equals_prediction\": true"));
        assert!(json.contains("\"all_checks_pass\": true"));
        // Review I-1: the hand-off bootstraps were decrypted and compared, every update.
        assert_eq!(number(&json, "slots_compared"), 2.0 * 8.0 * 16.0);
        assert!(number(&json, "log2_max_abs_err_vs_twin") < -10.0);
        assert_eq!(number(&json, "updates"), 2.0);
        assert_eq!(number(&json, "handoff_bootstraps"), 2.0);
        let ratio = number(&json, "measured_over_predicted");
        assert!((ratio - 1.0).abs() < 1e-12, "{ratio}");
        let diff = number(&json, "log2_max_abs_diff");
        assert!(diff < -10.0, "weights 2^{diff} from the twin");
    }
    fs::remove_dir_all(&dir).unwrap();
}
