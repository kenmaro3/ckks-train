//! Task 32 smoke test: the `mnist` binary end to end on a synthetic MNIST-format data set
//! (8 x 8 images, written as IDX files) with the insecure `small` parameters (N = 2^12): the
//! same code path as the n16_train run (PolyRelu2 -> identity, NAG, B = 32 lanes, zero-new-key
//! transpose, client refresh), with one full and one partial batch. Checks that every batch's
//! key switches equal the closed-form prediction (which is the cost model plus the documented
//! corrections) and that the weights stay within 2^-10 of the plaintext NAG twin.

use std::fs;
use std::path::Path;
use std::process::Command;

use hekit_bench::data::{Split, write_idx_u8};

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

#[test]
fn mnist_binary_small_params_end_to_end() {
    let dir = std::env::temp_dir().join(format!("hekit-mnist-smoke-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    write_split(&dir, Split::Train, 40, 7);
    write_split(&dir, Split::Test, 20, 9);
    let out = dir.join("out.json");
    let status = Command::new(env!("CARGO_BIN_EXE_mnist"))
        .args(["--params", "small", "--hidden", "16", "--subset", "40"])
        .args(["--test-subset", "20", "--batch", "32", "--lr", "0.05"])
        .arg("--data")
        .arg(&dir)
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success(), "mnist exited with {status}");
    let json = fs::read_to_string(&out).unwrap();
    eprintln!("{json}");
    assert!(json.contains("\"every_batch_equals_prediction\": true"));
    assert_eq!(number(&json, "batches"), 2.0);
    assert_eq!(number(&json, "samples"), 40.0);
    let exact = &json[json.find("\"exact\"").unwrap()..];
    assert_eq!(
        number(&json, "cost_model_plus_all_corrections"),
        number(exact, "key_switches"),
        "the corrections close the gap to the model exactly"
    );
    let measured = &json[json.find("\"measured\"").unwrap()..];
    assert_eq!(
        number(measured, "key_switches"),
        2.0 * number(exact, "key_switches"),
        "two batches at the predicted count each"
    );
    let diff = number(&json, "log2_max_abs_diff");
    assert!(diff < -10.0, "weights 2^{diff} from the twin");
    fs::remove_dir_all(&dir).unwrap();
}
