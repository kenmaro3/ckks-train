//! Task 41 smoke test: the `mnist` binary in `--mode noninteractive` end to end on a synthetic
//! MNIST-format data set (8 x 8 images, IDX files) with `mini_boot` (n16_boot's chain at
//! N = 2^10, insecure): the same code path as the n16_boot run (PolyRelu2 -> identity, NAG,
//! compact state, real bootstrapping, zero-new-key transpose), with `--batch 16` as 2
//! micro-batch ciphertexts of 8 lanes per update. Checks that every update's key switches equal
//! the closed form, that the closed form is the cost model plus the listed corrections, and that
//! the weights stay within 2^-10 of the plaintext NAG twin.

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

#[test]
fn mnist_binary_noninteractive_mini_boot() {
    let dir = std::env::temp_dir().join(format!("hekit-mnist-ni-smoke-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    write_split(&dir, Split::Train, 32, 7);
    write_split(&dir, Split::Test, 20, 9);
    // (placement, faithful, batch, updates, micro): batch 16 = 2 x 8 lanes; batch 6 = one
    // 8-lane ciphertext with 6 valid lanes (Task 41b: B = 48 in 64 lanes, in miniature).
    for (placement, faithful, batch, updates, micro) in [
        ("after_update", false, "16", 2.0, 2.0),
        ("when_needed", false, "16", 2.0, 2.0),
        ("after_update", true, "16", 2.0, 2.0),
        ("when_needed", true, "6", 6.0, 1.0),
        ("when_needed", true, "16s", 2.0, 2.0),
    ] {
        // "16s": batch 16 with a per-step lr schedule (R-M3-14), in --lr-sum units.
        let (batch, schedule) = match batch.strip_suffix('s') {
            Some(b) => {
                let f = dir.join("lr.txt");
                fs::write(&f, "# lr per step\n0.0005\n0.0002\n").unwrap();
                (b, Some(f))
            }
            None => (batch, None),
        };
        let out = dir.join(format!(
            "{placement}_{faithful}_{batch}_{}.json",
            schedule.is_some()
        ));
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_mnist"));
        if let Some(f) = &schedule {
            cmd.arg("--lr-schedule").arg(f);
        }
        if faithful {
            // R-M3-12: no biases, Xavier init, lr 5e-4 on the batch-summed gradient (the
            // explicit --lr below is overridden by --lr-sum's conversion).
            cmd.arg("--faithful");
        }
        let status = cmd
            .args(["--mode", "noninteractive", "--placement", placement])
            .args(["--params", "mini_boot", "--hidden", "32", "--subset", "32"])
            .args(["--test-subset", "20", "--batch", batch])
            .args(if faithful {
                vec![]
            } else {
                vec!["--lr", "0.05"]
            })
            .arg("--data")
            .arg(&dir)
            .arg("--out")
            .arg(&out)
            .status()
            .unwrap();
        assert!(status.success(), "mnist exited with {status}");
        let json = fs::read_to_string(&out).unwrap();
        eprintln!("{json}");
        assert!(json.contains("\"every_update_equals_prediction\": true"));
        assert_eq!(number(&json, "updates"), updates);
        assert_eq!(number(&json, "micro_batches_per_update"), micro);
        // The measured per-sample count is the prediction (closed form with the schedule).
        let ratio = number(&json, "measured_over_predicted");
        assert!(
            (ratio - 1.0).abs() < 1e-12,
            "measured / predicted = {ratio}"
        );
        let diff = number(&json, "log2_max_abs_diff");
        assert!(diff < -10.0, "weights 2^{diff} from the twin");
    }
    fs::remove_dir_all(&dir).unwrap();
}
