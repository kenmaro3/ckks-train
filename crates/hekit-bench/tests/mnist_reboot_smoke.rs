//! Smoke test of the `mnist_reboot_layout` binary: synthetic 8 x 8 IDX data, insecure `small`
//! parameters (N = 2^12, 2048 slots, `(r, c) = (64, 32)`), one full and one partial batch,
//! both relinearisation modes. The binary itself errors if a batch's key switches differ from
//! the closed form; this checks the exit status, the counts in the JSON and the twin diff.

use std::fs;
use std::path::Path;
use std::process::Command;

use hekit_bench::data::{Split, write_idx_u8};

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

fn number(json: &str, key: &str) -> f64 {
    let pat = format!("\"{key}\": ");
    let at = json.find(&pat).unwrap_or_else(|| panic!("{key} missing")) + pat.len();
    let end = json[at..].find([',', '\n']).map_or(json.len(), |e| at + e);
    json[at..end].trim().parse().unwrap()
}

#[test]
fn reboot_binary_small_params_end_to_end() {
    let dir = std::env::temp_dir().join(format!("hekit-reboot-smoke-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    write_split(&dir, Split::Train, 40, 7);
    write_split(&dir, Split::Test, 20, 9);
    // (mode, relins for the 32-sample batch + the 8-sample batch)
    for (mode, relins) in [("lazy", (5 * 32 + 2) + (5 * 8 + 2)), ("persample", 7 * 40)] {
        let out = dir.join(format!("{mode}.json"));
        let status = Command::new(env!("CARGO_BIN_EXE_mnist_reboot_layout"))
            .args([
                "--mode", mode, "--params", "small", "--r", "64", "--c", "32",
            ])
            .args(["--subset", "40", "--test-subset", "20", "--batch", "32"])
            .args(["--lr", "0.05"])
            .arg("--data")
            .arg(&dir)
            .arg("--out")
            .arg(&out)
            .status()
            .unwrap();
        assert!(status.success(), "{mode} exited with {status}");
        let json = fs::read_to_string(&out).unwrap();
        assert_eq!(number(&json, "samples"), 40.0);
        assert_eq!(number(&json, "batches"), 2.0);
        let total = &json[json.find("\"total\"").unwrap()..];
        assert_eq!(number(total, "rotations"), 40.0 * 22.0);
        assert_eq!(number(total, "relins"), relins as f64, "{mode}");
        let diff = number(&json, "log2_max_abs_diff");
        assert!(diff < -10.0, "{mode}: weights 2^{diff} from the twin");
    }
    let _ = fs::remove_dir_all(&dir);
}
