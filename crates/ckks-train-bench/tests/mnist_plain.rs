//! Task 41c (review m1): `mnist --mode plain` on a handmade `.npy` data set (with a SHA-256
//! manifest). Two epochs given by `--epoch-orders` train exactly as one pass over the
//! concatenated order given by `--indices` (the same per-step losses), and an lr schedule of the
//! wrong length is refused before training.

use std::fs;
use std::path::Path;
use std::process::Command;

use ckks_train_bench::data::sha256_hex;

/// A version-1 `.npy` (C order) with `descr` and raw little-endian `data`.
fn npy(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((h.len() as u16).to_le_bytes());
    out.extend(h.as_bytes());
    out.extend(data);
    out
}

/// xorshift64*: deterministic test data.
fn stream(seed: u64, n: usize) -> Vec<u64> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545_f491_4f6c_dd1d)
        })
        .collect()
}

fn write_dataset(dir: &Path) {
    let img = |n: usize, seed: u64| -> Vec<u8> {
        stream(seed, n * 16)
            .iter()
            .flat_map(|v| ((v >> 40) as f32 / (1u64 << 24) as f32).to_le_bytes())
            .collect()
    };
    let lab = |n: usize, seed: u64| -> Vec<u8> {
        stream(seed, n)
            .iter()
            .flat_map(|v| (v % 10).to_le_bytes())
            .collect()
    };
    let files = [
        (
            "train_images_14x14.npy",
            npy("<f4", "(24, 16)", &img(24, 1)),
        ),
        ("train_labels.npy", npy("<u8", "(24,)", &lab(24, 2))),
        ("test_images_14x14.npy", npy("<f4", "(10, 16)", &img(10, 3))),
        ("test_labels.npy", npy("<u8", "(10,)", &lab(10, 4))),
    ];
    let mut manifest = String::from("{\n");
    for (name, bytes) in &files {
        fs::write(dir.join(name), bytes).unwrap();
        manifest.push_str(&format!("  \"{name}\": \"{}\",\n", sha256_hex(bytes)));
    }
    manifest.push_str("}\n");
    fs::write(dir.join("data_manifest.json"), manifest).unwrap();
}

fn losses(json: &str) -> String {
    let at = json.find("\"step_batch_loss_sum\": [").unwrap();
    let end = json[at..].find(']').unwrap() + at;
    json[at..end].to_string()
}

#[test]
fn epoch_orders_equal_concatenated_indices() {
    let dir = std::env::temp_dir().join(format!("hekit-mnist-plain-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    write_dataset(&dir);
    // Two epochs of orders (a permutation each), and the same as one index list.
    let p0: Vec<u16> = (0..24).rev().collect();
    let p1: Vec<u16> = (0..24).map(|i| (i * 7) % 24).collect();
    let all: Vec<u16> = p0.iter().chain(&p1).copied().collect();
    let bytes: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(dir.join("orders.npy"), npy("<u2", "(2, 24)", &bytes)).unwrap();
    let text: Vec<String> = all.iter().map(u16::to_string).collect();
    fs::write(dir.join("idx.txt"), text.join(" ")).unwrap();
    let run = |extra: &[&str], out: &str| {
        let o = dir.join(out);
        let status = Command::new(env!("CARGO_BIN_EXE_mnist"))
            .args([
                "--mode",
                "plain",
                "--faithful",
                "--batch",
                "8",
                "--hidden",
                "8",
            ])
            .arg("--dataset-npy")
            .arg(&dir)
            .args(extra)
            .arg("--out")
            .arg(&o)
            .status()
            .unwrap();
        (status.success(), fs::read_to_string(&o).unwrap_or_default())
    };
    let orders = dir.join("orders.npy");
    let idx = dir.join("idx.txt");
    let (ok1, a) = run(
        &["--epochs", "2", "--epoch-orders", orders.to_str().unwrap()],
        "a.json",
    );
    let (ok2, b) = run(
        &["--epochs", "1", "--indices", idx.to_str().unwrap()],
        "b.json",
    );
    assert!(ok1 && ok2);
    assert_eq!(losses(&a), losses(&b), "the same 6 steps");
    // An lr schedule of the wrong length is refused before training (review I2).
    fs::write(dir.join("lr.txt"), "0.001\n0.001\n").unwrap();
    let lr = dir.join("lr.txt");
    let (ok3, _) = run(
        &[
            "--epochs",
            "2",
            "--epoch-orders",
            orders.to_str().unwrap(),
            "--lr-schedule",
            lr.to_str().unwrap(),
        ],
        "c.json",
    );
    assert!(!ok3, "2 values for 6 steps");
    fs::remove_dir_all(&dir).unwrap();
}
