//! Task 48: the `mnist` binary under the RQ3 ablation flags (`--no-hoist --no-bsgs
//! --no-lazy-relin`), on the CPU backend, on synthetic MNIST-format data (8 x 8 images):
//!
//! - interactive (`small`, N = 2^12, insecure), all 8 flag combinations: every batch's
//!   counter equals the flag-aware closed form (the bin's `predict` with `EvalConfig`, Task 43
//!   M5), `hoist` changes no count, `bsgs` off adds rotations, and the weights stay within
//!   2^-10 of the plaintext twin;
//! - non-interactive (`mini_boot`, N = 2^10, insecure, real bootstrapping), all flags off:
//!   every update's counter equals `predict_update_with` (plus the bootstraps), which differs
//!   from the default flags' count;
//! - `--top-level` (a level-capped interactive run, the GPU memory variant): the network and the
//!   refresh live on the chosen level, the counts are unchanged.
//!
//! `--matvec batch` and `--backend cuda` without the feature are refused before any work.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use hekit_bench::data::{Split, write_idx_u8};
use hekit_ckks::EvalConfig;

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

fn data_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hekit-mnist-{tag}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    write_split(&dir, Split::Train, 40, 7);
    write_split(&dir, Split::Test, 20, 9);
    dir
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

fn flags(c: &EvalConfig) -> Vec<&'static str> {
    let mut f = Vec::new();
    if !c.hoist {
        f.push("--no-hoist");
    }
    if !c.bsgs {
        f.push("--no-bsgs");
    }
    if !c.lazy_relin {
        f.push("--no-lazy-relin");
    }
    f
}

/// Runs the interactive `small` smoke configuration with `extra` flags; returns the JSON.
fn interactive(dir: &Path, tag: &str, extra: &[&str]) -> String {
    let out = dir.join(format!("{tag}.json"));
    let status = Command::new(env!("CARGO_BIN_EXE_mnist"))
        .args(["--params", "small", "--hidden", "16", "--subset", "40"])
        .args(["--test-subset", "20", "--batch", "32", "--lr", "0.05"])
        .args(extra)
        .arg("--data")
        .arg(dir)
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success(), "{tag}: mnist exited with {status}");
    fs::read_to_string(&out).unwrap()
}

#[test]
fn interactive_counts_follow_the_flags() {
    let dir = data_dir("flags");
    let mut by_cfg = Vec::new();
    for c in EvalConfig::all() {
        let tag = format!(
            "h{}b{}l{}",
            u8::from(c.hoist),
            u8::from(c.bsgs),
            u8::from(c.lazy_relin)
        );
        let json = interactive(&dir, &tag, &flags(&c));
        assert!(
            json.contains("\"every_batch_equals_prediction\": true"),
            "{tag}: measured counts differ from the flag-aware prediction"
        );
        assert!(json.contains(&format!("\"hoist\": {}", c.hoist)), "{tag}");
        let diff = number(&json, "log2_max_abs_diff");
        assert!(diff < -10.0, "{tag}: weights 2^{diff} from the twin");
        let exact = &json[json.find("\"exact\"").unwrap()..];
        by_cfg.push((
            c,
            number(exact, "rotations") as u64,
            number(exact, "relins") as u64,
        ));
    }
    let get = |c: EvalConfig| *by_cfg.iter().find(|x| x.0 == c).unwrap();
    for (c, rot, relin) in &by_cfg {
        // hoist changes no OpCounter value.
        let h = get(EvalConfig { hoist: true, ..*c });
        assert_eq!((*rot, *relin), (h.1, h.2), "{c:?}");
        // bsgs off adds rotations; lazy_relin off adds relins.
        if !c.bsgs {
            assert!(*rot > get(EvalConfig { bsgs: true, ..*c }).1, "{c:?}");
        }
        if !c.lazy_relin {
            assert!(
                *relin
                    > get(EvalConfig {
                        lazy_relin: true,
                        ..*c
                    })
                    .2,
                "{c:?}"
            );
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

/// `--top-level 7` on `small` (max level 10) with full batches only: the network starts and
/// is refreshed on level 7, the counts are those of the full-level run, and the weights still
/// track the twin. The eMLP-1-shaped step (z^2 + z, then linear) uses exactly 7 levels on a
/// full batch (`W1'` lands on level 0), so 6 is refused at the first step, and a partial batch
/// (one more level for the linear output's valid-lane mask) needs 8. This is the level cap of
/// the GPU memory variant (Task 48: n16_train keys at level 7 are 24 MiB instead of 288 MiB;
/// MNIST subset 1024 at B = 32 has full batches only).
#[test]
fn interactive_top_level_cap() {
    let dir = data_dir("toplevel");
    let run = |tag: &str, subset: &str, extra: &[&str]| {
        let out = dir.join(format!("{tag}.json"));
        let status = Command::new(env!("CARGO_BIN_EXE_mnist"))
            .args(["--params", "small", "--hidden", "16", "--subset", subset])
            .args(["--test-subset", "20", "--batch", "32", "--lr", "0.05"])
            .args(extra)
            .arg("--data")
            .arg(&dir)
            .arg("--out")
            .arg(&out)
            .status()
            .unwrap();
        (status.code(), fs::read_to_string(&out).unwrap_or_default())
    };
    let (c_full, full) = run("full", "32", &[]);
    let (c_cap, capped) = run("capped", "32", &["--top-level", "7"]);
    assert_eq!((c_full, c_cap), (Some(0), Some(0)));
    assert_eq!(number(&capped, "start_level"), 7.0);
    assert_eq!(number(&full, "start_level"), 10.0);
    assert!(capped.contains("\"every_batch_equals_prediction\": true"));
    let m = |j: &str| {
        let t = &j[j.find("\"measured\"").unwrap()..];
        (number(t, "rotations"), number(t, "relins"))
    };
    assert_eq!(m(&full), m(&capped));
    assert!(number(&capped, "log2_max_abs_diff") < -10.0);
    assert_eq!(
        number(&capped, "W1'"),
        0.0,
        "the full batch uses all 7 levels"
    );
    // One level fewer, or a partial batch at the cap: no levels left (exit 1, no output).
    assert_eq!(run("low", "32", &["--top-level", "6"]).0, Some(1));
    assert_eq!(run("partial", "40", &["--top-level", "7"]).0, Some(1));
    assert_eq!(run("partial8", "40", &["--top-level", "8"]).0, Some(0));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn noninteractive_counts_follow_the_flags() {
    let dir = data_dir("niflags");
    let out = dir.join("naive.json");
    let status = Command::new(env!("CARGO_BIN_EXE_mnist"))
        .args(["--mode", "noninteractive", "--placement", "after_update"])
        .args(["--params", "mini_boot", "--hidden", "32", "--subset", "16"])
        .args(["--test-subset", "20", "--batch", "16", "--lr", "0.05"])
        .args(["--no-hoist", "--no-bsgs", "--no-lazy-relin"])
        .arg("--data")
        .arg(&dir)
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success(), "mnist exited with {status}");
    let json = fs::read_to_string(&out).unwrap();
    assert!(
        json.contains("\"every_update_equals_prediction\": true"),
        "NI counts differ from predict_update_with under all flags off"
    );
    assert!(json.contains("\"bsgs\": false"));
    assert!(number(&json, "log2_max_abs_diff") < -10.0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn refused_options() {
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_mnist"))
            .args(args)
            .output()
            .unwrap()
    };
    let o = run(&["--matvec", "batch", "--plan"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("R-47-2"));
    let o = run(&["--backend", "cuda", "--plan"]);
    assert_eq!(
        o.status.code(),
        Some(if cfg!(feature = "cuda") { 0 } else { 2 })
    );
    let o = run(&["--mode", "noninteractive", "--top-level", "5", "--plan"]);
    assert_eq!(o.status.code(), Some(2));
    let o = run(&["--mode", "plain", "--no-bsgs"]);
    assert_eq!(o.status.code(), Some(2));
}
