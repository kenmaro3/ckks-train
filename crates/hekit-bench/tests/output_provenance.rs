//! New output provenance on previously uncovered binary paths, and a regression for
//! plaintext `--plan` accidentally running training. All data/results are tiny temporary
//! fixtures; every binary invocation has an explicit output path.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use hekit_bench::data::{Split, write_idx_u8};
use serde_json::Value;

struct Temp(PathBuf);

impl Temp {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = PathBuf::from("/tmp")
            .join(format!("hekit-output-{tag}-{}-{nanos}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn mnist(&self) {
        for split in [Split::Train, Split::Test] {
            let (images, labels) = split.files();
            let pixels: Vec<u8> = (0..128).map(|i| ((i * 17 + 3) % 256) as u8).collect();
            fs::write(self.0.join(images), write_idx_u8(&[2, 8, 8], &pixels)).unwrap();
            fs::write(self.0.join(labels), write_idx_u8(&[2], &[0, 1])).unwrap();
        }
    }

    fn iris(&self) {
        fs::write(
            self.0.join("iris.csv"),
            "a,b,c,d,label\n1,2,3,4,0\n2,3,4,5,1\n1.5,2.5,3.5,4.5,2\n",
        )
        .unwrap();
        fs::write(self.0.join("split.txt"), "train:0,1\ntest:2\n").unwrap();
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn success(output: Output) {
    assert!(
        output.status.success(),
        "status {}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

fn check_provenance(path: &Path, binary: &str, marker: &str, started: f64) -> Value {
    let ended = hekit_bench::provenance::unix_now();
    let result: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let p = &result["provenance"];
    assert!(p.is_object(), "{binary}: provenance block missing");
    assert_eq!(
        p["written_by"],
        format!("the {binary} binary (raw output, not assembled)")
    );
    let head = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(head.status.success());
    assert_eq!(
        p["git_commit"],
        String::from_utf8(head.stdout).unwrap().trim()
    );
    let command = p["command_line"].as_str().unwrap();
    assert!(command.contains(binary), "wrong binary: {command}");
    assert!(
        command.contains(marker),
        "missing mode arguments: {command}"
    );
    assert!(
        command.contains(path.to_str().unwrap()),
        "wrong output identity"
    );
    for key in [
        "git_modified_tracked_files",
        "rustc_on_path",
        "threads",
        "load_start",
        "load_end",
    ] {
        assert!(
            p[key].as_str().is_some_and(|v| !v.is_empty()),
            "{binary}: {key}"
        );
    }
    for key in ["cpu", "host", "os", "memory_bytes"] {
        assert!(
            p["machine"][key]
                .as_str()
                .is_some_and(|v| !v.is_empty() && v != "unknown"),
            "{binary}: machine.{key}"
        );
    }
    assert!(p["machine"]["logical_cpus"].as_u64().unwrap() > 0);
    assert_eq!(p["parallel_feature"], cfg!(feature = "parallel"));
    assert_eq!(
        p["build"]["features"]["parallel"],
        cfg!(feature = "parallel")
    );
    if cfg!(feature = "parallel") {
        let cap = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into());
        assert!(
            p["threads"]
                .as_str()
                .unwrap()
                .contains(&format!("RAYON_NUM_THREADS = {cap}"))
        );
    } else {
        assert_eq!(p["threads"], "1 (no parallel feature)");
    }
    assert!(p["build"]["features"]["cuda"].is_boolean());
    assert_eq!(p["build"]["target_os"], std::env::consts::OS);
    assert_eq!(p["build"]["target_arch"], std::env::consts::ARCH);
    assert!(p["peak_rss_bytes"].as_u64().unwrap() > 0);
    let start = p["started_unix"].as_f64().unwrap();
    let finish = p["finished_unix"].as_f64().unwrap();
    assert!(started <= start && start < finish && finish <= ended);
    result
}

#[test]
fn plain_plan_never_trains_or_replaces_result() {
    let dir = Temp::new("plain-plan");
    dir.mnist();
    // Exercise both the ordinary and multi-block plain dispatch, in both argument orders.
    for (i, extra) in [vec![], vec!["--blocks", "4,2"]].iter().enumerate() {
        let out = dir.0.join(format!("existing-{i}.json"));
        fs::write(&out, b"published measurement sentinel\n").unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_mnist"))
            .args(if i == 0 {
                vec!["--plan", "--mode", "plain"]
            } else {
                vec!["--mode", "plain", "--plan"]
            })
            .args([
                "--subset",
                "2",
                "--test-subset",
                "2",
                "--batch",
                "2",
                "--epochs",
                "1",
                "--hidden",
                "4",
            ])
            .args(extra)
            .arg("--data")
            .arg(&dir.0)
            .arg("--out")
            .arg(&out)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "plain plan must be rejected before training; stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(fs::read(&out).unwrap(), b"published measurement sentinel\n");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("--plan") && error.contains("plain"));
        fs::remove_file(&out).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_mnist"))
            .args(["--mode", "plain", "--plan"])
            .args(extra)
            .arg("--data")
            .arg(dir.0.join("missing-data"))
            .arg("--out")
            .arg(&out)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "argument error must precede data access"
        );
        assert!(!out.exists());
    }
}

#[test]
fn mnist_interactive_json_records_real_run_provenance() {
    let dir = Temp::new("mnist");
    dir.mnist();
    let out = dir.0.join("interactive.json");
    let started = hekit_bench::provenance::unix_now();
    success(
        Command::new(env!("CARGO_BIN_EXE_mnist"))
            .args([
                "--mode",
                "interactive",
                "--params",
                "small",
                "--hidden",
                "4",
            ])
            .args([
                "--subset",
                "2",
                "--test-subset",
                "2",
                "--batch",
                "2",
                "--epochs",
                "1",
            ])
            .arg("--data")
            .arg(&dir.0)
            .arg("--out")
            .arg(&out)
            .output()
            .unwrap(),
    );
    let j = check_provenance(&out, "mnist", "--mode interactive", started);
    assert_eq!(j["training"]["samples"], 2);
    assert!(j["measured"]["total"]["key_switches"].as_u64().unwrap() > 0);
    assert_eq!(j["measured"]["every_batch_equals_prediction"], true);
}

#[test]
fn layout_json_records_real_run_provenance() {
    let dir = Temp::new("layout");
    dir.mnist();
    let out = dir.0.join("layout.json");
    let started = hekit_bench::provenance::unix_now();
    success(
        Command::new(env!("CARGO_BIN_EXE_mnist_reboot_layout"))
            .args([
                "--mode", "lazy", "--params", "small", "--r", "64", "--c", "32",
            ])
            .args([
                "--subset",
                "2",
                "--test-subset",
                "2",
                "--batch",
                "2",
                "--hidden",
                "4",
            ])
            .arg("--data")
            .arg(&dir.0)
            .arg("--out")
            .arg(&out)
            .output()
            .unwrap(),
    );
    let j = check_provenance(&out, "mnist_reboot_layout", "--mode lazy", started);
    assert_eq!(j["mode"], "lazy");
    assert_eq!(j["training"]["samples"], 2);
    assert!(j["measured"]["total"]["key_switches"].as_u64().unwrap() > 0);
}

#[test]
fn iris_training_timing_and_ni_json_record_real_run_provenance() {
    let dir = Temp::new("iris");
    dir.iris();
    for (tag, extra) in [
        (
            "training",
            vec![
                "--epochs",
                "2",
                "--checkpoint-every",
                "1",
                "--top-level",
                "16",
            ],
        ),
        (
            "timing",
            vec!["--time-iters", "1", "--warmup", "0", "--top-level", "16"],
        ),
        (
            "noninteractive",
            vec!["--mode", "noninteractive", "--epochs", "1"],
        ),
    ] {
        let out = dir.0.join(format!("{tag}.json"));
        let started = hekit_bench::provenance::unix_now();
        success(
            Command::new(env!("CARGO_BIN_EXE_iris"))
                .args(["--preset", "mini_boot", "--batch", "2"])
                .args(extra)
                .arg("--data")
                .arg(dir.0.join("iris.csv"))
                .arg("--split")
                .arg(dir.0.join("split.txt"))
                .arg("--out")
                .arg(&out)
                .output()
                .unwrap(),
        );
        let j = check_provenance(&out, "iris", "--preset mini_boot", started);
        assert_eq!(j["mode"], if tag == "training" { "train" } else { tag });
        if tag == "training" {
            assert_eq!(j["epochs_done"], 2);
            assert_eq!(j["complete"], true);
        }
    }
}

#[test]
fn encrypted_plans_still_return_predictions_without_writing_results() {
    let dir = Temp::new("encrypted-plans");
    dir.mnist();
    for (mode, params, extra) in [
        ("interactive", "small", vec![]),
        ("noninteractive", "mini_boot", vec![]),
        ("noninteractive", "mini_boot", vec!["--blocks", "4,2"]),
    ] {
        let out = dir.0.join("untouched.json");
        fs::write(&out, b"published measurement sentinel\n").unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_mnist"))
            .args(["--plan", "--mode", mode, "--params", params])
            .args([
                "--subset",
                "2",
                "--test-subset",
                "2",
                "--batch",
                "2",
                "--hidden",
                "4",
            ])
            .args(extra)
            .arg("--data")
            .arg(&dir.0)
            .arg("--out")
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let plan: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(plan.is_object());
        assert!(
            plan.get("measured").is_none(),
            "plan cannot supply measured training results"
        );
        assert!(
            plan.get("provenance").is_none(),
            "prediction is not a measured result"
        );
        assert_eq!(fs::read(&out).unwrap(), b"published measurement sentinel\n");
    }
}
