//! Task 48 fix round 1 (review I1): the `ablation` binary writes the shared provenance block
//! (`hekit_bench::provenance`) into its result JSON: commit, modified files, command line,
//! build and features, machine, threads, load, peak RSS, timestamps, and the GPU (`null` on
//! the CPU). A cheap run: insecure `small`, the default configuration only, one iteration,
//! synthetic pixels.

use std::process::Command;

use serde_json::Value;

#[test]
fn ablation_json_carries_provenance() {
    let dir = std::env::temp_dir().join(format!("hekit-ablation-prov-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("ablation.json");
    let status = Command::new(env!("CARGO_BIN_EXE_ablation"))
        .args(["--params", "small", "--batch", "2", "--iters", "1"])
        .args(["--warmup", "0", "--configs", "default"])
        .arg("--data")
        .arg(dir.join("no-mnist-here"))
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success(), "ablation exited with {status}");
    let j: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let p = &j["provenance"];
    assert!(p.is_object(), "provenance block missing");
    for k in [
        "written_by",
        "git_commit",
        "git_modified_tracked_files",
        "command_line",
        "build",
        "rustc_on_path",
        "machine",
        "threads",
        "load_start",
        "load_end",
        "peak_rss_bytes",
        "started_unix",
        "finished_unix",
        "gpu",
    ] {
        assert!(p.get(k).is_some(), "provenance.{k} missing");
    }
    assert!(p["written_by"].as_str().unwrap().contains("ablation"));
    assert!(
        p["command_line"]
            .as_str()
            .unwrap()
            .contains("--configs default")
    );
    assert!(p["machine"]["cpu"].is_string());
    assert!(p["build"]["features"]["cuda"].is_boolean());
    assert!(p["gpu"].is_null(), "CPU run: no GPU");
    assert!(p["finished_unix"].as_f64().unwrap() >= p["started_unix"].as_f64().unwrap());
    assert_eq!(j["all_checks_pass"], true);
    let _ = std::fs::remove_dir_all(&dir);
}
