//! Task 48: the `iris` binary with the RQ3 ablation flags on the CPU (seal_2020, interactive,
//! timing mode, batch 20): `hoist` off gives the default counter and bit-identical weights;
//! all flags off changes exactly the three matvecs of a step (two forward, one backward, each
//! of dimension 4: +1 rotation and +2 relins each) and keeps the weights on the twin.
//! The GPU comparisons are in `iris_cuda.rs`.

use std::process::Command;

use serde_json::Value;

fn timing(tag: &str, extra: &[&str]) -> Value {
    let out = std::env::temp_dir().join(format!("hekit-iris-{tag}-{}.json", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_iris"))
        .args(["--batch", "20", "--time-iters", "2", "--warmup", "0"])
        .args(extra)
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success(), "{tag}: iris exited with {status}");
    let v = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let _ = std::fs::remove_file(&out);
    v
}

#[test]
fn iris_counts_follow_the_flags() {
    let def = timing("default", &[]);
    let nohoist = timing("nohoist", &["--no-hoist"]);
    let naive = timing("naive", &["--no-hoist", "--no-bsgs", "--no-lazy-relin"]);
    let c = |v: &Value, k: &str| v["counter_per_iteration"][k].as_u64().unwrap();
    for k in ["rotations", "relins", "ct_ct_tensor", "rescales"] {
        assert_eq!(c(&def, k), c(&nohoist, k), "{k}: hoist changes no count");
    }
    assert_eq!(c(&naive, "rotations"), c(&def, "rotations") + 3);
    assert_eq!(c(&naive, "relins"), c(&def, "relins") + 6);
    let errs = |v: &Value| -> Vec<f64> {
        v["iterations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["weight_err"].as_f64().unwrap())
            .collect()
    };
    assert_eq!(errs(&def), errs(&nohoist), "hoist is bit-identical");
    assert!(errs(&naive).iter().all(|&e| e < 2f64.powi(-15)));
    assert_eq!(def["config"]["backend"]["eval_config"]["default"], true);
    assert_eq!(naive["config"]["backend"]["eval_config"]["bsgs"], false);
}
