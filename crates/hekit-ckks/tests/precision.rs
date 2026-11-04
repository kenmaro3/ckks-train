//! Precision of every basic operation, for both presets (see `docs/precision.md`).

use hekit_ckks::params::CkksParams;
use hekit_ckks::precision::{PrecisionReport, precision_bits, run_precision_suite};

#[test]
fn precision_bits_definition() {
    // max abs error 2^-10 (the other error is smaller).
    let e = [1.0, 2.0, 3.0];
    let a = [1.0 + 2f64.powi(-10), 2.0 + 2f64.powi(-12), 3.0];
    assert!((precision_bits(&e, &a) - 10.0).abs() < 1e-9);
    // Sign of the error does not matter.
    let a = [1.0 - 2f64.powi(-10), 2.0, 3.0];
    assert!((precision_bits(&e, &a) - 10.0).abs() < 1e-9);
    // Exact match: infinite precision.
    assert_eq!(precision_bits(&e, &e), f64::INFINITY);
}

fn check(reports: &[PrecisionReport], min_bits: f64) {
    let ops: Vec<&str> = reports.iter().map(|r| r.op.as_str()).collect();
    assert_eq!(
        ops,
        [
            "encode/decode",
            "encrypt",
            "add",
            "mul+rescale",
            "rotate",
            "matvec",
            "matvec_enc"
        ]
    );
    for r in reports {
        eprintln!("{:<14} level {:>2}  {:.2} bits", r.op, r.level, r.bits);
        assert!(r.bits >= min_bits, "{} has {:.2} bits", r.op, r.bits);
    }
}

#[test]
fn seal_2020_precision() {
    check(&run_precision_suite(CkksParams::seal_2020()), 20.0);
}

#[test]
fn n16_train_precision() {
    check(&run_precision_suite(CkksParams::n16_train()), 24.0);
}
