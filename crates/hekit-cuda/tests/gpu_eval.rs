//! GPU tests of the CUDA evaluator (Task 47): `HeOps` on device ciphertexts, the RQ3 flags,
//! the operation counts, BSGS matrix-vector products and the multi-stream batch.
//!
//! Every result is compared with a CPU `Evaluator` holding the same keys **bit for bit**
//! (the matrix-vector products included: their operation sequence is the CPU's and every
//! primitive is bit-exact, so the brief's 2^-25 tolerance is not needed), and every
//! `OpCounter` **exactly**. The shared cases are in `tests/cases/eval.rs`; the always-built
//! unit tests run them with a second CPU evaluator.
//!
//! Each test is `#[ignore]`d, declares its device-memory need ([`need`]) and skips (prints
//! `SKIP <test>: needs .. MiB, only .. MiB free`, passes) when there is no usable GPU or not
//! enough free memory; the need is checked **before** keys are generated on the CPU. Run on a
//! CUDA machine with
//! `cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1`.
#![cfg(feature = "cuda")]

mod cases;

use std::sync::Arc;

use cases::eval::{EvalFixture, check_matvec_conformance, check_sequence};
use cases::ks::assert_same_ct;
use hekit_ckks::{CkksParams, Diagonals, EvalConfig, GaloisKey, HeOps, bsgs_rotation_steps};
use hekit_core::HekitError;
use hekit_cuda::matvec::matvec_enc_workspace_limbs;
use hekit_cuda::testing::{free_device_memory, ksk_bytes, require_gpu_mem, table_bytes};
use hekit_cuda::{CudaCiphertext, CudaDevice, CudaEvaluator};

const MIB: usize = 1 << 20;

/// Device-memory need of a test: the NTT tables, `keys` keys uploaded for levels
/// `<= key_level`, `polys` input/result polynomials over `Q_level` (a ciphertext is 2, a
/// plaintext 1), `workers` matvec working sets at `level` for dimension `d`
/// ([`matvec_enc_workspace_limbs`]), and 40 MiB for the caches, staging rings and allocator
/// slack.
fn need(
    p: &CkksParams,
    keys: usize,
    key_level: usize,
    level: usize,
    polys: usize,
    d: usize,
    workers: usize,
) -> usize {
    let limb = 8 * p.n();
    table_bytes(p)
        + keys * ksk_bytes(p, key_level).unwrap()
        + polys * (level + 1) * limb
        + workers * matvec_enc_workspace_limbs(p, level, d) * limb
        + 40 * MIB
}

/// A device with `streams` streams if `need_bytes` fit, else `SKIP`.
fn open(test: &str, p: &CkksParams, need_bytes: usize, streams: usize) -> Option<CudaDevice> {
    let dev = require_gpu_mem(test, need_bytes, p)?;
    if streams == 1 {
        return Some(dev);
    }
    drop(dev);
    Some(CudaDevice::new(0, p, streams).expect("CudaDevice with several streams"))
}

/// An evaluator with the relinearization key and the keys of `steps` (and every other key
/// of the fixture when `steps` is `None`), uploaded for levels `<= key_level`.
fn evaluator<'d>(
    dev: &'d CudaDevice,
    fx: &EvalFixture,
    steps: Option<&[i32]>,
    key_level: usize,
    config: EvalConfig,
) -> CudaEvaluator<'d> {
    let keys: Vec<&GaloisKey> = match steps {
        None => fx.galois_keys(),
        Some(s) => s.iter().map(|&s| fx.gks.get(s).unwrap()).collect(),
    };
    let ev = CudaEvaluator::with_keys(
        dev,
        Arc::clone(&fx.params),
        Some(&fx.rk),
        &keys,
        config,
        key_level,
    )
    .unwrap();
    assert_eq!(
        ev.key_bytes(),
        (keys.len() + 1) * ksk_bytes(&fx.params, key_level).unwrap()
    );
    ev
}

// ---------------------------------------------------------------------------------------
// The operation sequence: every HeOps operation, results and counters against the CPU.
// ---------------------------------------------------------------------------------------

/// Polynomials alive in `run_sequence` at once (inputs, diagonals, results), with margin
/// (trimmed in fix round 1: the full-key n16_train variant missed 2984.6 MiB free by 11 MiB).
fn sequence_polys(d: usize) -> usize {
    3 * d + 32
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn sequence_and_counts_match_cpu_seal_2020() {
    let p = CkksParams::seal_2020();
    let (d, l) = (16, p.max_level());
    // Keys: relin + 6 BSGS steps + 1 extra step + conjugation.
    let n = need(&p, 9, l, l, sequence_polys(d), d, 1);
    let Some(dev) = open("sequence_and_counts_match_cpu_seal_2020", &p, n, 1) else {
        return;
    };
    let fx = EvalFixture::new(p, d, 4701);
    let ev = evaluator(&dev, &fx, None, l, EvalConfig::default());
    let cpu = fx.cpu(EvalConfig::default());
    for level in [l, 5, 2] {
        check_sequence(&ev, &cpu, &fx, level, 10 + level as u64);
    }
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn sequence_and_counts_match_cpu_n16_train_capped() {
    // Keys capped at level 9 (42 MiB each; 3 digits of 4 + 4 + 2 primes): levels 9, 5, 2.
    let p = CkksParams::n16_train();
    let (d, kl) = (4, 9);
    // Keys: relin + 2 BSGS steps + 1 extra step + conjugation.
    let n = need(&p, 5, kl, kl, sequence_polys(d), d, 1);
    let Some(dev) = open("sequence_and_counts_match_cpu_n16_train_capped", &p, n, 1) else {
        return;
    };
    let fx = EvalFixture::new(p, d, 4702);
    let ev = evaluator(&dev, &fx, None, kl, EvalConfig::default());
    let cpu = fx.cpu(EvalConfig::default());
    for level in [kl, 5, 2] {
        check_sequence(&ev, &cpu, &fx, level, 20 + level as u64);
    }
    // Above the capped keys: InvalidParams before any work, nothing counted (the documented
    // deviation from the CPU, which holds full keys).
    let x = ev.upload_ct(&fx.encrypt(kl + 1, 7)).unwrap();
    ev.reset_counter();
    for r in [ev.rotate(&x, 1), ev.conjugate(&x), ev.mul(&x, &x)] {
        assert!(
            matches!(r, Err(HekitError::InvalidParams(_))),
            "{:?}",
            r.err()
        );
    }
    assert!(ev.hoisted_rotate(&x, &[1, 2]).is_err());
    assert_eq!(ev.counter(), hekit_ckks::OpCounter::default());
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn sequence_and_counts_match_cpu_n16_train_full_keys() {
    // Full keys (288 MiB each) at the top level: 2868 MiB; skips below that free.
    let p = CkksParams::n16_train();
    let (d, l) = (4, p.max_level());
    let n = need(&p, 5, l, l, sequence_polys(d), d, 1);
    let Some(dev) = open(
        "sequence_and_counts_match_cpu_n16_train_full_keys",
        &p,
        n,
        1,
    ) else {
        return;
    };
    let fx = EvalFixture::new(p, d, 4703);
    let ev = evaluator(&dev, &fx, None, l, EvalConfig::default());
    let cpu = fx.cpu(EvalConfig::default());
    check_sequence(&ev, &cpu, &fx, l, 31);
}

// ---------------------------------------------------------------------------------------
// Task 43 M3: matvec conformance for all 8 flag combinations at d = 16.
// ---------------------------------------------------------------------------------------

fn conformance(test: &str, p: CkksParams, key_level: usize, seed: u64) {
    let d = 16;
    let steps = bsgs_rotation_steps(d, 1);
    // x + d plaintext + d encrypted diagonals + results.
    let n = need(&p, steps.len() + 1, key_level, key_level, 3 * d + 8, d, 1);
    let Some(dev) = open(test, &p, n, 1) else {
        return;
    };
    let fx = EvalFixture::new(p, d, seed);
    let base = evaluator(&dev, &fx, Some(&steps), key_level, EvalConfig::default());
    for config in EvalConfig::all() {
        let ev = base.with_config(config);
        let cpu = fx.cpu(config);
        check_matvec_conformance(&ev, &cpu, &fx, key_level, seed + 1);
        eprintln!("{test}: {config:?}: {:?}", ev.counter());
    }
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn matvec_conformance_all_flags_d16_seal_2020() {
    let p = CkksParams::seal_2020();
    let l = p.max_level();
    conformance("matvec_conformance_all_flags_d16_seal_2020", p, l, 4710);
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn matvec_conformance_all_flags_d16_n16_train_capped() {
    conformance(
        "matvec_conformance_all_flags_d16_n16_train_capped",
        CkksParams::n16_train(),
        9,
        4720,
    );
}

// ---------------------------------------------------------------------------------------
// d = 64 (the brief's matvec test): bit-exact against the CPU.
// ---------------------------------------------------------------------------------------

/// `matvec_plain` and `matvec_enc` with d = 64 at `level` (keys capped at `level`): bit for
/// bit and count for count against the CPU (default flags).
fn matvec_d64(test: &str, p: CkksParams, level: usize, seed: u64) {
    let d = 64;
    let steps = bsgs_rotation_steps(d, 1);
    let n = need(&p, steps.len() + 1, level, level, 3 * d + 8, d, 1);
    let Some(dev) = open(test, &p, n, 1) else {
        return;
    };
    let fx = EvalFixture::new(p, d, seed);
    let ev = evaluator(&dev, &fx, Some(&steps), level, EvalConfig::default());
    let cpu = fx.cpu(EvalConfig::default());
    check_matvec_conformance(&ev, &cpu, &fx, level, seed + 1);
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn matvec_d64_matches_cpu_seal_2020() {
    let p = CkksParams::seal_2020();
    let l = p.max_level();
    matvec_d64("matvec_d64_matches_cpu_seal_2020", p, l, 4730);
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn matvec_d64_matches_cpu_n16_train_full() {
    // 15 full keys (4.2 GiB) and 64 top-level encrypted diagonals (2 GiB): skips on a GPU
    // with less than about 7 GiB free.
    let p = CkksParams::n16_train();
    let l = p.max_level();
    matvec_d64("matvec_d64_matches_cpu_n16_train_full", p, l, 4740);
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn matvec_d64_matches_cpu_n16_train_level9() {
    // Keys capped at level 9 (15 x 42 MiB) and level-9 diagonals: about 1.9 GiB.
    matvec_d64(
        "matvec_d64_matches_cpu_n16_train_level9",
        CkksParams::n16_train(),
        9,
        4750,
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn matvec_d64_matches_cpu_n16_train_level5() {
    // Keys capped at level 5 (15 x 20 MiB; 2 digits, the second clipped) and level-5
    // diagonals: about 1.1 GiB.
    matvec_d64(
        "matvec_d64_matches_cpu_n16_train_level5",
        CkksParams::n16_train(),
        5,
        4760,
    );
}

// ---------------------------------------------------------------------------------------
// Multi-stream batch.
// ---------------------------------------------------------------------------------------

/// `matvec_enc_batch` of `jobs` products on `streams` streams (the first half of the jobs
/// with their own `x` and diagonals, the rest sharing job 0's operands): every result equal
/// to `matvec_enc`'s and to the CPU's bit for bit, counts `jobs` times one product's.
fn batch(test: &str, p: CkksParams, d: usize, level: usize, streams: usize, seed: u64) {
    let jobs = 2 * streams;
    let steps = bsgs_rotation_steps(d, 1);
    let own = jobs / 2;
    let n = need(
        &p,
        steps.len() + 1,
        level,
        level,
        own * (2 * d + 2) + 2 * jobs + 8,
        d,
        streams + 1,
    );
    let Some(dev) = open(test, &p, n, streams) else {
        return;
    };
    assert_eq!(dev.n_streams(), streams);
    let fx = EvalFixture::new(p, d, seed);
    let ev = evaluator(&dev, &fx, Some(&steps), level, EvalConfig::default());
    let cpu = fx.cpu(EvalConfig::default());
    let mut host_x = Vec::new();
    let mut host_d = Vec::new();
    for j in 0..own {
        host_x.push(fx.encrypt(level, seed + 100 + j as u64));
        host_d.push(fx.diagonals(level, seed + 200 + 64 * j as u64).1);
    }
    let up_x: Vec<CudaCiphertext<'_>> = host_x.iter().map(|x| ev.upload_ct(x).unwrap()).collect();
    let up_d: Vec<Diagonals<CudaCiphertext<'_>>> = host_d
        .iter()
        .map(|dg| dg.try_map(|c| ev.upload_ct(c)).unwrap())
        .collect();
    // Jobs `own..` share job 0's operands (concurrent reads of one buffer on several streams).
    let src = |j: usize| if j < own { j } else { 0 };
    let xs: Vec<CudaCiphertext<'_>> = (0..jobs).map(|j| up_x[src(j)].clone()).collect();
    let ds: Vec<Diagonals<CudaCiphertext<'_>>> = (0..jobs).map(|j| up_d[src(j)].clone()).collect();
    ev.reset_counter();
    let got = ev.matvec_enc_batch(&ds, &xs).unwrap();
    let batch_counts = ev.counter();
    ev.reset_counter();
    let single = ev.matvec_enc(&ds[0], &xs[0]).unwrap();
    let mut want_counts = hekit_ckks::OpCounter::default();
    for _ in 0..jobs {
        want_counts += &ev.counter();
    }
    assert_eq!(batch_counts, want_counts, "{test}: batch counts");
    assert_eq!(got.len(), jobs);
    let single = single.download().unwrap();
    for (j, g) in got.iter().enumerate() {
        let want = cpu.matvec_enc(&host_d[src(j)], &host_x[src(j)]).unwrap();
        let g = g.download().unwrap();
        assert_same_ct(&g, &want, &format!("{test}: job {j} vs CPU"));
        if src(j) == 0 {
            assert_same_ct(&g, &single, &format!("{test}: job {j} vs matvec_enc"));
        }
    }
    // The device keeps working on stream 0 afterwards.
    let again = ev.matvec_enc(&ds[1], &xs[1]).unwrap().download().unwrap();
    assert_same_ct(&again, &got[1].download().unwrap(), "after the batch");
    eprintln!("{test}: {jobs} jobs on {streams} streams, counts {batch_counts:?}");
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn matvec_enc_batch_matches_individual_seal_2020() {
    let p = CkksParams::seal_2020();
    let l = p.max_level();
    batch(
        "matvec_enc_batch_matches_individual_seal_2020",
        p,
        16,
        l,
        4,
        4770,
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn matvec_enc_batch_matches_individual_n16_train_capped() {
    batch(
        "matvec_enc_batch_matches_individual_n16_train_capped",
        CkksParams::n16_train(),
        4,
        9,
        4,
        4780,
    );
}

/// The brief's simulated OOM: a device with so many streams that the batch's per-stream
/// pre-allocation exceeds the device's total memory. `matvec_enc_batch` returns `Device`
/// (no abort), counts nothing, and the device and evaluator keep working. (The pre-check
/// against total memory refuses before allocating, so the test never takes the GPU's free
/// memory from other tenants; an allocation failure itself is `Device` too, see Task 46's
/// `ks_errors_sharing_and_oom`.)
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn matvec_enc_batch_oom_is_a_device_error() {
    let test = "matvec_enc_batch_oom_is_a_device_error";
    let p = CkksParams::n16_train();
    let (d, level) = (4, 9);
    let steps = bsgs_rotation_steps(d, 1);
    let n = need(&p, steps.len() + 1, level, level, 2 * d + 8, d, 1);
    let Some(probe) = open(test, &p, n, 1) else {
        return;
    };
    let (_free, total) = probe.mem_info().unwrap();
    drop(probe);
    let ws = 8 * p.n() * matvec_enc_workspace_limbs(&p, level, d);
    let streams = total / ws + 8;
    eprintln!(
        "{test}: {streams} streams x {:.1} MiB > {:.1} MiB total (free before: {:.1} MiB)",
        ws as f64 / MIB as f64,
        total as f64 / MIB as f64,
        free_device_memory(0).unwrap() as f64 / MIB as f64
    );
    let dev = CudaDevice::new(0, &p, streams).unwrap();
    let fx = EvalFixture::new(p, d, 4790);
    let ev = evaluator(&dev, &fx, Some(&steps), level, EvalConfig::default());
    let x = ev.upload_ct(&fx.encrypt(level, 1)).unwrap();
    let dg = fx
        .diagonals(level, 2)
        .1
        .try_map(|c| ev.upload_ct(c))
        .unwrap();
    let xs = vec![x.clone(); streams];
    let ds = vec![dg.clone(); streams];
    let r = ev.matvec_enc_batch(&ds, &xs);
    assert!(matches!(r, Err(HekitError::Device(_))), "{:?}", r.err());
    if let Err(e) = r {
        eprintln!("{test}: {e}");
    }
    assert_eq!(ev.counter(), hekit_ckks::OpCounter::default());
    // Still usable: a single product and a 2-job batch match the CPU.
    let cpu = fx.cpu(EvalConfig::default());
    let want = cpu
        .matvec_enc(&fx.diagonals(level, 2).1, &fx.encrypt(level, 1))
        .unwrap();
    assert_same_ct(
        &ev.matvec_enc(&dg, &x).unwrap().download().unwrap(),
        &want,
        "after OOM",
    );
    let two = ev.matvec_enc_batch(&ds[..2], &xs[..2]).unwrap();
    for c in &two {
        assert_same_ct(&c.download().unwrap(), &want, "2-job batch after OOM");
    }
    // Bad stream index and length mismatch are InvalidParams.
    assert!(matches!(
        dev.use_stream(streams),
        Err(HekitError::InvalidParams(_))
    ));
    assert!(matches!(
        ev.matvec_enc_batch(&ds[..2], &xs[..1]),
        Err(HekitError::InvalidParams(_))
    ));
}

/// Constructor checks: parameters that are not the device's, a key level above `L`.
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn evaluator_constructor_errors() {
    let p = CkksParams::test_tiny();
    let Some(dev) = open("evaluator_constructor_errors", &p, 64 * MIB, 1) else {
        return;
    };
    let fx = EvalFixture::new(p.clone(), 2, 4799);
    // Another Arc with equal values but separately built bases.
    let other = Arc::new(CkksParams::test_tiny());
    let keys: Vec<&GaloisKey> = fx.galois_keys();
    let r = CudaEvaluator::with_keys(&dev, other, Some(&fx.rk), &keys, EvalConfig::default(), 1);
    assert!(matches!(r, Err(HekitError::InvalidParams(_))));
    // `dev` was built from `p`, whose clone `fx.params` shares its bases.
    let r = CudaEvaluator::with_keys(
        &dev,
        Arc::clone(&fx.params),
        Some(&fx.rk),
        &keys,
        EvalConfig::default(),
        fx.params.max_level() + 1,
    );
    assert!(matches!(r, Err(HekitError::InvalidParams(_))));
    let ev = CudaEvaluator::new(
        &dev,
        Arc::clone(&fx.params),
        &fx.rk,
        &fx.gks,
        EvalConfig::default(),
    )
    .unwrap();
    assert_eq!(ev.num_galois_keys(), fx.num_galois_keys());
    assert_eq!(ev.key_level(), fx.params.max_level());
}
