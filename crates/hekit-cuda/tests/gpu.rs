//! GPU tests of the device context, device polynomials and batched NTT (Task 44).
//!
//! All GPU tests are `#[ignore]`d; run them on a CUDA machine with
//! `cargo test -p hekit-cuda --features cuda --release -- --ignored`. Each test states its
//! device-memory need and skips (prints `SKIP <test>: ...`, passes) when there is no usable
//! GPU or not enough free memory (see `hekit_cuda::testing::require_gpu_mem`).
#![cfg(feature = "cuda")]

use std::sync::Arc;

use hekit_ckks::{CkksParams, Plaintext};
use hekit_core::{HekitError, RnsBasis, RnsPoly};
use hekit_cuda::testing::{require_gpu_mem, table_bytes};
use hekit_cuda::{CudaDevice, probe};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

/// Bytes of one polynomial with `limbs` limbs of degree `n`.
fn poly_bytes(n: usize, limbs: usize) -> usize {
    8 * n * limbs
}

/// Random polynomial at `level` over `basis` (residues uniform mod each prime).
fn random_poly(basis: &Arc<RnsBasis>, level: usize, ntt_form: bool, seed: u64) -> RnsPoly {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let limbs = basis.moduli()[..=level]
        .iter()
        .map(|m| (0..basis.n()).map(|_| rng.next_u64() % m.value()).collect())
        .collect();
    RnsPoly::from_limbs(basis, limbs, ntt_form).unwrap()
}

/// `download(ntt(upload(p)))` equals the CPU `to_ntt` (i.e. `NttTable::forward` per limb) bit
/// for bit, and `download(intt(upload(f)))` equals the CPU `to_coeff`.
fn check_ntt(dev: &CudaDevice, basis: &Arc<RnsBasis>, level: usize, seed: u64) {
    let p = random_poly(basis, level, false, seed);
    let mut expect = p.clone();
    expect.to_ntt();
    let mut d = dev.upload(&p).unwrap();
    assert!(!d.is_ntt_form());
    assert_eq!(d.num_limbs(), level + 1);
    dev.ntt(&mut d).unwrap();
    assert!(d.is_ntt_form());
    let got = d.download().unwrap();
    assert!(got.is_ntt_form());
    assert_eq!(
        got.limbs(),
        expect.limbs(),
        "GPU NTT != CPU NttTable::forward (N = {}, level {level})",
        basis.n()
    );

    let f = random_poly(basis, level, true, seed + 1);
    let mut expect = f.clone();
    expect.to_coeff();
    let mut d = dev.upload(&f).unwrap();
    dev.intt(&mut d).unwrap();
    let got = d.download().unwrap();
    assert!(!got.is_ntt_form());
    assert_eq!(
        got.limbs(),
        expect.limbs(),
        "GPU inverse NTT != CPU NttTable::inverse (N = {}, level {level})",
        basis.n()
    );
}

/// n16_train, all 32 Q limbs (the correctness gate): GPU forward NTT bit-identical to the CPU.
/// Also level 5 (6 limbs) and the 4 P limbs.
///
/// GPU memory need: NTT tables 36 primes x 4 x 2^16 x 8 B = 72 MiB, plus one 32-limb
/// polynomial (16 MiB) at a time: declared 72 MiB + 2 x 16 MiB = 104 MiB.
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn ntt_matches_cpu_n16_train_32_limbs() {
    let params = CkksParams::n16_train();
    let need = table_bytes(&params) + 2 * poly_bytes(params.n(), 32);
    let Some(dev) = require_gpu_mem("ntt_matches_cpu_n16_train_32_limbs", need, &params) else {
        return;
    };
    assert_eq!(params.q().len(), 32);
    check_ntt(&dev, params.basis_q(), 31, 1);
    check_ntt(&dev, params.basis_q(), 5, 2);
    check_ntt(&dev, params.basis_p(), params.p().len() - 1, 3);
    dev.sync().unwrap();
}

/// n16_train, 32 limbs: `intt(ntt(p)) == p` on the device (no download in between), and the
/// no-op cases (ntt of an NTT-form polynomial, intt of a coefficient-form one).
///
/// GPU memory need: 72 MiB tables + one 32-limb polynomial (16 MiB): declared 104 MiB.
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn intt_ntt_roundtrip_n16_train() {
    let params = CkksParams::n16_train();
    let need = table_bytes(&params) + 2 * poly_bytes(params.n(), 32);
    let Some(dev) = require_gpu_mem("intt_ntt_roundtrip_n16_train", need, &params) else {
        return;
    };
    let p = random_poly(params.basis_q(), 31, false, 10);
    let mut d = dev.upload(&p).unwrap();
    dev.intt(&mut d).unwrap(); // no-op: already coefficient form
    dev.ntt(&mut d).unwrap();
    dev.ntt(&mut d).unwrap(); // no-op: already NTT form
    dev.intt(&mut d).unwrap();
    dev.sync().unwrap();
    let back = d.download().unwrap();
    assert!(!back.is_ntt_form());
    assert_eq!(back.limbs(), p.limbs(), "intt(ntt(p)) != p");
}

/// Small parameter sets (all-shared-memory path N = 2^5; N = 2^15 with four global stages),
/// every level of Q and P.
///
/// GPU memory need: seal_2020 tables 11 primes x 4 x 2^15 x 8 B = 11 MiB plus a 10-limb
/// polynomial (2.5 MiB): declared 16 MiB (test_tiny is negligible).
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn ntt_matches_cpu_small_params() {
    for (k, params) in [CkksParams::test_tiny(), CkksParams::seal_2020()]
        .into_iter()
        .enumerate()
    {
        let need = table_bytes(&params) + 2 * poly_bytes(params.n(), params.q().len());
        let Some(dev) = require_gpu_mem("ntt_matches_cpu_small_params", need, &params) else {
            return;
        };
        for level in 0..params.q().len() {
            check_ntt(&dev, params.basis_q(), level, 100 * k as u64 + level as u64);
        }
        for level in 0..params.p().len() {
            check_ntt(&dev, params.basis_p(), level, 1000 + level as u64);
        }
    }
}

/// `upload_pt` / `download` keep polynomial, scale and level.
///
/// GPU memory need: n16_train tables 72 MiB + one 11-limb polynomial (5.5 MiB): declared
/// 72 MiB + 2 x 5.5 MiB = 83 MiB.
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn upload_pt_roundtrip_n16_train() {
    let params = CkksParams::n16_train();
    let need = table_bytes(&params) + 2 * poly_bytes(params.n(), 11);
    let Some(dev) = require_gpu_mem("upload_pt_roundtrip_n16_train", need, &params) else {
        return;
    };
    let pt = Plaintext::new(random_poly(params.basis_q(), 10, true, 20), 2f64.powi(42)).unwrap();
    let dpt = dev.upload_pt(&pt).unwrap();
    assert_eq!(dpt.level(), 10);
    assert_eq!(dpt.scale(), pt.scale());
    assert!(dpt.poly().is_ntt_form());
    let back = dpt.download().unwrap();
    assert_eq!(back.level(), pt.level());
    assert_eq!(back.scale(), pt.scale());
    assert_eq!(back.poly().limbs(), pt.poly().limbs());
}

/// A non-existent ordinal is a `Device` error, not a panic or abort.
///
/// GPU memory need: none (no device is opened).
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn bad_ordinal_is_device_error() {
    let params = CkksParams::test_tiny();
    let r = CudaDevice::new(1000, &params, 1);
    assert!(
        matches!(r, Err(HekitError::Device(_))),
        "expected Device error, got {r:?}"
    );
    if let Err(e) = r {
        eprintln!("bad_ordinal_is_device_error: {e}");
    }
}

/// Without a driver, `CudaDevice::new` returns `Device` instead of panicking (runs everywhere;
/// on a machine with a GPU there is nothing to check). GPU memory need: none.
#[test]
fn missing_driver_is_device_error() {
    match probe() {
        Ok(n) => eprintln!(
            "missing_driver_is_device_error: {n} CUDA device(s) present, nothing to check"
        ),
        Err(e) => {
            assert!(matches!(e, HekitError::Device(_)));
            let r = CudaDevice::new(0, &CkksParams::test_tiny(), 1);
            assert!(
                matches!(r, Err(HekitError::Device(_))),
                "expected Device error, got {r:?}"
            );
        }
    }
}

/// `n_streams == 0` is rejected before touching the GPU. GPU memory need: none.
#[test]
fn zero_streams_is_invalid() {
    let r = CudaDevice::new(0, &CkksParams::test_tiny(), 0);
    assert!(matches!(r, Err(HekitError::InvalidParams(_))));
}
