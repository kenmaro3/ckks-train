//! GPU tests of hybrid key switching, relinearization, rotations, hoisted rotations and
//! rescaling on device ciphertexts (Task 46).
//!
//! The checks are the shared cases of `tests/cases/ks.rs`; the CPU replay of the kernels
//! (`src/emu_ks.rs`, `replay_*` tests) runs the same checks with the same fixtures on every
//! machine, except the two n16_train 16-step tests (16 n16_train keys are too heavy for the
//! light test run; the replay covers the same launch sequence with 2 steps on n16_train and 16
//! steps on seal_2020). Every result is compared with the CPU **bit for bit**.
//!
//! Each test is `#[ignore]`d, declares its device-memory need (tables + keys + working set,
//! see [`need`]) and skips (prints `SKIP <test>: needs .. MiB, only .. MiB free`, passes)
//! when there is no usable GPU or not enough free memory. The need is checked **before** the
//! keys are generated on the CPU, so a skip is immediate. Run on a CUDA machine with
//! `cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1`.
#![cfg(feature = "cuda")]

mod cases;

use std::sync::OnceLock;

use cases::ks::{
    KeyChoice, KsExec, KsFixture, check_hoisted_rotate, check_key_switch, check_mul_relin_rescale,
    check_rotate,
};
use hekit_ckks::keyswitch::digit_ranges;
use hekit_ckks::{Ciphertext, CkksParams, KeyGenerator};
use hekit_core::sampler::Rng;
use hekit_core::{HekitError, RnsPoly};
use hekit_cuda::testing::{ksk_bytes, require_gpu_mem, table_bytes, try_alloc_limbs};
use hekit_cuda::{CudaCiphertext, CudaDevice, DeviceKsk, KskPurpose};

const MIB: usize = 1 << 20;

/// Device-memory need of a test: the parameter set's NTT tables, `keys` key-switching keys
/// uploaded for levels `<= key_level`, and a working set at `key_level`:
/// * 12 ciphertext polynomials over `Q_l` (inputs, the tensor product, outputs and
///   temporaries) plus the `hoisted` results of a hoisted rotation (2 polynomials each, all
///   alive until the call returns);
/// * `digit_sets` sets of lifted digits over `Q_l * P` (the decomposition, plus the rotated
///   digits of a hoisted rotation) and 4 more `Q_l * P` polynomials (inner-product
///   accumulators and `mod_up` / `mod_down` temporaries);
/// * 8 MiB for the cached limb maps, conversion tables and digit-address arrays and 32 MiB
///   of slack for the stream-ordered allocator.
fn need(
    params: &CkksParams,
    keys: usize,
    key_level: usize,
    digit_sets: usize,
    hoisted: usize,
) -> usize {
    let n8 = 8 * params.n();
    let q = n8 * (key_level + 1);
    let qp = n8 * (key_level + 1 + params.p().len());
    let digits = digit_ranges(params, key_level).len();
    table_bytes(params)
        + keys * ksk_bytes(params, key_level).unwrap()
        + (12 + 2 * hoisted) * q
        + (digit_sets * digits + 4) * qp
        + 40 * MIB
}

/// [`KsExec`] on the GPU: the fixture's keys uploaded once (for levels `<= key_level`), then
/// per call upload, device operations, download.
struct GpuKs<'d> {
    dev: &'d CudaDevice,
    relin: Option<DeviceKsk<'d>>,
    galois: Vec<DeviceKsk<'d>>,
}

impl<'d> GpuKs<'d> {
    /// Uploads the relinearization key (`relin`) and the first `galois` Galois keys of `fx`.
    fn new(dev: &'d CudaDevice, fx: &KsFixture, relin: bool, galois: usize) -> GpuKs<'d> {
        let l = fx.key_level;
        let relin = relin.then(|| {
            let rk = fx.relin_key().expect("fixture has a relinearization key");
            dev.upload_relin_key_with(rk, l).unwrap()
        });
        let galois = fx
            .galois_keys()
            .into_iter()
            .take(galois)
            .map(|gk| {
                let k = dev.upload_galois_key_with(gk, l).unwrap();
                assert_eq!(k.device_bytes(), ksk_bytes(&fx.params, l).unwrap());
                k
            })
            .collect();
        GpuKs { dev, relin, galois }
    }

    fn keys(&self) -> Vec<&DeviceKsk<'d>> {
        self.galois.iter().collect()
    }

    fn up(&self, ct: &Ciphertext) -> CudaCiphertext<'d> {
        self.dev.upload_ct(ct).unwrap()
    }
}

impl KsExec for GpuKs<'_> {
    /// Both `key_switch` and `apply_key(decompose(..))`, which must agree.
    fn key_switch(&self, d: &RnsPoly, key: KeyChoice) -> (RnsPoly, RnsPoly) {
        let k = match key {
            KeyChoice::Relin => self.relin.as_ref().unwrap(),
            KeyChoice::Galois(g) => self
                .galois
                .iter()
                .find(|k| k.galois_elt() == Some(g))
                .unwrap(),
        };
        let x = self.dev.upload(d).unwrap();
        let (kb, ka) = self.dev.key_switch(&x, k).unwrap();
        let digits = self.dev.decompose(&x).unwrap();
        assert_eq!(digits.len(), self.dev.num_digits(d.level()));
        let (kb2, ka2) = self.dev.apply_key(&digits, k).unwrap();
        let (kb, ka) = (kb.download().unwrap(), ka.download().unwrap());
        assert_eq!(
            kb2.download().unwrap().limbs(),
            kb.limbs(),
            "apply_key(decompose) kb"
        );
        assert_eq!(
            ka2.download().unwrap().limbs(),
            ka.limbs(),
            "apply_key(decompose) ka"
        );
        (kb, ka)
    }

    fn mul_relin_rescale(&self, a: &Ciphertext, b: &Ciphertext) -> [Ciphertext; 3] {
        let d = self.dev;
        let t = d.mul_no_relin(&self.up(a), &self.up(b)).unwrap();
        let r = d.relinearize(&t, self.relin.as_ref().unwrap()).unwrap();
        let s = d.rescale(&r).unwrap();
        [
            t.download().unwrap(),
            r.download().unwrap(),
            s.download().unwrap(),
        ]
    }

    fn rotate(&self, ct: &Ciphertext, steps: i32) -> Ciphertext {
        let r = self.dev.rotate(&self.up(ct), steps, &self.keys()).unwrap();
        r.download().unwrap()
    }

    fn conjugate(&self, ct: &Ciphertext) -> Ciphertext {
        let r = self.dev.conjugate(&self.up(ct), &self.keys()).unwrap();
        r.download().unwrap()
    }

    fn hoisted_rotate(&self, ct: &Ciphertext, steps: &[i32]) -> Vec<Ciphertext> {
        let rs = self
            .dev
            .hoisted_rotate(&self.up(ct), steps, &self.keys())
            .unwrap();
        rs.iter().map(|r| r.download().unwrap()).collect()
    }
}

// Fixtures (the same parameters, steps and seeds as the replay in src/emu_ks.rs). Built on
// first use, after the memory check of the test that needs them.

fn seal() -> &'static KsFixture {
    static F: OnceLock<KsFixture> = OnceLock::new();
    F.get_or_init(|| KsFixture::new(CkksParams::seal_2020(), true, &[1, -3], true, None, 46))
}

fn n16() -> &'static KsFixture {
    static F: OnceLock<KsFixture> = OnceLock::new();
    F.get_or_init(|| KsFixture::new(CkksParams::n16_train(), true, &[1, 5], false, None, 47))
}

fn sixteen(params: CkksParams, seed: u64) -> KsFixture {
    let steps: Vec<i32> = (1..=16).collect();
    KsFixture::new(params, false, &steps, false, None, seed)
}

fn seal_16() -> &'static KsFixture {
    static F: OnceLock<KsFixture> = OnceLock::new();
    F.get_or_init(|| sixteen(CkksParams::seal_2020(), 48))
}

fn n16_16() -> &'static KsFixture {
    static F: OnceLock<KsFixture> = OnceLock::new();
    F.get_or_init(|| sixteen(CkksParams::n16_train(), 49))
}

/// Opens the device for `params` if `need_bytes` fit, else prints SKIP and returns `None`.
fn open(test: &str, params: &CkksParams, need_bytes: usize) -> Option<CudaDevice> {
    require_gpu_mem(test, need_bytes, params)
}

/// `key_switch` and `apply_key(decompose(..))` against `hekit_ckks::keyswitch::key_switch`
/// with the relinearization key at levels L, L-1, L/2-2, 3, 1, 0 and a Galois key at L.
/// Need: tables + 2 keys + working set (seal_2020 230 MiB, n16_train 1096 MiB).
fn key_switch_test(test: &str, fx: fn() -> &'static KsFixture, params: CkksParams) {
    let l = params.max_level();
    let Some(dev) = open(test, &params, need(&params, 2, l, 1, 0)) else {
        return;
    };
    let fx = fx();
    check_key_switch(&GpuKs::new(&dev, fx, true, 1), fx);
    dev.sync().unwrap();
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn key_switch_matches_cpu_seal_2020() {
    key_switch_test(
        "key_switch_matches_cpu_seal_2020",
        seal,
        CkksParams::seal_2020(),
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn key_switch_matches_cpu_n16_train() {
    key_switch_test(
        "key_switch_matches_cpu_n16_train",
        n16,
        CkksParams::n16_train(),
    );
}

/// `relinearize(mul_no_relin(a, b))` and `rescale` of it, every stage against the CPU, at
/// levels L, L/2, 1. Need: tables + relinearization key + working set (seal_2020 175 MiB,
/// n16_train 808 MiB).
fn mul_test(test: &str, fx: fn() -> &'static KsFixture, params: CkksParams) {
    let l = params.max_level();
    let Some(dev) = open(test, &params, need(&params, 1, l, 1, 0)) else {
        return;
    };
    let fx = fx();
    check_mul_relin_rescale(&GpuKs::new(&dev, fx, true, 0), fx);
    dev.sync().unwrap();
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn mul_relin_rescale_matches_cpu_seal_2020() {
    mul_test(
        "mul_relin_rescale_matches_cpu_seal_2020",
        seal,
        CkksParams::seal_2020(),
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn mul_relin_rescale_matches_cpu_n16_train() {
    mul_test(
        "mul_relin_rescale_matches_cpu_n16_train",
        n16,
        CkksParams::n16_train(),
    );
}

/// `rotate` by the fixture's two steps, 0 and `slots`, `conjugate` (seal_2020), and
/// `hoisted_rotate(ct, [s1, s2, 0, s1])` against the CPU's individual rotations, at levels
/// L, L-1, L/2-2, 3, 1, 0 (hoisted: L, 1). Need: tables + the Galois keys (seal_2020 3, with
/// conjugation: 327 MiB in all; n16_train 2: 1336 MiB) + a working set with two digit
/// sets and 3 hoisted results.
fn rotate_test(test: &str, fx: fn() -> &'static KsFixture, params: CkksParams, keys: usize) {
    let l = params.max_level();
    let Some(dev) = open(test, &params, need(&params, keys, l, 2, 3)) else {
        return;
    };
    let fx = fx();
    let e = GpuKs::new(&dev, fx, false, keys);
    check_rotate(&e, fx);
    check_hoisted_rotate(&e, fx, &[l, 1]);
    dev.sync().unwrap();
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn rotate_matches_cpu_seal_2020() {
    rotate_test(
        "rotate_matches_cpu_seal_2020",
        seal,
        CkksParams::seal_2020(),
        3,
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn rotate_matches_cpu_n16_train() {
    rotate_test(
        "rotate_matches_cpu_n16_train",
        n16,
        CkksParams::n16_train(),
        2,
    );
}

/// `hoisted_rotate` by the 16 steps 1..=16 (plus 0 and a repeat) against the CPU's
/// individual rotations, with all 16 Galois keys on the device for levels `<= key_level`, at
/// `levels`.
fn hoisted_16_test(
    test: &str,
    fx: fn() -> &'static KsFixture,
    params: CkksParams,
    key_level: usize,
    levels: &[usize],
) {
    let Some(dev) = open(test, &params, need(&params, 16, key_level, 2, 17)) else {
        return;
    };
    let full = fx();
    let capped;
    let fx = if key_level == params.max_level() {
        full
    } else {
        capped = full.capped(key_level);
        &capped
    };
    let e = GpuKs::new(&dev, fx, false, 16);
    assert_eq!(e.galois.len(), 16);
    if key_level < params.max_level() {
        // The capped layout: the first Galois key at the key level and below (key_skip > 0).
        check_key_switch(&e, fx);
    }
    check_hoisted_rotate(&e, fx, levels);
    dev.sync().unwrap();
}

/// seal_2020, full keys (16 x 55 MiB): need 1112 MiB.
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn hoisted_rotate_16_matches_cpu_seal_2020() {
    let p = CkksParams::seal_2020();
    let l = p.max_level();
    hoisted_16_test(
        "hoisted_rotate_16_matches_cpu_seal_2020",
        seal_16,
        p,
        l,
        &[l, 1],
    );
}

/// n16_train, full keys: 16 x 288 MiB, need 5816 MiB (skips when the GPU has less free).
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn hoisted_rotate_16_matches_cpu_n16_train() {
    let p = CkksParams::n16_train();
    let l = p.max_level();
    hoisted_16_test(
        "hoisted_rotate_16_matches_cpu_n16_train",
        n16_16,
        p,
        l,
        &[l],
    );
}

/// n16_train, the same 16 keys uploaded for levels <= 9 only (16 x 42 MiB, 3 key parts):
/// hoisted rotations at levels 9 (digits 4 + 4 + 2 clipped, key_skip 0), 5 (4 + 2 clipped,
/// key_skip 4) and 1 (one 2-prime digit, key_skip 8), and key switching with the first Galois
/// key at 9, 5, 3, 1 and 0. Need 1084 MiB.
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn hoisted_rotate_16_low_level_matches_cpu_n16_train() {
    hoisted_16_test(
        "hoisted_rotate_16_low_level_matches_cpu_n16_train",
        n16_16,
        CkksParams::n16_train(),
        9,
        &[9, 5, 1],
    );
}

fn is_invalid<T: std::fmt::Debug>(r: hekit_core::Result<T>, what: &str) {
    assert!(
        matches!(r, Err(HekitError::InvalidParams(_))),
        "{what}: expected InvalidParams, got {r:?}"
    );
}

/// Error kinds (as the CPU evaluator's), shared storage, copy on write and out-of-memory, on
/// test_tiny (N = 32): need 4 MiB.
///
/// * `InvalidParams`: wrong component counts, a Galois key for relinearization, a missing
///   Galois key, a level above a capped key's, a wrong digit count for `apply_key`, a key of
///   another parameter set, a ciphertext or key of another device; `LevelMismatch`;
///   `NoLevelsLeft` for `mul_no_relin` and `rescale` at level 0.
/// * `clone` shares the device storage (no allocation); `ntt` on a shared polynomial copies on
///   write and leaves the other handle unchanged.
/// * An allocation larger than the whole device (through the allocation path of every
///   operation and of `upload_ksk`) is a `Device` error, not an abort; the device stays usable.
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn ks_errors_sharing_and_oom() {
    let params = CkksParams::test_tiny();
    let Some(dev) = require_gpu_mem("ks_errors_sharing_and_oom", 4 * MIB, &params) else {
        return;
    };
    let fx = KsFixture::new(params.clone(), true, &[1], false, None, 50);
    let l = params.max_level();
    let rk = dev.upload_relin_key(fx.relin_key().unwrap()).unwrap();
    let gk = dev.upload_galois_key(fx.galois_keys()[0]).unwrap();
    let low = dev
        .upload_relin_key_with(fx.relin_key().unwrap(), 2)
        .unwrap();
    assert_eq!(low.purpose(), KskPurpose::Relin);
    assert_eq!(gk.galois_elt(), Some(fx.galois_keys()[0].elt()));
    assert_eq!((rk.max_level(), low.max_level()), (l, 2));
    let a = dev.upload_ct(&fx.encrypt(l, 1)).unwrap();
    let b = dev.upload_ct(&fx.encrypt(l - 1, 2)).unwrap();
    let z = dev.upload_ct(&fx.encrypt(0, 3)).unwrap();

    let t = dev.mul_no_relin(&a, &a).unwrap();
    is_invalid(dev.mul_no_relin(&t, &a), "mul_no_relin of 3 components");
    assert_eq!(
        dev.mul_no_relin(&a, &b).unwrap_err(),
        HekitError::LevelMismatch
    );
    assert_eq!(
        dev.mul_no_relin(&z, &z).unwrap_err(),
        HekitError::NoLevelsLeft
    );
    assert_eq!(dev.rescale(&z).unwrap_err(), HekitError::NoLevelsLeft);
    is_invalid(dev.relinearize(&a, &rk), "relinearize of 2 components");
    is_invalid(dev.relinearize(&t, &gk), "relinearize with a Galois key");
    is_invalid(
        dev.relinearize(&t, &low),
        "relinearize above the key's level",
    );
    is_invalid(dev.rotate(&t, 1, &[&gk]), "rotate of 3 components");
    is_invalid(dev.rotate(&a, 2, &[&gk]), "rotate without its key");
    is_invalid(dev.conjugate(&a, &[&gk]), "conjugate without its key");
    is_invalid(
        dev.hoisted_rotate(&a, &[1, 2], &[&gk]),
        "hoisted_rotate with a missing key",
    );
    assert!(dev.rotate(&a, 0, &[]).is_ok()); // a copy needs no key
    let mut digits = dev.decompose(&a.polys()[1]).unwrap();
    digits.pop();
    is_invalid(
        dev.apply_key(&digits, &rk),
        "apply_key with a missing digit",
    );
    is_invalid(dev.apply_key(&[], &rk), "apply_key without digits");
    // A key of another parameter set (dnum and primes differ).
    let foreign_params = CkksParams::new(5, &[54, 42, 34], &[60], 42, 3).unwrap();
    let mut kg = KeyGenerator::new(
        std::sync::Arc::new(foreign_params.clone()),
        Rng::from_seed([8; 32]),
    )
    .unwrap();
    is_invalid(
        dev.upload_ksk(kg.relin_key().key()),
        "upload_ksk of another parameter set",
    );
    let dev2 = CudaDevice::new(0, &params, 1).unwrap();
    let a2 = dev2.upload_ct(&fx.encrypt(l, 4)).unwrap();
    is_invalid(dev.mul_no_relin(&a, &a2), "ciphertext of another device");
    let rk2 = dev2.upload_relin_key(fx.relin_key().unwrap()).unwrap();
    is_invalid(dev.relinearize(&t, &rk2), "key of another device");

    // Shared storage and copy on write.
    let host = a.download().unwrap();
    let shared = a.clone();
    assert!(a.polys()[0].is_shared() && shared.polys()[1].is_shared());
    let mut p = shared.polys()[0].clone();
    dev.intt(&mut p).unwrap();
    assert!(!p.is_ntt_form());
    assert_same_download(&a, &host, "original after intt of a clone");
    let mut back = p.clone();
    dev.ntt(&mut back).unwrap();
    assert_eq!(back.download().unwrap().limbs(), host.polys()[0].limbs());
    let relabelled = a.with_scale(2.0 * a.scale()).unwrap();
    assert!(relabelled.polys()[0].is_shared());

    // Out of memory: more than the device's total memory.
    let (_, total) = dev.mem_info().unwrap();
    let limbs = total / (8 * params.n()) + 1;
    let e = try_alloc_limbs(&dev, limbs).unwrap_err();
    assert!(
        matches!(e, HekitError::Device(_)),
        "oversized allocation: {e:?}"
    );
    eprintln!("ks_errors_sharing_and_oom: oversized allocation -> {e}");
    // The device is still usable.
    let s = dev.rescale(&a).unwrap();
    assert_eq!(s.level(), l - 1);
    dev.sync().unwrap();
}

fn assert_same_download(ct: &CudaCiphertext<'_>, host: &Ciphertext, what: &str) {
    let got = ct.download().unwrap();
    for (g, h) in got.polys().iter().zip(host.polys()) {
        assert_eq!(g.limbs(), h.limbs(), "{what}");
    }
}
