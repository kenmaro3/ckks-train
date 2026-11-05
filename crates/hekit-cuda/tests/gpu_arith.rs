//! GPU tests of the pointwise, automorphism, rescale and base-conversion kernels (Task 45).
//!
//! The checks are the shared cases of `tests/cases/mod.rs`; the CPU replay of the kernels
//! (`src/emu_arith.rs`, `replay_*` tests) runs the same checks with the same inputs on every
//! machine. Each test here is `#[ignore]`d, declares its device-memory need and skips (prints
//! `SKIP <test>: ...`, passes) when there is no usable GPU or not enough free memory. Run on a
//! CUDA machine with
//! `cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1`.
//!
//! Memory: every test declares the parameter set's tables plus four `Q * P` polynomials
//! (the most any operation keeps alive at once is the input, two temporaries and the output
//! of `mod_up`: 32 + 4 + 32 + 36 limbs for n16_train, under 4 x 36) plus 1 MiB for the cached
//! limb maps and conversion tables: n16_train 72 + 4 x 18 + 1 = 145 MiB, seal_2020 11 + 4 x
//! 2.75 + 1 = 23 MiB.
#![cfg(feature = "cuda")]

mod cases;

use cases::Exec;
use hekit_ckks::{CkksParams, QpPoly};
use hekit_core::{HekitError, RnsPoly};
use hekit_cuda::testing::{require_gpu_mem, table_bytes};
use hekit_cuda::{CudaDevice, DevicePoly};

/// [`Exec`] on the GPU: upload, one device operation, download.
struct Gpu<'a>(&'a CudaDevice);

impl Gpu<'_> {
    fn up(&self, p: &RnsPoly) -> DevicePoly<'_> {
        self.0.upload(p).unwrap()
    }

    fn up_qp(&self, x: &QpPoly) -> DevicePoly<'_> {
        self.0.upload_qp(x).unwrap()
    }
}

impl Exec for Gpu<'_> {
    fn add(&self, a: &RnsPoly, b: &RnsPoly) -> RnsPoly {
        let d = self.0;
        d.add(&self.up(a), &self.up(b)).unwrap().download().unwrap()
    }

    fn sub(&self, a: &RnsPoly, b: &RnsPoly) -> RnsPoly {
        let d = self.0;
        d.sub(&self.up(a), &self.up(b)).unwrap().download().unwrap()
    }

    fn mul_pointwise(&self, a: &RnsPoly, b: &RnsPoly) -> RnsPoly {
        let d = self.0;
        let r = d.mul_pointwise(&self.up(a), &self.up(b)).unwrap();
        r.download().unwrap()
    }

    fn neg(&self, a: &RnsPoly) -> RnsPoly {
        self.0.neg(&self.up(a)).unwrap().download().unwrap()
    }

    fn mul_scalar(&self, a: &RnsPoly, scalar: u64) -> RnsPoly {
        let r = self.0.mul_scalar(&self.up(a), scalar).unwrap();
        r.download().unwrap()
    }

    fn add_qp(&self, a: &QpPoly, b: &QpPoly) -> QpPoly {
        let d = self.0;
        let r = d.add(&self.up_qp(a), &self.up_qp(b)).unwrap();
        r.download_qp().unwrap()
    }

    fn ntt_qp(&self, x: &QpPoly) -> QpPoly {
        let mut d = self.up_qp(x);
        if d.is_ntt_form() {
            self.0.intt(&mut d).unwrap();
        } else {
            self.0.ntt(&mut d).unwrap();
        }
        d.download_qp().unwrap()
    }

    fn automorphism(&self, a: &RnsPoly, galois_elt: u64) -> RnsPoly {
        let r = self.0.automorphism(&self.up(a), galois_elt).unwrap();
        r.download().unwrap()
    }

    fn rescale(&self, a: &RnsPoly) -> RnsPoly {
        self.0
            .rescale_poly(&self.up(a))
            .unwrap()
            .download()
            .unwrap()
    }

    fn mod_down(&self, x: &QpPoly) -> RnsPoly {
        self.0.mod_down(&self.up_qp(x)).unwrap().download().unwrap()
    }

    fn mod_up(&self, x: &RnsPoly, digit: usize) -> QpPoly {
        let r = self.0.mod_up(&self.up(x), digit).unwrap();
        r.download_qp().unwrap()
    }
}

/// Opens the device for `params` with the declared need (module docs).
fn open(test: &str, params: &CkksParams) -> Option<CudaDevice> {
    let limbs = params.q().len() + params.p().len();
    let need = table_bytes(params) + 4 * 8 * params.n() * limbs + (1 << 20);
    require_gpu_mem(test, need, params)
}

macro_rules! gpu_tests {
    ($($name:ident: $params:ident => $check:ident;)*) => {$(
        #[test]
        #[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
        fn $name() {
            let params = CkksParams::$params();
            let Some(dev) = open(stringify!($name), &params) else {
                return;
            };
            cases::$check(&Gpu(&dev), &params);
            dev.sync().unwrap();
        }
    )*};
}

gpu_tests! {
    pointwise_matches_cpu_seal_2020: seal_2020 => check_pointwise;
    pointwise_matches_cpu_n16_train: n16_train => check_pointwise;
    qp_layout_matches_cpu_seal_2020: seal_2020 => check_qp_layout;
    qp_layout_matches_cpu_n16_train: n16_train => check_qp_layout;
    automorphism_matches_cpu_seal_2020: seal_2020 => check_automorphism;
    automorphism_matches_cpu_n16_train: n16_train => check_automorphism;
    rescale_matches_cpu_seal_2020: seal_2020 => check_rescale;
    rescale_matches_cpu_n16_train: n16_train => check_rescale;
    mod_down_matches_cpu_seal_2020: seal_2020 => check_mod_down;
    mod_down_matches_cpu_n16_train: n16_train => check_mod_down;
    mod_up_matches_cpu_seal_2020: seal_2020 => check_mod_up;
    mod_up_matches_cpu_n16_train: n16_train => check_mod_up;
    mod_down_near_ties_match_cpu_seal_2020: seal_2020 => check_mod_down_near_ties;
    mod_down_near_ties_match_cpu_n16_train: n16_train => check_mod_down_near_ties;
    mod_up_near_ties_match_cpu_seal_2020: seal_2020 => check_mod_up_near_ties;
    mod_up_near_ties_match_cpu_n16_train: n16_train => check_mod_up_near_ties;
}

fn is_invalid<T: std::fmt::Debug>(r: hekit_core::Result<T>, what: &str) {
    assert!(
        matches!(r, Err(HekitError::InvalidParams(_))),
        "{what}: expected InvalidParams, got {r:?}"
    );
}

/// Shape errors are `InvalidParams` (no launch): downloads check the prime layout (a
/// `Q_l * P` polynomial needs `download_qp`, a `Q` polynomial is not one), mismatched
/// operands, forms, Galois elements, levels, digits, and a polynomial of another device.
///
/// GPU memory need: two test_tiny devices (tables well under 1 MiB) and a few tiny
/// polynomials: declared 4 MiB.
#[test]
#[ignore = "needs an NVIDIA GPU (run on quadruped-gpu)"]
fn shape_errors_are_invalid_params() {
    let params = CkksParams::test_tiny();
    let Some(dev) = require_gpu_mem("shape_errors_are_invalid_params", 4 << 20, &params) else {
        return;
    };
    let (l, n) = (params.max_level(), params.n());
    let q = dev
        .upload(&cases::random_poly(params.basis_q(), l, true, 1))
        .unwrap();
    let q0 = dev
        .upload(&cases::random_poly(params.basis_q(), 0, true, 2))
        .unwrap();
    let qc = dev
        .upload(&cases::random_poly(params.basis_q(), l, false, 3))
        .unwrap();
    let qp = dev
        .upload_qp(&cases::random_qp(&params, 0, true, 4))
        .unwrap();
    assert_eq!(qp.primes()[0], 0);
    assert_eq!(qp.primes()[1] as usize, params.q().len());

    is_invalid(qp.download(), "download of Q_l * P");
    is_invalid(q.download_qp(), "download_qp of a Q polynomial");
    is_invalid(dev.add(&q, &q0), "add at different levels");
    is_invalid(dev.add(&q, &qc), "add in different forms");
    is_invalid(
        dev.mul_pointwise(&qc, &qc),
        "mul_pointwise in coefficient form",
    );
    is_invalid(dev.automorphism(&q, 2), "even Galois element");
    is_invalid(
        dev.automorphism(&q, 2 * n as u64 + 1),
        "Galois element >= 2N",
    );
    is_invalid(dev.rescale_poly(&q0), "rescale at level 0");
    is_invalid(dev.rescale_poly(&qp), "rescale of a Q_l * P polynomial");
    is_invalid(dev.mod_down(&q), "mod_down without P limbs");
    is_invalid(dev.mod_up(&qc, 0), "mod_up in coefficient form");
    is_invalid(dev.mod_up(&qp, 0), "mod_up of a Q_l * P polynomial");
    is_invalid(dev.mod_up(&q, params.dnum()), "mod_up of a missing digit");

    let other = CudaDevice::new(0, &params, 1).unwrap();
    let foreign = other
        .upload(&cases::random_poly(params.basis_q(), l, true, 5))
        .unwrap();
    is_invalid(dev.neg(&foreign), "polynomial of another device");
    dev.sync().unwrap();
}
