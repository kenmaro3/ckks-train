//! Plain-Rust replay of `kernels/keyswitch.cu`, one simulated CUDA thread at a time (tests
//! only), the [`KsBackend`] of the replay (the kernels of `pointwise.cu` from
//! [`crate::emu_arith`], the inner product below), and [`EmuKs`], which runs the shared launch
//! sequences of [`crate::ks`] on it.
//!
//! Variable names and the `[K*]` tags match `keyswitch.cu`; launch shapes are [`ElemLaunch`],
//! as on the device. The tests run the shared cases of `tests/cases/ks.rs` (the checks, inputs
//! and CPU references of the GPU tests in `tests/gpu_ks.rs`) through [`EmuKs`].

use std::collections::HashMap;

use hekit_ckks::Ciphertext;
use hekit_ckks::keys::conjugation_elt;
use hekit_core::{HekitError, Result, RnsPoly};

use crate::basis::check_galois_elt;
use crate::cases::ks::{KeyChoice, KsExec, KsFixture};
use crate::emu_arith::{
    Binary, EmuBackend, add_mod, automorphism_coeff, automorphism_ntt, each_thread, limb, mul_mod,
    pw_binary,
};
use crate::ks::{
    InnerProduct, KeyRef, Ks, KsBackend, KskShape, PwOp, ksk_limbs, merge_hoisted, q_primes,
    rescaled_scale, rotation_elt,
};
use crate::plan::ElemLaunch;

/// `ks_inner_product` (the kernel's argument list).
#[allow(clippy::too_many_arguments)]
fn ks_inner_product(
    l: &ElemLaunch,
    out_b: &mut [u64],
    out_a: &mut [u64],
    digit_ptrs: &[&[u64]],
    key: &[u64],
    consts: &[u64],
    limb_prime: &[u32],
    log_n: u32,
    n_digits: u32,
    q_limbs: u32,
    key_limbs: u32,
    key_skip: u32,
) {
    each_thread(l, log_n, |y, c| {
        // [K1], [K2]
        let (i, cp) = limb(y, c, log_n, consts, limb_prime);
        let (q, r0, r1) = (cp[0], cp[5], cp[6]);
        let t = if y < q_limbs {
            y as usize
        } else {
            y as usize + key_skip as usize
        }; // [K3]
        let (mut acc_b, mut acc_a) = (0u64, 0u64);
        for j in 0..n_digits as usize {
            let x = digit_ptrs[j][i]; // [K4]
            let kb = key[(((2 * j) * key_limbs as usize + t) << log_n) + c as usize];
            let ka = key[(((2 * j + 1) * key_limbs as usize + t) << log_n) + c as usize];
            acc_b = add_mod(acc_b, mul_mod(x, kb, q, r0, r1), q); // [K5]
            acc_a = add_mod(acc_a, mul_mod(x, ka, q, r0, r1), q);
        }
        out_b[i] = acc_b; // [K6]
        out_a[i] = acc_a;
    });
}

impl KsBackend for EmuBackend {
    fn binary(
        &self,
        op: PwOp,
        out: &mut Vec<u64>,
        a: &Vec<u64>,
        b: &Vec<u64>,
        primes: &[u32],
    ) -> Result<()> {
        let n = primes.len() * self.n();
        assert!(out.len() == n && a.len() == n && b.len() == n);
        let op = match op {
            PwOp::Add => Binary::Add,
            PwOp::Sub => Binary::Sub,
            PwOp::Mul => Binary::Mul,
        };
        pw_binary(&self.pw(primes), op, out, a, b);
        Ok(())
    }

    fn automorphism(
        &self,
        out: &mut Vec<u64>,
        a: &Vec<u64>,
        primes: &[u32],
        ntt_form: bool,
        galois_elt: u64,
    ) -> Result<()> {
        check_galois_elt(self.n(), galois_elt)?;
        assert!(out.len() == primes.len() * self.n() && a.len() == out.len());
        if ntt_form {
            let l = ElemLaunch::new(self.n(), primes.len())?;
            automorphism_ntt(&l, out, a, self.log_n, galois_elt);
        } else {
            automorphism_coeff(&self.pw(primes), out, a, galois_elt);
        }
        Ok(())
    }

    fn inner_product(
        &self,
        out_b: &mut Vec<u64>,
        out_a: &mut Vec<u64>,
        digits: &[&Vec<u64>],
        key: &Vec<u64>,
        ip: &InnerProduct,
    ) -> Result<()> {
        let n = self.n();
        let rows = ip.out_primes.len();
        assert_eq!(digits.len(), ip.n_digits);
        assert!(digits.iter().all(|d| d.len() == rows * n));
        assert!(out_b.len() == rows * n && out_a.len() == rows * n);
        assert!(key.len() >= 2 * ip.n_digits * ip.key_limbs * n);
        let l = ElemLaunch::new(n, rows)?;
        let ptrs: Vec<&[u64]> = digits.iter().map(|d| d.as_slice()).collect();
        ks_inner_product(
            &l,
            out_b,
            out_a,
            &ptrs,
            key,
            &self.tables.consts,
            &ip.out_primes,
            self.log_n,
            ip.n_digits as u32,
            ip.q_limbs as u32,
            ip.key_limbs as u32,
            ip.key_skip as u32,
        );
        Ok(())
    }
}

/// A key flattened in device order ([`ksk_limbs`]).
struct EmuKey {
    buf: Vec<u64>,
    shape: KskShape,
}

impl EmuKey {
    fn r(&self) -> KeyRef<'_, Vec<u64>> {
        KeyRef {
            buf: &self.buf,
            shape: &self.shape,
        }
    }
}

/// [`KsExec`] on the replay: host values to flat limbs (as the uploads), the shared launch
/// sequences of [`Ks`] on [`EmuBackend`], flat limbs back to host values (as the downloads),
/// with the scale and level bookkeeping of the device's ciphertext operations.
pub struct EmuKs<'f> {
    fx: &'f KsFixture,
    b: EmuBackend,
    relin: Option<EmuKey>,
    galois: HashMap<u64, EmuKey>,
}

impl<'f> EmuKs<'f> {
    /// The replay with the fixture's keys "uploaded" for levels `<= fx.key_level`.
    pub fn new(fx: &'f KsFixture) -> EmuKs<'f> {
        let b = EmuBackend::new(&fx.params);
        let shape = KskShape::new(&fx.params, fx.key_level).unwrap();
        let flat = |k| EmuKey {
            buf: ksk_limbs(&b.primes, &fx.params, k, &shape)
                .unwrap()
                .concat(),
            shape,
        };
        let relin = fx.relin_key().map(|rk| flat(rk.key()));
        let galois = fx
            .galois_keys()
            .into_iter()
            .map(|gk| (gk.elt(), flat(gk.key())))
            .collect();
        EmuKs {
            fx,
            b,
            relin,
            galois,
        }
    }

    fn ks(&self) -> Ks<'_, EmuBackend> {
        Ks {
            b: &self.b,
            params: &self.fx.params,
            table: &self.b.primes,
        }
    }

    fn up(&self, ct: &Ciphertext) -> Vec<Vec<u64>> {
        ct.polys().iter().map(|p| p.limbs().concat()).collect()
    }

    fn down_poly(&self, flat: &[u64], level: usize) -> RnsPoly {
        self.b
            .primes
            .host_poly(&q_primes(level), flat, true)
            .unwrap()
    }

    fn down(&self, c: &[Vec<u64>], level: usize, scale: f64) -> Ciphertext {
        let polys = c.iter().map(|x| self.down_poly(x, level)).collect();
        Ciphertext::new(polys, scale).unwrap()
    }

    fn galois_key(&self, elt: u64) -> Result<&EmuKey> {
        self.galois
            .get(&elt)
            .ok_or_else(|| HekitError::InvalidParams("missing galois key".into()))
    }

    fn galois(&self, ct: &Ciphertext, elt: u64) -> Ciphertext {
        let c = self.up(ct);
        let key = self.galois_key(elt).unwrap();
        let r = self
            .ks()
            .galois([&c[0], &c[1]], ct.level(), elt, key.r())
            .unwrap();
        self.down(&r, ct.level(), ct.scale())
    }
}

impl KsExec for EmuKs<'_> {
    fn key_switch(&self, d: &RnsPoly, key: KeyChoice) -> (RnsPoly, RnsPoly) {
        let k = match key {
            KeyChoice::Relin => self.relin.as_ref().unwrap(),
            KeyChoice::Galois(g) => self.galois_key(g).unwrap(),
        };
        let level = d.level();
        let (kb, ka) = self
            .ks()
            .key_switch(&d.limbs().concat(), level, k.r())
            .unwrap();
        (self.down_poly(&kb, level), self.down_poly(&ka, level))
    }

    fn mul_relin_rescale(&self, a: &Ciphertext, b: &Ciphertext) -> [Ciphertext; 3] {
        let (x, y, l) = (self.up(a), self.up(b), a.level());
        let ks = self.ks();
        let t = ks.tensor([&x[0], &x[1]], [&y[0], &y[1]], l).unwrap();
        let scale = a.scale() * b.scale();
        let rk = self.relin.as_ref().unwrap().r();
        let r = ks.relinearize([&t[0], &t[1], &t[2]], l, rk).unwrap();
        let s = ks.rescale(&[&r[0], &r[1]], l).unwrap();
        [
            self.down(&t, l, scale),
            self.down(&r, l, scale),
            self.down(&s, l - 1, rescaled_scale(&self.fx.params, scale, l)),
        ]
    }

    fn rotate(&self, ct: &Ciphertext, steps: i32) -> Ciphertext {
        match rotation_elt(&self.fx.params, steps) {
            None => ct.clone(),
            Some(elt) => self.galois(ct, elt),
        }
    }

    fn conjugate(&self, ct: &Ciphertext) -> Ciphertext {
        self.galois(ct, conjugation_elt(self.fx.params.n()))
    }

    fn hoisted_rotate(&self, ct: &Ciphertext, steps: &[i32]) -> Vec<Ciphertext> {
        let c = self.up(ct);
        let elts: Vec<Option<u64>> = steps
            .iter()
            .map(|&s| rotation_elt(&self.fx.params, s))
            .collect();
        let rots: Vec<(u64, KeyRef<'_, Vec<u64>>)> = elts
            .iter()
            .flatten()
            .map(|&g| (g, self.galois_key(g).unwrap().r()))
            .collect();
        let done = self
            .ks()
            .hoisted([&c[0], &c[1]], ct.level(), &rots)
            .unwrap();
        merge_hoisted(
            &elts,
            done,
            || Ok(ct.clone()),
            |r| Ok(self.down(&r, ct.level(), ct.scale())),
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use hekit_ckks::CkksParams;

    use super::*;
    use crate::cases::ks::{
        check_hoisted_rotate, check_key_switch, check_mul_relin_rescale, check_rotate,
    };

    /// The fixtures of `tests/gpu_ks.rs` (same parameters, keys and seeds; see there).
    fn seal() -> &'static KsFixture {
        static F: OnceLock<KsFixture> = OnceLock::new();
        F.get_or_init(|| KsFixture::new(CkksParams::seal_2020(), true, &[1, -3], true, None, 46))
    }

    fn n16() -> &'static KsFixture {
        static F: OnceLock<KsFixture> = OnceLock::new();
        F.get_or_init(|| KsFixture::new(CkksParams::n16_train(), true, &[1, 5], false, None, 47))
    }

    fn seal_16() -> &'static KsFixture {
        static F: OnceLock<KsFixture> = OnceLock::new();
        F.get_or_init(|| {
            let steps: Vec<i32> = (1..=16).collect();
            KsFixture::new(CkksParams::seal_2020(), false, &steps, false, None, 48)
        })
    }

    /// The replay above is a hand transcription of `kernels/keyswitch.cu` (review M4 of Task
    /// 45): the source is pinned by hash. After editing the kernel, update the replay, have
    /// both re-reviewed, then update the hash here.
    #[test]
    fn keyswitch_kernel_source_is_pinned() {
        let h = crate::KEYSWITCH_KERNEL_SRC
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
            });
        assert_eq!(
            h, 0x7c5c_ad94_1365_486d,
            "kernels/keyswitch.cu changed (FNV-1a {h:#018x}): update and re-review its replay \
             (src/emu_ks.rs), then pin the new hash"
        );
    }

    /// The modular helpers of `keyswitch.cu` (`add_mod`, `sub_mod`, `barrett_est`, `mul_mod`)
    /// are a verbatim copy of `pointwise.cu`'s, so the replay of those (`emu_arith`, with its
    /// Barrett tests) covers them.
    #[test]
    fn keyswitch_helpers_are_a_verbatim_copy() {
        let src = crate::KEYSWITCH_KERNEL_SRC;
        let begin = "// ---- begin verbatim copy of kernels/pointwise.cu\n";
        let end = "// ---- end verbatim copy of kernels/pointwise.cu";
        let i = src.find(begin).unwrap() + begin.len();
        let block = &src[i..src.find(end).unwrap()];
        assert!(block.contains("mul_mod") && block.contains("barrett_est"));
        assert!(
            crate::POINTWISE_KERNEL_SRC.contains(block),
            "the helpers of keyswitch.cu differ from pointwise.cu"
        );
        assert_eq!(src.matches("__global__").count(), 1);
    }

    #[test]
    fn replay_key_switch_seal_2020() {
        check_key_switch(&EmuKs::new(seal()), seal());
    }

    #[test]
    fn replay_key_switch_n16_train() {
        check_key_switch(&EmuKs::new(n16()), n16());
    }

    #[test]
    fn replay_mul_relin_rescale_seal_2020() {
        check_mul_relin_rescale(&EmuKs::new(seal()), seal());
    }

    #[test]
    fn replay_mul_relin_rescale_n16_train() {
        check_mul_relin_rescale(&EmuKs::new(n16()), n16());
    }

    #[test]
    fn replay_rotate_seal_2020() {
        let fx = seal();
        let e = EmuKs::new(fx);
        check_rotate(&e, fx);
        check_hoisted_rotate(&e, fx, &[fx.params.max_level(), 1]);
    }

    #[test]
    fn replay_rotate_n16_train() {
        let fx = n16();
        let e = EmuKs::new(fx);
        check_rotate(&e, fx);
        check_hoisted_rotate(&e, fx, &[fx.params.max_level(), 1]);
    }

    /// Keys put on the device for levels <= 9 only (3 of 8 key parts, `key_skip` 0 to 9):
    /// the cap, levels and checks of `hoisted_rotate_16_low_level_matches_cpu_n16_train`
    /// (key switching at 9, 5, 3, 1, 0; hoisted at 9, 5, 1 with 3- / 2- / 1-digit
    /// accumulation and clipped digits), with the 2 Galois keys of the n16_train fixture
    /// instead of 16 (16 n16_train keys are 4.8 GB of host memory) plus its relinearization
    /// key.
    #[test]
    fn replay_capped_keys_n16_train() {
        let fx = n16().capped(9);
        let e = EmuKs::new(&fx);
        check_key_switch(&e, &fx);
        check_hoisted_rotate(&e, &fx, &[9, 5, 1]);
    }

    #[test]
    fn replay_hoisted_rotate_16_seal_2020() {
        let fx = seal_16();
        check_hoisted_rotate(&EmuKs::new(fx), fx, &[fx.params.max_level(), 1]);
    }
}
