//! ModRaise with sparse-secret encapsulation (SSE; Bossuat, Troncoso-Pastoriza, Hubaux,
//! ACNS 2022), the first step of bootstrapping (Task 37, controller ruling R-M3-9,
//! `docs/bootstrap.md` §2, §3, §7).
//!
//! A level-0 ciphertext `(c_0, c_1)` decrypts to `m = c_0 + c_1 s mod q_0`. Reading its components
//! as integers (centered,
//! [`RnsPoly::raise_from_level0`](ckks_train_core::poly::RnsPoly::raise_from_level0)) and reducing them
//! modulo the whole chain gives a ciphertext at the top level that decrypts to `t = m + q_0 I`,
//! with `I` the integer overflow of `c_0 + c_1 s`. EvalMod removes `q_0 I` later, which works only
//! while `|I| < K`. With the dense main key `I` has `sigma_I = sqrt((N/2 + 1) / 12)` (about 52 at
//! `N = 2^16`); SSE first switches the ciphertext to an ephemeral key `s'` of Hamming weight `h'`,
//! so that `sigma_I = sqrt((h' + 1) / 12)` (1.66 for `h' = 32`, hence `K = 13`), and switches back
//! to `s` at the top level:
//!
//! 1. at level 0, key-switch `c_1` from `s` to `s'` ([`SseKeys::to_sparse`], modulo `q_0 P`);
//! 2. raise both components from level 0 to `max_level` (centered lift);
//! 3. at `max_level`, key-switch `c_1` from `s'` to `s` ([`SseKeys::to_dense`], modulo
//!    `Q P`).
//!
//! The scale is unchanged (Delta): the result decrypts under `s` to `t = Delta c + q_0 I + e`,
//! and Task 38 relabels it to the CoeffToSlot input scale `q_0` (as `q_0 as f64`, which is
//! not exact for the 54-bit `q_0`: a relative relabelling error of at most `2^-54`).
//!
//! Noise: step 1 adds `(d e)/P + r_b + r_a s'` with `|r| <= 1/2` (the rounding of the
//! division by `P`; the digit term is about `q_0 / P ~ 2^-186` of the key error), so about
//! `sqrt((h' + 1) / 12)` per coefficient, 1.66 for `h' = 32`. Step 3 is a standard key switch
//! at the top level, about `sqrt((N/2 + 1) / 12)` (52 at `N = 2^16`) plus the digit term. In
//! `x = t / q_0` units both are below the CoeffToSlot noise of `docs/bootstrap.md` §6 (the
//! binding input-noise term); `docs/bootstrap.md` §10 has the measured values.

use std::sync::Arc;

use ckks_train_core::error::{HekitError, Result};

use crate::ciphertext::Ciphertext;
use crate::evaluator::Evaluator;
use crate::keys::{KskKey, SseKeys};
use crate::keyswitch::key_switch;

impl Evaluator {
    /// ModRaise with sparse-secret encapsulation (the [module docs](self) list the three
    /// steps): from a level-0 ciphertext decrypting to `m` to a `max_level` ciphertext with
    /// the same scale decrypting (under the main key) to `m + q_0 I`, `I` the overflow under
    /// the ephemeral key of `sse` (`sigma_I = sqrt((h' + 1) / 12)`). This is the path of the
    /// n16_boot bootstrap. Each call performs **exactly 2 key switches** (one at level 0 with
    /// `to_sparse`, one at the top with `to_dense`), counted as
    /// [`OpCounter::sse_switches`](crate::OpCounter::sse_switches) `+= 2` on success (Task 38).
    ///
    /// Errors, before any work: `InvalidParams` if `ct` is not a 2-component ciphertext over
    /// this evaluator's basis, or the keys are not SSE keys of these parameters (`to_sparse`
    /// one part with level-0 `Q` parts, `to_dense` `dnum` parts at `max_level`, over this
    /// evaluator's bases); `LevelMismatch` unless `ct` is at level 0.
    pub fn mod_raise(&self, ct: &Ciphertext, sse: &SseKeys) -> Result<Ciphertext> {
        self.check_mod_raise_input(ct)?;
        self.check_sse_key(sse.to_sparse(), 1, 0)?;
        self.check_sse_key(
            sse.to_dense(),
            self.params().dnum(),
            self.params().max_level(),
        )?;
        let sparse = self.sse_to_sparse(ct, sse);
        let raised = raise(self, &sparse)?;
        let (kb, ka) = key_switch(self.params(), &raised.polys()[1], sse.to_dense());
        let mut c0 = raised.into_polys().swap_remove(0);
        c0.add_assign(&kb);
        let out = Ciphertext::new(vec![c0, ka], ct.scale())?;
        self.count(|c| c.sse_switches += 2);
        Ok(out)
    }

    /// ModRaise under the main key, without SSE: both components raised from level 0 to
    /// `max_level` (centered lift), scale unchanged. The overflow then has the main key's
    /// `sigma_I` (52 for the dense n16_boot key, far beyond `K = 13`), so this is for tests
    /// and sparse main keys only; n16_boot uses [`Evaluator::mod_raise`].
    ///
    /// Errors as [`Evaluator::mod_raise`] (without the key checks).
    pub fn mod_raise_main_key(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.check_mod_raise_input(ct)?;
        raise(self, ct)
    }

    /// Step 1 of [`Evaluator::mod_raise`]: `(c_0 + kb, ka)` with `(kb, ka)` the switch of
    /// `c_1` to `s'` at level 0 (inputs checked by the caller).
    pub(crate) fn sse_to_sparse(&self, ct: &Ciphertext, sse: &SseKeys) -> Ciphertext {
        let c = ct.polys();
        let (kb, ka) = key_switch(self.params(), &c[1], sse.to_sparse());
        let mut c0 = c[0].clone();
        c0.add_assign(&kb);
        Ciphertext::new(vec![c0, ka], ct.scale()).expect("level-0 NTT components")
    }

    /// `InvalidParams` unless `ct` has 2 components over this basis, `LevelMismatch` unless
    /// it is at level 0.
    fn check_mod_raise_input(&self, ct: &Ciphertext) -> Result<()> {
        self.check_basis(ct.polys())?;
        if ct.len() != 2 {
            return Err(HekitError::InvalidParams(format!(
                "mod_raise needs a 2-component ciphertext (got {}); relinearize first",
                ct.len()
            )));
        }
        if ct.level() != 0 {
            return Err(HekitError::LevelMismatch);
        }
        Ok(())
    }

    /// `InvalidParams` unless `key` has `parts` parts over this evaluator's bases with `Q`
    /// parts at `level`.
    fn check_sse_key(&self, key: &KskKey, parts: usize, level: usize) -> Result<()> {
        let (bq, bp) = (self.params().basis_q(), self.params().basis_p());
        let ok = key.dnum() == parts
            && key.parts().iter().all(|(b, a)| {
                [b, a].iter().all(|x| {
                    Arc::ptr_eq(x.q().basis(), bq)
                        && Arc::ptr_eq(x.p().basis(), bp)
                        && x.q().level() == level
                })
            });
        if ok {
            Ok(())
        } else {
            Err(HekitError::InvalidParams(
                "SSE keys are not for this evaluator's parameters".into(),
            ))
        }
    }
}

/// Both components of a level-0 ciphertext raised to `max_level`.
fn raise(eval: &Evaluator, ct: &Ciphertext) -> Result<Ciphertext> {
    let top = eval.params().max_level();
    let c = ct
        .polys()
        .iter()
        .map(|p| p.raise_from_level0(top))
        .collect();
    Ciphertext::new(c, ct.scale())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::keys::{KeyGenerator, SecretKey};
    use crate::params::CkksParams;
    use crate::plaintext::Plaintext;
    use ckks_train_core::poly::RnsPoly;
    use ckks_train_core::sampler::Rng;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    fn unit(r: &mut Rng) -> f64 {
        2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
    }

    /// A plaintext with the integer coefficients `c_j` at level 0 (scale `scale`).
    fn coeff_plaintext(params: &CkksParams, c: &[i64], scale: f64) -> Plaintext {
        let m = params.basis_q().moduli()[0];
        let q = m.value() as i128;
        let limb = c
            .iter()
            .map(|&x| (x as i128).rem_euclid(q) as u64)
            .collect();
        let mut poly = RnsPoly::from_limbs(params.basis_q(), vec![limb], false).unwrap();
        poly.to_ntt();
        Plaintext::new(poly, scale).unwrap()
    }

    /// The centered integer coefficients of `pt`, which must lie within `q_0 q_1 / 2`
    /// (exact CRT over the first two limbs, `i128`; level 0 uses `q_0` alone).
    fn coeffs_of(params: &CkksParams, pt: &Plaintext) -> Vec<i128> {
        let mut p = pt.poly().clone();
        p.to_coeff();
        let q0 = params.q()[0] as i128;
        if p.level() == 0 {
            return p.limbs()[0]
                .iter()
                .map(|&a| {
                    let a = a as i128;
                    if a > q0 / 2 { a - q0 } else { a }
                })
                .collect();
        }
        let q1 = params.q()[1] as i128;
        let inv = modinv(q0 % q1, q1);
        let big = q0 * q1;
        (0..params.n())
            .map(|i| {
                let (a, b) = (p.limbs()[0][i] as i128, p.limbs()[1][i] as i128);
                let v = a + q0 * ((b - a).rem_euclid(q1) * inv).rem_euclid(q1);
                if v > big / 2 { v - big } else { v }
            })
            .collect()
    }

    fn modinv(a: i128, m: i128) -> i128 {
        let (mut t, mut nt, mut r, mut nr) = (0i128, 1i128, m, a.rem_euclid(m));
        while nr != 0 {
            let q = r / nr;
            (t, nt) = (nt, t - q * nt);
            (r, nr) = (nr, r - q * nr);
        }
        t.rem_euclid(m)
    }

    fn rms(v: &[f64]) -> f64 {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }

    /// Hamming weight of a secret key (centered limb 0).
    fn weight(params: &CkksParams, sk: &SecretKey) -> usize {
        let mut s = sk.qp().q().clone();
        s.to_coeff();
        let q0 = params.q()[0];
        s.limbs()[0].iter().filter(|&&x| x != 0 && x != q0).count()
    }

    struct Run {
        /// Overflows `I_j` of every coefficient of every ciphertext.
        i: Vec<f64>,
        /// Residual noise `t_j - Delta c_j - q_0 I_j` (encryption and both key switches).
        residual: Vec<f64>,
        /// Coefficient error of the level-0 switch to `s'`: `dec_{s'}(switched) - dec_s(ct)`.
        ks0: Vec<f64>,
        /// Coefficient error of the top-level switch back to `s`:
        /// `dec_s(mod_raise(ct)) - dec_{s'}(raise(switched))`.
        ks_top: Vec<f64>,
    }

    /// `count` ciphertexts of random coefficients `round(Delta u)` (`u` uniform in [-1, 1])
    /// at level 0, mod-raised with SSE of weight `h`; returns the overflows, the residual
    /// noise and the level-0 key-switch error (measured with the ephemeral secret).
    fn sse_run(params: CkksParams, h: usize, count: usize, seed: u8) -> Run {
        let params = Arc::new(params);
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let pk = kg.public_key();
        let (sse, s_eph) = kg.sse_keys_with_secret(h).unwrap();
        assert_eq!(weight(&params, &s_eph), h);
        let eval = Evaluator::new(Arc::clone(&params), None, None);
        let encryptor = Encryptor::new(Arc::clone(&params), pk);
        let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
        let dec_eph = Decryptor::new(Arc::clone(&params), s_eph);
        let (q0, delta) = (params.q()[0] as f64, params.scale());
        let mut r = rng(seed.wrapping_add(1));
        let mut run = Run {
            i: Vec::new(),
            residual: Vec::new(),
            ks0: Vec::new(),
            ks_top: Vec::new(),
        };
        for _ in 0..count {
            let c: Vec<i64> = (0..params.n())
                .map(|_| (unit(&mut r) * delta).round() as i64)
                .collect();
            let ct = encryptor.encrypt(&coeff_plaintext(&params, &c, delta), &mut r);
            let before = coeffs_of(&params, &dec.decrypt(&ct));
            let sparse = eval.sse_to_sparse(&ct, &sse);
            let after = coeffs_of(&params, &dec_eph.decrypt(&sparse));
            run.ks0
                .extend(before.iter().zip(&after).map(|(a, b)| (b - a) as f64));
            let raised = eval.mod_raise(&ct, &sse).unwrap();
            assert_eq!(
                (raised.level(), raised.scale(), raised.len()),
                (params.max_level(), delta, 2)
            );
            let t = coeffs_of(&params, &dec.decrypt(&raised));
            let t_eph = coeffs_of(&params, &dec_eph.decrypt(&raise(&eval, &sparse).unwrap()));
            run.ks_top
                .extend(t.iter().zip(&t_eph).map(|(a, b)| (a - b) as f64));
            for (&tj, &cj) in t.iter().zip(&c) {
                let i = ((tj - cj as i128) as f64 / q0).round();
                run.i.push(i);
                run.residual
                    .push((tj - cj as i128 - i as i128 * params.q()[0] as i128) as f64);
            }
        }
        run
    }

    /// Bits of precision a coefficient error of rms `sigma` (integer units of the raised
    /// plaintext, so `sigma / q_0` in `x` units) allows on its own, with the model of
    /// `docs/bootstrap.md` §6: `-log2(4.2 sqrt(N/2) (q_0 / Delta) sigma / q_0)`.
    fn bits_allowed(sigma: f64, n: usize, delta: f64) -> f64 {
        -(4.2 * (n as f64 / 2.0).sqrt() * sigma / delta).log2()
    }

    /// Prints and checks one SSE run: `sigma_I` within `tol` of `sqrt((h' + 1) / 12)`,
    /// `|I| <= 13`, the level-0 switch error the `P` rounding (`sqrt((h' + 1) / 12)`).
    fn check_run(run: &Run, name: &str, h: usize, n: usize, delta: f64, tol: f64) {
        let want = ((h as f64 + 1.0) / 12.0).sqrt();
        let sigma = rms(&run.i);
        let mean = run.i.iter().sum::<f64>() / run.i.len() as f64;
        let max_i = run.i.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let (ks0, ks_top, res) = (rms(&run.ks0), rms(&run.ks_top), rms(&run.residual));
        eprintln!(
            "{name}, h' {h}, {} coefficients: sigma_I {sigma:.4} (want {want:.4}), mean \
             {mean:.4}, max |I| {max_i}; key-switch rms per coefficient: level 0 (s -> s') \
             {ks0:.3} ({:.1} bits alone), top (s' -> s) {ks_top:.2} ({:.1} bits alone); \
             residual rms {res:.1} ({:.1} bits alone)",
            run.i.len(),
            bits_allowed(ks0, n, delta),
            bits_allowed(ks_top, n, delta),
            bits_allowed(res, n, delta),
        );
        assert!((sigma / want - 1.0).abs() < tol, "sigma_I {sigma}");
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!(max_i <= 13.0, "max |I| {max_i}");
        assert!((ks0 / want - 1.0).abs() < 0.15, "level-0 switch {ks0}");
    }

    /// N = 2^12 with alpha = 2 (insecure, tests only): q_0 54 bits, the overflow fits the CRT
    /// over q_0 q_1.
    fn mid() -> CkksParams {
        CkksParams::new(12, &[54, 42, 42, 42, 56, 56], &[60, 60], 42, 3).unwrap()
    }

    /// R-M3-9: after the SSE mod-raise the plaintext under s is `m + q_0 I` with
    /// `|I| <= K = 13` and the empirical `sigma_I` within 5% of `sqrt((h' + 1) / 12)`
    /// (h' = 32: 1.658); the level-0 key-switch error is the `P` rounding,
    /// `sqrt((h' + 1) / 12)` per coefficient.
    #[test]
    fn sse_overflow_statistics() {
        let p = mid();
        let (n, delta) = (p.n(), p.scale());
        let run = sse_run(p, 32, 4, 1);
        check_run(&run, "N 2^12", 32, n, delta, 0.05);
        assert!(run.residual.iter().all(|e| e.abs() < 2f64.powi(14)));
    }

    /// **Heavy (N = 2^16, n16_boot keys: the s' -> s key is 0.3 GB).** R-M3-9 on n16_boot:
    /// dense main key, h' = 32, two ciphertexts (2^17 coefficients): `|I| <= K = 13`,
    /// `sigma_I` within 2% of `sqrt(33 / 12) = 1.658`, and the key-switch errors, which the
    /// report compares with the binding input-noise term of `docs/bootstrap.md` §6 (17.4
    /// bits).
    #[test]
    fn n16_boot_keys_sse_mod_raise() {
        let p = CkksParams::n16_boot();
        let (n, delta) = (p.n(), p.scale());
        let run = sse_run(p, crate::bootstrap::params::N16_EPHEMERAL_HW, 2, 21);
        check_run(&run, "n16_boot", 32, n, delta, 0.02);
        // Both switches allow far more than the 17.4 bits of the binding term.
        assert!(bits_allowed(rms(&run.ks0), n, delta) > 28.0);
        assert!(bits_allowed(rms(&run.ks_top), n, delta) > 24.0);
    }

    /// Without SSE the overflow has the dense key's `sigma_I = sqrt((h + 1) / 12)` (h ~ N/2):
    /// 13.1 at N = 2^12, above K = 13 for a third of the coefficients.
    #[test]
    fn main_key_mod_raise_has_the_dense_overflow() {
        let params = Arc::new(mid());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(5)).unwrap();
        let pk = kg.public_key();
        let h = weight(&params, &kg.secret_key());
        let eval = Evaluator::new(Arc::clone(&params), None, None);
        let encryptor = Encryptor::new(Arc::clone(&params), pk);
        let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
        let mut r = rng(6);
        let delta = params.scale();
        let c: Vec<i64> = (0..params.n())
            .map(|_| (unit(&mut r) * delta).round() as i64)
            .collect();
        let ct = encryptor.encrypt(&coeff_plaintext(&params, &c, delta), &mut r);
        let raised = eval.mod_raise_main_key(&ct).unwrap();
        assert_eq!(
            (raised.level(), raised.scale()),
            (params.max_level(), delta)
        );
        let t = coeffs_of(&params, &dec.decrypt(&raised));
        let q0 = params.q()[0] as f64;
        let i: Vec<f64> = t
            .iter()
            .zip(&c)
            .map(|(&tj, &cj)| ((tj - cj as i128) as f64 / q0).round())
            .collect();
        let want = ((h as f64 + 1.0) / 12.0).sqrt();
        eprintln!("main key h {h}: sigma_I {:.3} (want {want:.3})", rms(&i));
        assert!((rms(&i) / want - 1.0).abs() < 0.05);
    }

    /// The SSE path on test_tiny (N = 2^5, ephemeral weight 4): the raised ciphertext still
    /// decrypts to the message modulo q_0, and the checks fail before any work.
    #[test]
    fn tiny_mod_raise_and_errors() {
        let params = Arc::new(CkksParams::test_tiny());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(7)).unwrap();
        let pk = kg.public_key();
        let sse = kg.sse_keys(4).unwrap();
        assert_eq!(sse.hamming_weight(), 4);
        assert_eq!((sse.to_sparse().dnum(), sse.to_dense().dnum()), (1, 12));
        assert!(format!("{sse:?}").starts_with("SseKeys { hamming_weight: 4"));
        let eval = Evaluator::new(Arc::clone(&params), None, None);
        let encryptor = Encryptor::new(Arc::clone(&params), pk);
        let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
        let mut r = rng(8);
        let delta = params.scale();
        let c: Vec<i64> = (0..params.n())
            .map(|_| (unit(&mut r) * delta).round() as i64)
            .collect();
        let ct = encryptor.encrypt(&coeff_plaintext(&params, &c, delta), &mut r);
        let raised = eval.mod_raise(&ct, &sse).unwrap();
        // Task 38: one mod_raise is exactly 2 key switches, counted as sse_switches.
        let k = eval.counter();
        assert_eq!(
            k,
            crate::OpCounter {
                sse_switches: 2,
                ..Default::default()
            }
        );
        assert_eq!(k.key_switches(), 2);
        let low = eval.mod_switch_to_level(&raised, 0).unwrap();
        let got = coeffs_of(&params, &dec.decrypt(&low));
        let err = got
            .iter()
            .zip(&c)
            .map(|(&g, &w)| (g - w as i128).abs())
            .max()
            .unwrap();
        assert!(err < 1 << 12, "{err}");

        assert!(matches!(kg.sse_keys(0), Err(HekitError::InvalidParams(_))));
        assert!(matches!(kg.sse_keys(33), Err(HekitError::InvalidParams(_))));
        let inv = |e: Result<Ciphertext>| matches!(e, Err(HekitError::InvalidParams(_)));
        assert_eq!(
            eval.mod_raise(&raised, &sse).unwrap_err(),
            HekitError::LevelMismatch
        );
        assert_eq!(
            eval.mod_raise_main_key(&raised).unwrap_err(),
            HekitError::LevelMismatch
        );
        let other = Arc::new(CkksParams::test_tiny());
        let mut kg2 = KeyGenerator::new(Arc::clone(&other), rng(9)).unwrap();
        let foreign = kg2.sse_keys(4).unwrap();
        assert!(inv(eval.mod_raise(&ct, &foreign)));
        let swapped = SseKeys::from_parts(4, sse.to_dense().clone(), sse.to_sparse().clone());
        assert!(inv(eval.mod_raise(&ct, &swapped)));
        let enc = Encoder::new(Arc::clone(&params));
        let three = {
            let x = encryptor.encrypt(&enc.encode(&[0.5], delta, 1).unwrap(), &mut r);
            let rk = kg.relin_key();
            let e = Evaluator::new(Arc::clone(&params), Some(rk), None);
            e.mod_switch_to_level(&e.mul_no_relin(&x, &x).unwrap(), 0)
                .unwrap()
        };
        assert_eq!(
            eval.mod_raise(&three, &sse).unwrap_err(),
            HekitError::InvalidParams(
                "mod_raise needs a 2-component ciphertext (got 3); relinearize first".into()
            )
        );
        assert!(inv(eval.mod_raise_main_key(&three)));
        // Failures count nothing; the main-key raise has no switch.
        assert_eq!(eval.counter(), k);
        eval.mod_raise_main_key(&ct).unwrap();
        assert_eq!(eval.counter(), k);
    }
}
