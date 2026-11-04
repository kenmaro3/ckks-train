//! Hybrid (GHS-style, `dnum`-digit) key switching.
//!
//! A polynomial `d` (NTT form, level `l`) that decrypts under a source key `s'` is turned
//! into a pair `(kb, ka)` at level `l` with `kb + ka s ~ d s'` (see [`crate::keys`] for the
//! key layout). The switch is split in two so rotations can be hoisted (Task 18):
//!
//! 1. [`decompose`]: for every digit `j` whose first prime `j * alpha` is `<= l` (the digit
//!    is clipped to `j*alpha .. min((j+1)*alpha, l+1)`), the **centered** lift of the digit
//!    to `Q_l * P` ([`RnsPoly::mod_up_digit_centered`]). Each lifted digit is a [`QpPoly`]:
//!    `Q` part over `params.basis_q()` at level `l`, `P` part over `params.basis_p()` (all
//!    special primes), both in NTT form.
//! 2. [`apply_key`]: the inner product `sum_j d_j * (b_j, a_j)` over `Q_l * P` (key `Q`
//!    parts used through their first `l + 1` limbs, `P` parts whole), followed by
//!    [`RnsPoly::mod_down`] (division by `P` with rounding) of both accumulators.
//!
//! [`key_switch`] is `apply_key(decompose(d))`.
//!
//! # Noise
//!
//! `kb + ka s - d s' = (sum_j d_j e_j) / P + r_b + r_a s` (`r` the `mod_down` roundings,
//! `|r| <= k/2`-ish). Because `d_j` is the centered representative (`|d_j| <= Q_{D_j}/2`,
//! zero mean), the first term is a zero-mean sum; the non-centered lift of
//! [`RnsPoly::mod_up_digit`] has mean `Q_{D_j}/2`, which for seal_2020 (digit 0 = the 60-bit
//! `q_0 ~ P`) raised the slot error of a rotation from `2^-22.7` to `2^-19.8` times the scale
//! (Task 17 report; pinned by the tests below).
//!
//! # Hoisting (Task 18)
//!
//! An automorphism `sigma_g` commutes with the lift up to the `u * Q_D` term: apply
//! `automorphism(g)` to both parts of every digit returned by [`decompose`] (of `c_1`), then
//! call [`apply_key`] with the Galois key of `g`. For `alpha = 1` (seal_2020) the centered
//! lift is exact and `sigma_g(lift(d)) = lift(sigma_g(d))` holds exactly.

use std::sync::Arc;

use hekit_core::poly::RnsPoly;
use hekit_core::rns::RnsBasis;

use crate::keys::{KskKey, QpPoly};
use crate::params::CkksParams;

/// The prime-index ranges of the digits of a level-`l` polynomial: digit `j` covers
/// `j*alpha .. min((j+1)*alpha, l+1)`; digits starting above `l` are omitted.
pub fn digit_ranges(params: &CkksParams, level: usize) -> Vec<std::ops::Range<usize>> {
    let alpha = params.alpha();
    (0..params.dnum())
        .map(|j| j * alpha..((j + 1) * alpha).min(level + 1))
        .take_while(|r| r.start <= level)
        .collect()
}

/// The centered lifts of the digits of `poly` to `Q_l * P` (module docs, step 1), one
/// [`QpPoly`] per digit of [`digit_ranges`], in NTT form.
///
/// # Panics
/// If `poly` is not in NTT form, not over `params.basis_q()` (same `Arc`), or its level
/// exceeds `params.max_level()`.
pub fn decompose(params: &CkksParams, poly: &RnsPoly) -> Vec<QpPoly> {
    decompose_with(params, poly, true)
}

/// [`decompose`] with a choice of lift (`centered = false`: [`RnsPoly::mod_up_digit`]); the
/// non-centered variant is kept for the noise-comparison tests only.
fn decompose_with(params: &CkksParams, poly: &RnsPoly, centered: bool) -> Vec<QpPoly> {
    assert!(poly.is_ntt_form(), "key switching needs NTT form");
    assert!(
        Arc::ptr_eq(poly.basis(), params.basis_q()),
        "polynomial is not over this parameter set's data basis"
    );
    let l = poly.level();
    assert!(l <= params.max_level(), "level {l} above max_level");
    digit_ranges(params, l)
        .into_iter()
        .map(|digit| {
            let (q, p) = if centered {
                poly.mod_up_digit_centered(digit, params.basis_p())
            } else {
                poly.mod_up_digit(digit, params.basis_p())
            };
            QpPoly::new(q, p)
        })
        .collect()
}

/// Inner product of lifted digits with the key parts, divided by `P` (module docs, step 2):
/// returns `(kb, ka)` at the digits' level `l`, NTT form, over `params.basis_q()`.
///
/// `digits[j]` is used with `key.parts()[j]`; `digits` would normally come from
/// [`decompose`] (possibly with an automorphism applied to every part, for hoisting).
///
/// # Panics
/// If `digits` is empty or longer than `key.dnum()`, or the digits are not all at the same
/// level and in NTT form.
pub fn apply_key(params: &CkksParams, digits: &[QpPoly], key: &KskKey) -> (RnsPoly, RnsPoly) {
    assert!(!digits.is_empty(), "no digits to switch");
    assert!(
        digits.len() <= key.dnum(),
        "{} digits for a key with {} parts",
        digits.len(),
        key.dnum()
    );
    let l = digits[0].q().level();
    assert!(
        digits
            .iter()
            .all(|d| d.q().level() == l && d.q().is_ntt_form() && d.p().is_ntt_form()),
        "digits must share one level and be in NTT form"
    );
    let parts = &key.parts()[..digits.len()];
    let (basis_q, basis_p) = (params.basis_q(), params.basis_p());
    // One accumulator pass per (part, prime): acc_b += d * b_j, acc_a += d * a_j.
    let inner = |basis: &Arc<RnsBasis>, limbs: usize, pick: &dyn Fn(&QpPoly) -> &RnsPoly| {
        let moduli = &basis.moduli()[..limbs];
        let n = params.n();
        let mut acc_b = vec![vec![0u64; n]; limbs];
        let mut acc_a = vec![vec![0u64; n]; limbs];
        for (d, (b, a)) in digits.iter().zip(parts) {
            let (d, b, a) = (pick(d), pick(b), pick(a));
            for (i, m) in moduli.iter().enumerate() {
                let (dl, bl, al) = (&d.limbs()[i], &b.limbs()[i], &a.limbs()[i]);
                for (((xb, xa), &x), (&yb, &ya)) in acc_b[i]
                    .iter_mut()
                    .zip(acc_a[i].iter_mut())
                    .zip(dl)
                    .zip(bl.iter().zip(al))
                {
                    *xb = m.add(*xb, m.mul(x, yb));
                    *xa = m.add(*xa, m.mul(x, ya));
                }
            }
        }
        let wrap = |limbs| RnsPoly::from_limbs(basis, limbs, true).expect("reduced limbs");
        (wrap(acc_b), wrap(acc_a))
    };
    let (mut bq, mut aq) = inner(basis_q, l + 1, &|x: &QpPoly| x.q());
    let (bp, ap) = inner(basis_p, basis_p.len(), &|x: &QpPoly| x.p());
    bq.mod_down(&bp);
    aq.mod_down(&ap);
    (bq, aq)
}

/// Switches `poly` (NTT form, level `l`, decrypting under the key's source key `s'`) to the
/// secret key: `(kb, ka)` at level `l` with `kb + ka s ~ poly * s'`.
/// `= apply_key(params, &decompose(params, poly), key)`.
///
/// # Panics
/// As [`decompose`] and [`apply_key`].
pub fn key_switch(params: &CkksParams, poly: &RnsPoly, key: &KskKey) -> (RnsPoly, RnsPoly) {
    apply_key(params, &decompose(params, poly), key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::keys::{KeyGenerator, SecretKey, galois_elt};
    use crate::{Ciphertext, Encoder};
    use hekit_core::sampler::Rng;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    /// N = 2^12, 6 data primes in 3 digits of 2, 2 special primes.
    fn small_params() -> Arc<CkksParams> {
        Arc::new(CkksParams::new(12, &[60, 40, 40, 40, 40, 40], &[60, 60], 40, 3).unwrap())
    }

    fn rand_vals(seed: u8, len: usize) -> Vec<f64> {
        let mut r = rng(seed);
        (0..len)
            .map(|_| (r.next_u64() >> 11) as f64 * 2f64.powi(-52) - 1.0)
            .collect()
    }

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// `log2 ||kb + ka s - d s'||_inf` (centered, limb 0).
    fn ks_noise_log2(
        sk: &SecretKey,
        d: &RnsPoly,
        s_prime: &RnsPoly,
        kb: &RnsPoly,
        ka: &RnsPoly,
    ) -> f64 {
        let l = d.level();
        let mut t = ka.clone();
        t.mul_assign(&sk.at_level(l));
        t.add_assign(kb);
        let mut ds = d.clone();
        ds.mul_assign(&crate::keys::truncated(s_prime, l));
        t.sub_assign(&ds);
        t.to_coeff();
        let q = t.basis().moduli()[0].value();
        let max = t.limbs()[0]
            .iter()
            .map(|&x| if x > q / 2 { q - x } else { x })
            .max()
            .unwrap();
        (max.max(1) as f64).log2()
    }

    #[test]
    fn digit_ranges_clip_to_level() {
        let p = small_params();
        assert_eq!(digit_ranges(&p, 5), vec![0..2, 2..4, 4..6]);
        assert_eq!(digit_ranges(&p, 4), vec![0..2, 2..4, 4..5]);
        assert_eq!(digit_ranges(&p, 2), vec![0..2, 2..3]);
        assert_eq!(digit_ranges(&p, 1), vec![0..2]);
        assert_eq!(digit_ranges(&p, 0), vec![0..1]);
        let s = CkksParams::seal_2020();
        assert_eq!(digit_ranges(&s, 9).len(), 10);
        assert_eq!(digit_ranges(&s, 3), vec![0..1, 1..2, 2..3, 3..4]);
    }

    #[test]
    fn decompose_shapes() {
        let p = small_params();
        let mut d = rng(1).uniform_poly(p.basis_q(), 4);
        d.to_ntt();
        let digits = decompose(&p, &d);
        assert_eq!(digits.len(), 3);
        for (dig, range) in digits.iter().zip(digit_ranges(&p, 4)) {
            assert_eq!(dig.q().level(), 4);
            assert_eq!(dig.p().level(), 1);
            assert!(dig.q().is_ntt_form() && dig.p().is_ntt_form());
            assert!(Arc::ptr_eq(dig.q().basis(), p.basis_q()));
            assert!(Arc::ptr_eq(dig.p().basis(), p.basis_p()));
            for i in range {
                assert_eq!(dig.q().limbs()[i], d.limbs()[i]);
            }
        }
    }

    /// Relinearization-key and Galois-key switches at several levels: the coefficient noise
    /// `kb + ka s - d s'` is small, and the centered lift beats the non-centered one on
    /// seal_2020 (digit 0 = q_0 ~ P). Prints the measured numbers for the report.
    #[test]
    fn key_switch_noise_and_centered_lift() {
        for params in [small_params(), Arc::new(CkksParams::seal_2020())] {
            let mut kg = KeyGenerator::new(Arc::clone(&params), rng(2)).unwrap();
            let sk = kg.secret_key();
            let rk = kg.relin_key();
            let mut s2 = sk.qp().q().clone();
            s2.mul_assign(sk.qp().q());
            for level in [params.max_level(), 2, 0] {
                let mut d = rng(3).uniform_poly(params.basis_q(), level);
                d.to_ntt();
                let (kb, ka) = key_switch(&params, &d, rk.key());
                assert_eq!((kb.level(), ka.level()), (level, level));
                let centered = ks_noise_log2(&sk, &d, &s2, &kb, &ka);
                let (kb, ka) = apply_key(&params, &decompose_with(&params, &d, false), rk.key());
                let plain = ks_noise_log2(&sk, &d, &s2, &kb, &ka);
                eprintln!(
                    "N=2^{} level {level}: relin-key switch noise ||.||_inf = 2^{centered:.2} \
                     (centered), 2^{plain:.2} (non-centered)",
                    params.log_n()
                );
                // ~ sqrt(N) * sigma * (Q_D / P) * sqrt(dnum) + rounding * ||s||.
                assert!(centered < 0.5 * params.log_n() as f64 + 8.0, "{centered}");
            }
        }
    }

    /// Rotation by 3 through a Galois-key switch, decoded: slot error with the centered lift
    /// vs the non-centered lift (Task 16 measured 2^-19.8 non-centered on seal_2020).
    #[test]
    fn rotation_error_centered_vs_non_centered() {
        for params in [small_params(), Arc::new(CkksParams::seal_2020())] {
            let enc = Encoder::new(Arc::clone(&params));
            let mut kg = KeyGenerator::new(Arc::clone(&params), rng(7)).unwrap();
            let encryptor = Encryptor::new(Arc::clone(&params), kg.public_key());
            let gks = kg.galois_keys(&[3]);
            let decryptor = Decryptor::new(Arc::clone(&params), kg.secret_key());
            let slots = params.slots();
            let v = rand_vals(9, slots);
            let want: Vec<f64> = (0..slots).map(|i| v[(i + 3) % slots]).collect();
            let g = galois_elt(params.n(), 3);
            let mut r = rng(8);
            for level in [params.max_level(), 0] {
                let ct = encryptor.encrypt(&enc.encode(&v, params.scale(), level).unwrap(), &mut r);
                let d = ct.polys()[1].automorphism(g);
                let mut errs = [0.0; 2];
                for (e, centered) in errs.iter_mut().zip([true, false]) {
                    let digits = decompose_with(&params, &d, centered);
                    let (kb, ka) = apply_key(&params, &digits, gks.get(3).unwrap().key());
                    let mut r0 = ct.polys()[0].automorphism(g);
                    r0.add_assign(&kb);
                    let rot = Ciphertext::new(vec![r0, ka], ct.scale()).unwrap();
                    *e = max_err(&enc.decode(&decryptor.decrypt(&rot)), &want).log2();
                }
                eprintln!(
                    "N=2^{} level {level}: rotation slot error 2^{:.2} (centered), 2^{:.2} \
                     (non-centered)",
                    params.log_n(),
                    errs[0],
                    errs[1]
                );
                assert!(errs[0] < -21.0, "centered rotation error 2^{}", errs[0]);
                if params.log_n() == 15 && level == params.max_level() {
                    assert!(errs[0] < errs[1] - 1.0, "centering should help: {errs:?}");
                }
            }
        }
    }

    /// Hoisting shape for Task 18: applying the automorphism to every lifted digit and then
    /// `apply_key` equals switching the rotated polynomial (exactly, for alpha = 1, and up to
    /// small noise otherwise).
    #[test]
    fn hoisted_digits_match_direct_switch() {
        let params = Arc::new(CkksParams::new(12, &[60, 40, 40, 40], &[60], 40, 4).unwrap());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(11)).unwrap();
        let gks = kg.galois_keys(&[5]);
        let key = gks.get(5).unwrap().key();
        let g = galois_elt(params.n(), 5);
        let mut d = rng(12).uniform_poly(params.basis_q(), 3);
        d.to_ntt();
        let hoisted: Vec<QpPoly> = decompose(&params, &d)
            .into_iter()
            .map(|x| QpPoly::new(x.q().automorphism(g), x.p().automorphism(g)))
            .collect();
        let a = apply_key(&params, &hoisted, key);
        let b = key_switch(&params, &d.automorphism(g), key);
        assert_eq!(a.0.limbs(), b.0.limbs());
        assert_eq!(a.1.limbs(), b.1.limbs());
    }
}
