//! Hybrid (GHS-style, `dnum`-digit) key switching.
//!
//! A polynomial `d` (NTT form, level `l`) that decrypts under a source key `s'` is turned
//! into a pair `(kb, ka)` at level `l` with `kb + ka s ~ d s'` (see [`crate::keys`] for the
//! key layout). The switch is split in two so rotations can be hoisted (Task 18):
//!
//! 1. [`decompose`]: for every digit `j` whose first prime `j * alpha` is `<= l` (the digit
//!    is clipped to `j*alpha .. min((j+1)*alpha, l+1)`), the **exact centered** lift of the
//!    digit to `Q_l * P` (the representative in `(-Q_D/2, Q_D/2)` reduced modulo every other
//!    prime, via `BaseConverter::convert_exact`). Each lifted digit is a [`QpPoly`]:
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
//! `|r| <= 1/2`: `mod_down` rounds exactly). `d_j` is the exact centered representative
//! (`|d_j| <= Q_{D_j}/2`), so for a uniform digit it has zero mean and variance
//! `Q_{D_j}^2 / 12`, and the first term is a zero-mean sum with coefficient variance
//! `N sigma^2 sum_j Q_{D_j}^2 / (12 P^2)` (the test `key_switch_noise_and_centered_lift`
//! bounds the measured noise by 6 times the predicted standard deviation).
//!
//! The other lifts, kept for comparison in the tests, add `u Q_{D_j}` (`u` the overflow of
//! the fast base conversion, `0 <= u <= alpha - 1`, mean `~ (alpha - 1)/2`), so they are
//! **not** zero-mean for `alpha > 1`:
//!
//! * non-centered ([`RnsPoly::mod_up_digit`], digit in `[0, Q_D)`): mean `~ alpha Q_D / 2`.
//!   For seal_2020 (digit 0 = the 60-bit `q_0 ~ P`) this raised the slot error of a rotation
//!   from `2^-22.7` to `2^-19.8` times the scale (Task 17 report; pinned below);
//! * fast centered ([`RnsPoly::mod_up_digit_centered`], Task 17's lift): `x_c + u Q_D`, mean
//!   `~ (alpha - 1) Q_D / 2`. It equals the exact lift for `alpha = 1`. For `alpha > 1` the
//!   mean matters only when `Q_D` is close to `P` (n16_train: `Q_D / P <= 2^-48`), but the
//!   `u` term breaks the symmetry hoisting needs (next section), hence the exact lift.
//!
//! # Hoisting (Task 18)
//!
//! An automorphism `sigma_g` is a signed permutation of the coefficients (in NTT form an
//! index permutation, the same for every prime), so it commutes with reduction modulo every
//! prime of `Q_l * P`: `sigma_g(lift(d))` is an integer polynomial congruent to `sigma_g(d)`
//! modulo `Q_{D_j}` with the same coefficient bound, i.e. a valid lift of `sigma_g(d)`'s
//! digit, and switching it with the Galois key is correct.
//! [`crate::Evaluator::hoisted_rotate`] therefore applies `automorphism(g)` to both parts of
//! every digit returned by [`decompose`] (of `c_1`) and calls [`apply_key`] with the Galois
//! key of `g`. The exact centered lift is moreover symmetric (`Q_D` odd, so the centered
//! representative of `-x` is `-x_c`), hence `sigma_g(lift(d)) = lift(sigma_g(d))` exactly and
//! the hoisted result is bit-identical to the direct switch for every `alpha` (barring the
//! near-tie case of `BaseConverter::convert_exact`, probability `~ 2^-50` per coefficient).
//! With the fast centered lift and `alpha > 1` the two differ by a fresh rounding noise
//! (measured `2^-25.5` decoded on n16_train, too close to the `2^-25` requirement).

use std::ops::Range;
use std::sync::Arc;

use hekit_core::poly::RnsPoly;
use hekit_core::rns::RnsBasis;
use hekit_core::{BaseConverter, Modulus, NttTable};

use crate::keys::{KskKey, QpPoly};
use crate::params::CkksParams;

/// The prime-index ranges of the digits of a level-`l` polynomial: digit `j` covers
/// `j*alpha .. min((j+1)*alpha, l+1)`; digits starting above `l` are omitted.
pub fn digit_ranges(params: &CkksParams, level: usize) -> Vec<Range<usize>> {
    let alpha = params.alpha();
    (0..params.dnum())
        .map(|j| j * alpha..((j + 1) * alpha).min(level + 1))
        .take_while(|r| r.start <= level)
        .collect()
}

/// The exact centered lifts of the digits of `poly` to `Q_l * P` (module docs, step 1), one
/// [`QpPoly`] per digit of [`digit_ranges`], in NTT form.
///
/// # Panics
/// If `poly` is not in NTT form, not over `params.basis_q()` (same `Arc`), or its level
/// exceeds `params.max_level()`.
pub fn decompose(params: &CkksParams, poly: &RnsPoly) -> Vec<QpPoly> {
    decompose_with(params, poly, Lift::Exact)
}

/// How a digit is lifted from `Q_D` to `Q_l * P`. Only [`Lift::Exact`] is used by the
/// library; the others are kept for the noise-comparison tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lift {
    /// The centered representative `x_c` exactly (`u = 0`): [`lift_digit_exact`].
    Exact,
    /// `x_c + u Q_D`, `0 <= u <= alpha - 1` ([`RnsPoly::mod_up_digit_centered`]).
    #[cfg_attr(not(test), allow(dead_code))]
    CenteredFast,
    /// `x_D + u Q_D` with `x_D` in `[0, Q_D)` ([`RnsPoly::mod_up_digit`]).
    #[cfg_attr(not(test), allow(dead_code))]
    Plain,
}

fn decompose_with(params: &CkksParams, poly: &RnsPoly, lift: Lift) -> Vec<QpPoly> {
    assert!(poly.is_ntt_form(), "key switching needs NTT form");
    assert!(
        Arc::ptr_eq(poly.basis(), params.basis_q()),
        "polynomial is not over this parameter set's data basis"
    );
    let l = poly.level();
    assert!(l <= params.max_level(), "level {l} above max_level");
    let bp = params.basis_p();
    digit_ranges(params, l)
        .into_iter()
        .map(|digit| match lift {
            Lift::Exact => lift_digit_exact(poly, digit, bp),
            Lift::CenteredFast => {
                let (q, p) = poly.mod_up_digit_centered(digit, bp);
                QpPoly::new(q, p)
            }
            Lift::Plain => {
                let (q, p) = poly.mod_up_digit(digit, bp);
                QpPoly::new(q, p)
            }
        })
        .collect()
}

/// The exact centered lift of the digit `digit` of `poly` (NTT form) to `Q_l * P`: the
/// digit limbs are copied, and every other `Q_l` limb and every `P` limb is `x_c mod r` for
/// `x_c` the representative of the digit value in `[-(Q_D-1)/2, (Q_D-1)/2]` (`Q_D` odd).
///
/// Computed as [`BaseConverter::convert_exact`] of `[x + h]_{Q_D}` minus `h mod r`,
/// `h = (Q_D - 1)/2` (the offset trick of `RnsPoly::mod_up_digit_centered`, with the
/// overflow `u` of the fast conversion removed). For a one-prime digit this is exactly
/// `mod_up_digit_centered`; for `alpha > 1` it is exact except when `x_c` lies within about
/// `alpha * 2^-52 * Q_D` of `+-Q_D/2`, where it may be `x_c -+ Q_D` (still a valid lift).
/// Zero mean and symmetric (`lift(-x) = -lift(x)`), so it commutes with automorphisms
/// (module docs, "Hoisting").
///
/// Kept here rather than in `hekit_core::baseconv` (Task 18 was not to edit that file while
/// Task 17's fix was in flight); it only uses public core APIs.
fn lift_digit_exact(poly: &RnsPoly, digit: Range<usize>, basis_p: &Arc<RnsBasis>) -> QpPoly {
    let l = poly.level();
    assert!(
        digit.start < digit.end && digit.end <= l + 1,
        "digit {digit:?} must be a non-empty range within 0..={l}"
    );
    let basis = poly.basis();
    let (qs, tables) = (&basis.moduli()[..=l], basis.ntt_tables());
    let digit_moduli = &qs[digit.clone()];
    // [x + h]_{Q_D} in coefficient form; h mod q_j = (q_j - 1)/2 for a digit prime q_j.
    let src: Vec<Vec<u64>> = poly.limbs()[digit.clone()]
        .iter()
        .zip(digit_moduli)
        .zip(&tables[digit.clone()])
        .map(|((limb, m), t)| {
            let mut a = limb.clone();
            t.inverse(&mut a);
            let h = m.value() >> 1;
            a.iter_mut().for_each(|x| *x = m.add(*x, h));
            a
        })
        .collect();
    let others: Vec<usize> = (0..=l).filter(|i| !digit.contains(i)).collect();
    let targets: Vec<(&Modulus, &NttTable)> = others
        .iter()
        .map(|&i| (&qs[i], &tables[i]))
        .chain(basis_p.moduli().iter().zip(basis_p.ntt_tables()))
        .collect();
    let mut out = if digit.len() == 1 {
        // One-prime digit: src[0] = [x + h]_{q_D} is already the conversion's `y`, and its
        // output for target r is `y mod r - h mod r` (one fused pass, then the NTT).
        let y = &src[0];
        targets
            .iter()
            .map(|(m, t)| {
                let one = m.shoup(1);
                let h = half_mod(digit_moduli, m);
                let mut a: Vec<u64> = y.iter().map(|&v| m.sub(m.mul_shoup(v, &one), h)).collect();
                t.forward(&mut a);
                a
            })
            .collect()
    } else {
        let from: Vec<u64> = digit_moduli.iter().map(Modulus::value).collect();
        let to: Vec<u64> = targets.iter().map(|(m, _)| m.value()).collect();
        let mut out = BaseConverter::new(&from, &to).convert_exact(&src);
        for (a, (m, t)) in out.iter_mut().zip(&targets) {
            let h = half_mod(digit_moduli, m);
            a.iter_mut().for_each(|x| *x = m.sub(*x, h));
            t.forward(a);
        }
        out
    };
    let p_limbs = out.split_off(others.len());
    let mut converted = out.into_iter();
    let q_limbs = (0..=l)
        .map(|i| {
            if digit.contains(&i) {
                poly.limbs()[i].clone()
            } else {
                converted
                    .next()
                    .expect("one converted limb per other prime")
            }
        })
        .collect();
    QpPoly::new(
        RnsPoly::from_limbs(basis, q_limbs, true).expect("limbs are reduced"),
        RnsPoly::from_limbs(basis_p, p_limbs, true).expect("converted limbs are reduced"),
    )
}

/// `(D - 1)/2 mod m` for `D` the (odd) product of `factors`.
fn half_mod(factors: &[Modulus], m: &Modulus) -> u64 {
    let d = factors
        .iter()
        .fold(1 % m.value(), |acc, f| m.mul(acc, f.value() % m.value()));
    m.mul(m.sub(d, 1 % m.value()), m.value().div_ceil(2))
}

/// Inner product of lifted digits with the key parts, divided by `P` (module docs, step 2):
/// returns `(kb, ka)` at the digits' level `l`, NTT form, over `params.basis_q()`.
///
/// `digits[j]` is used with `key.parts()[j]`; `digits` would normally come from
/// [`decompose`] (possibly with an automorphism applied to every part, for hoisting).
///
/// # Panics
/// If `digits` is empty, its length is not `digit_ranges(params, l).len()` (`l` the level of
/// the digits) or exceeds `key.dnum()`, or the digits are not all at the same level and in
/// NTT form.
pub fn apply_key(params: &CkksParams, digits: &[QpPoly], key: &KskKey) -> (RnsPoly, RnsPoly) {
    assert!(!digits.is_empty(), "no digits to switch");
    let l = digits[0].q().level();
    let expected = digit_ranges(params, l).len();
    assert_eq!(
        digits.len(),
        expected,
        "{} digits for a level-{l} polynomial with {expected} digits",
        digits.len()
    );
    assert!(
        digits.len() <= key.dnum(),
        "{} digits for a key with {} parts",
        digits.len(),
        key.dnum()
    );
    assert!(
        digits
            .iter()
            .all(|d| d.q().level() == l && d.q().is_ntt_form() && d.p().is_ntt_form()),
        "digits must share one level and be in NTT form"
    );
    let parts = &key.parts()[..digits.len()];
    let (basis_q, basis_p) = (params.basis_q(), params.basis_p());
    // Per prime: acc_b = sum_j d_j * b_j, acc_a = sum_j d_j * a_j with lazy reduction (exact
    // u128 sums of the products, reduced once per coefficient as hi * [2^64]_m + lo via two
    // Shoup products, which accept any u64). Each product is <= (m - 1)^2, so the sum fits
    // in u128 when digits.len() * (m - 1)^2 <= u128::MAX (>= 256 digits for 60-bit primes).
    let inner = |basis: &Arc<RnsBasis>, limbs: usize, pick: &dyn Fn(&QpPoly) -> &RnsPoly| {
        let n = params.n();
        let (mut out_b, mut out_a) = (Vec::with_capacity(limbs), Vec::with_capacity(limbs));
        let (mut acc_b, mut acc_a) = (vec![0u128; n], vec![0u128; n]);
        for (i, m) in basis.moduli()[..limbs].iter().enumerate() {
            let max = (m.value() - 1) as u128;
            assert!(
                (digits.len() as u128) <= u128::MAX / (max * max).max(1),
                "too many digits for lazy reduction"
            );
            acc_b.fill(0);
            acc_a.fill(0);
            for (d, (b, a)) in digits.iter().zip(parts) {
                let (dl, bl, al) = (
                    &pick(d).limbs()[i],
                    &pick(b).limbs()[i],
                    &pick(a).limbs()[i],
                );
                for ((xb, xa), (&x, (&yb, &ya))) in acc_b
                    .iter_mut()
                    .zip(acc_a.iter_mut())
                    .zip(dl.iter().zip(bl.iter().zip(al)))
                {
                    *xb += x as u128 * yb as u128;
                    *xa += x as u128 * ya as u128;
                }
            }
            let two64 = m.shoup(((1u128 << 64) % m.value() as u128) as u64);
            let one = m.shoup(1);
            let reduce = |acc: &[u128]| -> Vec<u64> {
                acc.iter()
                    .map(|&s| {
                        m.add(
                            m.mul_shoup((s >> 64) as u64, &two64),
                            m.mul_shoup(s as u64, &one),
                        )
                    })
                    .collect()
            };
            out_b.push(reduce(&acc_b));
            out_a.push(reduce(&acc_a));
        }
        let wrap = |limbs| RnsPoly::from_limbs(basis, limbs, true).expect("reduced limbs");
        (wrap(out_b), wrap(out_a))
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

    /// Predicted standard deviation of a coefficient of `kb + ka s - d s'` for a uniform `d`
    /// at `level` with the exact centered lift (module docs, "Noise"):
    /// `N sigma^2 sum_j Q_{D_j}^2 / (12 P^2) + N Pr[s_i != 0] / 12 + 1/12`.
    fn predicted_ks_std(params: &CkksParams) -> impl Fn(usize) -> f64 + '_ {
        move |level| {
            let n = params.n() as f64;
            let p: f64 = params.p().iter().map(|&p| p as f64).product();
            let sigma = crate::keys::ERROR_STD_DEV;
            let digits: f64 = digit_ranges(params, level)
                .into_iter()
                .map(|r| {
                    let qd: f64 = params.q()[r].iter().map(|&q| q as f64).product();
                    (qd / p).powi(2) / 12.0
                })
                .sum();
            let nonzero = params.hamming_weight().map_or(0.5, |h| h as f64 / n);
            (n * sigma * sigma * digits + n * nonzero / 12.0 + 1.0 / 12.0).sqrt()
        }
    }

    /// Relinearization-key switches at several levels: the coefficient noise
    /// `kb + ka s - d s'` is within 6 predicted standard deviations ([`predicted_ks_std`]),
    /// and the centered lift beats the non-centered one on seal_2020 (digit 0 = q_0 ~ P).
    /// Prints the measured numbers for the report.
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
                let noise = |lift| {
                    let (kb, ka) = apply_key(&params, &decompose_with(&params, &d, lift), rk.key());
                    ks_noise_log2(&sk, &d, &s2, &kb, &ka)
                };
                let (fast, plain) = (noise(Lift::CenteredFast), noise(Lift::Plain));
                let bound = (6.0 * predicted_ks_std(&params)(level)).log2();
                eprintln!(
                    "N=2^{} level {level}: relin-key switch noise ||.||_inf = 2^{centered:.2} \
                     (exact centered), 2^{fast:.2} (fast centered), 2^{plain:.2} \
                     (non-centered); bound 6 sigma = 2^{bound:.2}",
                    params.log_n()
                );
                assert!(centered < bound, "2^{centered} >= 2^{bound}");
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
                for (e, lift) in errs.iter_mut().zip([Lift::Exact, Lift::Plain]) {
                    let digits = decompose_with(&params, &d, lift);
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

    /// The exact centered lift against a CRT reference (u128, digits of up to 2 primes):
    /// every converted limb is `x_c mod r` for `x_c` the centered digit value, and the
    /// digit limbs are copied.
    #[test]
    fn exact_lift_matches_crt_reference() {
        let params = small_params(); // digits {60, 40}, {40, 40}, {40, 40}
        let level = params.max_level();
        let mut d = rng(13).uniform_poly(params.basis_q(), level);
        d.to_ntt();
        let mut d_coeff = d.clone();
        d_coeff.to_coeff();
        let qs = params.basis_q().moduli();
        for (dig, range) in decompose(&params, &d)
            .iter()
            .zip(digit_ranges(&params, level))
        {
            let (mut q, mut p) = (dig.q().clone(), dig.p().clone());
            q.to_coeff();
            p.to_coeff();
            let (m0, m1) = (&qs[range.start], &qs[range.start + 1]);
            let (q0, q1) = (m0.value() as u128, m1.value() as u128);
            let q0_inv = m1.inv(m0.value() % m1.value()).unwrap() as u128;
            let big_q = q0 * q1;
            for c in 0..params.n() {
                let (x0, x1) = (
                    d_coeff.limbs()[range.start][c] as u128,
                    d_coeff.limbs()[range.start + 1][c] as u128,
                );
                // Garner: x = x0 + q0 * ((x1 - x0) q0^{-1} mod q1), in [0, Q_D).
                let t = (x1 + q1 - x0 % q1) % q1 * q0_inv % q1;
                let x = x0 + q0 * t;
                let residue = |r: u64| {
                    let r = r as u128;
                    if x <= (big_q - 1) / 2 {
                        (x % r) as u64
                    } else {
                        ((r - (big_q - x) % r) % r) as u64
                    }
                };
                for (i, (limb, m)) in q.limbs().iter().zip(qs).enumerate() {
                    assert_eq!(limb[c], residue(m.value()), "q_{i} coeff {c}");
                }
                for (t, m) in params.basis_p().moduli().iter().enumerate() {
                    assert_eq!(p.limbs()[t][c], residue(m.value()), "p_{t} coeff {c}");
                }
            }
        }
    }

    /// Hoisting (Task 18): applying the automorphism to every lifted digit and then
    /// `apply_key` is bit-identical to switching the rotated polynomial, for `alpha = 1` and
    /// `alpha = 2` (with the exact lift; also with a level-clipped last digit), and for
    /// conjugation. With the fast centered lift and `alpha = 2` it is not.
    #[test]
    fn hoisted_digits_match_direct_switch() {
        let alpha1 = Arc::new(CkksParams::new(12, &[60, 40, 40, 40], &[60], 40, 4).unwrap());
        for (params, level) in [(alpha1, 3), (small_params(), 5), (small_params(), 4)] {
            let mut kg = KeyGenerator::new(Arc::clone(&params), rng(11)).unwrap();
            let mut gks = kg.galois_keys(&[5]);
            gks.insert(kg.conjugation_key());
            let mut d = rng(12).uniform_poly(params.basis_q(), level);
            d.to_ntt();
            let digits = decompose(&params, &d);
            for g in [
                galois_elt(params.n(), 5),
                crate::keys::conjugation_elt(params.n()),
            ] {
                let key = gks.get_elt(g).unwrap().key();
                let hoisted: Vec<QpPoly> = digits
                    .iter()
                    .map(|x| QpPoly::new(x.q().automorphism(g), x.p().automorphism(g)))
                    .collect();
                let a = apply_key(&params, &hoisted, key);
                let b = key_switch(&params, &d.automorphism(g), key);
                assert_eq!(a.0.limbs(), b.0.limbs(), "alpha {} g {g}", params.alpha());
                assert_eq!(a.1.limbs(), b.1.limbs(), "alpha {} g {g}", params.alpha());
                if params.alpha() > 1 {
                    let fast: Vec<QpPoly> = decompose_with(&params, &d, Lift::CenteredFast)
                        .iter()
                        .map(|x| QpPoly::new(x.q().automorphism(g), x.p().automorphism(g)))
                        .collect();
                    let direct = decompose_with(&params, &d.automorphism(g), Lift::CenteredFast);
                    let (fa, fb) = (
                        apply_key(&params, &fast, key),
                        apply_key(&params, &direct, key),
                    );
                    assert_ne!(fa.1.limbs(), fb.1.limbs(), "fast lift is not symmetric");
                }
            }
        }
    }

    /// `apply_key` rejects a digit count that does not match the level.
    #[test]
    #[should_panic(expected = "digits for a level-4 polynomial")]
    fn apply_key_checks_digit_count() {
        let params = small_params();
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(14)).unwrap();
        let rk = kg.relin_key();
        let mut d = rng(15).uniform_poly(params.basis_q(), 4);
        d.to_ntt();
        let digits = decompose(&params, &d);
        apply_key(&params, &digits[..2], rk.key());
    }

    /// The lazily reduced inner product of `apply_key` is bit-identical to a reference that
    /// reduces every product (`Modulus::mul` / `add`) followed by `mod_down`; worst-case
    /// inputs (all residues `m - 1`) included.
    #[test]
    fn apply_key_matches_reduced_reference() {
        let params = Arc::new(CkksParams::seal_2020());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(16)).unwrap();
        let rk = kg.relin_key();
        let reference = |digits: &[QpPoly]| {
            let parts = &rk.key().parts()[..digits.len()];
            let side = |pick: &dyn Fn(&QpPoly) -> &RnsPoly| {
                let basis = pick(&digits[0]).basis();
                let limbs = pick(&digits[0]).limbs().len();
                let (mut ob, mut oa) = (vec![], vec![]);
                for (i, m) in basis.moduli()[..limbs].iter().enumerate() {
                    let (mut b, mut a) = (vec![0u64; params.n()], vec![0u64; params.n()]);
                    for (d, (kb, ka)) in digits.iter().zip(parts) {
                        let (x, y, z) = (
                            &pick(d).limbs()[i],
                            &pick(kb).limbs()[i],
                            &pick(ka).limbs()[i],
                        );
                        for c in 0..params.n() {
                            b[c] = m.add(b[c], m.mul(x[c], y[c]));
                            a[c] = m.add(a[c], m.mul(x[c], z[c]));
                        }
                    }
                    ob.push(b);
                    oa.push(a);
                }
                let wrap = |l| RnsPoly::from_limbs(basis, l, true).unwrap();
                (wrap(ob), wrap(oa))
            };
            let (mut bq, mut aq) = side(&|x: &QpPoly| x.q());
            let (bp, ap) = side(&|x: &QpPoly| x.p());
            bq.mod_down(&bp);
            aq.mod_down(&ap);
            (bq, aq)
        };
        let level = params.max_level();
        let mut d = rng(17).uniform_poly(params.basis_q(), level);
        d.to_ntt();
        let random = decompose(&params, &d);
        let worst: Vec<QpPoly> = random
            .iter()
            .map(|x| {
                let max = |p: &RnsPoly| {
                    let limbs = p.basis().moduli()[..p.limbs().len()]
                        .iter()
                        .map(|m| vec![m.value() - 1; params.n()])
                        .collect();
                    RnsPoly::from_limbs(p.basis(), limbs, true).unwrap()
                };
                QpPoly::new(max(x.q()), max(x.p()))
            })
            .collect();
        for digits in [random, worst] {
            let (a, b) = (apply_key(&params, &digits, rk.key()), reference(&digits));
            assert_eq!(a.0.limbs(), b.0.limbs());
            assert_eq!(a.1.limbs(), b.1.limbs());
        }
    }
}
