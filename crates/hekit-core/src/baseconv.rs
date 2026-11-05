//! Fast RNS base conversion, rescaling by the last prime, and the special-prime lift and
//! division (`mod_up` / `mod_down`) used by hybrid key switching.
//!
//! # Fast base conversion
//!
//! [`BaseConverter::convert`] is SEAL 4.1's `BaseConverter::fast_convert_array`: for
//! `x` given by its residues `x_j = x mod q_j` over `Q = q_0 * ... * q_{k-1}`,
//!
//! ```text
//! out_t = sum_j [x_j * (Q/q_j)^{-1}]_{q_j} * (Q/q_j)   mod p_t
//! ```
//!
//! The sum (before reduction mod `p_t`) is an integer in `[0, k*Q)` congruent to `x` mod `Q`,
//! so it equals `[x]_Q + e*Q` with `0 <= e <= k - 1` (`[x]_Q` the representative in
//! `[0, Q)`; `e` depends on the coefficient but is the same for every target prime). The
//! conversion is therefore exact modulo `Q` and off by a small multiple of `Q` as an integer.
//!
//! # Representation of polynomials over `Q_l * P`
//!
//! An [`RnsPoly`] always holds a *prefix* `q_0, ..., q_l` of its basis. In hybrid key
//! switching at level `l < L`, the working modulus `Q_l * P` (`P = p_0 * ... * p_{k-1}`, the
//! special primes) is not a prefix of the ciphertext chain `q_0, ..., q_L`, so this module
//! represents an element of `R_{Q_l * P}` as a **pair of polynomials in the same form**:
//!
//! * the `Q` part: an `RnsPoly` over the ciphertext basis at level `l`;
//! * the `P` part: an `RnsPoly` over a separate *special basis* (`RnsBasis` of the special
//!   primes, all `k` limbs active).
//!
//! Both are ordinary `RnsPoly`s, so `add_assign`, `mul_assign`, `to_ntt` and `automorphism`
//! apply to each part unchanged, and the special primes keep their own NTT tables (needed for
//! the NTT-form inner product with the key and for NTT-form `mod_down`). The brief's raw
//! `special: &[u64]` parameters therefore become `special: &Arc<RnsBasis>` in
//! [`RnsPoly::mod_up`] / [`RnsPoly::mod_up_digit`] and `special: &RnsPoly` (the `P` part)
//! in [`RnsPoly::mod_down`]; the semantics are unchanged.
//!
//! # Use in hybrid key switching (Task 17)
//!
//! With `dnum` digits of `alpha` consecutive primes, a key switch of `c1` (NTT form, level
//! `l`) is:
//!
//! 1. for each digit `j` with primes `D_j = j*alpha .. min((j+1)*alpha, l+1)` (the last digit
//!    may be clipped by the level; skip digits starting above `l`):
//!    `(d_q, d_p) = c1.mod_up_digit_centered(D_j, &basis_p)` (NTT form in, NTT form out;
//!    the centered lift keeps the key-switch noise zero-mean, see
//!    [`RnsPoly::mod_up_digit_centered`]);
//! 2. multiply `(d_q, d_p)` part-wise by key part `j` restricted to `Q_l * P`: its `Q` part
//!    truncated to level `l` (key `Q` parts are stored at level `L`; drop limbs `l+1..=L`, or
//!    work on `limbs()` directly) and its `P` part unchanged, and accumulate;
//! 3. `acc_q.mod_down(&acc_p)` gives the result at level `l` in NTT form.
//!
//! Keys can be generated over a combined basis `q_0..q_L, p_0..p_{k-1}` (the samplers take a
//! prefix) and split into the two parts with [`RnsPoly::from_limbs`]. For hoisted rotations
//! (Task 18) the digits from step 1 are kept and `automorphism` is applied to both parts of
//! each digit before step 2.
//!
//! # Preconditions
//!
//! As in [`crate::poly`], mismatched forms or ring degrees between the two parts and special
//! primes that also occur in the ciphertext basis are programming errors; they are
//! `debug_assert`ed (a shared prime makes `P^{-1} mod q_i` undefined and panics in
//! `mod_down`). Invalid levels and digit ranges panic (`assert!`), as in
//! [`RnsPoly::drop_last_limb`].
//!
//! Precomputed constants (`(Q/q_j)^{-1}`, `q_L^{-1} mod q_i`, `P^{-1} mod q_i`, ...) are
//! recomputed on every call (`O(k^2 * l)` word operations, negligible next to the `O(N)` work
//! per limb); caching them per level is left to the performance pass.

use std::ops::Range;
use std::sync::Arc;

use crate::modulus::{Modulus, Shoup};
use crate::poly::RnsPoly;
use crate::rns::RnsBasis;

/// Fast (approximate) RNS base conversion from `{q_j}` to `{p_t}` (see the module docs).
#[derive(Debug, Clone)]
pub struct BaseConverter {
    from: Vec<Modulus>,
    to: Vec<Modulus>,
    /// `q_hat_inv[j] = (Q / q_j)^{-1} mod q_j`.
    q_hat_inv: Vec<u64>,
    /// `q_hat_mod_p[t][j] = (Q / q_j) mod p_t` (one row per target prime).
    q_hat_mod_p: Vec<Vec<u64>>,
}

impl BaseConverter {
    /// Precomputes the conversion from the moduli `from` to the moduli `to`.
    ///
    /// `to` may be empty and may share moduli with `from` (a shared modulus is converted
    /// exactly).
    ///
    /// # Panics
    /// If `from` is empty, a modulus is outside `2..2^60`, or the `from` moduli are not
    /// pairwise coprime (e.g. a duplicate).
    pub fn new(from: &[u64], to: &[u64]) -> BaseConverter {
        assert!(
            !from.is_empty(),
            "base conversion needs at least one source modulus"
        );
        let modulus = |&q: &u64| Modulus::new(q).unwrap_or_else(|e| panic!("{e}"));
        let from: Vec<Modulus> = from.iter().map(modulus).collect();
        let to: Vec<Modulus> = to.iter().map(modulus).collect();
        // (Q / q_j) mod m, as a product of the other source moduli reduced mod m.
        let q_hat_mod = |j: usize, m: &Modulus| {
            from.iter()
                .enumerate()
                .filter(|&(i, _)| i != j)
                .fold(1 % m.value(), |acc, (_, qi)| {
                    m.mul(acc, qi.value() % m.value())
                })
        };
        let q_hat_inv = from
            .iter()
            .enumerate()
            .map(|(j, m)| {
                m.inv(q_hat_mod(j, m))
                    .expect("source moduli of a base conversion must be pairwise coprime")
            })
            .collect();
        let q_hat_mod_p = to
            .iter()
            .map(|p| (0..from.len()).map(|j| q_hat_mod(j, p)).collect())
            .collect();
        BaseConverter {
            from,
            to,
            q_hat_inv,
            q_hat_mod_p,
        }
    }

    /// The precomputed `(Q / q_j)^{-1} mod q_j`, one per source modulus (in order). Exposed
    /// for backends that evaluate the conversion elsewhere (the CUDA kernels).
    pub fn q_hat_inv(&self) -> &[u64] {
        &self.q_hat_inv
    }

    /// The precomputed `(Q / q_j) mod p_t`: one row per target modulus `p_t`, one entry per
    /// source modulus `q_j`.
    pub fn q_hat_mod_to(&self) -> &[Vec<u64>] {
        &self.q_hat_mod_p
    }

    /// Converts residues over `from` (one slice per source modulus, in order, all of the
    /// same length, entries reduced) to residues over `to` (one vector per target modulus).
    /// Input and output are coefficient-wise, so this works on coefficient-form limbs.
    ///
    /// Output `t` is `([x]_Q + e*Q) mod p_t` with `0 <= e <= from.len() - 1` (module docs).
    ///
    /// # Panics
    /// If `input.len() != from.len()`.
    pub fn convert(&self, input: &[Vec<u64>]) -> Vec<Vec<u64>> {
        self.convert_inner(input, false)
    }

    /// [`BaseConverter::convert`] with the overflow `e` removed: output `t` is
    /// `[x]_Q mod p_t` (the representative in `[0, Q)`), as in HPS-style exact conversion.
    ///
    /// For one source prime the plain conversion is already exact (`e = 0`) and is used
    /// unchanged. For `k = from.len() >= 2`, `e = floor(sum_j y_j / q_j)` (`y_j` as in the
    /// module docs) is evaluated in `f64`; the sum equals `e + [x]_Q / Q`, so the result is
    /// exact unless `[x]_Q / Q` lies within about `k * 2^-52` of 0 or 1, where it may be off
    /// by `+-Q`.
    ///
    /// # Panics
    /// If `input.len() != from.len()`.
    pub fn convert_exact(&self, input: &[Vec<u64>]) -> Vec<Vec<u64>> {
        self.convert_inner(input, true)
    }

    fn convert_inner(&self, input: &[Vec<u64>], exact: bool) -> Vec<Vec<u64>> {
        assert_eq!(
            input.len(),
            self.from.len(),
            "one input limb per source modulus"
        );
        let n = input[0].len();
        // y_j = [x_j * (Q/q_j)^{-1}]_{q_j}
        let y: Vec<Vec<u64>> = input
            .iter()
            .zip(&self.from)
            .zip(&self.q_hat_inv)
            .map(|((x, m), &w)| {
                debug_assert_eq!(x.len(), n, "input limbs of different lengths");
                let s = m.shoup(w);
                x.iter()
                    .map(|&v| {
                        debug_assert!(v < m.value(), "input not reduced");
                        m.mul_shoup(v, &s)
                    })
                    .collect()
            })
            .collect();
        // e = floor(sum_j y_j / q_j) (exact conversion only). With one source prime
        // y_0 = x_0 < q_0, so e = 0 and the plain conversion is already exact; skip the f64
        // estimate there, which would round y_0 / q_0 up to 1.0 for y_0 within ~2^9 of a
        // 60-bit q_0 (and break SEAL's exact divide_and_round for one special prime).
        let overflow: Option<Vec<u64>> = (exact && self.from.len() > 1).then(|| {
            let inv: Vec<f64> = self.from.iter().map(|m| 1.0 / m.value() as f64).collect();
            (0..n)
                .map(|c| {
                    let f: f64 = y.iter().zip(&inv).map(|(yj, &w)| yj[c] as f64 * w).sum();
                    f.floor() as u64
                })
                .collect()
        });
        // out_t = sum_j y_j * (Q/q_j) - e * Q mod p_t; Shoup products accept unreduced
        // y_j < q_j.
        self.to
            .iter()
            .zip(&self.q_hat_mod_p)
            .map(|(p, row)| {
                let mut out = vec![0u64; n];
                for (yj, &w) in y.iter().zip(row) {
                    let s: Shoup = p.shoup(w);
                    out.iter_mut()
                        .zip(yj)
                        .for_each(|(o, &v)| *o = p.add(*o, p.mul_shoup(v, &s)));
                }
                if let Some(e) = &overflow {
                    let q_mod_p = self
                        .from
                        .iter()
                        .fold(1 % p.value(), |acc, m| p.mul(acc, m.value() % p.value()));
                    let s = p.shoup(q_mod_p);
                    out.iter_mut()
                        .zip(e)
                        .for_each(|(o, &v)| *o = p.sub(*o, p.mul_shoup(v, &s)));
                }
                out
            })
            .collect()
    }
}

/// `(D - 1)/2 mod m` for `D` the (odd) product of `factors`: `(D - 1) * 2^{-1}` with
/// `2^{-1} = (m + 1)/2` (`m` odd).
fn half_mod(factors: &[Modulus], m: &Modulus) -> u64 {
    let d = factors
        .iter()
        .fold(1 % m.value(), |acc, f| m.mul(acc, f.value() % m.value()));
    m.mul(m.sub(d, 1 % m.value()), m.value().div_ceil(2))
}

impl RnsPoly {
    /// Divides by the last prime `q_l` with rounding and drops it (level `l -> l - 1`):
    /// `a_i <- (a_i - [a_l + h]_{q_l} + h) * q_l^{-1} mod q_i`, `h = (q_l - 1) / 2`
    /// (SEAL's `divide_and_round_q_last_inplace`).
    ///
    /// The result is exactly `round(x / q_l)` coefficient-wise for the integer `x` in
    /// `[0, Q)` represented by the limbs (equivalently for its centered representative; no
    /// ties since `q_l` is odd). Works in either form; in NTT form the last limb is brought to
    /// coefficient form and re-transformed under each remaining prime
    /// (`divide_and_round_q_last_ntt_inplace`). The form is preserved.
    ///
    /// # Panics
    /// At level 0.
    pub fn rescale_by_last(&mut self) {
        let l = self.level();
        assert!(l >= 1, "cannot rescale at level 0");
        let basis = Arc::clone(self.basis());
        let (moduli, tables) = (basis.moduli(), basis.ntt_tables());
        let ntt = self.is_ntt_form();
        let q_last = moduli[l];
        let half = q_last.value() >> 1;
        let mut last = self.limbs()[l].clone();
        if ntt {
            tables[l].inverse(&mut last);
        }
        last.iter_mut().for_each(|x| *x = q_last.add(*x, half));
        self.drop_last_limb();
        for (i, limb) in self.limbs_mut().iter_mut().enumerate() {
            let m = &moduli[i];
            let one = m.shoup(1); // mul_shoup(v, 1) reduces any u64 v
            let half_i = m.mul_shoup(half, &one);
            let mut t: Vec<u64> = last
                .iter()
                .map(|&v| m.sub(m.mul_shoup(v, &one), half_i))
                .collect();
            if ntt {
                tables[i].forward(&mut t);
            }
            let inv = m.shoup(
                m.inv(q_last.value() % m.value())
                    .expect("basis primes are distinct"),
            );
            limb.iter_mut()
                .zip(&t)
                .for_each(|(a, &b)| *a = m.mul_shoup(m.sub(*a, b), &inv));
        }
    }

    /// The `P` part of the lift of `self` from `Q_l` to `Q_l * P` (fast base conversion of
    /// all limbs; `special` is the basis of the special primes `P`, all of whose primes are
    /// used). The `Q` part of the lift is `self` itself (exact), so `(self, self.mod_up(P))`
    /// is the lifted pair (module docs).
    ///
    /// Each output limb is `([x]_{Q_l} + u*Q_l) mod p_t` with `0 <= u <= l`. Accepts either
    /// form and returns the same form. Equivalent to `self.mod_up_digit(0..l + 1, special).1`.
    ///
    /// # Panics
    /// If `special` has a different ring degree.
    pub fn mod_up(&self, special: &Arc<RnsBasis>) -> RnsPoly {
        let (_, p_limbs) = self.convert_digit(0..self.level() + 1, special, false);
        RnsPoly::from_limbs(special, p_limbs, self.is_ntt_form())
            .expect("converted limbs are reduced")
    }

    /// Lifts the digit formed by the limbs `digit` (a range of prime indices within
    /// `0..=level`) to `Q_l * P`: returns `(q_part, p_part)` where `q_part` has the digit's
    /// limbs copied unchanged and every other `Q_l` limb, like every limb of `p_part`
    /// (over `special`, all primes), obtained by fast base conversion from the digit primes.
    ///
    /// For `x_D` the integer in `[0, Q_D)` given by the digit limbs (`Q_D` = product of the
    /// digit primes), every converted limb is `(x_D + u*Q_D) mod r` with
    /// `0 <= u <= digit.len() - 1` (the same `u` for all target primes `r` of a coefficient).
    /// Accepts either form and returns both parts in the same form.
    ///
    /// # Panics
    /// If `digit` is empty or not within `0..=level`, or `special` has a different ring
    /// degree.
    pub fn mod_up_digit(&self, digit: Range<usize>, special: &Arc<RnsBasis>) -> (RnsPoly, RnsPoly) {
        self.lift_digit(digit, special, false)
    }

    /// [`RnsPoly::mod_up_digit`] with a **centered** lift: the digit value is taken as its
    /// representative `x_c` in `[-(Q_D-1)/2, (Q_D-1)/2]` (`Q_D` is odd) instead of
    /// `[0, Q_D)`, so every converted limb is `(x_c + u*Q_D) mod r` with
    /// `0 <= u <= digit.len() - 1` (`u = 0`, an exact lift, for a one-prime digit). The digit
    /// limbs are copied unchanged, as in `mod_up_digit`.
    ///
    /// Implemented as the fast conversion of `[x + h]_{Q_D}` followed by subtracting
    /// `h mod r`, `h = (Q_D - 1)/2` (the same offset trick as [`RnsPoly::mod_down`]).
    ///
    /// Hybrid key switching uses this lift: the lifted digit `d_j` multiplies the key error
    /// `e_j`, and a non-centered `d_j` has mean `Q_D/2`, whose constant component concentrates
    /// noise in the slots near `zeta = 1` (seal_2020, `Q_D = q_0 ~ P`: key-switch error
    /// `2^-19.8` instead of `2^-22.7` times the scale).
    ///
    /// # Panics
    /// As [`RnsPoly::mod_up_digit`].
    pub fn mod_up_digit_centered(
        &self,
        digit: Range<usize>,
        special: &Arc<RnsBasis>,
    ) -> (RnsPoly, RnsPoly) {
        self.lift_digit(digit, special, true)
    }

    fn lift_digit(
        &self,
        digit: Range<usize>,
        special: &Arc<RnsBasis>,
        centered: bool,
    ) -> (RnsPoly, RnsPoly) {
        let (others, mut converted) = self.convert_digit(digit.clone(), special, centered);
        let p_limbs = converted.split_off(others.len());
        let mut converted = converted.into_iter();
        let q_limbs = (0..=self.level())
            .map(|i| {
                if digit.contains(&i) {
                    self.limbs()[i].clone()
                } else {
                    converted
                        .next()
                        .expect("one converted limb per other prime")
                }
            })
            .collect();
        let ntt = self.is_ntt_form();
        (
            RnsPoly::from_limbs(self.basis(), q_limbs, ntt).expect("limbs are reduced"),
            RnsPoly::from_limbs(special, p_limbs, ntt).expect("converted limbs are reduced"),
        )
    }

    /// Converts the limbs `digit` to every other `Q_l` prime (indices returned, increasing)
    /// followed by every prime of `special`, in the form of `self`. With `centered`, the
    /// digit value is lifted from its centered representative (see
    /// [`RnsPoly::mod_up_digit_centered`]).
    fn convert_digit(
        &self,
        digit: Range<usize>,
        special: &Arc<RnsBasis>,
        centered: bool,
    ) -> (Vec<usize>, Vec<Vec<u64>>) {
        let l = self.level();
        assert!(
            digit.start < digit.end && digit.end <= l + 1,
            "digit {digit:?} must be a non-empty range within 0..={l}"
        );
        let basis = self.basis();
        assert_eq!(special.n(), basis.n(), "special basis has a different N");
        let (qs, tables) = (&basis.moduli()[..=l], basis.ntt_tables());
        debug_assert!(
            special.moduli().iter().all(|p| !qs.contains(p)),
            "special primes must be disjoint from the Q primes"
        );
        let ntt = self.is_ntt_form();
        let mut src = self.limbs()[digit.clone()].to_vec();
        if ntt {
            src.iter_mut()
                .zip(&tables[digit.clone()])
                .for_each(|(a, t)| t.inverse(a));
        }
        let others: Vec<usize> = (0..=l).filter(|i| !digit.contains(i)).collect();
        let digit_moduli = &qs[digit];
        let from: Vec<u64> = digit_moduli.iter().map(Modulus::value).collect();
        let targets: Vec<&Modulus> = others
            .iter()
            .map(|&i| &qs[i])
            .chain(special.moduli())
            .collect();
        let to: Vec<u64> = targets.iter().map(|m| m.value()).collect();
        if centered {
            // [x + h]_{Q_D}, h = (Q_D - 1)/2; h mod q_j = (q_j - 1)/2 for a digit prime q_j.
            for (a, m) in src.iter_mut().zip(digit_moduli) {
                let h = m.value() >> 1;
                a.iter_mut().for_each(|x| *x = m.add(*x, h));
            }
        }
        let mut out = BaseConverter::new(&from, &to).convert(&src);
        if centered {
            for (a, m) in out.iter_mut().zip(&targets) {
                let h = half_mod(digit_moduli, m);
                a.iter_mut().for_each(|x| *x = m.sub(*x, h));
            }
        }
        if ntt {
            let targets = others
                .iter()
                .map(|&i| &tables[i])
                .chain(special.ntt_tables());
            out.iter_mut().zip(targets).for_each(|(a, t)| t.forward(a));
        }
        (others, out)
    }

    /// Divides the pair `(self, special)` representing `X` in `Q_l * P` by `P` with
    /// rounding, leaving the result in `self` (level unchanged):
    /// `x_i <- (x_i - (conv_{P->Q}([X + h]_P) - h)) * P^{-1} mod q_i`, `h = (P - 1) / 2`.
    ///
    /// `special` is the `P` part (its active limbs define `P`) and must be in the same form
    /// as `self`; the form is preserved. With `X` in `[0, Q_l * P)` the result is
    /// `floor((X + h) / P) = round(X / P)`: `[X + h]_P` is converted with
    /// [`BaseConverter::convert_exact`], so the rounding is exact: always for one special
    /// prime (bit-identical to SEAL's `divide_and_round_q_last`), and for `k >= 2` except
    /// when `X / P` lies within about `k * 2^-52` of a tie `n + 1/2`, where it may round to
    /// either neighbour (rounding error `<= 1/2 + k 2^-52`).
    ///
    /// (With the plain fast conversion the result would be `round(X / P) - e`,
    /// `0 <= e <= k - 1`; that biased `e` multiplies the secret key in encryption and key
    /// switching and, being a nonzero-mean constant, concentrates in the slots near
    /// `zeta = 1`: n16_train (`k = 4`) public-key encryption measured `2^-21` slot error
    /// instead of `2^-25.8`.)
    ///
    /// # Panics
    /// If a special prime divides some `q_i` of `self` (no `P^{-1} mod q_i`).
    pub fn mod_down(&mut self, special: &RnsPoly) {
        let ntt = self.is_ntt_form();
        debug_assert_eq!(
            ntt,
            special.is_ntt_form(),
            "Q and P parts in different forms"
        );
        let basis = Arc::clone(self.basis());
        debug_assert_eq!(
            basis.n(),
            special.basis().n(),
            "Q and P parts of different N"
        );
        let l = self.level();
        let (qs, tables) = (&basis.moduli()[..=l], basis.ntt_tables());
        let sp = special.basis();
        let ps = &sp.moduli()[..=special.level()];
        debug_assert!(
            ps.iter().all(|p| !qs.contains(p)),
            "special primes must be disjoint from the Q primes"
        );
        // [X + h]_P with h = (P - 1)/2; P is odd, so h = (p_t - 1)/2 mod p_t.
        let mut src = special.limbs().to_vec();
        for ((a, p), t) in src.iter_mut().zip(ps).zip(sp.ntt_tables()) {
            if ntt {
                t.inverse(a);
            }
            let h = p.value() >> 1;
            a.iter_mut().for_each(|x| *x = p.add(*x, h));
        }
        let from: Vec<u64> = ps.iter().map(Modulus::value).collect();
        let to: Vec<u64> = qs.iter().map(Modulus::value).collect();
        let converted = BaseConverter::new(&from, &to).convert_exact(&src);
        for (i, (limb, mut t)) in self.limbs_mut().iter_mut().zip(converted).enumerate() {
            let m = &qs[i];
            let p_mod = ps
                .iter()
                .fold(1 % m.value(), |acc, p| m.mul(acc, p.value() % m.value()));
            let p_inv = m
                .inv(p_mod)
                .expect("special primes must be coprime to the Q primes");
            // h mod q_i = (P - 1) * 2^{-1}.
            let half = half_mod(ps, m);
            t.iter_mut().for_each(|x| *x = m.sub(*x, half));
            if ntt {
                tables[i].forward(&mut t);
            }
            let s = m.shoup(p_inv);
            limb.iter_mut()
                .zip(&t)
                .for_each(|(a, &b)| *a = m.mul_shoup(m.sub(*a, b), &s));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primes::{ntt_primes, ntt_primes_skip};
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn values(ms: &[Modulus]) -> Vec<u64> {
        ms.iter().map(Modulus::value).collect()
    }

    fn product(ms: &[u64]) -> u128 {
        ms.iter().map(|&m| m as u128).product()
    }

    /// CRT reconstruction by Garner's method; the product of `moduli` must be `< 2^127`.
    fn crt(residues: &[u64], moduli: &[u64]) -> u128 {
        let (mut x, mut big_m) = (0u128, 1u128);
        for (&r, &m) in residues.iter().zip(moduli) {
            let md = Modulus::new(m).unwrap();
            let xm = (x % m as u128) as u64;
            let mm = (big_m % m as u128) as u64;
            let t = md.mul(md.sub(r, xm), md.inv(mm).unwrap());
            x += big_m * t as u128;
            big_m *= m as u128;
        }
        x
    }

    fn rem_i(x: i128, q: u64) -> u64 {
        x.rem_euclid(q as i128) as u64
    }

    fn rand_u128(rng: &mut ChaCha20Rng, bound: u128) -> u128 {
        (((rng.next_u64() as u128) << 64) | rng.next_u64() as u128) % bound
    }

    /// Polynomial over `b` whose coefficient `c` is the integer `xs[c]` (`u128`, any size).
    fn poly_u128(b: &Arc<RnsBasis>, level: usize, xs: &[u128]) -> RnsPoly {
        let limbs = b.moduli()[..=level]
            .iter()
            .map(|m| xs.iter().map(|&x| (x % m.value() as u128) as u64).collect())
            .collect();
        RnsPoly::from_limbs(b, limbs, false).unwrap()
    }

    fn poly_i128(b: &Arc<RnsBasis>, level: usize, xs: &[i128]) -> RnsPoly {
        let limbs = b.moduli()[..=level]
            .iter()
            .map(|m| xs.iter().map(|&x| rem_i(x, m.value())).collect())
            .collect();
        RnsPoly::from_limbs(b, limbs, false).unwrap()
    }

    fn rand_poly(b: &Arc<RnsBasis>, level: usize, rng: &mut ChaCha20Rng) -> RnsPoly {
        let limbs = b.moduli()[..=level]
            .iter()
            .map(|m| (0..b.n()).map(|_| rng.next_u64() % m.value()).collect())
            .collect();
        RnsPoly::from_limbs(b, limbs, false).unwrap()
    }

    fn ntt_of(p: &RnsPoly) -> RnsPoly {
        let mut f = p.clone();
        f.to_ntt();
        f
    }

    fn coeff_of(p: &RnsPoly) -> RnsPoly {
        let mut c = p.clone();
        c.to_coeff();
        c
    }

    /// Multiplies limb `i` by `scalars[i]` (already reduced).
    fn mul_limbwise(p: &RnsPoly, scalars: &[u64]) -> RnsPoly {
        let limbs = p
            .limbs()
            .iter()
            .zip(p.basis().moduli())
            .zip(scalars)
            .map(|((l, m), &s)| l.iter().map(|&x| m.mul(x, s)).collect())
            .collect();
        RnsPoly::from_limbs(p.basis(), limbs, p.is_ntt_form()).unwrap()
    }

    /// `prod(ps) mod q` for each `q` in `qs`.
    fn prod_mod(ps: &[u64], qs: &[Modulus]) -> Vec<u64> {
        qs.iter()
            .map(|m| ps.iter().fold(1, |acc, &p| m.mul(acc, p % m.value())))
            .collect()
    }

    // ---------------------------------------------------------------- BaseConverter

    /// Brief: small `|x| < 2^20` over a 2-prime basis converted to another 2-prime basis and
    /// recovered there by CRT. The approximate conversion is exact modulo `Q_from`; as an
    /// integer it is `[x]_Q + e*Q` with `e in {0, 1}` (`k = 2`).
    #[test]
    fn convert_small_values_between_two_prime_bases() {
        let from = ntt_primes(30, 16, 2).unwrap();
        let to = ntt_primes(45, 16, 2).unwrap();
        let q = product(&from);
        assert!(product(&to) > 2 * q);
        let bc = BaseConverter::new(&from, &to);
        let mut rng = ChaCha20Rng::seed_from_u64(1301);
        let bound = 1i128 << 20;
        let mut xs: Vec<i128> = vec![0, 1, -1, bound - 1, -(bound - 1), 12345, -12345];
        xs.extend(
            (0..2000).map(|_| rand_u128(&mut rng, 2 * bound as u128 - 1) as i128 - bound + 1),
        );
        let input: Vec<Vec<u64>> = from
            .iter()
            .map(|&qj| xs.iter().map(|&x| rem_i(x, qj)).collect())
            .collect();
        let out = bc.convert(&input);
        assert_eq!(out.len(), 2);
        let mut seen_e = [0usize; 2];
        for (c, &x) in xs.iter().enumerate() {
            let r = crt(&[out[0][c], out[1][c]], &to);
            let xq = x.rem_euclid(q as i128) as u128;
            assert_eq!(r % q, xq, "x = {x}: not exact modulo Q_from");
            let e = (r / q) as usize;
            assert!(e < 2, "x = {x}: e = {e} exceeds k - 1");
            seen_e[e] += 1;
        }
        // x = 0 has all-zero residues, so the sum is exactly 0.
        assert_eq!(crt(&[out[0][0], out[1][0]], &to), 0);
        eprintln!(
            "convert e distribution over {} values: {seen_e:?}",
            xs.len()
        );
    }

    /// The exposed constants are `(Q/q_j)^{-1} mod q_j` and `(Q/q_j) mod p_t`.
    #[test]
    fn exposed_constants() {
        let from = ntt_primes(30, 16, 3).unwrap();
        let to = ntt_primes(45, 16, 2).unwrap();
        let bc = BaseConverter::new(&from, &to);
        let q_hat = |j: usize, m: u64| {
            from.iter()
                .enumerate()
                .filter(|&(i, _)| i != j)
                .fold(1u128, |acc, (_, &q)| acc * q as u128 % m as u128) as u64
        };
        assert_eq!(bc.q_hat_inv().len(), 3);
        for (j, (&qj, &inv)) in from.iter().zip(bc.q_hat_inv()).enumerate() {
            assert_eq!(q_hat(j, qj) as u128 * inv as u128 % qj as u128, 1);
        }
        assert_eq!(bc.q_hat_mod_to().len(), 2);
        for (row, &p) in bc.q_hat_mod_to().iter().zip(&to) {
            let want: Vec<u64> = (0..3).map(|j| q_hat(j, p)).collect();
            assert_eq!(row, &want);
        }
    }

    /// `convert_exact` returns `[x]_Q` itself (no overflow multiple) for random `x in [0, Q)`
    /// over 1, 2 and 3 source primes; for `k >= 2`, within `2^-40 * Q` of 0 or `Q` it may be
    /// off by `Q`, while `k = 1` is always exact.
    #[test]
    fn convert_exact_removes_the_overflow() {
        let to = ntt_primes(60, 16, 2).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(1310);
        for k in [1usize, 2, 3] {
            let from = ntt_primes(30, 16, k).unwrap();
            let q = product(&from);
            let pt = product(&to);
            assert!(pt > 2 * q);
            let bc = BaseConverter::new(&from, &to);
            let mut xs: Vec<u128> = vec![0, 1, q / 2, q / 2 + 1, q - 1, q - 2];
            xs.extend((0..2000).map(|_| rand_u128(&mut rng, q)));
            let input: Vec<Vec<u64>> = from
                .iter()
                .map(|&qj| xs.iter().map(|&x| (x % qj as u128) as u64).collect())
                .collect();
            let out = bc.convert_exact(&input);
            let mut edges = 0;
            for (c, &x) in xs.iter().enumerate() {
                let r = crt(&[out[0][c], out[1][c]], &to);
                if k == 1 {
                    assert_eq!(r, x, "k=1 must be exact");
                } else if r != x {
                    let edge = q >> 40;
                    let ok = (x < edge && r == x + q) || (x > q - edge && r == pt - (q - x));
                    assert!(ok, "k={k} x={x}: got {r}");
                    edges += 1;
                }
            }
            assert_eq!(crt(&[out[0][0], out[1][0]], &to), 0);
            eprintln!("convert_exact k={k}: {edges} edge values off by Q");
        }
    }

    /// The output is exactly SEAL's `fast_convert_array` formula, evaluated here with exact
    /// `u128` arithmetic, including the worst case `x_j = q_j - 1` and `x_j = 0`.
    #[test]
    fn convert_matches_fast_convert_formula() {
        let from = ntt_primes(20, 16, 3).unwrap();
        let mut to = ntt_primes(50, 16, 2).unwrap();
        to.push(from[1]); // a shared modulus converts exactly
        to.push(ntt_primes(21, 16, 1).unwrap()[0]);
        let q = product(&from);
        let bc = BaseConverter::new(&from, &to);
        let mut rng = ChaCha20Rng::seed_from_u64(1302);
        let mut input: Vec<Vec<u64>> = from
            .iter()
            .map(|&qj| {
                let mut v = vec![0, qj - 1];
                v.extend((0..500).map(|_| rng.next_u64() % qj));
                v
            })
            .collect();
        input[0][0] = 0;
        let out = bc.convert(&input);
        assert_eq!(out.len(), to.len());
        for c in 0..input[0].len() {
            let mut sum = 0u128;
            for (j, &qj) in from.iter().enumerate() {
                let m = Modulus::new(qj).unwrap();
                let hat = q / qj as u128;
                let hat_inv = m.inv((hat % qj as u128) as u64).unwrap();
                sum += m.mul(input[j][c], hat_inv) as u128 * hat;
            }
            assert!(sum < 3 * q);
            let x = crt(&[input[0][c], input[1][c], input[2][c]], &from);
            assert_eq!(sum % q, x);
            for (t, &p) in to.iter().enumerate() {
                assert_eq!(out[t][c] as u128, sum % p as u128, "c={c} t={t}");
            }
            assert_eq!(out[2][c], input[1][c]);
        }
        assert!(BaseConverter::new(&from, &[]).convert(&input).is_empty());
    }

    #[test]
    #[should_panic]
    fn converter_rejects_duplicate_source_moduli() {
        let ps = ntt_primes(30, 16, 2).unwrap();
        let _ = BaseConverter::new(&[ps[0], ps[1], ps[0]], &[ps[1]]);
    }

    // ---------------------------------------------------------------- rescale_by_last

    fn round_div(x: i128, q: u64) -> i128 {
        (x + (q as i128 - 1) / 2).div_euclid(q as i128)
    }

    /// Brief: `rescale_by_last == round(x / q_L)` for `|x| < 2^100` (built with `i128`,
    /// negative values included), in coefficient and NTT form, twice in a row.
    #[test]
    fn rescale_by_last_rounds_large_signed_values() {
        let n = 64;
        let mut qs = ntt_primes(60, n, 1).unwrap();
        qs.extend(ntt_primes(45, n, 3).unwrap());
        let b = RnsBasis::new(n, &qs).unwrap();
        let q_last = qs[3] as i128;
        let bound = 1i128 << 100;
        let mut rng = ChaCha20Rng::seed_from_u64(1303);
        for trial in 0..4 {
            let mut xs: Vec<i128> = if trial == 0 {
                let m = 987_654_321_987i128;
                let h = (q_last - 1) / 2;
                vec![
                    0,
                    1,
                    -1,
                    bound - 1,
                    -(bound - 1),
                    m * q_last + h,     // fraction just below 1/2: rounds down
                    m * q_last + h + 1, // just above 1/2: rounds up
                    -m * q_last + h,
                    -m * q_last + h + 1,
                    -m * q_last - h,     // -(m + 1/2 - 1/(2q)): rounds to -m
                    -m * q_last - h - 1, // rounds to -m - 1
                    q_last,
                    -q_last,
                ]
            } else {
                Vec::new()
            };
            while xs.len() < n {
                xs.push(rand_u128(&mut rng, 2 * bound as u128 - 1) as i128 - bound + 1);
            }
            let a = poly_i128(&b, 3, &xs);
            let want1: Vec<i128> = xs.iter().map(|&x| round_div(x, qs[3])).collect();
            let want2: Vec<i128> = want1.iter().map(|&x| round_div(x, qs[2])).collect();
            for ntt in [false, true] {
                let mut r = if ntt { ntt_of(&a) } else { a.clone() };
                r.rescale_by_last();
                assert_eq!(r.level(), 2);
                assert_eq!(r.is_ntt_form(), ntt);
                assert_eq!(
                    coeff_of(&r).limbs(),
                    poly_i128(&b, 2, &want1).limbs(),
                    "ntt={ntt}"
                );
                r.rescale_by_last();
                assert_eq!(r.level(), 1);
                assert_eq!(
                    coeff_of(&r).limbs(),
                    poly_i128(&b, 1, &want2).limbs(),
                    "ntt={ntt}"
                );
            }
        }
    }

    /// NTT-form rescale equals coefficient-form rescale on full-range random inputs and on the
    /// worst cases (all `q_i - 1`, i.e. `x = -1`, and all zero).
    #[test]
    fn rescale_ntt_form_matches_coeff_form() {
        let n = 1024;
        let mut qs = ntt_primes(60, n, 1).unwrap();
        qs.extend(ntt_primes(40, n, 3).unwrap());
        let b = RnsBasis::new(n, &qs).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(1304);
        for level in [1, 3] {
            for _ in 0..3 {
                let a = rand_poly(&b, level, &mut rng);
                let mut c = a.clone();
                c.rescale_by_last();
                let mut f = ntt_of(&a);
                f.rescale_by_last();
                assert!(f.is_ntt_form());
                assert_eq!(coeff_of(&f).limbs(), c.limbs(), "level={level}");
            }
        }
        let minus_one: Vec<Vec<u64>> = qs.iter().map(|&q| vec![q - 1; n]).collect();
        let a = RnsPoly::from_limbs(&b, minus_one, false).unwrap();
        for ntt in [false, true] {
            let mut r = if ntt { ntt_of(&a) } else { a.clone() };
            r.rescale_by_last();
            assert!(
                coeff_of(&r)
                    .limbs()
                    .iter()
                    .all(|l| l.iter().all(|&x| x == 0))
            );
            let mut z = RnsPoly::zero(&b, 3);
            if ntt {
                z.to_ntt();
            }
            z.rescale_by_last();
            assert!(z.limbs().iter().all(|l| l.iter().all(|&x| x == 0)));
        }
    }

    #[test]
    #[should_panic]
    fn rescale_at_level_zero_panics() {
        let b = RnsBasis::new(16, &ntt_primes(30, 16, 2).unwrap()).unwrap();
        RnsPoly::zero(&b, 0).rescale_by_last();
    }

    // ---------------------------------------------------------------- mod_up

    /// `mod_up` over a level below the basis maximum: the `P` limbs recover
    /// `[x]_Q + u*Q`, `0 <= u <= l`, and the NTT-form path agrees with the coefficient path.
    #[test]
    fn mod_up_lifts_with_bounded_multiple_of_q() {
        let n = 16;
        let b = RnsBasis::new(n, &ntt_primes(20, n, 4).unwrap()).unwrap();
        let ps = ntt_primes(45, n, 2).unwrap();
        let sp = RnsBasis::new(n, &ps).unwrap();
        let level = 2;
        let qs = values(&b.moduli()[..=level]);
        let q = product(&qs);
        assert!(product(&ps) > 3 * q);
        let mut rng = ChaCha20Rng::seed_from_u64(1305);
        for trial in 0..20 {
            let mut xs: Vec<u128> = (0..n).map(|_| rand_u128(&mut rng, q)).collect();
            if trial == 0 {
                xs[0] = 0;
                xs[1] = q - 1;
            }
            let a = poly_u128(&b, level, &xs);
            let up = a.mod_up(&sp);
            assert_eq!(up.level(), 1);
            assert!(Arc::ptr_eq(up.basis(), &sp));
            assert!(!up.is_ntt_form());
            for (c, &x) in xs.iter().enumerate() {
                let r = crt(&[up.limbs()[0][c], up.limbs()[1][c]], &ps);
                assert_eq!(r % q, x);
                assert!(r / q <= level as u128, "u = {}", r / q);
            }
            if trial == 0 {
                assert_eq!(crt(&[up.limbs()[0][0], up.limbs()[1][0]], &ps), 0);
            }
            let up_ntt = ntt_of(&a).mod_up(&sp);
            assert!(up_ntt.is_ntt_form());
            assert_eq!(coeff_of(&up_ntt).limbs(), up.limbs());
        }
    }

    /// `mod_up_digit`: digit limbs are copied (in the input form), the other `Q` limbs and
    /// the `P` limbs all hold `x_D + u*Q_D` for one `u in {0, 1}` per coefficient.
    #[test]
    fn mod_up_digit_copies_digit_and_converts_the_rest() {
        let n = 16;
        let qs = ntt_primes(20, n, 4).unwrap();
        let b = RnsBasis::new(n, &qs).unwrap();
        let p = ntt_primes(50, n, 1).unwrap();
        let sp = RnsBasis::new(n, &p).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(1306);
        for _ in 0..20 {
            let a = rand_poly(&b, 3, &mut rng);
            let (dq, dp) = a.mod_up_digit(1..3, &sp);
            assert_eq!(dq.level(), 3);
            assert!(Arc::ptr_eq(dq.basis(), &b));
            assert_eq!(dp.level(), 0);
            assert_eq!(dq.limbs()[1], a.limbs()[1]);
            assert_eq!(dq.limbs()[2], a.limbs()[2]);
            let q_d = qs[1] as u128 * qs[2] as u128;
            for c in 0..n {
                let x_d = crt(&[a.limbs()[1][c], a.limbs()[2][c]], &qs[1..3]);
                let u = (0..2u128)
                    .find(|&u| (x_d + u * q_d) % p[0] as u128 == dp.limbs()[0][c] as u128)
                    .expect("P limb is x_D + u*Q_D with u in {0, 1}");
                let v = x_d + u * q_d;
                assert_eq!(dq.limbs()[0][c] as u128, v % qs[0] as u128);
                assert_eq!(dq.limbs()[3][c] as u128, v % qs[3] as u128);
            }
            // NTT form in, NTT form out; digit limbs copied as-is.
            let fa = ntt_of(&a);
            let (fq, fp) = fa.mod_up_digit(1..3, &sp);
            assert!(fq.is_ntt_form() && fp.is_ntt_form());
            assert_eq!(fq.limbs()[1], fa.limbs()[1]);
            assert_eq!(coeff_of(&fq).limbs(), dq.limbs());
            assert_eq!(coeff_of(&fp).limbs(), dp.limbs());
            // The whole basis as one digit: Q part is self, P part is mod_up.
            let (wq, wp) = a.mod_up_digit(0..4, &sp);
            assert_eq!(wq.limbs(), a.limbs());
            assert_eq!(wp.limbs(), a.mod_up(&sp).limbs());
        }
    }

    /// Centered representative of `x in [0, q)` in `[-(q-1)/2, (q-1)/2]` (`q` odd).
    fn center(x: u128, q: u128) -> i128 {
        if x > q / 2 {
            x as i128 - q as i128
        } else {
            x as i128
        }
    }

    /// `mod_up_digit_centered`: digit limbs are copied, every converted limb holds
    /// `x_c + u*Q_D` with `x_c` the centered representative of the digit value and
    /// `0 <= u <= digit.len() - 1` (the same `u` for every target prime); a one-prime digit
    /// is lifted exactly (`u = 0`). Boundary values `0, (Q_D-1)/2, (Q_D+1)/2, Q_D-1`
    /// included; NTT form agrees with coefficient form.
    #[test]
    fn mod_up_digit_centered_lifts_the_centered_representative() {
        let n = 16;
        let qs = ntt_primes(20, n, 4).unwrap();
        let b = RnsBasis::new(n, &qs).unwrap();
        let p = ntt_primes(50, n, 2).unwrap();
        let sp = RnsBasis::new(n, &p).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(1307);
        for (digit, trial) in [(1..3usize, 0), (1..3, 1), (2..3, 0), (2..3, 1), (0..4, 0)]
            .into_iter()
            .chain((0..10).map(|t| (1..3, t + 2)))
        {
            let q_d: u128 = qs[digit.clone()].iter().map(|&q| q as u128).product();
            let mut a = rand_poly(&b, 3, &mut rng);
            if trial == 0 {
                // Boundary digit values in the first coefficients.
                let mut xs: Vec<u128> = (0..n as u128).map(|c| c * 977 % q_d).collect();
                xs[..4].copy_from_slice(&[0, (q_d - 1) / 2, (q_d + 1) / 2, q_d - 1]);
                let limbs = (0..4)
                    .map(|i| {
                        if digit.contains(&i) {
                            xs.iter().map(|&x| (x % qs[i] as u128) as u64).collect()
                        } else {
                            a.limbs()[i].clone()
                        }
                    })
                    .collect();
                a = RnsPoly::from_limbs(&b, limbs, false).unwrap();
            }
            let (dq, dp) = a.mod_up_digit_centered(digit.clone(), &sp);
            assert_eq!((dq.level(), dp.level()), (3, 1));
            assert!(Arc::ptr_eq(dq.basis(), &b) && Arc::ptr_eq(dp.basis(), &sp));
            assert!(!dq.is_ntt_form() && !dp.is_ntt_form());
            for i in digit.clone() {
                assert_eq!(dq.limbs()[i], a.limbs()[i]);
            }
            let digit_q: Vec<u64> = qs[digit.clone()].to_vec();
            for c in 0..n {
                let res: Vec<u64> = digit.clone().map(|i| a.limbs()[i][c]).collect();
                let x_c = center(crt(&res, &digit_q), q_d);
                // Recover the lifted integer from the two P limbs (P > alpha * Q_D).
                let lifted = crt(&[dp.limbs()[0][c], dp.limbs()[1][c]], &p);
                let lifted = center(lifted, product(&p));
                let u = lifted - x_c;
                assert_eq!(u % q_d as i128, 0, "coefficient {c}: not x_c + u*Q_D");
                let u = u / q_d as i128;
                assert!(
                    (0..digit.len() as i128).contains(&u),
                    "coefficient {c}: u = {u}"
                );
                if digit.len() == 1 {
                    assert_eq!(u, 0);
                }
                for i in (0..4).filter(|i| !digit.contains(i)) {
                    assert_eq!(dq.limbs()[i][c], rem_i(lifted, qs[i]), "limb {i}");
                }
                if trial == 0 && c < 4 {
                    let want = [0, (q_d as i128 - 1) / 2, -(q_d as i128 - 1) / 2, -1][c];
                    assert_eq!(x_c, want);
                }
            }
            let fa = ntt_of(&a);
            let (fq, fp) = fa.mod_up_digit_centered(digit.clone(), &sp);
            assert!(fq.is_ntt_form() && fp.is_ntt_form());
            assert_eq!(coeff_of(&fq).limbs(), dq.limbs());
            assert_eq!(coeff_of(&fp).limbs(), dp.limbs());
        }
    }

    #[test]
    #[should_panic]
    fn mod_up_digit_centered_rejects_range_above_level() {
        let b = RnsBasis::new(16, &ntt_primes(20, 16, 3).unwrap()).unwrap();
        let sp = RnsBasis::new(16, &ntt_primes(40, 16, 1).unwrap()).unwrap();
        let _ = RnsPoly::zero(&b, 1).mod_up_digit_centered(1..3, &sp);
    }

    #[test]
    #[should_panic]
    fn mod_up_digit_rejects_empty_range() {
        let b = RnsBasis::new(16, &ntt_primes(20, 16, 2).unwrap()).unwrap();
        let sp = RnsBasis::new(16, &ntt_primes(40, 16, 1).unwrap()).unwrap();
        let _ = RnsPoly::zero(&b, 1).mod_up_digit(1..1, &sp);
    }

    #[test]
    #[should_panic]
    fn mod_up_digit_rejects_range_above_level() {
        let b = RnsBasis::new(16, &ntt_primes(20, 16, 3).unwrap()).unwrap();
        let sp = RnsBasis::new(16, &ntt_primes(40, 16, 1).unwrap()).unwrap();
        let _ = RnsPoly::zero(&b, 1).mod_up_digit(1..3, &sp);
    }

    // ---------------------------------------------------------------- mod_down

    /// `mod_down` on random `X in [0, Q_l * P)` (exact `u128` reference): the result is
    /// exactly `floor((X + (P-1)/2) / P)`: always for `k = 1` (SEAL's `divide_and_round`;
    /// boundary values `[X + h]_P = 0` and `P - 1` included), and for `k = 2` except possibly
    /// off by one when `[X + h]_P / P` is within `2^-40` of 0 or 1 (a rounding tie); NTT form
    /// agrees with coefficient form. Level 1 of a 3-prime basis.
    #[test]
    fn mod_down_rounds_exactly() {
        let n = 16;
        let qs_all = ntt_primes(28, n, 3).unwrap();
        let b = RnsBasis::new(n, &qs_all).unwrap();
        let level = 1;
        let qs = &qs_all[..=level];
        let q = product(qs);
        let mut rng = ChaCha20Rng::seed_from_u64(1307);
        for k in [1usize, 2] {
            let ps = ntt_primes_skip(28, n, 3, k).unwrap();
            let sp = RnsBasis::new(n, &ps).unwrap();
            let p = product(&ps);
            let h = (p - 1) / 2;
            let mut ties = 0usize;
            for trial in 0..50 {
                let mut xs: Vec<u128> = (0..n).map(|_| rand_u128(&mut rng, q * p)).collect();
                if trial == 0 {
                    let m = q / 3;
                    xs[..9].copy_from_slice(&[
                        0,
                        q * p - 1,
                        h,
                        h + 1,
                        p - 1,
                        p,
                        m * p + h,
                        m * p + h + 1,
                        (q - 1) * p + h + 1, // rounds up to Q = 0 mod Q
                    ]);
                }
                let xq = poly_u128(&b, level, &xs);
                let xp = poly_u128(&sp, k - 1, &xs);
                let mut r = xq.clone();
                r.mod_down(&xp);
                assert_eq!(r.level(), level);
                assert!(!r.is_ntt_form());
                for (c, &x) in xs.iter().enumerate() {
                    let got = crt(&[r.limbs()[0][c], r.limbs()[1][c]], qs);
                    let want = ((x + h) / p) % q;
                    if k == 1 {
                        assert_eq!(got, want, "k=1 must round exactly: x={x}");
                    } else if got != want {
                        let f = (x + h) % p;
                        let edge = p >> 40;
                        let near_tie = (f < edge && (want + q - 1) % q == got)
                            || (f > p - edge && (want + 1) % q == got);
                        assert!(near_tie, "k={k} x={x}: got {got}, want {want}");
                        ties += 1;
                    }
                }
                let mut f = ntt_of(&xq);
                f.mod_down(&ntt_of(&xp));
                assert!(f.is_ntt_form());
                assert_eq!(coeff_of(&f).limbs(), r.limbs(), "k={k}");
            }
            eprintln!("mod_down k={k}: {ties} tie values rounded the other way");
        }
    }

    /// One 60-bit special prime (the seal_2020 regime, where `f64(y) * f64(1/p)` can round
    /// to 1.0 for `y` near `p`): `convert_exact` from one prime is the identity and
    /// `mod_down` is exactly SEAL's `divide_and_round`, including every `[X + h]_P` in
    /// `[P - 1024, P)` and `[0, 1024)`.
    #[test]
    fn one_60_bit_special_prime_rounds_exactly() {
        let n = 1024;
        let p = ntt_primes(60, n, 1).unwrap()[0] as u128;
        let to = ntt_primes_skip(60, n, 1, 2).unwrap();
        let bc = BaseConverter::new(&[p as u64], &to);
        let ys: Vec<u64> = (0..n as u64).flat_map(|t| [p as u64 - 1 - t, t]).collect();
        let out = bc.convert_exact(&[ys.clone()]);
        for (c, &y) in ys.iter().enumerate() {
            assert_eq!(crt(&[out[0][c], out[1][c]], &to), y as u128, "y = {y}");
        }

        let qs = ntt_primes(30, n, 2).unwrap();
        let b = RnsBasis::new(n, &qs).unwrap();
        let sp = RnsBasis::new(n, &[p as u64]).unwrap();
        let q = product(&qs);
        let h = (p - 1) / 2;
        let mut rng = ChaCha20Rng::seed_from_u64(1311);
        for side in [0u128, 1] {
            // [X + h]_P = P - 1 - t (side 0) or t (side 1), X = m P + ([X + h]_P - h mod P).
            let xs: Vec<u128> = (0..n as u128)
                .map(|t| {
                    let f = if side == 0 { p - 1 - t } else { t };
                    rand_u128(&mut rng, q) * p + (f + p - h) % p
                })
                .collect();
            let xq = poly_u128(&b, 1, &xs);
            let xp = poly_u128(&sp, 0, &xs);
            for ntt in [false, true] {
                let (mut r, xp) = if ntt {
                    (ntt_of(&xq), ntt_of(&xp))
                } else {
                    (xq.clone(), xp.clone())
                };
                r.mod_down(&xp);
                let r = coeff_of(&r);
                for (c, &x) in xs.iter().enumerate() {
                    let got = crt(&[r.limbs()[0][c], r.limbs()[1][c]], &qs);
                    assert_eq!(got, ((x + h) / p) % q, "side {side} c={c} x={x}");
                }
            }
        }
    }

    /// Key-switching shaped input: `X = P*x + E` with small `|E| < 2^50` and `x` uniform in
    /// `Q_l` (60/40-bit primes, `N = 1024`, `k = 2`, level 2 of 4). `mod_down` returns
    /// exactly `x` (`X / P` is within `2^-70` of `x`, far from a tie); NTT form agrees.
    #[test]
    fn mod_down_divides_p_times_x_plus_small_noise() {
        let n = 1024;
        let mut qs = ntt_primes(60, n, 1).unwrap();
        qs.extend(ntt_primes(40, n, 3).unwrap());
        let b = RnsBasis::new(n, &qs).unwrap();
        let ps = ntt_primes_skip(60, n, 1, 2).unwrap();
        let sp = RnsBasis::new(n, &ps).unwrap();
        let level = 2;
        let p_mod_q = prod_mod(&ps, &b.moduli()[..=level]);
        let mut rng = ChaCha20Rng::seed_from_u64(1308);
        let bound = 1i128 << 50;
        for _ in 0..3 {
            let x = rand_poly(&b, level, &mut rng);
            let es: Vec<i128> = (0..n)
                .map(|_| rand_u128(&mut rng, 2 * bound as u128 - 1) as i128 - bound + 1)
                .collect();
            let mut xq = mul_limbwise(&x, &p_mod_q);
            xq.add_assign(&poly_i128(&b, level, &es));
            let xp = poly_i128(&sp, 1, &es);
            let mut r = xq.clone();
            r.mod_down(&xp);
            for c in 0..n {
                for i in 0..=level {
                    assert_eq!(r.limbs()[i][c], x.limbs()[i][c], "c={c} i={i}");
                }
            }
            let mut f = ntt_of(&xq);
            f.mod_down(&ntt_of(&xp));
            assert_eq!(coeff_of(&f).limbs(), r.limbs());
        }
    }

    /// `mod_down(P * (x, mod_up(x)))` recovers exactly `x` for `k = 1, 2, 3` (the `P` part of
    /// `P * lift` is zero, so `[X + h]_P = h`, far from a tie).
    #[test]
    fn mod_down_inverts_p_times_mod_up() {
        let n = 256;
        let mut qs = ntt_primes(60, n, 1).unwrap();
        qs.extend(ntt_primes(40, n, 2).unwrap());
        let b = RnsBasis::new(n, &qs).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(1309);
        for k in [1usize, 2, 3] {
            let ps = ntt_primes_skip(60, n, 1, k).unwrap();
            let sp = RnsBasis::new(n, &ps).unwrap();
            for ntt in [false, true] {
                let x = rand_poly(&b, 2, &mut rng);
                let x = if ntt { ntt_of(&x) } else { x };
                let up = x.mod_up(&sp);
                let mut lq = mul_limbwise(&x, &prod_mod(&ps, b.moduli()));
                let lp = mul_limbwise(&up, &prod_mod(&ps, sp.moduli()));
                assert!(lp.limbs().iter().all(|l| l.iter().all(|&v| v == 0)));
                lq.mod_down(&lp);
                let (xc, rc) = (coeff_of(&x), coeff_of(&lq));
                let m0 = b.moduli()[0];
                let e0 = m0.sub(xc.limbs()[0][0], rc.limbs()[0][0]);
                assert_eq!(e0, 0, "k={k}");
                for (i, m) in b.moduli().iter().enumerate() {
                    for c in 0..n {
                        assert_eq!(rc.limbs()[i][c], m.sub(xc.limbs()[i][c], e0));
                    }
                }
            }
        }
    }
}
