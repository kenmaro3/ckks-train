//! RNS polynomials over `Z_Q[X]/(X^N + 1)`, `Q = q_0 * ... * q_level`.
//!
//! An [`RnsPoly`] at level `l` stores `l + 1` limbs; limb `i` holds the residues mod the
//! `i`-th prime of its [`RnsBasis`]. A polynomial is either in coefficient form or in NTT form
//! (each limb transformed by [`NttTable::forward`](crate::ntt::NttTable::forward), i.e.
//! bit-reversed evaluation order).
//!
//! # Preconditions
//!
//! Binary operations require both operands to share the same basis (`Arc::ptr_eq`), the same
//! level and the same form. These are programming errors at this layer and are only
//! `debug_assert`ed; checked `Result`-returning wrappers (`LevelMismatch`, ...) live in the
//! CKKS layer. [`RnsPoly::mul_assign`] additionally requires both operands in NTT form.

use std::sync::Arc;

use crate::error::{HekitError, Result};
use crate::rns::RnsBasis;

/// A polynomial in RNS representation (one limb of `N` residues per active prime).
#[derive(Debug, Clone)]
pub struct RnsPoly {
    basis: Arc<RnsBasis>,
    limbs: Vec<Vec<u64>>,
    ntt_form: bool,
}

impl RnsPoly {
    /// The zero polynomial at `level` (using the first `level + 1` primes), in coefficient
    /// form.
    ///
    /// # Panics
    /// If `level >= basis.len()`.
    pub fn zero(basis: &Arc<RnsBasis>, level: usize) -> RnsPoly {
        assert!(
            level < basis.len(),
            "level {level} out of range for a basis of {} primes",
            basis.len()
        );
        RnsPoly {
            basis: Arc::clone(basis),
            limbs: vec![vec![0u64; basis.n()]; level + 1],
            ntt_form: false,
        }
    }

    /// Wraps existing limbs. `limbs.len() - 1` is the level; every limb must have `N`
    /// entries reduced modulo its prime. Fails with `InvalidParams` otherwise.
    pub fn from_limbs(
        basis: &Arc<RnsBasis>,
        limbs: Vec<Vec<u64>>,
        ntt_form: bool,
    ) -> Result<RnsPoly> {
        if limbs.is_empty() || limbs.len() > basis.len() {
            return Err(HekitError::InvalidParams(format!(
                "{} limbs for a basis of {} primes",
                limbs.len(),
                basis.len()
            )));
        }
        for (limb, m) in limbs.iter().zip(basis.moduli()) {
            if limb.len() != basis.n() {
                return Err(HekitError::InvalidParams(format!(
                    "limb length {} != N = {}",
                    limb.len(),
                    basis.n()
                )));
            }
            if limb.iter().any(|&x| x >= m.value()) {
                return Err(HekitError::InvalidParams(format!(
                    "limb entry not reduced mod {}",
                    m.value()
                )));
            }
        }
        Ok(RnsPoly {
            basis: Arc::clone(basis),
            limbs,
            ntt_form,
        })
    }

    /// The current level (`number of limbs - 1`).
    pub fn level(&self) -> usize {
        self.limbs.len() - 1
    }

    /// The basis this polynomial lives over.
    pub fn basis(&self) -> &Arc<RnsBasis> {
        &self.basis
    }

    /// `true` when the limbs are in NTT (evaluation) form.
    pub fn is_ntt_form(&self) -> bool {
        self.ntt_form
    }

    /// The limbs (`level + 1` vectors of `N` residues).
    pub fn limbs(&self) -> &[Vec<u64>] {
        &self.limbs
    }

    /// `self += other`, limb-wise. See the module docs for the preconditions.
    pub fn add_assign(&mut self, other: &RnsPoly) {
        self.debug_check_compatible(other);
        for ((a, b), m) in self
            .limbs
            .iter_mut()
            .zip(&other.limbs)
            .zip(self.basis.moduli())
        {
            a.iter_mut().zip(b).for_each(|(x, &y)| *x = m.add(*x, y));
        }
    }

    /// `self -= other`, limb-wise. See the module docs for the preconditions.
    pub fn sub_assign(&mut self, other: &RnsPoly) {
        self.debug_check_compatible(other);
        for ((a, b), m) in self
            .limbs
            .iter_mut()
            .zip(&other.limbs)
            .zip(self.basis.moduli())
        {
            a.iter_mut().zip(b).for_each(|(x, &y)| *x = m.sub(*x, y));
        }
    }

    /// In-place negation `self = -self` (either form).
    pub fn neg(&mut self) {
        for (a, m) in self.limbs.iter_mut().zip(self.basis.moduli()) {
            a.iter_mut().for_each(|x| *x = m.sub(0, *x));
        }
    }

    /// `self *= other` as ring elements: the pointwise product of NTT forms.
    ///
    /// Both operands must be in NTT form (`debug_assert`ed; calling [`RnsPoly::to_ntt`] first
    /// is the caller's responsibility) and satisfy the module-level preconditions.
    pub fn mul_assign(&mut self, other: &RnsPoly) {
        debug_assert!(
            self.ntt_form && other.ntt_form,
            "mul_assign needs both operands in NTT form"
        );
        self.debug_check_compatible(other);
        for ((a, b), m) in self
            .limbs
            .iter_mut()
            .zip(&other.limbs)
            .zip(self.basis.moduli())
        {
            a.iter_mut().zip(b).for_each(|(x, &y)| *x = m.mul(*x, y));
        }
    }

    /// `self *= scalar` (any `u64`; reduced mod each prime). Valid in either form.
    pub fn mul_scalar_assign(&mut self, scalar: u64) {
        for (a, m) in self.limbs.iter_mut().zip(self.basis.moduli()) {
            let s = m.shoup(scalar);
            a.iter_mut().for_each(|x| *x = m.mul_shoup(*x, &s));
        }
    }

    /// Converts to NTT form (no-op when already in NTT form).
    pub fn to_ntt(&mut self) {
        if self.ntt_form {
            return;
        }
        for (a, t) in self.limbs.iter_mut().zip(self.basis.ntt_tables()) {
            t.forward(a);
        }
        self.ntt_form = true;
    }

    /// Converts to coefficient form (no-op when already in coefficient form).
    pub fn to_coeff(&mut self) {
        if !self.ntt_form {
            return;
        }
        for (a, t) in self.limbs.iter_mut().zip(self.basis.ntt_tables()) {
            t.inverse(a);
        }
        self.ntt_form = false;
    }

    /// Drops the last limb (level `l -> l - 1`); the remaining limbs are unchanged. This is
    /// plain truncation, not a rescale.
    ///
    /// # Panics
    /// At level 0 (a polynomial must keep at least one limb).
    pub fn drop_last_limb(&mut self) {
        assert!(self.limbs.len() > 1, "cannot drop the last limb at level 0");
        self.limbs.pop();
    }

    fn debug_check_compatible(&self, other: &RnsPoly) {
        debug_assert!(
            Arc::ptr_eq(&self.basis, &other.basis),
            "operands over different bases"
        );
        debug_assert_eq!(
            self.limbs.len(),
            other.limbs.len(),
            "operands at different levels"
        );
        debug_assert_eq!(self.ntt_form, other.ntt_form, "operands in different forms");
    }

    /// The Galois automorphism `a(X) -> a(X^g)` in the current form.
    ///
    /// `galois_elt` must be odd and `< 2N` (`debug_assert`ed). Coefficient form:
    /// `X^i -> X^(i*g mod 2N)`, with a sign flip when `i*g mod 2N >= N`. NTT form: the index
    /// permutation of [`RnsBasis::galois_permutation`] (SEAL's `apply_galois_ntt`), cached
    /// in the basis.
    pub fn automorphism(&self, galois_elt: u64) -> RnsPoly {
        let n = self.basis.n();
        let two_n = 2 * n as u64;
        debug_assert!(
            galois_elt % 2 == 1 && galois_elt < two_n,
            "Galois element {galois_elt} must be odd and < 2N = {two_n}"
        );
        let limbs = if self.ntt_form {
            let perm = self.basis.galois_permutation(galois_elt);
            self.limbs
                .iter()
                .map(|a| perm.iter().map(|&p| a[p as usize]).collect())
                .collect()
        } else {
            let g = galois_elt % two_n;
            self.limbs
                .iter()
                .zip(self.basis.moduli())
                .map(|(a, m)| {
                    let mut out = vec![0u64; n];
                    let mut e = 0u64; // i * g mod 2N
                    for &c in a {
                        if e < n as u64 {
                            out[e as usize] = c;
                        } else {
                            out[(e - n as u64) as usize] = m.sub(0, c);
                        }
                        e += g;
                        if e >= two_n {
                            e -= two_n;
                        }
                    }
                    out
                })
                .collect()
        };
        RnsPoly {
            basis: Arc::clone(&self.basis),
            limbs,
            ntt_form: self.ntt_form,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulus::Modulus;
    use crate::primes::ntt_primes;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn basis(n: usize, count: usize) -> Arc<RnsBasis> {
        let mut ps = ntt_primes(60, n, 1).unwrap();
        ps.extend(ntt_primes(40, n, count - 1).unwrap());
        RnsBasis::new(n, &ps).unwrap()
    }

    /// Uniform-ish random polynomial at `level` (coefficient form).
    fn rand_poly(b: &Arc<RnsBasis>, level: usize, rng: &mut ChaCha20Rng) -> RnsPoly {
        let limbs = b.moduli()[..=level]
            .iter()
            .map(|m| (0..b.n()).map(|_| rng.next_u64() % m.value()).collect())
            .collect();
        RnsPoly::from_limbs(b, limbs, false).unwrap()
    }

    /// A random polynomial with small signed coefficients, consistent across limbs.
    fn rand_small(b: &Arc<RnsBasis>, level: usize, rng: &mut ChaCha20Rng) -> RnsPoly {
        let coeffs: Vec<i64> = (0..b.n())
            .map(|_| (rng.next_u64() % 41) as i64 - 20)
            .collect();
        let limbs = b.moduli()[..=level]
            .iter()
            .map(|m| {
                let q = m.value() as i64;
                coeffs.iter().map(|&c| c.rem_euclid(q) as u64).collect()
            })
            .collect();
        RnsPoly::from_limbs(b, limbs, false).unwrap()
    }

    fn naive_negacyclic(a: &[u64], b: &[u64], m: &Modulus) -> Vec<u64> {
        let n = a.len();
        let mut c = vec![0u64; n];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let p = m.mul(ai, bj);
                let k = i + j;
                if k < n {
                    c[k] = m.add(c[k], p);
                } else {
                    c[k - n] = m.sub(c[k - n], p);
                }
            }
        }
        c
    }

    /// Coefficient-form automorphism straight from the definition, on one limb.
    fn naive_automorphism(a: &[u64], g: u64, m: &Modulus) -> Vec<u64> {
        let n = a.len() as u64;
        let mut out = vec![0u64; a.len()];
        for (i, &c) in a.iter().enumerate() {
            let e = i as u64 * g % (2 * n);
            if e < n {
                out[e as usize] = m.add(out[e as usize], c);
            } else {
                out[(e - n) as usize] = m.sub(out[(e - n) as usize], c);
            }
        }
        out
    }

    /// `1`, `5^k` (k = 1..=6), `3`, `2N - 5` and `2N - 1` (conjugation), reduced mod `2N`
    /// and deduplicated (for tiny `N` several coincide).
    fn galois_elts(n: usize) -> Vec<u64> {
        let two_n = 2 * n as u64;
        let mut gs = vec![1, 3 % two_n, (two_n * 3 - 5) % two_n, two_n - 1];
        let mut g = 1u64;
        for _ in 0..6 {
            g = g * 5 % two_n;
            gs.push(g);
        }
        let mut out = Vec::new();
        for g in gs {
            if !out.contains(&g) {
                out.push(g);
            }
        }
        out
    }

    /// Multiplicative order of `g` in `Z_2N^*`.
    fn order(g: u64, n: usize) -> u64 {
        let two_n = 2 * n as u64;
        let mut x = g % two_n;
        let mut k = 1;
        while x != 1 {
            x = x * g % two_n;
            k += 1;
        }
        k
    }

    #[test]
    fn zero_level_and_shape() {
        let b = basis(64, 4);
        for level in 0..4 {
            let z = RnsPoly::zero(&b, level);
            assert_eq!(z.level(), level);
            assert_eq!(z.limbs().len(), level + 1);
            assert!(!z.is_ntt_form());
            assert!(
                z.limbs()
                    .iter()
                    .all(|l| l.len() == 64 && l.iter().all(|&x| x == 0))
            );
        }
    }

    #[test]
    #[should_panic]
    fn zero_level_out_of_range_panics() {
        let b = basis(64, 2);
        let _ = RnsPoly::zero(&b, 2);
    }

    #[test]
    fn from_limbs_validates() {
        let b = basis(16, 2);
        let q0 = b.moduli()[0].value();
        let ok = vec![vec![0u64; 16], vec![1u64; 16]];
        assert_eq!(RnsPoly::from_limbs(&b, ok, true).unwrap().level(), 1);
        let bad = |l: Vec<Vec<u64>>| {
            matches!(
                RnsPoly::from_limbs(&b, l, false),
                Err(HekitError::InvalidParams(_))
            )
        };
        assert!(bad(vec![]));
        assert!(bad(vec![vec![0; 16]; 3]));
        assert!(bad(vec![vec![0; 15]]));
        assert!(bad(vec![vec![q0; 16]]));
    }

    #[test]
    fn add_sub_neg_scalar_match_modulus_limbwise() {
        let b = basis(256, 3);
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        for ntt in [false, true] {
            for level in [0, 2] {
                let mut a = rand_poly(&b, level, &mut rng);
                let mut c = rand_poly(&b, level, &mut rng);
                if ntt {
                    a.to_ntt();
                    c.to_ntt();
                }
                let s = rng.next_u64();
                let (mut sum, mut diff, mut neg, mut sc) =
                    (a.clone(), a.clone(), a.clone(), a.clone());
                sum.add_assign(&c);
                diff.sub_assign(&c);
                neg.neg();
                sc.mul_scalar_assign(s);
                for (i, m) in b.moduli()[..=level].iter().enumerate() {
                    let (x, y) = (&a.limbs()[i], &c.limbs()[i]);
                    for j in 0..b.n() {
                        assert_eq!(sum.limbs()[i][j], m.add(x[j], y[j]));
                        assert_eq!(diff.limbs()[i][j], m.sub(x[j], y[j]));
                        assert_eq!(neg.limbs()[i][j], m.sub(0, x[j]));
                        assert_eq!(sc.limbs()[i][j], m.mul(x[j], s % m.value()));
                    }
                }
                assert_eq!(sum.is_ntt_form(), ntt);
            }
        }
    }

    #[test]
    fn add_sub_worst_case() {
        let b = basis(32, 2);
        let limbs: Vec<Vec<u64>> = b.moduli().iter().map(|m| vec![m.value() - 1; 32]).collect();
        let a = RnsPoly::from_limbs(&b, limbs, false).unwrap();
        let mut s = a.clone();
        s.add_assign(&a);
        let mut d = RnsPoly::zero(&b, 1);
        d.sub_assign(&a);
        let mut n0 = RnsPoly::zero(&b, 1);
        n0.neg();
        for (i, m) in b.moduli().iter().enumerate() {
            let q = m.value();
            assert!(s.limbs()[i].iter().all(|&x| x == q - 2));
            assert!(d.limbs()[i].iter().all(|&x| x == 1));
            assert!(n0.limbs()[i].iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn mul_matches_modulus_limbwise_and_negacyclic_product() {
        let b = basis(64, 3);
        let mut rng = ChaCha20Rng::seed_from_u64(12);
        let a = rand_poly(&b, 2, &mut rng);
        let c = rand_poly(&b, 2, &mut rng);
        let (mut fa, mut fc) = (a.clone(), c.clone());
        fa.to_ntt();
        fc.to_ntt();
        let mut prod = fa.clone();
        prod.mul_assign(&fc);
        assert!(prod.is_ntt_form());
        for (i, m) in b.moduli().iter().enumerate() {
            for j in 0..b.n() {
                assert_eq!(
                    prod.limbs()[i][j],
                    m.mul(fa.limbs()[i][j], fc.limbs()[i][j])
                );
            }
        }
        prod.to_coeff();
        for (i, m) in b.moduli().iter().enumerate() {
            assert_eq!(
                prod.limbs()[i],
                naive_negacyclic(&a.limbs()[i], &c.limbs()[i], m)
            );
        }
    }

    #[test]
    fn ntt_roundtrip_multi_limb() {
        let b = basis(1024, 4);
        let mut rng = ChaCha20Rng::seed_from_u64(13);
        for level in 0..4 {
            let a = rand_poly(&b, level, &mut rng);
            let mut f = a.clone();
            f.to_ntt();
            assert!(f.is_ntt_form());
            assert_ne!(f.limbs(), a.limbs());
            // Each limb is that prime's forward NTT.
            for (i, t) in b.ntt_tables()[..=level].iter().enumerate() {
                let mut l = a.limbs()[i].clone();
                t.forward(&mut l);
                assert_eq!(f.limbs()[i], l);
            }
            f.to_ntt(); // idempotent
            f.to_coeff();
            assert!(!f.is_ntt_form());
            assert_eq!(f.limbs(), a.limbs());
            f.to_coeff(); // idempotent
            assert_eq!(f.limbs(), a.limbs());
        }
    }

    #[test]
    fn drop_last_limb_keeps_remaining_limbs() {
        let b = basis(128, 4);
        let mut rng = ChaCha20Rng::seed_from_u64(14);
        for ntt in [false, true] {
            let mut a = rand_poly(&b, 3, &mut rng);
            if ntt {
                a.to_ntt();
            }
            let orig = a.clone();
            for level in (0..3).rev() {
                a.drop_last_limb();
                assert_eq!(a.level(), level);
                assert_eq!(a.limbs(), &orig.limbs()[..=level]);
                assert_eq!(a.is_ntt_form(), ntt);
            }
        }
    }

    #[test]
    #[should_panic]
    fn drop_last_limb_at_level_zero_panics() {
        let b = basis(16, 2);
        RnsPoly::zero(&b, 0).drop_last_limb();
    }

    #[test]
    fn automorphism_coeff_form_matches_definition() {
        let b = basis(64, 3);
        let mut rng = ChaCha20Rng::seed_from_u64(15);
        let a = rand_poly(&b, 2, &mut rng);
        for g in galois_elts(64) {
            let r = a.automorphism(g);
            assert!(!r.is_ntt_form());
            assert_eq!(r.level(), 2);
            for (i, m) in b.moduli().iter().enumerate() {
                assert_eq!(
                    r.limbs()[i],
                    naive_automorphism(&a.limbs()[i], g, m),
                    "g={g}"
                );
            }
        }
        // Monomial check: X^1 -> X^g; X^(N-1) under g = 2N-1 -> X^(N-1)(2N-1) = X^N... sign.
        let n = 64usize;
        let mut mono = vec![vec![0u64; n]; 3];
        mono.iter_mut().for_each(|l| l[1] = 1);
        let x = RnsPoly::from_limbs(&b, mono, false).unwrap();
        // X -> X^(2N-1) = X^(-1) = -X^(N-1).
        let conj = x.automorphism(2 * n as u64 - 1);
        for (i, m) in b.moduli().iter().enumerate() {
            let mut want = vec![0u64; n];
            want[n - 1] = m.value() - 1;
            assert_eq!(conj.limbs()[i], want);
        }
        let x5 = x.automorphism(5);
        assert!(
            x5.limbs()
                .iter()
                .all(|l| l[5] == 1 && l.iter().filter(|&&v| v != 0).count() == 1)
        );
    }

    #[test]
    fn automorphism_ntt_form_matches_coeff_form() {
        let mut rng = ChaCha20Rng::seed_from_u64(16);
        for n in [2usize, 8, 64, 4096] {
            let b = basis(n, 3);
            for level in [0, 2] {
                let a = rand_poly(&b, level, &mut rng);
                let mut fa = a.clone();
                fa.to_ntt();
                for g in galois_elts(n) {
                    let want = a.automorphism(g);
                    let mut got = fa.automorphism(g);
                    assert!(got.is_ntt_form());
                    got.to_coeff();
                    assert_eq!(got.limbs(), want.limbs(), "n={n} level={level} g={g}");
                }
            }
        }
    }

    #[test]
    fn automorphism_order_times_is_identity() {
        let n = 256;
        let b = basis(n, 2);
        let mut rng = ChaCha20Rng::seed_from_u64(17);
        let a = rand_poly(&b, 1, &mut rng);
        let mut fa = a.clone();
        fa.to_ntt();
        for g in galois_elts(n) {
            let ord = order(g, n);
            if g == 5 {
                assert_eq!(ord, n as u64 / 2);
            }
            if g == 2 * n as u64 - 1 {
                assert_eq!(ord, 2);
            }
            for start in [&a, &fa] {
                let mut x = start.clone();
                for k in 1..=ord {
                    x = x.automorphism(g);
                    if k < ord && g != 1 {
                        assert_ne!(x.limbs(), start.limbs(), "g={g} k={k}");
                    }
                }
                assert_eq!(
                    x.limbs(),
                    start.limbs(),
                    "g={g} ntt={}",
                    start.is_ntt_form()
                );
            }
        }
    }

    #[test]
    fn automorphism_is_ring_homomorphism() {
        let n = 128;
        let b = basis(n, 2);
        let mut rng = ChaCha20Rng::seed_from_u64(18);
        let mut a = rand_small(&b, 1, &mut rng);
        let mut c = rand_poly(&b, 1, &mut rng);
        a.to_ntt();
        c.to_ntt();
        for g in [5u64, 2 * n as u64 - 1, 25] {
            let mut lhs = a.clone();
            lhs.mul_assign(&c);
            let lhs = lhs.automorphism(g);
            let mut rhs = a.automorphism(g);
            rhs.mul_assign(&c.automorphism(g));
            assert_eq!(lhs.limbs(), rhs.limbs(), "g={g}");
        }
    }
}
