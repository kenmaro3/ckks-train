//! ChaCha20-based CSPRNG and polynomial samplers.
//!
//! All samplers return polynomials in **coefficient form**; callers convert with
//! [`RnsPoly::to_ntt`]. A signed integer coefficient `x` is stored in limb `i` as `x mod q_i`
//! (`q_i - |x|` for negative `x`), so every limb represents the same integer polynomial.
//!
//! # Side channels
//!
//! The samplers are **not constant-time** (rejection sampling, data-dependent branches). This
//! is acceptable for a research library but must be revisited before production use.

use std::sync::Arc;

use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use crate::poly::RnsPoly;
use crate::rns::RnsBasis;

/// Tail cut of the discrete Gaussian, in standard deviations.
const TAIL_SIGMAS: f64 = 6.0;

/// Cryptographically secure random number generator (ChaCha20) with polynomial samplers.
#[derive(Debug, Clone)]
pub struct Rng(ChaCha20Rng);

impl Rng {
    /// Deterministic generator from a 32-byte seed (for tests and reproducible runs only).
    pub fn from_seed(seed: [u8; 32]) -> Rng {
        Rng(ChaCha20Rng::from_seed(seed))
    }

    /// Generator seeded from operating-system entropy.
    ///
    /// # Panics
    /// If the OS entropy source is unavailable.
    pub fn from_entropy() -> Rng {
        Rng(ChaCha20Rng::from_os_rng())
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    /// Uniform integer in `[0, bound)` by rejection sampling (no modulo bias).
    ///
    /// # Panics
    /// If `bound == 0`.
    pub fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "bound must be positive");
        let zone = (u64::MAX / bound) * bound;
        loop {
            let x = self.0.next_u64();
            if x < zone {
                return x % bound;
            }
        }
    }

    /// Uniformly random polynomial: every coefficient is independent and uniform modulo each
    /// `q_i` (the RNS-uniform distribution on `Z_Q`). Coefficient form.
    ///
    /// # Panics
    /// If `level >= basis.len()`.
    pub fn uniform_poly(&mut self, basis: &Arc<RnsBasis>, level: usize) -> RnsPoly {
        assert!(level < basis.len(), "level out of range");
        let limbs = basis.moduli()[..=level]
            .iter()
            .map(|m| {
                let q = m.value();
                (0..basis.n()).map(|_| self.below(q)).collect()
            })
            .collect();
        RnsPoly::from_limbs(basis, limbs, false).expect("sampled limbs are reduced")
    }

    /// Ternary polynomial with coefficients in `{-1, 0, 1}`. Coefficient form.
    ///
    /// * `Some(h)`: exactly `h` nonzero coefficients at uniformly random distinct positions
    ///   (partial Fisher–Yates), each `±1` with probability 1/2.
    /// * `None`: each coefficient is `-1, 0, 1` with probability 1/4, 1/2, 1/4.
    ///
    /// # Panics
    /// If `level >= basis.len()` or `hamming_weight > N` (callers such as the key generator
    /// validate `h <= N` and return `InvalidParams` beforehand).
    pub fn ternary_poly(
        &mut self,
        basis: &Arc<RnsBasis>,
        level: usize,
        hamming_weight: Option<usize>,
    ) -> RnsPoly {
        assert!(level < basis.len(), "level out of range");
        let n = basis.n();
        let mut coeffs = vec![0i64; n];
        match hamming_weight {
            Some(h) => {
                assert!(h <= n, "hamming weight {h} exceeds N = {n}");
                let mut idx: Vec<usize> = (0..n).collect();
                for k in 0..h {
                    let j = k + self.below((n - k) as u64) as usize;
                    idx.swap(k, j);
                    coeffs[idx[k]] = if self.0.next_u64() & 1 == 0 { 1 } else { -1 };
                }
            }
            None => {
                for c in coeffs.iter_mut() {
                    *c = match self.0.next_u64() & 3 {
                        0 => -1,
                        1 => 1,
                        _ => 0,
                    };
                }
            }
        }
        signed_to_poly(basis, level, &coeffs)
    }

    /// Discrete Gaussian polynomial with standard deviation `sigma`, tails cut at
    /// `|x| <= 6 sigma`. Coefficient form.
    ///
    /// Method: inversion with a cumulative distribution table (CDT). The weights
    /// `exp(-x^2 / (2 sigma^2))` of `x in [-B, B]`, `B = floor(6 sigma)`, are normalised and
    /// accumulated into 64-bit thresholds once per call; each coefficient is the first `x`
    /// whose threshold exceeds one uniform 64-bit draw. This is the truncated discrete
    /// Gaussian with probabilities exact up to `f64` rounding (~2^-53 relative), using one
    /// draw per coefficient (rejection sampling needed about ten).
    ///
    /// # Panics
    /// If `level >= basis.len()` or `sigma` is not finite and positive.
    pub fn gaussian_poly(&mut self, basis: &Arc<RnsBasis>, level: usize, sigma: f64) -> RnsPoly {
        assert!(level < basis.len(), "level out of range");
        assert!(sigma.is_finite() && sigma > 0.0, "sigma must be positive");
        let bound = (TAIL_SIGMAS * sigma).floor() as i64;
        let denom = 2.0 * sigma * sigma;
        let weights: Vec<f64> = (-bound..=bound)
            .map(|x| (-(x as f64) * (x as f64) / denom).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        // cdt[i] = round(2^64 * P(X <= x_i)); the last threshold is u64::MAX, so every draw
        // falls in some bucket (r = u64::MAX, probability 2^-64, lands in the last one).
        let mut acc = 0.0;
        let mut cdt: Vec<u64> = weights
            .iter()
            .map(|w| {
                acc += w;
                // `as` saturates at u64::MAX for values >= 2^64.
                (acc / total * 18_446_744_073_709_551_616.0) as u64
            })
            .collect();
        *cdt.last_mut().expect("bound >= 0") = u64::MAX;
        let last = cdt.len() - 1;
        let coeffs: Vec<i64> = (0..basis.n())
            .map(|_| {
                let r = self.0.next_u64();
                cdt.partition_point(|&c| c <= r).min(last) as i64 - bound
            })
            .collect();
        signed_to_poly(basis, level, &coeffs)
    }
}

/// Embeds small signed integers into every limb (`x mod q_i`), coefficient form.
fn signed_to_poly(basis: &Arc<RnsBasis>, level: usize, coeffs: &[i64]) -> RnsPoly {
    let limbs = basis.moduli()[..=level]
        .iter()
        .map(|m| {
            let q = m.value();
            coeffs
                .iter()
                .map(|&x| {
                    if x >= 0 {
                        x as u64
                    } else {
                        q - x.unsigned_abs()
                    }
                })
                .collect()
        })
        .collect();
    RnsPoly::from_limbs(basis, limbs, false).expect("embedded limbs are reduced")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primes::ntt_primes;

    fn basis(n: usize) -> Arc<RnsBasis> {
        let mut qs = ntt_primes(50, n, 2).unwrap();
        qs.extend(ntt_primes(40, n, 1).unwrap());
        RnsBasis::new(n, &qs).unwrap()
    }

    fn rng() -> Rng {
        Rng::from_seed([7u8; 32])
    }

    /// Centered representative of limb 0.
    fn signed(p: &RnsPoly, limb: usize) -> Vec<i64> {
        let q = p.basis().moduli()[limb].value();
        p.limbs()[limb]
            .iter()
            .map(|&x| {
                if x > q / 2 {
                    x as i64 - q as i64
                } else {
                    x as i64
                }
            })
            .collect()
    }

    #[test]
    fn ternary_hamming_weight_exact() {
        let b = basis(1 << 12);
        let p = rng().ternary_poly(&b, 2, Some(64));
        assert!(!p.is_ntt_form());
        for l in 0..=2 {
            let v = signed(&p, l);
            assert_eq!(v.iter().filter(|&&x| x != 0).count(), 64);
            assert!(v.iter().all(|&x| (-1..=1).contains(&x)));
        }
        // Full weight and zero weight edge cases.
        let b = basis(1 << 6);
        let full = rng().ternary_poly(&b, 0, Some(64));
        assert!(signed(&full, 0).iter().all(|&x| x != 0));
        let zero = rng().ternary_poly(&b, 0, Some(0));
        assert!(signed(&zero, 0).iter().all(|&x| x == 0));
    }

    #[test]
    fn ternary_hamming_signs_balanced() {
        let b = basis(1 << 12);
        let v = signed(&rng().ternary_poly(&b, 0, Some(2048)), 0);
        let pos = v.iter().filter(|&&x| x == 1).count();
        assert!((900..=1150).contains(&pos), "pos = {pos}");
    }

    #[test]
    fn ternary_hamming_positions_uniform() {
        let n = 1 << 12;
        let h = 256;
        let b = basis(n);
        let support = |seed: u8| -> Vec<usize> {
            let v = signed(&Rng::from_seed([seed; 32]).ternary_poly(&b, 0, Some(h)), 0);
            (0..n).filter(|&i| v[i] != 0).collect()
        };
        // Count in each of 8 equal blocks: hypergeometric, mean h/8 = 32, sigma ~ 5.3.
        for seed in [1u8, 2, 3] {
            let s = support(seed);
            assert_eq!(s.len(), h);
            for blk in 0..8 {
                let c = s.iter().filter(|&&i| i / (n / 8) == blk).count() as f64;
                assert!((c - 32.0).abs() < 5.0 * 5.3, "seed {seed} block {blk}: {c}");
            }
        }
        assert_ne!(support(1), support(2));
        assert_ne!(support(2), support(3));
    }

    #[test]
    fn same_seed_same_output() {
        let b = basis(1 << 10);
        let (mut r1, mut r2) = (rng(), rng());
        assert_eq!(
            r1.uniform_poly(&b, 2).limbs(),
            r2.uniform_poly(&b, 2).limbs()
        );
        assert_eq!(
            r1.ternary_poly(&b, 1, Some(32)).limbs(),
            r2.ternary_poly(&b, 1, Some(32)).limbs()
        );
        assert_eq!(
            r1.ternary_poly(&b, 1, None).limbs(),
            r2.ternary_poly(&b, 1, None).limbs()
        );
        assert_eq!(
            r1.gaussian_poly(&b, 1, 3.2).limbs(),
            r2.gaussian_poly(&b, 1, 3.2).limbs()
        );
        let mut r3 = Rng::from_seed([8u8; 32]);
        assert_ne!(
            rng().uniform_poly(&b, 0).limbs(),
            r3.uniform_poly(&b, 0).limbs()
        );
    }

    #[test]
    fn entropy_rngs_differ() {
        let b = basis(1 << 6);
        let a = Rng::from_entropy().uniform_poly(&b, 0);
        let c = Rng::from_entropy().uniform_poly(&b, 0);
        assert_ne!(a.limbs(), c.limbs());
    }

    #[test]
    fn ternary_none_proportions() {
        let n = 1 << 14;
        let b = basis(n);
        let v = signed(&rng().ternary_poly(&b, 0, None), 0);
        let count = |t: i64| v.iter().filter(|&&x| x == t).count() as f64 / n as f64;
        assert!((count(-1) - 0.25).abs() < 0.03, "{}", count(-1));
        assert!((count(0) - 0.5).abs() < 0.03, "{}", count(0));
        assert!((count(1) - 0.25).abs() < 0.03, "{}", count(1));
    }

    #[test]
    fn gaussian_variance_and_tail() {
        let n = 1 << 14;
        let b = basis(n);
        let sigma = 3.2;
        let p = rng().gaussian_poly(&b, 0, sigma);
        let v = signed(&p, 0);
        let bound = (6.0 * sigma) as i64;
        assert!(v.iter().all(|&x| x.abs() <= bound));
        let mean = v.iter().sum::<i64>() as f64 / n as f64;
        let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let target = sigma * sigma;
        assert!((var - target).abs() < 0.1 * target, "var = {var}");
        assert!(mean.abs() < 0.2, "mean = {mean}");
    }

    /// Empirical frequencies of the CDT sampler against the exact truncated discrete
    /// Gaussian pmf (2^16 samples, each |x| <= 8 within 5 standard errors).
    #[test]
    fn gaussian_pmf_matches_exact() {
        let n = 1 << 16;
        let b = basis(n);
        let sigma = 3.2;
        let v = signed(&rng().gaussian_poly(&b, 0, sigma), 0);
        let bound = (6.0 * sigma) as i64;
        let w = |x: i64| (-(x as f64).powi(2) / (2.0 * sigma * sigma)).exp();
        let total: f64 = (-bound..=bound).map(w).sum();
        for x in -8..=8 {
            let p = w(x) / total;
            let f = v.iter().filter(|&&y| y == x).count() as f64 / n as f64;
            let se = (p * (1.0 - p) / n as f64).sqrt();
            assert!((f - p).abs() < 5.0 * se, "x = {x}: freq {f}, pmf {p}");
        }
    }

    #[test]
    fn uniform_values_reduced_and_spread() {
        let n = 1 << 12;
        let b = basis(n);
        let p = rng().uniform_poly(&b, 2);
        assert_eq!(p.level(), 2);
        for (l, limb) in p.limbs().iter().enumerate() {
            let q = b.moduli()[l].value();
            assert!(limb.iter().all(|&x| x < q));
            let mean = limb.iter().map(|&x| x as f64 / q as f64).sum::<f64>() / n as f64;
            assert!((mean - 0.5).abs() < 0.03, "limb {l} mean {mean}");
        }
    }

    #[test]
    fn below_is_in_range_and_covers() {
        let mut r = rng();
        let mut seen = [false; 5];
        for _ in 0..200 {
            let x = r.below(5);
            seen[x as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn limbs_represent_same_integer() {
        let b = basis(1 << 10);
        let polys = [
            rng().ternary_poly(&b, 2, None),
            rng().ternary_poly(&b, 2, Some(100)),
            rng().gaussian_poly(&b, 2, 3.2),
        ];
        for p in &polys {
            let base = signed(p, 0);
            for l in 1..=2 {
                assert_eq!(signed(p, l), base);
            }
        }
    }
}
