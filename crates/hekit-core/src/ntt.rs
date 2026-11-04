//! Negacyclic number-theoretic transform over `Z_q[X]/(X^N+1)` (Harvey butterflies,
//! Shoup precomputation, lazy reduction in `[0, 4q)`).
//!
//! # Conventions (normative; SEAL 4.1 interop depends on them)
//!
//! * `N >= 2` is a power of two, `q` is prime with `q = 1 (mod 2N)` and `q < 2^60`.
//! * `psi` is the **minimal primitive `2N`-th root of unity** mod `q`: among all elements of
//!   multiplicative order exactly `2N`, the numerically smallest (this is what SEAL's
//!   `try_minimal_primitive_root` selects).
//! * **Twiddle table**: `roots[i] = psi^bitrev(i) mod q` for `i in 0..N`, where `bitrev`
//!   reverses the `log2 N` low bits. `inv_roots[i] = psi^(-bitrev(i)) mod q`.
//! * **Forward** ([`NttTable::forward`]): Cooley-Tukey, natural-order input, **bit-reversed**
//!   output with `out[bitrev(k)] = a(psi^(2k+1))`. Output fully reduced to `[0, q)`.
//! * **Inverse** ([`NttTable::inverse`]): Gentleman-Sande, bit-reversed input, natural-order
//!   output, includes the `N^-1` scaling, fully reduced to `[0, q)`.
//!
//! Pointwise products of forward outputs are the forward output of the negacyclic product.
//!
//! # Lazy-reduction invariants
//!
//! Forward: every value entering a stage is in `[0, 4q)` (`4q < 2^62`). A butterfly brings
//! `x` into `[0, 2q)`, computes `t = mul_shoup_lazy(y, w) in [0, 2q)` and outputs
//! `x + t` and `x - t + 2q`, both in `[0, 4q)`; a final pass reduces to `[0, q)`.
//! Inverse: every value entering a stage is in `[0, 2q)`; sums are reduced by a conditional
//! `-2q`, differences `x - y + 2q in (0, 4q)` go through the lazy Shoup product into `[0, 2q)`.
//! The last stage fuses `N^-1` and the final reduction.

use crate::error::{HekitError, Result};
use crate::modulus::{Modulus, Shoup};
use crate::primes::is_prime_u64;

/// Precomputed tables for the negacyclic NTT of size `n` modulo a prime `q`.
#[derive(Debug, Clone)]
pub struct NttTable {
    n: usize,
    modulus: Modulus,
    psi: u64,
    roots: Vec<Shoup>,
    inv_roots: Vec<Shoup>,
    n_inv: Shoup,
    /// `inv_roots[1] * n_inv`, used by the fused last inverse stage.
    w_n_inv: Shoup,
}

fn bitrev(i: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        i.reverse_bits() >> (usize::BITS - bits)
    }
}

/// `table[bitrev(i)] = base^i`, as Shoup pairs.
fn bitrev_power_table(n: usize, m: &Modulus, base: u64) -> Vec<Shoup> {
    let bits = n.trailing_zeros();
    let mut table = vec![m.shoup(0); n];
    let mut p = 1u64;
    for i in 0..n {
        table[bitrev(i, bits)] = m.shoup(p);
        p = m.mul(p, base);
    }
    table
}

/// The minimal primitive `2n`-th root of unity mod prime `q` (`2n | q - 1`).
fn minimal_primitive_root(n: usize, m: &Modulus) -> u64 {
    let q = m.value();
    let e = (q - 1) / (2 * n as u64);
    // Find one primitive root: c = g^e has order 2n iff c^n = -1.
    let c = (2u64..)
        .map(|g| m.pow(g, e))
        .find(|&c| m.pow(c, n as u64) == q - 1)
        .expect("a primitive 2n-th root exists when q is prime and 2n | q-1");
    // All primitive 2n-th roots are the odd powers c^1, c^3, ..., c^(2n-1).
    let c2 = m.mul(c, c);
    let (mut cur, mut best) = (c, c);
    for _ in 1..n {
        cur = m.mul(cur, c2);
        best = best.min(cur);
    }
    best
}

impl NttTable {
    /// Builds the tables for size `n` over `modulus`.
    ///
    /// Requires `n >= 2` a power of two, `modulus` prime, and `q = 1 (mod 2n)`;
    /// otherwise returns `InvalidParams`.
    pub fn new(n: usize, modulus: Modulus) -> Result<NttTable> {
        if n < 2 || !n.is_power_of_two() {
            return Err(HekitError::InvalidParams(format!(
                "NTT size {n} must be a power of two >= 2"
            )));
        }
        let q = modulus.value();
        // 2n < 2^60 is implied by q < 2^60 once 2n | q - 1 (checked first to avoid overflow).
        if (n as u64) >= q || (q - 1) % (2 * n as u64) != 0 {
            return Err(HekitError::InvalidParams(format!(
                "modulus {q} is not 1 mod 2n for n = {n}"
            )));
        }
        if !is_prime_u64(q) {
            return Err(HekitError::InvalidParams(format!(
                "modulus {q} is not prime"
            )));
        }
        let psi = minimal_primitive_root(n, &modulus);
        let psi_inv = modulus.inv(psi).expect("psi is invertible");
        let n_inv_v = modulus.inv(n as u64).expect("n is invertible mod prime q");
        let roots = bitrev_power_table(n, &modulus, psi);
        let inv_roots = bitrev_power_table(n, &modulus, psi_inv);
        let w_n_inv = modulus.shoup(modulus.mul(inv_roots[1].value(), n_inv_v));
        Ok(NttTable {
            n,
            modulus,
            psi,
            roots,
            inv_roots,
            n_inv: modulus.shoup(n_inv_v),
            w_n_inv,
        })
    }

    /// The transform size `N`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The modulus `q`.
    pub fn modulus(&self) -> &Modulus {
        &self.modulus
    }

    /// The minimal primitive `2N`-th root of unity `psi`.
    pub fn psi(&self) -> u64 {
        self.psi
    }

    /// In-place forward NTT. `a.len()` must equal `n` and every `a[i] < q`
    /// (panics on a length mismatch). Output is in bit-reversed order, in `[0, q)`.
    pub fn forward(&self, a: &mut [u64]) {
        assert_eq!(a.len(), self.n, "NTT input length must equal n");
        let m = &self.modulus;
        let q = m.value();
        debug_assert!(a.iter().all(|&x| x < q));
        let two_q = 2 * q;
        let n = self.n;
        let mut t = n;
        let mut groups = 1;
        // Every stage but the last (t >= 2); group i of a stage uses roots[groups + i].
        while groups < n / 2 {
            t >>= 1;
            for (chunk, w) in a
                .chunks_exact_mut(2 * t)
                .zip(&self.roots[groups..2 * groups])
            {
                let (lo, hi) = chunk.split_at_mut(t);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let xr = if *x >= two_q { *x - two_q } else { *x };
                    let tt = m.mul_shoup_lazy(*y, w);
                    *x = xr + tt;
                    *y = xr + two_q - tt;
                }
            }
            groups <<= 1;
        }
        // Last stage (t = 1, groups = n/2) fused with the final reduction [0, 4q) -> [0, q).
        let reduce = |r: u64| {
            let r = if r >= two_q { r - two_q } else { r };
            if r >= q { r - q } else { r }
        };
        for (pair, w) in a.chunks_exact_mut(2).zip(&self.roots[n / 2..]) {
            let x = pair[0];
            let xr = if x >= two_q { x - two_q } else { x };
            let tt = m.mul_shoup_lazy(pair[1], w);
            pair[0] = reduce(xr + tt);
            pair[1] = reduce(xr + two_q - tt);
        }
    }

    /// In-place inverse NTT (bit-reversed input, natural-order output, includes the `N^-1`
    /// scaling). `a.len()` must equal `n` and every `a[i] < q` (panics on a length mismatch).
    /// Output is in `[0, q)`.
    pub fn inverse(&self, a: &mut [u64]) {
        assert_eq!(a.len(), self.n, "NTT input length must equal n");
        let m = &self.modulus;
        let q = m.value();
        debug_assert!(a.iter().all(|&x| x < q));
        let two_q = 2 * q;
        let mut t = 1;
        let mut h = self.n >> 1;
        // First stage (t = 1) on adjacent pairs, without the per-group split.
        if h > 1 {
            for (pair, w) in a.chunks_exact_mut(2).zip(&self.inv_roots[h..]) {
                let (x, y) = (pair[0], pair[1]);
                let s = x + y;
                pair[0] = if s >= two_q { s - two_q } else { s };
                pair[1] = m.mul_shoup_lazy(x + two_q - y, w);
            }
            t = 2;
            h >>= 1;
        }
        while h > 1 {
            for (chunk, w) in a.chunks_exact_mut(2 * t).zip(&self.inv_roots[h..2 * h]) {
                let (lo, hi) = chunk.split_at_mut(t);
                for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                    let s = *x + *y;
                    let d = *x + two_q - *y;
                    *x = if s >= two_q { s - two_q } else { s };
                    *y = m.mul_shoup_lazy(d, w);
                }
            }
            t <<= 1;
            h >>= 1;
        }
        // Last stage: x' = (x + y) * n_inv, y' = (x - y) * (w * n_inv); inputs < 2q so sums < 4q.
        let (lo, hi) = a.split_at_mut(self.n / 2);
        for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
            let s = *x + *y;
            let d = *x + two_q - *y;
            let mut r0 = m.mul_shoup_lazy(s, &self.n_inv);
            let mut r1 = m.mul_shoup_lazy(d, &self.w_n_inv);
            if r0 >= q {
                r0 -= q;
            }
            if r1 >= q {
                r1 -= q;
            }
            *x = r0;
            *y = r1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primes::ntt_primes;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn rand_poly(n: usize, q: u64, seed: u64) -> Vec<u64> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n).map(|_| rng.next_u64() % q).collect()
    }

    fn table(n: usize, bits: u32) -> NttTable {
        let q = ntt_primes(bits, n, 1).unwrap()[0];
        NttTable::new(n, Modulus::new(q).unwrap()).unwrap()
    }

    /// Schoolbook product in `Z_q[X]/(X^n + 1)`.
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

    fn check_conv(t: &NttTable, a: &[u64], b: &[u64]) {
        let m = t.modulus();
        let (mut fa, mut fb) = (a.to_vec(), b.to_vec());
        t.forward(&mut fa);
        t.forward(&mut fb);
        let mut fc: Vec<u64> = fa.iter().zip(&fb).map(|(&x, &y)| m.mul(x, y)).collect();
        t.inverse(&mut fc);
        assert_eq!(fc, naive_negacyclic(a, b, m));
    }

    #[test]
    fn roundtrip() {
        for (n, bits) in [(2, 20), (4, 30), (256, 40), (1024, 60), (4096, 60)] {
            let t = table(n, bits);
            let a = rand_poly(n, t.modulus().value(), n as u64);
            let mut b = a.clone();
            t.forward(&mut b);
            assert!(b.iter().all(|&x| x < t.modulus().value()));
            t.inverse(&mut b);
            assert_eq!(a, b, "n={n}");
        }
    }

    #[test]
    fn pointwise_product_is_negacyclic_convolution() {
        for n in [8usize, 256, 1024] {
            let t = table(n, 60);
            let q = t.modulus().value();
            check_conv(&t, &rand_poly(n, q, 1), &rand_poly(n, q, 2));
        }
    }

    #[test]
    fn worst_case_inputs() {
        let n = 1024;
        let t = table(n, 60);
        let q = t.modulus().value();
        let maxv = vec![q - 1; n];
        let zero = vec![0u64; n];
        check_conv(&t, &maxv, &maxv);
        check_conv(&t, &zero, &maxv);
        check_conv(&t, &zero, &zero);
        let mut z = zero.clone();
        t.forward(&mut z);
        assert_eq!(z, zero);
        let mut mx = maxv.clone();
        t.forward(&mut mx);
        t.inverse(&mut mx);
        assert_eq!(mx, maxv);
    }

    #[test]
    fn q_close_to_2_pow_60_n_2_pow_16() {
        let n = 1 << 16;
        let q = ntt_primes(60, n, 1).unwrap()[0];
        assert!(q > (1u64 << 60) - (1u64 << 40));
        let t = NttTable::new(n, Modulus::new(q).unwrap()).unwrap();
        for a in [
            rand_poly(n, q, 7),
            vec![q - 1; n],
            (0..n as u64).map(|i| q - 1 - (i & 1)).collect(),
        ] {
            let mut b = a.clone();
            t.forward(&mut b);
            assert!(b.iter().all(|&x| x < q));
            t.inverse(&mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn forward_output_order_is_bitreversed_evaluation() {
        for n in [16usize, 256] {
            let t = table(n, 50);
            let m = t.modulus();
            let q = m.value();
            let a = rand_poly(n, q, 99);
            let mut f = a.clone();
            t.forward(&mut f);
            let bits = n.trailing_zeros();
            for k in 0..n {
                let x = m.pow(t.psi(), 2 * k as u64 + 1);
                let mut acc = 0u64; // Horner
                for &c in a.iter().rev() {
                    acc = m.add(m.mul(acc, x), c);
                }
                assert_eq!(f[bitrev(k, bits)], acc, "n={n} k={k}");
            }
        }
    }

    #[test]
    fn psi_is_minimal_primitive_root() {
        for (n, q) in [(8usize, 17u64), (8, 97), (16, 193), (16, 12289)] {
            let m = Modulus::new(q).unwrap();
            let t = NttTable::new(n, m).unwrap();
            // Brute force: smallest g with order exactly 2n, i.e. g^n = -1 (2n is a power of 2).
            let expect = (1..q).find(|&g| m.pow(g, n as u64) == q - 1).unwrap();
            assert_eq!(t.psi(), expect, "n={n} q={q}");
        }
        assert_eq!(
            NttTable::new(8, Modulus::new(17).unwrap()).unwrap().psi(),
            3
        );
    }

    #[test]
    fn invalid_params() {
        let m = Modulus::new(ntt_primes(60, 1 << 10, 1).unwrap()[0]).unwrap();
        for n in [0usize, 1, 3, 1000] {
            assert!(matches!(
                NttTable::new(n, m),
                Err(HekitError::InvalidParams(_))
            ));
        }
        // q = 1 mod 2n fails: 17 = 1 mod 16 but not mod 32.
        let m17 = Modulus::new(17).unwrap();
        assert!(NttTable::new(8, m17).is_ok());
        assert!(NttTable::new(16, m17).is_err());
        // 1 mod 2n but composite: 65 = 5 * 13, 64 | 64.
        assert!(NttTable::new(32, Modulus::new(65).unwrap()).is_err());
        // n too large for q.
        assert!(NttTable::new(1 << 20, m).is_err());
    }
}
