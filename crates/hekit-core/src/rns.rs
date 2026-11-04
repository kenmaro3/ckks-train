//! RNS basis: an ordered list of NTT-friendly primes `q_0, ..., q_L` sharing one ring degree
//! `N`, with one [`NttTable`] per prime and a cache of Galois permutations for NTT-form
//! automorphisms.
//!
//! # Galois permutation (NTT form)
//!
//! [`NttTable::forward`] stores `a(psi^(2k+1))` at index `bitrev(k)`. The automorphism
//! `sigma_g: a(X) -> a(X^g)` (`g` odd, `g < 2N`) therefore satisfies
//! `sigma_g(a)(psi^(2k+1)) = a(psi^(g(2k+1) mod 2N))`, and `g(2k+1) mod 2N` is again odd,
//! `= 2k'+1`. Writing `i = bitrev(k)`, the NTT-form automorphism is the index permutation
//!
//! ```text
//! out[i] = in[perm[i]],   perm[i] = bitrev( ((g * (2*bitrev(i) + 1)) mod 2N - 1) / 2 )
//! ```
//!
//! which is exactly SEAL 4.1's `GaloisTool::generate_table_ntt` / `apply_galois_ntt`. The
//! permutation depends only on `N` and `g` (not on the prime), so one table serves every limb.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{HekitError, Result};
use crate::modulus::Modulus;
use crate::ntt::NttTable;

/// Reverses the `bits` low bits of `i` (`bitrev(i, 0) = 0`).
pub(crate) fn bitrev(i: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        i.reverse_bits() >> (usize::BITS - bits)
    }
}

/// An RNS basis `{q_0, ..., q_L}` for the ring `Z[X]/(X^N + 1)`.
///
/// Built once and shared through `Arc` by every [`RnsPoly`](crate::poly::RnsPoly) over it.
#[derive(Debug)]
pub struct RnsBasis {
    moduli: Vec<Modulus>,
    ntt: Vec<NttTable>,
    n: usize,
    /// NTT-form Galois permutations, computed on first use per Galois element.
    galois_ntt: Mutex<HashMap<u64, Arc<[u32]>>>,
}

impl RnsBasis {
    /// Builds a basis of degree `n` over `moduli` (in the given order; limb `i` uses
    /// `moduli[i]`).
    ///
    /// Fails with `InvalidParams` when `moduli` is empty, contains a duplicate, `n` is not a
    /// power of two `>= 2` (or `>= 2^32`), or some modulus is not a prime `= 1 (mod 2n)`
    /// below `2^60`.
    pub fn new(n: usize, moduli: &[u64]) -> Result<Arc<RnsBasis>> {
        if moduli.is_empty() {
            return Err(HekitError::InvalidParams(
                "RNS basis needs at least one modulus".into(),
            ));
        }
        if n as u64 >= 1u64 << 32 {
            return Err(HekitError::InvalidParams(format!(
                "ring degree {n} must be below 2^32"
            )));
        }
        for (i, q) in moduli.iter().enumerate() {
            if moduli[..i].contains(q) {
                return Err(HekitError::InvalidParams(format!(
                    "duplicate modulus {q} in RNS basis"
                )));
            }
        }
        let mut ms = Vec::with_capacity(moduli.len());
        let mut tables = Vec::with_capacity(moduli.len());
        for &q in moduli {
            let m = Modulus::new(q)?;
            tables.push(NttTable::new(n, m)?);
            ms.push(m);
        }
        Ok(Arc::new(RnsBasis {
            moduli: ms,
            ntt: tables,
            n,
            galois_ntt: Mutex::new(HashMap::new()),
        }))
    }

    /// The ring degree `N`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of primes in the basis (`L + 1`).
    pub fn len(&self) -> usize {
        self.moduli.len()
    }

    /// Always `false`: a basis has at least one prime.
    pub fn is_empty(&self) -> bool {
        self.moduli.is_empty()
    }

    /// The primes, in limb order.
    pub fn moduli(&self) -> &[Modulus] {
        &self.moduli
    }

    /// The NTT tables, in limb order.
    pub fn ntt_tables(&self) -> &[NttTable] {
        &self.ntt
    }

    /// The NTT-form Galois permutation for `galois_elt` (see the module docs):
    /// `sigma_g(a)_ntt[i] = a_ntt[perm[i]]`.
    ///
    /// `galois_elt` must be odd and `< 2N` (`debug_assert`ed; in release it is reduced
    /// mod `2N`). Computed on first use and cached in the basis; later calls return the
    /// same `Arc`.
    pub fn galois_permutation(&self, galois_elt: u64) -> Arc<[u32]> {
        let two_n = 2 * self.n as u64;
        debug_assert!(
            galois_elt % 2 == 1 && galois_elt < two_n,
            "Galois element {galois_elt} must be odd and < 2N = {two_n}"
        );
        let g = galois_elt % two_n;
        let mut cache = self.galois_ntt.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .entry(g)
            .or_insert_with(|| galois_permutation_ntt(self.n, g))
            .clone()
    }
}

/// SEAL 4.1 `GaloisTool::generate_table_ntt`: for `i` in `N..2N`, with `logN + 1`-bit
/// reversal, `table[i - N] = bitrev_{logN}(((g * bitrev_{logN+1}(i)) >> 1) & (N - 1))`.
/// For `i = N + j`, `bitrev_{logN+1}(i) = 2 * bitrev_{logN}(j) + 1`, so this is the formula
/// in the module docs.
fn galois_permutation_ntt(n: usize, g: u64) -> Arc<[u32]> {
    let log_n = n.trailing_zeros();
    let mask = n as u64 - 1;
    (n..2 * n)
        .map(|i| {
            let reversed = bitrev(i, log_n + 1) as u64;
            let index_raw = (g.wrapping_mul(reversed) >> 1) & mask;
            bitrev(index_raw as usize, log_n) as u32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primes::ntt_primes;

    #[test]
    fn new_rejects_bad_params() {
        let n = 1 << 10;
        let ps = ntt_primes(50, n, 3).unwrap();
        assert!(RnsBasis::new(n, &ps).is_ok());
        let bad = |r: Result<Arc<RnsBasis>>| matches!(r, Err(HekitError::InvalidParams(_)));
        assert!(bad(RnsBasis::new(n, &[])));
        assert!(bad(RnsBasis::new(n, &[ps[0], ps[1], ps[0]])));
        assert!(bad(RnsBasis::new(1000, &ps)));
        assert!(bad(RnsBasis::new(n, &[ps[0], 17])));
        assert!(bad(RnsBasis::new(n, &[ps[0], 1 << 61])));
        // Composite = 1 mod 2n: 2049 = 3 * 683.
        assert!(bad(RnsBasis::new(n, &[ps[0], 2049])));
    }

    #[test]
    fn basis_accessors() {
        let n = 64;
        let ps = ntt_primes(40, n, 3).unwrap();
        let b = RnsBasis::new(n, &ps).unwrap();
        assert_eq!(b.n(), n);
        assert_eq!(b.len(), 3);
        assert!(!b.is_empty());
        let qs: Vec<u64> = b.moduli().iter().map(|m| m.value()).collect();
        assert_eq!(qs, ps);
        assert_eq!(b.ntt_tables().len(), 3);
        assert!(b.ntt_tables().iter().all(|t| t.n() == n));
    }

    #[test]
    fn galois_permutation_matches_direct_formula() {
        for n in [2usize, 8, 64, 1024] {
            let b = RnsBasis::new(n, &ntt_primes(40, n, 1).unwrap()).unwrap();
            let bits = n.trailing_zeros();
            let two_n = 2 * n as u64;
            let mut gs = vec![1u64, 3 % two_n, two_n - 1];
            let mut g = 1u64;
            for _ in 0..6 {
                g = g * 5 % two_n;
                gs.push(g);
            }
            for g in gs {
                let perm = b.galois_permutation(g);
                assert_eq!(perm.len(), n);
                for (i, &p) in perm.iter().enumerate() {
                    let k = bitrev(i, bits) as u64;
                    let odd = g * (2 * k + 1) % two_n;
                    assert_eq!(p as usize, bitrev(((odd - 1) / 2) as usize, bits));
                }
                // A permutation of 0..n.
                let mut seen = vec![false; n];
                perm.iter().for_each(|&p| seen[p as usize] = true);
                assert!(seen.iter().all(|&s| s), "n={n} g={g}");
            }
        }
    }

    #[test]
    fn galois_permutation_is_cached() {
        let n = 256;
        let b = RnsBasis::new(n, &ntt_primes(40, n, 2).unwrap()).unwrap();
        let p1 = b.galois_permutation(5);
        let p2 = b.galois_permutation(5);
        assert!(Arc::ptr_eq(&p1, &p2));
        let id = b.galois_permutation(1);
        assert!(id.iter().enumerate().all(|(i, &p)| p as usize == i));
    }
}
