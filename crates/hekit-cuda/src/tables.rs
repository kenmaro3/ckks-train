//! Host-side twiddle tables in the flat layout the kernels read (`kernels/ntt.cu`). Host-only,
//! no CUDA needed.
//!
//! The tables of [`hekit_core::NttTable`] are private, so they are rebuilt here from its public
//! `psi()` and `modulus()` with the same definitions (`roots[bitrev(i)] = psi^i`,
//! `inv_roots[bitrev(i)] = psi^-i`, Shoup companions `floor(w 2^64 / q)`,
//! `w_n_inv = inv_roots[1] * N^-1`). The CPU replay of the kernels in the tests checks the
//! result against `NttTable::forward` / `inverse` bit for bit, which also checks these tables.

use hekit_ckks::CkksParams;
use hekit_core::{HekitError, NttTable, Result};

/// Per-prime constants per entry of [`HostNttTables::consts`]:
/// `{ q, n_inv, n_inv_shoup, w_n_inv, w_n_inv_shoup, barrett_lo, barrett_hi }` (`NCONST` in
/// the kernels; `barrett = floor(2^128 / q)`, the constant of `Modulus::mul`, used by the
/// pointwise product in `kernels/pointwise.cu`).
pub const NCONST: usize = 7;

/// Twiddle tables of several primes, concatenated: prime `p` occupies `[p * N, (p + 1) * N)` of
/// each table and `[p * NCONST, (p + 1) * NCONST)` of `consts`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostNttTables {
    /// Ring degree `N`.
    pub n: usize,
    /// The primes, in table order.
    pub primes: Vec<u64>,
    /// `roots[p * N + i] = psi_p^bitrev(i) mod q_p`.
    pub roots: Vec<u64>,
    /// Shoup companions of `roots`.
    pub roots_shoup: Vec<u64>,
    /// `inv_roots[p * N + i] = psi_p^-bitrev(i) mod q_p`.
    pub inv_roots: Vec<u64>,
    /// Shoup companions of `inv_roots`.
    pub inv_roots_shoup: Vec<u64>,
    /// [`NCONST`] constants per prime.
    pub consts: Vec<u64>,
}

fn bitrev(i: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        i.reverse_bits() >> (usize::BITS - bits)
    }
}

impl HostNttTables {
    /// Tables for `tables` (all of the same degree), in the given order.
    /// `InvalidParams` if `tables` is empty or the degrees differ.
    pub fn new<'a>(tables: impl IntoIterator<Item = &'a NttTable>) -> Result<HostNttTables> {
        let tables: Vec<&NttTable> = tables.into_iter().collect();
        let Some(first) = tables.first() else {
            return Err(HekitError::InvalidParams(
                "device NTT tables need at least one prime".into(),
            ));
        };
        let n = first.n();
        if tables.iter().any(|t| t.n() != n) {
            return Err(HekitError::InvalidParams(
                "device NTT tables must share one degree".into(),
            ));
        }
        let bits = n.trailing_zeros();
        let len = n * tables.len();
        let mut out = HostNttTables {
            n,
            primes: Vec::with_capacity(tables.len()),
            roots: vec![0; len],
            roots_shoup: vec![0; len],
            inv_roots: vec![0; len],
            inv_roots_shoup: vec![0; len],
            consts: Vec::with_capacity(NCONST * tables.len()),
        };
        for (p, t) in tables.iter().enumerate() {
            let m = t.modulus();
            let psi = t.psi();
            let psi_inv = m
                .inv(psi)
                .ok_or_else(|| HekitError::InvalidParams("psi is not invertible".into()))?;
            let n_inv = m
                .inv(n as u64 % m.value())
                .ok_or_else(|| HekitError::InvalidParams("N is not invertible".into()))?;
            let base = p * n;
            let (mut w, mut w_inv) = (1u64, 1u64);
            for i in 0..n {
                let k = base + bitrev(i, bits);
                let s = m.shoup(w);
                out.roots[k] = s.value();
                out.roots_shoup[k] = s.companion();
                let s = m.shoup(w_inv);
                out.inv_roots[k] = s.value();
                out.inv_roots_shoup[k] = s.companion();
                w = m.mul(w, psi);
                w_inv = m.mul(w_inv, psi_inv);
            }
            let n_inv_s = m.shoup(n_inv);
            let w_n_inv_s = m.shoup(m.mul(out.inv_roots[base + 1], n_inv));
            let barrett = barrett_ratio(m.value());
            out.consts.extend([
                m.value(),
                n_inv_s.value(),
                n_inv_s.companion(),
                w_n_inv_s.value(),
                w_n_inv_s.companion(),
                barrett as u64,
                (barrett >> 64) as u64,
            ]);
            out.primes.push(m.value());
        }
        Ok(out)
    }

    /// Tables of a CKKS parameter set: the Q primes (`q_0, ..., q_L`, prime indices
    /// `0..=L`) followed by the special primes P (prime indices `L + 1, ...`).
    pub fn from_params(params: &CkksParams) -> Result<HostNttTables> {
        HostNttTables::new(
            params
                .basis_q()
                .ntt_tables()
                .iter()
                .chain(params.basis_p().ntt_tables()),
        )
    }

    /// Number of primes.
    pub fn num_primes(&self) -> usize {
        self.primes.len()
    }

    /// Bytes these tables occupy on the device (four `N`-entry tables per prime and the
    /// constants).
    pub fn device_bytes(&self) -> usize {
        device_bytes_for(self.n, self.num_primes())
    }
}

/// `floor(2^128 / q)` for `2 <= q < 2^60`, as in `hekit_core::Modulus::new` (whose constant is
/// private).
pub fn barrett_ratio(q: u64) -> u128 {
    let qq = q as u128;
    let mut ratio = u128::MAX / qq;
    if u128::MAX % qq + 1 == qq {
        ratio += 1;
    }
    ratio
}

/// Device bytes of the tables of `num_primes` primes of degree `n`: four `N`-entry u64 tables
/// and [`NCONST`] u64 constants per prime. For `n16_train` (36 primes, `N = 2^16`) about
/// 72 MiB. (The per-polynomial limb maps and conversion tables a device caches on use are
/// not included; they total about 1 MiB for `n16_train`.)
pub fn device_bytes_for(n: usize, num_primes: usize) -> usize {
    8 * (4 * num_primes * n + NCONST * num_primes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekit_core::Modulus;
    use hekit_core::primes::ntt_primes;

    #[test]
    fn layout_and_constants() {
        let n = 16;
        let qs = ntt_primes(30, n, 2).unwrap();
        let tabs: Vec<NttTable> = qs
            .iter()
            .map(|&q| NttTable::new(n, Modulus::new(q).unwrap()).unwrap())
            .collect();
        let h = HostNttTables::new(&tabs).unwrap();
        assert_eq!(h.primes, qs);
        assert_eq!(h.roots.len(), 2 * n);
        for (p, t) in tabs.iter().enumerate() {
            let m = t.modulus();
            let q = m.value();
            // roots[bitrev(1)] = roots[N/2] = psi, roots[0] = 1, inv_roots[N/2] * psi = 1.
            assert_eq!(h.roots[p * n], 1);
            assert_eq!(h.roots[p * n + n / 2], t.psi());
            assert_eq!(m.mul(h.inv_roots[p * n + n / 2], t.psi()), 1);
            for i in 0..n {
                let k = p * n + i;
                assert_eq!(h.roots_shoup[k], m.shoup(h.roots[k]).companion());
                assert_eq!(m.mul(h.roots[k], h.inv_roots[k]), 1);
            }
            let c = &h.consts[p * NCONST..(p + 1) * NCONST];
            assert_eq!(c[0], q);
            assert_eq!(m.mul(c[1], n as u64), 1);
            assert_eq!(c[2], m.shoup(c[1]).companion());
            assert_eq!(c[3], m.mul(h.inv_roots[p * n + 1], c[1]));
            assert_eq!(c[4], m.shoup(c[3]).companion());
            // floor(2^128 / q): ratio * q <= 2^128 < (ratio + 1) * q.
            let ratio = (c[5] as u128) | ((c[6] as u128) << 64);
            assert_eq!(ratio, barrett_ratio(q));
            let lo = ratio.wrapping_mul(q as u128);
            assert!(lo.wrapping_neg() < q as u128 && lo != 0);
        }
    }

    #[test]
    fn from_params_is_q_then_p() {
        let params = CkksParams::test_tiny();
        let h = HostNttTables::from_params(&params).unwrap();
        let expect: Vec<u64> = params.q().iter().chain(params.p()).copied().collect();
        assert_eq!(h.primes, expect);
        assert_eq!(h.n, params.n());
        assert_eq!(
            h.device_bytes(),
            8 * (4 * expect.len() * params.n() + NCONST * expect.len())
        );
    }

    #[test]
    fn rejects_empty_and_mixed_degrees() {
        assert!(matches!(
            HostNttTables::new(std::iter::empty()),
            Err(HekitError::InvalidParams(_))
        ));
        let q = ntt_primes(30, 32, 1).unwrap()[0];
        let a = NttTable::new(16, Modulus::new(q).unwrap()).unwrap();
        let b = NttTable::new(32, Modulus::new(q).unwrap()).unwrap();
        assert!(matches!(
            HostNttTables::new([&a, &b]),
            Err(HekitError::InvalidParams(_))
        ));
    }
}
