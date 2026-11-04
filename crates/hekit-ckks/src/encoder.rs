//! CKKS encoder: the canonical embedding with Microsoft SEAL 4.1's slot ordering.
//!
//! # Embedding and slot order (SEAL 4.1 `CKKSEncoder`)
//!
//! Let `zeta = exp(2 pi i / 2N)` and `g = GALOIS_GENERATOR = 3`. A plaintext polynomial
//! `m(X)` with real integer coefficients decodes to the `N/2` slots
//!
//! ```text
//! slot_j = m(zeta^(g^j mod 2N)) / Delta,     j = 0 .. N/2 - 1,
//! ```
//!
//! and the remaining evaluation points `zeta^(-g^j)` carry the complex conjugates. Hence the
//! automorphism `X -> X^(g^k mod 2N)` maps `slot_j` to `slot_{j+k mod N/2}` (rotation **left**
//! by `k`), and `X -> X^(2N-1)` conjugates every slot. [`Encoder::rot_group`] lists
//! `g^j mod 2N`, so `rot_group()[k]` is the Galois element of a left rotation by `k`.
//!
//! SEAL uses generator **3** for CKKS (`CKKSEncoder::matrix_reps_index_map_` and
//! `GaloisTool::generator_`, both verified in the SEAL 4.1.2 sources), not 5. Both 3 and 5
//! generate an order-`N/2` subgroup of `Z_2N^*` that misses `-1`, so either gives a valid
//! embedding, but the slot orders differ; SEAL interoperability requires 3.
//!
//! Concretely SEAL stores slot `j` at FFT index `bitrev_{logN}((g^j - 1) / 2)` and its
//! conjugate at `bitrev_{logN}((2N - g^j - 1) / 2)` (`index_map` here), and runs the
//! bit-reversed negacyclic DWT of `util::DWTHandler` with the root table of
//! `util::ComplexRoots` (8-fold symmetry of `exp(2 pi i k / 2N)`). Both are ported verbatim,
//! so encodings agree with SEAL coefficient by coefficient (pinned by a known-answer test).
//! This ported code is derived from Microsoft SEAL 4.1.2 (Copyright (c) Microsoft
//! Corporation, MIT License); its copyright and permission notice is reproduced in
//! `THIRD_PARTY_NOTICES` at the repository root.
//!
//! # Encoding
//!
//! Values -> inverse DWT (scaled by `Delta / N`) -> round half away from zero (as
//! `std::round`) -> reduce into limbs `0..=level` -> NTT. Every rounded `f64` is an exact
//! integer, reduced exactly into each limb: `|c| < 2^63` through `u64`, larger `|c|` through
//! its exact decomposition `mantissa * 2^exponent` (so no `i128`/bigint bound applies beyond
//! the modulus itself). Encoding fails with `InvalidParams` when `max |c| >= Q_level / 2`
//! (checked as `2^(log2 Q_level - 1)`), the analogue of SEAL's "encoded values are too
//! large".
//!
//! # Decoding
//!
//! NTT -> coefficient form -> exact CRT composition by Garner's algorithm with **balanced**
//! mixed-radix digits `d_k` in `[-(q_k-1)/2, (q_k-1)/2]`; the balanced representation is
//! exactly the centered representative in `(-Q/2, Q/2)`, so no multi-precision integer is
//! needed. The value is then evaluated in `f64` by Horner's rule on `d_k / Delta` (top digit
//! first). Since `|sum_{j<k} d_j P_j| <= (P_k - 1) / 2`, the top non-zero digit dominates and
//! the relative error is `O(level * 2^-53)`. Values beyond the `f64` range (garbage, e.g. a
//! uniformly random polynomial at a high level) decode to `+-inf`. Finally the forward DWT
//! evaluates at the roots and the slots are read through `index_map`.
//!
//! # Floating-point accuracy
//!
//! The DWTs are `f64`; their error is `O(log N * 2^-53)` relative to the largest coefficient
//! (about `2^-40` absolute for `Delta = 2^40`, `|v| <= 1`), far below the `0.5 / Delta`
//! rounding of the integer coefficients, which dominates the round-trip error
//! (`~2^-32` for `seal_2020`, see the tests).

use std::sync::Arc;

use hekit_core::error::{HekitError, Result};
use hekit_core::poly::RnsPoly;
use num_complex::Complex64;

use crate::params::CkksParams;
use crate::plaintext::Plaintext;

/// `2^63` (exact in `f64`).
const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;

/// Generator of the slot-rotation subgroup of `Z_2N^*` (SEAL 4.1 uses 3 for CKKS).
/// The Galois element of a left rotation by `k` slots is `3^k mod 2N`.
pub const GALOIS_GENERATOR: u64 = 3;

/// Reverses the `bits` low bits of `i`.
fn bitrev(i: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        i.reverse_bits() >> (usize::BITS - bits)
    }
}

/// SEAL `util::ComplexRoots::get_root`: `exp(2 pi i index / m)` from a table of the first
/// `m/8 + 1` roots, using the 8-fold symmetry (requires `m >= 8`).
fn complex_root(table: &[Complex64], m: usize, index: usize) -> Complex64 {
    let index = index & (m - 1);
    if index <= m / 8 {
        table[index]
    } else if index <= m / 4 {
        let a = table[m / 4 - index];
        Complex64::new(a.im, a.re)
    } else if index <= m / 2 {
        -complex_root(table, m, m / 2 - index).conj()
    } else if index <= 3 * m / 4 {
        -complex_root(table, m, index - m / 2)
    } else {
        complex_root(table, m, m - index).conj()
    }
}

/// SEAL `DWTHandler::transform_to_rev` (no scalar): input in normal order, output in
/// bit-reversed order; `roots[1..n]` are the root powers in bit-reversed order.
fn transform_to_rev(values: &mut [Complex64], roots: &[Complex64]) {
    let n = values.len();
    let mut ri = 0;
    let mut gap = n >> 1;
    let mut m = 1;
    while m < n {
        let mut offset = 0;
        for _ in 0..m {
            ri += 1;
            let r = roots[ri];
            for x in offset..offset + gap {
                let u = values[x];
                let v = values[x + gap] * r;
                values[x] = u + v;
                values[x + gap] = u - v;
            }
            offset += gap << 1;
        }
        gap >>= 1;
        m <<= 1;
    }
}

/// SEAL `DWTHandler::transform_from_rev` with a scalar: input in bit-reversed order, output
/// in normal order multiplied by `scalar`; `roots[1..n]` in SEAL's scrambled inverse order.
fn transform_from_rev(values: &mut [Complex64], roots: &[Complex64], scalar: f64) {
    let n = values.len();
    let mut ri = 0;
    let mut gap = 1;
    let mut m = n >> 1;
    while m > 1 {
        let mut offset = 0;
        for _ in 0..m {
            ri += 1;
            let r = roots[ri];
            for x in offset..offset + gap {
                let u = values[x];
                let v = values[x + gap];
                values[x] = u + v;
                values[x + gap] = (u - v) * r;
            }
            offset += gap << 1;
        }
        gap <<= 1;
        m >>= 1;
    }
    ri += 1;
    let scaled_r = roots[ri] * scalar;
    for x in 0..gap {
        let u = values[x];
        let v = values[x + gap];
        values[x] = (u + v) * scalar;
        values[x + gap] = (u - v) * scaled_r;
    }
}

/// `d mod q` for a signed digit `d`.
fn signed_mod(d: i64, q: u64) -> u64 {
    let r = d.unsigned_abs() % q;
    if d < 0 && r != 0 { q - r } else { r }
}

/// CKKS encoder/decoder for one parameter set (SEAL 4.1-compatible; see the module docs).
#[derive(Debug, Clone)]
pub struct Encoder {
    params: Arc<CkksParams>,
    /// SEAL `root_powers_`: `roots[i] = zeta^bitrev(i)`, `i in 1..N` (decoding DWT).
    roots: Vec<Complex64>,
    /// SEAL `inv_root_powers_`: `conj(zeta^(bitrev(i-1) + 1))`, `i in 1..N` (encoding DWT).
    inv_roots: Vec<Complex64>,
    /// `rot_group[j] = 3^j mod 2N`, `j < N/2`.
    rot_group: Vec<usize>,
    /// SEAL `matrix_reps_index_map_` (length `N`): DWT index of slot `j` (`j < N/2`) and of
    /// its conjugate (`N/2 + j`).
    index_map: Vec<usize>,
    /// Garner tables: `crt_prefix[k][j] = (q_0 ... q_{j-1}) mod q_k` for `j < k`.
    crt_prefix: Vec<Vec<u64>>,
    /// Garner tables: `crt_prefix_inv[k] = (q_0 ... q_{k-1})^{-1} mod q_k`.
    crt_prefix_inv: Vec<u64>,
}

impl Encoder {
    /// Precomputes the slot permutation, the DWT root tables and the CRT tables.
    pub fn new(params: Arc<CkksParams>) -> Encoder {
        let n = params.n();
        let slots = params.slots();
        let log_n = params.log_n();
        let m = 2 * n;

        let mut index_map = vec![0usize; n];
        let mut rot_group = Vec::with_capacity(slots);
        let mut pos = 1usize;
        for i in 0..slots {
            index_map[i] = bitrev((pos - 1) >> 1, log_n);
            index_map[slots | i] = bitrev((m - pos - 1) >> 1, log_n);
            rot_group.push(pos);
            pos = (pos * GALOIS_GENERATOR as usize) & (m - 1);
        }

        let zero = Complex64::new(0.0, 0.0);
        let mut roots = vec![zero; n];
        let mut inv_roots = vec![zero; n];
        if m >= 8 {
            let table: Vec<Complex64> = (0..=m / 8)
                .map(|i| {
                    Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * i as f64 / m as f64)
                })
                .collect();
            for i in 1..n {
                roots[i] = complex_root(&table, m, bitrev(i, log_n));
                inv_roots[i] = complex_root(&table, m, bitrev(i - 1, log_n) + 1).conj();
            }
        } else {
            // m = 4 (N = 2), as in SEAL.
            roots[1] = Complex64::new(0.0, 1.0);
            inv_roots[1] = Complex64::new(0.0, -1.0);
        }

        let moduli = params.basis_q().moduli();
        let mut crt_prefix = Vec::with_capacity(moduli.len());
        let mut crt_prefix_inv = Vec::with_capacity(moduli.len());
        for (k, mk) in moduli.iter().enumerate() {
            let qk = mk.value();
            let mut p = 1 % qk;
            let mut row = Vec::with_capacity(k);
            for mj in &moduli[..k] {
                row.push(p);
                p = mk.mul(p, mj.value() % qk);
            }
            crt_prefix.push(row);
            crt_prefix_inv.push(mk.inv(p).expect("RNS primes are distinct"));
        }

        Encoder {
            params,
            roots,
            inv_roots,
            rot_group,
            index_map,
            crt_prefix,
            crt_prefix_inv,
        }
    }

    /// The parameter set.
    pub fn params(&self) -> &Arc<CkksParams> {
        &self.params
    }

    /// `rot_group()[k] = 3^k mod 2N` for `k < N/2`: the Galois element whose automorphism
    /// `X -> X^(3^k)` rotates the slots left by `k` (slot `j` receives slot `j + k`).
    pub fn rot_group(&self) -> &[usize] {
        &self.rot_group
    }

    /// Encodes real values (zero-padded to `N/2` slots) at `scale` and `level`; the result is
    /// in NTT form with `level + 1` limbs.
    ///
    /// Errors (`InvalidParams`): `values.len() > N/2`, `level > max_level`, `scale` not
    /// positive and finite, a non-finite value, or a coefficient `|c| >= Q_level / 2`.
    pub fn encode(&self, values: &[f64], scale: f64, level: usize) -> Result<Plaintext> {
        self.check_len(values.len())?;
        let z: Vec<Complex64> = values.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        self.encode_complex(&z, scale, level)
    }

    /// Encodes complex values (zero-padded to `N/2` slots); see [`Encoder::encode`].
    pub fn encode_complex(
        &self,
        values: &[Complex64],
        scale: f64,
        level: usize,
    ) -> Result<Plaintext> {
        let bad = |m: String| Err(HekitError::InvalidParams(m));
        self.check_len(values.len())?;
        if level > self.params.max_level() {
            return bad(format!(
                "level {level} exceeds max_level {}",
                self.params.max_level()
            ));
        }
        if !(scale.is_finite() && scale > 0.0) {
            return bad(format!("scale {scale} must be positive and finite"));
        }
        if let Some(z) = values.iter().find(|z| !z.is_finite()) {
            return bad(format!("non-finite value {z}"));
        }

        let n = self.params.n();
        let slots = self.params.slots();
        let mut cv = vec![Complex64::new(0.0, 0.0); n];
        for (i, &z) in values.iter().enumerate() {
            cv[self.index_map[i]] = z;
            cv[self.index_map[slots + i]] = z.conj();
        }
        transform_from_rev(&mut cv, &self.inv_roots, scale / n as f64);
        let coeffs: Vec<f64> = cv.iter().map(|c| c.re.round()).collect();

        let basis = self.params.basis_q();
        let moduli = &basis.moduli()[..=level];
        let log_q: f64 = moduli.iter().map(|m| (m.value() as f64).log2()).sum();
        let limit = (log_q - 1.0).exp2();
        if let Some(c) = coeffs.iter().find(|c| c.is_nan() || c.abs() >= limit) {
            return bad(format!(
                "encoded coefficient {c:e} does not fit in Q/2 ~ 2^{:.1} at level {level}",
                log_q - 1.0
            ));
        }

        let mut limbs = vec![vec![0u64; n]; level + 1];
        for (i, &c) in coeffs.iter().enumerate() {
            let neg = c < 0.0;
            let a = c.abs();
            for (limb, m) in limbs.iter_mut().zip(moduli) {
                let q = m.value();
                let r = if a < TWO_POW_63 {
                    // |c| < 2^63: exact as u64.
                    (a as u64) % q
                } else {
                    // a = mantissa * 2^exponent exactly (a >= 2^63 is a normal f64).
                    let bits = a.to_bits();
                    let exponent = ((bits >> 52) & 0x7ff) - 1075;
                    let mantissa = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
                    m.mul(mantissa % q, m.pow(2, exponent))
                };
                limb[i] = if neg && r != 0 { q - r } else { r };
            }
        }
        let mut poly = RnsPoly::from_limbs(basis, limbs, false)?;
        poly.to_ntt();
        Plaintext::new(poly, scale)
    }

    /// Decodes the real parts of the `N/2` slots.
    ///
    /// # Panics
    /// If `pt` was not built over this encoder's modulus chain.
    pub fn decode(&self, pt: &Plaintext) -> Vec<f64> {
        self.decode_complex(pt).into_iter().map(|z| z.re).collect()
    }

    /// Decodes the `N/2` complex slots.
    ///
    /// # Panics
    /// If `pt` was not built over this encoder's modulus chain.
    pub fn decode_complex(&self, pt: &Plaintext) -> Vec<Complex64> {
        let n = self.params.n();
        let slots = self.params.slots();
        let mut poly = pt.poly().clone();
        let level = poly.level();
        let moduli = &self.params.basis_q().moduli()[..=level.min(self.params.max_level())];
        assert!(
            poly.basis().n() == n && poly.basis().moduli()[..=level] == *moduli,
            "plaintext is not over this encoder's modulus chain"
        );
        if poly.is_ntt_form() {
            poly.to_coeff();
        }
        let limbs = poly.limbs();
        let inv_scale = 1.0 / pt.scale();
        let q_f64: Vec<f64> = moduli.iter().map(|m| m.value() as f64).collect();

        let mut res = vec![Complex64::new(0.0, 0.0); n];
        let mut digits = vec![0i64; level + 1];
        for (i, out) in res.iter_mut().enumerate() {
            // Garner with balanced digits: X = sum_k d_k * (q_0 ... q_{k-1}).
            for k in 0..=level {
                let mk = &moduli[k];
                let qk = mk.value();
                let mut s = 0u64;
                for (&d, &p) in digits[..k].iter().zip(&self.crt_prefix[k]) {
                    s = mk.add(s, mk.mul(signed_mod(d, qk), p));
                }
                let t = mk.mul(mk.sub(limbs[k][i], s), self.crt_prefix_inv[k]);
                digits[k] = if t > qk / 2 {
                    t as i64 - qk as i64
                } else {
                    t as i64
                };
            }
            let mut acc = digits[level] as f64 * inv_scale;
            for k in (0..level).rev() {
                acc = acc * q_f64[k] + digits[k] as f64 * inv_scale;
            }
            *out = Complex64::new(acc, 0.0);
        }

        transform_to_rev(&mut res, &self.roots);
        (0..slots).map(|j| res[self.index_map[j]]).collect()
    }

    fn check_len(&self, len: usize) -> Result<()> {
        if len > self.params.slots() {
            return Err(HekitError::InvalidParams(format!(
                "{len} values exceed the {} slots",
                self.params.slots()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekit_core::error::HekitError;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    fn rng(seed: u8) -> ChaCha20Rng {
        ChaCha20Rng::from_seed([seed; 32])
    }

    /// Uniform in [-1, 1).
    fn unit(rng: &mut ChaCha20Rng) -> f64 {
        (rng.next_u64() >> 11) as f64 * 2f64.powi(-52) - 1.0
    }

    fn rand_real(rng: &mut ChaCha20Rng, len: usize) -> Vec<f64> {
        (0..len).map(|_| unit(rng)).collect()
    }

    fn rand_complex(rng: &mut ChaCha20Rng, len: usize) -> Vec<Complex64> {
        (0..len)
            .map(|_| Complex64::new(unit(rng), unit(rng)))
            .collect()
    }

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    fn max_err_c(a: &[Complex64], b: &[Complex64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    fn small_params(log_n: u32) -> Arc<CkksParams> {
        Arc::new(CkksParams::new(log_n, &[60, 40, 40], &[60], 40, 1).unwrap())
    }

    /// Coefficient-form automorphism `X -> X^g` applied to a plaintext.
    fn galois_coeff(pt: &Plaintext, g: u64) -> Plaintext {
        let mut p = pt.poly().clone();
        p.to_coeff();
        let mut out = p.automorphism(g);
        out.to_ntt();
        Plaintext::new(out, pt.scale()).unwrap()
    }

    /// Centered coefficients of limb 0 (coefficient form).
    fn centered_coeffs(pt: &Plaintext) -> Vec<i64> {
        let mut p = pt.poly().clone();
        p.to_coeff();
        let q = p.basis().moduli()[0].value();
        p.limbs()[0]
            .iter()
            .map(|&c| {
                if c > q / 2 {
                    -((q - c) as i64)
                } else {
                    c as i64
                }
            })
            .collect()
    }

    const TWO_M30: f64 = 9.313225746154785e-10; // 2^-30

    #[test]
    fn roundtrip_seal_2020_top_level() {
        let params = Arc::new(CkksParams::seal_2020());
        let enc = Encoder::new(Arc::clone(&params));
        let v = rand_real(&mut rng(1), params.slots());
        let pt = enc.encode(&v, 2f64.powi(40), params.max_level()).unwrap();
        assert_eq!(pt.level(), 9);
        assert_eq!(pt.scale(), 2f64.powi(40));
        assert!(pt.poly().is_ntt_form());
        let e = max_err(&enc.decode(&pt), &v);
        eprintln!("seal_2020 level 9 max abs error = 2^{:.2}", e.log2());
        assert!(e < TWO_M30, "max error {e:e}");
    }

    #[test]
    fn roundtrip_seal_2020_level_0_and_intermediate() {
        let params = Arc::new(CkksParams::seal_2020());
        let enc = Encoder::new(Arc::clone(&params));
        let v = rand_real(&mut rng(2), params.slots());
        for level in [0, 4] {
            let pt = enc.encode(&v, 2f64.powi(40), level).unwrap();
            assert_eq!(pt.level(), level);
            assert_eq!(pt.poly().limbs().len(), level + 1);
            let e = max_err(&enc.decode(&pt), &v);
            eprintln!("seal_2020 level {level} max abs error = 2^{:.2}", e.log2());
            assert!(e < TWO_M30, "level {level}: max error {e:e}");
        }
    }

    #[test]
    fn roundtrip_n16_train_top_level() {
        let params = Arc::new(CkksParams::n16_train());
        let enc = Encoder::new(Arc::clone(&params));
        let v = rand_real(&mut rng(3), params.slots());
        let pt = enc.encode(&v, params.scale(), params.max_level()).unwrap();
        let e = max_err(&enc.decode(&pt), &v);
        eprintln!(
            "n16_train level {} max abs error = 2^{:.2}",
            params.max_level(),
            e.log2()
        );
        assert!(e < 2f64.powi(-24), "max error {e:e}");
    }

    #[test]
    fn roundtrip_complex() {
        let params = Arc::new(CkksParams::seal_2020());
        let enc = Encoder::new(Arc::clone(&params));
        let v = rand_complex(&mut rng(4), params.slots());
        let pt = enc.encode_complex(&v, 2f64.powi(40), 3).unwrap();
        let e = max_err_c(&enc.decode_complex(&pt), &v);
        eprintln!(
            "seal_2020 complex level 3 max abs error = 2^{:.2}",
            e.log2()
        );
        assert!(e < TWO_M30, "max error {e:e}");
        // decode() returns the real parts.
        let re: Vec<f64> = v.iter().map(|z| z.re).collect();
        assert!(max_err(&enc.decode(&pt), &re) < TWO_M30);
    }

    /// Review Focus #3: zeros and tiny values decode within 2^-30.
    #[test]
    fn zero_and_tiny_values() {
        let params = Arc::new(CkksParams::seal_2020());
        let enc = Encoder::new(Arc::clone(&params));
        let delta = 2f64.powi(40);
        for level in [0, params.max_level()] {
            let zeros = enc.decode(&enc.encode(&[0.0; 8], delta, level).unwrap());
            assert_eq!(zeros.len(), params.slots());
            assert!(
                zeros.iter().all(|x| x.abs() < TWO_M30),
                "zeros at level {level}"
            );

            let tiny = enc.decode(&enc.encode(&[1e-12], delta, level).unwrap());
            let mut want = vec![0.0; params.slots()];
            want[0] = 1e-12;
            assert!(max_err(&tiny, &want) < TWO_M30, "1e-12 at level {level}");

            let empty = enc.decode(&enc.encode(&[], delta, level).unwrap());
            assert!(
                empty.iter().all(|x| x.abs() < TWO_M30),
                "empty at level {level}"
            );
        }
    }

    #[test]
    fn short_input_is_zero_padded() {
        let params = small_params(6);
        let enc = Encoder::new(Arc::clone(&params));
        let short = [0.75, -0.5, 0.125];
        let mut padded = vec![0.0; params.slots()];
        padded[..3].copy_from_slice(&short);
        let a = enc.encode(&short, 2f64.powi(40), 2).unwrap();
        let b = enc.encode(&padded, 2f64.powi(40), 2).unwrap();
        assert_eq!(a.poly().limbs(), b.poly().limbs());
        assert!(max_err(&enc.decode(&a), &padded) < TWO_M30);
    }

    #[test]
    fn invalid_inputs_rejected() {
        let params = small_params(6);
        let enc = Encoder::new(Arc::clone(&params));
        let d = 2f64.powi(40);
        let too_many = vec![0.5; params.slots() + 1];
        assert!(matches!(
            enc.encode(&too_many, d, 2),
            Err(HekitError::InvalidParams(_))
        ));
        let too_many_c = vec![Complex64::new(0.5, 0.0); params.slots() + 1];
        assert!(enc.encode_complex(&too_many_c, d, 2).is_err());
        assert!(enc.encode(&[0.5], d, params.max_level() + 1).is_err());
        for bad_scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                enc.encode(&[0.5], bad_scale, 2).is_err(),
                "scale {bad_scale}"
            );
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(enc.encode(&[bad], d, 2).is_err(), "value {bad}");
        }
        // Level 0: Q = q_0 ~ 2^60, so |coefficients| ~ 2^40 * 2^30 cannot be represented.
        assert!(enc.encode(&[2f64.powi(30)], d, 0).is_err());
        // ... but the same value fits at level 2 (Q ~ 2^140).
        let big = enc.encode(&[2f64.powi(30)], d, 2).unwrap();
        assert!((enc.decode(&big)[0] - 2f64.powi(30)).abs() < 1e-3);
    }

    /// Coefficients larger than 2^63 (and 2^127) are reduced exactly into every limb.
    #[test]
    fn large_coefficients_exact() {
        let params = Arc::new(CkksParams::seal_2020());
        let enc = Encoder::new(Arc::clone(&params));
        let mut r = rng(5);
        for (mag, delta) in [
            (2f64.powi(30), 2f64.powi(40)),
            (2f64.powi(100), 2f64.powi(40)),
        ] {
            let v: Vec<f64> = rand_real(&mut r, params.slots())
                .into_iter()
                .map(|x| x * mag)
                .collect();
            let pt = enc.encode(&v, delta, params.max_level()).unwrap();
            let e = max_err(&enc.decode(&pt), &v);
            // Relative precision limited by f64 (53 bits) on coefficients of size mag * delta.
            assert!(e / mag < 2f64.powi(-40), "mag {mag:e}: error {e:e}");
        }
    }

    #[test]
    fn rot_group_is_powers_of_three() {
        let params = small_params(6);
        let enc = Encoder::new(Arc::clone(&params));
        let two_n = 2 * params.n();
        let mut g = 1usize;
        for (i, &r) in enc.rot_group().iter().enumerate() {
            assert_eq!(r, g, "rot_group[{i}]");
            g = g * 3 % two_n; // literal 3 (SEAL), not the constant
        }
        assert_eq!(enc.rot_group().len(), params.slots());
        assert_eq!(g, 1, "3 has order N/2 mod 2N");
    }

    /// Pins the slot ordering for Tasks 18/23: `X -> X^(3^k mod 2N)` rotates slots left by k.
    #[test]
    fn galois_3_pow_k_rotates_left() {
        for log_n in [3, 6] {
            let params = small_params(log_n);
            let enc = Encoder::new(Arc::clone(&params));
            let slots = params.slots();
            let v = rand_complex(&mut rng(6), slots);
            let pt = enc.encode_complex(&v, 2f64.powi(40), 2).unwrap();
            for k in [0, 1, 2, 5 % slots, slots - 1] {
                let g = enc.rot_group()[k] as u64;
                let got = enc.decode_complex(&galois_coeff(&pt, g));
                let want: Vec<Complex64> = (0..slots).map(|i| v[(i + k) % slots]).collect();
                let e = max_err_c(&got, &want);
                assert!(e < TWO_M30, "N=2^{log_n}, k={k}: error {e:e}");
                // The NTT-form permutation (hekit-core) agrees with the coefficient form.
                let ntt = Plaintext::new(pt.poly().automorphism(g), pt.scale()).unwrap();
                assert!(max_err_c(&enc.decode_complex(&ntt), &want) < TWO_M30);
            }
        }
    }

    #[test]
    fn galois_3_pow_k_rotates_left_seal_2020() {
        let params = Arc::new(CkksParams::seal_2020());
        let enc = Encoder::new(Arc::clone(&params));
        let slots = params.slots();
        let v = rand_real(&mut rng(7), slots);
        let pt = enc.encode(&v, 2f64.powi(40), params.max_level()).unwrap();
        for k in [1, 1234] {
            let got = enc.decode(&galois_coeff(&pt, enc.rot_group()[k] as u64));
            let want: Vec<f64> = (0..slots).map(|i| v[(i + k) % slots]).collect();
            assert!(max_err(&got, &want) < TWO_M30, "k={k}");
        }
    }

    /// `X -> X^(2N-1)` conjugates every slot.
    #[test]
    fn galois_minus_one_conjugates() {
        for params in [small_params(6), Arc::new(CkksParams::seal_2020())] {
            let enc = Encoder::new(Arc::clone(&params));
            let v = rand_complex(&mut rng(8), params.slots());
            let pt = enc.encode_complex(&v, 2f64.powi(40), 2).unwrap();
            let g = 2 * params.n() as u64 - 1;
            let got = enc.decode_complex(&galois_coeff(&pt, g));
            let want: Vec<Complex64> = v.iter().map(|z| z.conj()).collect();
            assert!(max_err_c(&got, &want) < TWO_M30, "N={}", params.n());
        }
    }

    /// Analytic pin of the embedding: decoding the plaintext `Delta * X` gives
    /// `slot_i = zeta^(3^i mod 2N)` with `zeta = exp(2 pi i / 2N)` (SEAL's root convention).
    #[test]
    fn decode_of_x_is_zeta_to_rot_group() {
        let params = small_params(5);
        let enc = Encoder::new(Arc::clone(&params));
        let n = params.n();
        let delta = 2f64.powi(30);
        let basis = params.basis_q();
        let limbs: Vec<Vec<u64>> = (0..=2)
            .map(|_| {
                let mut l = vec![0u64; n];
                l[1] = 1u64 << 30;
                l
            })
            .collect();
        let mut poly = RnsPoly::from_limbs(basis, limbs, false).unwrap();
        poly.to_ntt();
        let got = enc.decode_complex(&Plaintext::new(poly, delta).unwrap());
        // 3^i mod 2N computed here, independently of rot_group().
        let mut e = 1usize;
        let want: Vec<Complex64> = (0..params.slots())
            .map(|_| {
                let z = Complex64::from_polar(1.0, std::f64::consts::PI * e as f64 / n as f64);
                e = e * 3 % (2 * n);
                z
            })
            .collect();
        assert!(max_err_c(&got, &want) < 1e-12);
    }

    /// Known-answer tests generated with Microsoft SEAL 4.1.2 `CKKSEncoder::encode`
    /// (CoeffModulus::Create(N, {60, 40, 60}), scale 2^30, first_parms_id; centered
    /// coefficients of limb 0 after `inverse_ntt_negacyclic_harvey`). Generator and
    /// build/run instructions: `tools/seal-kat/encoder_kat/` (`kat.cpp`, `README.md`).
    #[test]
    #[allow(clippy::excessive_precision)] // inputs are SEAL's %.17g printout, verbatim
    fn matches_seal_4_1_encoder() {
        type KatCase<'a> = (u32, &'a [i64], &'a [i64], &'a [i64]);
        let real = [
            -0.85000000000000009,
            -0.85500000000000009,
            -0.4850000000000001,
            0.2599999999999999,
            0.25499999999999989,
            0.62499999999999989,
            1.3699999999999999,
            1.3649999999999998,
            1.7349999999999999,
            2.48,
            2.4749999999999996,
            2.8449999999999998,
            3.5899999999999999,
            3.585,
            3.9550000000000001,
            4.7000000000000002,
        ];
        let cplx: Vec<Complex64> = [
            (0.5, -0.40000000000000002),
            (0.29000000000000004, -0.27000000000000002),
            (0.080000000000000016, -0.14000000000000001),
            (-0.12999999999999998, -0.010000000000000009),
            (-0.33999999999999997, 0.12),
            (-0.54999999999999993, 0.25),
            (-0.76000000000000001, 0.38),
            (-0.96999999999999997, 0.51000000000000001),
            (-1.1799999999999999, 0.64000000000000001),
            (-1.3899999999999999, 0.77000000000000002),
            (-1.5999999999999999, 0.90000000000000002),
            (-1.8099999999999998, 1.03),
            (-2.02, 1.1600000000000001),
            (-2.23, 1.29),
            (-2.4399999999999999, 1.4199999999999999),
            (-2.6499999999999999, 1.55),
        ]
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
        let short = [0.75, -0.5, 0.125];

        let seal_n16_real: [i64; 16] = [
            226156872, -344450380, -278799679, -77424207, -104871424, 290508693, 154004784,
            -286079917, 0, 286079917, -154004784, -290508693, 104871424, 77424207, 278799679,
            344450380,
        ];
        let seal_n16_cplx: [i64; 16] = [
            -252329329, 182148167, 56116452, -11773007, 121480020, -298561333, -98787831,
            205647104, -69793219, -121707514, 23244196, -59186903, -37962506, -59387542,
            -238494921, -307772642,
        ];
        let seal_n16_short: [i64; 16] = [
            50331648, 39657025, 51819182, 106111683, 130496115, 107795172, 94102319, 73376969, 0,
            -73376969, -94102319, -107795172, -130496115, -106111683, -51819182, -39657025,
        ];
        let seal_n32_real: [i64; 32] = [
            1815294771, -366965883, -405762676, 113740191, -245918945, -651538087, -60837129,
            -407048896, -140461273, 522620323, 207119823, 101595331, 140385138, 366908900,
            -327047483, -349302239, 0, 349302239, 327047483, -366908900, -140385138, -101595331,
            -207119823, -522620323, 140461273, 407048896, 60837129, 651538087, 245918945,
            -113740191, 405762676, 366965883,
        ];
        let seal_n32_cplx: [i64; 32] = [
            -1154272461,
            -42596489,
            182148167,
            -253314633,
            56116452,
            343023669,
            -11773007,
            258655548,
            516290085,
            -428571270,
            -298561333,
            25785504,
            -98787831,
            -326776372,
            205647104,
            209963553,
            -69793219,
            -199054864,
            -121707514,
            38039102,
            23244196,
            172905378,
            -59186903,
            92075228,
            356847559,
            -237070061,
            -59387542,
            -463716161,
            -238494921,
            -164221017,
            -307772642,
            -402302888,
        ];
        let seal_n32_short: [i64; 32] = [
            25165824, 23301375, 19828513, 19479586, 25909591, 38664557, 53055842, 63072554,
            65248058, 60700199, 53897586, 49091636, 47051159, 44502795, 36688484, 21158183, 0,
            -21158183, -36688484, -44502795, -47051159, -49091636, -53897586, -60700199, -65248058,
            -63072554, -53055842, -38664557, -25909591, -19479586, -19828513, -23301375,
        ];

        let cases: [KatCase; 2] = [
            (4, &seal_n16_real, &seal_n16_cplx, &seal_n16_short),
            (5, &seal_n32_real, &seal_n32_cplx, &seal_n32_short),
        ];
        for (log_n, want_real, want_cplx, want_short) in cases {
            let params = Arc::new(CkksParams::new(log_n, &[60, 40], &[60], 30, 1).unwrap());
            let enc = Encoder::new(Arc::clone(&params));
            let slots = params.slots();
            let s = 2f64.powi(30);
            let pt = enc.encode(&real[..slots], s, 1).unwrap();
            assert_eq!(centered_coeffs(&pt), want_real, "real, N=2^{log_n}");
            let pt = enc.encode_complex(&cplx[..slots], s, 1).unwrap();
            assert_eq!(centered_coeffs(&pt), want_cplx, "complex, N=2^{log_n}");
            let pt = enc.encode(&short, s, 1).unwrap();
            assert_eq!(centered_coeffs(&pt), want_short, "short, N=2^{log_n}");
        }
    }

    #[test]
    fn smallest_ring_n2() {
        let params = Arc::new(CkksParams::new(1, &[30, 30], &[30], 20, 1).unwrap());
        let enc = Encoder::new(Arc::clone(&params));
        let v = [Complex64::new(0.25, -0.75)];
        let pt = enc.encode_complex(&v, 2f64.powi(20), 1).unwrap();
        assert!(max_err_c(&enc.decode_complex(&pt), &v) < 1e-5);
    }

    #[test]
    fn plaintext_new_validates() {
        let params = small_params(4);
        let mut p = RnsPoly::zero(params.basis_q(), 1);
        assert!(Plaintext::new(p.clone(), 1.0).is_err(), "coefficient form");
        p.to_ntt();
        assert!(Plaintext::new(p.clone(), 0.0).is_err());
        assert!(Plaintext::new(p.clone(), f64::NAN).is_err());
        let pt = Plaintext::new(p, 2.0).unwrap();
        assert_eq!((pt.level(), pt.scale()), (1, 2.0));
        assert_eq!(pt.into_poly().level(), 1);
    }
}
