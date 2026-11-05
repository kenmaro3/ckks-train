//! Host-side layer of GPU hybrid key switching (no CUDA needed): the device layout of
//! key-switching keys ([`KskShape`], [`ksk_limbs`]), the inner-product launch
//! ([`InnerProduct`], `kernels/keyswitch.cu`), and the launch sequences of key switching and
//! of the ciphertext operations built on it (relinearization, rotations, hoisted rotations,
//! rescaling), written once against `KsBackend`.
//!
//! As in [`crate::basis`], the CUDA device (`src/keyswitch.rs`, `src/ciphertext.rs`) and the
//! CPU replay of the kernels (tests) both implement `KsBackend`, so the tests replay the
//! exact launch sequence the device runs.
//!
//! The sequences are those of `hekit_ckks` (`keyswitch::{decompose, apply_key}` and
//! `Evaluator::{mul_no_relin, relinearize, rotate, conjugate, hoisted_rotate, rescale}`),
//! operation by operation. Every step's output is the unique reduced residue the CPU step
//! computes, so the results are bit-identical.

use hekit_ckks::keyswitch::digit_ranges;
use hekit_ckks::{CkksParams, KskKey};
use hekit_core::{HekitError, Result};

use crate::basis::{Backend, ModUpPlan, PrimeTable, divide_round_tail, mod_up_digit};

fn invalid(msg: impl Into<String>) -> HekitError {
    HekitError::InvalidParams(msg.into())
}

/// Which limbs of a key-switching key are on the device: the parts whose digit starts at or
/// below `max_level` and, in each, the `Q` limbs `0..=max_level` and every `P` limb.
///
/// A full key has `max_level = L` (all `dnum` parts, `L + 1 + k` limbs per polynomial); a key
/// uploaded for levels `<= max_level` only is smaller (key switching at level `l` reads only
/// the parts of the digits of `l` and the limbs `0..=l` and `P`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KskShape {
    /// Number of parts `(b_j, a_j)` stored: `digit_ranges(params, max_level).len()`.
    pub parts: usize,
    /// The highest level the key serves.
    pub max_level: usize,
    /// Number of `P` primes `k`.
    pub np: usize,
}

impl KskShape {
    /// The shape of a key for levels `0..=max_level`. `InvalidParams` when `max_level` exceeds
    /// `params.max_level()`.
    pub fn new(params: &CkksParams, max_level: usize) -> Result<KskShape> {
        if max_level > params.max_level() {
            return Err(invalid(format!(
                "key level {max_level} above max_level {}",
                params.max_level()
            )));
        }
        Ok(KskShape {
            parts: digit_ranges(params, max_level).len(),
            max_level,
            np: params.p().len(),
        })
    }

    /// Limbs of one key polynomial (`b_j` or `a_j`): `max_level + 1 + k`.
    pub fn limbs_per_poly(&self) -> usize {
        self.max_level + 1 + self.np
    }

    /// Limbs of the whole key: `parts * 2 * limbs_per_poly`.
    pub fn total_limbs(&self) -> usize {
        self.parts * 2 * self.limbs_per_poly()
    }

    /// Device bytes of the key at ring degree `n` (n16_train, full key: 288 MiB).
    pub fn device_bytes(&self, n: usize) -> usize {
        8 * n * self.total_limbs()
    }
}

/// The limbs of `key` in device order (module docs of `kernels/keyswitch.cu`): for each part
/// `j < shape.parts`, `b_j` then `a_j`, each as its `Q` limbs `0..=shape.max_level` then its
/// `P` limbs.
///
/// `InvalidParams` unless the key has `params.dnum()` parts, each polynomial in NTT form with
/// its `Q` part over all of this table's `Q` primes and its `P` part over all `P` primes
/// (primes compared by value, as for uploads).
pub fn ksk_limbs<'k>(
    table: &PrimeTable,
    params: &CkksParams,
    key: &'k KskKey,
    shape: &KskShape,
) -> Result<Vec<&'k [u64]>> {
    if key.dnum() != params.dnum() || shape.parts > key.dnum() {
        return Err(invalid(format!(
            "key has {} parts, the parameters {}",
            key.dnum(),
            params.dnum()
        )));
    }
    let full_q = table.nq() - 1;
    let mut out = Vec::with_capacity(shape.total_limbs());
    for (b, a) in &key.parts()[..shape.parts] {
        for x in [b, a] {
            let (q, p) = (table.primes_of(x.q())?, table.primes_of(x.p())?);
            if table.q_prefix_level(&q) != Some(full_q)
                || table.p_prefix_len(&p) != Some(table.np())
                || !x.q().is_ntt_form()
                || !x.p().is_ntt_form()
            {
                return Err(invalid(
                    "key polynomials must be NTT-form pairs over all Q and all P primes",
                ));
            }
            out.extend(x.q().limbs()[..=shape.max_level].iter().map(Vec::as_slice));
            out.extend(x.p().limbs().iter().map(Vec::as_slice));
        }
    }
    debug_assert_eq!(out.len(), shape.total_limbs());
    Ok(out)
}

/// One launch of `ks_inner_product` (`kernels/keyswitch.cu`): the digits of a level-`level`
/// polynomial against a key of shape `key`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InnerProduct {
    /// The digits' level `l`.
    pub level: usize,
    /// Number of digits (`digit_ranges(params, l).len()`).
    pub n_digits: usize,
    /// `l + 1`: output limbs below this are `Q` limbs.
    pub q_limbs: usize,
    /// Limbs per key polynomial ([`KskShape::limbs_per_poly`]).
    pub key_limbs: usize,
    /// `key.max_level - l`: added to a `P` limb's index to find its key limb.
    pub key_skip: usize,
    /// Limbs of the digits and of both outputs: [`PrimeTable::qp_primes`]`(l)`.
    pub out_primes: Vec<u32>,
}

impl InnerProduct {
    /// The launch for `n_digits` digits at `level` against `key`. `InvalidParams` unless
    /// `n_digits == digit_ranges(params, level).len()` and `level <= key.max_level` (so the
    /// key has every part the digits need).
    pub fn new(
        params: &CkksParams,
        table: &PrimeTable,
        level: usize,
        n_digits: usize,
        key: &KskShape,
    ) -> Result<InnerProduct> {
        if level > key.max_level {
            return Err(invalid(format!(
                "key switching at level {level} with a key uploaded for levels <= {}",
                key.max_level
            )));
        }
        let expected = digit_ranges(params, level).len();
        if n_digits != expected {
            return Err(invalid(format!(
                "{n_digits} digits for a level-{level} polynomial with {expected} digits"
            )));
        }
        debug_assert!(n_digits <= key.parts);
        Ok(InnerProduct {
            level,
            n_digits,
            q_limbs: level + 1,
            key_limbs: key.limbs_per_poly(),
            key_skip: key.max_level - level,
            out_primes: table.qp_primes(level),
        })
    }
}

/// The element-wise kernels with two inputs (`pw_add`, `pw_sub`, `pw_mul`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))] // `Sub` is the device's `sub` only
pub(crate) enum PwOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b` (pointwise).
    Mul,
}

/// The kernels key switching adds to [`Backend`]: implemented by the CUDA device and by the
/// CPU replay (tests). Buffers hold whole limbs; `primes` gives each limb's prime index.
#[cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
pub(crate) trait KsBackend: Backend {
    /// `pw_add` / `pw_sub` / `pw_mul` into `out` (same limbs as `a` and `b`).
    fn binary(
        &self,
        op: PwOp,
        out: &mut Self::Buf,
        a: &Self::Buf,
        b: &Self::Buf,
        primes: &[u32],
    ) -> Result<()>;
    /// `automorphism_ntt` (NTT form) or `automorphism_coeff` into `out`.
    fn automorphism(
        &self,
        out: &mut Self::Buf,
        a: &Self::Buf,
        primes: &[u32],
        ntt_form: bool,
        galois_elt: u64,
    ) -> Result<()>;
    /// `ks_inner_product`: `out_b`, `out_a` (limbs `ip.out_primes`) from `digits` (each with
    /// limbs `ip.out_primes`) and the key buffer ([`ksk_limbs`] order).
    fn inner_product(
        &self,
        out_b: &mut Self::Buf,
        out_a: &mut Self::Buf,
        digits: &[&Self::Buf],
        key: &Self::Buf,
        ip: &InnerProduct,
    ) -> Result<()>;
}

/// A key on a backend: its buffer and shape.
#[derive(Debug)]
pub(crate) struct KeyRef<'a, Buf> {
    /// The limbs in [`ksk_limbs`] order.
    pub buf: &'a Buf,
    /// The shape.
    pub shape: &'a KskShape,
}

impl<Buf> Clone for KeyRef<'_, Buf> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Buf> Copy for KeyRef<'_, Buf> {}

/// The launch sequences of key switching and the ciphertext operations on a backend. A
/// ciphertext component is a buffer over the `Q` primes `0..=level`, NTT form.
#[cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
pub(crate) struct Ks<'a, B> {
    /// The backend.
    pub b: &'a B,
    /// The parameter set.
    pub params: &'a CkksParams,
    /// Its prime table.
    pub table: &'a PrimeTable,
}

/// `[0, ..., level]`: the limbs of a level-`level` ciphertext component.
pub fn q_primes(level: usize) -> Vec<u32> {
    (0..=level as u32).collect()
}

#[cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
impl<B: KsBackend> Ks<'_, B> {
    /// `a op c` (limbs `primes`) as a new buffer.
    pub fn pw(&self, op: PwOp, a: &B::Buf, c: &B::Buf, primes: &[u32]) -> Result<B::Buf> {
        let mut out = self.b.alloc(primes.len())?;
        self.b.binary(op, &mut out, a, c, primes)?;
        Ok(out)
    }

    /// `sigma_g(a)` (limbs `primes`, NTT form) as a new buffer.
    pub fn auto(&self, a: &B::Buf, primes: &[u32], galois_elt: u64) -> Result<B::Buf> {
        let mut out = self.b.alloc(primes.len())?;
        self.b.automorphism(&mut out, a, primes, true, galois_elt)?;
        Ok(out)
    }

    /// `hekit_ckks::keyswitch::decompose`: the exact centered lift of every digit of `x`
    /// (level `level`, NTT form) to `Q_l * P` (limbs `qp_primes(level)`, NTT form), one
    /// `mod_up` launch sequence per digit.
    pub fn decompose(&self, x: &B::Buf, level: usize) -> Result<Vec<B::Buf>> {
        let primes = q_primes(level);
        (0..digit_ranges(self.params, level).len())
            .map(|j| {
                let plan = ModUpPlan::new(self.params, self.table, &primes, j)?;
                mod_up_digit(self.b, x, &plan)
            })
            .collect()
    }

    /// `hekit_ckks::keyswitch::apply_key`: one `ks_inner_product` launch over every limb of
    /// `Q_l * P` and both accumulators, then `mod_down` (division by `P` with rounding) of
    /// each: `(kb, ka)` at level `level`, NTT form.
    pub fn apply_key(
        &self,
        digits: &[&B::Buf],
        level: usize,
        key: KeyRef<'_, B::Buf>,
    ) -> Result<(B::Buf, B::Buf)> {
        let ip = InnerProduct::new(self.params, self.table, level, digits.len(), key.shape)?;
        let (mut acc_b, mut acc_a) = (
            self.b.alloc(ip.out_primes.len())?,
            self.b.alloc(ip.out_primes.len())?,
        );
        self.b
            .inner_product(&mut acc_b, &mut acc_a, digits, key.buf, &ip)?;
        let kb = divide_round_tail(self.b, &acc_b, &ip.out_primes, level + 1, true)?;
        drop(acc_b);
        let ka = divide_round_tail(self.b, &acc_a, &ip.out_primes, level + 1, true)?;
        Ok((kb, ka))
    }

    /// `hekit_ckks::keyswitch::key_switch`: `apply_key(decompose(x))`.
    pub fn key_switch(
        &self,
        x: &B::Buf,
        level: usize,
        key: KeyRef<'_, B::Buf>,
    ) -> Result<(B::Buf, B::Buf)> {
        let digits = self.decompose(x, level)?;
        let refs: Vec<&B::Buf> = digits.iter().collect();
        self.apply_key(&refs, level, key)
    }

    /// `Evaluator::mul_no_relin`: `(x0 y0, x0 y1 + x1 y0, x1 y1)`.
    pub fn tensor(&self, x: [&B::Buf; 2], y: [&B::Buf; 2], level: usize) -> Result<[B::Buf; 3]> {
        let p = q_primes(level);
        let d0 = self.pw(PwOp::Mul, x[0], y[0], &p)?;
        let t01 = self.pw(PwOp::Mul, x[0], y[1], &p)?;
        let t10 = self.pw(PwOp::Mul, x[1], y[0], &p)?;
        let d1 = self.pw(PwOp::Add, &t01, &t10, &p)?;
        drop((t01, t10));
        let d2 = self.pw(PwOp::Mul, x[1], y[1], &p)?;
        Ok([d0, d1, d2])
    }

    /// `Evaluator::relinearize`: `(c0 + kb, c1 + ka)`, `(kb, ka)` the key switch of `c2`.
    pub fn relinearize(
        &self,
        c: [&B::Buf; 3],
        level: usize,
        key: KeyRef<'_, B::Buf>,
    ) -> Result<[B::Buf; 2]> {
        let p = q_primes(level);
        let (kb, ka) = self.key_switch(c[2], level, key)?;
        Ok([
            self.pw(PwOp::Add, c[0], &kb, &p)?,
            self.pw(PwOp::Add, c[1], &ka, &p)?,
        ])
    }

    /// `(sigma_g(c0) + kb, ka)` (`finish_galois` of the CPU evaluator).
    fn finish_galois(
        &self,
        c0: &B::Buf,
        level: usize,
        galois_elt: u64,
        kb: &B::Buf,
        ka: B::Buf,
    ) -> Result<[B::Buf; 2]> {
        let p = q_primes(level);
        let r0 = self.auto(c0, &p, galois_elt)?;
        Ok([self.pw(PwOp::Add, &r0, kb, &p)?, ka])
    }

    /// `Evaluator::rotate` / `conjugate` (no hoisting): `(sigma_g(c0) + kb, ka)`, `(kb, ka)`
    /// the key switch of `sigma_g(c1)` by the Galois key of `g`.
    pub fn galois(
        &self,
        c: [&B::Buf; 2],
        level: usize,
        galois_elt: u64,
        key: KeyRef<'_, B::Buf>,
    ) -> Result<[B::Buf; 2]> {
        let d = self.auto(c[1], &q_primes(level), galois_elt)?;
        let (kb, ka) = self.key_switch(&d, level, key)?;
        drop(d);
        self.finish_galois(c[0], level, galois_elt, &kb, ka)
    }

    /// `Evaluator::hoisted_rotate` for the nontrivial rotations `rots` (Galois element and
    /// key): `c1` is decomposed once; per rotation, `sigma_g` is applied to every limb of every
    /// lifted digit (one launch per digit over its `Q_l * P` limbs), then `apply_key` and
    /// `finish_galois`. Results in the order of `rots`.
    pub fn hoisted(
        &self,
        c: [&B::Buf; 2],
        level: usize,
        rots: &[(u64, KeyRef<'_, B::Buf>)],
    ) -> Result<Vec<[B::Buf; 2]>> {
        if rots.is_empty() {
            return Ok(Vec::new());
        }
        let digits = self.decompose(c[1], level)?;
        let qp = self.table.qp_primes(level);
        rots.iter()
            .map(|&(g, key)| {
                let rotated = digits
                    .iter()
                    .map(|d| self.auto(d, &qp, g))
                    .collect::<Result<Vec<_>>>()?;
                let refs: Vec<&B::Buf> = rotated.iter().collect();
                let (kb, ka) = self.apply_key(&refs, level, key)?;
                drop(rotated);
                self.finish_galois(c[0], level, g, &kb, ka)
            })
            .collect()
    }

    /// `Evaluator::rescale`: `rescale_by_last` of every component (level `level >= 1`).
    pub fn rescale(&self, c: &[&B::Buf], level: usize) -> Result<Vec<B::Buf>> {
        let p = q_primes(level);
        c.iter()
            .map(|x| divide_round_tail(self.b, x, &p, level, true))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hekit_ckks::KeyGenerator;
    use hekit_core::sampler::Rng;

    use super::*;

    /// Device order of a key's limbs (full and level-capped) and rejection of a key of
    /// another parameter set (the GPU error test uses the same foreign parameters).
    #[test]
    fn ksk_limbs_layout_and_rejections() {
        let p = Arc::new(CkksParams::test_tiny()); // L = 11, alpha 1, dnum 12, k 1
        let t = PrimeTable::from_params(&p);
        let mut kg = KeyGenerator::new(Arc::clone(&p), Rng::from_seed([1; 32])).unwrap();
        let rk = kg.relin_key();
        let parts = rk.key().parts();
        let full = KskShape::new(&p, 11).unwrap();
        let limbs = ksk_limbs(&t, &p, rk.key(), &full).unwrap();
        assert_eq!(limbs.len(), full.total_limbs());
        let lp = full.limbs_per_poly(); // 13
        assert_eq!(limbs[0], parts[0].0.q().limbs()[0].as_slice());
        assert_eq!(limbs[12], parts[0].0.p().limbs()[0].as_slice());
        assert_eq!(limbs[lp], parts[0].1.q().limbs()[0].as_slice());
        assert_eq!(limbs[2 * lp + 5], parts[1].0.q().limbs()[5].as_slice());
        let low = KskShape::new(&p, 2).unwrap(); // 3 parts, limbs q0..q2 + p0
        let l2 = ksk_limbs(&t, &p, rk.key(), &low).unwrap();
        assert_eq!(l2.len(), 3 * 2 * 4);
        assert_eq!(l2[3], parts[0].0.p().limbs()[0].as_slice());
        assert_eq!(l2[4], parts[0].1.q().limbs()[0].as_slice());
        assert_eq!(l2[2 * 4 + 2], parts[1].0.q().limbs()[2].as_slice());
        let foreign = Arc::new(CkksParams::new(5, &[54, 42, 34], &[60], 42, 3).unwrap());
        let mut kg = KeyGenerator::new(foreign, Rng::from_seed([2; 32])).unwrap();
        assert!(matches!(
            ksk_limbs(&t, &p, kg.relin_key().key(), &full),
            Err(HekitError::InvalidParams(_))
        ));
    }

    #[test]
    fn ksk_shapes() {
        let p = CkksParams::n16_train(); // L = 31, alpha 4, dnum 8, k 4
        let full = KskShape::new(&p, p.max_level()).unwrap();
        assert_eq!((full.parts, full.limbs_per_poly()), (8, 36));
        assert_eq!(full.device_bytes(p.n()), (8 * 2 * 36 * 8) << 16); // 288 MiB
        assert_eq!(full.device_bytes(p.n()), 301_989_888);
        let low = KskShape::new(&p, 3).unwrap();
        assert_eq!((low.parts, low.limbs_per_poly()), (1, 8));
        assert_eq!(KskShape::new(&p, 4).unwrap().parts, 2);
        assert!(KskShape::new(&p, 32).is_err());
        let s = CkksParams::seal_2020(); // L = 9, alpha 1, dnum 10, k 1
        let full = KskShape::new(&s, 9).unwrap();
        assert_eq!((full.parts, full.limbs_per_poly()), (10, 11));
        assert_eq!(full.device_bytes(s.n()), (10 * 2 * 11 * 8) << 15); // 55 MiB
    }

    #[test]
    fn inner_product_launches() {
        let p = CkksParams::n16_train();
        let t = PrimeTable::from_params(&p);
        let full = KskShape::new(&p, 31).unwrap();
        let ip = InnerProduct::new(&p, &t, 30, 8, &full).unwrap();
        assert_eq!((ip.q_limbs, ip.key_limbs, ip.key_skip), (31, 36, 1));
        assert_eq!(ip.out_primes, t.qp_primes(30));
        // P limb y = 31 (prime 32) reads key limb 32 = prime 32 of the stored key.
        assert_eq!(ip.out_primes[31] as usize, 31 + ip.key_skip);
        assert!(InnerProduct::new(&p, &t, 30, 7, &full).is_err());
        let low = KskShape::new(&p, 3).unwrap();
        let ip = InnerProduct::new(&p, &t, 1, 1, &low).unwrap();
        assert_eq!((ip.q_limbs, ip.key_limbs, ip.key_skip), (2, 8, 2));
        assert!(InnerProduct::new(&p, &t, 4, 2, &low).is_err()); // above the key's level
    }
}
