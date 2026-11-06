//! CKKS ciphertexts in device memory ([`CudaCiphertext`]) and the ciphertext operations that
//! need key switching or rescaling: [`CudaDevice::mul_no_relin`],
//! [`CudaDevice::relinearize`], [`CudaDevice::rescale`], [`CudaDevice::rotate`],
//! [`CudaDevice::conjugate`] and [`CudaDevice::hoisted_rotate`].
//!
//! Each has the semantics, the checks and the error kinds of the `ckks_train_ckks::Evaluator`
//! method of the same name, and the result is bit-identical: the same residues (the shared
//! launch sequences of [`crate::ks`]), the same level and the same `f64` scale (computed by
//! the same formula). Operations are **functional**: they read their operands and return new
//! ciphertexts, so ciphertexts can share device storage ([`CudaCiphertext`]'s `Clone`).
//!
//! These operations do not count anything: operation counting (`ckks_train_ckks::OpCounter`,
//! rotations / relinearizations / conjugations as the CPU counts them) is the device
//! evaluator's (Task 47), which calls them.

use std::sync::Arc;

use ckks_train_ckks::Ciphertext;
use ckks_train_ckks::keys::conjugation_elt;
use ckks_train_ckks::linalg::DiagonalEntry;
use ckks_train_ckks::ops::HeCiphertext;
use ckks_train_core::{HekitError, Result};
use cudarc::driver::CudaSlice;

use crate::arith::Dev;
use crate::device::CudaDevice;
use crate::keyswitch::{DeviceKsk, KskPurpose};
pub use crate::ks::rotation_elt;
use crate::ks::{KeyRef, merge_hoisted, q_primes, rescaled_scale};
use crate::poly::DevicePoly;

fn invalid(msg: impl Into<String>) -> HekitError {
    HekitError::InvalidParams(msg.into())
}

/// A CKKS ciphertext `(c_0, c_1, ...)` in device memory: NTT-form polynomials over the `Q`
/// primes `0..=level` of one [`CudaDevice`], with a scale and a level (as
/// [`ckks_train_ckks::Ciphertext`]; 2 components, or 3 before relinearization).
///
/// **`Clone` is a shared handle**: the components are [`DevicePoly`] handles to `Arc`'d device
/// storage, so `clone` copies a few pointers and increments reference counts. It never
/// allocates device memory and never panics, as `ckks_train_ckks::HeOps` requires of its
/// ciphertexts. Sharing is safe because every ciphertext operation is functional (it returns
/// a new ciphertext and never writes its operands); the only in-place device operations
/// ([`CudaDevice::ntt`] / [`CudaDevice::intt`] on a component) copy on write.
///
/// Borrows its device (`'d`) like [`DevicePoly`], so it cannot outlive it (the error code of
/// a `compile_fail` example is not checked by stable rustdoc; the `no_run` twin differs only
/// in where the device is dropped):
///
/// ```compile_fail,E0505
/// fn f(dev: ckks_train_cuda::CudaDevice, ct: &ckks_train_ckks::Ciphertext) {
///     let d = dev.upload_ct(ct).unwrap();
///     drop(dev); // error[E0505]: cannot move out of `dev` because it is borrowed
///     let _ = d.download();
/// }
/// ```
///
/// ```no_run
/// fn f(dev: ckks_train_cuda::CudaDevice, ct: &ckks_train_ckks::Ciphertext) {
///     let d = dev.upload_ct(ct).unwrap();
///     let _ = d.download();
///     drop(dev);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CudaCiphertext<'d> {
    c: Vec<DevicePoly<'d>>,
    scale: f64,
    level: usize,
}

// The `HeOps` contract: device ciphertexts and plaintexts are `Clone + Send` (and the device
// is shared by reference across threads).
const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<CudaCiphertext<'static>>();
    send_sync::<crate::DevicePlaintext<'static>>();
    send_sync::<DeviceKsk<'static>>();
    send_sync::<CudaDevice>();
};

impl<'d> CudaCiphertext<'d> {
    /// Wraps device polynomials with their scale; the level is the common level.
    ///
    /// Errors (as `Ciphertext::new`): `InvalidParams` for fewer than 2 components, a component
    /// in coefficient form or not over the `Q` primes `0..=l`, components of different
    /// devices, or a scale that is not positive and finite; `LevelMismatch` for components at
    /// different levels.
    pub fn new(c: Vec<DevicePoly<'d>>, scale: f64) -> Result<CudaCiphertext<'d>> {
        if c.len() < 2 {
            return Err(invalid("a ciphertext needs at least 2 components"));
        }
        let dev = c[0].device();
        let t = dev.prime_table();
        let mut levels = Vec::with_capacity(c.len());
        for p in &c {
            if !p.is_ntt_form() {
                return Err(invalid("ciphertext components must be in NTT form"));
            }
            if !std::ptr::eq(p.device(), dev) {
                return Err(invalid("ciphertext components on different devices"));
            }
            levels.push(
                t.q_prefix_level(p.primes()).ok_or_else(|| {
                    invalid("ciphertext components must be over the Q primes 0..=l")
                })?,
            );
        }
        let level = levels[0];
        if levels.iter().any(|&l| l != level) {
            return Err(HekitError::LevelMismatch);
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(invalid("ciphertext scale must be positive and finite"));
        }
        Ok(CudaCiphertext { c, scale, level })
    }

    /// The components (NTT form).
    pub fn polys(&self) -> &[DevicePoly<'d>] {
        &self.c
    }

    /// Number of components (2, or 3 before relinearization).
    pub fn len(&self) -> usize {
        self.c.len()
    }

    /// Always `false`: a ciphertext has at least 2 components.
    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    /// The scale.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The level (every component has `level + 1` limbs).
    pub fn level(&self) -> usize {
        self.level
    }

    /// The device holding the ciphertext.
    pub fn device(&self) -> &'d CudaDevice {
        self.c[0].device()
    }

    /// Device bytes held by the components (shared storage counted once per handle).
    pub fn device_bytes(&self) -> usize {
        self.c.iter().map(DevicePoly::device_bytes).sum()
    }

    /// The same ciphertext labelled with `scale` (no device work; storage shared).
    /// `InvalidParams` unless `scale` is positive and finite.
    pub fn with_scale(&self, scale: f64) -> Result<CudaCiphertext<'d>> {
        CudaCiphertext::new(self.c.clone(), scale)
    }

    /// Copies the ciphertext back to the host (waits for pending work on it): components over
    /// the device's `params.basis_q()`.
    pub fn download(&self) -> Result<Ciphertext> {
        let polys = self
            .c
            .iter()
            .map(DevicePoly::download)
            .collect::<Result<Vec<_>>>()?;
        Ciphertext::new(polys, self.scale)
    }

    fn bufs(&self) -> Vec<&CudaSlice<u64>> {
        self.c.iter().map(DevicePoly::limbs).collect()
    }
}

impl DiagonalEntry for CudaCiphertext<'_> {
    fn level(&self) -> usize {
        self.level
    }
    fn scale(&self) -> f64 {
        self.scale
    }
    fn slots(&self) -> usize {
        self.c[0].n() / 2
    }
}

impl HeCiphertext for CudaCiphertext<'_> {
    fn size(&self) -> usize {
        self.c.len()
    }
}

impl CudaDevice {
    /// Copies a host ciphertext (components over this device's `Q` primes, compared by value)
    /// to the device. Synchronous with respect to `ct`.
    pub fn upload_ct<'d>(&'d self, ct: &Ciphertext) -> Result<CudaCiphertext<'d>> {
        let c = ct
            .polys()
            .iter()
            .map(|p| self.upload(p))
            .collect::<Result<Vec<_>>>()?;
        CudaCiphertext::new(c, ct.scale())
    }

    fn check_ct(&self, ct: &CudaCiphertext<'_>) -> Result<()> {
        if std::ptr::eq(ct.device(), self) {
            Ok(())
        } else {
            Err(invalid("CudaCiphertext belongs to another CudaDevice"))
        }
    }

    /// A ciphertext from device buffers over the `Q` primes `0..=level` (NTT form).
    fn ct_from(
        &self,
        bufs: impl IntoIterator<Item = CudaSlice<u64>>,
        level: usize,
        scale: f64,
    ) -> Result<CudaCiphertext<'_>> {
        let primes: Arc<[u32]> = q_primes(level).into();
        let c = bufs
            .into_iter()
            .map(|b| self.new_poly(b, Arc::clone(&primes), true))
            .collect();
        CudaCiphertext::new(c, scale)
    }

    /// `Evaluator::mul_no_relin`: the tensor product `(a0 b0, a0 b1 + a1 b0, a1 b1)` of two
    /// 2-component ciphertexts at the same level `>= 1`; scale `s_a * s_b`.
    ///
    /// Errors: `InvalidParams` for a ciphertext of another device or not of 2 components;
    /// `LevelMismatch`; `NoLevelsLeft` at level 0; `InvalidParams` when the scale product is
    /// not finite.
    pub fn mul_no_relin<'d>(
        &'d self,
        a: &CudaCiphertext<'d>,
        b: &CudaCiphertext<'d>,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_ct(a)?;
        self.check_ct(b)?;
        if a.len() != 2 || b.len() != 2 {
            return Err(invalid(format!(
                "multiplication needs 2-component ciphertexts (got {} and {}); relinearize first",
                a.len(),
                b.len()
            )));
        }
        if a.level() != b.level() {
            return Err(HekitError::LevelMismatch);
        }
        if a.level() == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        let scale = a.scale() * b.scale();
        if !(scale.is_finite() && scale > 0.0) {
            return Err(invalid("ciphertext scale must be positive and finite"));
        }
        let (x, y) = (a.bufs(), b.bufs());
        let d = self
            .ks(&Dev(self))
            .tensor([x[0], x[1]], [y[0], y[1]], a.level())?;
        self.ct_from(d, a.level(), scale)
    }

    /// `Evaluator::relinearize`: `(c0 + kb, c1 + ka)` for a 3-component ciphertext, `(kb, ka)`
    /// the key switch of `c2` with the relinearization key `rk`.
    ///
    /// Errors (`InvalidParams`): another device's ciphertext or key, not 3 components, a key
    /// that is not a relinearization key ([`KskPurpose::Relin`]), a level above the key's.
    pub fn relinearize<'d>(
        &'d self,
        ct: &CudaCiphertext<'d>,
        rk: &DeviceKsk<'d>,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_ct(ct)?;
        if ct.len() != 3 {
            return Err(invalid(format!(
                "relinearize needs a 3-component ciphertext (got {})",
                ct.len()
            )));
        }
        self.check_key_owner(rk)?;
        if rk.purpose() != KskPurpose::Relin {
            return Err(invalid("relinearize needs a relinearization key"));
        }
        check_key_level(rk, ct.level())?;
        let c = ct.bufs();
        let r = self
            .ks(&Dev(self))
            .relinearize([c[0], c[1], c[2]], ct.level(), rk.key_ref())?;
        self.ct_from(r, ct.level(), ct.scale())
    }

    /// `Evaluator::rescale`: divides every component by the last prime `q_l` with rounding:
    /// level `l - 1`, scale `/ q_l` (as `f64`). `NoLevelsLeft` at level 0.
    pub fn rescale<'d>(&'d self, ct: &CudaCiphertext<'d>) -> Result<CudaCiphertext<'d>> {
        self.check_ct(ct)?;
        let l = ct.level();
        if l == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        let c = self.ks(&Dev(self)).rescale(&ct.bufs(), l)?;
        self.ct_from(c, l - 1, rescaled_scale(self.params(), ct.scale(), l))
    }

    fn check_galois_input(&self, ct: &CudaCiphertext<'_>) -> Result<()> {
        self.check_ct(ct)?;
        if ct.len() != 2 {
            return Err(invalid(format!(
                "rotation needs a 2-component ciphertext (got {}); relinearize first",
                ct.len()
            )));
        }
        Ok(())
    }

    /// The first key of `keys` for Galois element `elt` that belongs to this device and serves
    /// `level`. `InvalidParams("missing galois key")` when `keys` has no key for `elt`;
    /// otherwise, when none of them is usable, the error of the first one (another device's
    /// key, or a key uploaded for levels below `level`).
    fn galois_key<'k, 'd>(
        &self,
        keys: &[&'k DeviceKsk<'d>],
        elt: u64,
        level: usize,
    ) -> Result<&'k DeviceKsk<'d>> {
        let mut first_err = None;
        for &k in keys.iter().filter(|k| k.galois_elt() == Some(elt)) {
            match self
                .check_key_owner(k)
                .and_then(|()| check_key_level(k, level))
            {
                Ok(()) => return Ok(k),
                Err(e) => {
                    first_err.get_or_insert(e);
                }
            }
        }
        Err(first_err.unwrap_or_else(|| invalid("missing galois key")))
    }

    /// The Galois element of a rotation by `steps`, `None` for a multiple of the slot count.
    fn rotation_elt(&self, steps: i32) -> Option<u64> {
        rotation_elt(self.params(), steps)
    }

    fn apply_galois<'d>(
        &'d self,
        ct: &CudaCiphertext<'d>,
        elt: u64,
        key: &DeviceKsk<'d>,
    ) -> Result<CudaCiphertext<'d>> {
        let c = ct.bufs();
        let r = self
            .ks(&Dev(self))
            .galois([c[0], c[1]], ct.level(), elt, key.key_ref())?;
        self.ct_from(r, ct.level(), ct.scale())
    }

    /// `Evaluator::rotate`: rotates the slots left by `steps` (negative: right), with the
    /// Galois key of `g = galois_elt(N, steps mod slots)` found in `keys` (by element).
    /// `(c0, c1) -> (sigma_g(c0) + kb, ka)`, `(kb, ka)` the key switch of `sigma_g(c1)`. A
    /// multiple of the slot count returns a copy (shared storage, no key needed).
    ///
    /// Errors (`InvalidParams`): another device's ciphertext or key, not 2 components, no key
    /// for `g` (`"missing galois key"`), a level above the key's.
    pub fn rotate<'d>(
        &'d self,
        ct: &CudaCiphertext<'d>,
        steps: i32,
        keys: &[&DeviceKsk<'d>],
    ) -> Result<CudaCiphertext<'d>> {
        self.check_galois_input(ct)?;
        let Some(elt) = self.rotation_elt(steps) else {
            return Ok(ct.clone());
        };
        let key = self.galois_key(keys, elt, ct.level())?;
        self.apply_galois(ct, elt, key)
    }

    /// `Evaluator::conjugate`: complex conjugation of every slot (Galois element `2N - 1`),
    /// with the conjugation key found in `keys`. Errors as [`CudaDevice::rotate`].
    pub fn conjugate<'d>(
        &'d self,
        ct: &CudaCiphertext<'d>,
        keys: &[&DeviceKsk<'d>],
    ) -> Result<CudaCiphertext<'d>> {
        self.check_galois_input(ct)?;
        let elt = conjugation_elt(self.n());
        let key = self.galois_key(keys, elt, ct.level())?;
        self.apply_galois(ct, elt, key)
    }

    /// `Evaluator::hoisted_rotate`: `steps.iter().map(|s| rotate(ct, s, keys))` with the digit
    /// decomposition of `c1` done once. Per nontrivial step, `sigma_g` is applied to every
    /// lifted digit (one launch per digit over its `Q_l * P` limbs) and the result goes
    /// through [`CudaDevice::apply_key`] with the Galois key of `g`; the exact centered lift
    /// commutes with `sigma_g`, so every result is bit-identical to [`CudaDevice::rotate`]
    /// (and to the CPU).
    ///
    /// Results are in the order of `steps`; multiples of the slot count give copies, repeated
    /// steps are allowed. Every key is looked up before any work. Errors as
    /// [`CudaDevice::rotate`].
    ///
    /// Device memory: the `dnum_l` lifted digits of `c1` stay alive for the whole call, plus
    /// one set of rotated digits at a time (n16_train at the top level: 2 x 144 MiB).
    pub fn hoisted_rotate<'d>(
        &'d self,
        ct: &CudaCiphertext<'d>,
        steps: &[i32],
        keys: &[&DeviceKsk<'d>],
    ) -> Result<Vec<CudaCiphertext<'d>>> {
        self.check_galois_input(ct)?;
        let elts: Vec<Option<u64>> = steps.iter().map(|&s| self.rotation_elt(s)).collect();
        let rots = elts
            .iter()
            .flatten()
            .map(|&g| Ok((g, self.galois_key(keys, g, ct.level())?.key_ref())))
            .collect::<Result<Vec<(u64, KeyRef<'_, CudaSlice<u64>>)>>>()?;
        let c = ct.bufs();
        let done = self
            .ks(&Dev(self))
            .hoisted([c[0], c[1]], ct.level(), &rots)?;
        merge_hoisted(
            &elts,
            done,
            || Ok(ct.clone()),
            |r| self.ct_from(r, ct.level(), ct.scale()),
        )
    }
}

/// `InvalidParams` when `key` was uploaded for levels below `level` (checked before any work).
pub(crate) fn check_key_level(key: &DeviceKsk<'_>, level: usize) -> Result<()> {
    if level <= key.max_level() {
        Ok(())
    } else {
        Err(invalid(format!(
            "key switching at level {level} with a key uploaded for levels <= {}",
            key.max_level()
        )))
    }
}
