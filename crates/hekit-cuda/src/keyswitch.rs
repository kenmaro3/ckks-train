//! Hybrid key switching on the device: key-switching keys in device memory ([`DeviceKsk`]),
//! the digit decomposition ([`CudaDevice::decompose`]), the inner product with the key and
//! the division by `P` ([`CudaDevice::apply_key`]), and [`CudaDevice::key_switch`]. Each is
//! bit-identical to its `hekit_ckks::keyswitch` counterpart; the launch sequences are the
//! shared ones of [`crate::ks`].

use std::sync::Arc;

use cudarc::driver::{CudaSlice, DevicePtr, PushKernelArg};
use hekit_ckks::keyswitch::digit_ranges;
use hekit_ckks::{GaloisKey, KskKey, RelinKey};
use hekit_core::{HekitError, Result};

use crate::arith::Dev;
use crate::basis::check_galois_elt;
use crate::device::{CudaDevice, dev_err, guarded};
use crate::ks::{InnerProduct, KeyRef, Ks, KsBackend, KskShape, PwOp, ksk_limbs, q_primes};
use crate::plan::ElemLaunch;
use crate::poly::DevicePoly;

fn invalid(msg: impl Into<String>) -> HekitError {
    HekitError::InvalidParams(msg.into())
}

/// What a [`DeviceKsk`] switches from, as far as the device operations check it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KskPurpose {
    /// Any key-switching key ([`CudaDevice::upload_ksk`]): usable with
    /// [`CudaDevice::key_switch`] / [`CudaDevice::apply_key`] only.
    Generic,
    /// A relinearization key (`s^2 -> s`): also [`CudaDevice::relinearize`].
    Relin,
    /// A Galois key for this element (`sigma_g(s) -> s`): also the rotations and conjugation.
    Galois(u64),
}

/// A hybrid key-switching key in device memory: the parts `(b_j, a_j)`, `j < parts`, of a
/// [`KskKey`], each polynomial with its `Q` limbs `0..=max_level` and all its `P` limbs, in
/// **one** allocation of [`KskShape::device_bytes`] (n16_train, full key: 288 MiB; seal_2020:
/// 55 MiB). Layout in `kernels/keyswitch.cu`.
///
/// Key switching at level `l` reads the key's `Q` limbs `0..=l` and its `P` limbs, which are
/// not contiguous for `l < max_level`; the inner-product kernel skips the limbs in between in
/// place (`key_skip`), so nothing is gathered or copied per level.
///
/// Borrows its [`CudaDevice`] like [`DevicePoly`]. `Clone` is a shared handle (a
/// reference-count increment, no device memory).
#[derive(Debug, Clone)]
pub struct DeviceKsk<'d> {
    buf: Arc<CudaSlice<u64>>,
    shape: KskShape,
    purpose: KskPurpose,
    dev: &'d CudaDevice,
}

impl<'d> DeviceKsk<'d> {
    /// The stored parts and limbs.
    pub fn shape(&self) -> &KskShape {
        &self.shape
    }

    /// The highest level this key can switch at.
    pub fn max_level(&self) -> usize {
        self.shape.max_level
    }

    /// What the key switches from.
    pub fn purpose(&self) -> KskPurpose {
        self.purpose
    }

    /// The Galois element of a Galois key.
    pub fn galois_elt(&self) -> Option<u64> {
        match self.purpose {
            KskPurpose::Galois(g) => Some(g),
            _ => None,
        }
    }

    /// Device bytes held by the key.
    pub fn device_bytes(&self) -> usize {
        8 * self.buf.len()
    }

    /// The device holding this key.
    pub fn device(&self) -> &'d CudaDevice {
        self.dev
    }

    pub(crate) fn key_ref(&self) -> KeyRef<'_, CudaSlice<u64>> {
        KeyRef {
            buf: &self.buf,
            shape: &self.shape,
        }
    }
}

impl CudaDevice {
    /// Copies a key-switching key (all `dnum` parts, all limbs) to the device: a
    /// [`KskPurpose::Generic`] key for [`CudaDevice::key_switch`] / [`CudaDevice::apply_key`]
    /// at every level.
    ///
    /// Errors: `InvalidParams` when the key is not over this device's parameters (part
    /// count, primes by value, NTT form); `Device` when it does not fit in device memory (the
    /// allocation fails, nothing is allocated: never an abort) or the copy fails.
    pub fn upload_ksk<'d>(&'d self, key: &KskKey) -> Result<DeviceKsk<'d>> {
        self.upload_ksk_with(key, self.params().max_level())
    }

    /// [`CudaDevice::upload_ksk`] of a relinearization key ([`KskPurpose::Relin`]).
    pub fn upload_relin_key<'d>(&'d self, rk: &RelinKey) -> Result<DeviceKsk<'d>> {
        self.upload_relin_key_with(rk, self.params().max_level())
    }

    /// [`CudaDevice::upload_ksk`] of a Galois key ([`KskPurpose::Galois`] with its element).
    pub fn upload_galois_key<'d>(&'d self, gk: &GaloisKey) -> Result<DeviceKsk<'d>> {
        self.upload_galois_key_with(gk, self.params().max_level())
    }

    /// A [`KskPurpose::Generic`] key uploaded for levels `<= max_level` only: the parts of the
    /// digits of `max_level`, the `Q` limbs `0..=max_level` and the `P` limbs (see
    /// [`KskShape`]). With `max_level` below `L` the device key is smaller (n16_train at
    /// `max_level = 9`: 42 MiB, at 3: 8 MiB, instead of 288 MiB); switching above `max_level`
    /// is then `InvalidParams`.
    ///
    /// Errors as [`CudaDevice::upload_ksk`]; `InvalidParams` also for `max_level > L`.
    pub fn upload_ksk_with<'d>(&'d self, key: &KskKey, max_level: usize) -> Result<DeviceKsk<'d>> {
        self.upload_key(key, KskPurpose::Generic, max_level)
    }

    /// [`CudaDevice::upload_relin_key`] for levels `<= max_level` only (as
    /// [`CudaDevice::upload_ksk_with`]); the purpose comes from the key's type.
    pub fn upload_relin_key_with<'d>(
        &'d self,
        rk: &RelinKey,
        max_level: usize,
    ) -> Result<DeviceKsk<'d>> {
        self.upload_key(rk.key(), KskPurpose::Relin, max_level)
    }

    /// [`CudaDevice::upload_galois_key`] for levels `<= max_level` only (as
    /// [`CudaDevice::upload_ksk_with`]); the purpose comes from the key's type.
    pub fn upload_galois_key_with<'d>(
        &'d self,
        gk: &GaloisKey,
        max_level: usize,
    ) -> Result<DeviceKsk<'d>> {
        self.upload_key(gk.key(), KskPurpose::Galois(gk.elt()), max_level)
    }

    fn upload_key<'d>(
        &'d self,
        key: &KskKey,
        purpose: KskPurpose,
        max_level: usize,
    ) -> Result<DeviceKsk<'d>> {
        if let KskPurpose::Galois(g) = purpose {
            check_galois_elt(self.n(), g)?;
        }
        let shape = KskShape::new(self.params(), max_level)?;
        let limbs = ksk_limbs(self.prime_table(), self.params(), key, &shape)?;
        // One allocation (a fallible driver call: out of memory is `Device`), then one copy per
        // limb from the host key (no host staging copy of the whole key).
        let mut buf = self.alloc_limbs(limbs.len())?;
        let n = self.n();
        guarded("key upload", || {
            for (i, limb) in limbs.iter().enumerate() {
                self.stream()
                    .memcpy_htod(*limb, &mut buf.slice_mut(i * n..(i + 1) * n))
                    .map_err(|e| dev_err("key upload (host-to-device copy)", e))?;
            }
            Ok(())
        })?;
        Ok(DeviceKsk {
            buf: Arc::new(buf),
            shape,
            purpose,
            dev: self,
        })
    }

    pub(crate) fn check_key_owner(&self, key: &DeviceKsk<'_>) -> Result<()> {
        if std::ptr::eq(key.dev, self) {
            Ok(())
        } else {
            Err(invalid("DeviceKsk belongs to another CudaDevice"))
        }
    }

    /// The shared launch sequences ([`Ks`]) on this device: `self.ks(&Dev(self))`.
    pub(crate) fn ks<'a>(&'a self, d: &'a Dev<'a>) -> Ks<'a, Dev<'a>> {
        Ks {
            b: d,
            params: self.params(),
            table: self.prime_table(),
        }
    }

    /// The level of a key-switching input: a polynomial of this device over the `Q` primes
    /// `0..=l`, NTT form.
    fn switch_input_level(&self, x: &DevicePoly<'_>) -> Result<usize> {
        self.check_owner(x)?;
        if !x.is_ntt_form() {
            return Err(invalid("key switching needs an NTT-form polynomial"));
        }
        self.prime_table()
            .q_prefix_level(x.primes())
            .ok_or_else(|| invalid("key switching needs a polynomial over the Q primes 0..=l"))
    }

    /// `hekit_ckks::keyswitch::decompose`: the exact centered lift of every digit of `x`
    /// (NTT form, over the `Q` primes `0..=l`) to `Q_l * P`, one [`CudaDevice::mod_up`] per
    /// digit of `digit_ranges(params, l)`. Each digit has limbs `[0..=l, L+1..L+1+k]`, NTT form
    /// ([`DevicePoly::download_qp`]).
    pub fn decompose<'d>(&'d self, x: &DevicePoly<'d>) -> Result<Vec<DevicePoly<'d>>> {
        let level = self.switch_input_level(x)?;
        let primes: Arc<[u32]> = self.prime_table().qp_primes(level).into();
        let digits = self.ks(&Dev(self)).decompose(x.limbs(), level)?;
        Ok(digits
            .into_iter()
            .map(|d| self.new_poly(d, Arc::clone(&primes), true))
            .collect())
    }

    /// `hekit_ckks::keyswitch::apply_key`: the inner product of the lifted digits with the
    /// key parts (**one** `ks_inner_product` launch for every digit, every limb of `Q_l * P`
    /// and both accumulators), then `mod_down` of each accumulator: `(kb, ka)` at level `l`
    /// over the `Q` primes, NTT form. `digits[j]` is used with key part `j`; the digits would
    /// normally come from [`CudaDevice::decompose`], possibly with an automorphism applied to
    /// each (hoisting).
    ///
    /// `InvalidParams` unless the digits belong to this device, are NTT-form `Q_l * P`
    /// polynomials of one level `l <= key.max_level()`, and number
    /// `digit_ranges(params, l).len()`.
    pub fn apply_key<'d>(
        &'d self,
        digits: &[DevicePoly<'d>],
        key: &DeviceKsk<'d>,
    ) -> Result<(DevicePoly<'d>, DevicePoly<'d>)> {
        self.check_key_owner(key)?;
        let first = digits
            .first()
            .ok_or_else(|| invalid("no digits to switch"))?;
        let t = self.prime_table();
        let level = t.qp_q_limbs(first.primes())? - 1;
        for d in digits {
            self.check_owner(d)?;
            if d.primes() != first.primes() || !d.is_ntt_form() {
                return Err(invalid(
                    "digits must be NTT-form Q_l * P polynomials of one level",
                ));
            }
        }
        let refs: Vec<&CudaSlice<u64>> = digits.iter().map(DevicePoly::limbs).collect();
        let (kb, ka) = self.ks(&Dev(self)).apply_key(&refs, level, key.key_ref())?;
        Ok(self.q_pair(kb, ka, level))
    }

    /// `hekit_ckks::keyswitch::key_switch`: `apply_key(decompose(x))` for `x` NTT form over
    /// the `Q` primes `0..=l`, `l <= key.max_level()`: `(kb, ka)` at level `l`.
    pub fn key_switch<'d>(
        &'d self,
        x: &DevicePoly<'d>,
        key: &DeviceKsk<'d>,
    ) -> Result<(DevicePoly<'d>, DevicePoly<'d>)> {
        self.check_key_owner(key)?;
        let level = self.switch_input_level(x)?;
        crate::ciphertext::check_key_level(key, level)?;
        let (kb, ka) = self
            .ks(&Dev(self))
            .key_switch(x.limbs(), level, key.key_ref())?;
        Ok(self.q_pair(kb, ka, level))
    }

    fn q_pair(
        &self,
        a: CudaSlice<u64>,
        b: CudaSlice<u64>,
        level: usize,
    ) -> (DevicePoly<'_>, DevicePoly<'_>) {
        let primes: Arc<[u32]> = q_primes(level).into();
        (
            self.new_poly(a, Arc::clone(&primes), true),
            self.new_poly(b, primes, true),
        )
    }

    /// Number of digits of a level-`l` polynomial (`digit_ranges(params, l).len()`).
    pub fn num_digits(&self, level: usize) -> usize {
        digit_ranges(self.params(), level).len()
    }
}

impl KsBackend for Dev<'_> {
    fn binary(
        &self,
        op: PwOp,
        out: &mut CudaSlice<u64>,
        a: &CudaSlice<u64>,
        b: &CudaSlice<u64>,
        primes: &[u32],
    ) -> Result<()> {
        self.0.launch_binary(op, out, a, b, primes)
    }

    fn automorphism(
        &self,
        out: &mut CudaSlice<u64>,
        a: &CudaSlice<u64>,
        primes: &[u32],
        ntt_form: bool,
        galois_elt: u64,
    ) -> Result<()> {
        self.0
            .launch_automorphism(out, a, primes, ntt_form, galois_elt)
    }

    fn inner_product(
        &self,
        out_b: &mut CudaSlice<u64>,
        out_a: &mut CudaSlice<u64>,
        digits: &[&CudaSlice<u64>],
        key: &CudaSlice<u64>,
        ip: &InnerProduct,
    ) -> Result<()> {
        let dev = self.0;
        let n = dev.n();
        let rows = ip.out_primes.len();
        if digits.len() != ip.n_digits
            || digits.iter().any(|d| d.len() != rows * n)
            || out_b.len() != rows * n
            || out_a.len() != rows * n
            || key.len() < 2 * ip.n_digits * ip.key_limbs * n
        {
            return Err(invalid("inner-product buffers do not match the launch"));
        }
        let map = dev.limb_map(&ip.out_primes)?;
        let l = ElemLaunch::new(n, rows)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: l.grid,
            block_dim: l.block,
            shared_mem_bytes: 0,
        };
        let (log_n, n_digits, q_limbs, key_limbs, key_skip) = (
            n.trailing_zeros(),
            ip.n_digits as u32,
            ip.q_limbs as u32,
            ip.key_limbs as u32,
            ip.key_skip as u32,
        );
        let stream = dev.stream();
        guarded("ks_inner_product", || {
            // The digit addresses, uploaded as one small array. Each `device_ptr` guard records
            // the digit's read on drop (event tracking), after the launch below is queued.
            let (ptrs, _reads): (Vec<u64>, Vec<_>) =
                digits.iter().map(|d| d.device_ptr(stream)).unzip();
            let ptr_buf = stream
                .clone_htod(&ptrs)
                .map_err(|e| dev_err("uploading the digit addresses", e))?;
            let mut b = stream.launch_builder(&dev.kernels.ks_inner_product);
            b.arg(out_b)
                .arg(out_a)
                .arg(&ptr_buf)
                .arg(key)
                .arg(&dev.ntt_tables.consts)
                .arg(&*map)
                .arg(&log_n)
                .arg(&n_digits)
                .arg(&q_limbs)
                .arg(&key_limbs)
                .arg(&key_skip);
            // SAFETY: matches `(u64*, u64*, const u64*, const u64*, const u64*, const u32*, u32,
            // u32, u32, u32, u32)` in kernels/keyswitch.cu. out_b, out_a and every digit hold
            // rows = q_limbs + k limbs (checked above); ptr_buf holds n_digits addresses of
            // live digit allocations (borrowed for this call; their frees are stream-ordered
            // after the launch); the key holds at least 2 * n_digits * key_limbs limbs and
            // every key limb index t = y or y + key_skip is < key_limbs (y < q_limbs = l + 1
            // <= max_level + 1, or y - q_limbs < k with y + key_skip = max_level + 1 + (y -
            // q_limbs)); the map has rows valid prime indices; out_b and out_a are distinct
            // allocations (`&mut` borrows) not aliasing the inputs.
            unsafe { b.launch(cfg) }.map_err(|e| dev_err("ks_inner_product", e))?;
            Ok(())
        })
    }
}
