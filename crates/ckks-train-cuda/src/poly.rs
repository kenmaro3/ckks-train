//! Device-resident RNS polynomials and plaintexts.

use std::sync::Arc;

use ckks_train_ckks::linalg::DiagonalEntry;
use ckks_train_ckks::{Plaintext, QpPoly};
use ckks_train_core::{HekitError, Result, RnsPoly};
use cudarc::driver::CudaSlice;

use crate::device::{CudaDevice, dev_err, guarded};

/// An RNS polynomial in device memory: `num_limbs` limbs of `N` residues, limb `l` contiguous
/// at offset `l * N` of one allocation (`num_limbs * N * 8` bytes). Limb `l` holds the
/// residues modulo the device prime with index [`DevicePoly::primes`]`[l]` (`Q` primes
/// `0..=L`, then `P` primes; see [`CudaDevice`]), so a `Q_l * P` element is one polynomial.
///
/// Invariant: every residue is fully reduced (`< q`): uploads check it and every kernel
/// preserves it, which is what the inverse NTT's `< 2q` precondition needs.
///
/// **`Clone` is a shared handle**: the limbs live in an `Arc`'d device allocation, so `clone`
/// is a reference-count increment. It never allocates device memory and never fails, as
/// `ckks_train_ckks::HeOps` requires of ciphertexts and plaintexts. Sharing is safe because every
/// operation of this crate is functional: it reads its operands and returns a **new**
/// polynomial. The two in-place operations, [`CudaDevice::ntt`] and [`CudaDevice::intt`],
/// copy on write: on a polynomial whose storage is shared, they first make a private copy of
/// the limbs. That copy is fallible (`Device` on out-of-memory), so other handles never see
/// the change. The allocation is freed when the last handle is dropped (stream-ordered).
///
/// It borrows the [`CudaDevice`] it was uploaded to, so it cannot outlive it (stable rustdoc
/// does not check the error code of a `compile_fail` example; the `no_run` twin below differs
/// only in where the device is dropped):
///
/// ```compile_fail,E0505
/// use ckks_train_ckks::CkksParams;
/// use ckks_train_core::RnsPoly;
/// use ckks_train_cuda::CudaDevice;
///
/// let params = CkksParams::test_tiny();
/// let dev = CudaDevice::new(0, &params, 1).unwrap();
/// let poly = dev.upload(&RnsPoly::zero(params.basis_q(), 0)).unwrap();
/// drop(dev); // error[E0505]: cannot move out of `dev` because it is borrowed
/// let _ = poly.download();
/// ```
///
/// The same code with `drop(dev)` after the last use of `poly` compiles:
///
/// ```no_run
/// use ckks_train_ckks::CkksParams;
/// use ckks_train_core::RnsPoly;
/// use ckks_train_cuda::CudaDevice;
///
/// let params = CkksParams::test_tiny();
/// let dev = CudaDevice::new(0, &params, 1).unwrap();
/// let poly = dev.upload(&RnsPoly::zero(params.basis_q(), 0)).unwrap();
/// let _ = poly.download();
/// drop(dev);
/// ```
#[derive(Debug, Clone)]
pub struct DevicePoly<'d> {
    /// Shared device storage (see the type docs: functional operations, copy on write).
    limbs: Arc<CudaSlice<u64>>,
    n: usize,
    dev: &'d CudaDevice,
    /// Prime index (in the device tables) of every limb; distinct.
    primes: Arc<[u32]>,
    ntt_form: bool,
}

impl<'d> DevicePoly<'d> {
    pub(crate) fn from_parts(
        limbs: CudaSlice<u64>,
        n: usize,
        dev: &'d CudaDevice,
        primes: Arc<[u32]>,
        ntt_form: bool,
    ) -> DevicePoly<'d> {
        debug_assert_eq!(limbs.len(), n * primes.len());
        DevicePoly {
            limbs: Arc::new(limbs),
            n,
            dev,
            primes,
            ntt_form,
        }
    }

    /// Ring degree `N`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of limbs (`level + 1` for a polynomial over a prefix of Q or P).
    pub fn num_limbs(&self) -> usize {
        self.primes.len()
    }

    /// `num_limbs - 1` (the level of a polynomial over a prefix of Q or P).
    pub fn level(&self) -> usize {
        self.primes.len() - 1
    }

    /// The prime index of every limb (`0..=L` the Q primes, then the P primes).
    pub fn primes(&self) -> &[u32] {
        &self.primes
    }

    /// `true` when the limbs are in NTT form.
    pub fn is_ntt_form(&self) -> bool {
        self.ntt_form
    }

    /// The device holding this polynomial.
    pub fn device(&self) -> &'d CudaDevice {
        self.dev
    }

    /// Device bytes held by the limbs.
    pub fn device_bytes(&self) -> usize {
        8 * self.n * self.num_limbs()
    }

    pub(crate) fn primes_arc(&self) -> &Arc<[u32]> {
        &self.primes
    }

    pub(crate) fn limbs(&self) -> &CudaSlice<u64> {
        &self.limbs
    }

    pub(crate) fn set_ntt_form(&mut self, ntt_form: bool) {
        self.ntt_form = ntt_form;
    }

    /// Mutable access to the limbs for an in-place operation: when the storage is shared
    /// with another handle (a clone), the limbs are first copied to a private allocation
    /// (copy on write; `Device` on out-of-memory).
    pub(crate) fn limbs_mut(&mut self) -> Result<&mut CudaSlice<u64>> {
        if Arc::get_mut(&mut self.limbs).is_none() {
            let private = self.dev.clone_limbs(&self.limbs)?;
            self.limbs = Arc::new(private);
        }
        Arc::get_mut(&mut self.limbs).ok_or_else(|| {
            HekitError::Device("copy-on-write left the limbs shared (internal error)".into())
        })
    }

    /// `true` when another handle shares this polynomial's storage.
    pub fn is_shared(&self) -> bool {
        Arc::strong_count(&self.limbs) > 1
    }

    fn download_flat(&self) -> Result<Vec<u64>> {
        guarded("download", || {
            self.dev
                .stream()
                .clone_dtoh(&*self.limbs)
                .map_err(|e| dev_err("device-to-host copy", e))
        })
    }

    /// Copies the polynomial back to the host (waits for pending work on it). Its limbs must
    /// be a prefix of the Q primes (returned over `params.basis_q()` of the device's
    /// parameters) or of the P primes (over `params.basis_p()`); `InvalidParams` otherwise,
    /// e.g. for a `Q_l * P` polynomial ([`DevicePoly::download_qp`]). Checked before copying.
    pub fn download(&self) -> Result<RnsPoly> {
        let t = self.dev.prime_table();
        t.host_basis(&self.primes)?; // no copy for a bad layout
        t.host_poly(&self.primes, &self.download_flat()?, self.ntt_form)
    }

    /// Copies a `Q_l * P` polynomial (limbs `[0..=l, L+1..L+1+k]`, all P primes; e.g. the
    /// output of [`CudaDevice::mod_up`]) back to the host as a [`QpPoly`]. `InvalidParams` for
    /// any other prime list.
    pub fn download_qp(&self) -> Result<QpPoly> {
        let t = self.dev.prime_table();
        t.qp_q_limbs(&self.primes)?;
        t.host_qp(&self.primes, &self.download_flat()?, self.ntt_form)
    }
}

/// A plaintext in device memory: NTT-form polynomial, scale and level (as
/// [`ckks_train_ckks::Plaintext`]). Borrows its [`CudaDevice`] like [`DevicePoly`]; `Clone` is a
/// shared handle (a reference-count increment, see [`DevicePoly`]).
#[derive(Debug, Clone)]
pub struct DevicePlaintext<'d> {
    poly: DevicePoly<'d>,
    scale: f64,
    level: usize,
}

impl<'d> DevicePlaintext<'d> {
    pub(crate) fn from_parts(poly: DevicePoly<'d>, scale: f64, level: usize) -> Self {
        DevicePlaintext { poly, scale, level }
    }

    /// The polynomial (NTT form).
    pub fn poly(&self) -> &DevicePoly<'d> {
        &self.poly
    }

    /// The scale.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The level.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Copies the plaintext back to the host.
    pub fn download(&self) -> Result<Plaintext> {
        Plaintext::new(self.poly.download()?, self.scale)
    }
}

impl DiagonalEntry for DevicePlaintext<'_> {
    fn level(&self) -> usize {
        self.level
    }
    fn scale(&self) -> f64 {
        self.scale
    }
    fn slots(&self) -> usize {
        self.poly.n() / 2
    }
}
