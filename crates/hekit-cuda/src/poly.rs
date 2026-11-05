//! Device-resident RNS polynomials and plaintexts.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use hekit_ckks::Plaintext;
use hekit_core::{Result, RnsBasis, RnsPoly};

use crate::device::{CudaDevice, dev_err, guarded};

/// An RNS polynomial in device memory: `num_limbs` limbs of `N` residues, limb `l` contiguous
/// at offset `l * N` of one allocation (`num_limbs * N * 8` bytes).
///
/// It borrows the [`CudaDevice`] it was uploaded to, so it cannot outlive it:
///
/// ```compile_fail,E0505
/// use hekit_ckks::CkksParams;
/// use hekit_core::RnsPoly;
/// use hekit_cuda::CudaDevice;
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
/// use hekit_ckks::CkksParams;
/// use hekit_core::RnsPoly;
/// use hekit_cuda::CudaDevice;
///
/// let params = CkksParams::test_tiny();
/// let dev = CudaDevice::new(0, &params, 1).unwrap();
/// let poly = dev.upload(&RnsPoly::zero(params.basis_q(), 0)).unwrap();
/// let _ = poly.download();
/// drop(dev);
/// ```
#[derive(Debug)]
pub struct DevicePoly<'d> {
    limbs: CudaSlice<u64>,
    n: usize,
    num_limbs: usize,
    dev: &'d CudaDevice,
    /// The host basis, to rebuild an [`RnsPoly`] on download.
    basis: Arc<RnsBasis>,
    /// Index of limb 0's prime in the device tables (0 for Q, `L + 1` for P).
    prime_offset: usize,
    ntt_form: bool,
}

impl<'d> DevicePoly<'d> {
    pub(crate) fn from_parts(
        limbs: CudaSlice<u64>,
        n: usize,
        num_limbs: usize,
        dev: &'d CudaDevice,
        basis: Arc<RnsBasis>,
        prime_offset: usize,
        ntt_form: bool,
    ) -> DevicePoly<'d> {
        DevicePoly {
            limbs,
            n,
            num_limbs,
            dev,
            basis,
            prime_offset,
            ntt_form,
        }
    }

    /// Ring degree `N`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of limbs (`level + 1`).
    pub fn num_limbs(&self) -> usize {
        self.num_limbs
    }

    /// The level (`num_limbs - 1`).
    pub fn level(&self) -> usize {
        self.num_limbs - 1
    }

    /// `true` when the limbs are in NTT form.
    pub fn is_ntt_form(&self) -> bool {
        self.ntt_form
    }

    /// The device holding this polynomial.
    pub fn device(&self) -> &'d CudaDevice {
        self.dev
    }

    /// The host basis this polynomial was uploaded from.
    pub fn basis(&self) -> &Arc<RnsBasis> {
        &self.basis
    }

    /// Device bytes held by the limbs.
    pub fn device_bytes(&self) -> usize {
        8 * self.n * self.num_limbs
    }

    pub(crate) fn prime_offset(&self) -> usize {
        self.prime_offset
    }

    pub(crate) fn set_ntt_form(&mut self, ntt_form: bool) {
        self.ntt_form = ntt_form;
    }

    pub(crate) fn limbs_mut(&mut self) -> &mut CudaSlice<u64> {
        &mut self.limbs
    }

    /// Copies the polynomial back to the host (waits for pending work on it).
    pub fn download(&self) -> Result<RnsPoly> {
        let flat = guarded("download", || {
            self.dev
                .stream()
                .clone_dtoh(&self.limbs)
                .map_err(|e| dev_err("device-to-host copy", e))
        })?;
        let limbs: Vec<Vec<u64>> = flat.chunks_exact(self.n).map(<[u64]>::to_vec).collect();
        RnsPoly::from_limbs(&self.basis, limbs, self.ntt_form)
    }
}

/// A plaintext in device memory: NTT-form polynomial, scale and level (as
/// [`hekit_ckks::Plaintext`]). Borrows its [`CudaDevice`] like [`DevicePoly`].
#[derive(Debug)]
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
