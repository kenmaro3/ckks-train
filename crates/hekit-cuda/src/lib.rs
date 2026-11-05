//! hekit-cuda: CUDA backend for hekit RNS-CKKS (temporary crate name).
//!
//! With the `cuda` feature (off by default) the crate provides a device context
//! ([`CudaDevice`]), device-resident RNS polynomials ([`DevicePoly`], [`DevicePlaintext`]) and
//! the batched NTT / inverse NTT over all limbs of a polynomial (one launch sequence for every
//! limb at once), bit-identical to [`hekit_core::NttTable::forward`] / `inverse`.
//!
//! Without the feature only the host-side pieces are compiled: the kernel launch
//! configuration ([`plan`]) and the flat twiddle tables ([`tables`]). Their tests replay the
//! CUDA kernels thread by thread on the CPU and compare with `NttTable` bit for bit.
//!
//! # Runtime requirements (`cuda` feature)
//!
//! `cudarc` loads `libcuda` and `libnvrtc` at first use (nothing is linked at build time), so
//! the crate builds on machines without CUDA. At run time it needs an NVIDIA driver supporting
//! CUDA >= 13.1 and NVRTC 13.1 on the library path (e.g. `/usr/local/cuda-13.1/lib64` in
//! `LD_LIBRARY_PATH`). The kernels (`kernels/ntt.cu`) are compiled by NVRTC when a
//! [`CudaDevice`] is created. Every failure (no driver, no device, bad ordinal, out of device
//! memory, compilation or launch failure) is returned as [`hekit_core::HekitError::Device`].
//!
//! # Lifetimes
//!
//! [`DevicePoly<'d>`](DevicePoly) and [`DevicePlaintext<'d>`](DevicePlaintext) borrow the
//! [`CudaDevice`] they live on, so they cannot outlive it (see the `compile_fail` example on
//! [`DevicePoly`]).
#![deny(missing_docs)]

pub mod plan;
pub mod tables;

#[cfg(test)]
mod emu;

#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
mod poly;
#[cfg(feature = "cuda")]
pub mod testing;

#[cfg(feature = "cuda")]
pub use device::{CudaDevice, probe};
#[cfg(feature = "cuda")]
pub use poly::{DevicePlaintext, DevicePoly};

/// Source of the NTT kernels, compiled at run time with NVRTC.
pub const NTT_KERNEL_SRC: &str = include_str!("../kernels/ntt.cu");
