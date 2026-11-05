//! hekit-cuda: CUDA backend for hekit RNS-CKKS (temporary crate name).
//!
//! With the `cuda` feature (off by default) the crate provides a device context
//! ([`CudaDevice`]), device-resident RNS polynomials ([`DevicePoly`], [`DevicePlaintext`]) and,
//! each bit-identical to the CPU code:
//!
//! * the batched NTT / inverse NTT over all limbs of a polynomial (one launch sequence for
//!   every limb at once): `hekit_core::NttTable::forward` / `inverse`;
//! * pointwise `add`, `sub`, `neg`, `mul_pointwise`, `mul_scalar` and the Galois
//!   `automorphism` (either form): the `RnsPoly` methods of the same names;
//! * `rescale_poly` (`RnsPoly::rescale_by_last`), `mod_down` (`RnsPoly::mod_down` of a
//!   `Q_l * P` pair) and `mod_up` (the exact centered digit lift of
//!   `hekit_ckks::keyswitch::decompose`).
//!
//! A device polynomial carries one prime index per limb, so a `Q_l * P` element (limbs
//! `0..=l` then the special primes) is one polynomial handled by single launches.
//!
//! Without the feature only the host-side pieces are compiled: the kernel launch
//! configurations ([`plan`]), the flat twiddle and prime tables ([`tables`]), and the
//! base-conversion tables and launch sequences ([`basis`]). Their tests replay the CUDA
//! kernels thread by thread on the CPU, run the device's launch sequences on the replay, and
//! compare with the CPU implementation bit for bit (the same checks as the GPU tests).
//!
//! # Runtime requirements (`cuda` feature)
//!
//! `cudarc` loads `libcuda` and `libnvrtc` at first use (nothing is linked at build time), so
//! the crate builds on machines without CUDA. At run time it needs an NVIDIA driver supporting
//! CUDA >= 13.1 and NVRTC 13.1 on the library path (e.g. `/usr/local/cuda-13.1/lib64` in
//! `LD_LIBRARY_PATH`). The kernels (`kernels/ntt.cu`, `pointwise.cu`, `basis.cu`) are compiled
//! by NVRTC (with `--fmad=false`) when a [`CudaDevice`] is created. Every failure (no driver,
//! no device, bad ordinal, out of device memory, compilation or launch failure) is returned as
//! [`hekit_core::HekitError::Device`].
//!
//! # Lifetimes
//!
//! [`DevicePoly<'d>`](DevicePoly) and [`DevicePlaintext<'d>`](DevicePlaintext) borrow the
//! [`CudaDevice`] they live on, so they cannot outlive it (see the `compile_fail` example on
//! [`DevicePoly`]).
#![deny(missing_docs)]

pub mod basis;
pub mod plan;
pub mod tables;

#[cfg(test)]
#[path = "../tests/cases/mod.rs"]
mod cases;
#[cfg(test)]
mod emu;
#[cfg(test)]
mod emu_arith;

#[cfg(feature = "cuda")]
mod arith;
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
/// Source of the element-wise kernels (add, sub, neg, products, automorphisms).
pub const POINTWISE_KERNEL_SRC: &str = include_str!("../kernels/pointwise.cu");
/// Source of the base-conversion kernels (rescale, mod_down, mod_up).
pub const BASIS_KERNEL_SRC: &str = include_str!("../kernels/basis.cu");
