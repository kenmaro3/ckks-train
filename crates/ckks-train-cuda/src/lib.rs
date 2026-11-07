//! ckks-train-cuda: CUDA backend for hekit RNS-CKKS (temporary crate name).
//!
//! With the `cuda` feature (off by default) the crate provides a device context
//! ([`CudaDevice`]), device-resident RNS polynomials ([`DevicePoly`], [`DevicePlaintext`]) and,
//! each bit-identical to the CPU code:
//!
//! * the batched NTT / inverse NTT over all limbs of a polynomial (one launch sequence for
//!   every limb at once): `ckks_train_core::NttTable::forward` / `inverse`;
//! * pointwise `add`, `sub`, `neg`, `mul_pointwise`, `mul_scalar` and the Galois
//!   `automorphism` (either form): the `RnsPoly` methods of the same names;
//! * `rescale_poly` (`RnsPoly::rescale_by_last`), `mod_down` (`RnsPoly::mod_down` of a
//!   `Q_l * P` pair) and `mod_up` (the exact centered digit lift of
//!   `ckks_train_ckks::keyswitch::decompose`);
//! * hybrid key switching with key-switching keys in device memory ([`DeviceKsk`]):
//!   `decompose`, `apply_key` (one inner-product launch for all digits) and `key_switch`
//!   (`ckks_train_ckks::keyswitch`), and on device ciphertexts ([`CudaCiphertext`]) the
//!   `ckks_train_ckks::Evaluator` operations `mul_no_relin`, `relinearize`, `rescale`, `rotate`,
//!   `conjugate` and `hoisted_rotate`.
//!
//! * the evaluator [`CudaEvaluator`]: `ckks_train_ckks::HeOps` on device ciphertexts (the CPU's
//!   checks, errors and `OpCounter` counts, the RQ3 flags), BSGS matrix-vector products
//!   (the shared orchestration of [`matvec`], bit-identical to the CPU's) and
//!   [`CudaEvaluator::matvec_enc_batch`], independent products on several streams.
//!
//! Device ciphertexts, plaintexts, polynomials and keys are shared handles: `Clone` is a
//! reference-count increment (no device memory, cannot fail), which is safe because every
//! operation is functional (returns new values; the in-place NTTs copy on write).
//!
//! A device polynomial carries one prime index per limb, so a `Q_l * P` element (limbs
//! `0..=l` then the special primes) is one polynomial handled by single launches.
//!
//! Without the feature only the host-side pieces are compiled: the kernel launch
//! configurations ([`plan`]), the flat twiddle and prime tables ([`tables`]), the
//! base-conversion tables and launch sequences ([`basis`]), and the key-switching key layout
//! and launch sequences ([`ks`]), and the backend-neutral BSGS orchestration ([`matvec`]).
//! Their tests replay the CUDA
//! kernels thread by thread on the CPU, run the device's launch sequences on the replay, and
//! compare with the CPU implementation bit for bit (the same checks as the GPU tests).
//!
//! # Runtime requirements (`cuda` feature)
//!
//! `cudarc` loads `libcuda` and `libnvrtc` at first use (nothing is linked at build time), so
//! the crate builds on machines without CUDA. At run time it needs an NVIDIA driver supporting
//! CUDA >= 13.1 and NVRTC 13.1 on the library path (e.g. `/usr/local/cuda-13.1/lib64` in
//! `LD_LIBRARY_PATH`). The kernels (`kernels/ntt.cu`, `pointwise.cu`, `basis.cu`,
//! `keyswitch.cu`) are compiled
//! by NVRTC (with `--fmad=false`) when a [`CudaDevice`] is created. Every failure (no driver,
//! no device, bad ordinal, out of device memory, compilation or launch failure) is returned as
//! [`ckks_train_core::HekitError::Device`].
//!
//! # Lifetimes
//!
//! [`DevicePoly<'d>`](DevicePoly), [`DevicePlaintext<'d>`](DevicePlaintext),
//! [`CudaCiphertext<'d>`](CudaCiphertext) and [`DeviceKsk<'d>`](DeviceKsk) borrow the
//! [`CudaDevice`] they live on, so they cannot outlive it (see the `compile_fail` examples on
//! [`DevicePoly`] and [`CudaCiphertext`]).
#![deny(missing_docs)]
// The crate docs above link to the device types, which exist only with the `cuda` feature.
// Without it those links cannot resolve; with it (and on docs.rs, see Cargo.toml) every link
// is checked.
#![cfg_attr(not(feature = "cuda"), allow(rustdoc::broken_intra_doc_links))]

pub mod basis;
#[cfg(any(test, feature = "cuda"))]
pub mod bootstrap;
pub mod ks;
pub mod matvec;
pub mod plan;
pub mod tables;

#[cfg(test)]
#[path = "../tests/cases/mod.rs"]
mod cases;
#[cfg(test)]
mod emu;
#[cfg(test)]
mod emu_arith;
#[cfg(test)]
mod emu_ks;

#[cfg(feature = "cuda")]
mod arith;
#[cfg(feature = "cuda")]
mod ciphertext;
#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
pub mod evaluator;
#[cfg(feature = "cuda")]
mod keyswitch;
#[cfg(feature = "cuda")]
mod poly;
#[cfg(feature = "cuda")]
mod runtime_observation;
#[cfg(feature = "cuda")]
pub mod testing;

#[cfg(feature = "cuda")]
pub use ciphertext::{CudaCiphertext, rotation_elt};
#[cfg(feature = "cuda")]
pub use device::{CudaDevice, StreamGuard, probe};
#[cfg(feature = "cuda")]
pub use evaluator::CudaEvaluator;
#[cfg(feature = "cuda")]
pub use keyswitch::{DeviceKsk, KskPurpose};
#[cfg(feature = "cuda")]
pub use poly::{DevicePlaintext, DevicePoly};
#[cfg(feature = "cuda")]
pub use runtime_observation::{
    CudaExportedFunction, CudaLibraryKind, CudaPhysicalIdentity, CudaRuntimeObservation,
};

/// Source of the NTT kernels, compiled at run time with NVRTC.
pub const NTT_KERNEL_SRC: &str = include_str!("../kernels/ntt.cu");
/// Source of the element-wise kernels (add, sub, neg, products, automorphisms).
pub const POINTWISE_KERNEL_SRC: &str = include_str!("../kernels/pointwise.cu");
/// Source of the base-conversion kernels (rescale, mod_down, mod_up).
pub const BASIS_KERNEL_SRC: &str = include_str!("../kernels/basis.cu");
/// Source of the key-switching inner-product kernel.
pub const KEYSWITCH_KERNEL_SRC: &str = include_str!("../kernels/keyswitch.cu");

#[cfg(feature = "cuda")]
pub use bootstrap::CudaBootstrapper;
