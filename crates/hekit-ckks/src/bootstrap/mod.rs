//! CKKS bootstrapping (M3): mod-raise, CoeffToSlot, EvalMod, SlotToCoeff.
//!
//! Task 35 provides the parameters and the level design (`docs/bootstrap.md`): the
//! [`CkksParams::n16_boot`](crate::params::CkksParams::n16_boot) chain, per-level scales
//! ([`CkksParams::scale_at`](crate::params::CkksParams::scale_at)) and [`BootParams`].
//! Task 36 adds the CoeffToSlot / SlotToCoeff linear transforms ([`linear`]).

pub mod linear;
pub mod params;

pub use linear::{
    DiagMatrix, Direction, LinearTransform, StageSpec, dft_stages, encode_stages, galois_steps_for,
    stage_radices,
};
pub use params::{BootParams, levels_after_boot};
