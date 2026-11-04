//! CKKS bootstrapping (M3): mod-raise, CoeffToSlot, EvalMod, SlotToCoeff.
//!
//! Task 35 provides the parameters and the level design (`docs/bootstrap.md`): the
//! [`CkksParams::n16_boot`](crate::params::CkksParams::n16_boot) chain, per-level scales
//! ([`CkksParams::scale_at`](crate::params::CkksParams::scale_at)) and [`BootParams`].

pub mod params;

pub use params::{BootParams, levels_after_boot};
