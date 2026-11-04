//! CKKS bootstrapping (M3): mod-raise, CoeffToSlot, EvalMod, SlotToCoeff.
//!
//! Task 35 provides the parameters and the level design (`docs/bootstrap.md`): the
//! [`CkksParams::n16_boot`](crate::params::CkksParams::n16_boot) chain, per-level scales
//! ([`CkksParams::scale_at`](crate::params::CkksParams::scale_at)) and [`BootParams`].
//! Task 36 adds the CoeffToSlot / SlotToCoeff linear transforms ([`linear`]). Task 37 adds
//! the mod-raise with sparse-secret encapsulation ([`modraise`], keys in
//! [`SseKeys`](crate::SseKeys)), the level-optimal Chebyshev evaluation ([`poly`]) and
//! EvalMod ([`evalmod`]).

pub mod evalmod;
pub mod linear;
pub mod modraise;
pub mod params;
pub mod poly;

pub use evalmod::evalmod_coeffs;
pub use linear::{
    DiagMatrix, Direction, LinearTransform, StageSpec, dft_stages, encode_stages, galois_steps_for,
    stage_radices,
};
pub use params::{BootParams, levels_after_boot};
pub use poly::{chebyshev_coeffs, chebyshev_depth, chebyshev_eval, optimal_split};
