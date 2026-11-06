//! CKKS bootstrapping (M3): mod-raise, CoeffToSlot, EvalMod, SlotToCoeff.
//!
//! Task 35 provides the parameters and the level design (`docs/bootstrap.md`): the
//! [`CkksParams::n16_boot`](crate::params::CkksParams::n16_boot) chain, per-level scales
//! ([`CkksParams::scale_at`](crate::params::CkksParams::scale_at)) and [`BootParams`].
//! Task 36 adds the CoeffToSlot / SlotToCoeff linear transforms ([`linear`]). Task 37 adds
//! the mod-raise with sparse-secret encapsulation ([`modraise`], keys in
//! [`SseKeys`](crate::SseKeys)), the level-optimal Chebyshev evaluation ([`poly`]) and
//! EvalMod ([`evalmod`]). Task 38 assembles them into the [`Bootstrapper`] ([`bootstrapper`]).

pub mod bootstrapper;
pub mod evalmod;
pub mod linear;
pub mod metadata;
pub mod modraise;
pub mod params;
pub mod poly;

pub use bootstrapper::{BootstrapKeys, Bootstrapper};
pub use evalmod::{evalmod_coeffs, evalmod_cost};
pub use linear::{
    DiagMatrix, Direction, LinearTransform, StageSpec, dft_stages, encode_stages, galois_steps_for,
    stage_radices, transforms_cost,
};
pub use params::{BootParams, levels_after_boot};
pub use poly::{chebyshev_coeffs, chebyshev_cost, chebyshev_depth, chebyshev_eval, optimal_split};
