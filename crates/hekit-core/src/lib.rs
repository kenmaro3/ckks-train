//! hekit-core: RNS arithmetic, NTT and sampling primitives (temporary crate name).
#![deny(missing_docs)]

pub mod error;
pub mod modulus;
pub mod primes;

pub use error::{HekitError, Result};
pub use modulus::{Modulus, Shoup};
