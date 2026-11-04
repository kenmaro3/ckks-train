//! hekit-ckks: RNS-CKKS scheme built on hekit-core (temporary crate name).
#![deny(missing_docs)]

pub mod encoder;
pub mod params;
pub mod plaintext;

pub use encoder::Encoder;
pub use params::CkksParams;
pub use plaintext::Plaintext;
