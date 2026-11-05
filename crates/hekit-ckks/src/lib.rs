//! hekit-ckks: RNS-CKKS scheme built on hekit-core (temporary crate name).
#![deny(missing_docs)]

pub mod bootstrap;
pub mod ciphertext;
pub mod counter;
pub mod encoder;
pub mod encrypt;
pub mod evaluator;
pub mod keys;
pub mod keyswitch;
pub mod linalg;
pub mod ops;
mod par;
pub mod params;
pub mod plaintext;
pub mod precision;
pub mod seal_compat;

pub use ciphertext::Ciphertext;
pub use counter::OpCounter;
pub use encoder::Encoder;
pub use encrypt::{Decryptor, Encryptor};
pub use evaluator::Evaluator;
pub use keys::{
    GaloisKey, GaloisKeys, KeyGenerator, KskKey, PublicKey, QpPoly, RelinKey, SecretKey, SseKeys,
};
pub use linalg::{
    DiagonalEntry, Diagonals, bsgs_rotation_steps, bsgs_rotation_steps_with_offset, bsgs_split,
    diagonals_from_matrix,
};
pub use ops::{BootstrapOps, EvalConfig, HeCiphertext, HeOps, HostBootstrapper};
pub use params::CkksParams;
pub use plaintext::Plaintext;
