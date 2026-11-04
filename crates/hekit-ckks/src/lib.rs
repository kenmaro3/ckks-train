//! hekit-ckks: RNS-CKKS scheme built on hekit-core (temporary crate name).
#![deny(missing_docs)]

pub mod ciphertext;
pub mod counter;
pub mod encoder;
pub mod encrypt;
pub mod evaluator;
pub mod keys;
pub mod keyswitch;
pub mod linalg;
pub mod params;
pub mod plaintext;
pub mod seal_compat;

pub use ciphertext::Ciphertext;
pub use counter::OpCounter;
pub use encoder::Encoder;
pub use encrypt::{Decryptor, Encryptor};
pub use evaluator::Evaluator;
pub use keys::{
    GaloisKey, GaloisKeys, KeyGenerator, KskKey, PublicKey, QpPoly, RelinKey, SecretKey,
};
pub use linalg::{Diagonals, bsgs_rotation_steps, bsgs_split, diagonals_from_matrix};
pub use params::CkksParams;
pub use plaintext::Plaintext;
