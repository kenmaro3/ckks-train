//! ckks-train-nn: encrypted neural-network training on ckks-train-ckks (temporary crate name).
#![deny(missing_docs)]

pub mod activation;
pub mod block;
pub mod compact;
pub mod dense;
pub mod diagonal_plan;
pub mod layout;
pub mod layout_reboot;
pub mod loss;
pub mod optim;
pub mod packed_training;
pub mod refresh;
pub mod sim;
pub mod trainer;
pub mod trainer_ni;
