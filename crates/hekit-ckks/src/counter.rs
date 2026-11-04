//! Counts of the expensive homomorphic operations performed by an
//! [`Evaluator`](crate::Evaluator).
//!
//! The evaluator keeps one [`OpCounter`] behind a `Mutex` (so `Evaluator` stays `Sync`) and
//! increments it after each successful operation; [`Evaluator::counter`](crate::Evaluator::counter)
//! returns a snapshot. Measure an operation by the difference of two snapshots.
//!
//! What is counted (Task 19; Task 21 completes the counter):
//!
//! * `rotations`: every rotation that performs a key switch, i.e. each step of
//!   [`Evaluator::rotate`](crate::Evaluator::rotate) and of
//!   [`Evaluator::hoisted_rotate`](crate::Evaluator::hoisted_rotate) that is not a multiple of
//!   the slot count (those are copies and count 0), and each
//!   [`Evaluator::conjugate`](crate::Evaluator::conjugate) (also a Galois key switch).
//! * `relins`: each [`Evaluator::relinearize`](crate::Evaluator::relinearize) (so each `mul`).
//! * `ct_ct_tensor`: each [`Evaluator::mul_no_relin`](crate::Evaluator::mul_no_relin) (so
//!   each `mul`).
//! * `ct_pt_mults`: each [`Evaluator::mul_plain`](crate::Evaluator::mul_plain).
//! * `rescales`: each [`Evaluator::rescale`](crate::Evaluator::rescale).
//! * `bootstraps`: none yet (no bootstrapping in M1).
//!
//! Failed operations (any `Err`) count nothing.

/// Operation counts; see the [module docs](self) for what increments each field.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct OpCounter {
    /// Key-switching rotations (including conjugations; trivial rotations excluded).
    pub rotations: u64,
    /// Relinearizations.
    pub relins: u64,
    /// Ciphertext-ciphertext tensor products (`mul_no_relin`, also inside `mul`).
    pub ct_ct_tensor: u64,
    /// Ciphertext-plaintext multiplications.
    pub ct_pt_mults: u64,
    /// Rescales.
    pub rescales: u64,
    /// Bootstraps.
    pub bootstraps: u64,
}
