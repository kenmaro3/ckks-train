//! Counts of the expensive homomorphic operations performed by an
//! [`Evaluator`](crate::Evaluator).
//!
//! The evaluator keeps one [`OpCounter`] behind a `Mutex` (so `Evaluator` stays `Sync`) and
//! increments it after each successful operation; [`Evaluator::counter`](crate::Evaluator::counter)
//! returns a snapshot. Measure an operation by the difference of two snapshots.
//!
//! What is counted:
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
//! Failed operations (any `Err`) count nothing. Composite operations count their parts:
//! `mul` is one `ct_ct_tensor` plus one `relin`; `matvec_plain` adds `d` `ct_pt_mults`;
//! `matvec_enc` adds `d` `ct_ct_tensor` and `giant` `relins`. A rescale that is part of the
//! caller's pipeline (the matvec results are returned unrescaled) is counted only when the
//! caller calls [`Evaluator::rescale`](crate::Evaluator::rescale).
//!
//! [`OpCounter::key_switches`] sums the key-switching operations (`rotations + relins +
//! bootstraps * BOOT_KS_EQUIV`); [`Evaluator::reset_counter`](crate::Evaluator::reset_counter)
//! zeroes the counts.

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

/// Key switches charged per bootstrap in [`OpCounter::key_switches`]. `0` until bootstrapping
/// exists (M3); at that point it is set to the number of key switches one bootstrap performs.
pub const BOOT_KS_EQUIV: u64 = 0;

impl OpCounter {
    /// Total key switches, the dominant cost of CKKS: `rotations + relins + bootstraps *
    /// `[`BOOT_KS_EQUIV`]. Conjugations are included in `rotations`.
    pub fn key_switches(&self) -> u64 {
        self.rotations + self.relins + self.bootstraps * BOOT_KS_EQUIV
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hekit_core::sampler::Rng;
    use ndarray::Array2;

    use super::*;
    use crate::encoder::Encoder;
    use crate::encrypt::Encryptor;
    use crate::evaluator::Evaluator;
    use crate::keys::KeyGenerator;
    use crate::linalg::{Diagonals, bsgs_rotation_steps, bsgs_split, diagonals_from_matrix};
    use crate::params::CkksParams;
    use crate::{Ciphertext, Plaintext};

    struct Ctx {
        enc: Encoder,
        encryptor: Encryptor,
        eval: Evaluator,
        rng: Rng,
        slots: usize,
        scale: f64,
        top: usize,
    }

    /// A small context with relin key and Galois keys for steps 1..=3 and BSGS d = 16.
    fn ctx() -> Ctx {
        let params = Arc::new(CkksParams::new(12, &[60, 40, 40, 40], &[60], 40, 4).unwrap());
        let mut kg = KeyGenerator::new(Arc::clone(&params), Rng::from_seed([5; 32])).unwrap();
        let mut steps = bsgs_rotation_steps(16, 1);
        steps.extend([1, 2, 3]);
        steps.sort_unstable();
        steps.dedup();
        let pk = kg.public_key();
        let rk = kg.relin_key();
        let gks = kg.galois_keys(&steps);
        Ctx {
            enc: Encoder::new(Arc::clone(&params)),
            encryptor: Encryptor::new(Arc::clone(&params), pk),
            eval: Evaluator::new(Arc::clone(&params), Some(rk), Some(gks)),
            rng: Rng::from_seed([6; 32]),
            slots: params.slots(),
            scale: params.scale(),
            top: params.max_level(),
        }
    }

    impl Ctx {
        fn ct(&mut self, v: f64) -> Ciphertext {
            let pt = self.enc.encode(&vec![v; self.slots], self.scale, self.top);
            self.encryptor.encrypt(&pt.unwrap(), &mut self.rng)
        }
    }

    #[test]
    fn key_switches_formula() {
        let c = OpCounter {
            rotations: 5,
            relins: 3,
            bootstraps: 2,
            ..Default::default()
        };
        assert_eq!(c.key_switches(), 8 + 2 * BOOT_KS_EQUIV);
        assert_eq!(BOOT_KS_EQUIV, 0, "0 until bootstrapping lands in M3");
        assert_eq!(OpCounter::default().key_switches(), 0);
    }

    #[test]
    fn mul_counts_one_relin_and_one_tensor() {
        let mut c = ctx();
        let (a, b) = (c.ct(0.5), c.ct(0.25));
        assert_eq!(c.eval.counter(), OpCounter::default());
        c.eval.mul(&a, &b).unwrap();
        let k = c.eval.counter();
        assert_eq!((k.relins, k.ct_ct_tensor), (1, 1));
        assert_eq!((k.rotations, k.ct_pt_mults, k.rescales), (0, 0, 0));
        assert_eq!(k.key_switches(), 1);
    }

    #[test]
    fn hoisted_rotate_counts_each_step() {
        let mut c = ctx();
        let a = c.ct(1.0);
        c.eval.hoisted_rotate(&a, &[1, 2, 3]).unwrap();
        let k = c.eval.counter();
        assert_eq!(k.rotations, 3);
        assert_eq!(k.key_switches(), 3);
    }

    #[test]
    fn matvec_enc_counts_giant_relins() {
        let mut c = ctx();
        let d = 16;
        let (baby, giant) = bsgs_split(d);
        let w = Array2::from_shape_fn((d, d), |(i, j)| (i + j) as f64 / 64.0);
        let diags = diagonals_from_matrix(&w, c.slots, 1);
        let x = c.ct(0.5);
        let cd = Diagonals::<Ciphertext>::encrypt(
            &c.enc,
            &c.encryptor,
            &diags,
            1,
            c.scale,
            c.top,
            &mut c.rng,
        )
        .unwrap();
        c.eval.reset_counter();
        c.eval.matvec_enc(&cd, &x).unwrap();
        let k = c.eval.counter();
        assert_eq!(giant, 4);
        assert_eq!(k.relins, giant as u64);
        assert_eq!(k.ct_ct_tensor, d as u64);
        assert_eq!(k.rotations, (baby + giant - 2) as u64);
        assert_eq!(k.key_switches(), (giant + baby + giant - 2) as u64);
    }

    #[test]
    fn plain_products_and_rescales_are_counted() {
        let mut c = ctx();
        let a = c.ct(0.5);
        let p: Plaintext = c.enc.encode(&vec![2.0; c.slots], c.scale, c.top).unwrap();
        let m = c.eval.mul_plain(&a, &p).unwrap();
        c.eval.rescale(&m).unwrap();
        let k = c.eval.counter();
        assert_eq!((k.ct_pt_mults, k.rescales), (1, 1));
        assert_eq!(k.key_switches(), 0);
        // A failed operation counts nothing (level 0 cannot rescale).
        let low = c.eval.mod_switch_to_level(&a, 0).unwrap();
        assert!(c.eval.rescale(&low).is_err());
        assert_eq!(c.eval.counter(), k);
    }

    #[test]
    fn reset_counter_restores_default() {
        let mut c = ctx();
        let (a, b) = (c.ct(0.5), c.ct(0.25));
        c.eval.mul(&a, &b).unwrap();
        c.eval.rotate(&a, 1).unwrap();
        assert_ne!(c.eval.counter(), OpCounter::default());
        c.eval.reset_counter();
        assert_eq!(c.eval.counter(), OpCounter::default());
    }
}
