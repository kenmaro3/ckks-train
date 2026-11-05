//! The backend-neutral evaluator interface [`HeOps`] (Task 43) and the RQ3 ablation flags
//! [`EvalConfig`].
//!
//! [`HeOps`] is the public surface of [`Evaluator`] that the training code (`hekit-nn`) uses,
//! with the ciphertext and plaintext types as associated types, so a device evaluator (the
//! CUDA backend, Task 47) can replace the CPU one. For the CPU [`Evaluator`],
//! `Ct = `[`Ciphertext`], `Pt = `[`Plaintext`], every method delegates to the inherent method
//! of the same name (no extra work, the same counts), and upload / download are clones (the
//! `_owned` variants are moves).
//!
//! # Contract for implementations
//!
//! * Every operation has the semantics, the checks, the errors and the counts of the
//!   [`Evaluator`] method of the same name ([`crate::evaluator`], [`crate::counter`]), and
//!   the same exact-scale bookkeeping: scales are `f64` labels computed by the same formulas,
//!   so a device backend produces bit-identical scales and levels.
//! * `Ct::clone` and `Pt::clone` must not allocate device memory and must not panic: the
//!   training code clones ciphertexts freely (e.g. a copy for a trivial rotation). A backend
//!   whose deep copy could fail (out of device memory) uses shared, reference-counted handles
//!   so that `clone` is a reference-count increment.
//! * [`HeOps::upload_ct`] / [`HeOps::download_ct`] move values between the host
//!   representation ([`Ciphertext`]) and the backend's; host-side code (encoding, encryption,
//!   decryption, the client refresh, the CPU bootstrapper) stays on [`Ciphertext`] and
//!   [`Plaintext`].
//!
//! # The ablation flags ([`EvalConfig`]) and the code paths they change
//!
//! All three default to `true` (the optimised library). They change only costs, never the
//! value (up to CKKS noise; bit-identical for `hoist`). Every other operation ignores them.
//! The bootstrapper always runs its own evaluator with the default flags: bootstrapping
//! (its DFT included) is not part of the ablation, and
//! [`Bootstrapper::cost_per_boot`](crate::bootstrap::Bootstrapper::cost_per_boot) holds
//! under every configuration (tested).
//!
//! | flag | `true` (default) | `false` (naive) | code paths |
//! |---|---|---|---|
//! | `hoist` | [`Evaluator::hoisted_rotate`] shares one digit decomposition | one [`Evaluator::rotate`] per step (same results bit for bit, same counts) | every `hoisted_rotate` of the training step: the baby steps of `matvec_plain` / `matvec_enc` (`Dense::forward`, `Dense::backward_input`) and the input and `delta` rotations of `Dense::weight_grad_scaled` / `Dense::weight_grad_reduced` |
//! | `bsgs` | the giant rotation is shared by the `baby` products of its block | the giant rotation is not shared (definition below) | `matvec_plain`, `matvec_enc` (so `Dense::forward`, `Dense::backward_input`), `matvec_plain_sum[_sparse]` |
//! | `lazy_relin` | degree-2 products are summed and relinearized once | every product is relinearized at once ([`Evaluator::mul`]) | `matvec_enc`; [`HeOps::mul_lazy`] / [`HeOps::relin_lazy`]: the micro-batch sum of `Dense::weight_grad_reduced` (`m` relins become `k m`); `RebootLayout::outer_sum` with `GradRelin::Lazy` (acts as `PerSample`) |
//!
//! ## What the RQ3 table must state (Task 48 copies this)
//!
//! 1. **Definition of `bsgs = false`.** The giant rotation is not shared; the weights keep
//!    the same pre-rotated (BSGS) layout and the same Galois keys. Every product with a
//!    nonzero giant shift is finished (relinearized) and rotated on its own; the products of
//!    the zero-shift block are summed as with BSGS. Per matvec of dimension `d` (offset 0):
//!    `d - 1` rotations, of which only the `baby - 1` rotations of `x` can be hoisted; with
//!    `lazy_relin` `1 + d - baby` relins, without it `d`
//!    ([`EvalConfig::matvec_rotations`], [`EvalConfig::matvec_enc_relins`]). At `d = 16`
//!    (`baby = giant = 4`), as (hoistable rotations / individual rotations / relins):
//!
//!    | form | hoistable rot. | individual rot. | relins (lazy) | relins (not lazy) |
//!    |---|---|---|---|---|
//!    | default BSGS | 3 | 3 | 4 | 16 |
//!    | `bsgs = false` (this library) | 3 | 12 | 13 | 16 |
//!    | textbook diagonal method (not implemented) | 15 | 0 | 1 | 16 |
//!
//!    The textbook (Halevi-Shoup) diagonal method rotates `x` by all `d - 1` steps (all
//!    hoistable) and sums the products lazily; it needs un-pre-rotated diagonals, i.e.
//!    another weight layout and gradient form, so it is not implemented. The all-off
//!    configuration has the same counts in both forms (15 rotations, 16 relins at `d = 16`).
//! 2. **Bias.** Against the strongest naive baseline (the textbook method with hoisting and
//!    lazy relinearization), the cost of the `bsgs = false` rows overstates what BSGS gains:
//!    the measured BSGS gain is an **upper bound**. Conversely, the marginal gains of `hoist`
//!    and `lazy_relin` measured inside `bsgs = false` configurations are **lower bounds** (only
//!    `baby - 1` rotations can be hoisted and only the zero-shift block summed lazily: e.g.
//!    `lazy_relin` saves 3 relins at `d = 16` instead of 15).
//! 3. **`hoist` changes no [`OpCounter`] value.** The counter does not distinguish hoisted from
//!    individual rotations, so the effect of `hoist` shows only in wall-clock time, never in
//!    key-switch counts.
//! 4. **`bsgs` does not touch the gradient's outer product** (`Dense::weight_grad_scaled`,
//!    `Dense::weight_grad_reduced`): its `(baby - 1) + (giant - 1)` input rotations are what
//!    the pre-rotated weight layout requires, not a BSGS choice; only `hoist` changes how they
//!    run. The interactive gradient multiplies each diagonal with one [`Evaluator::mul`]
//!    (nothing to sum lazily), so `lazy_relin` does not change it either.
//!
//! # Counters across evaluators ([`HeOps::add_counts`])
//!
//! Work done for a backend on another evaluator (a refresh or bootstrap on the CPU for a
//! device backend: `hekit_nn::refresh::HostRefresher`, [`HostBootstrapper`]) is merged into
//! the backend's counter with [`HeOps::add_counts`], so its epoch counts and
//! [`OpCounter::key_switches`] include it. Every implementation must honour `add_counts`
//! (add field by field, like [`Evaluator`]); otherwise host-side work silently drops out of
//! the counts.

use std::sync::Arc;

use hekit_core::error::Result;

use crate::bootstrap::Bootstrapper;
use crate::ciphertext::Ciphertext;
use crate::counter::OpCounter;
use crate::evaluator::{Evaluator, check_plain_scale, exact_product_scale};
use crate::linalg::{DiagonalEntry, Diagonals, bsgs_split};
use crate::params::CkksParams;
use crate::plaintext::Plaintext;

/// The RQ3 ablation flags of an evaluator (see the [module docs](self) for the code paths
/// each one changes). [`EvalConfig::default`] has all three on.
///
/// Caveats for reporting (module docs, "What the RQ3 table must state"):
/// * `bsgs = false` means the giant rotation is not shared, with the same pre-rotated layout
///   and Galois keys: per matvec `d - 1` rotations (`baby - 1` hoistable) and `1 + d - baby`
///   relins with `lazy_relin`, `d` without; at `d = 16`: 15 rotations (3 hoistable) and 13 or
///   16 relins, against BSGS's 6 and 4 or 16. Compared with the textbook diagonal method
///   (15 hoistable rotations, 1 relin), the measured BSGS gain is an upper bound and the
///   `hoist` / `lazy_relin` gains inside `bsgs = false` configurations are lower bounds.
/// * `hoist` changes no [`OpCounter`] value (wall-clock time only).
/// * `bsgs` does not touch the gradient's outer product.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvalConfig {
    /// Hoisted rotations: one shared digit decomposition per [`HeOps::hoisted_rotate`].
    pub hoist: bool,
    /// Baby-step giant-step matrix-vector products: one giant rotation per block.
    pub bsgs: bool,
    /// Lazy relinearization: sums of degree-2 products are relinearized once.
    pub lazy_relin: bool,
}

impl EvalConfig {
    /// All optimisations on (the default).
    pub const OPTIMIZED: EvalConfig = EvalConfig {
        hoist: true,
        bsgs: true,
        lazy_relin: true,
    };

    /// All optimisations off: individual rotations, `d` separately rotated diagonal
    /// products, one relinearization per multiplication.
    pub const NAIVE: EvalConfig = EvalConfig {
        hoist: false,
        bsgs: false,
        lazy_relin: false,
    };

    /// Key-switching rotations of [`Evaluator::matvec_plain`] / [`Evaluator::matvec_enc`] of
    /// dimension `d` (offset 0, no zero entries): `baby + giant - 2` with `bsgs`, `d - 1`
    /// without ([`crate::linalg::bsgs_split`]; `hoist` changes no count).
    ///
    /// # Panics
    /// If `d` is not a power of two.
    pub fn matvec_rotations(&self, d: usize) -> u64 {
        let (baby, giant) = bsgs_split(d);
        (if self.bsgs { baby + giant - 2 } else { d - 1 }) as u64
    }

    /// Relinearizations of [`Evaluator::matvec_enc`] of dimension `d` (offset 0): `giant`
    /// with `bsgs` and `lazy_relin`, `1 + d - baby` with `lazy_relin` only (the zero-shift
    /// block summed, every rotated product relinearized on its own), `d` without
    /// `lazy_relin`.
    ///
    /// # Panics
    /// If `d` is not a power of two.
    pub fn matvec_enc_relins(&self, d: usize) -> u64 {
        let (baby, giant) = bsgs_split(d);
        (match (self.bsgs, self.lazy_relin) {
            (_, false) => d,
            (true, true) => giant,
            (false, true) => 1 + d - baby,
        }) as u64
    }

    /// The 8 combinations of the three flags, in the order `(hoist, bsgs, lazy_relin)` =
    /// `(1,1,1), (1,1,0), (1,0,1), ..., (0,0,0)` (the RQ3 ablation grid).
    pub fn all() -> [EvalConfig; 8] {
        std::array::from_fn(|i| EvalConfig {
            hoist: i & 4 == 0,
            bsgs: i & 2 == 0,
            lazy_relin: i & 1 == 0,
        })
    }
}

impl Default for EvalConfig {
    fn default() -> EvalConfig {
        EvalConfig::OPTIMIZED
    }
}

/// The metadata the training code reads from a ciphertext of any backend: the level, scale
/// and slot count of [`DiagonalEntry`] and the number of components.
pub trait HeCiphertext: DiagonalEntry {
    /// The number of polynomial components (2, or 3 before a relinearization).
    fn size(&self) -> usize;
}

impl HeCiphertext for Ciphertext {
    fn size(&self) -> usize {
        self.len()
    }
}

/// The evaluator operations of a CKKS backend (see the [module docs](self) for the contract).
///
/// Every method has the meaning, the checks, the errors and the counts of the
/// [`Evaluator`] method of the same name; the links point there. The methods with a default
/// body are derived from the others and need no backend code.
pub trait HeOps: Sized {
    /// A ciphertext of this backend. `clone` must not allocate device memory or panic
    /// (shared handles; [module docs](self)).
    type Ct: Clone + Send + HeCiphertext;
    /// A plaintext of this backend. `clone` must not allocate device memory or panic.
    type Pt: Clone + Send + DiagonalEntry;

    /// The parameters ([`Evaluator::params`]).
    fn params(&self) -> &Arc<CkksParams>;

    /// The ablation flags ([`Evaluator::config`]).
    fn config(&self) -> &EvalConfig;

    /// A snapshot of the operation counts ([`Evaluator::counter`]).
    fn counter(&self) -> OpCounter;

    /// Zeroes the operation counts ([`Evaluator::reset_counter`]).
    fn reset_counter(&self);

    /// Adds `counts` to this evaluator's counter, field by field: merges the operations done
    /// for this backend on another evaluator (a CPU refresh or bootstrap of a device
    /// ciphertext, see the [module docs](self)). Required: every implementation must honour
    /// it, or that work drops out of the counts.
    fn add_counts(&self, counts: &OpCounter);

    /// An evaluator with the same parameters, keys (shared, not copied) and flags and an own
    /// counter starting at zero: for probes whose operations must not show on this
    /// evaluator's counter (the CPU [`Evaluator`]'s `clone` with a reset counter).
    fn fork(&self) -> Self;

    /// Whether a relinearization key is present (`relin_key().is_some()`).
    fn has_relin_key(&self) -> bool;

    /// Whether a Galois key for a rotation by `step` is present
    /// (`galois_keys().get(step).is_some()`).
    fn has_galois_key(&self, step: i32) -> bool;

    /// Whether `ct` belongs to this evaluator (for the CPU: every polynomial over the data
    /// basis of [`HeOps::params`], the same `Arc`). Lets callers report foreign operands
    /// before any work.
    fn owns(&self, ct: &Self::Ct) -> bool;

    /// The backend copy of a host ciphertext (a clone for the CPU).
    fn upload_ct(&self, ct: &Ciphertext) -> Result<Self::Ct>;

    /// [`HeOps::upload_ct`] taking the host ciphertext by value (a move for the CPU).
    fn upload_ct_owned(&self, ct: Ciphertext) -> Result<Self::Ct> {
        self.upload_ct(&ct)
    }

    /// The host copy of a backend ciphertext (a clone for the CPU).
    fn download_ct(&self, ct: &Self::Ct) -> Result<Ciphertext>;

    /// The backend copy of a host plaintext (a clone for the CPU).
    fn upload_pt(&self, pt: &Plaintext) -> Result<Self::Pt>;

    /// [`HeOps::upload_pt`] taking the host plaintext by value (a move for the CPU).
    fn upload_pt_owned(&self, pt: Plaintext) -> Result<Self::Pt> {
        self.upload_pt(&pt)
    }

    /// `a + b` ([`Evaluator::add`]).
    fn add(&self, a: &Self::Ct, b: &Self::Ct) -> Result<Self::Ct>;

    /// `a - b` ([`Evaluator::sub`]).
    fn sub(&self, a: &Self::Ct, b: &Self::Ct) -> Result<Self::Ct>;

    /// `-a` ([`Evaluator::negate`]).
    fn negate(&self, a: &Self::Ct) -> Result<Self::Ct>;

    /// `a + p` ([`Evaluator::add_plain`]).
    fn add_plain(&self, a: &Self::Ct, p: &Self::Pt) -> Result<Self::Ct>;

    /// `a * p` ([`Evaluator::mul_plain`]).
    fn mul_plain(&self, a: &Self::Ct, p: &Self::Pt) -> Result<Self::Ct>;

    /// `ct * value`, labelled to rescale to exactly `target_scale` ([`Evaluator::mul_scalar`]).
    fn mul_scalar(&self, ct: &Self::Ct, value: f64, target_scale: f64) -> Result<Self::Ct>;

    /// `a * p`, labelled to rescale to exactly `target_scale`
    /// ([`Evaluator::mul_plain_to_scale`]).
    fn mul_plain_to_scale(&self, a: &Self::Ct, p: &Self::Pt, target_scale: f64)
    -> Result<Self::Ct>;

    /// The tensor product ([`Evaluator::mul_no_relin`]).
    fn mul_no_relin(&self, a: &Self::Ct, b: &Self::Ct) -> Result<Self::Ct>;

    /// Relinearization of a 3-component ciphertext ([`Evaluator::relinearize`]).
    fn relinearize(&self, ct: &Self::Ct) -> Result<Self::Ct>;

    /// `relinearize(mul_no_relin(a, b))` ([`Evaluator::mul`]).
    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> Result<Self::Ct>;

    /// Division by the last prime ([`Evaluator::rescale`]).
    fn rescale(&self, ct: &Self::Ct) -> Result<Self::Ct>;

    /// Drops primes down to `level` ([`Evaluator::mod_switch_to_level`]).
    fn mod_switch_to_level(&self, ct: &Self::Ct, level: usize) -> Result<Self::Ct>;

    /// The same ciphertext labelled with `scale` (no arithmetic: the message is multiplied by
    /// `scale / ct.scale()`), as `Ciphertext::new(ct.into_polys(), scale)` does on the CPU.
    /// Used to relabel a ciphertext to `Delta` around a bootstrap.
    ///
    /// Errors: `InvalidParams` for a scale that is not positive and finite.
    fn relabel(&self, ct: Self::Ct, scale: f64) -> Result<Self::Ct>;

    /// Left rotation of the slots by `steps` ([`Evaluator::rotate`]).
    fn rotate(&self, ct: &Self::Ct, steps: i32) -> Result<Self::Ct>;

    /// Complex conjugation of the slots ([`Evaluator::conjugate`]).
    fn conjugate(&self, ct: &Self::Ct) -> Result<Self::Ct>;

    /// Several rotations of one ciphertext, hoisted when [`EvalConfig::hoist`] is on
    /// ([`Evaluator::hoisted_rotate`]).
    fn hoisted_rotate(&self, ct: &Self::Ct, steps: &[i32]) -> Result<Vec<Self::Ct>>;

    /// `W x` for plaintext diagonals, BSGS when [`EvalConfig::bsgs`] is on
    /// ([`Evaluator::matvec_plain`]).
    fn matvec_plain(&self, diag: &Diagonals<Self::Pt>, x: &Self::Ct) -> Result<Self::Ct>;

    /// `W x` for encrypted diagonals, honouring all three flags ([`Evaluator::matvec_enc`]).
    fn matvec_enc(&self, diag: &Diagonals<Self::Ct>, x: &Self::Ct) -> Result<Self::Ct>;

    /// The pre-rescale label landing exactly on `target_scale`
    /// ([`Evaluator::exact_product_scale`]); a function of the parameters only.
    fn exact_product_scale(&self, level: usize, target_scale: f64) -> Result<f64> {
        exact_product_scale(self.params(), level, target_scale)
    }

    /// The plaintext scale for [`HeOps::mul_plain_to_scale`] of `ct` to `target_scale`
    /// ([`Evaluator::target_plain_scale`]); a function of the parameters and of `ct`'s level
    /// and scale only.
    fn target_plain_scale(&self, ct: &Self::Ct, target_scale: f64) -> Result<f64> {
        let sp = self.exact_product_scale(ct.level(), target_scale)? / ct.scale();
        check_plain_scale(sp)?;
        Ok(sp)
    }

    /// A product destined for a sum that [`HeOps::relin_lazy`] finishes: the degree-2
    /// [`HeOps::mul_no_relin`] when [`EvalConfig::lazy_relin`] is on, else the relinearized
    /// [`HeOps::mul`] (the naive form: one relinearization per product).
    fn mul_lazy(&self, a: &Self::Ct, b: &Self::Ct) -> Result<Self::Ct> {
        if self.config().lazy_relin {
            self.mul_no_relin(a, b)
        } else {
            self.mul(a, b)
        }
    }

    /// The end of a sum of [`HeOps::mul_lazy`] products: one [`HeOps::relinearize`] when
    /// [`EvalConfig::lazy_relin`] is on, else `sum` as it is (already relinearized).
    fn relin_lazy(&self, sum: Self::Ct) -> Result<Self::Ct> {
        if self.config().lazy_relin {
            self.relinearize(&sum)
        } else {
            Ok(sum)
        }
    }
}

impl HeOps for Evaluator {
    type Ct = Ciphertext;
    type Pt = Plaintext;

    #[inline]
    fn params(&self) -> &Arc<CkksParams> {
        Evaluator::params(self)
    }
    #[inline]
    fn config(&self) -> &EvalConfig {
        Evaluator::config(self)
    }
    #[inline]
    fn counter(&self) -> OpCounter {
        Evaluator::counter(self)
    }
    #[inline]
    fn reset_counter(&self) {
        Evaluator::reset_counter(self)
    }
    #[inline]
    fn add_counts(&self, counts: &OpCounter) {
        self.count(|c| *c += counts);
    }
    fn fork(&self) -> Evaluator {
        let e = self.clone();
        e.reset_counter();
        e
    }
    #[inline]
    fn has_relin_key(&self) -> bool {
        self.relin_key().is_some()
    }
    #[inline]
    fn has_galois_key(&self, step: i32) -> bool {
        self.galois_keys().is_some_and(|g| g.get(step).is_some())
    }
    #[inline]
    fn owns(&self, ct: &Ciphertext) -> bool {
        ct.polys()
            .iter()
            .all(|p| Arc::ptr_eq(p.basis(), Evaluator::params(self).basis_q()))
    }
    #[inline]
    fn upload_ct(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        Ok(ct.clone())
    }
    #[inline]
    fn upload_ct_owned(&self, ct: Ciphertext) -> Result<Ciphertext> {
        Ok(ct)
    }
    #[inline]
    fn download_ct(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        Ok(ct.clone())
    }
    #[inline]
    fn upload_pt(&self, pt: &Plaintext) -> Result<Plaintext> {
        Ok(pt.clone())
    }
    #[inline]
    fn upload_pt_owned(&self, pt: Plaintext) -> Result<Plaintext> {
        Ok(pt)
    }
    #[inline]
    fn add(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::add(self, a, b)
    }
    #[inline]
    fn sub(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::sub(self, a, b)
    }
    #[inline]
    fn negate(&self, a: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::negate(self, a)
    }
    #[inline]
    fn add_plain(&self, a: &Ciphertext, p: &Plaintext) -> Result<Ciphertext> {
        Evaluator::add_plain(self, a, p)
    }
    #[inline]
    fn mul_plain(&self, a: &Ciphertext, p: &Plaintext) -> Result<Ciphertext> {
        Evaluator::mul_plain(self, a, p)
    }
    #[inline]
    fn mul_scalar(&self, ct: &Ciphertext, value: f64, target_scale: f64) -> Result<Ciphertext> {
        Evaluator::mul_scalar(self, ct, value, target_scale)
    }
    #[inline]
    fn mul_plain_to_scale(
        &self,
        a: &Ciphertext,
        p: &Plaintext,
        target_scale: f64,
    ) -> Result<Ciphertext> {
        Evaluator::mul_plain_to_scale(self, a, p, target_scale)
    }
    #[inline]
    fn mul_no_relin(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::mul_no_relin(self, a, b)
    }
    #[inline]
    fn relinearize(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::relinearize(self, ct)
    }
    #[inline]
    fn mul(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::mul(self, a, b)
    }
    #[inline]
    fn rescale(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::rescale(self, ct)
    }
    #[inline]
    fn mod_switch_to_level(&self, ct: &Ciphertext, level: usize) -> Result<Ciphertext> {
        Evaluator::mod_switch_to_level(self, ct, level)
    }
    #[inline]
    fn relabel(&self, ct: Ciphertext, scale: f64) -> Result<Ciphertext> {
        Ciphertext::new(ct.into_polys(), scale)
    }
    #[inline]
    fn rotate(&self, ct: &Ciphertext, steps: i32) -> Result<Ciphertext> {
        Evaluator::rotate(self, ct, steps)
    }
    #[inline]
    fn conjugate(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::conjugate(self, ct)
    }
    #[inline]
    fn hoisted_rotate(&self, ct: &Ciphertext, steps: &[i32]) -> Result<Vec<Ciphertext>> {
        Evaluator::hoisted_rotate(self, ct, steps)
    }
    #[inline]
    fn matvec_plain(&self, diag: &Diagonals<Plaintext>, x: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::matvec_plain(self, diag, x)
    }
    #[inline]
    fn matvec_enc(&self, diag: &Diagonals<Ciphertext>, x: &Ciphertext) -> Result<Ciphertext> {
        Evaluator::matvec_enc(self, diag, x)
    }
    #[inline]
    fn exact_product_scale(&self, level: usize, target_scale: f64) -> Result<f64> {
        Evaluator::exact_product_scale(self, level, target_scale)
    }
    #[inline]
    fn target_plain_scale(&self, ct: &Ciphertext, target_scale: f64) -> Result<f64> {
        Evaluator::target_plain_scale(self, ct, target_scale)
    }
}

/// A bootstrapper for the ciphertexts of the backend `E`: what the non-interactive trainer
/// (`hekit-nn`'s `NonInteractiveTrainer`) needs to refresh its state. The CPU
/// [`Bootstrapper`] implements it for `E = `[`Evaluator`] (it stays CPU-only); a device
/// bootstrapper (Task 49), or an adapter that downloads, bootstraps on the CPU and uploads
/// ("level ops on the GPU, bootstrapping on the CPU"), implements it for its evaluator.
pub trait BootstrapOps<E: HeOps> {
    /// Bootstraps `ct` (level 0, scale exactly `Delta`) to [`BootstrapOps::output_level`] at
    /// exactly `Delta`, adding its operations and one `bootstraps` to `ev`'s counter
    /// ([`Bootstrapper::bootstrap`]).
    fn bootstrap(&self, ev: &E, ct: &E::Ct) -> Result<E::Ct>;

    /// The level of every bootstrap output ([`Bootstrapper::output_level`]).
    fn output_level(&self) -> usize;
}

impl BootstrapOps<Evaluator> for Bootstrapper {
    #[inline]
    fn bootstrap(&self, ev: &Evaluator, ct: &Ciphertext) -> Result<Ciphertext> {
        Bootstrapper::bootstrap(self, ev, ct)
    }
    #[inline]
    fn output_level(&self) -> usize {
        Bootstrapper::output_level(self)
    }
}

/// The CPU [`Bootstrapper`] serving the ciphertexts of any backend `E` ("level operations on
/// the device, bootstrapping on the CPU", research plan §5): every bootstrap downloads the
/// ciphertext, bootstraps it on the host evaluator `host` and uploads the result, and merges
/// the host evaluator's counts of that bootstrap into `E`'s counter ([`HeOps::add_counts`]).
///
/// `host` must hold the relinearization key of the bootstrapper's parameters, must not be the
/// evaluator `E` itself (its counts would be added twice; for a CPU `E` pass a
/// [`HeOps::fork`]), and must not be used by another thread during a bootstrap (the merged
/// counts are the difference of its counter around the bootstrap).
#[derive(Debug)]
pub struct HostBootstrapper<'h> {
    bs: &'h Bootstrapper,
    host: &'h Evaluator,
}

impl<'h> HostBootstrapper<'h> {
    /// Bootstraps with `bs` on the host evaluator `host` (see the type docs).
    pub fn new(bs: &'h Bootstrapper, host: &'h Evaluator) -> HostBootstrapper<'h> {
        HostBootstrapper { bs, host }
    }
}

impl<E: HeOps> BootstrapOps<E> for HostBootstrapper<'_> {
    /// `upload(bootstrap(download(ct)))`, with the bootstrap's counts merged into `ev`.
    ///
    /// Errors: those of the transfers and of [`Bootstrapper::bootstrap`].
    fn bootstrap(&self, ev: &E, ct: &E::Ct) -> Result<E::Ct> {
        let host_ct = ev.download_ct(ct)?;
        let before = self.host.counter();
        let out = self.bs.bootstrap(self.host, &host_ct);
        ev.add_counts(&(&self.host.counter() - &before));
        ev.upload_ct_owned(out?)
    }

    fn output_level(&self) -> usize {
        self.bs.output_level()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::keys::KeyGenerator;
    use hekit_core::sampler::Rng;

    /// The limbs of every component (bit-exact comparison).
    fn bits(c: &Ciphertext) -> Vec<Vec<Vec<u64>>> {
        c.polys().iter().map(|p| p.limbs().to_vec()).collect()
    }

    #[test]
    fn default_is_all_on_and_grid_is_complete() {
        assert_eq!(EvalConfig::default(), EvalConfig::OPTIMIZED);
        let all = EvalConfig::all();
        assert_eq!(all[0], EvalConfig::OPTIMIZED);
        assert_eq!(all[7], EvalConfig::NAIVE);
        let distinct: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(distinct.len(), 8);
        // The closed forms (checked against measured counts in linalg's ablation test).
        let n = EvalConfig::NAIVE;
        assert_eq!((n.matvec_rotations(16), n.matvec_enc_relins(16)), (15, 16));
        let d = EvalConfig::default();
        assert_eq!((d.matvec_rotations(16), d.matvec_enc_relins(16)), (6, 4));
        assert_eq!((d.matvec_rotations(32), d.matvec_enc_relins(32)), (10, 4));
        for c in EvalConfig::all() {
            assert_eq!((c.matvec_rotations(1), c.matvec_enc_relins(1)), (0, 1));
        }
    }

    /// The CPU impl is a delegation: same values, same counts, upload/download are copies,
    /// `fork` shares the keys with a zero counter, `mul_lazy`/`relin_lazy` follow the flag.
    #[test]
    fn evaluator_impl_delegates() {
        let params = Arc::new(CkksParams::new(12, &[60, 40, 40, 40], &[60], 40, 4).unwrap());
        let mut kg = KeyGenerator::new(Arc::clone(&params), Rng::from_seed([5; 32])).unwrap();
        let pk = kg.public_key();
        let rk = kg.relin_key();
        let gk = kg.galois_keys(&[1]);
        let ev = Evaluator::new(Arc::clone(&params), Some(rk), Some(gk));
        let enc = Encoder::new(Arc::clone(&params));
        let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
        let encryptor = Encryptor::new(Arc::clone(&params), pk);
        let mut r = Rng::from_seed([6; 32]);
        let top = params.max_level();
        let pt = enc.encode(&[0.5, -0.25], params.scale(), top).unwrap();
        let ct = encryptor.encrypt(&pt, &mut r);

        fn generic<E: HeOps>(ev: &E, a: &E::Ct) -> E::Ct {
            let s = ev.mul_lazy(a, a).unwrap();
            let s = ev.relin_lazy(s).unwrap();
            ev.rescale(&ev.rotate(&s, 1).unwrap()).unwrap()
        }
        let g = generic(&ev, &ct);
        let counts = ev.counter();
        ev.reset_counter();
        let c = Evaluator::rescale(
            &ev,
            &Evaluator::rotate(
                &ev,
                &Evaluator::relinearize(&ev, &Evaluator::mul_no_relin(&ev, &ct, &ct).unwrap())
                    .unwrap(),
                1,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(ev.counter(), counts);
        assert_eq!(bits(&g), bits(&c));
        assert_eq!((g.level(), g.scale()), (c.level(), c.scale()));

        let up = ev.upload_ct(&ct).unwrap();
        assert_eq!(bits(&up), bits(&ct));
        assert_eq!(bits(&ev.download_ct(&up).unwrap()), bits(&ct));
        assert!(HeOps::owns(&ev, &up));
        assert_eq!(ev.upload_pt(&pt).unwrap().poly().limbs(), pt.poly().limbs());
        assert!(ev.has_relin_key() && ev.has_galois_key(1) && !ev.has_galois_key(2));
        assert_eq!(HeCiphertext::size(&ct), 2);

        // add_counts adds field by field.
        let g = ev.fork();
        g.add_counts(&counts);
        g.add_counts(&counts);
        let mut twice = counts.clone();
        twice += &counts;
        assert_eq!(g.counter(), twice);

        let f = ev.fork();
        assert_eq!(HeOps::counter(&f), OpCounter::default());
        f.rotate(&ct, 1).unwrap();
        assert_eq!(f.counter().rotations, 1);
        assert_eq!(ev.counter(), counts);

        // lazy_relin off: mul_lazy relinearizes at once, relin_lazy is the identity.
        let naive = Evaluator::with_config(
            Arc::clone(&params),
            Some(kg.relin_key()),
            None,
            EvalConfig::NAIVE,
        );
        assert_eq!(*HeOps::config(&naive), EvalConfig::NAIVE);
        let p = naive.mul_lazy(&ct, &ct).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(naive.counter().relins, 1);
        let q = naive.relin_lazy(p.clone()).unwrap();
        assert_eq!(bits(&q), bits(&p));
        assert_eq!(naive.counter().relins, 1);
        let v = enc.decode(&dec.decrypt(&naive.rescale(&q).unwrap()));
        assert!((v[0] - 0.25).abs() < 1e-6 && (v[1] - 0.0625).abs() < 1e-6);

        // target_plain_scale / exact_product_scale defaults agree with the inherent ones.
        assert_eq!(
            HeOps::target_plain_scale(&ev, &ct, params.scale()).unwrap(),
            Evaluator::target_plain_scale(&ev, &ct, params.scale()).unwrap()
        );
    }
}
