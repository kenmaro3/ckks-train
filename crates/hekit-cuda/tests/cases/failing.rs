//! A test backend that fails on demand (Task 47 fix round 1, review I1 and M4): the CPU
//! `Evaluator` behind `HeOps`, where every primitive operation first takes a ticket and the
//! `fail_at`-th one (counted across forks) returns `Device`, as a device allocation failure
//! part-way through a composite operation would; `fail_upload` makes every upload fail.
//! Counts are those of the wrapped CPU evaluator.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use hekit_ckks::{
    Ciphertext, CkksParams, Diagonals, EvalConfig, Evaluator, HeOps, OpCounter, Plaintext,
};
use hekit_core::{HekitError, Result};

/// The failing backend (see the module docs).
pub struct Failing {
    inner: Evaluator,
    calls: Arc<AtomicUsize>,
    fail_at: Arc<AtomicUsize>,
    fail_upload: Arc<AtomicBool>,
}

impl Failing {
    /// Wraps `inner`; nothing fails until [`Failing::fail_at`] / [`Failing::fail_uploads`].
    pub fn new(inner: Evaluator) -> Failing {
        Failing {
            inner,
            calls: Arc::new(AtomicUsize::new(0)),
            fail_at: Arc::new(AtomicUsize::new(usize::MAX)),
            fail_upload: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Restarts the ticket count and fails the `k`-th primitive from now (`None`: never).
    pub fn fail_at(&self, k: Option<usize>) {
        self.calls.store(0, Ordering::SeqCst);
        self.fail_at
            .store(k.unwrap_or(usize::MAX), Ordering::SeqCst);
    }

    /// Primitive operations since the last [`Failing::fail_at`].
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Makes every upload fail (`Device`).
    pub fn fail_uploads(&self, on: bool) {
        self.fail_upload.store(on, Ordering::SeqCst);
    }

    fn tick(&self) -> Result<()> {
        let k = self.calls.fetch_add(1, Ordering::SeqCst);
        if k == self.fail_at.load(Ordering::SeqCst) {
            Err(HekitError::Device(format!(
                "injected failure at primitive {k}"
            )))
        } else {
            Ok(())
        }
    }

    fn up(&self) -> Result<()> {
        if self.fail_upload.load(Ordering::SeqCst) {
            Err(HekitError::Device("injected upload failure".into()))
        } else {
            Ok(())
        }
    }
}

impl HeOps for Failing {
    type Ct = Ciphertext;
    type Pt = Plaintext;

    fn params(&self) -> &Arc<CkksParams> {
        HeOps::params(&self.inner)
    }
    fn config(&self) -> &EvalConfig {
        HeOps::config(&self.inner)
    }
    fn counter(&self) -> OpCounter {
        HeOps::counter(&self.inner)
    }
    fn reset_counter(&self) {
        HeOps::reset_counter(&self.inner)
    }
    fn add_counts(&self, counts: &OpCounter) {
        self.inner.add_counts(counts)
    }
    fn fork(&self) -> Failing {
        Failing {
            inner: self.inner.fork(),
            calls: Arc::clone(&self.calls),
            fail_at: Arc::clone(&self.fail_at),
            fail_upload: Arc::clone(&self.fail_upload),
        }
    }
    fn has_relin_key(&self) -> bool {
        self.inner.has_relin_key()
    }
    fn has_galois_key(&self, step: i32) -> bool {
        self.inner.has_galois_key(step)
    }
    fn owns(&self, ct: &Ciphertext) -> bool {
        HeOps::owns(&self.inner, ct)
    }
    fn upload_ct(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.up()?;
        self.inner.upload_ct(ct)
    }
    fn download_ct(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.inner.download_ct(ct)
    }
    fn upload_pt(&self, pt: &Plaintext) -> Result<Plaintext> {
        self.up()?;
        self.inner.upload_pt(pt)
    }
    fn add(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::add(&self.inner, a, b)
    }
    fn sub(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::sub(&self.inner, a, b)
    }
    fn negate(&self, a: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::negate(&self.inner, a)
    }
    fn add_plain(&self, a: &Ciphertext, p: &Plaintext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::add_plain(&self.inner, a, p)
    }
    fn mul_plain(&self, a: &Ciphertext, p: &Plaintext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::mul_plain(&self.inner, a, p)
    }
    fn mul_scalar(&self, ct: &Ciphertext, value: f64, target: f64) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::mul_scalar(&self.inner, ct, value, target)
    }
    fn mul_plain_to_scale(&self, a: &Ciphertext, p: &Plaintext, t: f64) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::mul_plain_to_scale(&self.inner, a, p, t)
    }
    fn mul_no_relin(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::mul_no_relin(&self.inner, a, b)
    }
    fn relinearize(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::relinearize(&self.inner, ct)
    }
    fn mul(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::mul(&self.inner, a, b)
    }
    fn rescale(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::rescale(&self.inner, ct)
    }
    fn mod_switch_to_level(&self, ct: &Ciphertext, level: usize) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::mod_switch_to_level(&self.inner, ct, level)
    }
    fn relabel(&self, ct: Ciphertext, scale: f64) -> Result<Ciphertext> {
        self.inner.relabel(ct, scale)
    }
    fn rotate(&self, ct: &Ciphertext, steps: i32) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::rotate(&self.inner, ct, steps)
    }
    fn conjugate(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::conjugate(&self.inner, ct)
    }
    fn hoisted_rotate(&self, ct: &Ciphertext, steps: &[i32]) -> Result<Vec<Ciphertext>> {
        self.tick()?;
        HeOps::hoisted_rotate(&self.inner, ct, steps)
    }
    fn matvec_plain(&self, diag: &Diagonals<Plaintext>, x: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::matvec_plain(&self.inner, diag, x)
    }
    fn matvec_enc(&self, diag: &Diagonals<Ciphertext>, x: &Ciphertext) -> Result<Ciphertext> {
        self.tick()?;
        HeOps::matvec_enc(&self.inner, diag, x)
    }
}
