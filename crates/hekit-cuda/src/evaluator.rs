//! The CUDA evaluator ([`CudaEvaluator`]): `hekit_ckks::HeOps` on device ciphertexts, with
//! the relinearization and Galois keys in device memory, the RQ3 ablation flags
//! ([`EvalConfig`]), the CPU's operation counts ([`OpCounter`]), BSGS matrix-vector products
//! (the shared orchestration of [`crate::matvec`]) and a multi-stream batch of encrypted
//! products ([`CudaEvaluator::matvec_enc_batch`]).
//!
//! # Contract (`hekit_ckks::ops`)
//!
//! Every operation has the semantics, the checks (in the CPU's order), the error kinds and
//! the counts of the `hekit_ckks::Evaluator` method of the same name, and returns the CPU's
//! result bit for bit: the residues (the device kernels and launch sequences are bit-exact,
//! Tasks 44 to 46), the level and the `f64` scale (the CPU's formulas). The matrix-vector
//! products run the CPU's operation sequence ([`crate::matvec`]), so they are bit-exact too.
//!
//! # Counting
//!
//! As the CPU ([`hekit_ckks::counter`]): `rotations` for every nontrivial rotation (each step
//! of a hoisted rotation) and every conjugation, `relins`, `ct_ct_tensor`, `ct_pt_mults`,
//! `rescales`; composite operations count their parts. A failed operation counts nothing.
//! `sse_switches` and `bootstraps` are never incremented here (no mod-raise or bootstrap runs
//! on the device); host-side bootstraps merge theirs with [`HeOps::add_counts`]
//! (`hekit_ckks::HostBootstrapper`). Counts are taken when the work is queued, not when the
//! device finishes it; a device error that surfaces later (at a download or a sync) does not
//! retract them.
//!
//! # Keys and device memory (deviations from the CPU, documented)
//!
//! * Galois keys are held by **Galois element** (`k` and `k + N/2` share a key).
//! * Keys can be uploaded for levels `<= key_level` only ([`CudaEvaluator::with_keys`]; an
//!   n16_train key is 288 MiB in full, 42 MiB at level 9). A key switch above `key_level` is
//!   then `InvalidParams`, reported before any work, where the CPU (which holds full keys)
//!   would succeed. With full keys (`key_level = L`, [`CudaEvaluator::new`]) there is no
//!   difference.
//! * `has_galois_key` / `has_relin_key` say whether a key is present, whatever its level.
//!
//! # Interface deviations from the Task 47 brief
//!
//! The brief's `galois: HashMap<i32, DeviceKsk>` is keyed by element (`u64`), as above; the
//! keys and ciphertexts borrow `&'d CudaDevice` as in Tasks 44 to 46 (cudarc 0.19's
//! `Arc<CudaContext>` lives inside the device); `params` is passed as the
//! `Arc<CkksParams>` the CPU code shares (it must be the device's parameter set).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use hekit_ckks::evaluator::{MIN_PLAIN_SCALE, SCALE_TOLERANCE};
use hekit_ckks::keys::{conjugation_elt, galois_elt};
use hekit_ckks::{
    Ciphertext, CkksParams, Diagonals, EvalConfig, GaloisKey, GaloisKeys, HeOps, OpCounter,
    Plaintext, RelinKey,
};
use hekit_core::{HekitError, Result};

use crate::CudaCiphertext;
use crate::arith::Dev;
use crate::basis::Backend;
use crate::device::CudaDevice;
use crate::keyswitch::DeviceKsk;
use crate::ks::{q_primes, rotation_elt};
use crate::matvec::{check_shape, matvec_enc_bsgs, matvec_enc_workspace_limbs, matvec_plain_bsgs};
use crate::poly::{DevicePlaintext, DevicePoly};

/// Bound on the integer constant of `mul_scalar`, `2^62` (as the CPU).
const MAX_SCALAR_INT: f64 = 4_611_686_018_427_387_904.0;

fn invalid(msg: impl Into<String>) -> HekitError {
    HekitError::InvalidParams(msg.into())
}

fn check_level(a: usize, b: usize) -> Result<()> {
    if a == b {
        Ok(())
    } else {
        Err(HekitError::LevelMismatch)
    }
}

fn check_scale(a: f64, b: f64) -> Result<()> {
    if (a - b).abs() < SCALE_TOLERANCE * a.min(b) {
        Ok(())
    } else {
        Err(HekitError::ScaleMismatch)
    }
}

fn check_plain_scale(sp: f64) -> Result<()> {
    if sp < MIN_PLAIN_SCALE {
        return Err(invalid(format!(
            "plaintext scale {sp:e} is below 2^20: the target scale is too small for this \
             ciphertext's scale and level (the constant would be rounded to a few bits)"
        )));
    }
    Ok(())
}

/// A CKKS evaluator on a [`CudaDevice`]: `hekit_ckks::HeOps` with
/// `Ct = `[`CudaCiphertext<'d>`](CudaCiphertext), `Pt = `[`DevicePlaintext<'d>`](DevicePlaintext)
/// (see the [module docs](self) for the contract, the counting and the key deviations).
///
/// `Sync`: one evaluator can be shared by threads (each thread issues its work on its current
/// stream, [`CudaDevice::use_stream`]); the counter is behind a mutex.
#[derive(Debug)]
pub struct CudaEvaluator<'d> {
    dev: &'d CudaDevice,
    params: Arc<CkksParams>,
    relin: Option<DeviceKsk<'d>>,
    /// Galois keys by Galois element.
    galois: HashMap<u64, DeviceKsk<'d>>,
    key_level: usize,
    config: EvalConfig,
    counter: Mutex<OpCounter>,
}

const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<CudaEvaluator<'static>>();
};

impl<'d> CudaEvaluator<'d> {
    /// Uploads the relinearization key and every Galois key (full keys, all levels) to `dev`
    /// with the ablation flags `config`.
    ///
    /// Errors: `InvalidParams` when `params` is not the device's parameter set (the same
    /// bases, as `CudaDevice::new(.., &params, ..)` shares them) or a key is not over it;
    /// `Device` when the keys do not fit in device memory (nothing is kept).
    pub fn new(
        dev: &'d CudaDevice,
        params: Arc<CkksParams>,
        relin: &RelinKey,
        galois: &GaloisKeys,
        config: EvalConfig,
    ) -> Result<CudaEvaluator<'d>> {
        let keys: Vec<&GaloisKey> = galois.iter().collect();
        let level = params.max_level();
        CudaEvaluator::with_keys(dev, params, Some(relin), &keys, config, level)
    }

    /// [`CudaEvaluator::new`] with an optional relinearization key, a chosen list of Galois
    /// keys (e.g. `steps.iter().map(|&s| gks.get(s))`) and every key uploaded for levels
    /// `<= key_level` only (smaller device keys; switching above `key_level` is
    /// `InvalidParams`, see the [module docs](self)). Duplicate elements are uploaded once.
    ///
    /// Errors as [`CudaEvaluator::new`], and `InvalidParams` for `key_level > L`.
    pub fn with_keys(
        dev: &'d CudaDevice,
        params: Arc<CkksParams>,
        relin: Option<&RelinKey>,
        galois: &[&GaloisKey],
        config: EvalConfig,
        key_level: usize,
    ) -> Result<CudaEvaluator<'d>> {
        let dp = dev.params();
        if !(Arc::ptr_eq(params.basis_q(), dp.basis_q())
            && Arc::ptr_eq(params.basis_p(), dp.basis_p())
            && params.dnum() == dp.dnum()
            && params.scale().to_bits() == dp.scale().to_bits())
        {
            return Err(invalid(
                "CudaEvaluator: the parameters are not the device's parameter set",
            ));
        }
        if key_level > params.max_level() {
            return Err(invalid(format!(
                "key level {key_level} above max_level {}",
                params.max_level()
            )));
        }
        let relin = relin
            .map(|rk| dev.upload_relin_key_with(rk, key_level))
            .transpose()?;
        let mut keys = HashMap::new();
        for gk in galois {
            if let std::collections::hash_map::Entry::Vacant(e) = keys.entry(gk.elt()) {
                e.insert(dev.upload_galois_key_with(gk, key_level)?);
            }
        }
        Ok(CudaEvaluator {
            dev,
            params,
            relin,
            galois: keys,
            key_level,
            config,
            counter: Mutex::new(OpCounter::default()),
        })
    }

    /// An evaluator sharing this one's device keys (no upload, no device memory) with the
    /// ablation flags `config` and its own counter starting at zero (e.g. for the RQ3 grid).
    pub fn with_config(&self, config: EvalConfig) -> CudaEvaluator<'d> {
        let mut e = self.fork();
        e.config = config;
        e
    }

    /// The device.
    pub fn device(&self) -> &'d CudaDevice {
        self.dev
    }

    /// The highest level the uploaded keys serve (`L` for full keys).
    pub fn key_level(&self) -> usize {
        self.key_level
    }

    /// Device bytes held by the keys (shared with forks).
    pub fn key_bytes(&self) -> usize {
        self.relin.as_ref().map_or(0, DeviceKsk::device_bytes)
            + self
                .galois
                .values()
                .map(DeviceKsk::device_bytes)
                .sum::<usize>()
    }

    /// Number of Galois keys on the device.
    pub fn num_galois_keys(&self) -> usize {
        self.galois.len()
    }

    /// Applies `f` to the counter (after an operation succeeded).
    fn count(&self, f: impl FnOnce(&mut OpCounter)) {
        f(&mut self.counter.lock().unwrap_or_else(PoisonError::into_inner));
    }

    fn check_ct(&self, ct: &CudaCiphertext<'_>) -> Result<()> {
        if std::ptr::eq(ct.device(), self.dev) {
            Ok(())
        } else {
            Err(invalid("operand is not on this evaluator's device"))
        }
    }

    fn check_pt(&self, p: &DevicePlaintext<'_>) -> Result<()> {
        if std::ptr::eq(p.poly().device(), self.dev) {
            Ok(())
        } else {
            Err(invalid("plaintext is not on this evaluator's device"))
        }
    }

    /// `InvalidParams` unless `ct` is this device's and has 2 components (the CPU's
    /// `check_galois_input`).
    fn check_galois_input(&self, ct: &CudaCiphertext<'_>) -> Result<()> {
        self.check_ct(ct)?;
        if ct.len() != 2 {
            return Err(invalid(format!(
                "rotation needs a 2-component ciphertext (got {}); relinearize first",
                ct.len()
            )));
        }
        Ok(())
    }

    /// The Galois key of element `elt` usable at `level`: `"missing galois key"`, or a level
    /// above the uploaded keys (`InvalidParams`).
    fn galois_key(&self, elt: u64, level: usize) -> Result<&DeviceKsk<'d>> {
        let k = self
            .galois
            .get(&elt)
            .ok_or_else(|| invalid("missing galois key"))?;
        crate::ciphertext::check_key_level(k, level)?;
        Ok(k)
    }

    /// `Ok` when every nontrivial step of `steps` has a key usable at `level`.
    fn check_rotation_keys(&self, steps: &[i32], level: usize) -> Result<()> {
        for &s in steps {
            if let Some(elt) = rotation_elt(&self.params, s) {
                self.galois_key(elt, level)?;
            }
        }
        Ok(())
    }

    /// The relinearization key usable at `level`.
    fn relin_key(&self, level: usize) -> Result<&DeviceKsk<'d>> {
        let rk = self
            .relin
            .as_ref()
            .ok_or_else(|| invalid("evaluator has no relinearization key"))?;
        crate::ciphertext::check_key_level(rk, level)?;
        Ok(rk)
    }

    /// The operand checks of `mul_no_relin` / `mul`, in the CPU's order: device, component
    /// count, level mismatch, level 0.
    fn check_mul_operands(&self, a: &CudaCiphertext<'_>, b: &CudaCiphertext<'_>) -> Result<()> {
        self.check_ct(a)?;
        self.check_ct(b)?;
        if a.len() != 2 || b.len() != 2 {
            return Err(invalid(format!(
                "multiplication needs 2-component ciphertexts (got {} and {}); relinearize first",
                a.len(),
                b.len()
            )));
        }
        check_level(a.level(), b.level())?;
        if a.level() == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        Ok(())
    }

    fn add_sub(
        &self,
        a: &CudaCiphertext<'d>,
        b: &CudaCiphertext<'d>,
        sub: bool,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_ct(a)?;
        self.check_ct(b)?;
        check_level(a.level(), b.level())?;
        check_scale(a.scale(), b.scale())?;
        let (x, y) = (a.polys(), b.polys());
        let c = (0..x.len().max(y.len()))
            .map(|i| match (x.get(i), y.get(i), sub) {
                (Some(p), Some(q), false) => self.dev.add(p, q),
                (Some(p), Some(q), true) => self.dev.sub(p, q),
                (Some(p), None, _) => Ok(p.clone()),
                (None, Some(q), false) => Ok(q.clone()),
                (None, Some(q), true) => self.dev.neg(q),
                (None, None, _) => unreachable!("index below the longer length"),
            })
            .collect::<Result<Vec<_>>>()?;
        CudaCiphertext::new(c, a.scale())
    }

    /// The first `level + 1` limbs of a ciphertext component (a device copy).
    fn truncate(&self, p: &DevicePoly<'d>, level: usize) -> Result<DevicePoly<'d>> {
        let d = Dev(self.dev);
        let mut out = d.alloc(level + 1)?;
        d.copy_limbs(p.limbs(), 0, &mut out, 0, level + 1)?;
        Ok(self
            .dev
            .new_poly(out, q_primes(level).into(), p.is_ntt_form()))
    }

    /// The checks of `matvec_enc` in the CPU's order (all before any work): `x` and every
    /// entry are 2-component ciphertexts of this device, [`check_shape`], the Galois keys of
    /// [`Diagonals::rotation_steps`] at `x`'s level, `NoLevelsLeft` at level 0, the
    /// relinearization key.
    fn check_matvec_enc(
        &self,
        diag: &Diagonals<CudaCiphertext<'d>>,
        x: &CudaCiphertext<'d>,
    ) -> Result<()> {
        self.check_galois_input(x)?;
        for c in diag.entries() {
            self.check_galois_input(c)?;
        }
        check_shape(diag, x.level(), self.params.slots())?;
        self.check_rotation_keys(&diag.rotation_steps(), x.level())?;
        if x.level() == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        self.relin_key(x.level())?;
        Ok(())
    }

    /// Device bytes [`CudaEvaluator::matvec_enc_batch`] reserves per worker stream for
    /// products at `level` of dimension `d` ([`matvec_enc_workspace_limbs`]).
    pub fn matvec_enc_workspace_bytes(&self, level: usize, d: usize) -> usize {
        8 * self.params.n() * matvec_enc_workspace_limbs(&self.params, level, d)
    }

    /// `[matvec_enc(diags[i], xs[i]) for i]`, the independent products spread over the
    /// device's streams: job `i` runs on stream `i mod w`, `w = min(n_streams, jobs)`, each
    /// stream driven by its own worker thread. Results (in job order), counts and errors are
    /// those of the sequential loop; every result is bit-identical to `matvec_enc`'s (and the
    /// CPU's).
    ///
    /// Before any work, every job is checked (as `matvec_enc`), and the working set of each
    /// worker stream ([`CudaEvaluator::matvec_enc_workspace_bytes`] of its largest job) is
    /// pre-allocated on that stream, all at once: `Device` (no abort, nothing counted) when
    /// the `w` reservations exceed the device's total memory or do not fit now. The
    /// reservations are then freed into the stream-ordered pool the jobs allocate from.
    /// After the jobs, every worker stream is synchronized, so no read of a shared operand
    /// is in flight when the call returns (cudarc tracks only the last reader of a buffer;
    /// Task 46 review M7g). Counts are merged only when every job succeeded.
    ///
    /// Errors: `InvalidParams` when `diags` and `xs` differ in length; those of
    /// `matvec_enc` (the first failing job in order); `Device` as above.
    pub fn matvec_enc_batch(
        &self,
        diags: &[Diagonals<CudaCiphertext<'d>>],
        xs: &[CudaCiphertext<'d>],
    ) -> Result<Vec<CudaCiphertext<'d>>> {
        if diags.len() != xs.len() {
            return Err(invalid(format!(
                "matvec_enc_batch: {} diagonal sets for {} vectors",
                diags.len(),
                xs.len()
            )));
        }
        for (d, x) in diags.iter().zip(xs) {
            self.check_matvec_enc(d, x)?;
        }
        let jobs = xs.len();
        let workers = self.dev.n_streams().min(jobs);
        if workers <= 1 {
            let local = self.fork();
            let out = diags
                .iter()
                .zip(xs)
                .map(|(d, x)| local.matvec_enc(d, x))
                .collect::<Result<Vec<_>>>()?;
            self.add_counts(&local.counter());
            return Ok(out);
        }
        // Per worker: its jobs and the largest working set among them.
        let ws: Vec<usize> = (0..workers)
            .map(|w| {
                (w..jobs)
                    .step_by(workers)
                    .map(|i| self.matvec_enc_workspace_bytes(xs[i].level(), diags[i].d()))
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        self.reserve(&ws)?;
        type Done<'a> =
            std::result::Result<(Vec<(usize, CudaCiphertext<'a>)>, OpCounter), (usize, HekitError)>;
        let results: Vec<Done<'d>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..workers)
                .map(|w| {
                    s.spawn(move || -> Done<'d> {
                        let _g = self.dev.use_stream(w).map_err(|e| (w, e))?;
                        let local = self.fork();
                        let mut out = Vec::new();
                        for i in (w..jobs).step_by(workers) {
                            let r = local.matvec_enc(&diags[i], &xs[i]).map_err(|e| (i, e))?;
                            out.push((i, r));
                        }
                        Ok((out, local.counter()))
                    })
                })
                .collect();
            handles
                .into_iter()
                .enumerate()
                .map(|(w, h)| {
                    h.join().unwrap_or_else(|_| {
                        Err((
                            w,
                            HekitError::Device("matvec_enc_batch: a worker thread panicked".into()),
                        ))
                    })
                })
                .collect()
        });
        // Join every worker stream before shared operands can be dropped (all of them, even
        // after an error).
        let mut synced = Ok(());
        for w in 0..workers {
            let r = self.dev.sync_stream(w);
            if synced.is_ok() {
                synced = r;
            }
        }
        let mut slots: Vec<Option<CudaCiphertext<'d>>> = vec![None; jobs];
        let mut total = OpCounter::default();
        let mut first_err: Option<(usize, HekitError)> = None;
        for r in results {
            match r {
                Ok((out, c)) => {
                    total += &c;
                    for (i, ct) in out {
                        slots[i] = Some(ct);
                    }
                }
                // The error of the earliest failing job (job order), as the sequential loop.
                Err((i, e)) => {
                    if first_err.as_ref().is_none_or(|(j, _)| i < *j) {
                        first_err = Some((i, e));
                    }
                }
            }
        }
        if let Some((_, e)) = first_err {
            return Err(e);
        }
        synced?;
        self.add_counts(&total);
        slots
            .into_iter()
            .map(|c| c.ok_or_else(|| HekitError::Device("matvec_enc_batch: missing result".into())))
            .collect()
    }

    /// Pre-allocates `ws[w]` bytes on stream `w` for every `w` at once, then frees them
    /// (stream-ordered, into the pool the jobs allocate from): `Device` when they cannot all
    /// be held, checked first against the device's total memory (then nothing is allocated).
    fn reserve(&self, ws: &[usize]) -> Result<()> {
        const MIB: f64 = (1u64 << 20) as f64;
        let need: usize = ws.iter().sum();
        let (_free, total) = self.dev.mem_info()?;
        if need > total {
            return Err(HekitError::Device(format!(
                "matvec_enc_batch: {} worker streams need {:.1} MiB of working memory, more \
                 than the device's {:.1} MiB (use fewer streams)",
                ws.len(),
                need as f64 / MIB,
                total as f64 / MIB
            )));
        }
        let limbs_per = |b: usize| b.div_ceil(8 * self.params.n()).max(1);
        let mut held = Vec::with_capacity(ws.len());
        for (w, &b) in ws.iter().enumerate() {
            let _g = self.dev.use_stream(w)?;
            match self.dev.alloc_limbs(limbs_per(b)) {
                Ok(buf) => held.push(buf),
                Err(e) => {
                    return Err(HekitError::Device(format!(
                        "matvec_enc_batch: reserving {:.1} MiB on each of {} streams \
                         ({:.1} MiB in all) failed at stream {w}: {e}",
                        b as f64 / MIB,
                        ws.len(),
                        need as f64 / MIB
                    )));
                }
            }
        }
        // Each buffer is freed on the stream it was allocated on (the guard is not needed for
        // the free: cudarc frees on the allocating stream).
        drop(held);
        Ok(())
    }
}

impl<'d> HeOps for CudaEvaluator<'d> {
    type Ct = CudaCiphertext<'d>;
    type Pt = DevicePlaintext<'d>;

    fn params(&self) -> &Arc<CkksParams> {
        &self.params
    }

    fn config(&self) -> &EvalConfig {
        &self.config
    }

    fn counter(&self) -> OpCounter {
        self.counter
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn reset_counter(&self) {
        self.count(|c| *c = OpCounter::default());
    }

    fn add_counts(&self, counts: &OpCounter) {
        self.count(|c| *c += counts);
    }

    fn fork(&self) -> CudaEvaluator<'d> {
        CudaEvaluator {
            dev: self.dev,
            params: Arc::clone(&self.params),
            relin: self.relin.clone(),
            galois: self.galois.clone(),
            key_level: self.key_level,
            config: self.config,
            counter: Mutex::new(OpCounter::default()),
        }
    }

    fn has_relin_key(&self) -> bool {
        self.relin.is_some()
    }

    fn has_galois_key(&self, step: i32) -> bool {
        self.galois.contains_key(&galois_elt(self.params.n(), step))
    }

    fn owns(&self, ct: &CudaCiphertext<'d>) -> bool {
        std::ptr::eq(ct.device(), self.dev)
    }

    fn upload_ct(&self, ct: &Ciphertext) -> Result<CudaCiphertext<'d>> {
        self.dev.upload_ct(ct)
    }

    fn download_ct(&self, ct: &CudaCiphertext<'d>) -> Result<Ciphertext> {
        self.check_ct(ct)?;
        ct.download()
    }

    fn upload_pt(&self, pt: &Plaintext) -> Result<DevicePlaintext<'d>> {
        self.dev.upload_pt(pt)
    }

    fn add(&self, a: &CudaCiphertext<'d>, b: &CudaCiphertext<'d>) -> Result<CudaCiphertext<'d>> {
        self.add_sub(a, b, false)
    }

    fn sub(&self, a: &CudaCiphertext<'d>, b: &CudaCiphertext<'d>) -> Result<CudaCiphertext<'d>> {
        self.add_sub(a, b, true)
    }

    fn negate(&self, a: &CudaCiphertext<'d>) -> Result<CudaCiphertext<'d>> {
        self.check_ct(a)?;
        let c = a
            .polys()
            .iter()
            .map(|p| self.dev.neg(p))
            .collect::<Result<Vec<_>>>()?;
        CudaCiphertext::new(c, a.scale())
    }

    fn add_plain(
        &self,
        a: &CudaCiphertext<'d>,
        p: &DevicePlaintext<'d>,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_ct(a)?;
        self.check_pt(p)?;
        check_level(a.level(), p.level())?;
        check_scale(a.scale(), p.scale())?;
        let mut c = a.polys().to_vec();
        c[0] = self.dev.add(&c[0], p.poly())?;
        CudaCiphertext::new(c, a.scale())
    }

    fn mul_plain(
        &self,
        a: &CudaCiphertext<'d>,
        p: &DevicePlaintext<'d>,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_ct(a)?;
        self.check_pt(p)?;
        check_level(a.level(), p.level())?;
        let c = a
            .polys()
            .iter()
            .map(|x| self.dev.mul_pointwise(x, p.poly()))
            .collect::<Result<Vec<_>>>()?;
        let r = CudaCiphertext::new(c, a.scale() * p.scale())?;
        self.count(|c| c.ct_pt_mults += 1);
        Ok(r)
    }

    fn mul_scalar(
        &self,
        ct: &CudaCiphertext<'d>,
        value: f64,
        target_scale: f64,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_ct(ct)?;
        if !value.is_finite() {
            return Err(invalid(format!("scalar {value} must be finite")));
        }
        let label = self.exact_product_scale(ct.level(), target_scale)?;
        check_plain_scale(label / ct.scale())?;
        let k = (value * (label / ct.scale())).round();
        if k.abs() >= MAX_SCALAR_INT {
            return Err(invalid(format!(
                "scalar {value:e} at plaintext scale {:e} is the integer {k:e} >= 2^62",
                label / ct.scale()
            )));
        }
        let c = ct
            .polys()
            .iter()
            .map(|p| {
                let m = self.dev.mul_scalar(p, k.abs() as u64)?;
                if k < 0.0 { self.dev.neg(&m) } else { Ok(m) }
            })
            .collect::<Result<Vec<_>>>()?;
        let r = CudaCiphertext::new(c, label)?;
        self.count(|c| c.ct_pt_mults += 1);
        Ok(r)
    }

    fn mul_plain_to_scale(
        &self,
        a: &CudaCiphertext<'d>,
        p: &DevicePlaintext<'d>,
        target_scale: f64,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_ct(a)?;
        self.check_pt(p)?;
        check_level(a.level(), p.level())?;
        let label = self.exact_product_scale(a.level(), target_scale)?;
        check_plain_scale(label / a.scale())?;
        if p.scale() != label / a.scale() {
            return Err(HekitError::ScaleMismatch);
        }
        let c = a
            .polys()
            .iter()
            .map(|x| self.dev.mul_pointwise(x, p.poly()))
            .collect::<Result<Vec<_>>>()?;
        let r = CudaCiphertext::new(c, label)?;
        self.count(|c| c.ct_pt_mults += 1);
        Ok(r)
    }

    fn mul_no_relin(
        &self,
        a: &CudaCiphertext<'d>,
        b: &CudaCiphertext<'d>,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_mul_operands(a, b)?;
        let r = self.dev.mul_no_relin(a, b)?;
        self.count(|c| c.ct_ct_tensor += 1);
        Ok(r)
    }

    fn relinearize(&self, ct: &CudaCiphertext<'d>) -> Result<CudaCiphertext<'d>> {
        self.check_ct(ct)?;
        if ct.len() != 3 {
            return Err(invalid(format!(
                "relinearize needs a 3-component ciphertext (got {})",
                ct.len()
            )));
        }
        let rk = self.relin_key(ct.level())?;
        let r = self.dev.relinearize(ct, rk)?;
        self.count(|c| c.relins += 1);
        Ok(r)
    }

    fn mul(&self, a: &CudaCiphertext<'d>, b: &CudaCiphertext<'d>) -> Result<CudaCiphertext<'d>> {
        self.check_mul_operands(a, b)?;
        self.relin_key(a.level())?;
        self.relinearize(&self.mul_no_relin(a, b)?)
    }

    fn rescale(&self, ct: &CudaCiphertext<'d>) -> Result<CudaCiphertext<'d>> {
        self.check_ct(ct)?;
        let r = self.dev.rescale(ct)?;
        self.count(|c| c.rescales += 1);
        Ok(r)
    }

    fn mod_switch_to_level(
        &self,
        ct: &CudaCiphertext<'d>,
        level: usize,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_ct(ct)?;
        if level > ct.level() {
            return Err(invalid(format!(
                "cannot switch a level-{} ciphertext up to level {level}",
                ct.level()
            )));
        }
        if level == ct.level() {
            return Ok(ct.clone());
        }
        let c = ct
            .polys()
            .iter()
            .map(|p| self.truncate(p, level))
            .collect::<Result<Vec<_>>>()?;
        CudaCiphertext::new(c, ct.scale())
    }

    fn relabel(&self, ct: CudaCiphertext<'d>, scale: f64) -> Result<CudaCiphertext<'d>> {
        ct.with_scale(scale)
    }

    fn rotate(&self, ct: &CudaCiphertext<'d>, steps: i32) -> Result<CudaCiphertext<'d>> {
        self.check_galois_input(ct)?;
        let Some(elt) = rotation_elt(&self.params, steps) else {
            return Ok(ct.clone());
        };
        let key = self.galois_key(elt, ct.level())?;
        let r = self.dev.rotate(ct, steps, &[key])?;
        self.count(|c| c.rotations += 1);
        Ok(r)
    }

    fn conjugate(&self, ct: &CudaCiphertext<'d>) -> Result<CudaCiphertext<'d>> {
        self.check_galois_input(ct)?;
        let key = self.galois_key(conjugation_elt(self.params.n()), ct.level())?;
        let r = self.dev.conjugate(ct, &[key])?;
        self.count(|c| c.rotations += 1);
        Ok(r)
    }

    fn hoisted_rotate(
        &self,
        ct: &CudaCiphertext<'d>,
        steps: &[i32],
    ) -> Result<Vec<CudaCiphertext<'d>>> {
        self.check_galois_input(ct)?;
        self.check_rotation_keys(steps, ct.level())?;
        if !self.config.hoist {
            return steps.iter().map(|&s| self.rotate(ct, s)).collect();
        }
        let mut keys: Vec<&DeviceKsk<'d>> = Vec::new();
        let mut nontrivial = 0u64;
        for &s in steps {
            if let Some(elt) = rotation_elt(&self.params, s) {
                nontrivial += 1;
                let k = self.galois_key(elt, ct.level())?;
                if !keys.iter().any(|o| std::ptr::eq(*o, k)) {
                    keys.push(k);
                }
            }
        }
        let r = self.dev.hoisted_rotate(ct, steps, &keys)?;
        self.count(|c| c.rotations += nontrivial);
        Ok(r)
    }

    fn matvec_plain(
        &self,
        diag: &Diagonals<DevicePlaintext<'d>>,
        x: &CudaCiphertext<'d>,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_galois_input(x)?;
        for p in diag.entries() {
            self.check_pt(p)?;
        }
        check_shape(diag, x.level(), self.params.slots())?;
        self.check_rotation_keys(&diag.rotation_steps(), x.level())?;
        matvec_plain_bsgs(self, diag, x)
    }

    fn matvec_enc(
        &self,
        diag: &Diagonals<CudaCiphertext<'d>>,
        x: &CudaCiphertext<'d>,
    ) -> Result<CudaCiphertext<'d>> {
        self.check_matvec_enc(diag, x)?;
        matvec_enc_bsgs(self, diag, x)
    }
}
