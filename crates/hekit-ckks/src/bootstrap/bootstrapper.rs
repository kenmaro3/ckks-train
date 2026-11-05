//! The assembled CKKS bootstrap (Task 38): mod-raise with sparse-secret encapsulation,
//! CoeffToSlot, EvalMod on the real and the imaginary part, SlotToCoeff
//! (`docs/bootstrap.md` §2 has the pipeline, §11 the measurements).
//!
//! # Scales and levels
//!
//! The input is a ciphertext at level 0 with the preset scale `Delta`
//! ([`CkksParams::scale`]); its plaintext coefficients are `Delta c_j`, `c = encode(m)`.
//!
//! 1. [`Evaluator::mod_raise`] (two key switches, to the ephemeral key and back) gives
//!    `t_j = Delta c_j + q_0 I_j` at the top level. It is relabelled to the scale `q_0`
//!    (`params.q()[0] as f64`, the value [`StageSpec::input_scale`] asks for), so its slots
//!    hold `x_j = t_j / q_0 = I_j + Delta c_j / q_0`.
//! 2. [`Evaluator::coeff_to_slot`] with the factor `1 / K` folded in: two ciphertexts whose
//!    slots are `y_j = x_j / K` (bit-reversed order), at the scale `scale_at` of the level
//!    below the CoeffToSlot band (2^58 at level 27 for n16_boot), as [`Evaluator::eval_mod`]
//!    expects them.
//! 3. [`Evaluator::eval_mod`] on both: `sin(2 pi x_j) ~ 2 pi Delta c_j / q_0` at the same
//!    scale `S`, `evalmod_depth` levels lower.
//! 4. Both are relabelled to `S Delta / q_0` (about 2^46), which turns the slots into
//!    `2 pi c_j` for free, and dropped to the top of the SlotToCoeff band if EvalMod used
//!    fewer than `evalmod_levels` levels. [`Evaluator::slot_to_coeff`] with the factor
//!    `1 / (2 pi)` folded in lands exactly on `Delta` at [`levels_after_boot`]: the slots
//!    hold `m` again.
//!
//! # Counting
//!
//! The bootstrap runs on an internal evaluator that shares the caller's relinearization key
//! and the bootstrapper's Galois keys. On success every operation it performed is added to
//! the caller's [`OpCounter`] (rotations and conjugations, relinearizations, products,
//! rescales, and the 2 `sse_switches` of the mod-raise), plus `bootstraps += 1`; a failed
//! bootstrap counts nothing. [`Bootstrapper::cost_per_boot`] plans these counts from the
//! encoded stages and the EvalMod series, and the tests check it against the counter.

use std::f64::consts::PI;
use std::sync::Arc;

use num_complex::Complex64;

use hekit_core::error::{HekitError, Result};

use crate::bootstrap::evalmod::evalmod_cost;
use crate::bootstrap::linear::{
    Direction, LinearTransform, StageSpec, dft_stages, encode_stages, galois_steps_for,
    transforms_cost,
};
use crate::bootstrap::params::{BootParams, levels_after_boot};
use crate::ciphertext::Ciphertext;
use crate::counter::OpCounter;
use crate::encoder::Encoder;
use crate::evaluator::Evaluator;
use crate::keys::{GaloisKeys, KeyGenerator, KskKey, SseKeys, conjugation_elt};
use crate::params::CkksParams;

/// The keys of a bootstrap besides the caller's relinearization key: the Galois keys of the
/// CoeffToSlot and SlotToCoeff rotation steps ([`galois_steps_for`], 32 for n16_boot) **with
/// the conjugation key inserted** (CoeffToSlot conjugates twice), and the two switching keys
/// of sparse-secret encapsulation when [`BootParams::ephemeral_hw`] is set. About 10 GB at
/// n16_boot (33 Galois keys of 0.30 GB plus the SSE keys; [`BootstrapKeys::bytes`]).
///
/// The Galois keys are behind an `Arc` that the bootstrap's internal evaluator shares, so no
/// key is copied per bootstrap. (The brief's `pub conj: GaloisKey` field is the conjugation
/// entry of [`BootstrapKeys::galois`]: [`Evaluator::conjugate`] looks it up there.)
#[derive(Clone)]
pub struct BootstrapKeys {
    galois: Arc<GaloisKeys>,
    steps: Vec<i32>,
    sse: Option<SseKeys>,
}

impl std::fmt::Debug for BootstrapKeys {
    /// The shape only (no key material).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BootstrapKeys")
            .field("rotation_steps", &self.steps.len())
            .field("galois_keys", &self.galois.len())
            .field("sse", &self.sse)
            .field("bytes", &self.bytes())
            .finish()
    }
}

/// The CoeffToSlot and SlotToCoeff stage matrices of `bp` (full packing).
fn stages(
    bp: &BootParams,
) -> Result<(
    Vec<crate::bootstrap::DiagMatrix>,
    Vec<crate::bootstrap::DiagMatrix>,
)> {
    Ok((
        dft_stages(bp.log_slots, bp.ctos_levels, true)?,
        dft_stages(bp.log_slots, bp.stoc_levels, false)?,
    ))
}

impl BootstrapKeys {
    /// Generates the keys of `bp` with `kg` (its secret key): the Galois keys of
    /// [`galois_steps_for`] of the CoeffToSlot and SlotToCoeff stages, the conjugation key,
    /// and [`KeyGenerator::sse_keys`]`(h')` if `bp.ephemeral_hw = Some(h')`. At n16_boot this
    /// takes minutes and about 10 GB.
    ///
    /// Errors (`InvalidParams`): `bp` does not validate against `kg`'s parameters
    /// ([`BootParams::validate`]), or `bp.log_slots` is not full packing (`log_n - 1`).
    pub fn generate(kg: &mut KeyGenerator, bp: &BootParams) -> Result<BootstrapKeys> {
        check_boot_params(kg.params(), bp)?;
        let (cts, stc) = stages(bp)?;
        let all: Vec<_> = cts.into_iter().chain(stc).collect();
        let steps = galois_steps_for(&all);
        let mut galois = kg.galois_keys(&steps);
        galois.insert(kg.conjugation_key());
        let sse = bp.ephemeral_hw.map(|h| kg.sse_keys(h)).transpose()?;
        Ok(BootstrapKeys {
            galois: Arc::new(galois),
            steps,
            sse,
        })
    }

    /// The Galois keys: one per rotation step ([`BootstrapKeys::rotation_steps`]) and the
    /// conjugation key.
    pub fn galois(&self) -> &GaloisKeys {
        &self.galois
    }

    /// The rotation steps of the Galois keys (sorted, in `1..slots`).
    pub fn rotation_steps(&self) -> &[i32] {
        &self.steps
    }

    /// The sparse-secret-encapsulation keys, if `ephemeral_hw` was set.
    pub fn sse(&self) -> Option<&SseKeys> {
        self.sse.as_ref()
    }

    /// The memory of the key material in bytes (8 bytes per stored word).
    pub fn bytes(&self) -> usize {
        let n = self.galois_n();
        let conj = conjugation_elt(n);
        let galois: usize = self
            .steps
            .iter()
            .filter_map(|&s| self.galois.get(s))
            .chain(self.galois.get_elt(conj))
            .map(|k| ksk_bytes(k.key()))
            .sum();
        let sse = self
            .sse
            .as_ref()
            .map_or(0, |k| ksk_bytes(k.to_sparse()) + ksk_bytes(k.to_dense()));
        galois + sse
    }

    /// The ring degree of the keys.
    fn galois_n(&self) -> usize {
        self.steps
            .first()
            .and_then(|&s| self.galois.get(s))
            .and_then(|k| k.key().parts().first())
            .map_or(0, |(b, _)| b.q().basis().n())
    }
}

/// The bytes of a key-switching key: every limb of both polynomials of every part.
fn ksk_bytes(k: &KskKey) -> usize {
    k.parts()
        .iter()
        .flat_map(|(b, a)| [b, a])
        .map(|x| {
            let n = x.q().basis().n();
            (x.q().limbs().len() + x.p().limbs().len()) * n * 8
        })
        .sum()
}

/// `InvalidParams` unless `bp` validates against `params` with full packing.
fn check_boot_params(params: &CkksParams, bp: &BootParams) -> Result<()> {
    bp.validate(params)?;
    if bp.log_slots + 1 != params.log_n() {
        return Err(HekitError::InvalidParams(format!(
            "bootstrapping needs full packing: log_slots {} for log_n {}",
            bp.log_slots,
            params.log_n()
        )));
    }
    Ok(())
}

/// CKKS bootstrapping for one parameter set: the encoded CoeffToSlot and SlotToCoeff
/// stages, the EvalMod parameters and the [`BootstrapKeys`] (see the [module docs](self)).
/// At n16_boot it holds about 13 GB (keys 10 GB, encoded diagonals about 3 GB).
pub struct Bootstrapper {
    params: Arc<CkksParams>,
    bp: BootParams,
    ctos: Vec<LinearTransform>,
    stoc: Vec<LinearTransform>,
    keys: BootstrapKeys,
    /// The CoeffToSlot output scale, which EvalMod keeps exactly.
    evalmod_scale: f64,
    /// The scale SlotToCoeff expects: the EvalMod output scale times `Delta / q_0`.
    stoc_input_scale: f64,
    cost: OpCounter,
}

impl std::fmt::Debug for Bootstrapper {
    /// A summary without keys or diagonals.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bootstrapper")
            .field("n", &self.params.n())
            .field("bp", &self.bp)
            .field("keys", &self.keys)
            .field("key_switches_per_boot", &self.key_switches_per_boot())
            .finish()
    }
}

impl Bootstrapper {
    /// Encodes the stages of `bp` with `encoder` and generates the [`BootstrapKeys`] with
    /// `kg`; all three must be over `params` (the same `Arc`). The bootstrap also needs the
    /// relinearization key, which comes with the evaluator passed to
    /// [`Bootstrapper::bootstrap`].
    ///
    /// CoeffToSlot runs from the top level, from the scale `q_0` to the nominal
    /// [`CkksParams::scale_at`] of the level below its band, with the factor `1 / K`;
    /// SlotToCoeff from the top of its band (`levels_after_boot + stoc_levels`), from the
    /// EvalMod output scale times `Delta / q_0` to exactly `Delta`, with the factor
    /// `1 / (2 pi)` (module docs).
    ///
    /// Errors (`InvalidParams`): `encoder` or `kg` over other parameters; `bp` not valid for
    /// `params` ([`BootParams::validate`]) or not full packing; and the errors of
    /// [`encode_stages`].
    pub fn new(
        params: Arc<CkksParams>,
        bp: BootParams,
        encoder: &Encoder,
        kg: &mut KeyGenerator,
    ) -> Result<Bootstrapper> {
        if !Arc::ptr_eq(encoder.params(), &params) || !Arc::ptr_eq(kg.params(), &params) {
            return Err(HekitError::InvalidParams(
                "Bootstrapper::new: encoder and key generator must share the parameters".into(),
            ));
        }
        check_boot_params(&params, &bp)?;
        let (cts, stc) = stages(&bp)?;
        let top = params.max_level();
        let q0 = params.q()[0] as f64;
        let k = bp.k_intervals as f64;
        let evalmod_scale = params.scale_at(top - bp.ctos_levels);
        let ctos = encode_stages(
            &cts,
            encoder,
            &StageSpec {
                direction: Direction::CoeffToSlot,
                start_level: top,
                input_scale: q0,
                output_scale: evalmod_scale,
                factor: Complex64::new(1.0 / k, 0.0),
            },
        )?;
        let stoc_input_scale = evalmod_scale * params.scale() / q0;
        let stoc = encode_stages(
            &stc,
            encoder,
            &StageSpec {
                direction: Direction::SlotToCoeff,
                start_level: levels_after_boot(&params, &bp) + bp.stoc_levels,
                input_scale: stoc_input_scale,
                output_scale: params.scale(),
                factor: Complex64::new(1.0 / (2.0 * PI), 0.0),
            },
        )?;
        let keys = BootstrapKeys::generate(kg, &bp)?;
        let mut cost = transforms_cost(&ctos, params.slots());
        let em = evalmod_cost(&bp);
        cost += &em;
        cost += &em;
        cost += &transforms_cost(&stoc, params.slots());
        cost.sse_switches = if keys.sse.is_some() { 2 } else { 0 };
        cost.bootstraps = 1;
        Ok(Bootstrapper {
            params,
            bp,
            ctos,
            stoc,
            keys,
            evalmod_scale,
            stoc_input_scale,
            cost,
        })
    }

    /// The exact shared parameter object used to encode and generate this bootstrap.
    pub fn params(&self) -> &Arc<CkksParams> {
        &self.params
    }

    /// Encoded CoeffToSlot stages, available for device upload without re-encoding.
    pub fn coeff_to_slot_stages(&self) -> &[LinearTransform] {
        &self.ctos
    }

    /// Encoded SlotToCoeff stages, available for device upload without re-encoding.
    pub fn slot_to_coeff_stages(&self) -> &[LinearTransform] {
        &self.stoc
    }

    /// The bootstrap parameters.
    pub fn boot_params(&self) -> &BootParams {
        &self.bp
    }

    /// The keys.
    pub fn keys(&self) -> &BootstrapKeys {
        &self.keys
    }

    /// The level of every bootstrap output ([`levels_after_boot`]: 16 for n16_boot).
    pub fn output_level(&self) -> usize {
        levels_after_boot(&self.params, &self.bp)
    }

    /// The key switches of the mod-raise: 2 with sparse-secret encapsulation (to the
    /// ephemeral key at level 0 and back at the top), 0 without.
    pub fn sse_switches_per_boot(&self) -> u64 {
        if self.keys.sse.is_some() { 2 } else { 0 }
    }

    /// What one [`Bootstrapper::bootstrap`] adds to the caller's counter, planned from the
    /// encoded stages ([`transforms_cost`]) and the EvalMod series ([`evalmod_cost`], twice);
    /// `sse_switches` = [`Bootstrapper::sse_switches_per_boot`], `bootstraps = 1`.
    pub fn cost_per_boot(&self) -> OpCounter {
        self.cost.clone()
    }

    /// The key switches of one bootstrap: the rotations and conjugations of CoeffToSlot and
    /// SlotToCoeff, the relinearizations of the two EvalMods and the mod-raise switches
    /// (`cost_per_boot().key_switches()`, SSE switches included). This
    /// replaces M1's `BOOT_KS_EQUIV` for planned bootstraps (see
    /// [`OpCounter::key_switches`]).
    pub fn key_switches_per_boot(&self) -> u64 {
        self.cost.key_switches()
    }

    /// Bootstraps `ct`: from level 0 at the scale `Delta` to [`Bootstrapper::output_level`]
    /// at exactly `Delta`, with the same slots up to the bootstrap error (module docs). The
    /// slots must satisfy `|m| <= 1` (the EvalMod precondition; precision falls
    /// quadratically above). `ev` supplies the relinearization key and the counter: every
    /// operation of the bootstrap is added to it, and `bootstraps` by one.
    ///
    /// Errors, before any work: `InvalidParams` if `ev` is over other parameters or has no
    /// relinearization key, `ct` is not at level 0 (drop it there first with
    /// [`Evaluator::mod_switch_to_level`]), or `ct` is not a 2-component ciphertext over these
    /// parameters; `ScaleMismatch` unless `ct.scale()` is exactly `Delta`
    /// ([`CkksParams::scale`]).
    pub fn bootstrap(&self, ev: &Evaluator, ct: &Ciphertext) -> Result<Ciphertext> {
        if !Arc::ptr_eq(ev.params(), &self.params) {
            return Err(HekitError::InvalidParams(
                "bootstrap: evaluator over other parameters".into(),
            ));
        }
        let relin = ev.relin_key_shared().cloned();
        if relin.is_none() {
            return Err(HekitError::InvalidParams(
                "bootstrap: the evaluator has no relinearization key (EvalMod needs it)".into(),
            ));
        }
        if ct.level() != 0 {
            return Err(HekitError::InvalidParams(format!(
                "bootstrap: input at level {} (must be 0)",
                ct.level()
            )));
        }
        if ct.scale() != self.params.scale() {
            return Err(HekitError::ScaleMismatch);
        }
        let inner = Evaluator::with_shared_keys(
            Arc::clone(&self.params),
            relin,
            Some(Arc::clone(&self.keys.galois)),
        );
        let out = self.pipeline(&inner, ct, None)?;
        let done = inner.counter();
        ev.count(|c| {
            *c += &done;
            c.bootstraps += 1;
        });
        Ok(out)
    }

    /// The four steps of the module docs on the internal evaluator (inputs checked by the
    /// caller). With `trace`, the intermediate ciphertexts are pushed in order: the relabelled
    /// mod-raise output, the CoeffToSlot pair, the EvalMod pair (tests only).
    fn pipeline(
        &self,
        inner: &Evaluator,
        ct: &Ciphertext,
        mut trace: Option<&mut Vec<Ciphertext>>,
    ) -> Result<Ciphertext> {
        let mut keep = |cts: &[&Ciphertext]| {
            if let Some(t) = trace.as_deref_mut() {
                t.extend(cts.iter().map(|&c| c.clone()));
            }
        };
        let raised = match &self.keys.sse {
            Some(sse) => inner.mod_raise(ct, sse)?,
            None => inner.mod_raise_main_key(ct)?,
        };
        let raised = Ciphertext::new(raised.into_polys(), self.params.q()[0] as f64)?;
        keep(&[&raised]);
        let (re, im) = inner.coeff_to_slot(&raised, &self.ctos)?;
        keep(&[&re, &im]);
        // Independent (in parallel with the `parallel` feature, crate::par).
        let (re, im) = crate::par::join(
            || inner.eval_mod(&re, &self.bp),
            || inner.eval_mod(&im, &self.bp),
        );
        let (re, im) = (re?, im?);
        keep(&[&re, &im]);
        let start = self.stoc[0].level;
        let to_stoc = |c: Ciphertext| -> Result<Ciphertext> {
            // The relabel to S Delta / q_0 is only valid from exactly the planned EvalMod
            // scale S (review m2); eval_mod guarantees it, this keeps a future change honest.
            if c.scale() != self.evalmod_scale {
                return Err(HekitError::ScaleMismatch);
            }
            let c = inner.mod_switch_to_level(&c, start)?;
            Ciphertext::new(c.into_polys(), self.stoc_input_scale)
        };
        inner.slot_to_coeff(&to_stoc(re)?, &to_stoc(im)?, &self.stoc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::precision::precision_bits;
    use hekit_core::sampler::Rng;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    /// Uniform values in `[-a, a)`.
    fn uniform(r: &mut Rng, len: usize, a: f64) -> Vec<f64> {
        (0..len)
            .map(|_| a * (2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0))
            .collect()
    }

    /// `±a` with random signs.
    fn signs(r: &mut Rng, len: usize, a: f64) -> Vec<f64> {
        (0..len)
            .map(|_| if r.next_u64() & 1 == 0 { a } else { -a })
            .collect()
    }

    /// The n16_boot bands in miniature at N = 2^10 (insecure, tests only): q_0 54, two
    /// 42-bit compute primes, StC 4 x 34, EvalMod 7 x 58, CtS 4 x 56, P 2 x 60, dnum 9.
    fn mini() -> CkksParams {
        let mut q = vec![54, 42, 42];
        q.extend([34; 4]);
        q.extend([58; 7]);
        q.extend([56; 4]);
        CkksParams::new(10, &q, &[60, 60], 42, 9).unwrap()
    }

    fn mini_bp() -> BootParams {
        BootParams {
            log_slots: 9,
            ..BootParams::default_n16()
        }
    }

    struct Fx {
        params: Arc<CkksParams>,
        enc: Encoder,
        encryptor: Encryptor,
        dec: Decryptor,
        ev: Evaluator,
        bs: Bootstrapper,
    }

    fn fixture(params: CkksParams, bp: BootParams, seed: u8) -> Fx {
        let params = Arc::new(params);
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let enc = Encoder::new(Arc::clone(&params));
        let pk = kg.public_key();
        let rk = kg.relin_key();
        let t = std::time::Instant::now();
        let bs = Bootstrapper::new(Arc::clone(&params), bp, &enc, &mut kg).unwrap();
        eprintln!(
            "Bootstrapper::new (N = 2^{}): {:.1} s, {:?}",
            params.log_n(),
            t.elapsed().as_secs_f64(),
            bs
        );
        Fx {
            encryptor: Encryptor::new(Arc::clone(&params), pk),
            dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
            ev: Evaluator::new(Arc::clone(&params), Some(rk), None),
            enc,
            params,
            bs,
        }
    }

    impl Fx {
        /// `m` encrypted at the top level, dropped to level 0.
        fn level0(&self, m: &[f64], r: &mut Rng) -> Ciphertext {
            let top = self.params.max_level();
            let pt = self.enc.encode(m, self.params.scale(), top).unwrap();
            let ct = self.encryptor.encrypt(&pt, r);
            self.ev.mod_switch_to_level(&ct, 0).unwrap()
        }

        fn decrypt(&self, ct: &Ciphertext) -> Vec<f64> {
            self.enc.decode(&self.dec.decrypt(ct))
        }

        /// Bootstraps `m`; checks level, scale and the counts; returns the precision in bits
        /// and the seconds of the bootstrap.
        fn run(&self, m: &[f64], r: &mut Rng) -> (f64, f64) {
            let ct = self.level0(m, r);
            let before = self.ev.counter();
            let t = std::time::Instant::now();
            let out = self.bs.bootstrap(&self.ev, &ct).unwrap();
            let secs = t.elapsed().as_secs_f64();
            let done = &self.ev.counter() - &before;
            assert_eq!(out.level(), self.bs.output_level());
            assert!(out.level() >= levels_after_boot(&self.params, self.bs.boot_params()));
            assert_eq!(out.scale(), self.params.scale());
            assert_eq!(done.bootstraps, 1);
            assert_eq!(done, self.bs.cost_per_boot());
            // RQ1: the measured key switches (SSE included) are the planned count.
            assert_eq!(done.sse_switches, self.bs.sse_switches_per_boot());
            assert_eq!(done.key_switches(), self.bs.key_switches_per_boot());
            (precision_bits(m, &self.decrypt(&out)), secs)
        }
    }

    /// Brief, Step 1 in miniature: random `m` in [-1, 1] on all 2^9 slots comes back with at
    /// least 16 bits at `levels_after_boot` and exactly `Delta`; the counter gains exactly the
    /// planned operations and one bootstrap.
    #[test]
    fn mini_bootstrap_precision_level_and_counts() {
        let fx = fixture(mini(), mini_bp(), 1);
        assert_eq!(fx.bs.output_level(), 2);
        assert_eq!(fx.bs.sse_switches_per_boot(), 2);
        let mut r = rng(2);
        let m = uniform(&mut r, fx.params.slots(), 1.0);
        let (bits, secs) = fx.run(&m, &mut r);
        let k = fx.bs.cost_per_boot();
        eprintln!(
            "mini bootstrap (N = 2^10): {bits:.2} bits, {secs:.2} s, {} key switches ({} rot, \
             {} relin, {} SSE)",
            fx.bs.key_switches_per_boot(),
            k.rotations,
            k.relins,
            fx.bs.sse_switches_per_boot()
        );
        assert!(bits >= 16.0, "{bits}");
        // EvalMod's relinearizations: 20 per call, two calls.
        assert_eq!(k.relins, 40);
        // A second bootstrap of the output (dropped to level 0) also works.
        let again = fx.level0(&m, &mut r);
        let out = fx.bs.bootstrap(&fx.ev, &again).unwrap();
        let low = fx.ev.mod_switch_to_level(&out, 0).unwrap();
        let twice = fx.bs.bootstrap(&fx.ev, &low).unwrap();
        let bits2 = precision_bits(&m, &fx.decrypt(&twice));
        eprintln!("mini bootstrap twice: {bits2:.2} bits");
        assert!(bits2 >= 15.0, "{bits2}");
        assert_eq!(fx.ev.counter().bootstraps, 3);
    }

    /// With the `parallel` feature the bootstrap is bit-identical on 1 and on 4 threads
    /// (`crate::par` collects every parallel section in its sequential order).
    #[cfg(feature = "parallel")]
    #[test]
    fn mini_bootstrap_is_bit_identical_across_thread_counts() {
        let fx = fixture(mini(), mini_bp(), 8);
        let mut r = rng(9);
        let m = uniform(&mut r, fx.params.slots(), 1.0);
        let ct = fx.level0(&m, &mut r);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| fx.bs.bootstrap(&fx.ev, &ct).unwrap())
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.scale(), b.scale());
        assert!(
            a.polys()
                .iter()
                .zip(b.polys())
                .all(|(x, y)| x.limbs() == y.limbs())
        );
    }

    /// Task 43 review N1/N2 (folded into Task 47): `HostBootstrapper` refuses the backend
    /// evaluator itself as its host (address check, nothing counted) and, with a fork as host,
    /// merges exactly one bootstrap's counts after the upload.
    #[test]
    fn host_bootstrapper_guards_double_counting() {
        use crate::ops::{BootstrapOps, HeOps, HostBootstrapper};
        let fx = fixture(mini(), mini_bp(), 11);
        let mut r = rng(12);
        let m = uniform(&mut r, fx.params.slots(), 1.0);
        let ct = fx.level0(&m, &mut r);
        let ev = fx.ev.fork();
        let same = HostBootstrapper::new(&fx.bs, &ev);
        assert!(matches!(
            same.bootstrap(&ev, &ct),
            Err(HekitError::InvalidParams(_))
        ));
        assert_eq!(ev.counter(), OpCounter::default());
        let host = ev.fork();
        let hb = HostBootstrapper::new(&fx.bs, &host);
        let out = hb.bootstrap(&ev, &ct).unwrap();
        assert_eq!(ev.counter(), host.counter());
        assert_eq!(ev.counter().bootstraps, 1);
        assert_eq!(out.level(), fx.bs.output_level());
    }

    /// Review Focus 1 and 2: every input error is reported before any work, and the counter
    /// does not move.
    #[test]
    fn errors_before_any_work() {
        let fx = fixture(mini(), mini_bp(), 3);
        let mut r = rng(4);
        let m = uniform(&mut r, fx.params.slots(), 1.0);
        let top = fx.params.max_level();
        let pt = fx.enc.encode(&m, fx.params.scale(), top).unwrap();
        let fresh = fx.encryptor.encrypt(&pt, &mut r);
        let before = fx.ev.counter();
        let inv = |e: Result<Ciphertext>| matches!(e, Err(HekitError::InvalidParams(_)));
        // Level != 0.
        assert!(inv(fx.bs.bootstrap(&fx.ev, &fresh)));
        let one = fx.ev.mod_switch_to_level(&fresh, 1).unwrap();
        assert!(inv(fx.bs.bootstrap(&fx.ev, &one)));
        // Wrong scale at level 0.
        let low = fx.ev.mod_switch_to_level(&fresh, 0).unwrap();
        let wrong = Ciphertext::new(low.polys().to_vec(), 2.0 * fx.params.scale()).unwrap();
        assert_eq!(
            fx.bs.bootstrap(&fx.ev, &wrong).unwrap_err(),
            HekitError::ScaleMismatch
        );
        let near = Ciphertext::new(low.polys().to_vec(), fx.params.scale() * (1.0 + 1e-15));
        assert_eq!(
            fx.bs.bootstrap(&fx.ev, &near.unwrap()).unwrap_err(),
            HekitError::ScaleMismatch
        );
        assert_eq!(fx.ev.counter(), before);
        // A 3-component ciphertext.
        let sq = fx.ev.mul_no_relin(&one, &one).unwrap();
        let sq = fx.ev.mod_switch_to_level(&sq, 0).unwrap();
        let sq = Ciphertext::new(sq.into_polys(), fx.params.scale()).unwrap();
        assert_eq!(sq.len(), 3);
        fx.ev.reset_counter();
        let before = fx.ev.counter();
        assert!(inv(fx.bs.bootstrap(&fx.ev, &sq)));
        // No relinearization key; an evaluator of other parameters.
        let no_rk = Evaluator::new(Arc::clone(&fx.params), None, None);
        assert!(inv(fx.bs.bootstrap(&no_rk, &low)));
        let other = Evaluator::new(Arc::new(mini()), None, None);
        assert!(inv(fx.bs.bootstrap(&other, &low)));
        assert_eq!(fx.ev.counter(), before);
        assert_eq!(no_rk.counter(), OpCounter::default());

        // Construction: not full packing, invalid parameters, foreign encoder.
        let p = Arc::clone(&fx.params);
        let mut kg = KeyGenerator::new(Arc::clone(&p), rng(5)).unwrap();
        let half = BootParams {
            log_slots: 8,
            ..mini_bp()
        };
        let inv_new = |e: Result<Bootstrapper>| matches!(e, Err(HekitError::InvalidParams(_)));
        assert!(inv_new(Bootstrapper::new(
            Arc::clone(&p),
            half,
            &fx.enc,
            &mut kg
        )));
        let bad_k = BootParams {
            k_intervals: 5,
            ..mini_bp()
        };
        assert!(inv_new(Bootstrapper::new(
            Arc::clone(&p),
            bad_k,
            &fx.enc,
            &mut kg
        )));
        let foreign = Encoder::new(Arc::new(mini()));
        assert!(inv_new(Bootstrapper::new(
            Arc::clone(&p),
            mini_bp(),
            &foreign,
            &mut kg
        )));
    }

    /// `x` with the bits of `i` reversed (`bits` bits).
    fn bitrev(i: usize, bits: u32) -> usize {
        i.reverse_bits() >> (usize::BITS - bits)
    }

    /// The centered value of coefficient `i` from limbs 0 and 1 of a plaintext (exact for
    /// `|v| < q_0 q_1 / 2`).
    fn coeff2(params: &CkksParams, pt: &crate::Plaintext) -> Vec<f64> {
        let mut p = pt.poly().clone();
        p.to_coeff();
        let (q0, q1) = (params.q()[0] as i128, params.q()[1] as i128);
        let inv = modinv(q0 % q1, q1);
        (0..params.n())
            .map(|i| {
                let a = p.limbs()[0][i] as i128;
                let b = p.limbs()[1][i] as i128;
                let t = ((b - a).rem_euclid(q1) * inv).rem_euclid(q1);
                let v = a + q0 * t;
                let big = q0 * q1;
                (if v > big / 2 { v - big } else { v }) as f64
            })
            .collect()
    }

    fn modinv(a: i128, m: i128) -> i128 {
        let (mut t, mut nt, mut r, mut nr) = (0i128, 1i128, m, a.rem_euclid(m));
        while nr != 0 {
            let q = r / nr;
            (t, nt) = (nt, t - q * nt);
            (r, nr) = (nr, r - q * nr);
        }
        t.rem_euclid(m)
    }

    fn stats(e: &[f64]) -> (f64, f64) {
        let max = e.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let rms = (e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64).sqrt();
        (max, rms)
    }

    /// The per-stage error breakdown of one bootstrap of `m` (printed): the mod-raise noise
    /// and overflow, the CoeffToSlot error in `x = t / q_0` units, the EvalMod error, and the
    /// final slot precision. `bits` is the `docs/bootstrap.md` §6 model
    /// `-log2(4.2 sqrt(N/2) (q_0 / Delta) sigma)` for an `x`-unit rms `sigma`.
    fn breakdown(fx: &Fx, m: &[f64], r: &mut Rng) -> f64 {
        let p = &fx.params;
        let (n, q0, delta) = (p.n(), p.q()[0] as f64, p.scale());
        let k = fx.bs.boot_params().k_intervals as f64;
        let bits = |sigma: f64| -(4.2 * (n as f64 / 2.0).sqrt() * (q0 / delta) * sigma).log2();
        let ct = fx.level0(m, r);
        // Delta c_j, exactly as encoded.
        let dc = coeff2(p, &fx.enc.encode(m, delta, 1).unwrap());
        let inner = Evaluator::with_shared_keys(
            Arc::clone(p),
            fx.ev.relin_key_shared().cloned(),
            Some(Arc::clone(&fx.bs.keys.galois)),
        );
        let mut trace = Vec::new();
        let t = std::time::Instant::now();
        let out = fx.bs.pipeline(&inner, &ct, Some(&mut trace)).unwrap();
        eprintln!("traced bootstrap: {:.1} s", t.elapsed().as_secs_f64());
        // 1. Mod-raise: t = Delta c + q_0 I + e.
        let raised = fx.dec.decrypt(&trace[0]);
        let tv = coeff2(p, &raised);
        let i: Vec<f64> = tv
            .iter()
            .zip(&dc)
            .map(|(t, d)| ((t - d) / q0).round())
            .collect();
        let e_raise: Vec<f64> = (0..n).map(|j| tv[j] - dc[j] - q0 * i[j]).collect();
        let (mx, rms) = stats(&e_raise);
        let max_i = i.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        eprintln!(
            "  mod_raise: max |I| = {max_i}, coefficient noise max {mx:.1} rms {rms:.1} \
             (x units rms 2^{:.2}; {:.1} bits alone)",
            (rms / q0).log2(),
            bits(rms / q0)
        );
        // x_j as encoded (without the raise noise), in slot order of CoeffToSlot.
        let half = n / 2;
        let lb = half.trailing_zeros();
        let x_true = |j: usize, second: bool| {
            let c = bitrev(j, lb) + if second { half } else { 0 };
            i[c] + dc[c] / q0
        };
        // 2. CoeffToSlot: slots y = x / K.
        let mut e_cts = Vec::with_capacity(n);
        for (h, ct) in trace[1..3].iter().enumerate() {
            let y = fx.decrypt(ct);
            e_cts.extend((0..half).map(|j| k * y[j] - x_true(j, h == 1)));
        }
        let (mx, rms) = stats(&e_cts);
        eprintln!(
            "  after CoeffToSlot (level {}, 2^{:.2}): x error max 2^{:.2} rms 2^{:.2} \
             ({:.1} bits alone, raise included)",
            trace[1].level(),
            trace[1].scale().log2(),
            mx.log2(),
            rms.log2(),
            bits(rms)
        );
        // 3. EvalMod: sin(2 pi x) against the true x, in x units (divide by 2 pi).
        let mut e_em = Vec::with_capacity(n);
        let mut e_mod = Vec::with_capacity(n);
        for (h, ct) in trace[3..5].iter().enumerate() {
            let v = fx.decrypt(ct);
            for (j, vj) in v.iter().enumerate().take(half) {
                let x = x_true(j, h == 1);
                e_em.push((vj - (2.0 * PI * x).sin()) / (2.0 * PI));
                e_mod.push(vj / (2.0 * PI) - (x - x.round()));
            }
        }
        let (mx, rms) = stats(&e_em);
        let (mx2, rms2) = stats(&e_mod);
        eprintln!(
            "  after EvalMod (level {}): vs sin(2 pi x): max 2^{:.2} rms 2^{:.2} ({:.1} bits \
             alone); vs x mod 1: max 2^{:.2} rms 2^{:.2} ({:.1} bits alone)",
            trace[3].level(),
            mx.log2(),
            rms.log2(),
            bits(rms),
            mx2.log2(),
            rms2.log2(),
            bits(rms2)
        );
        // 4. SlotToCoeff: the slots.
        let got = fx.decrypt(&out);
        let e_out: Vec<f64> = got.iter().zip(m).map(|(a, b)| a - b).collect();
        let (mx, rms) = stats(&e_out);
        let prec = precision_bits(m, &got);
        eprintln!(
            "  after SlotToCoeff (level {}, scale 2^{:.2}): slot error max 2^{:.2} rms 2^{:.2}: \
             precision {prec:.2} bits",
            out.level(),
            out.scale().log2(),
            mx.log2(),
            rms.log2()
        );
        prec
    }

    /// The breakdown on the miniature (fast): the stages are consistent with the precision.
    #[test]
    fn mini_breakdown() {
        let fx = fixture(mini(), mini_bp(), 6);
        let mut r = rng(7);
        let m = uniform(&mut r, fx.params.slots(), 1.0);
        let bits = breakdown(&fx, &m, &mut r);
        assert!(bits >= 16.0, "{bits}");
    }

    /// The brief's gate on an n16_boot chain (`n16_boot` or `n16_boot_sparse`): random `m` in
    /// [-1, 1] on all 2^15 slots, bootstrapped from level 0, comes back with
    /// `precision_bits >= 16` at level `levels_after_boot` = 16 and exactly `Delta`, with
    /// exactly the planned counts (`key_switches_per_boot`) and `bootstraps == 1`.
    fn n16_boot_gate(name: &str, params: CkksParams, bp: BootParams, model: f64, seed: u8) {
        let fx = fixture(params, bp, seed);
        assert_eq!(fx.bs.output_level(), 16);
        assert_eq!(fx.bs.keys().rotation_steps().len(), 32);
        let mut r = rng(seed + 1);
        let m = uniform(&mut r, fx.params.slots(), 1.0);
        let (bits, secs) = fx.run(&m, &mut r);
        let k = fx.bs.cost_per_boot();
        eprintln!(
            "{name} bootstrap: {bits:.2} bits (model {model}), {secs:.1} s on this thread, \
             level {} ; key switches per boot {} = {} rotations (incl. 2 conjugations) + {} \
             relins + {} SSE; {} products, {} plaintext/constant products, {} rescales; keys \
             {} Galois ({:.3} GB) and SSE: {:.2} GB",
            fx.bs.output_level(),
            fx.bs.key_switches_per_boot(),
            k.rotations,
            k.relins,
            fx.bs.sse_switches_per_boot(),
            k.ct_ct_tensor,
            k.ct_pt_mults,
            k.rescales,
            fx.bs.keys().galois().len(),
            ksk_bytes(fx.bs.keys().galois().get(1).unwrap().key()) as f64 / 1e9,
            fx.bs.keys().bytes() as f64 / 1e9
        );
        assert!(bits >= 16.0, "{name}: {bits}");
    }

    /// **Heavy (N = 2^16; about 13 GB: 33 Galois keys, the SSE and relinearization keys and
    /// the encoded diagonals).** The gate ([`n16_boot_gate`]) on n16_boot, dense main key.
    /// The model predicts 17.4 bits.
    #[test]
    #[ignore = "heavy: n16_boot keys (N = 2^16, ~13 GB); run with `-- --ignored n16_boot_keys`, see docs/bootstrap.md §11.6"]
    fn n16_boot_keys_bootstrap() {
        n16_boot_gate(
            "n16_boot",
            CkksParams::n16_boot(),
            BootParams::default_n16(),
            17.4,
            38,
        );
    }

    /// **Heavy (as above).** The same gate on `n16_boot_sparse` (main-key weight h = 192,
    /// ruling R-M3-10) with [`BootParams::sparse_n16`]. SSE with h' = 32 still runs the
    /// mod-raise, so the EvalMod constants (K = 13) apply unchanged; the smaller main key only
    /// lowers the rescale and key-switch rounding terms (model 21.0 bits).
    #[test]
    #[ignore = "heavy: n16_boot keys (N = 2^16, ~13 GB); run with `-- --ignored n16_boot_keys`, see docs/bootstrap.md §11.6"]
    fn n16_boot_keys_bootstrap_sparse() {
        let params = CkksParams::n16_boot_sparse();
        assert_eq!(params.hamming_weight(), Some(192));
        n16_boot_gate(
            "n16_boot_sparse",
            params,
            BootParams::sparse_n16(),
            21.0,
            38,
        );
    }

    /// Observation (ignored; run with `--ignored`): n16_boot at `|m| = 1` (random signs),
    /// with the per-stage error breakdown. Prints only.
    #[test]
    #[ignore = "observation, heavy: n16_boot keys (N = 2^16, ~13 GB); run with `-- --ignored n16_boot_keys`, see docs/bootstrap.md §11.6"]
    fn n16_boot_keys_observe_m1() {
        let fx = fixture(CkksParams::n16_boot(), BootParams::default_n16(), 40);
        let mut r = rng(41);
        let m = signs(&mut r, fx.params.slots(), 1.0);
        let bits = breakdown(&fx, &m, &mut r);
        eprintln!("n16_boot |m| = 1: {bits:.2} bits");
        let u = uniform(&mut r, fx.params.slots(), 1.0);
        let bits = breakdown(&fx, &u, &mut r);
        eprintln!("n16_boot m uniform in [-1, 1]: {bits:.2} bits");
    }

    /// Observation (ignored): n16_boot at `|m| = 4` (random signs), outside the `|m| <= 1`
    /// precondition: the sine-versus-modulus term grows quadratically. Prints only.
    #[test]
    #[ignore = "observation, heavy: n16_boot keys (N = 2^16, ~13 GB); run with `-- --ignored n16_boot_keys`, see docs/bootstrap.md §11.6"]
    fn n16_boot_keys_observe_m4() {
        let fx = fixture(CkksParams::n16_boot(), BootParams::default_n16(), 42);
        let mut r = rng(43);
        let m = signs(&mut r, fx.params.slots(), 4.0);
        let bits = breakdown(&fx, &m, &mut r);
        eprintln!("n16_boot |m| = 4: {bits:.2} bits (absolute error)");
    }
}
