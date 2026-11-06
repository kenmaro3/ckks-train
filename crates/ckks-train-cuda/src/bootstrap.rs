//! GPU bootstrap orchestration and CPU-testable sparse linear transforms.
//! Ciphertexts stay on the device; host work encodes public constants and prepares keys.
//! GPU precision, parity and timing require the ignored hardware tests.

use ckks_train_ckks::bootstrap::evalmod::eval_mod;
use ckks_train_ckks::bootstrap::poly::BootstrapArithmetic;
use ckks_train_ckks::bootstrap::{BootParams, Bootstrapper, LinearTransform};
use ckks_train_ckks::linalg::DiagonalEntry;
use ckks_train_ckks::{CkksParams, Diagonals, HeOps, Plaintext};
use ckks_train_core::{HekitError, Result, RnsPoly};
use std::sync::Arc;

fn invalid(message: impl Into<String>) -> HekitError {
    HekitError::InvalidParams(message.into())
}

struct Stage<P> {
    diags: Diagonals<Option<P>>,
    second: Option<Diagonals<Option<P>>>,
    level: usize,
    scale: f64,
}

struct BootstrapPlan<E: HeOps> {
    params: Arc<CkksParams>,
    bp: BootParams,
    ctos: Vec<Stage<E::Pt>>,
    stoc: Vec<Stage<E::Pt>>,
    monomial: E::Pt,
}

impl<E: BootstrapArithmetic> BootstrapPlan<E> {
    fn upload(ev: &E, host: &Bootstrapper) -> Result<Self> {
        if !Arc::ptr_eq(ev.params(), host.params()) {
            return Err(invalid("bootstrap parameters must share the same Arc"));
        }
        let upload = |s: &LinearTransform| -> Result<Stage<E::Pt>> {
            let map = |d: &Diagonals<Option<Plaintext>>| {
                d.try_map(|p| p.as_ref().map(|p| ev.upload_pt(p)).transpose())
            };
            Ok(Stage {
                diags: map(&s.diags)?,
                second: s.second.as_ref().map(map).transpose()?,
                level: s.level,
                scale: s.scale,
            })
        };
        let params = ev.params();
        let level = host
            .coeff_to_slot_stages()
            .last()
            .ok_or_else(|| invalid("no CtS stages"))?
            .level
            - 1;
        let limbs = (0..=level)
            .map(|_| {
                let mut l = vec![0; params.n()];
                l[params.n() / 2] = 1;
                l
            })
            .collect();
        let mut mono = RnsPoly::from_limbs(params.basis_q(), limbs, false)?;
        mono.to_ntt();
        Ok(Self {
            params: params.clone(),
            bp: host.boot_params().clone(),
            ctos: host
                .coeff_to_slot_stages()
                .iter()
                .map(upload)
                .collect::<Result<_>>()?,
            stoc: host
                .slot_to_coeff_stages()
                .iter()
                .map(upload)
                .collect::<Result<_>>()?,
            monomial: ev.upload_pt(&Plaintext::new(mono, 1.0)?)?,
        })
    }

    fn check_input(&self, ev: &E, ct: &E::Ct) -> Result<()> {
        if !Arc::ptr_eq(ev.params(), &self.params) {
            return Err(invalid("bootstrap evaluator over other parameters"));
        }
        ev.check_boot_input(ct)?;
        if ct.level() != 0 {
            return Err(invalid("bootstrap input must be at level 0"));
        }
        if ct.scale() != self.params.scale() {
            return Err(HekitError::ScaleMismatch);
        }
        Ok(())
    }

    fn pipeline(&self, ev: &E, raised: E::Ct) -> Result<E::Ct> {
        let mut x = ev.boot_relabel(raised, self.params.q()[0] as f64)?;
        for stage in &self.ctos {
            let conjugate = stage
                .second
                .as_ref()
                .map(|_| ev.conjugate(&x))
                .transpose()?;
            x = apply_stage(ev, stage, &x, conjugate.as_ref())?;
        }
        let conj = ev.conjugate(&x)?;
        let re = ev.add(&x, &conj)?;
        // Exact monomial multiply is not an HE plaintext product in CPU accounting.
        let mono_ev = ev.fork();
        let im = mono_ev.mul_plain(&ev.sub(&conj, &x)?, &self.monomial)?;
        let expected = self
            .params
            .scale_at(self.params.max_level() - self.bp.ctos_levels);
        let convert = |ct: E::Ct| -> Result<E::Ct> {
            let ct = eval_mod(ev, &ct, &self.bp)?;
            if ct.scale() != expected {
                return Err(HekitError::ScaleMismatch);
            }
            let ct = ev.mod_switch_to_level(&ct, self.stoc[0].level)?;
            ev.boot_relabel(
                ct,
                expected * self.params.scale() / self.params.q()[0] as f64,
            )
        };
        let (re, im) = (convert(re)?, convert(im)?);
        let mut x = apply_stage(ev, &self.stoc[0], &re, Some(&im))?;
        for stage in &self.stoc[1..] {
            x = apply_stage(ev, stage, &x, None)?;
        }
        Ok(x)
    }
}

// Treat work and asynchronous device completion as one transaction. A failed launch,
// allocation or final synchronization never publishes partial counts.
fn bootstrap_transaction<E: HeOps>(
    caller: &E,
    local: &E,
    work: impl FnOnce() -> Result<E::Ct>,
    synchronize: impl FnOnce() -> Result<()>,
) -> Result<E::Ct> {
    let out = work()?;
    synchronize()?;
    let mut done = local.counter();
    done.bootstraps += 1;
    caller.add_counts(&done);
    Ok(out)
}

fn apply_stage<E: BootstrapArithmetic>(
    ev: &E,
    stage: &Stage<E::Pt>,
    x: &E::Ct,
    x2: Option<&E::Ct>,
) -> Result<E::Ct> {
    let mut terms = vec![(&stage.diags, x)];
    if let (Some(d), Some(x)) = (&stage.second, x2) {
        terms.push((d, x));
    }
    let y = sparse_sum(ev, &terms)?;
    let label = ev.exact_product_scale(stage.level, stage.scale)?;
    if ((y.scale() - label) / label).abs() > 1e-15 {
        return Err(HekitError::ScaleMismatch);
    }
    ev.rescale(&ev.boot_relabel(y, label)?)
}

// Identical sparse shared-giant operation order to CPU bootstrap BSGS. Bootstrap always
// uses optimized flags, as the CPU Bootstrapper's private evaluator does.
type SparseTerm<'a, E> = (
    &'a Diagonals<Option<<E as HeOps>::Pt>>,
    &'a <E as HeOps>::Ct,
);

fn sparse_sum<E: HeOps>(ev: &E, terms: &[SparseTerm<'_, E>]) -> Result<E::Ct> {
    let first = terms[0].0;
    let (baby, giant, stride, offset) =
        (first.baby(), first.giant(), first.stride(), first.offset());
    let mut xs = Vec::new();
    for &(diag, x) in terms {
        let used: Vec<usize> = (0..baby)
            .filter(|&j| (0..giant).any(|i| diag.entries()[i * baby + j].is_some()))
            .collect();
        let steps: Vec<i32> = used.iter().map(|&j| (j * stride) as i32).collect();
        let rotated = ev.hoisted_rotate(x, &steps)?;
        let mut row = vec![None; baby];
        for (j, ct) in used.into_iter().zip(rotated) {
            row[j] = Some(ct);
        }
        xs.push(row);
    }
    let mut acc = None;
    for i in 0..giant {
        let mut inner = None;
        for (&(d, _), xr) in terms.iter().zip(&xs) {
            for (w, x) in d.entries()[i * baby..(i + 1) * baby].iter().zip(xr) {
                if let Some(w) = w {
                    let x = x
                        .as_ref()
                        .ok_or_else(|| invalid("missing bootstrap baby rotation"))?;
                    let t = ev.mul_plain(x, w)?;
                    inner = Some(match inner {
                        None => t,
                        Some(s) => ev.add(&s, &t)?,
                    });
                }
            }
        }
        if let Some(mut block) = inner {
            let shift = (offset + (baby * i) as i64) * stride as i64;
            if shift != 0 {
                block = ev.rotate(&block, shift as i32)?;
            }
            acc = Some(match acc {
                None => block,
                Some(s) => ev.add(&s, &block)?,
            });
        }
    }
    acc.ok_or_else(|| invalid("bootstrap stage has no diagonals"))
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::{CudaCiphertext, CudaEvaluator, DeviceKsk, KskPurpose};
    use ckks_train_ckks::bootstrap::poly::rounded_constant;
    use ckks_train_ckks::{BootstrapOps, OpCounter};

    /// Device-resident CKKS bootstrap prepared from CPU-encoded public stages and keys.
    ///
    /// `new` uploads bootstrap keys/diagonals once and shares the evaluator's relin key.
    /// No ciphertext is downloaded during a bootstrap. Input slots must be in [-1,1],
    /// at level zero with the exact preset scale. GPU numerical parity and performance
    /// are unmeasured until the ignored hardware tests are run.
    pub struct CudaBootstrapper<'d> {
        evaluator: CudaEvaluator<'d>,
        plan: BootstrapPlan<CudaEvaluator<'d>>,
        sse: Option<(DeviceKsk<'d>, DeviceKsk<'d>)>,
        cost: OpCounter,
        output_level: usize,
    }

    impl<'d> CudaBootstrapper<'d> {
        /// Upload existing CPU bootstrap material, using the same parameter Arc and a
        /// full-level relin key. All device allocations are fallible (`Device` on OOM).
        pub fn new(ev: &CudaEvaluator<'d>, host: &Bootstrapper) -> Result<Self> {
            if !Arc::ptr_eq(ev.params(), host.params()) {
                return Err(invalid("bootstrap parameters must share the same Arc"));
            }
            let evaluator = ev.with_bootstrap_galois(host.keys().galois())?;
            let sse = host
                .keys()
                .sse()
                .map(|s| ev.device().upload_sse_keys(s))
                .transpose()?;
            let plan = BootstrapPlan::upload(&evaluator, host)?;
            ev.device().sync()?;
            Ok(Self {
                evaluator,
                plan,
                sse,
                cost: host.cost_per_boot(),
                output_level: host.output_level(),
            })
        }

        /// Bootstrap and record successful work on this bootstrapper's own counter.
        /// A device synchronization precedes committing any counts.
        pub fn bootstrap(&self, ct: &CudaCiphertext<'d>) -> Result<CudaCiphertext<'d>> {
            self.execute(&self.evaluator, ct)
        }

        /// The output level (16 for n16_boot).
        pub fn output_level(&self) -> usize {
            self.output_level
        }

        /// The planned CPU-equivalent operation counts per bootstrap.
        pub fn cost_per_boot(&self) -> OpCounter {
            self.cost.clone()
        }

        /// Counts of successful calls to the inherent `bootstrap`; trait calls instead
        /// merge into the passed evaluator, exactly once.
        pub fn counter(&self) -> OpCounter {
            self.evaluator.counter()
        }

        fn execute(
            &self,
            caller: &CudaEvaluator<'d>,
            ct: &CudaCiphertext<'d>,
        ) -> Result<CudaCiphertext<'d>> {
            self.plan.check_input(caller, ct)?;
            let local = self.evaluator.bootstrap_fork(caller)?;
            bootstrap_transaction(
                caller,
                &local,
                || {
                    let raised = self.mod_raise(&local, ct)?;
                    self.plan.pipeline(&local, raised)
                },
                || local.device().sync(),
            )
        }

        fn mod_raise(
            &self,
            ev: &CudaEvaluator<'d>,
            ct: &CudaCiphertext<'d>,
        ) -> Result<CudaCiphertext<'d>> {
            let dev = ev.device();
            let switch =
                |ct: &CudaCiphertext<'d>, key: &DeviceKsk<'d>| -> Result<CudaCiphertext<'d>> {
                    let (b, a) = dev.key_switch(&ct.polys()[1], key)?;
                    CudaCiphertext::new(vec![dev.add(&ct.polys()[0], &b)?, a], ct.scale())
                };
            let sparse = match &self.sse {
                Some((s, d)) => {
                    if s.purpose() != KskPurpose::SseSparse
                        || s.max_level() != 0
                        || d.purpose() != KskPurpose::SseDense
                        || d.max_level() != ev.params().max_level()
                    {
                        return Err(invalid("bootstrap SSE key purpose or level mismatch"));
                    }
                    switch(ct, s)?
                }
                None => ct.clone(),
            };
            let primes: Arc<[u32]> = (0..=ev.params().max_level() as u32)
                .collect::<Vec<_>>()
                .into();
            let components = sparse
                .polys()
                .iter()
                .map(|p| {
                    let out = crate::basis::raise_level0(
                        &crate::arith::Dev(dev),
                        p.limbs(),
                        ev.params().max_level(),
                    )?;
                    Ok(dev.new_poly(out, primes.clone(), true))
                })
                .collect::<Result<Vec<_>>>()?;
            let raised = CudaCiphertext::new(components, ct.scale())?;
            match &self.sse {
                None => Ok(raised),
                Some((_, d)) => {
                    let out = switch(&raised, d)?;
                    ev.add_counts(&OpCounter {
                        sse_switches: 2,
                        ..OpCounter::default()
                    });
                    Ok(out)
                }
            }
        }
    }

    impl<'d> BootstrapOps<CudaEvaluator<'d>> for CudaBootstrapper<'d> {
        fn bootstrap(
            &self,
            ev: &CudaEvaluator<'d>,
            ct: &CudaCiphertext<'d>,
        ) -> Result<CudaCiphertext<'d>> {
            self.execute(ev, ct)
        }
        fn output_level(&self) -> usize {
            self.output_level
        }
    }

    // Public integer constants only: CPU prepares their reduced residues, CUDA performs
    // every operation involving ciphertext data. No encrypted value is read by the host.
    fn constant(ev: &CudaEvaluator<'_>, level: usize, k: i128) -> Result<RnsPoly> {
        let p = ev.params();
        let limbs = p.q()[..=level]
            .iter()
            .map(|&q| vec![k.rem_euclid(q as i128) as u64; p.n()])
            .collect();
        RnsPoly::from_limbs(p.basis_q(), limbs, true)
    }

    impl<'d> BootstrapArithmetic for CudaEvaluator<'d> {
        fn check_boot_input(&self, ct: &CudaCiphertext<'d>) -> Result<()> {
            if !self.owns(ct) || ct.len() != 2 {
                return Err(invalid("bootstrap needs an owned two-component ciphertext"));
            }
            self.relin_key(ct.level())?;
            Ok(())
        }
        fn boot_mul_const(
            &self,
            ct: &CudaCiphertext<'d>,
            value: f64,
            scale: f64,
        ) -> Result<CudaCiphertext<'d>> {
            if !self.owns(ct) {
                return Err(invalid("foreign bootstrap ciphertext"));
            }
            let ratio = scale / ct.scale();
            if ratio.is_nan() || ratio < ckks_train_ckks::evaluator::MIN_PLAIN_SCALE {
                return Err(invalid("constant scale below 2^20"));
            }
            let k = rounded_constant(value * ratio)?;
            let scalar = self.device().upload(&constant(self, ct.level(), k)?)?;
            let c = ct
                .polys()
                .iter()
                .map(|p| self.device().mul_pointwise(p, &scalar))
                .collect::<Result<_>>()?;
            let out = CudaCiphertext::new(c, scale)?;
            self.add_counts(&OpCounter {
                ct_pt_mults: 1,
                ..OpCounter::default()
            });
            Ok(out)
        }
        fn boot_add_const(
            &self,
            ct: &CudaCiphertext<'d>,
            value: f64,
        ) -> Result<CudaCiphertext<'d>> {
            if !self.owns(ct) {
                return Err(invalid("foreign bootstrap ciphertext"));
            }
            let k = rounded_constant(value * ct.scale())?;
            let scalar = self.device().upload(&constant(self, ct.level(), k)?)?;
            let mut polys = ct.polys().to_vec();
            polys[0] = self.device().add(&polys[0], &scalar)?;
            CudaCiphertext::new(polys, ct.scale())
        }
        fn boot_zero(&self, level: usize, scale: f64) -> Result<CudaCiphertext<'d>> {
            if level > self.params().max_level() {
                return Err(invalid("zero ciphertext level above chain"));
            }
            let z = self.device().upload(&constant(self, level, 0)?)?;
            CudaCiphertext::new(vec![z.clone(), z], scale)
        }
        fn boot_relabel(&self, ct: CudaCiphertext<'d>, scale: f64) -> Result<CudaCiphertext<'d>> {
            if !self.owns(&ct) {
                return Err(invalid("foreign bootstrap ciphertext"));
            }
            ct.with_scale(scale)
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::CudaBootstrapper;

#[cfg(test)]
mod tests {
    use super::*;
    use ckks_train_ckks::encrypt::{Decryptor, Encryptor};
    use ckks_train_ckks::{Encoder, Evaluator, KeyGenerator};
    use ckks_train_core::sampler::Rng;

    #[test]
    fn bootstrap_counters_commit_only_after_device_completion() {
        use ckks_train_ckks::OpCounter;
        let p = Arc::new(CkksParams::new(5, &[54, 42], &[60], 42, 2).unwrap());
        let ev = Evaluator::new(p.clone(), None, None);
        let local = ev.fork();
        let ct = local.boot_zero(0, p.scale()).unwrap();
        local.add_counts(&OpCounter {
            rotations: 7,
            relins: 3,
            ..OpCounter::default()
        });
        let fail = || Err(HekitError::Device("injected failure".into()));
        assert!(bootstrap_transaction(&ev, &local, fail, || Ok(())).is_err());
        assert_eq!(ev.counter(), OpCounter::default());
        assert!(
            bootstrap_transaction(
                &ev,
                &local,
                || Ok(ct.clone()),
                || Err(HekitError::Device("injected sync failure".into()))
            )
            .is_err()
        );
        assert_eq!(ev.counter(), OpCounter::default());
        bootstrap_transaction(&ev, &local, || Ok(ct), || Ok(())).unwrap();
        assert_eq!(
            ev.counter(),
            OpCounter {
                rotations: 7,
                relins: 3,
                bootstraps: 1,
                ..OpCounter::default()
            }
        );
    }

    #[test]
    fn centered_raise_kernel_replay_at_rounding_boundary() {
        let p = CkksParams::new(5, &[54, 42, 58], &[60], 42, 3).unwrap();
        let q = p.q()[0];
        let edge = [0, 1, (q - 1) / 2, (q + 1) / 2, q - 2, q - 1];
        let limb = (0..p.n()).map(|i| edge[i % edge.len()]).collect();
        let mut poly = RnsPoly::from_limbs(p.basis_q(), vec![limb], false).unwrap();
        poly.to_ntt();
        let b = crate::emu_arith::EmuBackend::new(&p);
        let actual = crate::basis::raise_level0(&b, &poly.limbs().concat(), p.max_level()).unwrap();
        assert_eq!(
            actual,
            poly.raise_from_level0(p.max_level()).limbs().concat()
        );
    }

    #[test]
    fn sse_key_shape_and_kernel_replay_match_cpu() {
        use crate::ks::{KeyRef, Ks, KskShape, ksk_limbs, sse_sparse_limbs};
        let p = Arc::new(CkksParams::new(6, &[54, 42, 58, 56], &[60, 60], 42, 2).unwrap());
        let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([52; 32])).unwrap();
        let keys = kg.sse_keys(32).unwrap();
        let b = crate::emu_arith::EmuBackend::new(&p);
        let low = KskShape::new(&p, 0).unwrap();
        let high = KskShape::new(&p, p.max_level()).unwrap();
        assert!(ksk_limbs(&b.primes, &p, keys.to_sparse(), &low).is_err());
        assert!(sse_sparse_limbs(&p, keys.to_dense()).is_err());
        let foreign = CkksParams::new(6, &[54, 42, 58, 56], &[60, 60], 42, 2).unwrap();
        assert!(sse_sparse_limbs(&foreign, keys.to_sparse()).is_err());
        let sparse = sse_sparse_limbs(&p, keys.to_sparse()).unwrap().concat();
        let dense = ksk_limbs(&b.primes, &p, keys.to_dense(), &high)
            .unwrap()
            .concat();
        assert_eq!(sparse.len() * 8, low.device_bytes(p.n()));
        let ks = Ks {
            b: &b,
            params: &p,
            table: &b.primes,
        };
        let enc = Encoder::new(p.clone());
        let encryptor = Encryptor::new(p.clone(), kg.public_key());
        let ct = encryptor.encrypt(
            &enc.encode(&[0.3, -0.2], p.scale(), 0).unwrap(),
            &mut Rng::from_seed([53; 32]),
        );
        let ev = Evaluator::new(p.clone(), None, None);
        let expected = ev.mod_raise(&ct, &keys).unwrap();
        let (kb, ka) = ks
            .key_switch(
                &ct.polys()[1].limbs().concat(),
                0,
                KeyRef {
                    buf: &sparse,
                    shape: &low,
                },
            )
            .unwrap();
        let make = |flat: Vec<u64>, level| {
            RnsPoly::from_limbs(
                p.basis_q(),
                flat.chunks(p.n())
                    .take(level + 1)
                    .map(|l| l.to_vec())
                    .collect(),
                true,
            )
            .unwrap()
        };
        let mut c0 = ct.polys()[0].clone();
        c0.add_assign(&make(kb, 0));
        let c0 = crate::basis::raise_level0(&b, &c0.limbs().concat(), p.max_level()).unwrap();
        let c1 = crate::basis::raise_level0(&b, &ka, p.max_level()).unwrap();
        let (kb, ka) = ks
            .key_switch(
                &c1,
                p.max_level(),
                KeyRef {
                    buf: &dense,
                    shape: &high,
                },
            )
            .unwrap();
        let mut c0 = make(c0, p.max_level());
        c0.add_assign(&make(kb, p.max_level()));
        assert_eq!(c0.limbs(), expected.polys()[0].limbs());
        assert_eq!(ka, expected.polys()[1].limbs().concat());
    }

    #[test]
    fn bootstrap_pipeline_cpu_replay_precision_bits_and_counts() {
        let mut bits = vec![54];
        bits.extend([42; 16]);
        bits.extend([34; 4]);
        bits.extend([58; 7]);
        bits.extend([56; 4]);
        let p = Arc::new(CkksParams::new(10, &bits, &[60; 4], 42, 8).unwrap());
        let mut bp = BootParams::default_n16();
        bp.log_slots = 9;
        let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([49; 32])).unwrap();
        let enc = Encoder::new(p.clone());
        let encryptor = Encryptor::new(p.clone(), kg.public_key());
        let dec = Decryptor::new(p.clone(), kg.secret_key());
        let bs = Bootstrapper::new(p.clone(), bp, &enc, &mut kg).unwrap();
        let rk = kg.relin_key();
        let ev = Evaluator::new(
            p.clone(),
            Some(rk.clone()),
            Some(bs.keys().galois().clone()),
        );
        let mut rng = Rng::from_seed([51; 32]);
        let v: Vec<f64> = (0..p.slots())
            .map(|_| 2.0 * ((rng.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0)
            .collect();
        let ct = encryptor.encrypt(&enc.encode(&v, p.scale(), 0).unwrap(), &mut rng);
        let expected = bs.bootstrap(&ev, &ct).unwrap();
        let counts = ev.counter();
        ev.reset_counter();
        let plan = BootstrapPlan::upload(&ev, &bs).unwrap();
        plan.check_input(&ev, &ct).unwrap();
        let high = ev.boot_zero(1, p.scale()).unwrap();
        assert!(matches!(
            plan.check_input(&ev, &high),
            Err(HekitError::InvalidParams(_))
        ));
        let wrong_scale = ev.boot_relabel(ct.clone(), p.scale() * 2.0).unwrap();
        assert!(matches!(
            plan.check_input(&ev, &wrong_scale),
            Err(HekitError::ScaleMismatch)
        ));
        let foreign = Evaluator::new(Arc::new((*p).clone()), Some(rk), None);
        assert!(matches!(
            plan.check_input(&foreign, &ct),
            Err(HekitError::InvalidParams(_))
        ));
        let missing_key = Evaluator::new(p.clone(), None, None);
        assert!(matches!(
            plan.check_input(&missing_key, &ct),
            Err(HekitError::InvalidParams(_))
        ));
        assert_eq!(ev.counter(), ckks_train_ckks::OpCounter::default());
        let raised = ev.mod_raise(&ct, bs.keys().sse().unwrap()).unwrap();
        let actual = plan.pipeline(&ev, raised).unwrap();
        let mut got = ev.counter();
        got.bootstraps += 1;
        assert_eq!(got, counts);
        assert_eq!(got, bs.cost_per_boot());
        assert_eq!(actual.scale(), expected.scale());
        assert_eq!(actual.level(), 16);
        for (a, b) in actual.polys().iter().zip(expected.polys()) {
            assert_eq!(a.limbs(), b.limbs());
        }
        let decoded = enc.decode(&dec.decrypt(&actual));
        let error = v
            .iter()
            .zip(decoded)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "CPU bootstrap pipeline replay precision {} bits",
            -error.log2()
        );
        assert!(error < 2f64.powi(-16));
    }
}
