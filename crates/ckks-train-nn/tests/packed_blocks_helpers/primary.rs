//! Registered actual-arithmetic programs: ignored and compile/list only until phase release.
use super::{fixture::*, *};
use ckks_train_ckks::{
    Decryptor, Encryptor, Evaluator, KeyGenerator,
    bootstrap::{BootParams, Bootstrapper},
};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::layout_reboot::GradRelin;
use std::time::Instant;
type Z = (f64, f64);
struct Context {
    p: Arc<CkksParams>,
    enc: Encoder,
    ev: Evaluator,
    encrypt: Encryptor,
    dec: Decryptor,
    bs: Option<Bootstrapper>,
}
impl Context {
    fn new(ni: bool) -> Self {
        let (q, a) = ckks_train_ckks::params::n16_boot_bits();
        let p = Arc::new(CkksParams::new(10, &q, &a, 42, 8).unwrap());
        let enc = Encoder::new(p.clone());
        let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([167; 32])).unwrap();
        let bp = BootParams {
            log_slots: 9,
            ..BootParams::default_n16()
        };
        let bs = ni.then(|| Bootstrapper::new(p.clone(), bp.clone(), &enc, &mut kg).unwrap());
        let steps = geometry(3).layout().galois_steps();
        assert_eq!(steps, [-2, -1, 1, 2, 4, 8, 16, 32, 64, 128, 256]);
        let ev = Evaluator::new(
            p.clone(),
            Some(kg.relin_key()),
            Some(kg.galois_keys(&steps)),
        );
        let encrypt = Encryptor::new(p.clone(), kg.public_key());
        let dec = Decryptor::new(p.clone(), kg.secret_key());
        println!(
            "PARAMS ni={ni} N={} scale_bits={} dnum={} q={:?} p={:?} bp={bp:?} held_out=167",
            p.n(),
            p.scale().to_bits(),
            p.dnum(),
            p.q(),
            p.p()
        );
        Self {
            p,
            enc,
            ev,
            encrypt,
            dec,
            bs,
        }
    }
    fn ct(&self, v: &[f64], seed: [u8; 32]) -> Ciphertext {
        self.encrypt.encrypt(
            &self.enc.encode(v, self.p.scale(), 16).unwrap(),
            &mut Rng::from_seed(seed),
        )
    }
    fn decode(&self, c: &Ciphertext) -> Vec<Z> {
        self.enc
            .decode_complex(&self.dec.decrypt(c))
            .iter()
            .map(|z| (z.re, z.im))
            .collect()
    }
    fn graph(&self, g: usize, mode: u8) -> PackedBlockGraph<Ciphertext> {
        let geometry = geometry(g);
        let stages = (0..g)
            .map(|s| {
                let weights = std::array::from_fn(|k| {
                    PackedWeights::new(
                        &self.ev,
                        geometry.layers()[2 * s + k].clone(),
                        self.ct(
                            &matrix(&W[k], (s + k) % 2 == 0),
                            seed(1, g, mode, 255, s as u8, 255, k as u8),
                        ),
                    )
                    .unwrap()
                });
                PackedStage::new(
                    &self.ev,
                    &geometry,
                    s,
                    weights,
                    None,
                    PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap(),
                )
                .unwrap()
            })
            .collect();
        PackedBlockGraph::new(&self.ev, geometry, stages).unwrap()
    }
    fn batch(
        &self,
        graph: &PackedBlockGraph<Ciphertext>,
        mode: u8,
        t: usize,
    ) -> PackedGraphBatch<Ciphertext> {
        let g = graph.stages().len();
        let geometry = graph.geometry();
        let records = (0..4)
            .map(|r| {
                let input = PackedVector::new(
                    &self.ev,
                    geometry.layers()[0].input(),
                    self.ct(
                        &vector(&X[r], true),
                        seed(1, g, mode, t as u8, 255, r as u8, 2),
                    ),
                )
                .unwrap();
                let targets = (0..g)
                    .map(|s| {
                        PackedVector::new(
                            &self.ev,
                            geometry.layers()[2 * s + 1].output(),
                            self.ct(
                                &vector(&Y[r], s % 2 == 0),
                                seed(1, g, mode, t as u8, s as u8, r as u8, 3),
                            ),
                        )
                        .unwrap()
                    })
                    .collect();
                PackedGraphRecord::new(r, input, targets, IDS[t].contains(&r))
            })
            .collect();
        let ids = IDS[t];
        PackedGraphBatch::new(
            &self.ev,
            graph,
            records,
            vec![
                PackedPartition::new(0, 2, ids.iter().filter(|&&i| i < 2).count()),
                PackedPartition::new(2, 4, ids.iter().filter(|&&i| i >= 2).count()),
            ],
        )
        .unwrap()
    }
    fn inference(
        &self,
        graph: &PackedBlockGraph<Ciphertext>,
        mode: u8,
    ) -> PackedForwardBatch<Ciphertext> {
        let g = graph.stages().len();
        let records = (0..3)
            .map(|r| {
                PackedForwardRecord::new(
                    r,
                    PackedVector::new(
                        &self.ev,
                        graph.geometry().layers()[0].input(),
                        self.ct(&vector(&X[r], true), seed(1, g, mode, 4, 255, r as u8, 2)),
                    )
                    .unwrap(),
                    true,
                )
            })
            .collect();
        PackedForwardBatch::new(
            &self.ev,
            graph,
            records,
            vec![PackedPartition::new(0, 2, 2), PackedPartition::new(2, 3, 1)],
        )
        .unwrap()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttemptPhase {
    Training,
    Inference,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttemptIdentity {
    pub ordinal: usize,
    pub graph: usize,
    pub placement: PackedNiPlacement,
    pub phase: AttemptPhase,
    pub update: Option<usize>,
    pub stage: usize,
    pub record: Option<usize>,
    pub role: PackedRefreshRole,
}
impl AttemptIdentity {
    pub fn bound(&self) -> f64 {
        match self.role {
            PackedRefreshRole::Weight(_) => 0.4,
            PackedRefreshRole::Velocity(_) => 0.1,
            PackedRefreshRole::Hidden { .. } => {
                if self.phase == AttemptPhase::Training {
                    0.5
                } else {
                    0.6
                }
            }
            PackedRefreshRole::Bias(_) => panic!("registered fixture has no bias bootstrap"),
        }
    }
    fn new(
        ordinal: usize,
        case: &Schedule,
        phase: AttemptPhase,
        update: Option<usize>,
        role: PackedRefreshRole,
    ) -> Self {
        let (stage, record) = match role {
            PackedRefreshRole::Hidden { stage, record } => (stage, Some(record as usize)),
            PackedRefreshRole::Weight(k)
            | PackedRefreshRole::Velocity(k)
            | PackedRefreshRole::Bias(k) => (k / 2, None),
        };
        Self {
            ordinal,
            graph: case.g,
            placement: case.placement,
            phase,
            update,
            stage,
            record,
            role,
        }
    }
}
pub fn frozen_attempts(cases: &[Schedule]) -> Vec<AttemptIdentity> {
    let mut out = vec![];
    for case in cases {
        for (t, plan) in case.updates.iter().enumerate() {
            for (s, child) in plan.children().iter().enumerate() {
                if case.placement == PackedNiPlacement::Aa {
                    for &id in IDS[t] {
                        out.push(AttemptIdentity::new(
                            out.len(),
                            case,
                            AttemptPhase::Training,
                            Some(t),
                            PackedRefreshRole::Hidden {
                                stage: s,
                                record: id as u64,
                            },
                        ));
                    }
                }
                if child.refresh_state() {
                    for role in [
                        PackedRefreshRole::Weight(2 * s),
                        PackedRefreshRole::Velocity(2 * s),
                        PackedRefreshRole::Weight(2 * s + 1),
                        PackedRefreshRole::Velocity(2 * s + 1),
                    ] {
                        out.push(AttemptIdentity::new(
                            out.len(),
                            case,
                            AttemptPhase::Training,
                            Some(t),
                            role,
                        ));
                    }
                }
                if s + 1 < case.g && case.placement != PackedNiPlacement::Aa {
                    for &id in IDS[t] {
                        out.push(AttemptIdentity::new(
                            out.len(),
                            case,
                            AttemptPhase::Training,
                            Some(t),
                            PackedRefreshRole::Hidden {
                                stage: s,
                                record: id as u64,
                            },
                        ));
                    }
                }
            }
        }
        for s in 0..case.g {
            if s + 1 < case.g || case.placement == PackedNiPlacement::Aa {
                for id in 0..3 {
                    out.push(AttemptIdentity::new(
                        out.len(),
                        case,
                        AttemptPhase::Inference,
                        None,
                        PackedRefreshRole::Hidden {
                            stage: s,
                            record: id,
                        },
                    ));
                }
            }
        }
    }
    out
}
pub fn attempt_range(identity: AttemptIdentity, slots: &[(f64, f64)]) -> (f64, f64, bool) {
    let bound = identity.bound();
    let finite = slots.len() == 512 && slots.iter().all(|z| z.0.is_finite() && z.1.is_finite());
    let range = if finite {
        slots.iter().map(|z| z.0.hypot(z.1)).fold(0.0_f64, f64::max)
    } else {
        f64::INFINITY
    };
    (bound, range, finite && range <= bound && range <= 1.0)
}
pub struct Attempt<C> {
    pub identity: AttemptIdentity,
    pub input: C,
    pub output: Option<C>,
    pub before: OpCounter,
    pub after: OpCounter,
    pub response_validated: Cell<bool>,
}
// Public-key-only recorder. Immutable role schedule is selected before invoking B.
// No secret, plaintext, range diagnostic or callback exists in this object.
pub struct RecordedBoot<'a, E: HeOps, B: PlannedBootstrap<E>> {
    bs: &'a B,
    schedule: Vec<AttemptIdentity>,
    pub attempts: RefCell<Vec<Attempt<E::Ct>>>,
}
impl<'a, E: HeOps, B: PlannedBootstrap<E>> RecordedBoot<'a, E, B> {
    pub fn new(bs: &'a B, schedule: Vec<AttemptIdentity>) -> Self {
        Self {
            bs,
            schedule,
            attempts: RefCell::new(vec![]),
        }
    }
}
impl<E: HeOps, B: PlannedBootstrap<E>> BootstrapOps<E> for RecordedBoot<'_, E, B> {
    fn bootstrap(&self, ev: &E, input: &E::Ct) -> Result<E::Ct> {
        let i = self.attempts.borrow().len();
        if i >= 372 {
            return Err(HekitError::InvalidParams(
                "D4c3 guard refuses call 373 before bootstrap".into(),
            ));
        }
        let identity = *self.schedule.get(i).ok_or_else(|| {
            HekitError::InvalidParams("no frozen identity for bootstrap attempt".into())
        })?;
        if identity.ordinal != i {
            return Err(HekitError::InvalidParams(
                "frozen attempt ordinal mismatch".into(),
            ));
        }
        let before = ev.counter();
        self.attempts.borrow_mut().push(Attempt {
            identity,
            input: input.clone(),
            output: None,
            before: before.clone(),
            after: before,
            response_validated: Cell::new(false),
        });
        let result = self.bs.bootstrap(ev, input);
        let mut attempts = self.attempts.borrow_mut();
        attempts[i].after = ev.counter();
        if let Ok(out) = &result {
            attempts[i].output = Some(out.clone());
        }
        result
    }
    fn output_level(&self) -> usize {
        self.bs.output_level()
    }
}
impl<E: HeOps, B: PlannedBootstrap<E>> PlannedBootstrap<E> for RecordedBoot<'_, E, B> {
    fn contract(&self) -> Result<PackedBootstrapContract> {
        self.bs.contract()
    }
    fn validate_binding(&self, ev: &E, c: &PackedBootstrapContract) -> Result<()> {
        self.bs.validate_binding(ev, c)
    }
}
// One shared external walk is exercised on controlled failures and used by both actual
// failure branches. Every available encrypted primitive/response remains distinguishable.
pub fn visit_forward_failure<C: ckks_train_ckks::HeCiphertext + Clone>(
    failure: &PackedGraphForwardFailure<C>,
    mut visit: impl FnMut(usize, usize, &str, &C),
) {
    for (s, child) in failure.children().iter().enumerate() {
        visit_observations(s, child.observations(), &mut visit);
    }
    if let Some(child) = failure.child_failure() {
        visit_observations(failure.stage(), child.observations(), &mut visit);
    }
    for h in failure.handoffs() {
        let (s, id) = (h.stage(), h.record());
        visit(s, id, "handoff_source", h.source().ciphertext());
        if let Some(out) = h.output() {
            visit(s, id, "handoff_output", out.ciphertext());
        }
        if let Some(request) = h.request() {
            visit(s, id, "handoff_request", request.ciphertext());
        }
        if let Some(response) = h.response() {
            visit(s, id, "handoff_response", response.ciphertext());
        }
        if let Some(raw) = h.raw_attempt() {
            visit(s, id, "raw_boot_input", raw.input());
            if let Some(out) = raw.output() {
                visit(s, id, "raw_boot_output", out);
            }
        }
        println!(
            "FAILED_HANDOFF stage={s} record={id} attempted={} completed={} validated={} counts={:?}",
            h.attempted(),
            h.completed(),
            h.validated(),
            h.counter_delta()
        );
    }
}
fn visit_observations<C: ckks_train_ckks::HeCiphertext>(
    stage: usize,
    observations: &[PackedForwardObservation<C>],
    visit: &mut impl FnMut(usize, usize, &str, &C),
) {
    for o in observations {
        let id = o.id();
        visit(stage, id, "input", o.input().ciphertext());
        for (kind, value) in [
            ("affine", o.affine()),
            ("z", o.preactivation()),
            ("hidden_pre", o.hidden_pre()),
            ("hidden_post", o.hidden_post()),
            ("head_affine", o.head_affine()),
            ("head_z", o.head_preactivation()),
            ("output", o.output()),
        ] {
            if let Some(v) = value {
                visit(stage, id, kind, v.ciphertext());
            }
        }
        if let Some(a) = o.bootstrap_attempt() {
            visit(stage, id, "raw_boot_input", a.input());
            if let Some(v) = a.output() {
                visit(stage, id, "raw_boot_output", v);
            }
            println!(
                "FAILED_RAW_BOOT stage={stage} record={id} role={:?} completed={} validated={} counts={:?}",
                a.role(),
                a.completed(),
                a.validated(),
                a.counter_delta()
            );
        }
        println!(
            "FAILED_RECORD stage={stage} record={id} counts={:?}",
            o.counter_delta()
        );
    }
}
struct ClientAttempt {
    role: PackedRefreshRole,
    input: Ciphertext,
    output: Option<Ciphertext>,
}
struct RealClient<'a> {
    c: &'a Context,
    g: usize,
    contract: PackedRefreshContract,
    attempts: Vec<ClientAttempt>,
}
impl PackedClientRefresh<Evaluator> for RealClient<'_> {
    fn output_contract(&self) -> &PackedRefreshContract {
        &self.contract
    }
    fn refresh(
        &mut self,
        request: &PackedRefreshRequest<Ciphertext>,
    ) -> Result<(PackedRefreshResponse<Ciphertext>, ClientRefreshTiming)> {
        let index = self.attempts.len();
        if index >= 116 {
            return Err(HekitError::InvalidParams(
                "D4c3 client cap 116 reached".into(),
            ));
        }
        self.attempts.push(ClientAttempt {
            role: request.role(),
            input: request.ciphertext().clone(),
            output: None,
        });
        let begin = Instant::now();
        let decoded = self.c.decode(request.ciphertext());
        let decoded_time = begin.elapsed();
        let mut packed = vec![0.; 512];
        match request.shape() {
            PackedRefreshShape::Matrix(layer) => {
                for o in 0..layer.output().logical_len() {
                    for i in 0..layer.input().logical_len() {
                        let s = if layer.orientation() == Orientation::Row {
                            4 * i + o
                        } else {
                            4 * o + i
                        };
                        packed[s] = decoded[s].0;
                    }
                }
            }
            PackedRefreshShape::Vector(spec) => {
                let expanded = spec.repr() == VectorRepr::Expanded;
                for k in 0..spec.logical_len() {
                    let slots = (0..512)
                        .filter(|&s| if expanded { s / 4 == k } else { s % 4 == k })
                        .collect::<Vec<_>>();
                    let mean =
                        slots.iter().map(|&s| decoded[s].0).sum::<f64>() / slots.len() as f64;
                    for s in slots {
                        packed[s] = mean;
                    }
                }
            }
        }
        let begin = Instant::now();
        let out = self.c.ct(&packed, client_seed(self.g, index as u64));
        let timing = ClientRefreshTiming {
            decrypt_decode: Some(decoded_time),
            canonical_repack_encrypt: Some(begin.elapsed()),
            ..Default::default()
        };
        self.attempts[index].output = Some(out.clone());
        Ok((
            PackedRefreshResponse::new(&self.c.ev, request, out)?,
            timing,
        ))
    }
}
struct Diagnostic<'a> {
    c: &'a Context,
    ok: bool,
    observed_inputs: Vec<(AttemptIdentity, Ciphertext)>,
}
impl<'a> Diagnostic<'a> {
    fn new(c: &'a Context) -> Self {
        Self {
            c,
            ok: true,
            observed_inputs: vec![],
        }
    }
    fn raw(&mut self, label: &str, ct: &Ciphertext) -> Vec<Z> {
        if !self.c.ev.owns(ct)
            || ct.len() != 2
            || ct.level() > self.c.p.max_level()
            || !ct.scale().is_finite()
            || ct.scale() <= 0.0
        {
            self.ok = false;
            println!(
                "UNDECODABLE_RAW {label} level={} scale_bits={} components={} owns_context={} encrypted_limbs={:?}",
                ct.level(),
                ct.scale().to_bits(),
                ct.len(),
                self.c.ev.owns(ct),
                ct.polys().iter().map(|p| p.limbs()).collect::<Vec<_>>()
            );
            return vec![];
        }
        let slots = self.c.decode(ct);
        self.ok &= slots.len() == 512 && slots.iter().all(|z| z.0.is_finite() && z.1.is_finite());
        println!(
            "SLOTS {label} context={:p} level={} scale_bits={} components={} values={:?}",
            Arc::as_ptr(&self.c.p),
            ct.level(),
            ct.scale().to_bits(),
            ct.len(),
            slots
                .iter()
                .map(|z| (z.0.to_bits(), z.1.to_bits()))
                .collect::<Vec<_>>()
        );
        slots
    }
    fn parity(&mut self, label: &str, ct: &Ciphertext, expected: &[f64], shape: Option<bool>) {
        let a = self.raw(label, ct);
        if a.len() != 512 {
            return;
        }
        let mut error = 0_f64;
        let mut worst = 0;
        let mut active = 0_f64;
        let mut padding = 0_f64;
        let mut replica = 0_f64;
        let mut imaginary = 0_f64;
        for (s, (z, &want)) in a.iter().zip(expected).enumerate() {
            let err = (z.0 - want).hypot(z.1);
            if err > error {
                error = err;
                worst = s;
            }
            let used = shape.map_or(s / 4 < 3 && s % 4 < 3, |expanded| {
                if expanded { s / 4 < 3 } else { s % 4 < 3 }
            });
            if used {
                active = active.max(err);
            } else {
                padding = padding.max(err);
            }
            imaginary = imaginary.max(z.1.abs());
            if let Some(expanded) = shape {
                let k = if expanded { s / 4 } else { s % 4 };
                if k < 3 {
                    let first = if expanded { 4 * k } else { k };
                    replica = replica.max((z.0 - a[first].0).hypot(z.1 - a[first].1));
                }
            }
        }
        println!(
            "PARITY {label} max={error:.17e} worst={worst} active={active:.17e} padding={padding:.17e} replica={replica:.17e} imaginary={imaginary:.17e}"
        );
        self.ok &= error <= E;
    }
    fn states(
        &mut self,
        label: &str,
        graph: &PackedBlockGraph<Ciphertext>,
        states: &[ScalarStage],
    ) {
        for (s, stage) in graph.stages().iter().enumerate() {
            self.stage(&format!("{label} stage={s}"), stage, &states[s], s);
        }
    }
    fn stage(
        &mut self,
        label: &str,
        stage: &PackedStage<Ciphertext>,
        state: &ScalarStage,
        s: usize,
    ) {
        for k in 0..2 {
            self.parity(
                &format!("{label} W{k}"),
                stage.layers()[k].weights().ciphertext(),
                &matrix(&state.w[k], (s + k) % 2 == 0),
                None,
            );
            if let Some(u) = stage.layers()[k].velocity() {
                self.parity(
                    &format!("{label} U{k}"),
                    u.ciphertext(),
                    &matrix(&state.u(k), (s + k) % 2 == 0),
                    None,
                );
            }
        }
    }
    fn role_expected(
        role: PackedRefreshRole,
        states: &[ScalarStage],
        snaps: &[Snapshot],
        ids: &[usize],
        inference: bool,
    ) -> (Vec<f64>, Option<bool>, f64) {
        match role {
            PackedRefreshRole::Weight(k) => (
                matrix(&states[k / 2].w[k % 2], (k / 2 + k % 2) % 2 == 0),
                None,
                0.4,
            ),
            PackedRefreshRole::Velocity(k) => (
                matrix(&states[k / 2].u(k % 2), (k / 2 + k % 2) % 2 == 0),
                None,
                0.1,
            ),
            PackedRefreshRole::Hidden { stage, record } => {
                let j = ids.iter().position(|&id| id == record as usize).unwrap();
                (
                    vector(&snaps[stage].hidden[j], stage % 2 == 1),
                    Some(stage % 2 == 1),
                    if inference { 0.6 } else { 0.5 },
                )
            }
            PackedRefreshRole::Bias(_) => panic!("registered no-bias fixture"),
        }
    }
    fn events(
        &mut self,
        label: &str,
        events: &[PackedBootstrapEvent<Ciphertext>],
        states: &[ScalarStage],
        snaps: &[Snapshot],
        ids: &[usize],
        inference: bool,
    ) {
        for (i, e) in events.iter().enumerate() {
            let (expected, shape, bound) =
                Self::role_expected(e.role(), states, snaps, ids, inference);
            let ordinal = self
                .observed_inputs
                .iter()
                .find(|(_, ct)| same_ciphertext(ct, e.input()))
                .map(|(identity, _)| *identity);
            self.ok &= ordinal.is_some_and(|identity| {
                identity.role == e.role()
                    && identity.phase
                        == if inference {
                            AttemptPhase::Inference
                        } else {
                            AttemptPhase::Training
                        }
            });
            let name = format!("{label} event={i} global={ordinal:?} role={:?}", e.role());
            self.parity(&format!("{name} source"), e.source(), &expected, shape);
            if let Some(p) = e.projected() {
                let source = self.raw(&format!("{name} projection_source"), e.source());
                let projected = self.raw(&format!("{name} projected_raw"), p);
                println!(
                    "PROJECTION {name} drift={:.17e}",
                    distance(&source, &projected)
                );
                self.parity(&format!("{name} projected"), p, &expected, shape);
            }
            self.parity(&format!("{name} input"), e.input(), &expected, shape);
            self.parity(&format!("{name} output"), e.output(), &expected, shape);
            let input = self.raw(&format!("{name} range"), e.input());
            let range = input.iter().map(|z| z.0.hypot(z.1)).fold(0., f64::max);
            println!(
                "EVENT {name} range={range:.17e} bound={bound} counts={:?}",
                e.counter_delta()
            );
            self.ok &= range <= bound && range <= 1.;
        }
    }
    fn attempts(&mut self, attempts: &[Attempt<Ciphertext>], start: usize) {
        for (i, a) in attempts.iter().enumerate().skip(start) {
            self.observed_inputs.push((a.identity, a.input.clone()));
            let input = self.raw(&format!("attempt={i} input"), &a.input);
            let (bound, range, passed) = attempt_range(a.identity, &input);
            let interval = attempt_delta(&a.after, &a.before);
            self.ok &= passed && interval.is_some();
            println!("ATTEMPT_INTERVAL {i} {interval:?}");
            println!(
                "ATTEMPT {i} identity={:?} bound={bound} range={range:.17e} before={:?} after={:?} completed={} validated={}",
                a.identity,
                a.before,
                a.after,
                a.output.is_some(),
                a.response_validated.get()
            );
            if let Some(out) = &a.output {
                let output = self.raw(&format!("attempt={i} output"), out);
                let err = distance(&input, &output);
                println!("INCREMENTAL {i} error={err:.17e} epsilon={EPS:.17e}");
                self.ok &= err <= EPS;
            } else {
                self.ok = false;
            }
        }
    }
    fn clients(
        &mut self,
        attempts: &[ClientAttempt],
        start: usize,
        states: &[ScalarStage],
        snaps: &[Snapshot],
        ids: &[usize],
    ) {
        for (i, a) in attempts.iter().enumerate().skip(start) {
            let (expected, shape, _) = Self::role_expected(a.role, states, snaps, ids, false);
            self.parity(
                &format!("client={i} role={:?} input", a.role),
                &a.input,
                &expected,
                shape,
            );
            if let Some(out) = &a.output {
                self.parity(&format!("client={i} output"), out, &expected, shape);
            } else {
                self.ok = false;
            }
        }
    }
    fn training(
        &mut self,
        label: &str,
        receipt: &PackedGraphReceipt<Ciphertext>,
        states: &[ScalarStage],
        snaps: &[Snapshot],
        ids: &[usize],
    ) {
        for (s, child) in receipt.children().iter().enumerate() {
            let step = child.step();
            let heads = step
                .trace
                .iter()
                .filter(|e| e.name == "affine2.product")
                .count();
            let losses = step.trace.iter().filter(|e| e.name == "residual").count();
            let outers = step
                .trace
                .iter()
                .filter(|e| e.name == "outer0" || e.name == "outer1")
                .count();
            let updates = step
                .trace
                .iter()
                .filter(|e| e.name == "weight0" || e.name == "weight1")
                .count();
            println!(
                "TRACE_FLAGS {label} stage={s} head_affines={heads} residuals={losses} outer_sums={outers} state_updates={updates}"
            );
            self.ok &= heads == ids.len() && losses == ids.len() && outers == 2 && updates == 2;
            self.ok &= step.forward().valid_ids() == ids;
            for (j, &id) in ids.iter().enumerate() {
                self.parity(
                    &format!("{label} s={s} id={id} hidden_pre"),
                    step.forward().hidden_pre()[j].ciphertext(),
                    &vector(&snaps[s].hidden[j], s % 2 == 1),
                    Some(s % 2 == 1),
                );
                if let Some(post) = step.forward().hidden_post() {
                    self.parity(
                        &format!("{label} s={s} id={id} hidden_post"),
                        post[j].ciphertext(),
                        &vector(&snaps[s].hidden[j], s % 2 == 1),
                        Some(s % 2 == 1),
                    );
                }
                self.parity(
                    &format!("{label} s={s} id={id} output"),
                    step.forward().output()[j].ciphertext(),
                    &vector(&snaps[s].output[j], s % 2 == 0),
                    Some(s % 2 == 0),
                );
            }
            match child {
                PackedGraphChildReceipt::Ni(r) => {
                    self.stage(
                        &format!("{label} candidate stage={s}"),
                        r.updated_stage(),
                        &states[s],
                        s,
                    );
                    self.events(label, r.events(), states, snaps, ids, false);
                }
                PackedGraphChildReceipt::Interactive(r) => self.stage(
                    &format!("{label} candidate stage={s}"),
                    r.updated_stage(),
                    &states[s],
                    s,
                ),
            }
        }
        for h in receipt.handoffs() {
            if let HandoffOrigin::AaPost {
                stage,
                valid_position,
                child_event_index,
            } = h.origin()
            {
                let child = &receipt.children()[*stage];
                let selected =
                    child.step().forward().hidden_post().unwrap()[*valid_position].ciphertext();
                self.ok &= same_ciphertext(selected, h.source().ciphertext())
                    && same_ciphertext(selected, h.output().unwrap().ciphertext());
                if let PackedGraphChildReceipt::Ni(child) = child {
                    self.ok &=
                        same_ciphertext(selected, child.events()[*child_event_index].output());
                }
            }
            let s = h.stage();
            let j = ids.iter().position(|&id| id == h.record()).unwrap();
            self.parity(
                &format!("{label} handoff s={s} id={} source", h.record()),
                h.source().ciphertext(),
                &vector(&snaps[s].hidden[j], s % 2 == 1),
                Some(s % 2 == 1),
            );
            if let Some(out) = h.output() {
                self.parity(
                    &format!("{label} handoff s={s} id={} output", h.record()),
                    out.ciphertext(),
                    &vector(&snaps[s + 1].input[j], s % 2 == 1),
                    Some(s % 2 == 1),
                );
            }
            if let Some(e) = h.bootstrap() {
                self.events(label, std::slice::from_ref(e), states, snaps, ids, false);
            }
        }
    }
    fn inference(
        &mut self,
        label: &str,
        out: &PackedForwardOutput<Ciphertext>,
        r: &PackedGraphForwardReceipt<Ciphertext>,
        states: &[ScalarStage],
        snaps: &[Snapshot],
    ) {
        for (s, child) in r.children().iter().enumerate() {
            let heads = child
                .trace()
                .iter()
                .filter(|e| e.name == "affine2.product")
                .count();
            let forbidden = child
                .trace()
                .iter()
                .filter(|e| {
                    e.name == "residual"
                        || e.name.starts_with("outer")
                        || e.name.starts_with("weight")
                        || e.name.starts_with("velocity")
                        || e.name.starts_with("derivative")
                })
                .count();
            println!(
                "INFERENCE_TRACE_FLAGS {label} stage={s} head_affines={heads} loss_outer_update_derivative_boundaries={forbidden} graph_commits=0"
            );
            self.ok &= forbidden == 0 && heads == if s + 1 == states.len() { 3 } else { 0 };
            for j in 0..3 {
                self.parity(
                    &format!("{label} s={s} id={j} input"),
                    child.inputs()[j].ciphertext(),
                    &vector(&snaps[s].input[j], s % 2 == 0),
                    Some(s % 2 == 0),
                );
                self.parity(
                    &format!("{label} s={s} id={j} z"),
                    child.preactivations()[j].ciphertext(),
                    &vector(&snaps[s].z[j], s % 2 == 1),
                    Some(s % 2 == 1),
                );
                let output_expanded = if s + 1 == states.len() {
                    s % 2 == 0
                } else {
                    s % 2 == 1
                };
                self.parity(
                    &format!("{label} s={s} id={j} path_output"),
                    child.outputs()[j].ciphertext(),
                    &vector(&snaps[s].output[j], output_expanded),
                    Some(output_expanded),
                );
                self.parity(
                    &format!("{label} s={s} id={j} hidden_pre"),
                    child.hidden_pre()[j].ciphertext(),
                    &vector(&snaps[s].hidden[j], s % 2 == 1),
                    Some(s % 2 == 1),
                );
                if let Some(post) = child.hidden_post() {
                    self.parity(
                        &format!("{label} s={s} id={j} hidden_post"),
                        post[j].ciphertext(),
                        &vector(&snaps[s].hidden[j], s % 2 == 1),
                        Some(s % 2 == 1),
                    );
                }
            }
            self.events(label, child.events(), states, snaps, &[0, 1, 2], true);
        }
        for h in r.handoffs() {
            let s = h.stage();
            let j = h.record();
            self.parity(
                &format!("{label} handoff s={s} id={j} source"),
                h.source().ciphertext(),
                &vector(&snaps[s].hidden[j], s % 2 == 1),
                Some(s % 2 == 1),
            );
            if let Some(v) = h.output() {
                self.parity(
                    &format!("{label} handoff s={s} id={j} out"),
                    v.ciphertext(),
                    &vector(&snaps[s + 1].input[j], s % 2 == 1),
                    Some(s % 2 == 1),
                );
            }
        }
        let s = states.len() - 1;
        for j in 0..3 {
            self.parity(
                &format!("{label} final id={j}"),
                out.values()[j].ciphertext(),
                &vector(&snaps[s].output[j], s % 2 == 0),
                Some(s % 2 == 0),
            );
        }
    }
    fn failed_inference(
        &mut self,
        f: &PackedGraphForwardFailure<Ciphertext>,
        states: &[ScalarStage],
        snaps: &[Snapshot],
    ) {
        println!(
            "FAILED_INFERENCE stage={} interval={:?} error={}",
            f.stage(),
            f.counter_delta(),
            f.error()
        );
        visit_forward_failure(f, |s, id, kind, ct| {
            let (expected, expanded) = match kind {
                "input" => (vector(&snaps[s].input[id], s % 2 == 0), s % 2 == 0),
                "affine" | "z" => (vector(&snaps[s].z[id], s % 2 == 1), s % 2 == 1),
                "head_affine" | "head_z" => (vector(&snaps[s].output[id], s % 2 == 0), s % 2 == 0),
                "output" => {
                    let expanded = if s + 1 == states.len() {
                        s % 2 == 0
                    } else {
                        s % 2 == 1
                    };
                    (vector(&snaps[s].output[id], expanded), expanded)
                }
                _ => (vector(&snaps[s].hidden[id], s % 2 == 1), s % 2 == 1),
            };
            self.parity(
                &format!("failed inference stage={s} record={id} {kind}"),
                ct,
                &expected,
                Some(expanded),
            );
        });
        self.ok = false;
    }
    fn failed_graph(
        &mut self,
        f: &PackedGraphFailure<Ciphertext>,
        states: &[ScalarStage],
        snaps: &[Snapshot],
        ids: &[usize],
    ) {
        println!(
            "FAILURE phase={:?} stage={} role={:?} interval={:?} error={}",
            f.phase(),
            f.stage(),
            f.role(),
            f.counter_delta(),
            f.error()
        );
        for (s, child) in f.children().iter().enumerate() {
            for (j, &id) in ids.iter().enumerate() {
                self.parity(
                    &format!("failed completed s={s} id={id} hidden_pre"),
                    child.step().forward().hidden_pre()[j].ciphertext(),
                    &vector(&snaps[s].hidden[j], s % 2 == 1),
                    Some(s % 2 == 1),
                );
                self.parity(
                    &format!("failed completed s={s} id={id} selected"),
                    child.step().forward().hidden()[j].ciphertext(),
                    &vector(&snaps[s].hidden[j], s % 2 == 1),
                    Some(s % 2 == 1),
                );
                self.parity(
                    &format!("failed completed s={s} id={id} output"),
                    child.step().forward().output()[j].ciphertext(),
                    &vector(&snaps[s].output[j], s % 2 == 0),
                    Some(s % 2 == 0),
                );
            }
            if let PackedGraphChildReceipt::Ni(r) = child {
                self.stage(
                    "failed previous candidate",
                    r.updated_stage(),
                    &states[s],
                    s,
                );
                self.events("failed previous", r.events(), states, snaps, ids, false);
            }
        }
        if let Some(PackedGraphChildFailure::Ni(child)) = f.child_failure() {
            self.events("failed child", child.events(), states, snaps, ids, false);
            if let Some(s) = child.updated_stage() {
                self.stage("failed current candidate", s, &states[f.stage()], f.stage());
            }
        }
        for h in f.handoffs() {
            self.raw("failed handoff source", h.source().ciphertext());
            if let Some(o) = h.output() {
                self.raw("failed handoff output", o.ciphertext());
            }
            if let Some(raw) = h.raw_attempt() {
                self.raw("failed handoff raw input", raw.input());
                if let Some(output) = raw.output() {
                    self.raw("failed handoff raw returned output", output);
                }
            }
            if let Some(e) = h.bootstrap() {
                self.events(
                    "failed handoff",
                    std::slice::from_ref(e),
                    states,
                    snaps,
                    ids,
                    false,
                );
            }
        }
        self.ok = false;
    }
}
fn distance(a: &[Z], b: &[Z]) -> f64 {
    if a.len() != 512 || b.len() != 512 {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let d = (a.0 - b.0).hypot(a.1 - b.1);
            if d.is_finite() { d } else { f64::INFINITY }
        })
        .fold(0., f64::max)
}
fn histories_match(actual: &PackedGraphNiPlan, frozen: &PackedGraphNiPlan) {
    assert_eq!(actual.predicted_counts(), frozen.predicted_counts());
    for (a, b) in actual.children().iter().zip(frozen.children()) {
        assert_eq!(a.dag().step().trace, b.dag().step().trace);
        assert_eq!(a.committed().weights(), b.committed().weights());
        assert_eq!(a.committed().velocities(), b.committed().velocities());
        assert_eq!(a.refresh_state(), b.refresh_state());
    }
}
pub fn ni() {
    // Plan with public parameters before held-out key generation; no hypothetical plan executes.
    let (p, enc, contract) = parameters();
    let frozen = schedules(&p, &enc, &contract);
    let expected = frozen.iter().map(|c| c.total.bootstraps).sum::<u64>();
    assert_eq!(expected, 332);
    assert!(expected <= 372);
    let c = Context::new(true);
    let identities = frozen_attempts(&frozen);
    assert_eq!(identities.len() as u64, expected);
    let boot = RecordedBoot::new(c.bs.as_ref().unwrap(), identities);
    let mut committed = 0;
    let mut inferences = 0;
    let mut cases = 0;
    let mut validated = 0;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for (case_index, case) in frozen.into_iter().enumerate() {
            assert_eq!(case.total, CASE_COUNTS[case_index]);
            let mode = match case.placement {
                PackedNiPlacement::Au => 1,
                PackedNiPlacement::Wn => 2,
                PackedNiPlacement::Aa => 3,
            };
            let mut graph = c.graph(case.g, mode);
            let mut scalar = vec![ScalarStage::new(); case.g];
            let server = PackedBlockNiTrainer::new(&c.ev, &c.enc, &boot, case.placement).unwrap();
            let case_before = c.ev.counter();
            for (t, &rate) in RATES.iter().enumerate() {
                let label = format!("NI G{} {:?} t={t}", case.g, case.placement);
                println!("BEGIN {label}");
                let batch = c.batch(&graph, mode, t);
                let plan = server
                    .plan_update(
                        &graph,
                        &batch,
                        rate,
                        GradRelin::Lazy,
                        next(&c.p, graph.geometry(), t),
                    )
                    .unwrap();
                histories_match(&plan, &case.updates[t]);
                let previous = scalar.clone();
                let snaps = scalar_step(&mut scalar, t, &vec![Y.to_vec(); case.g]);
                let start = boot.attempts.borrow().len();
                let prepared = server.prepare_planned(&graph, &batch, &plan);
                let mut binding_ok = true;
                match &prepared {
                    Ok((_, r)) => {
                        for child in r.children() {
                            if let PackedGraphChildReceipt::Ni(child) = child {
                                binding_ok &= validate_recorded_events(&boot, child.events());
                            }
                        }
                        for h in r.handoffs() {
                            if let Some(e) = h.bootstrap() {
                                binding_ok &=
                                    validate_recorded_events(&boot, std::slice::from_ref(e));
                            }
                        }
                    }
                    Err(f) => {
                        for child in f.children() {
                            if let PackedGraphChildReceipt::Ni(child) = child {
                                binding_ok &= validate_recorded_events(&boot, child.events());
                            }
                        }
                        if let Some(PackedGraphChildFailure::Ni(child)) = f.child_failure() {
                            binding_ok &= validate_recorded_events(&boot, child.events());
                        }
                        for h in f.handoffs() {
                            if let Some(e) = h.bootstrap() {
                                binding_ok &=
                                    validate_recorded_events(&boot, std::slice::from_ref(e));
                            }
                        }
                        if let Some(role) = f.role() {
                            binding_ok &= boot
                                .attempts
                                .borrow()
                                .last()
                                .is_some_and(|a| a.identity.role == role);
                        }
                    }
                }
                // All decrypt/diagnostic work is after synchronous preparation and outside NI capability.
                let mut diag = Diagnostic::new(&c);
                diag.ok &= binding_ok;
                diag.attempts(&boot.attempts.borrow(), start);
                diag.states(&format!("{label} old"), &graph, &previous);
                match prepared {
                    Ok((candidate, receipt)) => {
                        diag.training(&label, &receipt, &scalar, &snaps, IDS[t]);
                        diag.states(
                            &format!("{label} committed_candidate"),
                            candidate.graph(),
                            &scalar,
                        );
                        println!("UPDATE_COUNTS {label} {:?}", receipt.actual_counts());
                        assert!(
                            diag.ok,
                            "{label}: all-slot gate failed; stop before next update"
                        );
                        assert_eq!(
                            boot.attempts.borrow().len() - start,
                            case.updates[t].predicted_counts().bootstraps as usize
                        );
                        validated += boot.attempts.borrow().len() - start;
                        graph = candidate.into_graph();
                        committed += 1;
                        println!(
                            "GRAPH_COMMIT {label} count={committed} indices={:?}",
                            graph
                                .stages()
                                .iter()
                                .map(|s| s.layers()[0].update_index())
                                .collect::<Vec<_>>()
                        );
                    }
                    Err(f) => {
                        diag.failed_graph(&f, &scalar, &snaps, IDS[t]);
                        panic!("{label}: preparation failed, no commit");
                    }
                }
            }
            let inputs = c.inference(&graph, mode);
            let policy = if case.placement == PackedNiPlacement::Aa {
                PackedInferencePlan::AllHiddenBootstrap
            } else {
                PackedInferencePlan::BoundaryBootstrap
            };
            let plan = server.plan_forward(&graph, &inputs, policy).unwrap();
            assert_eq!(plan.predicted_counts(), case.inference.predicted_counts());
            for (a, b) in plan.children().iter().zip(case.inference.children()) {
                assert_eq!(a.trace(), b.trace());
            }
            let start = boot.attempts.borrow().len();
            let prepared = server.prepare_forward(&graph, &inputs, &plan);
            let mut binding_ok = true;
            match &prepared {
                Ok((_, r)) => {
                    for child in r.children() {
                        binding_ok &= validate_recorded_events(&boot, child.events());
                    }
                }
                Err(f) => {
                    for child in f.children() {
                        binding_ok &= validate_recorded_events(&boot, child.events());
                    }
                    if let Some(child) = f.child_failure() {
                        binding_ok &= validate_recorded_events(&boot, child.events());
                        if let Some(role) = child.role() {
                            binding_ok &= boot
                                .attempts
                                .borrow()
                                .last()
                                .is_some_and(|a| a.identity.role == role);
                        }
                    }
                }
            }
            let mut diag = Diagnostic::new(&c);
            diag.ok &= binding_ok;
            diag.attempts(&boot.attempts.borrow(), start);
            let snaps = scalar_forward(&scalar);
            match prepared {
                Ok((out, receipt)) => {
                    diag.inference("NI inference", &out, &receipt, &scalar, &snaps);
                    diag.states("post inference unchanged", &graph, &scalar);
                    assert!(diag.ok, "NI inference all-slot gate failed");
                    validated += boot.attempts.borrow().len() - start;
                    inferences += 1;
                }
                Err(f) => {
                    diag.failed_inference(&f, &scalar, &snaps);
                    panic!("NI inference failed: {}", f.error());
                }
            }
            let after = c.ev.counter();
            assert_eq!(
                after.bootstraps - case_before.bootstraps,
                case.total.bootstraps
            );
            cases += 1;
        }
        assert_eq!((cases, committed, inferences), (6, 24, 6));
        assert_eq!(boot.attempts.borrow().len() as u64, expected);
        assert_eq!(validated as u64, expected);
        assert!(boot.attempts.borrow().iter().all(|a| a.output.is_some()));
    }));
    println!(
        "NI_END success={} cases={cases} committed={committed} inferences={inferences} attempted={} completed={} validated={validated} expected={expected} cap=372",
        result.is_ok(),
        boot.attempts.borrow().len(),
        boot.attempts
            .borrow()
            .iter()
            .filter(|a| a.output.is_some())
            .count()
    );
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
pub fn interactive() {
    let (params, encoder, _) = parameters();
    let frozen_cases = [
        interactive_schedule(&params, &encoder, 2),
        interactive_schedule(&params, &encoder, 3),
    ];
    let frozen_forwards = [
        interactive_forward_schedule(&params, &encoder, 2),
        interactive_forward_schedule(&params, &encoder, 3),
    ];
    let c = Context::new(false);
    let server = PackedBlockInteractiveTrainer::new(&c.ev, &c.enc).unwrap();
    let mut client = RealClient {
        c: &c,
        g: 2,
        contract: PackedRefreshContract::new(&c.ev, 16).unwrap(),
        attempts: vec![],
    };
    let mut committed = 0;
    let mut inferences = 0;
    let mut graphs = 0;
    let mut validated = 0;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for g in [2, 3] {
            client.g = g;
            let mut graph = c.graph(g, 0);
            let mut scalar = vec![ScalarStage::new(); g];
            let frozen = &frozen_cases[g - 2];
            for (t, &rate) in RATES.iter().enumerate() {
                let label = format!("I G{g} t={t}");
                println!("BEGIN {label}");
                let data = c.batch(&graph, 0, t);
                let previous = scalar.clone();
                let snaps = scalar_step(&mut scalar, t, &vec![Y.to_vec(); g]);
                let start = client.attempts.len();
                let prepared =
                    server.prepare_update(&graph, &data, rate, GradRelin::Lazy, &mut client);
                let mut diag = Diagnostic::new(&c);
                diag.clients(&client.attempts, start, &scalar, &snaps, IDS[t]);
                diag.states(&format!("{label} old"), &graph, &previous);
                match prepared {
                    Ok((candidate, receipt)) => {
                        for (s, child) in receipt.children().iter().enumerate() {
                            assert_eq!(child.step().trace, frozen[t][s].step().trace);
                            assert_eq!(
                                child.step().actual_counts,
                                frozen[t][s].step().predicted_counts
                            );
                        }
                        diag.training(&label, &receipt, &scalar, &snaps, IDS[t]);
                        diag.states(&format!("{label} candidate"), candidate.graph(), &scalar);
                        assert!(diag.ok, "I all-slot gate failed; stop before next update");
                        assert_eq!(
                            client.attempts.len() - start,
                            4 * g + IDS[t].len() * (g - 1)
                        );
                        validated += client.attempts.len() - start;
                        graph = candidate.into_graph();
                        committed += 1;
                        println!(
                            "GRAPH_COMMIT {label} count={committed} indices={:?}",
                            graph
                                .stages()
                                .iter()
                                .map(|s| s.layers()[0].update_index())
                                .collect::<Vec<_>>()
                        );
                    }
                    Err(f) => {
                        diag.failed_graph(&f, &scalar, &snaps, IDS[t]);
                        panic!("I preparation failed; no commit");
                    }
                }
            }
            let inputs = c.inference(&graph, 0);
            let start = client.attempts.len();
            let prepared = server.prepare_forward(
                &graph,
                &inputs,
                PackedInferencePlan::ClientBoundary,
                &mut client,
            );
            let snaps = scalar_forward(&scalar);
            let mut diag = Diagnostic::new(&c);
            diag.clients(&client.attempts, start, &scalar, &snaps, &[0, 1, 2]);
            match prepared {
                Ok((out, receipt)) => {
                    for (s, child) in receipt.children().iter().enumerate() {
                        assert_eq!(child.trace(), frozen_forwards[g - 2][s].trace());
                        assert_eq!(
                            child.actual_counts(),
                            frozen_forwards[g - 2][s].predicted_counts()
                        );
                    }
                    diag.inference("I inference", &out, &receipt, &scalar, &snaps);
                    diag.states("I inference unchanged", &graph, &scalar);
                    assert!(diag.ok, "I inference all-slot gate failed");
                    assert_eq!(client.attempts.len() - start, 3 * (g - 1));
                    validated += client.attempts.len() - start;
                    inferences += 1;
                }
                Err(f) => {
                    diag.failed_inference(&f, &scalar, &snaps);
                    panic!("I inference failed: {}", f.error());
                }
            }
            graphs += 1;
        }
        assert_eq!((graphs, committed, inferences), (2, 8, 2));
        assert_eq!(client.attempts.len(), 116);
        assert_eq!(validated, 116);
        assert!(client.attempts.iter().all(|a| a.output.is_some()));
        assert_eq!(c.ev.counter().bootstraps, 0);
    }));
    println!(
        "I_END success={} graphs={graphs} committed={committed} inferences={inferences} attempted={} completed={} validated={validated}",
        result.is_ok(),
        client.attempts.len(),
        client
            .attempts
            .iter()
            .filter(|a| a.output.is_some())
            .count()
    );
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

fn same_ciphertext(a: &Ciphertext, b: &Ciphertext) -> bool {
    a.scale().to_bits() == b.scale().to_bits()
        && a.level() == b.level()
        && a.len() == b.len()
        && a.polys()
            .iter()
            .zip(b.polys())
            .all(|(a, b)| a.limbs() == b.limbs())
}

fn validate_recorded_events<E: HeOps<Ct = Ciphertext>, B: PlannedBootstrap<E>>(
    recorder: &RecordedBoot<'_, E, B>,
    events: &[PackedBootstrapEvent<Ciphertext>],
) -> bool {
    let attempts = recorder.attempts.borrow();
    let mut valid = true;
    for event in events {
        if let Some(attempt) = attempts
            .iter()
            .find(|a| same_ciphertext(&a.input, event.input()))
        {
            let ok = attempt.identity.role == event.role()
                && attempt
                    .output
                    .as_ref()
                    .is_some_and(|out| same_ciphertext(out, event.output()))
                && attempt_delta(&attempt.after, &attempt.before).as_ref()
                    == Some(event.counter_delta());
            attempt.response_validated.set(ok);
            valid &= ok;
        } else {
            valid = false;
        }
    }
    valid
}
fn attempt_delta(a: &OpCounter, b: &OpCounter) -> Option<OpCounter> {
    Some(OpCounter {
        rotations: a.rotations.checked_sub(b.rotations)?,
        relins: a.relins.checked_sub(b.relins)?,
        ct_ct_tensor: a.ct_ct_tensor.checked_sub(b.ct_ct_tensor)?,
        ct_pt_mults: a.ct_pt_mults.checked_sub(b.ct_pt_mults)?,
        rescales: a.rescales.checked_sub(b.rescales)?,
        sse_switches: a.sse_switches.checked_sub(b.sse_switches)?,
        bootstraps: a.bootstraps.checked_sub(b.bootstraps)?,
    })
}
