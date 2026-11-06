//! D4c2: registered finite full-chain bootstrap calibration, not packed training.
//! Construction validation must select individual `non_he_*` tests with `--exact`.
//! Run the real test only after independent review and root's one-attempt release.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use ckks_train_ckks::bootstrap::{BootParams, Bootstrapper};
use ckks_train_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter,
};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::packed_training::{PackedGeometryPlan, StageShape};

const SLOTS: usize = 512;
const VECTORS: usize = 21;
const CALLS: usize = 84;
const DELTA: f64 = 4_398_046_511_104.0;
const EPSILON: f64 = 0.000_000_953_674_316_406_25;
const W1: [[f64; 3]; 3] = [
    [0.25, -0.125, 0.0625],
    [-0.0625, 0.1875, 0.125],
    [0.125, 0.0625, -0.1875],
];
const W2: [[f64; 3]; 3] = [
    [0.1875, -0.125, 0.0625],
    [0.0625, 0.25, -0.125],
    [-0.125, 0.0625, 0.1875],
];
type Complex = (f64, f64);
type Check<T = ()> = Result<T, String>;

struct Vector {
    name: &'static str,
    m: Vec<f64>,
}

// Independent literal slot maps: no production packing or mask helper calls.
fn matrix(w: [[f64; 3]; 3], re: bool) -> Vec<f64> {
    (0..SLOTS)
        .map(|s| {
            let (i, j) = (s / 4, s % 4);
            if i < 3 && j < 3 {
                if re { w[j][i] } else { w[i][j] }
            } else {
                0.0
            }
        })
        .collect()
}

fn vector(a: [f64; 3], expanded: bool) -> Vec<f64> {
    (0..SLOTS)
        .map(|s| {
            let i = if expanded { s / 4 } else { s % 4 };
            if i < 3 { a[i] } else { 0.0 }
        })
        .collect()
}

fn old_hidden(square: bool) -> [f64; 3] {
    let x = [0.75, -0.375, 0.375];
    let bias = [0.25, 0.1875, -0.125];
    std::array::from_fn(|o| {
        let mut z = 0.0;
        for (i, x_i) in x.iter().enumerate() {
            z += W1[o][i] * x_i;
        }
        if square {
            z += bias[o];
            z * z
        } else {
            z * z + z
        }
    })
}

fn corpus() -> Vec<Vector> {
    let mut out = Vec::new();
    let mut push = |name, m| out.push(Vector { name, m });
    push("zero", vec![0.0; SLOTS]);
    push("constant_plus_half", vec![0.5; SLOTS]);
    push("constant_minus_half", vec![-0.5; SLOTS]);
    push(
        "alternating",
        (0..SLOTS)
            .map(|s| if s % 2 == 0 { 0.5 } else { -0.5 })
            .collect(),
    );
    push(
        "ramp",
        (0..SLOTS)
            .map(|s| 0.5 * (2.0 * s as f64 / 511.0 - 1.0))
            .collect(),
    );
    let mut impulse = vec![0.0; SLOTS];
    impulse[0] = 0.5;
    push("impulse_0", impulse);
    let mut impulse = vec![0.0; SLOTS];
    impulse[511] = -0.5;
    push("impulse_511", impulse);
    for (amp, names) in [
        (0.4, ["w_envelope_re", "w_envelope_ce"]),
        (0.1, ["u_envelope_re", "u_envelope_ce"]),
    ] {
        let w = std::array::from_fn(|o| {
            std::array::from_fn(|i| if (o + 2 * i) % 2 == 0 { amp } else { -amp })
        });
        push(names[0], matrix(w, true));
        push(names[1], matrix(w, false));
    }
    for (a, names) in [
        ([0.4, -0.4, 0.4], ["bias_expanded", "bias_repeated"]),
        ([0.5, -0.5, 0.5], ["hidden_expanded", "hidden_repeated"]),
        (
            old_hidden(true),
            ["square_old_expanded", "square_old_repeated"],
        ),
        (
            old_hidden(false),
            ["poly_old_expanded", "poly_old_repeated"],
        ),
    ] {
        push(names[0], vector(a, true));
        push(names[1], vector(a, false));
    }
    push("initial_w1_re", matrix(W1, true));
    push("initial_w2_ce", matrix(W2, false));
    out
}

fn encryption_seed(j: usize, k: usize) -> u8 {
    assert!(j < 2 && k < VECTORS);
    (100 + 21 * j + k) as u8
}

#[derive(Debug)]
struct Metric {
    max: f64,
    slot: usize,
    expected: Complex,
    actual: Complex,
    real_error: f64,
    imaginary_error: f64,
}

fn metric(actual: &[Complex], expected: &[Complex]) -> Check<Metric> {
    if actual.len() != SLOTS || expected.len() != SLOTS {
        return Err(format!(
            "slot length actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    let mut worst = Metric {
        max: -1.0,
        slot: 0,
        expected: (0.0, 0.0),
        actual: (0.0, 0.0),
        real_error: 0.0,
        imaginary_error: 0.0,
    };
    for (slot, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let re = a.0 - e.0;
        let im = a.1 - e.1;
        let error = re.hypot(im);
        if ![a.0, a.1, e.0, e.1, re, im, error]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(format!(
                "nonfinite slot={slot} expected={e:?} actual={a:?} error={error}"
            ));
        }
        if error > worst.max {
            worst = Metric {
                max: error,
                slot,
                expected: e,
                actual: a,
                real_error: re,
                imaginary_error: im,
            };
        }
    }
    Ok(worst)
}

fn observe(name: &str, actual: &[Complex], expected: &[Complex]) -> Check<f64> {
    let m = metric(actual, expected)?;
    println!(
        "metric={name} complex_error={:.17e} worst_slot={} expected_re={:.17e} expected_im={:.17e} actual_re={:.17e} actual_im={:.17e} real_error={:.17e} imaginary_error={:.17e}",
        m.max,
        m.slot,
        m.expected.0,
        m.expected.1,
        m.actual.0,
        m.actual.1,
        m.real_error,
        m.imaginary_error
    );
    Ok(m.max)
}

fn limit(name: &str, actual: f64, bound: f64) -> Check {
    if !actual.is_finite() || !bound.is_finite() || actual < 0.0 || actual > bound {
        Err(format!(
            "gate={name} actual={actual:.17e} bound={bound:.17e}"
        ))
    } else {
        Ok(())
    }
}

fn input_range(pass: usize, norm: f64, error: f64) -> Check {
    limit("input_range_relative", norm, 0.5 + error)?;
    limit("input_range_hard", norm, 0.5 + pass as f64 * EPSILON)?;
    limit("input_range_universal", norm, 1.0)
}

#[derive(Debug)]
struct Metadata {
    level: usize,
    scale_bits: u64,
    components: usize,
    component_levels: Vec<usize>,
    component_context: Vec<bool>,
    component_ntt: Vec<bool>,
}

fn metadata(ct: &Ciphertext, params: &Arc<CkksParams>) -> Metadata {
    Metadata {
        level: ct.level(),
        scale_bits: ct.scale().to_bits(),
        components: ct.len(),
        component_levels: ct.polys().iter().map(|p| p.level()).collect(),
        component_context: ct
            .polys()
            .iter()
            .map(|p| Arc::ptr_eq(p.basis(), params.basis_q()))
            .collect(),
        component_ntt: ct.polys().iter().map(|p| p.is_ntt_form()).collect(),
    }
}

fn check_metadata(m: &Metadata, level: usize) -> Check {
    if m.level != level
        || m.scale_bits != DELTA.to_bits()
        || m.components != 2
        || m.component_levels != [level, level]
        || m.component_context != [true, true]
        || m.component_ntt != [true, true]
    {
        Err(format!(
            "metadata expected_level={level} expected_scale_bits={} actual={m:?}",
            DELTA.to_bits()
        ))
    } else {
        Ok(())
    }
}

fn check_counter(observed: &OpCounter, configured: &OpCounter) -> Check {
    if observed != configured || observed.bootstraps != 1 || configured.bootstraps != 1 {
        Err(format!(
            "counter observed={observed:?} configured={configured:?}"
        ))
    } else {
        Ok(())
    }
}

struct Fixture {
    params: Arc<CkksParams>,
    encoder: Encoder,
    encryptor: Encryptor,
    decryptor: Decryptor,
    ev: Evaluator,
    bs: Bootstrapper,
}

// No Debug on Fixture: the diagnostic secret remains private.
impl Fixture {
    fn new(seed: u8, progress: &mut Progress) -> Check<Self> {
        progress.stage = "params";
        let (q, p) = ckks_train_ckks::params::n16_boot_bits();
        let params = Arc::new(CkksParams::new(10, &q, &p, 42, 8).map_err(|e| e.to_string())?);
        let bp = BootParams {
            log_slots: 9,
            ..BootParams::default_n16()
        };
        println!(
            "parameters n={} slots={} dnum={} delta={:.17e} delta_bits={} max_level={} q_bits={:?} p_bits={:?} q={:?} p={:?} boot_params={bp:?}",
            params.n(),
            params.slots(),
            params.dnum(),
            params.scale(),
            params.scale().to_bits(),
            params.max_level(),
            params.q_bits(),
            params.p_bits(),
            params.q(),
            params.p()
        );
        progress.stage = "key_generator";
        let mut kg = KeyGenerator::new(Arc::clone(&params), Rng::from_seed([seed; 32]))
            .map_err(|e| e.to_string())?;
        progress.stage = "encoder";
        let encoder = Encoder::new(Arc::clone(&params));
        progress.stage = "bootstrap_constructor";
        let bs = Bootstrapper::new(Arc::clone(&params), bp.clone(), &encoder, &mut kg)
            .map_err(|e| e.to_string())?;
        // No standalone training Galois-key generation, including no unused keys.
        progress.stage = "evaluator_relin_key";
        let relin = kg.relin_key();
        progress.stage = "public_key";
        let public = kg.public_key();
        progress.stage = "diagnostic_secret_clone";
        let secret = kg.secret_key();
        let ev = Evaluator::new(Arc::clone(&params), Some(relin), None);
        let encryptor = Encryptor::new(Arc::clone(&params), public);
        let decryptor = Decryptor::new(Arc::clone(&params), secret);
        progress.stage = "configured_context";
        if !Arc::ptr_eq(bs.params(), &params)
            || bs.boot_params() != &bp
            || bs.output_level() != 16
            || params.scale().to_bits() != DELTA.to_bits()
            || params.n() != 1024
            || params.slots() != SLOTS
            || params.dnum() != 8
        {
            return Err("configured context/spec/output mismatch".into());
        }
        println!(
            "configured_cost={:?} configured_key_switches={} output_level={} bootstrap_rotation_steps={:?}",
            bs.cost_per_boot(),
            bs.cost_per_boot().key_switches(),
            bs.output_level(),
            bs.keys().rotation_steps()
        );
        check_counter(&bs.cost_per_boot(), &bs.cost_per_boot())?;
        Ok(Self {
            params,
            encoder,
            encryptor,
            decryptor,
            ev,
            bs,
        })
    }

    fn decode(&self, ct: &Ciphertext) -> Vec<Complex> {
        self.encoder
            .decode_complex(&self.decryptor.decrypt(ct))
            .into_iter()
            .map(|z| (z.re, z.im))
            .collect()
    }

    fn boot(&self, ct: &Ciphertext, progress: &mut Progress) -> Check<Ciphertext> {
        progress.stage = "bootstrap_input_metadata";
        let input = metadata(ct, &self.params);
        println!("input_metadata={input:?}");
        check_metadata(&input, 0)?;
        let index = progress.index();
        if progress.attempted >= CALLS || progress.coverage[index] != 0 {
            return Err("call bound or duplicate attempt".into());
        }
        let before = self.ev.counter();
        progress.stage = "bootstrap_runtime";
        progress.attempted += 1;
        progress.coverage[index] = 1;
        let result = self.bs.bootstrap(&self.ev, ct);
        if result.is_ok() {
            progress.completed += 1;
            progress.coverage[index] = 2;
        }
        let observed = &self.ev.counter() - &before;
        println!(
            "call_index={index} observed_counter={observed:?} inclusive_key_switches={}",
            observed.key_switches()
        );
        let out = result.map_err(|e| e.to_string())?;
        progress.stage = "bootstrap_counter";
        check_counter(&observed, &self.bs.cost_per_boot())?;
        progress.stage = "bootstrap_output_metadata";
        let output = metadata(&out, &self.params);
        println!("output_metadata={output:?}");
        check_metadata(&output, 16)?;
        Ok(out)
    }
}

struct Progress {
    key: usize,
    vector: usize,
    pass: usize,
    stage: &'static str,
    attempted: usize,
    completed: usize,
    validated: usize,
    // 0 not attempted; 1 attempted, no returned output; 2 returned; 3 all gates passed.
    coverage: [u8; CALLS],
}

impl Progress {
    fn new() -> Self {
        Self {
            key: 0,
            vector: 0,
            pass: 1,
            stage: "startup",
            attempted: 0,
            completed: 0,
            validated: 0,
            coverage: [0; CALLS],
        }
    }
    fn index(&self) -> usize {
        (self.key * VECTORS + self.vector) * 2 + self.pass - 1
    }
    fn validate(&mut self) {
        self.coverage[self.index()] = 3;
        self.validated += 1;
    }
    fn context(&self) -> String {
        format!(
            "key_index={} key_seed={} vector_index={} pass={} stage={}",
            self.key,
            151 + self.key,
            self.vector,
            self.pass,
            self.stage
        )
    }
    fn summary(&self, status: &str) {
        println!(
            "status={status} {} attempted={} completed={} validated={} required={CALLS}",
            self.context(),
            self.attempted,
            self.completed,
            self.validated
        );
        for (i, state) in self
            .coverage
            .iter()
            .enumerate()
            .filter(|(_, state)| **state != 3)
        {
            println!(
                "missing_validated_coverage key_index={} vector_index={} pass={} state={} (0=unattempted,1=attempted,2=returned)",
                i / (VECTORS * 2),
                (i / 2) % VECTORS,
                i % 2 + 1,
                state
            );
        }
    }
}

fn run(progress: &mut Progress) -> Check {
    let corpus = corpus();
    if corpus.len() != VECTORS {
        return Err("fixed corpus length mismatch".into());
    }
    let zeros = vec![(0.0, 0.0); SLOTS];
    for key in 0..2 {
        progress.key = key;
        progress.vector = 0;
        progress.pass = 1;
        println!(
            "construction key_index={key} key_seed={} no_training_galois_keys=true",
            151 + key
        );
        let fx = Fixture::new((151 + key) as u8, progress)?;
        for (k, vector) in corpus.iter().enumerate() {
            progress.vector = k;
            progress.pass = 1;
            progress.stage = "fresh_encode_encrypt";
            let seed = encryption_seed(key, k);
            println!(
                "case {} vector_name={} encryption_seed={seed}",
                progress.context(),
                vector.name
            );
            let intended: Vec<_> = vector.m.iter().map(|&v| (v, 0.0)).collect();
            let pt = fx
                .encoder
                .encode(&vector.m, DELTA, 0)
                .map_err(|e| e.to_string())?;
            let mut rng = Rng::from_seed([seed; 32]);
            let ct0 = fx.encryptor.encrypt(&pt, &mut rng);
            progress.stage = "fresh_input_diagnostics";
            check_metadata(&metadata(&ct0, &fx.params), 0)?;
            let x0 = fx.decode(&ct0);
            let e0 = observe("e0=x0-m", &x0, &intended)?;
            let range0 = observe("norm_x0", &x0, &zeros)?;
            limit("e0", e0, EPSILON)?;
            input_range(1, range0, e0)?;
            let ct_y1 = fx.boot(&ct0, progress)?;
            progress.stage = "first_output_diagnostics";
            let y1 = fx.decode(&ct_y1);
            let e1 = observe("e1=y1-x0", &y1, &x0)?;
            observe("norm_y1", &y1, &zeros)?;
            limit("e1", e1, EPSILON)?;
            // The exact first output is lowered, not projected, relabelled or re-encrypted.
            progress.stage = "lower_first_output";
            let ct1 = fx
                .ev
                .mod_switch_to_level(&ct_y1, 0)
                .map_err(|e| e.to_string())?;
            check_metadata(&metadata(&ct1, &fx.params), 0)?;
            progress.stage = "lowered_input_diagnostics";
            let x1 = fx.decode(&ct1);
            let a1 = observe("a1=x1-m", &x1, &intended)?;
            let drift = observe("lowering_drift=x1-y1", &x1, &y1)?;
            let range1 = observe("norm_x1", &x1, &zeros)?;
            println!(
                "partial_metrics e0={e0:.17e} e1={e1:.17e} a1={a1:.17e} drift={drift:.17e} e2=unobserved final=unobserved norm_x0={range0:.17e} norm_x1={range1:.17e}"
            );
            limit("a1", a1, e0 + EPSILON)?;
            input_range(2, range1, a1)?;
            progress.validate();
            progress.pass = 2;
            println!(
                "case {} vector_name={} encryption=none",
                progress.context(),
                vector.name
            );
            let ct_y2 = fx.boot(&ct1, progress)?;
            progress.stage = "second_output_diagnostics";
            let y2 = fx.decode(&ct_y2);
            let e2 = observe("e2=y2-x1", &y2, &x1)?;
            let final_error = observe("final=y2-m", &y2, &intended)?;
            observe("norm_y2", &y2, &zeros)?;
            println!(
                "metrics e0={e0:.17e} e1={e1:.17e} e2={e2:.17e} a1={a1:.17e} final={final_error:.17e} drift={drift:.17e} norm_x0={range0:.17e} norm_x1={range1:.17e}"
            );
            limit("e2", e2, EPSILON)?;
            limit("final", final_error, e0 + 2.0 * EPSILON)?;
            progress.validate();
        }
        println!(
            "key_completed key_index={key} inclusive_counter={:?}",
            fx.ev.counter()
        );
    }
    if progress.attempted != CALLS || progress.completed != CALLS || progress.validated != CALLS {
        return Err("completion count mismatch".into());
    }
    Ok(())
}

#[test]
fn packed_ni_calibration_primary_84_calls() {
    let mut progress = Progress::new();
    // Embedded-source fingerprint is explicitly noncryptographic; runner supplies SHA256.
    let source_fnv1a = include_bytes!("packed_ni_calibration.rs")
        .iter()
        .fold(0xcbf29ce484222325_u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
    println!(
        "program package={} version={} source_file={} source_fnv1a64={source_fnv1a:016x} arch={} os={} pointer_bits={} debug_assertions={} executable={:?} epsilon={EPSILON:.17e} calls_cap={CALLS} held_out_seed=153",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        file!(),
        std::env::consts::ARCH,
        std::env::consts::OS,
        usize::BITS,
        cfg!(debug_assertions),
        std::env::current_exe()
    );
    let result = catch_unwind(AssertUnwindSafe(|| run(&mut progress)));
    match result {
        Ok(Ok(())) => progress.summary("PASS"),
        Ok(Err(error)) => {
            println!("failure {} error={error}", progress.context());
            progress.summary("FAIL");
            panic!("calibration criterion not met: {error}");
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic");
            println!("runtime_panic {} error={message}", progress.context());
            progress.summary("FAIL");
            panic!("calibration runtime panic; no retry");
        }
    }
}

#[test]
fn non_he_corpus_maps_and_old_hidden() {
    let c = corpus();
    assert_eq!(c.len(), VECTORS);
    assert!(
        c.iter()
            .all(|v| v.m.len() == SLOTS && v.m.iter().all(|x| x.is_finite() && x.abs() <= 0.5))
    );
    assert_eq!(
        old_hidden(true),
        [4225.0 / 16384.0, 225.0 / 16384.0, 1.0 / 64.0]
    );
    assert_eq!(
        old_hidden(false),
        [5313.0 / 16384.0, -1071.0 / 16384.0, 0.0]
    );
    assert_eq!(c[4].m[0], -0.5);
    assert_eq!(c[4].m[511], 0.5);
    // Rational oracle for the ramp, allowing only f64 expression rounding here.
    // This plaintext oracle tolerance is never used by any calibration gate.
    assert!((c[4].m[255] + 1.0 / 1022.0).abs() <= f64::EPSILON);
    assert_eq!(c[5].m.iter().filter(|&&x| x != 0.0).count(), 1);
    assert_eq!(c[5].m[0], 0.5);
    assert_eq!(c[6].m.iter().filter(|&&x| x != 0.0).count(), 1);
    assert_eq!(c[6].m[511], -0.5);
    for (k, amp) in [(7, 0.4), (9, 0.1)] {
        for s in 0..SLOTS {
            let (row, col) = (s / 4, s % 4);
            let re = if row < 3 && col < 3 {
                if col == 1 { -amp } else { amp }
            } else {
                0.0
            };
            let ce = if row < 3 && col < 3 {
                if row == 1 { -amp } else { amp }
            } else {
                0.0
            };
            assert_eq!(c[k].m[s], re);
            assert_eq!(c[k + 1].m[s], ce);
        }
        assert_ne!(c[k].m[1], c[k + 1].m[1]);
    }
    for (k, a) in [
        (11, [0.4, -0.4, 0.4]),
        (13, [0.5, -0.5, 0.5]),
        (15, [4225.0 / 16384.0, 225.0 / 16384.0, 1.0 / 64.0]),
        (17, [5313.0 / 16384.0, -1071.0 / 16384.0, 0.0]),
    ] {
        for s in 0..SLOTS {
            assert_eq!(
                c[k].m[s],
                match s {
                    0..=3 => a[0],
                    4..=7 => a[1],
                    8..=11 => a[2],
                    _ => 0.0,
                }
            );
            assert_eq!(
                c[k + 1].m[s],
                match s % 4 {
                    0 => a[0],
                    1 => a[1],
                    2 => a[2],
                    _ => 0.0,
                }
            );
        }
    }
    let initial_w1_re = [
        0.25, -0.0625, 0.125, 0.0, -0.125, 0.1875, 0.0625, 0.0, 0.0625, 0.125, -0.1875, 0.0,
    ];
    let initial_w2_ce = [
        0.1875, -0.125, 0.0625, 0.0, 0.0625, 0.25, -0.125, 0.0, -0.125, 0.0625, 0.1875, 0.0,
    ];
    assert_eq!(&c[19].m[..12], initial_w1_re);
    assert_eq!(&c[20].m[..12], initial_w2_ce);
    assert!(
        c[19].m[12..]
            .iter()
            .chain(&c[20].m[12..])
            .all(|&x| x == 0.0)
    );
}

#[test]
fn non_he_seed_schedule_and_call_cap() {
    let seeds: Vec<_> = (0..2)
        .flat_map(|j| (0..VECTORS).map(move |k| encryption_seed(j, k)))
        .collect();
    assert_eq!(seeds, (100..=141).collect::<Vec<u8>>());
    let mut progress = Progress::new();
    for j in 0..2 {
        for k in 0..VECTORS {
            for pass in 1..=2 {
                progress.key = j;
                progress.vector = k;
                progress.pass = pass;
                assert_eq!(progress.index(), progress.validated);
                progress.validate();
            }
        }
    }
    assert_eq!(progress.validated, CALLS);
    assert_eq!(progress.coverage, [3; CALLS]);
}

#[test]
fn non_he_complex_metrics_and_gate_negatives() {
    let zero = vec![(0.0, 0.0); SLOTS];
    let mut actual = zero.clone();
    actual[511] = (3.0, 4.0); // includes final padding slot and imaginary contribution
    let m = metric(&actual, &zero).unwrap();
    assert_eq!(
        (m.max, m.slot, m.real_error, m.imaginary_error),
        (5.0, 511, 3.0, 4.0)
    );
    actual[511] = (0.0, EPSILON * 2.0);
    assert!(limit("imaginary", metric(&actual, &zero).unwrap().max, EPSILON).is_err());
    actual[511] = (f64::NAN, 0.0);
    assert!(metric(&actual, &zero).is_err());
    actual[511] = (0.0, f64::INFINITY);
    assert!(metric(&actual, &zero).is_err());
    assert!(metric(&zero[..511], &zero).is_err());
    assert!(limit("boundary", EPSILON, EPSILON).is_ok());
    assert!(limit("next_float", f64::from_bits(EPSILON.to_bits() + 1), EPSILON).is_err());
    assert!(limit("a1", 1.5 * EPSILON, EPSILON).is_err());
    assert!(limit("final", 2.5 * EPSILON, 2.0 * EPSILON).is_err());
    assert!(input_range(1, 0.5 + EPSILON, EPSILON).is_ok());
    assert!(input_range(1, 0.5 + EPSILON, 0.0).is_err());
    assert!(input_range(1, 0.5 + 2.0 * EPSILON, 2.0 * EPSILON).is_err());
    assert!(input_range(2, 0.5 + 2.0 * EPSILON, 2.0 * EPSILON).is_ok());
    assert!(input_range(2, 0.5 + 3.0 * EPSILON, 3.0 * EPSILON).is_err());
}

#[test]
fn non_he_metadata_and_counter_negatives() {
    let good = || Metadata {
        level: 16,
        scale_bits: DELTA.to_bits(),
        components: 2,
        component_levels: vec![16, 16],
        component_context: vec![true, true],
        component_ntt: vec![true, true],
    };
    check_metadata(&good(), 16).unwrap();
    let mut bad = good();
    bad.level = 15;
    assert!(check_metadata(&bad, 16).is_err());
    let mut bad = good();
    bad.scale_bits += 1;
    assert!(check_metadata(&bad, 16).is_err());
    let mut bad = good();
    bad.components = 3;
    assert!(check_metadata(&bad, 16).is_err());
    let mut bad = good();
    bad.component_levels[1] = 15;
    assert!(check_metadata(&bad, 16).is_err());
    let mut bad = good();
    bad.component_context[1] = false;
    assert!(check_metadata(&bad, 16).is_err());
    let mut bad = good();
    bad.component_ntt[1] = false;
    assert!(check_metadata(&bad, 16).is_err());
    let expected = OpCounter {
        bootstraps: 1,
        ..OpCounter::default()
    };
    check_counter(&expected, &expected).unwrap();
    for field in 0..7 {
        let mut bad = expected.clone();
        let fields = [
            &mut bad.rotations,
            &mut bad.relins,
            &mut bad.ct_ct_tensor,
            &mut bad.ct_pt_mults,
            &mut bad.rescales,
            &mut bad.sse_switches,
            &mut bad.bootstraps,
        ];
        *fields.into_iter().nth(field).unwrap() += 1;
        assert!(check_counter(&bad, &expected).is_err());
    }
    assert!(check_counter(&OpCounter::default(), &OpCounter::default()).is_err());
}

#[test]
fn non_he_registered_geometry_lattice() {
    for classes in [3, 2] {
        let stage = StageShape::new(3, 3, classes).unwrap();
        let plan = PackedGeometryPlan::new(3, &[stage, stage], SLOTS).unwrap();
        assert_eq!(plan.layout().r(), 128);
        assert_eq!(plan.layout().c(), 4);
        let mut steps: Vec<_> = plan
            .layers()
            .iter()
            .flat_map(|layer| layer.layout().galois_steps())
            .collect();
        steps.sort_unstable();
        steps.dedup();
        assert_eq!(steps, [-2, -1, 1, 2, 4, 8, 16, 32, 64, 128, 256]);
    }
}
