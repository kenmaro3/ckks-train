//! Test cases of the CUDA evaluator (Task 47), generic over `hekit_ckks::HeOps`: the GPU
//! tests (`tests/gpu_eval.rs`) run them with `CudaEvaluator`, the always-built unit tests
//! (`src/matvec.rs`) with a second CPU `Evaluator`, which checks the harness and anchors its
//! expected counts on every machine.
//!
//! Every check compares with a CPU `Evaluator` holding the same keys: each result **bit for
//! bit** (components, level, scale bits, residues) and the whole `OpCounter` **exactly**,
//! including after operations that must fail (which count nothing).

use std::sync::Arc;

use hekit_ckks::keys::conjugation_elt;
use hekit_ckks::{
    Ciphertext, CkksParams, Diagonals, Encoder, Encryptor, EvalConfig, Evaluator, GaloisKey,
    GaloisKeys, HeOps, KeyGenerator, OpCounter, Plaintext, RelinKey, bsgs_rotation_steps,
};
use hekit_core::HekitError;
use hekit_core::sampler::Rng;

use super::ks::assert_same_ct;

fn rng(seed: u64) -> Rng {
    let mut s = [0u8; 32];
    s[..8].copy_from_slice(&seed.to_le_bytes());
    Rng::from_seed(s)
}

/// Keys from a fixed seed and the host-side encoder / encryptor.
pub struct EvalFixture {
    /// The parameters.
    pub params: Arc<CkksParams>,
    /// The relinearization key.
    pub rk: RelinKey,
    /// The Galois keys: the steps of [`EvalFixture::steps`] and the conjugation key.
    pub gks: GaloisKeys,
    /// The rotation steps with a key (the BSGS steps of `d` and one extra step).
    pub steps: Vec<i32>,
    /// The matrix dimension of the matvec checks.
    pub d: usize,
    enc: Encoder,
    encryptor: Encryptor,
}

impl EvalFixture {
    /// Keys for the BSGS steps of dimension `d` (stride 1), the extra step `d + 1` and the
    /// conjugation.
    pub fn new(params: CkksParams, d: usize, seed: u64) -> EvalFixture {
        let params = Arc::new(params);
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let rk = kg.relin_key();
        let mut steps = bsgs_rotation_steps(d, 1);
        steps.push(d as i32 + 1);
        let mut gks = kg.galois_keys(&steps);
        gks.insert(kg.conjugation_key());
        EvalFixture {
            enc: Encoder::new(Arc::clone(&params)),
            encryptor: Encryptor::with_secret_key(Arc::clone(&params), kg.secret_key()),
            params,
            rk,
            gks,
            steps,
            d,
        }
    }

    /// Every Galois key (for a device upload).
    pub fn galois_keys(&self) -> Vec<&GaloisKey> {
        self.gks.iter().collect()
    }

    /// Number of Galois keys (steps plus conjugation).
    pub fn num_galois_keys(&self) -> usize {
        self.gks.len()
    }

    /// A CPU evaluator with copies of the keys and the flags `config`.
    pub fn cpu(&self, config: EvalConfig) -> Evaluator {
        Evaluator::with_config(
            Arc::clone(&self.params),
            Some(self.rk.clone()),
            Some(self.gks.clone()),
            config,
        )
    }

    fn values(&self, seed: u64) -> Vec<f64> {
        let mut r = rng(seed);
        (0..self.params.slots())
            .map(|_| (r.next_u64() >> 11) as f64 * 2f64.powi(-52) - 1.0)
            .collect()
    }

    /// Random values encoded at `scale` and `level`.
    pub fn encode(&self, scale: f64, level: usize, seed: u64) -> Plaintext {
        self.enc.encode(&self.values(seed), scale, level).unwrap()
    }

    /// A fresh encryption of random values at `level` (scale `Delta`).
    pub fn encrypt(&self, level: usize, seed: u64) -> Ciphertext {
        let pt = self.encode(self.params.scale(), level, seed);
        self.encryptor.encrypt(&pt, &mut rng(seed ^ 0xe1))
    }

    /// `d` random diagonals (pre-rotated layout, stride 1) at `level`, scale `Delta`: as
    /// plaintexts and encrypted.
    pub fn diagonals(
        &self,
        level: usize,
        seed: u64,
    ) -> (Diagonals<Plaintext>, Diagonals<Ciphertext>) {
        let pts: Vec<Plaintext> = (0..self.d)
            .map(|i| self.encode(self.params.scale(), level, seed + i as u64))
            .collect();
        let cts: Vec<Ciphertext> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                self.encryptor
                    .encrypt(p, &mut rng(seed ^ (0x100 + i as u64)))
            })
            .collect();
        (
            Diagonals::from_prerotated(self.d, 1, pts).unwrap(),
            Diagonals::from_prerotated(self.d, 1, cts).unwrap(),
        )
    }
}

/// The downloaded results of [`run_sequence`], by name.
pub type Results = Vec<(&'static str, Ciphertext)>;

/// A sequence of every `HeOps` operation on `ev` at `level >= 2`, with operands made on the
/// host from fixed seeds, returning every result downloaded; operations that must fail are
/// checked to fail with the CPU's error kind (and count nothing, which the caller sees in the
/// counter). The counts of one run with the default flags are [`sequence_counts`].
pub fn run_sequence<E: HeOps>(ev: &E, fx: &EvalFixture, level: usize, seed: u64) -> Results {
    assert!(level >= 2);
    let p = &fx.params;
    let delta = p.scale();
    let up = |c: &Ciphertext| ev.upload_ct(c).unwrap();
    let down = |c: &E::Ct| ev.download_ct(c).unwrap();
    let a = up(&fx.encrypt(level, seed));
    let b = up(&fx.encrypt(level, seed + 1));
    let pt = ev.upload_pt(&fx.encode(delta, level, seed + 2)).unwrap();
    let host_a = ev.download_ct(&a).unwrap();
    let sp = ev.target_plain_scale(&a, delta).unwrap();
    let pts = ev.upload_pt(&fx.encode(sp, level, seed + 3)).unwrap();
    let (pd, cd) = fx.diagonals(level, seed + 10);
    let pd = pd.try_map(|x| ev.upload_pt(x)).unwrap();
    let cd = cd.try_map(|x| ev.upload_ct(x)).unwrap();
    let s0 = fx.steps[0];
    let extra = *fx.steps.last().unwrap();
    let mut out: Results = Vec::new();
    let mut push = |name: &'static str, c: &E::Ct| out.push((name, down(c)));

    assert!(ev.owns(&a) && ev.has_relin_key() && ev.has_galois_key(s0));
    assert!(!ev.has_galois_key(0) && !ev.has_galois_key(extra + 1));
    push("upload/download", &a);
    push("add", &ev.add(&a, &b).unwrap());
    push("sub", &ev.sub(&a, &b).unwrap());
    push("negate", &ev.negate(&a).unwrap());
    push("add_plain", &ev.add_plain(&a, &pt).unwrap());
    let ap = ev.mul_plain(&a, &pt).unwrap();
    push("mul_plain", &ap);
    push("mul_scalar+", &ev.mul_scalar(&a, 0.75, delta).unwrap());
    push("mul_scalar-", &ev.mul_scalar(&a, -1.5, delta).unwrap());
    push(
        "mul_plain_to_scale",
        &ev.mul_plain_to_scale(&a, &pts, delta).unwrap(),
    );
    let t = ev.mul_no_relin(&a, &b).unwrap();
    push("mul_no_relin", &t);
    push("add 3+2", &ev.add(&t, &ap).unwrap());
    push("sub 2-3", &ev.sub(&ap, &t).unwrap());
    let r = ev.relinearize(&t).unwrap();
    push("relinearize", &r);
    let m = ev.mul(&a, &b).unwrap();
    push("mul", &m);
    push("rescale", &ev.rescale(&m).unwrap());
    let low = ev.mod_switch_to_level(&a, level - 1).unwrap();
    push("mod_switch_to_level", &low);
    push(
        "mod_switch same",
        &ev.mod_switch_to_level(&a, level).unwrap(),
    );
    push("relabel", &ev.relabel(a.clone(), 2.0 * delta).unwrap());
    push("rotate", &ev.rotate(&a, s0).unwrap());
    push("rotate extra", &ev.rotate(&a, extra).unwrap());
    push(
        "rotate slots (copy)",
        &ev.rotate(&a, p.slots() as i32).unwrap(),
    );
    push("rotate low level", &ev.rotate(&low, s0).unwrap());
    push("conjugate", &ev.conjugate(&a).unwrap());
    let h = ev.hoisted_rotate(&a, &[0, s0, extra, s0]).unwrap();
    assert_eq!(h.len(), 4);
    for c in &h {
        push("hoisted_rotate", c);
    }
    push(
        "lazy",
        &ev.relin_lazy(ev.mul_lazy(&a, &b).unwrap()).unwrap(),
    );
    push("matvec_plain", &ev.matvec_plain(&pd, &a).unwrap());
    push("matvec_enc", &ev.matvec_enc(&cd, &a).unwrap());

    // Failures: the CPU's error kinds, nothing counted.
    let before = ev.counter();
    let inv = |r: hekit_core::Result<E::Ct>, what: &str| {
        assert!(
            matches!(r, Err(HekitError::InvalidParams(_))),
            "{what}: {r:?}",
            r = r.err()
        );
    };
    inv(ev.rotate(&a, extra + 1), "rotate without a key");
    assert!(
        ev.hoisted_rotate(&a, &[s0, extra + 1]).is_err(),
        "hoisted without a key"
    );
    inv(ev.relinearize(&a), "relinearize 2 components");
    inv(ev.mul(&t, &a), "mul of 3 components");
    inv(ev.mul_scalar(&a, f64::NAN, delta), "mul_scalar NaN");
    inv(ev.mod_switch_to_level(&low, level), "mod_switch up");
    assert!(matches!(ev.add(&a, &low), Err(HekitError::LevelMismatch)));
    assert!(matches!(ev.mul(&a, &low), Err(HekitError::LevelMismatch)));
    assert!(matches!(
        ev.add(&a, &m),
        Err(HekitError::LevelMismatch) | Err(HekitError::ScaleMismatch)
    ));
    assert!(matches!(
        ev.add(&a, &ev.relabel(b.clone(), 4.0 * delta).unwrap()),
        Err(HekitError::ScaleMismatch)
    ));
    assert!(matches!(
        ev.mul_plain_to_scale(&a, &pt, delta),
        Err(HekitError::ScaleMismatch)
    ));
    let zero = ev.mod_switch_to_level(&a, 0).unwrap();
    assert!(matches!(
        ev.mul(&zero, &zero),
        Err(HekitError::NoLevelsLeft)
    ));
    assert!(matches!(ev.rescale(&zero), Err(HekitError::NoLevelsLeft)));
    assert!(matches!(
        ev.matvec_enc(&cd, &low),
        Err(HekitError::LevelMismatch)
    ));
    assert_eq!(ev.counter(), before, "failed operations must count nothing");
    // The host copy of `a` is unchanged by all of the above (functional operations).
    assert_same_ct(&ev.download_ct(&a).unwrap(), &host_a, "operand unchanged");
    out
}

/// The counts of one [`run_sequence`] with the default flags and dimension `d` (BSGS:
/// `baby + giant - 2` rotations, `giant` relins and `d` tensors in `matvec_enc`, `d`
/// plaintext products in `matvec_plain`).
pub fn sequence_counts(d: usize) -> OpCounter {
    let (baby, giant) = hekit_ckks::bsgs_split(d);
    let mv_rot = (baby + giant - 2) as u64;
    OpCounter {
        // rotate x3 (+1 trivial), conjugate, hoisted 3 nontrivial, two matvecs.
        rotations: 3 + 1 + 3 + 2 * mv_rot,
        // relinearize, mul, lazy, matvec_enc.
        relins: 3 + giant as u64,
        // mul_no_relin, mul, lazy, matvec_enc.
        ct_ct_tensor: 3 + d as u64,
        // mul_plain, mul_scalar x2, mul_plain_to_scale, matvec_plain.
        ct_pt_mults: 4 + d as u64,
        rescales: 1,
        sse_switches: 0,
        bootstraps: 0,
    }
}

/// [`run_sequence`] on `ev` and on the CPU (same keys, `ev`'s flags): every result equal bit
/// for bit, and the counters equal exactly (and equal to [`sequence_counts`] for the default
/// flags). Counters are reset first.
pub fn check_sequence<E: HeOps>(
    ev: &E,
    cpu: &Evaluator,
    fx: &EvalFixture,
    level: usize,
    seed: u64,
) {
    ev.reset_counter();
    cpu.reset_counter();
    let got = run_sequence(ev, fx, level, seed);
    let want = run_sequence(cpu, fx, level, seed);
    assert_eq!(got.len(), want.len());
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        assert_same_ct(g, w, &format!("{name} at level {level}"));
    }
    assert_eq!(ev.counter(), cpu.counter(), "counters at level {level}");
    if *cpu.config() == EvalConfig::default() {
        assert_eq!(cpu.counter(), sequence_counts(fx.d), "expected counts");
    }
}

/// Task 43 M3 conformance: `matvec_plain` / `matvec_enc` at `level` under `ev`'s flags (the
/// CPU evaluator `cpu` has the same flags): bit-identical results, equal counters, and the
/// closed forms `EvalConfig::matvec_rotations` / `matvec_enc_relins` (`d` tensors, `d`
/// plaintext products, nothing else).
pub fn check_matvec_conformance<E: HeOps>(
    ev: &E,
    cpu: &Evaluator,
    fx: &EvalFixture,
    level: usize,
    seed: u64,
) {
    let config = *ev.config();
    assert_eq!(config, *cpu.config());
    let d = fx.d;
    let x = fx.encrypt(level, seed);
    let (pd, cd) = fx.diagonals(level, seed + 50);
    let ux = ev.upload_ct(&x).unwrap();
    let upd = pd.try_map(|p| ev.upload_pt(p)).unwrap();
    let ucd = cd.try_map(|c| ev.upload_ct(c)).unwrap();
    let what = format!("{config:?} d {d} level {level}");

    ev.reset_counter();
    cpu.reset_counter();
    let got = ev
        .download_ct(&ev.matvec_plain(&upd, &ux).unwrap())
        .unwrap();
    let want = cpu.matvec_plain(&pd, &x).unwrap();
    assert_same_ct(&got, &want, &format!("matvec_plain {what}"));
    let c = ev.counter();
    assert_eq!(c, cpu.counter(), "matvec_plain counts {what}");
    assert_eq!(
        (c.rotations, c.relins, c.ct_pt_mults, c.ct_ct_tensor),
        (config.matvec_rotations(d), 0, d as u64, 0),
        "matvec_plain closed form {what}"
    );

    ev.reset_counter();
    cpu.reset_counter();
    let got = ev.download_ct(&ev.matvec_enc(&ucd, &ux).unwrap()).unwrap();
    let want = cpu.matvec_enc(&cd, &x).unwrap();
    assert_same_ct(&got, &want, &format!("matvec_enc {what}"));
    let c = ev.counter();
    assert_eq!(c, cpu.counter(), "matvec_enc counts {what}");
    assert_eq!(
        (c.rotations, c.relins, c.ct_ct_tensor, c.ct_pt_mults),
        (
            config.matvec_rotations(d),
            config.matvec_enc_relins(d),
            d as u64,
            0
        ),
        "matvec_enc closed form {what}"
    );
}

/// The conjugation element of the fixture's ring (for key lookups in tests).
pub fn conj_elt(fx: &EvalFixture) -> u64 {
    conjugation_elt(fx.params.n())
}
