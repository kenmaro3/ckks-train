//! Test cases of GPU key switching, relinearization and rotations (Task 46), shared by the GPU
//! tests (`tests/gpu_ks.rs`) and the CPU replay of the kernels (`src/emu_ks.rs`): both
//! implement [`KsExec`] for a [`KsFixture`] (keys, CPU reference evaluator) and call the same
//! `check_*` functions.
//!
//! Every check compares with the CPU (`hekit_ckks::keyswitch::key_switch`, `Evaluator`) **bit
//! for bit**: every polynomial of every component, the level and the scale (as `f64` bits).

use std::sync::Arc;

use hekit_ckks::keys::conjugation_elt;
use hekit_ckks::keyswitch::key_switch;
use hekit_ckks::{
    Ciphertext, CkksParams, Encoder, Encryptor, Evaluator, GaloisKey, KeyGenerator, RelinKey,
};
use hekit_core::RnsPoly;
use hekit_core::sampler::Rng;

use super::{assert_same, random_poly, tag};

/// Keys (generated on the CPU from a fixed seed) and the CPU reference evaluator.
pub struct KsFixture {
    /// The parameters.
    pub params: Arc<CkksParams>,
    /// The CPU evaluator holding the relinearization key (if any) and the Galois keys.
    pub eval: Evaluator,
    /// The (nontrivial) rotation steps with a Galois key.
    pub steps: Vec<i32>,
    /// Whether the conjugation key is present.
    pub conjugation: bool,
    /// Keys are put on the device for levels `<= key_level` only (`params.max_level()` for
    /// full keys).
    pub key_level: usize,
    enc: Encoder,
    encryptor: Encryptor,
}

fn rng(seed: u64) -> Rng {
    let mut s = [0u8; 32];
    s[..8].copy_from_slice(&seed.to_le_bytes());
    Rng::from_seed(s)
}

impl KsFixture {
    /// Generates, in this order, the relinearization key (`relin`), the Galois keys of `steps`
    /// and the conjugation key (`conjugation`). `key_level`: `None` for full keys.
    pub fn new(
        params: CkksParams,
        relin: bool,
        steps: &[i32],
        conjugation: bool,
        key_level: Option<usize>,
        seed: u64,
    ) -> KsFixture {
        let params = Arc::new(params);
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let rk = relin.then(|| kg.relin_key());
        let mut gks = kg.galois_keys(steps);
        if conjugation {
            gks.insert(kg.conjugation_key());
        }
        KsFixture {
            eval: Evaluator::new(Arc::clone(&params), rk, Some(gks)),
            steps: steps.to_vec(),
            conjugation,
            key_level: key_level.unwrap_or(params.max_level()),
            enc: Encoder::new(Arc::clone(&params)),
            encryptor: Encryptor::with_secret_key(Arc::clone(&params), kg.secret_key()),
            params,
        }
    }

    /// The same keys (shared, not copied) put on the device for levels `<= key_level` only
    /// (smaller device keys: [`KsFixture::key_level`]).
    pub fn capped(&self, key_level: usize) -> KsFixture {
        assert!(key_level <= self.params.max_level());
        KsFixture {
            params: Arc::clone(&self.params),
            eval: self.eval.clone(),
            steps: self.steps.clone(),
            conjugation: self.conjugation,
            key_level,
            enc: self.enc.clone(),
            encryptor: self.encryptor.clone(),
        }
    }

    /// The relinearization key, if generated.
    pub fn relin_key(&self) -> Option<&RelinKey> {
        self.eval.relin_key()
    }

    /// The Galois keys: one per step of [`KsFixture::steps`] (distinct elements), then the
    /// conjugation key.
    pub fn galois_keys(&self) -> Vec<&GaloisKey> {
        let gks = self.eval.galois_keys().expect("fixture has Galois keys");
        let mut out: Vec<&GaloisKey> = Vec::new();
        for &s in &self.steps {
            let k = gks.get(s).expect("key for every step");
            if !out.iter().any(|o| o.elt() == k.elt()) {
                out.push(k);
            }
        }
        if self.conjugation {
            out.push(gks.get_elt(conjugation_elt(self.params.n())).unwrap());
        }
        out
    }

    /// A fresh encryption of random values in `[-1, 1)` at `level` (scale `Delta`).
    pub fn encrypt(&self, level: usize, seed: u64) -> Ciphertext {
        let mut r = rng(seed);
        let v: Vec<f64> = (0..self.params.slots())
            .map(|_| (r.next_u64() >> 11) as f64 * 2f64.powi(-52) - 1.0)
            .collect();
        let pt = self.enc.encode(&v, self.params.scale(), level).unwrap();
        self.encryptor.encrypt(&pt, &mut r)
    }

    /// The levels of `wanted` that the device keys serve (`<= key_level`), deduplicated.
    pub fn levels(&self, wanted: &[usize]) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        for &l in wanted {
            if l <= self.key_level && !out.contains(&l) {
                out.push(l);
            }
        }
        out
    }
}

/// Which key a key switch uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyChoice {
    /// The relinearization key.
    Relin,
    /// The Galois key of this element.
    Galois(u64),
}

/// One implementation of the operations under test, on host values (the GPU version uploads
/// the inputs, runs the device operations and downloads every result; the replay runs the
/// shared launch sequences on the kernel replays). Keys come from the fixture the
/// implementation was built for, put on the device for levels `<= key_level`.
pub trait KsExec {
    /// `hekit_ckks::keyswitch::key_switch(params, d, key)` (decompose, one inner-product
    /// launch, two `mod_down`s).
    fn key_switch(&self, d: &RnsPoly, key: KeyChoice) -> (RnsPoly, RnsPoly);
    /// `[mul_no_relin(a, b), relinearize(..), rescale(..)]`, chained on the device.
    fn mul_relin_rescale(&self, a: &Ciphertext, b: &Ciphertext) -> [Ciphertext; 3];
    /// `rotate(ct, steps)`.
    fn rotate(&self, ct: &Ciphertext, steps: i32) -> Ciphertext;
    /// `conjugate(ct)`.
    fn conjugate(&self, ct: &Ciphertext) -> Ciphertext;
    /// `hoisted_rotate(ct, steps)`.
    fn hoisted_rotate(&self, ct: &Ciphertext, steps: &[i32]) -> Vec<Ciphertext>;
}

/// Bit-for-bit equality of two ciphertexts: component count, level, scale bits, polynomials.
pub fn assert_same_ct(got: &Ciphertext, want: &Ciphertext, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: component count");
    assert_eq!(got.level(), want.level(), "{what}: level");
    assert_eq!(
        got.scale().to_bits(),
        want.scale().to_bits(),
        "{what}: scale {} vs CPU {}",
        got.scale(),
        want.scale()
    );
    for (i, (g, w)) in got.polys().iter().zip(want.polys()).enumerate() {
        assert_same(g, w, &format!("{what}: c{i}"));
    }
}

/// The levels most checks use: top, top - 1 (a level-clipped last digit when `alpha > 1`),
/// `L/2 - 2` (another clipped digit), 1 and 0 (one digit), restricted to the fixture's key
/// level.
fn spread(fx: &KsFixture) -> Vec<usize> {
    let l = fx.params.max_level();
    fx.levels(&[fx.key_level, l, l - 1, l / 2 - 2, 3, 1, 0])
}

/// `key_switch` of random NTT-form polynomials against `hekit_ckks::keyswitch::key_switch`:
/// with the relinearization key at every level of [`spread`], with the first Galois key at the
/// top level.
pub fn check_key_switch<E: KsExec>(e: &E, fx: &KsFixture) {
    let p = &fx.params;
    let mut cases: Vec<(usize, KeyChoice)> = Vec::new();
    if fx.relin_key().is_some() {
        cases.extend(spread(fx).into_iter().map(|l| (l, KeyChoice::Relin)));
    }
    if let Some(gk) = fx.galois_keys().first() {
        cases.push((fx.key_level, KeyChoice::Galois(gk.elt())));
    }
    assert!(!cases.is_empty(), "fixture without keys");
    for (i, (level, key)) in cases.into_iter().enumerate() {
        let d = random_poly(p.basis_q(), level, true, 4600 + i as u64);
        let ksk = match key {
            KeyChoice::Relin => fx.relin_key().unwrap().key(),
            KeyChoice::Galois(g) => fx.eval.galois_keys().unwrap().get_elt(g).unwrap().key(),
        };
        let (wb, wa) = key_switch(p, &d, ksk);
        let (gb, ga) = e.key_switch(&d, key);
        let what = format!("key_switch {key:?} {} level {level}", tag(p));
        assert_same(&gb, &wb, &format!("{what} (kb)"));
        assert_same(&ga, &wa, &format!("{what} (ka)"));
    }
}

/// `relinearize(mul_no_relin(a, b))` then `rescale` of fresh encryptions, every stage against
/// the CPU evaluator, at the top level, `L / 2` and 1.
pub fn check_mul_relin_rescale<E: KsExec>(e: &E, fx: &KsFixture) {
    let l = fx.params.max_level();
    for (i, level) in fx
        .levels(&[fx.key_level, l, l / 2, 1])
        .into_iter()
        .enumerate()
    {
        let (a, b) = (
            fx.encrypt(level, 4700 + 2 * i as u64),
            fx.encrypt(level, 4701 + 2 * i as u64),
        );
        let t = fx.eval.mul_no_relin(&a, &b).unwrap();
        let r = fx.eval.relinearize(&t).unwrap();
        let s = fx.eval.rescale(&r).unwrap();
        let got = e.mul_relin_rescale(&a, &b);
        let what = |op: &str| format!("{op} {} level {level}", tag(&fx.params));
        assert_same_ct(&got[0], &t, &what("mul_no_relin"));
        assert_same_ct(&got[1], &r, &what("relinearize(mul_no_relin)"));
        assert_same_ct(&got[2], &s, &what("rescale(relinearize(mul_no_relin))"));
    }
}

/// `rotate` by every step of the fixture, by 0 and by `slots` (copies, no key), and
/// `conjugate` when the fixture has the key, against the CPU evaluator at the levels of
/// [`spread`].
pub fn check_rotate<E: KsExec>(e: &E, fx: &KsFixture) {
    let slots = fx.params.slots() as i32;
    let mut steps = fx.steps.clone();
    steps.extend([0, slots]);
    for (i, level) in spread(fx).into_iter().enumerate() {
        let ct = fx.encrypt(level, 4800 + i as u64);
        for &s in &steps {
            let what = format!("rotate({s}) {} level {level}", tag(&fx.params));
            assert_same_ct(&e.rotate(&ct, s), &fx.eval.rotate(&ct, s).unwrap(), &what);
        }
        if fx.conjugation {
            let what = format!("conjugate {} level {level}", tag(&fx.params));
            assert_same_ct(&e.conjugate(&ct), &fx.eval.conjugate(&ct).unwrap(), &what);
        }
    }
}

/// `hoisted_rotate(ct, steps ++ [0, steps[0]])` (a copy and a repeated step included) against
/// the CPU's **individual** `rotate` of every step, at each of `levels` (restricted to the key
/// level), and the empty step list.
pub fn check_hoisted_rotate<E: KsExec>(e: &E, fx: &KsFixture, levels: &[usize]) {
    let mut steps = fx.steps.clone();
    steps.extend([0, fx.steps[0]]);
    for (i, level) in fx.levels(levels).into_iter().enumerate() {
        let ct = fx.encrypt(level, 4900 + i as u64);
        let got = e.hoisted_rotate(&ct, &steps);
        assert_eq!(got.len(), steps.len());
        for (g, &s) in got.iter().zip(&steps) {
            let what = format!(
                "hoisted_rotate step {s} of {} {} level {level}",
                steps.len(),
                tag(&fx.params)
            );
            assert_same_ct(g, &fx.eval.rotate(&ct, s).unwrap(), &what);
        }
    }
    let ct = fx.encrypt(fx.key_level, 4999);
    assert!(e.hoisted_rotate(&ct, &[]).is_empty());
}
