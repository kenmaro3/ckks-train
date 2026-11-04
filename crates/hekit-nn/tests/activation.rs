//! Task 29: polynomial activations and the MSE loss (`hekit_nn::activation`, `hekit_nn::loss`).
//!
//! Every activation's forward, derivative factor (with a folded constant and mask) and backward
//! are decrypted and compared slot by slot with the plaintext polynomial within 2^-15; the
//! levels actually consumed equal `depth()` / `derivative_depth()`; the exact-scale rule holds.
//! seal_2020 (one shared `Arc<CkksParams>`), ChaCha20 fixed seeds.

use std::sync::Arc;

use hekit_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter,
};
use hekit_core::error::HekitError;
use hekit_core::sampler::Rng;
use hekit_nn::activation::{Activation, Poly, PolyRelu2, Square};
use hekit_nn::loss::Mse;

const TOL: f64 = 1.0 / 32768.0;

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

struct Ctx {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
    rng: Rng,
}

fn ctx(seed: u8) -> Ctx {
    let params = Arc::new(CkksParams::seal_2020());
    let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
    let pk = kg.public_key();
    let rk = kg.relin_key();
    Ctx {
        encoder: Encoder::new(Arc::clone(&params)),
        enc: Encryptor::new(Arc::clone(&params), pk),
        dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
        ev: Evaluator::new(Arc::clone(&params), Some(rk), None),
        params,
        rng: rng(seed.wrapping_add(100)),
    }
}

impl Ctx {
    fn slots(&self) -> usize {
        self.params.slots()
    }
    fn rand(&mut self) -> Vec<f64> {
        (0..self.slots()).map(|_| uni(&mut self.rng)).collect()
    }
    fn encrypt_at(&mut self, v: &[f64], scale: f64, level: usize) -> Ciphertext {
        let pt = self.encoder.encode(v, scale, level).unwrap();
        self.enc.encrypt(&pt, &mut self.rng)
    }
    fn encrypt_top(&mut self, v: &[f64]) -> Ciphertext {
        self.encrypt_at(v, self.params.scale(), self.params.max_level())
    }
    fn decrypt(&self, ct: &Ciphertext) -> Vec<f64> {
        self.encoder.decode(&self.dec.decrypt(ct))
    }
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.min(b)
}

/// Plaintext model of an activation: `(f, f')`.
type Model = (Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>);

fn poly_model(c: Vec<f64>) -> Model {
    let d = c.clone();
    (
        Box::new(move |z| c.iter().rev().fold(0.0, |a, &k| a * z + k)),
        Box::new(move |z| {
            d.iter()
                .enumerate()
                .skip(1)
                .rev()
                .fold(0.0, |a, (i, &k)| a * z + i as f64 * k)
        }),
    )
}

fn cases() -> Vec<(&'static str, Box<dyn Activation>, Model)> {
    let mut v: Vec<(&'static str, Box<dyn Activation>, Model)> = vec![
        ("square", Box::new(Square), poly_model(vec![0.0, 0.0, 1.0])),
        (
            "polyrelu2",
            Box::new(PolyRelu2),
            poly_model(vec![0.0, 1.0, 1.0]),
        ),
    ];
    let polys: [(&'static str, Vec<f64>); 4] = [
        ("poly2", vec![0.2, -0.3, 0.7]),
        ("poly3", vec![0.1, 0.4, -0.5, 0.3]),
        ("poly4", vec![0.1, -0.3, 0.5, 0.2, -0.4]),
        ("poly4_sparse", vec![0.0, 0.0, 0.5, 0.0, 0.25]),
    ];
    for (name, c) in polys {
        v.push((name, Box::new(Poly::new(c.clone()).unwrap()), poly_model(c)));
    }
    v
}

#[test]
fn depths() {
    assert_eq!(Square.depth(), 1);
    assert_eq!(PolyRelu2.depth(), 1);
    assert_eq!(
        Poly::new(vec![0.1, -0.3, 0.5, 0.2, -0.4]).unwrap().depth(),
        3
    );
    assert_eq!(Poly::new(vec![0.1, 0.4, -0.5, 0.3]).unwrap().depth(), 2);
    assert_eq!(Poly::new(vec![0.2, -0.3, 0.7]).unwrap().depth(), 2);
    // Trailing zeros are trimmed.
    assert_eq!(
        Poly::new(vec![0.2, -0.3, 0.7, 0.0, 0.0]).unwrap().depth(),
        2
    );
    assert!(
        Poly::new(vec![1.0, 2.0]).is_err(),
        "degree 1: constant derivative"
    );
    assert!(
        Poly::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).is_err(),
        "degree 5"
    );
}

#[test]
fn forward_matches_plaintext_and_consumes_depth() {
    let mut c = ctx(1);
    for (name, act, (f, _)) in cases() {
        let zv = c.rand();
        let z = c.encrypt_top(&zv);
        c.ev.reset_counter();
        let a = act.forward(&c.ev, &c.encoder, &z).unwrap();
        assert_eq!(
            z.level() - a.level(),
            act.depth(),
            "{name}: levels consumed"
        );
        let want: Vec<f64> = zv.iter().map(|&x| f(x)).collect();
        let err = max_diff(&c.decrypt(&a), &want);
        assert!(err < TOL, "{name}: forward err {err:e}");
        // The result is at (essentially) the scale of its own history: usable downstream.
        assert!(a.scale() > 2f64.powi(30), "{name}: scale {}", a.scale());
        eprintln!("{name}: forward err 2^{:.1}", err.log2());
    }
}

#[test]
fn forward_from_lower_level() {
    let mut c = ctx(2);
    let zv = c.rand();
    let lvl = 5;
    let z = c.encrypt_at(&zv, c.params.scale(), lvl);
    let p = Poly::new(vec![0.1, -0.3, 0.5, 0.2, -0.4]).unwrap();
    let a = p.forward(&c.ev, &c.encoder, &z).unwrap();
    assert_eq!(a.level(), lvl - 3);
    let (f, _) = poly_model(vec![0.1, -0.3, 0.5, 0.2, -0.4]);
    let want: Vec<f64> = zv.iter().map(|&x| f(x)).collect();
    assert!(max_diff(&c.decrypt(&a), &want) < TOL);
    // Not enough levels is an error, not a panic.
    let z2 = c.encrypt_at(&zv, c.params.scale(), 2);
    assert!(matches!(
        p.forward(&c.ev, &c.encoder, &z2),
        Err(HekitError::NoLevelsLeft)
    ));
}

#[test]
fn derivative_factor_folds_constant_and_mask() {
    let mut c = ctx(3);
    let k = 0.37;
    for (name, act, (_, df)) in cases() {
        let zv = c.rand();
        let mask: Vec<f64> = (0..c.slots())
            .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 })
            .collect();
        let z = c.encrypt_top(&zv);
        c.ev.reset_counter();
        let s = act
            .derivative_factor(&c.ev, &c.encoder, &z, k, Some(&mask))
            .unwrap();
        let cnt: OpCounter = c.ev.counter();
        assert_eq!(
            z.level() - s.level(),
            act.derivative_depth(),
            "{name}: levels"
        );
        assert!(
            rel(s.scale(), z.scale()) < 1e-12,
            "{name}: scale kept at z's"
        );
        let want: Vec<f64> = zv.iter().zip(&mask).map(|(&x, &m)| k * m * df(x)).collect();
        let err = max_diff(&c.decrypt(&s), &want);
        assert!(err < TOL, "{name}: s err {err:e}");
        // Padded lanes (mask 0) decrypt to ~0: the mask costs no extra level.
        let zero_lane = c.decrypt(&s)[0].abs();
        assert!(zero_lane < TOL, "{name}: masked lane {zero_lane:e}");
        eprintln!("{name}: derivative err 2^{:.1}, {cnt:?}", err.log2());
    }
}

#[test]
fn derivative_factor_target_scale_knob() {
    let mut c = ctx(4);
    let target = 2f64.powi(40) * 1.001; // deliberately not Delta
    for (name, act, (_, df)) in cases() {
        let zv = c.rand();
        let z = c.encrypt_top(&zv);
        let s = act
            .derivative_factor_at(&c.ev, &c.encoder, &z, 1.0, None, Some(target))
            .unwrap();
        assert!(
            rel(s.scale(), target) < 1e-12,
            "{name}: {} vs {target}",
            s.scale()
        );
        let want: Vec<f64> = zv.iter().map(|&x| df(x)).collect();
        assert!(max_diff(&c.decrypt(&s), &want) < TOL, "{name}");
    }
}

#[test]
fn backward_matches_plaintext() {
    let mut c = ctx(5);
    for (name, act, (_, df)) in cases() {
        let zv = c.rand();
        let uv = c.rand();
        let z = c.encrypt_top(&zv);
        // Upstream at a different level than the derivative factor, to exercise alignment.
        let u = c.encrypt_at(&uv, c.params.scale(), c.params.max_level() - 1);
        let b = act.backward(&c.ev, &c.encoder, &z, &u).unwrap();
        let lvl = (z.level() - act.derivative_depth()).min(u.level());
        assert_eq!(b.level(), lvl - 1, "{name}: backward level");
        let want: Vec<f64> = zv.iter().zip(&uv).map(|(&x, &y)| df(x) * y).collect();
        let err = max_diff(&c.decrypt(&b), &want);
        assert!(err < TOL, "{name}: backward err {err:e}");
        eprintln!("{name}: backward err 2^{:.1}", err.log2());
    }
}

#[test]
fn mse_grad_is_pred_minus_target_without_relin() {
    let mut c = ctx(6);
    let tv = c.rand();
    let act = Square;
    let r = c.rand();
    let z = c.encrypt_top(&r);
    let pred = act.forward(&c.ev, &c.encoder, &z).unwrap();
    // The client encrypts the target at pred's exact scale; it may be at a higher level.
    let target = c.encrypt_at(&tv, pred.scale(), pred.level() + 1);
    c.ev.reset_counter();
    let g = Mse.grad(&c.ev, &pred, &target).unwrap();
    let cnt = c.ev.counter();
    assert_eq!(cnt.relins, 0);
    assert_eq!(cnt.ct_ct_tensor, 0);
    assert_eq!(cnt.ct_pt_mults, 0);
    assert_eq!(cnt.rescales, 0);
    assert_eq!(g.level(), pred.level());
    let zv = c.decrypt(&z);
    let want: Vec<f64> = zv.iter().zip(&tv).map(|(&x, &t)| x * x - t).collect();
    assert!(max_diff(&c.decrypt(&g), &want) < TOL);

    // A pred below the target's level and a target below pred's level.
    let low = c.ev.mod_switch_to_level(&pred, pred.level() - 1).unwrap();
    let g2 = Mse.grad(&c.ev, &low, &target).unwrap();
    assert_eq!(g2.level(), low.level());
    let g3 = Mse.grad(&c.ev, &target, &low).unwrap();
    assert_eq!(g3.level(), low.level());
}

#[test]
fn mse_rejects_mismatched_scales() {
    let mut c = ctx(7);
    let v = c.rand();
    let a = c.encrypt_top(&v);
    // Within the evaluator's 2^-10 tolerance, but not the same history: rejected.
    let b = c.encrypt_at(
        &v,
        c.params.scale() * (1.0 + 2f64.powi(-14)),
        c.params.max_level(),
    );
    assert!(matches!(
        Mse.grad(&c.ev, &a, &b),
        Err(HekitError::ScaleMismatch)
    ));
}
