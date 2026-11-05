//! Task 43: the training paths are generic over `HeOps` and honour the RQ3 ablation flags
//! (`EvalConfig`).
//!
//! - Interactive `train_batch` (the Task 30 network on seal_2020): `hoist` off gives the
//!   default counter and bit-identical weights; all flags off changes exactly the BSGS matvecs
//!   (forward and backward: `m - 1` rotations and `m` relins each instead of
//!   `baby + giant - 2` and `giant`), and the trained weights agree with the default within
//!   2^-20.
//! - The micro-batch weight gradient of the non-interactive update relinearizes every product
//!   with `lazy_relin` off (`k m` relins instead of `m`), same values.
//! - `RebootLayout::outer_sum` with `GradRelin::Lazy` acts as `PerSample` with `lazy_relin`
//!   off.
//! - A non-interactive update (real bootstrapper on the `mini` chain, 2 micro-batches, NAG)
//!   under every flag combination: the measured counter equals the config-aware closed form
//!   `predict_update` (plus the bootstraps' planned counts), and the state agrees with the
//!   default run within 2^-18.
//! - The backend helpers on the CPU: `Dense::upload` / `download` / `upload_owned`,
//!   `Mlp::predict_plain_on`, and `HostRefresher` around a `ClientRefresher`.

use std::sync::{Arc, OnceLock};

use hekit_ckks::bootstrap::{BootParams, Bootstrapper};
use hekit_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, EvalConfig, Evaluator, HeOps,
    KeyGenerator, OpCounter,
};
use hekit_core::sampler::Rng;
use hekit_nn::activation::PolyRelu2;
use hekit_nn::activation::{Activation, Square};
use hekit_nn::dense::{Dense, TransposeKeys, grad_reduced_rotation_steps};
use hekit_nn::layout::{LayerShape, Layout, pack_activations};
use hekit_nn::layout_reboot::{GradRelin, RebootLayout};
use hekit_nn::optim::Sgd;
use hekit_nn::refresh::{ClientRefresher, HostRefresher, Refresher};
use hekit_nn::trainer::{LayerOptimizer, Mlp, Trainer, mlp_rotation_steps};
use hekit_nn::trainer_ni::{
    BootPlacement, NiOptimizer, NonInteractiveTrainer, ni_rotation_steps, predict_update,
};
use ndarray::Array2;

const DIMS: [usize; 3] = [4, 8, 2];

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

fn rand_mat(r: &mut Rng, rows: usize, cols: usize, amp: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| amp * uni(r))
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

struct Ctx {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
    layout: Layout,
}

impl Ctx {
    /// An evaluator with this context's keys and the flags `cfg`.
    fn eval(&self, cfg: EvalConfig) -> Evaluator {
        Evaluator::with_config(
            Arc::clone(&self.params),
            self.ev.relin_key().cloned(),
            self.ev.galois_keys().cloned(),
            cfg,
        )
    }

    fn client(&self, seed: u8) -> ClientRefresher {
        ClientRefresher::new(
            self.dec.clone(),
            self.enc.clone(),
            Encoder::new(Arc::clone(&self.params)),
            rng(seed),
        )
    }

    fn decrypt(&self, ct: &Ciphertext) -> Vec<f64> {
        self.encoder.decode(&self.dec.decrypt(ct))
    }
}

/// seal_2020, layout d = 8, batch 8, keys for the Task 30 network's training step and for the
/// reduced gradient of its first layer.
fn ctx() -> &'static Ctx {
    static C: OnceLock<Ctx> = OnceLock::new();
    C.get_or_init(|| {
        let params = Arc::new(CkksParams::seal_2020());
        let layout = Layout::new(params.slots(), 8, 8).unwrap();
        let mut steps = mlp_rotation_steps(&DIMS, &layout, &TransposeKeys::Dedicated).unwrap();
        steps.extend(grad_reduced_rotation_steps(8, 4, &layout));
        steps.sort_unstable();
        steps.dedup();
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(43)).unwrap();
        let pk = kg.public_key();
        let rk = kg.relin_key();
        let gks = kg.galois_keys(&steps);
        Ctx {
            encoder: Encoder::new(Arc::clone(&params)),
            enc: Encryptor::new(Arc::clone(&params), pk),
            dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
            ev: Evaluator::new(Arc::clone(&params), Some(rk), Some(gks)),
            layout,
            params,
        }
    })
}

fn layers(r: &mut Rng) -> Vec<(Array2<f64>, Vec<f64>)> {
    DIMS.windows(2)
        .map(|p| {
            (
                rand_mat(r, p[1], p[0], 0.8 / (p[0] as f64).sqrt()),
                (0..p[1]).map(|_| 0.2 * uni(r)).collect(),
            )
        })
        .collect()
}

/// One SGD `train_batch` on `ev`: the counter of the step and the decrypted weights and biases
/// (all slots) after the refresh.
fn one_step(c: &Ctx, ev: &Evaluator) -> (OpCounter, Vec<Vec<f64>>) {
    let mut r = rng(77);
    let acts: Vec<Box<dyn Activation>> = vec![Box::new(Square), Box::new(Square)];
    let mut mlp = Mlp::encrypt(
        ev,
        &layers(&mut r),
        acts,
        &c.layout,
        &c.enc,
        &c.encoder,
        &mut r,
    )
    .unwrap();
    let spec = mlp.batch_spec(ev, &c.encoder).unwrap();
    let x = rand_mat(&mut r, 8, 4, 1.0);
    let y = rand_mat(&mut r, 8, 2, 1.0);
    let (xct, yct) = spec
        .encrypt(&c.enc, &c.encoder, &c.layout, &x, &y, &mut r)
        .unwrap();
    let optims: Vec<Box<dyn LayerOptimizer>> =
        vec![Box::new(Sgd { lr: 0.1 }), Box::new(Sgd { lr: 0.1 })];
    let mut trainer = Trainer::new(ev, Box::new(c.client(9)), optims, c.layout).unwrap();
    let before = ev.counter();
    trainer.train_batch(&mut mlp, &xct, &yct).unwrap();
    let count = &ev.counter() - &before;
    let mut vals = Vec::new();
    for (d, _) in &mlp.layers {
        for e in d.w.diagonals().entries() {
            vals.push(c.decrypt(e));
        }
        vals.push(c.decrypt(&d.b));
    }
    (count, vals)
}

#[test]
fn interactive_step_honours_the_flags() {
    let c = ctx();
    let (base, w0) = one_step(c, &c.eval(EvalConfig::default()));
    eprintln!("default: {base:?}");

    // hoist off: identical counts, and (the hoisted rotation being exact, the client refresh
    // seeded identically) identical weights.
    let no_hoist = EvalConfig {
        hoist: false,
        ..EvalConfig::default()
    };
    let (cnt, w1) = one_step(c, &c.eval(no_hoist));
    assert_eq!(cnt, base, "hoist changes no count");
    assert_eq!(w1, w0, "hoist changes no value");

    // All off: exactly the matvecs change.
    let (cnt, w2) = one_step(c, &c.eval(EvalConfig::NAIVE));
    eprintln!("naive: {cnt:?}");
    let mut want = base.clone();
    for (i, p) in DIMS.windows(2).enumerate() {
        let sh = LayerShape::new(p[1], p[0], &c.layout);
        let mut shapes = vec![sh];
        if i > 0 {
            shapes.push(sh.transposed());
        }
        for s in shapes {
            want.rotations = want.rotations + (s.m - 1) as u64 - (s.baby + s.giant - 2) as u64;
            want.relins = want.relins + s.m as u64 - s.giant as u64;
        }
    }
    assert_eq!(cnt, want, "naive matvecs: m - 1 rotations, m relins");
    assert!(cnt.key_switches() > base.key_switches());
    let err = w0
        .iter()
        .zip(&w2)
        .map(|(a, b)| max_diff(a, b))
        .fold(0.0, f64::max);
    eprintln!("naive vs default weights: 2^{:.1}", err.log2());
    assert!(err < 2f64.powi(-20), "{err:e}");
}

/// The non-interactive micro-batch gradient: `k m` relins with `lazy_relin` off, `m` with it on,
/// same values.
#[test]
fn reduced_gradient_lazy_relin_flag() {
    let c = ctx();
    let mut r = rng(5);
    let (w, b) = layers(&mut r).remove(0);
    let x_scale = c.params.scale();
    let dense = Dense::encrypt(&w, &b, &c.layout, x_scale, &c.enc, &c.encoder, &mut r).unwrap();
    let top = c.params.max_level();
    let enc_act = |r: &mut Rng, cols: usize| {
        let v = pack_activations(&rand_mat(r, 8, cols, 1.0), &c.layout);
        c.enc
            .encrypt(&c.encoder.encode(&v, c.params.scale(), top).unwrap(), r)
    };
    let k = 3;
    let deltas: Vec<Ciphertext> = (0..k).map(|_| enc_act(&mut r, 8)).collect();
    let inputs: Vec<Ciphertext> = (0..k).map(|_| enc_act(&mut r, 4)).collect();
    let d_refs: Vec<&Ciphertext> = deltas.iter().collect();
    let a_refs: Vec<&Ciphertext> = inputs.iter().collect();
    let m = LayerShape::new(8, 4, &c.layout).m as u64;
    let run = |ev: &Evaluator| {
        let before = ev.counter();
        let g = dense
            .weight_grad_reduced(ev, &d_refs, &a_refs, &c.layout, 0.125)
            .unwrap();
        (&ev.counter() - &before, g)
    };
    let (lazy, g0) = run(&c.eval(EvalConfig::default()));
    let eager_ev = c.eval(EvalConfig {
        lazy_relin: false,
        ..EvalConfig::default()
    });
    let (eager, g1) = run(&eager_ev);
    assert_eq!(lazy.relins, m);
    assert_eq!(eager.relins, k as u64 * m);
    assert_eq!(
        (eager.rotations, eager.ct_ct_tensor),
        (lazy.rotations, lazy.ct_ct_tensor)
    );
    for (a, b) in g0.iter().zip(&g1) {
        assert_eq!((a.level(), a.scale()), (b.level(), b.scale()));
        let e = max_diff(&c.decrypt(a), &c.decrypt(b));
        assert!(e < 2f64.powi(-20), "{e:e}");
    }
}

#[test]
fn reboot_outer_sum_lazy_needs_the_flag() {
    let c = ctx();
    let slots = c.params.slots();
    let lay = RebootLayout::new(128, slots / 128, slots).unwrap();
    let mut r = rng(6);
    let top = c.params.max_level();
    let enc = |r: &mut Rng| {
        let v: Vec<f64> = (0..slots).map(|_| uni(r)).collect();
        c.enc
            .encrypt(&c.encoder.encode(&v, c.params.scale(), top).unwrap(), r)
    };
    let xs: Vec<Ciphertext> = (0..4).map(|_| enc(&mut r)).collect();
    let ys: Vec<Ciphertext> = (0..4).map(|_| enc(&mut r)).collect();
    let pairs: Vec<(&Ciphertext, &Ciphertext)> = xs.iter().zip(&ys).collect();
    let relins = |ev: &Evaluator, mode| {
        let before = ev.counter();
        let s = lay.outer_sum(ev, &pairs, mode).unwrap();
        ((&ev.counter() - &before).relins, c.decrypt(&s))
    };
    let ev = c.eval(EvalConfig::default());
    let (lazy, v0) = relins(&ev, GradRelin::Lazy);
    let (per, _) = relins(&ev, GradRelin::PerSample);
    let eager_ev = c.eval(EvalConfig::NAIVE);
    let (lazy_off, v1) = relins(&eager_ev, GradRelin::Lazy);
    assert_eq!((lazy, per, lazy_off), (1, 4, 4));
    assert!(max_diff(&v0, &v1) < 2f64.powi(-20));
}

#[test]
fn backend_helpers_on_the_cpu() {
    let c = ctx();
    let mut r = rng(8);
    let acts: Vec<Box<dyn Activation>> = vec![Box::new(Square), Box::new(Square)];
    let mlp = Mlp::encrypt(
        &c.ev,
        &layers(&mut r),
        acts,
        &c.layout,
        &c.enc,
        &c.encoder,
        &mut r,
    )
    .unwrap();
    let x = rand_mat(&mut r, 8, 4, 1.0);
    let p0 = mlp.predict_plain(&c.dec, &c.encoder, &c.layout, &x);
    let p1 = mlp
        .predict_plain_on(&c.ev, &c.dec, &c.encoder, &c.layout, &x)
        .unwrap();
    assert_eq!(p0, p1);

    let d = &mlp.layers[0].0;
    let up: Dense = d.upload(&c.ev).unwrap();
    let down = up.download(&c.ev).unwrap();
    let owned = down.clone().upload_owned(&c.ev).unwrap();
    for (a, b) in
        d.w.diagonals()
            .entries()
            .iter()
            .chain([&d.b])
            .zip(owned.w.diagonals().entries().iter().chain([&owned.b]))
    {
        assert_eq!(c.decrypt(a), c.decrypt(b));
        assert_eq!((a.level(), a.scale()), (b.level(), b.scale()));
    }

    // HostRefresher around a ClientRefresher: the host refresh, with the CPU "upload".
    let ct = c.ev.mod_switch_to_level(&d.b, 2).unwrap();
    let mut host = HostRefresher::new(&c.ev, c.client(10));
    let fresh = Refresher::<Evaluator>::refresh(&mut host, &ct).unwrap();
    assert_eq!(fresh.level(), host.output_level());
    assert_eq!(fresh.scale(), ct.scale());
    assert!(max_diff(&c.decrypt(&fresh), &c.decrypt(&ct)) < 2f64.powi(-20));
    assert_eq!(host.inner().refreshed(), 1);
    // `HeOps` on the CPU: upload is a copy.
    let up = HeOps::upload_ct(&c.ev, &ct).unwrap();
    assert_eq!(c.decrypt(&up), c.decrypt(&ct));
}

/// n16_boot's chain at N = 2^10 (insecure, tests only), as in `tests/trainer_ni.rs`.
struct Mini {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
    bs: Bootstrapper,
    layout: Layout,
}

const NI_DIMS: [usize; 3] = [16, 8, 4];

fn mini() -> &'static Mini {
    static M: OnceLock<Mini> = OnceLock::new();
    M.get_or_init(|| {
        let (q, p) = hekit_ckks::params::n16_boot_bits();
        let params = Arc::new(CkksParams::new(10, &q, &p, 42, 8).unwrap());
        let layout = Layout::new(512, 16, 32).unwrap();
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(2)).unwrap();
        let encoder = Encoder::new(Arc::clone(&params));
        let bp = BootParams {
            log_slots: 9,
            ..BootParams::default_n16()
        };
        let bs = Bootstrapper::new(Arc::clone(&params), bp, &encoder, &mut kg).unwrap();
        let steps = ni_rotation_steps(&NI_DIMS, &layout, &TransposeKeys::Dedicated).unwrap();
        let gks = kg.galois_keys(&steps);
        Mini {
            enc: Encryptor::new(Arc::clone(&params), kg.public_key()),
            dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
            ev: Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), Some(gks)),
            encoder,
            bs,
            layout,
            params,
        }
    })
}

#[test]
fn non_interactive_update_honours_the_flags() {
    let f = mini();
    let nag = NiOptimizer::Nag {
        lr: 0.05,
        momentum: 0.9,
    };
    let placement = BootPlacement::AfterUpdate;
    let micro = 2;
    let run = |cfg: EvalConfig| {
        let ev = Evaluator::with_config(
            Arc::clone(&f.params),
            f.ev.relin_key().cloned(),
            f.ev.galois_keys().cloned(),
            cfg,
        );
        let mut r = rng(11);
        let plain: Vec<(Array2<f64>, Vec<f64>)> = NI_DIMS
            .windows(2)
            .map(|p| {
                (
                    rand_mat(&mut r, p[1], p[0], 0.2),
                    (0..p[1]).map(|_| 0.1 * uni(&mut r)).collect(),
                )
            })
            .collect();
        let acts: Vec<Box<dyn Activation>> = vec![Box::new(PolyRelu2), Box::new(PolyRelu2)];
        let (mut mlp, state) = NonInteractiveTrainer::encrypt_network(
            &ev, &f.bs, &plain, acts, &f.layout, placement, &f.enc, &f.encoder, &mut r,
        )
        .unwrap();
        let mut t =
            NonInteractiveTrainer::new(&ev, &f.bs, nag, f.layout, placement, &mlp, state).unwrap();
        let spec = t.batch_spec(&mlp).unwrap();
        let cts: Vec<(Ciphertext, Ciphertext)> = (0..micro)
            .map(|_| {
                let x = rand_mat(&mut r, f.layout.batch, NI_DIMS[0], 1.0);
                let y = rand_mat(&mut r, f.layout.batch, NI_DIMS[2], 0.5);
                spec.encrypt(&f.enc, &f.encoder, &f.layout, &x, &y, &mut r)
                    .unwrap()
            })
            .collect();
        let before = ev.counter();
        t.train_step(&mut mlp, &cts).unwrap();
        let got = &ev.counter() - &before;
        let pred = predict_update(
            &ev,
            &mlp,
            &f.layout,
            &TransposeKeys::Dedicated,
            placement,
            micro,
            true,
            t.last_update_booted(),
        )
        .unwrap();
        let per = f.bs.cost_per_boot();
        let nb = pred.bootstraps();
        assert!(nb > 0);
        assert_eq!(got.bootstraps, nb, "{cfg:?}");
        assert_eq!(
            got.rotations,
            pred.rotations() + nb * per.rotations,
            "{cfg:?}"
        );
        assert_eq!(got.relins, pred.relins() + nb * per.relins, "{cfg:?}");
        let state: Vec<Vec<f64>> = t
            .state()
            .weights
            .iter()
            .map(|c| f.encoder.decode(&f.dec.decrypt(c)))
            .collect();
        (got, state)
    };
    let (base, s0) = run(EvalConfig::default());
    for cfg in EvalConfig::all() {
        let (got, s) = run(cfg);
        let err = s0
            .iter()
            .zip(&s)
            .map(|(a, b)| max_diff(a, b))
            .fold(0.0, f64::max);
        eprintln!(
            "{cfg:?}: rotations {} relins {} (default {} / {}), state vs default 2^{:.1}",
            got.rotations,
            got.relins,
            base.rotations,
            base.relins,
            err.log2()
        );
        assert!(err < 2f64.powi(-18), "{cfg:?}: {err:e}");
        if cfg.hoist != EvalConfig::default().hoist && cfg.bsgs && cfg.lazy_relin {
            assert_eq!(got, base, "hoist changes no count");
        }
    }
}
