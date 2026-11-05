//! Task 41: non-interactive training with bootstrapping (`hekit_nn::trainer_ni`,
//! `docs/layout.md` §7).
//!
//! Small tests run a **real** bootstrapper on `mini`: the n16_boot modulus chain (32 primes,
//! 16 compute levels, the same bands and `BootParams`) at N = 2^10 (512 slots, insecure, tests
//! only). They check:
//! - the weights after 3 updates against a plaintext NAG twin (bootstrap error included), for
//!   a single-block and a replicated layout (ruling R-M3-3: the compact path re-lays the block
//!   copies), with every placement and with 2 micro-batches per update;
//! - the measured counter of every update against [`predict_update`] (rotations, relins,
//!   bootstraps, SSE switches, and the key switches with the bootstraps), and `bootstraps` =
//!   `boot_count` x updates;
//! - the levels of every value against the structural prediction, and [`levels_per_iter`]
//!   (exact values, the depth-overflow configuration refused by `new`);
//! - that training runs with no secret key in scope: the `Decryptor` is built only after
//!   training (the trainer holds none, by type).
//!
//! The n16_boot gate of the brief is `#[ignore]` (bootstrap keys ~10 GB).

use std::sync::{Arc, OnceLock};

use hekit_ckks::bootstrap::{BootParams, Bootstrapper};
use hekit_ckks::keys::SecretKey;
use hekit_ckks::{CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter};
use hekit_core::error::HekitError;
use hekit_core::sampler::Rng;
use hekit_nn::activation::{Activation, Identity, Poly, PolyRelu2};
use hekit_nn::compact::boot_count;
use hekit_nn::dense::{Dense, TransposeKeys};
use hekit_nn::layout::{Layout, pack_activations, unpack_activations};
use hekit_nn::trainer::{Mlp, mlp_key_pool};
use hekit_nn::trainer_ni::{
    ActCost, BootPlacement, CompactLayout, NiOptimizer, NonInteractiveTrainer, boot_schedule,
    levels_per_iter, ni_rotation_steps, predict_update,
};
use ndarray::{Array2, Axis};

const SLOTS: usize = 512;
const LR: f64 = 0.05;
const MU: f64 = 0.9;

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

/// n16_boot's chain at N = 2^10 (insecure, tests only).
fn mini() -> CkksParams {
    let (q, p) = hekit_ckks::params::n16_boot_bits();
    CkksParams::new(10, &q, &p, 42, 8).unwrap()
}

struct Fx {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    ev: Evaluator,
    bs: Bootstrapper,
    sk: SecretKey,
}

/// The layouts and networks of these tests (their keys are generated once).
fn single() -> Layout {
    Layout::new(SLOTS, 16, 32).unwrap()
}

fn replicated() -> Layout {
    Layout::new(SLOTS, 16, 8).unwrap()
}

const NET_A: [usize; 3] = [16, 16, 4];
const NET_B: [usize; 3] = [16, 8, 4];
/// One wide layer whose compact ciphertext is full (m L = 16 x 32 = 512 slots).
const NET_C: [usize; 2] = [32, 16];

fn wide() -> Layout {
    Layout::new(SLOTS, 32, 16).unwrap()
}

fn fx() -> &'static Fx {
    static FX: OnceLock<Fx> = OnceLock::new();
    FX.get_or_init(|| {
        let params = Arc::new(mini());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(1)).unwrap();
        let encoder = Encoder::new(Arc::clone(&params));
        let bp = BootParams {
            log_slots: 9,
            ..BootParams::default_n16()
        };
        let bs = Bootstrapper::new(Arc::clone(&params), bp, &encoder, &mut kg).unwrap();
        let mut steps = Vec::new();
        for layout in [single(), replicated()] {
            for dims in [&NET_A[..], &NET_B[..]] {
                steps.extend(ni_rotation_steps(dims, &layout, &TransposeKeys::Dedicated).unwrap());
            }
        }
        steps.extend(ni_rotation_steps(&NET_C, &wide(), &TransposeKeys::Dedicated).unwrap());
        steps.sort_unstable();
        steps.dedup();
        let gks = kg.galois_keys(&steps);
        let ev = Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), Some(gks));
        Fx {
            enc: Encryptor::new(Arc::clone(&params), kg.public_key()),
            sk: kg.secret_key(),
            encoder,
            ev,
            bs,
            params,
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Plaintext twin.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Act {
    Relu2,
    Id,
}

impl Act {
    fn boxed(self) -> Box<dyn Activation> {
        match self {
            Act::Relu2 => Box::new(PolyRelu2),
            Act::Id => Box::new(Identity),
        }
    }
    fn f(self, z: f64) -> f64 {
        match self {
            Act::Relu2 => z * z + z,
            Act::Id => z,
        }
    }
    fn df(self, z: f64) -> f64 {
        match self {
            Act::Relu2 => 2.0 * z + 1.0,
            Act::Id => 1.0,
        }
    }
}

type Net = Vec<(Array2<f64>, Vec<f64>)>;

fn init(dims: &[usize], r: &mut Rng) -> Net {
    init_scaled(dims, r, 0.2)
}

/// Weights uniform in `±scale`, biases in `±0.1`.
fn init_scaled(dims: &[usize], r: &mut Rng, scale: f64) -> Net {
    dims.windows(2)
        .map(|p| {
            let w = Array2::from_shape_fn((p[1], p[0]), |_| scale * uni(r));
            let b = (0..p[1]).map(|_| 0.1 * uni(r)).collect();
            (w, b)
        })
        .collect()
}

fn affine(a: &Array2<f64>, w: &Array2<f64>, b: &[f64]) -> Array2<f64> {
    let mut z = a.dot(&w.t());
    for mut row in z.rows_mut() {
        row.iter_mut().zip(b).for_each(|(v, bb)| *v += bb);
    }
    z
}

/// One NAG step (`docs/layout.md` §4 form), biases plain SGD with `lr`, means over the rows.
fn twin_step(
    net: &mut Net,
    vel: &mut [Array2<f64>],
    acts: &[Act],
    x: &Array2<f64>,
    y: &Array2<f64>,
    nag: bool,
) {
    let n = x.nrows() as f64;
    let mut a = vec![x.clone()];
    let mut zs = Vec::new();
    for (l, (w, b)) in net.iter().enumerate() {
        let z = affine(a.last().unwrap(), w, b);
        a.push(z.mapv(|v| acts[l].f(v)));
        zs.push(z);
    }
    let mut delta = a.last().unwrap() - y;
    for i in (0..net.len()).rev() {
        let dz = &delta * &zs[i].mapv(|v| acts[i].df(v));
        let next = dz.dot(&net[i].0);
        let g = dz.t().dot(&a[i]) / n;
        if nag {
            vel[i] = MU * &vel[i] + &g;
            net[i].0 = &net[i].0 - &(LR * &(&g + &(MU * &vel[i])));
        } else {
            net[i].0 = &net[i].0 - &(LR * &g);
        }
        let gb = dz.sum_axis(Axis(0)) / n;
        net[i].1.iter_mut().zip(&gb).for_each(|(b, g)| *b -= LR * g);
        delta = next;
    }
}

fn max_err(a: &Net, b: &Net) -> f64 {
    let mut e: f64 = 0.0;
    for ((wa, ba), (wb, bb)) in a.iter().zip(b) {
        e = wa.iter().zip(wb).fold(e, |m, (x, y)| m.max((x - y).abs()));
        e = ba.iter().zip(bb).fold(e, |m, (x, y)| m.max((x - y).abs()));
    }
    e
}

fn decrypt_net(mlp: &Mlp, dec: &Decryptor, encoder: &Encoder, layout: &Layout) -> Net {
    mlp.layers
        .iter()
        .map(|(d, _)| {
            let w = d.w.decrypt(dec, encoder, layout);
            let b = unpack_activations(&encoder.decode(&dec.decrypt(&d.b)), layout, w.nrows());
            (w, b.row(0).to_vec())
        })
        .collect()
}

/// `max_s |v[s] - v[s mod block]|` over every weight diagonal and bias.
fn block_spread(mlp: &Mlp, dec: &Decryptor, encoder: &Encoder, layout: &Layout) -> f64 {
    let bk = layout.block();
    let mut e: f64 = 0.0;
    for (d, _) in &mlp.layers {
        for ct in d.w.diagonals().entries().iter().chain([&d.b]) {
            let v = encoder.decode(&dec.decrypt(ct));
            e = v
                .iter()
                .enumerate()
                .fold(e, |m, (s, x)| m.max((x - v[s % bk]).abs()));
        }
    }
    e
}

// ---------------------------------------------------------------------------------------------
// Runs.
// ---------------------------------------------------------------------------------------------

struct Run {
    err: f64,
    spread: f64,
    boots: Vec<bool>,
    counters: Vec<OpCounter>,
    /// AfterActivation: max |bootstrapped hidden activation - twin's| over all updates.
    act_err: f64,
    /// Max |hidden activation| the twin sees (the activation bootstraps' input).
    act_max: f64,
}

/// Trains `dims` with `acts` for `updates` updates of `micro` micro-batches on `layout`,
/// checking every update's counter against [`predict_update`] and its levels against the
/// structural trace; returns the twin error and the schedule.
#[allow(clippy::too_many_arguments)]
fn run(
    dims: &[usize],
    acts: &[Act],
    layout: Layout,
    placement: BootPlacement,
    optim: NiOptimizer,
    micro: usize,
    updates: usize,
    seed: u8,
) -> Run {
    run_with(
        dims,
        acts,
        layout,
        placement,
        optim,
        micro,
        updates,
        seed,
        TransposeKeys::Dedicated,
        0.2,
    )
}

/// [`run`] with the transpose keys and the weight range.
#[allow(clippy::too_many_arguments)]
fn run_with(
    dims: &[usize],
    acts: &[Act],
    layout: Layout,
    placement: BootPlacement,
    optim: NiOptimizer,
    micro: usize,
    updates: usize,
    seed: u8,
    keys: TransposeKeys,
    wscale: f64,
) -> Run {
    let f = fx();
    // Our own counter: tests run in parallel on the shared keys (Evaluator::clone shares the
    // keys and starts a separate counter).
    let ev = f.ev.clone();
    let mut r = rng(seed);
    let plain = init_scaled(dims, &mut r, wscale);
    let boxed: Vec<Box<dyn Activation>> = acts.iter().map(|a| a.boxed()).collect();
    let (mut mlp, state) = NonInteractiveTrainer::encrypt_network(
        &ev, &f.bs, &plain, boxed, &layout, placement, &f.enc, &f.encoder, &mut r,
    )
    .unwrap();
    let mut t = NonInteractiveTrainer::new(&ev, &f.bs, optim, layout, placement, &mlp, state)
        .unwrap()
        .with_transpose_keys(keys.clone());
    // The client keeps the secret key; it decrypts only for the activation diagnostic below
    // (after an update, outside the trainer) and at the end.
    let diag = Decryptor::new(Arc::clone(&f.params), f.sk.clone());
    let spec = t.batch_spec(&mlp).unwrap();
    assert_eq!(spec.x_level, f.bs.output_level() - 1);
    let nag = optim.has_velocity();
    let mut twin = plain.clone();
    let mut vel: Vec<Array2<f64>> = plain.iter().map(|(w, _)| Array2::zeros(w.dim())).collect();
    let (d0, dl) = (dims[0], *dims.last().unwrap());
    let cl = CompactLayout::new(dims, &layout).unwrap();
    let mut out = Run {
        err: 0.0,
        spread: 0.0,
        boots: Vec::new(),
        counters: Vec::new(),
        act_err: 0.0,
        act_max: 0.0,
    };
    for it in 0..updates {
        let mut cts = Vec::new();
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..micro {
            let x = Array2::from_shape_fn((layout.batch, d0), |_| uni(&mut r));
            let y = Array2::from_shape_fn((layout.batch, dl), |_| 0.5 * uni(&mut r));
            cts.push(
                spec.encrypt(&f.enc, &f.encoder, &layout, &x, &y, &mut r)
                    .unwrap(),
            );
            xs.push(x);
            ys.push(y);
        }
        let x_all = ndarray::concatenate(Axis(0), &xs.iter().map(|x| x.view()).collect::<Vec<_>>())
            .unwrap();
        let y_all = ndarray::concatenate(Axis(0), &ys.iter().map(|x| x.view()).collect::<Vec<_>>())
            .unwrap();
        let hidden = affine(&x_all, &twin[0].0, &twin[0].1).mapv(|v| acts[0].f(v));
        out.act_max = out
            .act_max
            .max(hidden.iter().fold(0.0, |m, v| m.max(v.abs())));
        twin_step(&mut twin, &mut vel, acts, &x_all, &y_all, nag);

        let predicted_levels = t.predicted_trace(&mlp);
        let before = ev.counter();
        t.set_next_batch_full((it + 1 < updates).then_some(true));
        t.train_step(&mut mlp, &cts).unwrap();
        let got = &ev.counter() - &before;
        // Levels: every value of the step, as predicted.
        let measured: Vec<(String, i64)> = t
            .trace()
            .iter()
            .map(|(k, v)| (k.clone(), *v as i64))
            .collect();
        assert_eq!(measured, predicted_levels, "levels of the step");
        // Counts: the closed form, plus the bootstraps' own counts.
        let booted = t.last_update_booted();
        for (mb, outs) in t.last_activation_bootstraps().iter().enumerate() {
            let a =
                unpack_activations(&f.encoder.decode(&diag.decrypt(&outs[0])), &layout, dims[1]);
            for i in 0..layout.batch {
                for j in 0..dims[1] {
                    let e = (a[[i, j]] - hidden[[mb * layout.batch + i, j]]).abs();
                    out.act_err = out.act_err.max(e);
                }
            }
        }
        let pred =
            predict_update(&ev, &mlp, &layout, &keys, placement, micro, nag, booted).unwrap();
        let per = f.bs.cost_per_boot();
        let nb = pred.bootstraps();
        assert_eq!(got.bootstraps, nb, "bootstraps");
        assert_eq!(
            got.rotations,
            pred.rotations() + nb * per.rotations,
            "rotations"
        );
        assert_eq!(got.relins, pred.relins() + nb * per.relins, "relins");
        assert_eq!(got.sse_switches, nb * per.sse_switches, "sse switches");
        assert_eq!(
            got.key_switches(),
            pred.key_switches(f.bs.key_switches_per_boot()),
            "key switches incl. bootstraps"
        );
        if booted {
            let per_layer: usize = cl
                .plans
                .iter()
                .map(|p| boot_count(p, p.d_out))
                .sum::<usize>();
            let bias_only = cl.containers.iter().filter(|c| c.layer.is_none()).count();
            assert_eq!(cl.containers.len(), per_layer + bias_only);
            assert_eq!(
                pred.weight_bootstraps,
                (per_layer * (1 + usize::from(nag)) + bias_only) as u64,
                "boot_count x (W, U) + bias-only containers"
            );
        }
        out.boots.push(booted);
        out.counters.push(got);
    }
    let act_refs: Vec<&dyn Activation> = mlp.layers.iter().map(|(_, a)| a.as_ref()).collect();
    let schedule = boot_schedule(
        dims,
        &act_refs,
        &layout,
        placement,
        nag,
        f.bs.output_level() - 1,
        f.bs.output_level(),
        &vec![true; updates],
    )
    .unwrap();
    assert_eq!(out.boots, schedule, "the structural schedule");
    let dec = diag;
    let he = decrypt_net(&mlp, &dec, &f.encoder, &layout);
    out.err = max_err(&he, &twin);
    out.spread = block_spread(&mlp, &dec, &f.encoder, &layout);
    let moved = max_err(&plain, &twin);
    eprintln!(
        "{dims:?} B {} x {micro} {:?}: twin error 2^{:.1} (weights moved 2^{:.1}), block spread \
         2^{:.1}, boots {:?}, max |a1| {:.3}, activation-bootstrap error 2^{:.1}",
        layout.batch,
        placement,
        out.err.log2(),
        moved.log2(),
        out.spread.log2(),
        out.boots,
        out.act_max,
        out.act_err.log2()
    );
    assert!(
        moved > 4.0 * out.err,
        "the update is visible above the error"
    );
    out
}

/// The brief's gate (n16_boot, 17.5-bit bootstrap).
const TOL: f64 = 1.0 / 256.0;
/// The small tests' bound: `mini` bootstraps at about 26 bits (`docs/bootstrap.md` §11), so a
/// cross-position leak or a wrong-scale bias (~2^-9 and above) cannot hide under it.
const TOL_MINI: f64 = 1.0 / (1u64 << 20) as f64;

const NAG: NiOptimizer = NiOptimizer::Nag {
    lr: LR,
    momentum: MU,
};

#[test]
fn after_update_matches_plaintext_nag_single_block() {
    let r = run(
        &NET_A,
        &[Act::Relu2, Act::Relu2],
        single(),
        BootPlacement::AfterUpdate,
        NAG,
        1,
        3,
        7,
    );
    assert!(r.err <= TOL_MINI, "2^{}", r.err.log2());
    assert_eq!(
        r.boots,
        vec![true; 3],
        "AfterUpdate bootstraps every update"
    );
    let boots: u64 = r.counters.iter().map(|c| c.bootstraps).sum();
    // boot_count x updates: one container per layer, W and U each.
    assert_eq!(boots, 2 * 2 * 3);
}

#[test]
fn after_update_relays_replicated_blocks() {
    // 4 block copies (d B = 128 of 512 slots): BootstrapRefresher refuses this layout, the
    // compact path re-lays the copies by construction (ruling R-M3-3).
    let r = run(
        &NET_B,
        &[Act::Relu2, Act::Id],
        replicated(),
        BootPlacement::AfterUpdate,
        NAG,
        1,
        3,
        8,
    );
    assert!(r.err <= TOL_MINI, "2^{}", r.err.log2());
    assert!(
        r.spread <= 1e-5,
        "block copies re-laid: 2^{}",
        r.spread.log2()
    );
}

#[test]
fn lazy_placements_skip_weight_bootstraps() {
    // eMLP-1-shaped network (z^2 + z, then linear): 7 levels per step, 5 with activation
    // bootstraps; 15 levels after a bootstrap and an unpack.
    let acts = [Act::Relu2, Act::Id];
    let a = run(
        &NET_B,
        &acts,
        single(),
        BootPlacement::AfterActivation,
        NAG,
        1,
        4,
        9,
    );
    assert!(a.err <= TOL_MINI, "2^{}", a.err.log2());
    assert_eq!(
        a.boots,
        vec![false, true, false, true],
        "every second update"
    );
    let w = run(
        &NET_B,
        &acts,
        single(),
        BootPlacement::WhenNeeded,
        NAG,
        1,
        4,
        9,
    );
    assert!(w.err <= TOL_MINI, "2^{}", w.err.log2());
    assert_eq!(w.boots, vec![false, true, false, true]);
    // AfterActivation adds one activation bootstrap per update.
    for (ca, cw) in a.counters.iter().zip(&w.counters) {
        assert_eq!(ca.bootstraps, cw.bootstraps + 1);
    }
}

#[test]
fn micro_batches_accumulate_before_one_update() {
    let r = run(
        &NET_B,
        &[Act::Relu2, Act::Id],
        single(),
        BootPlacement::AfterUpdate,
        NAG,
        2,
        2,
        10,
    );
    assert!(r.err <= TOL_MINI, "2^{}", r.err.log2());
    // 4 bootstraps per update for 2 x 32 samples.
    assert!(r.counters.iter().all(|c| c.bootstraps == 4));
}

#[test]
fn sgd_with_bias_only_container() {
    // A one-layer 32 -> 16 network in d = 32, B = 16: its container is full (16 of 16
    // positions), so the bias needs a bias-only container (one more bootstrap).
    let layout = wide();
    let cl = CompactLayout::new(&NET_C, &layout).unwrap();
    assert_eq!(cl.containers.len(), 2);
    assert_eq!(cl.containers[1].layer, None);
    let r = run(
        &NET_C,
        &[Act::Relu2],
        layout,
        BootPlacement::AfterUpdate,
        NiOptimizer::Sgd { lr: LR },
        1,
        2,
        11,
    );
    assert!(r.err <= TOL_MINI, "2^{}", r.err.log2());
    assert!(r.counters.iter().all(|c| c.bootstraps == 2));
}

/// AfterActivation with the zero-new-key (Reuse) transpose, the configuration the predictions
/// are for, and hidden activations above 1 (outside the bootstrap's `|m| <= 1` range: not
/// clipped, the error is measured against the twin).
#[test]
fn after_activation_reuse_and_large_activations() {
    let layout = single();
    let reuse = TransposeKeys::Reuse(mlp_key_pool(&NET_B, &layout).unwrap());
    let r = run_with(
        &NET_B,
        &[Act::Relu2, Act::Id],
        layout,
        BootPlacement::AfterActivation,
        NAG,
        1,
        2,
        13,
        reuse,
        0.6,
    );
    assert!(r.act_max > 1.0, "activations above 1: {}", r.act_max);
    // The bootstrap still runs; its absolute error grows with |m| but stays far below the
    // update.
    assert!(r.act_err < 1e-4, "2^{}", r.act_err.log2());
    assert!(r.err <= 1e-4, "2^{}", r.err.log2());
}

/// Review I1: a lazy placement skips a bootstrap only if the next step fits, and a partial
/// batch costs one more level at a linear output layer. Undeclared next steps are planned as
/// partial; a declared-full next step that turns out partial is refused before any work; and
/// `train_epoch_valid` declares the fullness itself.
#[test]
fn lazy_placement_partial_batch_after_skip() {
    let f = fx();
    // Our own counter: tests run in parallel on the shared keys (Evaluator::clone shares the
    // keys and starts a separate counter).
    let ev = f.ev.clone();
    let layout = single();
    let acts = || -> Vec<Box<dyn Activation>> { vec![Box::new(PolyRelu2), Box::new(Identity)] };
    let mut r = rng(14);
    let plain = init(&NET_B, &mut r);
    let fresh = |r: &mut Rng| {
        let (mlp, state) = NonInteractiveTrainer::encrypt_network(
            &ev,
            &f.bs,
            &plain,
            acts(),
            &layout,
            BootPlacement::WhenNeeded,
            &f.enc,
            &f.encoder,
            r,
        )
        .unwrap();
        let t = NonInteractiveTrainer::new(
            &ev,
            &f.bs,
            NAG,
            layout,
            BootPlacement::WhenNeeded,
            &mlp,
            state,
        )
        .unwrap();
        (mlp, t)
    };
    let batch = |t: &NonInteractiveTrainer<'_>, mlp: &Mlp, r: &mut Rng| {
        let x = Array2::from_shape_fn((32, 16), |_| uni(r));
        let y = Array2::from_shape_fn((32, 4), |_| 0.5 * uni(r));
        t.batch_spec(mlp)
            .unwrap()
            .encrypt(&f.enc, &f.encoder, &layout, &x, &y, r)
            .unwrap()
    };

    // (a) Undeclared: full, then 31 of 32 lanes. The first update bootstraps (conservative).
    let (mut mlp, mut t) = fresh(&mut r);
    let (x, y) = batch(&t, &mlp, &mut r);
    t.train_batch(&mut mlp, &x, &y).unwrap();
    assert!(
        t.last_update_booted(),
        "unknown next step: planned as partial"
    );
    let (x, y) = batch(&t, &mlp, &mut r);
    t.train_batch_valid(&mut mlp, &x, &y, 31).unwrap();

    // (b) Declared full, then partial: the skip happens and the partial step is refused before
    // any work.
    let (mut mlp, mut t) = fresh(&mut r);
    let (x, y) = batch(&t, &mlp, &mut r);
    t.set_next_batch_full(Some(true));
    t.train_batch(&mut mlp, &x, &y).unwrap();
    assert!(!t.last_update_booted(), "declared full next step: skipped");
    let (x, y) = batch(&t, &mlp, &mut r);
    let before = ev.counter();
    let e = t.train_batch_valid(&mut mlp, &x, &y, 31).unwrap_err();
    assert!(matches!(e, HekitError::InvalidParams(_)), "{e:?}");
    assert_eq!(ev.counter(), before, "refused before any work");
    // A full step still fits.
    t.train_batch(&mut mlp, &x, &y).unwrap();

    // (c) Declared partial: bootstraps, then the partial step trains.
    let (mut mlp, mut t) = fresh(&mut r);
    let (x, y) = batch(&t, &mlp, &mut r);
    t.set_next_batch_full(Some(false));
    t.train_batch(&mut mlp, &x, &y).unwrap();
    assert!(t.last_update_booted());
    let (x, y) = batch(&t, &mlp, &mut r);
    t.train_batch_valid(&mut mlp, &x, &y, 31).unwrap();

    // (d) Four updates with explicit declarations (the epoch's look-ahead): every update's
    // decision equals boot_schedule's.
    let lanes_of = [32, 32, 32, 31];
    let (mut mlp, mut t) = fresh(&mut r);
    let mut decided = Vec::new();
    for (i, &n) in lanes_of.iter().enumerate() {
        let (x, y) = batch(&t, &mlp, &mut r);
        t.set_next_batch_full(lanes_of.get(i + 1).map(|&k| k == 32));
        t.train_step_valid(&mut mlp, &[(&x, &y, n)]).unwrap();
        decided.push(t.last_update_booted());
    }

    // (e) The same epoch through train_epoch_valid (which declares the fullness itself).
    let (mut mlp, mut t) = fresh(&mut r);
    let mut batches = Vec::new();
    for n in lanes_of {
        let (x, y) = batch(&t, &mlp, &mut r);
        batches.push((x, y, n));
    }
    let log = t.train_epoch_valid(&mut mlp, &batches).unwrap();
    let refs: Vec<&dyn Activation> = mlp.layers.iter().map(|(_, a)| a.as_ref()).collect();
    let sched = boot_schedule(
        &NET_B,
        &refs,
        &layout,
        BootPlacement::WhenNeeded,
        true,
        f.bs.output_level() - 1,
        f.bs.output_level(),
        &[true, true, true, false],
    )
    .unwrap();
    // full -> skip, full -> boot, full (next partial) -> boot, partial (last) -> boot.
    assert_eq!(sched, vec![false, true, true, true]);
    assert_eq!(decided, sched, "every update's decision");
    let boots = sched.iter().filter(|&&b| b).count() as u64 * 4;
    assert_eq!(log.counter.bootstraps, boots);
    assert_eq!(log.refreshes as u64, boots);
    assert!(log.refresh_seconds > 0.0 && t.bootstrap_seconds() >= log.refresh_seconds);
}

/// Ruling R-M3-12: a bias-free network (ReBoot's `bias=False`) as zero biases that are never
/// updated (`with_frozen_biases`): the counts are the closed form's, the biases stay zero and the
/// weights follow a bias-free twin.
#[test]
fn frozen_biases_stay_zero() {
    let f = fx();
    let ev = f.ev.clone();
    let layout = single();
    let mut r = rng(15);
    let mut plain = init(&NET_B, &mut r);
    plain
        .iter_mut()
        .for_each(|(_, b)| b.iter_mut().for_each(|v| *v = 0.0));
    let acts = [Act::Relu2, Act::Id];
    let (mut mlp, state) = NonInteractiveTrainer::encrypt_network(
        &ev,
        &f.bs,
        &plain,
        acts.iter().map(|a| a.boxed()).collect(),
        &layout,
        BootPlacement::AfterUpdate,
        &f.enc,
        &f.encoder,
        &mut r,
    )
    .unwrap();
    let mut t = NonInteractiveTrainer::new(
        &ev,
        &f.bs,
        NAG,
        layout,
        BootPlacement::AfterUpdate,
        &mlp,
        state,
    )
    .unwrap()
    .with_frozen_biases();
    let spec = t.batch_spec(&mlp).unwrap();
    let mut twin = plain.clone();
    let mut vel: Vec<Array2<f64>> = plain.iter().map(|(w, _)| Array2::zeros(w.dim())).collect();
    for _ in 0..2 {
        let x = Array2::from_shape_fn((32, 16), |_| uni(&mut r));
        let y = Array2::from_shape_fn((32, 4), |_| 0.5 * uni(&mut r));
        let (xc, yc) = spec
            .encrypt(&f.enc, &f.encoder, &layout, &x, &y, &mut r)
            .unwrap();
        // The twin with biases frozen: its NAG step updates the biases by lr * mean(dz); undo it.
        let b_before: Vec<Vec<f64>> = twin.iter().map(|(_, b)| b.clone()).collect();
        twin_step(&mut twin, &mut vel, &acts, &x, &y, true);
        twin.iter_mut()
            .zip(b_before)
            .for_each(|((_, b), old)| *b = old);
        let before = ev.counter();
        t.train_batch(&mut mlp, &xc, &yc).unwrap();
        let got = &ev.counter() - &before;
        let pred = predict_update(
            &ev,
            &mlp,
            &layout,
            &TransposeKeys::Dedicated,
            BootPlacement::AfterUpdate,
            1,
            true,
            true,
        )
        .unwrap();
        assert_eq!(
            got.key_switches(),
            pred.key_switches(f.bs.key_switches_per_boot())
        );
    }
    let dec = Decryptor::new(Arc::clone(&f.params), f.sk.clone());
    let he = decrypt_net(&mlp, &dec, &f.encoder, &layout);
    let bias_max = he
        .iter()
        .flat_map(|(_, b)| b.iter())
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(bias_max < 1e-6, "biases stay zero: {bias_max}");
    let err = max_err(&he, &twin);
    assert!(err <= TOL_MINI, "2^{}", err.log2());
}

// ---------------------------------------------------------------------------------------------
// Levels and errors.
// ---------------------------------------------------------------------------------------------

fn mlp_of(dims: &[usize], acts: Vec<Box<dyn Activation>>, layout: &Layout) -> Mlp {
    let f = fx();
    // Our own counter: tests run in parallel on the shared keys (Evaluator::clone shares the
    // keys and starts a separate counter).
    let ev = f.ev.clone();
    let mut r = rng(3);
    let plain = init(dims, &mut r);
    let start = NonInteractiveTrainer::start_level_for(&f.bs);
    Mlp::encrypt_at(&ev, &plain, acts, layout, start, &f.enc, &f.encoder, &mut r).unwrap()
}

fn poly4() -> Box<dyn Activation> {
    Box::new(Poly::new(vec![0.0, 0.5, 0.1, 0.01, 0.001]).unwrap())
}

#[test]
fn levels_per_iter_exact_and_depth_overflow_refused() {
    let f = fx();
    // Our own counter: tests run in parallel on the shared keys (Evaluator::clone shares the
    // keys and starts a separate counter).
    let ev = f.ev.clone();
    let layout = single();
    use BootPlacement::*;
    // eMLP-1's structure: z^2 + z, then linear.
    let b = mlp_of(
        &NET_B,
        vec![Box::new(PolyRelu2), Box::new(Identity)],
        &layout,
    );
    assert_eq!(levels_per_iter(&b, AfterUpdate), 7);
    assert_eq!(levels_per_iter(&b, WhenNeeded), 7);
    assert_eq!(levels_per_iter(&b, AfterActivation), 5);
    assert_eq!(ActCost::probe(&ev, &PolyRelu2).unwrap(), ActCost::QUADRATIC);
    assert_eq!(ActCost::probe(&ev, &Identity).unwrap(), ActCost::IDENTITY);
    let a = mlp_of(
        &NET_A,
        vec![Box::new(PolyRelu2), Box::new(PolyRelu2)],
        &layout,
    );
    assert_eq!(levels_per_iter(&a, AfterUpdate), 9, "4L + 1 with L = 2");
    assert_eq!(levels_per_iter(&a, AfterActivation), 7);

    // The depth-overflow configuration: degree-4 polynomials in 4 layers.
    let dims = [16, 16, 16, 16, 16];
    let after_boot = NonInteractiveTrainer::start_level_for(&f.bs);
    assert_eq!(after_boot, 15);
    let plain = init(&dims, &mut rng(4));
    // Built layer by layer (every bias at Delta^2: only the structure matters here).
    let delta = f.params.scale();
    let mut r5 = rng(5);
    let layers = plain
        .iter()
        .map(|(w, b)| {
            let d = Dense::encrypt_at(
                w, b, &layout, delta, after_boot, &f.enc, &f.encoder, &mut r5,
            )
            .unwrap();
            (d, poly4())
        })
        .collect();
    let deep = Mlp { layers };
    assert_eq!(levels_per_iter(&deep, AfterUpdate), 25);
    assert_eq!(levels_per_iter(&deep, AfterActivation), 13);
    let cl = CompactLayout::new(&dims, &layout).unwrap();
    let state = hekit_nn::trainer_ni::CompactState::encrypt(
        &cl,
        &plain,
        &layout,
        f.bs.output_level(),
        &f.enc,
        &f.encoder,
        &mut rng(6),
    )
    .unwrap();
    let before = ev.counter();
    let err =
        NonInteractiveTrainer::new(&ev, &f.bs, NAG, layout, AfterUpdate, &deep, state.clone())
            .unwrap_err();
    assert!(matches!(err, HekitError::InvalidParams(_)), "{err:?}");
    assert_eq!(ev.counter(), before, "refused before any work");
    // With activation bootstraps the same network fits (13 <= 15): the level check passes
    // (and the first step's levels are checked next).
    let ok = NonInteractiveTrainer::new(&ev, &f.bs, NAG, layout, AfterActivation, &deep, state);
    assert!(ok.is_ok(), "{:?}", ok.err());
}

#[test]
fn wrong_state_and_errors_before_work() {
    let f = fx();
    // Our own counter: tests run in parallel on the shared keys (Evaluator::clone shares the
    // keys and starts a separate counter).
    let ev = f.ev.clone();
    let layout = single();
    let mut r = rng(12);
    let plain = init(&NET_B, &mut r);
    let acts: Vec<Box<dyn Activation>> = vec![Box::new(PolyRelu2), Box::new(Identity)];
    let (mut mlp, mut state) = NonInteractiveTrainer::encrypt_network(
        &ev,
        &f.bs,
        &plain,
        acts,
        &layout,
        BootPlacement::AfterUpdate,
        &f.enc,
        &f.encoder,
        &mut r,
    )
    .unwrap();
    let extra = state.weights[0].clone();
    state.weights.push(extra);
    state.velocity.push(None);
    assert!(matches!(
        NonInteractiveTrainer::new(
            &ev,
            &f.bs,
            NAG,
            layout,
            BootPlacement::AfterUpdate,
            &mlp,
            state.clone()
        ),
        Err(HekitError::InvalidParams(_))
    ));
    state.weights.pop();
    state.velocity.pop();
    let mut t = NonInteractiveTrainer::new(
        &ev,
        &f.bs,
        NAG,
        layout,
        BootPlacement::AfterUpdate,
        &mlp,
        state,
    )
    .unwrap();
    let spec = t.batch_spec(&mlp).unwrap();
    let x = Array2::from_shape_fn((32, 16), |_| uni(&mut r));
    let y = Array2::from_shape_fn((32, 4), |_| uni(&mut r));
    let (xc, yc) = spec
        .encrypt(&f.enc, &f.encoder, &layout, &x, &y, &mut r)
        .unwrap();
    let before = ev.counter();
    assert!(matches!(
        t.train_batch_valid(&mut mlp, &xc, &yc, 0),
        Err(HekitError::InvalidParams(_))
    ));
    assert!(matches!(
        t.train_step(&mut mlp, &[]),
        Err(HekitError::InvalidParams(_))
    ));
    assert_eq!(ev.counter(), before);
    // A batch packed at the wrong level is refused by the level check, not run.
    let low = ev.mod_switch_to_level(&xc, 3).unwrap();
    assert!(t.train_batch(&mut mlp, &low, &yc).is_err());
    assert_eq!(ev.counter(), before);
    let _ = pack_activations(&x, &layout);
}

// ---------------------------------------------------------------------------------------------
// The brief's gate on n16_boot.
// ---------------------------------------------------------------------------------------------

/// 16 -> 16 -> 4, PolyRelu2, batch 32 on n16_boot (2^15 slots, 64 block copies), 3 updates:
/// weights within 2^-8 of plaintext NAG, `bootstraps = boot_count x updates`.
#[test]
#[ignore = "heavy: n16_boot keys (N = 2^16, ~13 GB); run with `-- --ignored n16_boot_keys`, see docs/bootstrap.md §11.6"]
fn n16_boot_keys_three_updates() {
    let params = Arc::new(CkksParams::n16_boot());
    let mut kg = KeyGenerator::new(Arc::clone(&params), rng(21)).unwrap();
    let encoder = Encoder::new(Arc::clone(&params));
    let bs = Bootstrapper::new(
        Arc::clone(&params),
        BootParams::default_n16(),
        &encoder,
        &mut kg,
    )
    .unwrap();
    let layout = Layout::new(params.slots(), 16, 32).unwrap();
    let steps = ni_rotation_steps(&NET_A, &layout, &TransposeKeys::Dedicated).unwrap();
    eprintln!("{} training Galois keys", steps.len());
    let ev = Evaluator::new(
        Arc::clone(&params),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&steps)),
    );
    let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
    let sk = kg.secret_key();
    drop(kg);
    let mut r = rng(22);
    let plain = init(&NET_A, &mut r);
    let acts = [Act::Relu2, Act::Relu2];
    let (mut mlp, state) = NonInteractiveTrainer::encrypt_network(
        &ev,
        &bs,
        &plain,
        acts.iter().map(|a| a.boxed()).collect(),
        &layout,
        BootPlacement::AfterUpdate,
        &enc,
        &encoder,
        &mut r,
    )
    .unwrap();
    let mut t = NonInteractiveTrainer::new(
        &ev,
        &bs,
        NAG,
        layout,
        BootPlacement::AfterUpdate,
        &mlp,
        state,
    )
    .unwrap();
    let spec = t.batch_spec(&mlp).unwrap();
    let mut twin = plain.clone();
    let mut vel: Vec<Array2<f64>> = plain.iter().map(|(w, _)| Array2::zeros(w.dim())).collect();
    let cl = CompactLayout::new(&NET_A, &layout).unwrap();
    let before = ev.counter();
    for it in 0..3 {
        let x = Array2::from_shape_fn((32, 16), |_| uni(&mut r));
        let y = Array2::from_shape_fn((32, 4), |_| 0.5 * uni(&mut r));
        let (xc, yc) = spec
            .encrypt(&enc, &encoder, &layout, &x, &y, &mut r)
            .unwrap();
        twin_step(&mut twin, &mut vel, &acts, &x, &y, true);
        let t0 = std::time::Instant::now();
        t.train_batch(&mut mlp, &xc, &yc).unwrap();
        eprintln!("update {it}: {:.1} s", t0.elapsed().as_secs_f64());
    }
    let got = &ev.counter() - &before;
    let boot_count: u64 = cl.plans.iter().map(|p| boot_count(p, p.d_out) as u64).sum();
    assert_eq!(
        got.bootstraps,
        2 * boot_count * 3,
        "boot_count x 2 (W, U) x updates"
    );
    let dec = Decryptor::new(Arc::clone(&params), sk);
    let he = decrypt_net(&mlp, &dec, &encoder, &layout);
    let err = max_err(&he, &twin);
    let spread = block_spread(&mlp, &dec, &encoder, &layout);
    eprintln!(
        "n16_boot: twin error 2^{:.2}, block spread 2^{:.2}",
        err.log2(),
        spread.log2()
    );
    assert!(err <= TOL, "2^{}", err.log2());
}
