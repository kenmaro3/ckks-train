//! Task 52: ReBoot's eMLP-2/3 block structure (`ckks_train_nn::block`).
//!
//! The network is local blocks plus a separate output stack (R-REB-5): a block is a linear
//! layer that sends no gradient below itself, `z^2 + z`, and a local head (linear); each block
//! is trained by its own squared-error loss only. The output stack is linear, `z^2 + z`,
//! linear, trained by the main loss; its first linear sends no gradient down either. The
//! activation of every block's first layer is the next stage's input.
//!
//! - Interactive (seal_2020, batch 8): 2 blocks of widths 8 and 4 (heads 8 -> 4 and 4 -> 4)
//!   and an output stack 4 -> 4 -> 4. After one iteration every weight is within 2^-10 of the
//!   plaintext block training (and after a second one, which carries the NAG velocities);
//!   block 0's weights do not change when block 1's labels change (bit for bit), while block
//!   1's do.
//! - Non-interactive (`mini`: n16_boot's chain at N = 2^10, real bootstrapping): one block and
//!   the output stack, 3 updates with the lazy placement; the weights match the plaintext block
//!   training, and every update's counter equals the closed form
//!   ([`predict_block_update_with`]: the stages' updates plus one hand-off bootstrap per
//!   micro-batch and stage boundary).
//! - AA: one/two blocks and a mixed AA/AU/WN graph, two updates with two micro-batches,
//!   preserve the same plaintext local-loss graph and NAG state. Every AA handoff reuses the
//!   recorded stage bootstrap ciphertext exactly, with canonical slots, Delta and level;
//!   additional AA handoff bootstraps/time are zero and counters match the predictor.

use std::sync::{Arc, OnceLock};

use ckks_train_ckks::bootstrap::{BootParams, Bootstrapper};
use ckks_train_ckks::keys::SecretKey;
use ckks_train_ckks::{
    CkksParams, Decryptor, Encoder, Encryptor, EvalConfig, Evaluator, HeOps, KeyGenerator,
};
use ckks_train_core::error::HekitError;
use ckks_train_core::sampler::Rng;
use ckks_train_nn::activation::{Activation, Identity, PolyRelu2};
use ckks_train_nn::block::{
    BlockMlp, BlockNiTrainer, BlockTrainer, block_rotation_steps, ni_block_rotation_steps,
    predict_block_update_with,
};
use ckks_train_nn::dense::TransposeKeys;
use ckks_train_nn::layout::{Layout, pack_activations, unpack_activations};
use ckks_train_nn::optim::Nag;
use ckks_train_nn::refresh::ClientRefresher;
use ckks_train_nn::trainer::{LayerOptimizer, Mlp, Trainer};
use ckks_train_nn::trainer_ni::{
    ActCost, BootPlacement, NiOptimizer, NonInteractiveTrainer, boot_schedule,
};
use ndarray::Array2;

/// The brief's gate.
const TOL: f64 = 1.0 / 1024.0;
const LR: f64 = 0.05;
const MU: f64 = 0.9;

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

// ---------------------------------------------------------------------------------------------
// The plaintext block training (independent of the library).
// ---------------------------------------------------------------------------------------------

/// One stage: `(W, b)` per layer; layer 0 `z^2 + z`, layer 1 linear.
type Stage = Vec<(Array2<f64>, Vec<f64>)>;

fn affine(a: &Array2<f64>, w: &Array2<f64>, b: &[f64]) -> Array2<f64> {
    let mut z = a.dot(&w.t());
    for mut row in z.rows_mut() {
        row.iter_mut().zip(b).for_each(|(v, bb)| *v += bb);
    }
    z
}

fn relu2(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|v| v * v + v)
}

/// The block's forward activation: `z^2 + z` of its first layer.
fn stage_hidden(st: &Stage, x: &Array2<f64>) -> Array2<f64> {
    relu2(&affine(x, &st[0].0, &st[0].1))
}

/// One NAG step of a two-layer stage on its own loss (`delta = out - y`, batch means, biases
/// SGD at `lr`, Sutskever NAG `v = mu v + g`, `W -= lr (g + mu v)`), no gradient below layer 0.
fn stage_step(st: &mut Stage, vel: &mut [Array2<f64>], x: &Array2<f64>, y: &Array2<f64>) {
    let n = x.nrows() as f64;
    let z1 = affine(x, &st[0].0, &st[0].1);
    let a1 = relu2(&z1);
    let out = affine(&a1, &st[1].0, &st[1].1);
    let d2 = &out - y;
    let d1 = d2.dot(&st[1].0) * z1.mapv(|v| 2.0 * v + 1.0);
    let grads = [(d1.t().dot(x) / n, d1), (d2.t().dot(&a1) / n, d2)];
    for (i, (g, dz)) in grads.into_iter().enumerate() {
        vel[i] = MU * &vel[i] + &g;
        st[i].0 = &st[i].0 - &(LR * &(&g + &(MU * &vel[i])));
        for (b, s) in st[i].1.iter_mut().zip(dz.sum_axis(ndarray::Axis(0))) {
            *b -= LR * s / n;
        }
    }
}

/// One block-training step: every stage's input from the old weights, then every stage on its
/// own labels (the update order does not matter: no stage reads another's new weights).
fn block_step(
    stages: &mut [Stage],
    vels: &mut [Vec<Array2<f64>>],
    x: &Array2<f64>,
    ys: &[&Array2<f64>],
) {
    let mut inputs = vec![x.clone()];
    for st in &stages[..stages.len() - 1] {
        let h = stage_hidden(st, inputs.last().unwrap());
        inputs.push(h);
    }
    for (s, st) in stages.iter_mut().enumerate() {
        stage_step(st, &mut vels[s], &inputs[s], ys[s]);
    }
}

fn zero_vels(stages: &[Stage]) -> Vec<Vec<Array2<f64>>> {
    stages
        .iter()
        .map(|st| st.iter().map(|(w, _)| Array2::zeros(w.dim())).collect())
        .collect()
}

fn init_stages(stage_dims: &[[usize; 3]], r: &mut Rng, scale: f64) -> Vec<Stage> {
    stage_dims
        .iter()
        .map(|d| {
            d.windows(2)
                .map(|p| {
                    let w = Array2::from_shape_fn((p[1], p[0]), |_| scale * uni(r));
                    let b = (0..p[1]).map(|_| 0.1 * uni(r)).collect();
                    (w, b)
                })
                .collect()
        })
        .collect()
}

fn one_hot(n: usize, c: usize, r: &mut Rng) -> Array2<f64> {
    let mut y = Array2::zeros((n, c));
    for i in 0..n {
        y[[i, (r.next_u64() % c as u64) as usize]] = 1.0;
    }
    y
}

fn acts<E: ckks_train_ckks::HeOps>() -> Vec<Box<dyn Activation<E>>> {
    vec![Box::new(PolyRelu2), Box::new(Identity)]
}

fn decrypt_stage(mlp: &Mlp, dec: &Decryptor, encoder: &Encoder, layout: &Layout) -> Stage {
    mlp.layers
        .iter()
        .map(|(d, _)| {
            let w = d.w.decrypt(dec, encoder, layout);
            let b = unpack_activations(&encoder.decode(&dec.decrypt(&d.b)), layout, w.nrows())
                .row(0)
                .to_vec();
            (w, b)
        })
        .collect()
}

fn max_err(a: &[Stage], b: &[Stage]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(sa, sb)| sa.iter().zip(sb))
        .map(|((wa, ba), (wb, bb))| {
            let w = wa
                .iter()
                .zip(wb)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max);
            let b = ba
                .iter()
                .zip(bb)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max);
            w.max(b)
        })
        .fold(0.0, f64::max)
}

fn decrypt_all(net: &BlockMlp, dec: &Decryptor, encoder: &Encoder, layout: &Layout) -> Vec<Stage> {
    (0..net.num_stages())
        .map(|s| decrypt_stage(net.stage(s), dec, encoder, layout))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Interactive, seal_2020.
// ---------------------------------------------------------------------------------------------

/// Input 6, blocks of widths 8 and 4 with heads to 4 outputs, output stack 4 -> 4 -> 4.
const STAGES: [[usize; 3]; 3] = [[6, 8, 4], [8, 4, 4], [4, 4, 4]];
const BATCH: usize = 8;

struct Seal {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
    layout: Layout,
}

fn seal() -> &'static Seal {
    static S: OnceLock<Seal> = OnceLock::new();
    S.get_or_init(|| {
        let params = Arc::new(CkksParams::seal_2020());
        let layout = Layout::new(params.slots(), 8, BATCH).unwrap();
        let dims: Vec<Vec<usize>> = STAGES.iter().map(|d| d.to_vec()).collect();
        let keys = vec![TransposeKeys::Dedicated; dims.len()];
        let steps = block_rotation_steps(&dims, &layout, &keys).unwrap();
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(3)).unwrap();
        let gks = kg.galois_keys(&steps);
        Seal {
            encoder: Encoder::new(Arc::clone(&params)),
            enc: Encryptor::new(Arc::clone(&params), kg.public_key()),
            dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
            ev: Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), Some(gks)),
            layout,
            params,
        }
    })
}

fn client(s: &Seal, seed: u8) -> ClientRefresher {
    ClientRefresher::new(
        s.dec.clone(),
        s.enc.clone(),
        Encoder::new(Arc::clone(&s.params)),
        rng(seed),
    )
}

fn block_trainer(s: &Seal) -> BlockTrainer<'_> {
    let stages = (0..STAGES.len())
        .map(|i| {
            let optims: Vec<Box<dyn LayerOptimizer>> =
                vec![Box::new(Nag::new(LR, MU)), Box::new(Nag::new(LR, MU))];
            Trainer::new(&s.ev, Box::new(client(s, 10 + i as u8)), optims, s.layout).unwrap()
        })
        .collect();
    BlockTrainer::new(&s.ev, stages, Box::new(client(s, 20))).unwrap()
}

/// Encrypts the network and runs `steps` iterations; `ys[step][stage]`. Returns the decrypted
/// stages after every step.
fn run_interactive(init: &[Stage], x: &[Array2<f64>], ys: &[Vec<Array2<f64>>]) -> Vec<Vec<Stage>> {
    let s = seal();
    let mut r = rng(7);
    let top = s.params.max_level();
    let mut net = BlockMlp::encrypt_at(
        &s.ev,
        init,
        (0..init.len()).map(|_| acts()).collect(),
        &s.layout,
        top,
        &s.enc,
        &s.encoder,
        &mut r,
    )
    .unwrap();
    let mut t = block_trainer(s);
    let specs: Vec<_> = (0..net.num_stages())
        .map(|i| net.stage(i).batch_spec(&s.ev, &s.encoder).unwrap())
        .collect();
    let mut out = Vec::new();
    for (k, xk) in x.iter().enumerate() {
        // Every ciphertext of the step is encrypted before training (block 1's labels last, so
        // changing them changes no other encryption).
        let xc = s.enc.encrypt(
            &s.encoder
                .encode(
                    &pack_activations(xk, &s.layout),
                    specs[0].x_scale,
                    specs[0].x_level,
                )
                .unwrap(),
            &mut r,
        );
        let ycs: Vec<_> = (0..net.num_stages())
            .map(|i| {
                let sp = specs[i];
                s.enc.encrypt(
                    &s.encoder
                        .encode(
                            &pack_activations(&ys[k][i], &s.layout),
                            sp.y_scale,
                            sp.y_level,
                        )
                        .unwrap(),
                    &mut r,
                )
            })
            .collect();
        // Stage order 0, 2, 1 of encryption would change the rng stream; keep 0, 1, 2 and make
        // the label test change only stage 1 (encrypted after stage 0's).
        let refs: Vec<_> = ycs.iter().collect();
        net.train_batch(&mut t, &xc, &refs).unwrap();
        assert_eq!(t.handoffs(), (k + 1) * (net.num_stages() - 1));
        out.push(decrypt_all(&net, &s.dec, &s.encoder, &s.layout));
    }
    out
}

#[allow(clippy::type_complexity)]
fn interactive_data() -> (Vec<Stage>, Vec<Array2<f64>>, Vec<Vec<Array2<f64>>>) {
    let mut r = rng(5);
    let init = init_stages(&STAGES, &mut r, 0.4);
    let x: Vec<Array2<f64>> = (0..2)
        .map(|_| Array2::from_shape_fn((BATCH, STAGES[0][0]), |_| uni(&mut r)))
        .collect();
    let ys: Vec<Vec<Array2<f64>>> = (0..2)
        .map(|_| {
            let y = one_hot(BATCH, 4, &mut r);
            vec![y.clone(), y.clone(), y]
        })
        .collect();
    (init, x, ys)
}

#[test]
fn interactive_block_step_matches_plaintext_block_training() {
    let (init, x, ys) = interactive_data();
    let got = run_interactive(&init, &x, &ys);
    let mut want = init.clone();
    let mut vels = zero_vels(&want);
    for (k, g) in got.iter().enumerate() {
        let yr: Vec<&Array2<f64>> = ys[k].iter().collect();
        block_step(&mut want, &mut vels, &x[k], &yr);
        let e = max_err(g, &want);
        eprintln!(
            "iteration {}: max |W_he - W_plain| = 2^{:.1}",
            k + 1,
            e.log2()
        );
        assert!(e < TOL, "iteration {}: {e}", k + 1);
    }
    // The training moved every stage (the gate is not met by standing still).
    let moved = max_err(&want, &init);
    assert!(moved > 16.0 * TOL, "{moved}");
    // The hand-off is the block activation itself: the output stack's prediction uses it.
    let s = seal();
    let net_final = &got[1];
    let xt = Array2::from_shape_fn((5, STAGES[0][0]), |(i, j)| {
        ((i * 7 + j) as f64 * 0.37).sin()
    });
    let mut h = xt.clone();
    for st in &net_final[..2] {
        h = stage_hidden(st, &h);
    }
    let want_out = affine(
        &relu2(&affine(&h, &net_final[2][0].0, &net_final[2][0].1)),
        &net_final[2][1].0,
        &net_final[2][1].1,
    );
    let mut r = rng(7);
    let net = BlockMlp::encrypt_at(
        &s.ev,
        net_final,
        (0..3).map(|_| acts()).collect(),
        &s.layout,
        s.params.max_level(),
        &s.enc,
        &s.encoder,
        &mut r,
    )
    .unwrap();
    let got_out = net.predict_plain(&s.dec, &s.encoder, &s.layout, &xt);
    let e = got_out
        .iter()
        .zip(&want_out)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(e < 1e-6, "predict_plain {e}");
}

#[test]
fn block0_weights_do_not_depend_on_block1_labels() {
    let (init, x, ys) = interactive_data();
    let x1 = vec![x[0].clone()];
    let a = run_interactive(&init, &x1, &ys[..1]);
    let mut ys_b = ys[..1].to_vec();
    let mut r = rng(99);
    ys_b[0][1] = one_hot(BATCH, 4, &mut r);
    assert_ne!(ys_b[0][1], ys[0][1], "block 1's labels changed");
    let b = run_interactive(&init, &x1, &ys_b);
    // Block 0: bit for bit the same weights.
    for ((wa, ba), (wb, bb)) in a[0][0].iter().zip(&b[0][0]) {
        assert_eq!(wa, wb);
        assert_eq!(ba, bb);
    }
    // Block 1 did see its labels.
    let d1 = max_err(&a[0][1..2], &b[0][1..2]);
    assert!(d1 > 1e-3, "block 1 unchanged ({d1})");
    // The output stack reads block 1's activation from its old weights: unchanged too.
    let d2 = max_err(&a[0][2..], &b[0][2..]);
    assert!(d2 < 1e-9, "output stack {d2}");
}

#[test]
fn block_mlp_shapes_are_checked() {
    let s = seal();
    let mut r = rng(8);
    let init = init_stages(&STAGES, &mut r, 0.4);
    let enc_stage = |st: &Stage| {
        Mlp::encrypt(
            &s.ev,
            st,
            acts(),
            &s.layout,
            &s.enc,
            &s.encoder,
            &mut rng(1),
        )
        .unwrap()
    };
    let blocks: Vec<Mlp> = init[..2].iter().map(enc_stage).collect();
    let output = enc_stage(&init[2]);
    assert!(BlockMlp::new(blocks, output).is_ok());
    // A block whose width is not the next stage's input.
    let blocks = vec![enc_stage(&init[0]), enc_stage(&init[2])];
    let output = enc_stage(&init[2]);
    assert!(matches!(
        BlockMlp::new(blocks, output),
        Err(HekitError::InvalidParams(_))
    ));
    // A one-layer block has no head.
    let one = Mlp::encrypt(
        &s.ev,
        &init[0][..1],
        vec![Box::new(PolyRelu2)],
        &s.layout,
        &s.enc,
        &s.encoder,
        &mut rng(1),
    )
    .unwrap();
    assert!(matches!(
        BlockMlp::new(vec![one], enc_stage(&init[1])),
        Err(HekitError::InvalidParams(_))
    ));
}

// ---------------------------------------------------------------------------------------------
// Non-interactive, mini (real bootstrapping).
// ---------------------------------------------------------------------------------------------

const SLOTS: usize = 512;
/// One block 16 -> 16 (head -> 4) and the output stack 16 -> 8 -> 4.
const NI_STAGES: [[usize; 3]; 2] = [[16, 16, 4], [16, 8, 4]];

fn mini() -> CkksParams {
    let (q, p) = ckks_train_ckks::params::n16_boot_bits();
    CkksParams::new(10, &q, &p, 42, 8).unwrap()
}

struct Mini {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    ev: Evaluator,
    bs: Bootstrapper,
    sk: SecretKey,
    layout: Layout,
}

fn ni_keys(dims: &[Vec<usize>]) -> Vec<TransposeKeys> {
    vec![TransposeKeys::Dedicated; dims.len()]
}

fn mini_fx() -> &'static Mini {
    static M: OnceLock<Mini> = OnceLock::new();
    M.get_or_init(|| {
        let params = Arc::new(mini());
        let layout = Layout::new(SLOTS, 16, 32).unwrap();
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(1)).unwrap();
        let encoder = Encoder::new(Arc::clone(&params));
        let bp = BootParams {
            log_slots: 9,
            ..BootParams::default_n16()
        };
        let bs = Bootstrapper::new(Arc::clone(&params), bp, &encoder, &mut kg).unwrap();
        let dims: Vec<Vec<usize>> = NI_STAGES.iter().map(|d| d.to_vec()).collect();
        let steps = ni_block_rotation_steps(&dims, &layout, &ni_keys(&dims)).unwrap();
        let gks = kg.galois_keys(&steps);
        Mini {
            enc: Encryptor::new(Arc::clone(&params), kg.public_key()),
            ev: Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), Some(gks)),
            sk: kg.secret_key(),
            encoder,
            bs,
            params,
            layout,
        }
    })
}

#[test]
fn ni_block_updates_match_twin_and_closed_form() {
    let m = mini_fx();
    // A fork of the shared evaluator (same keys, own counter): other tests run in parallel.
    let ev = m.ev.fork();
    let mut r = rng(4);
    let init = init_stages(&NI_STAGES, &mut r, 0.2);
    let placement = BootPlacement::WhenNeeded;
    let (mut net, states) = BlockNiTrainer::encrypt_network(
        &ev,
        &m.bs,
        &init,
        (0..init.len()).map(|_| acts()).collect(),
        &m.layout,
        placement,
        &m.enc,
        &m.encoder,
        &mut r,
    )
    .unwrap();
    let dims = net.stage_dims();
    let keys = ni_keys(&dims);
    let optim = NiOptimizer::Nag {
        lr: LR,
        momentum: MU,
    };
    let stages = states
        .into_iter()
        .enumerate()
        .map(|(s, st)| {
            NonInteractiveTrainer::new(&ev, &m.bs, optim, m.layout, placement, net.stage(s), st)
                .unwrap()
                .with_transpose_keys(keys[s].clone())
        })
        .collect();
    let mut t = BlockNiTrainer::new(&ev, &m.bs, stages).unwrap();
    let specs: Vec<_> = (0..net.num_stages())
        .map(|s| t.stages()[s].batch_spec(net.stage(s)).unwrap())
        .collect();
    let per_boot = m.bs.cost_per_boot();
    let boot_ks = m.bs.key_switches_per_boot();
    let act_costs: Vec<Vec<ActCost>> = dims
        .iter()
        .map(|_| vec![ActCost::QUADRATIC, ActCost::IDENTITY])
        .collect();
    // The a-priori schedule of every stage (review M-4: the counter is checked against it, not
    // only against the observed decisions).
    let start = t.stages()[0].start_level();
    let relu_id: [&dyn Activation; 2] = [&PolyRelu2, &Identity];
    let planned: Vec<Vec<bool>> = dims
        .iter()
        .map(|d| {
            boot_schedule(
                d,
                &relu_id,
                &m.layout,
                placement,
                true,
                start,
                start + 1,
                &[true; 3],
            )
            .unwrap()
        })
        .collect();
    let mut want = init.clone();
    let mut vels = zero_vels(&want);
    let mut booted_any = false;
    for k in 0..3 {
        let x = Array2::from_shape_fn((32, 16), |_| 0.5 * uni(&mut r));
        let y = one_hot(32, 4, &mut r);
        let enc_at = |a: &Array2<f64>, scale: f64, level: usize, r: &mut Rng| {
            m.enc.encrypt(
                &m.encoder
                    .encode(&pack_activations(a, &m.layout), scale, level)
                    .unwrap(),
                r,
            )
        };
        let xc = enc_at(&x, specs[0].x_scale, specs[0].x_level, &mut r);
        let ycs: Vec<_> = specs
            .iter()
            .map(|sp| enc_at(&y, sp.y_scale, sp.y_level, &mut r))
            .collect();
        let yr: Vec<_> = ycs.iter().collect();
        t.set_next_batch_full(Some(true));
        let before = ev.counter();
        net.train_step_ni(&mut t, &[(&xc, &yr[..], 32)]).unwrap();
        let c = &ev.counter() - &before;
        let booted: Vec<bool> = t.stages().iter().map(|s| s.last_update_booted()).collect();
        booted_any |= booted.iter().any(|&b| b);
        // The last update's next step is undeclared in the a-priori schedule (planned as
        // partial); here it was declared full, so compare the first two only.
        if k < 2 {
            let want_b: Vec<bool> = planned.iter().map(|p| p[k]).collect();
            assert_eq!(booted, want_b, "update {k}: a-priori schedule");
        }
        let p = predict_block_update_with(
            &dims,
            &act_costs,
            &m.layout,
            &keys,
            placement,
            1,
            true,
            &booted,
            EvalConfig::default(),
        )
        .unwrap();
        assert_eq!(p.handoff_bootstraps, 1, "one boundary, one micro-batch");
        let nb = p.bootstraps();
        assert_eq!(c.bootstraps, nb, "update {k}");
        assert_eq!(
            c.rotations,
            p.rotations() + nb * per_boot.rotations,
            "update {k}"
        );
        assert_eq!(c.relins, p.relins() + nb * per_boot.relins, "update {k}");
        assert_eq!(c.key_switches(), p.key_switches(boot_ks), "update {k}");
        let yrr = [&y, &y];
        block_step(&mut want, &mut vels, &x, &yrr);
    }
    assert_eq!(t.handoff_bootstraps(), 3);
    assert!(
        booted_any,
        "the lazy schedule bootstrapped the weights at least once"
    );
    let dec = Decryptor::new(Arc::clone(&m.params), m.sk.clone());
    let got = decrypt_all(&net, &dec, &m.encoder, &m.layout);
    let e = max_err(&got, &want);
    let moved = max_err(&want, &init);
    eprintln!(
        "NI blocks: weights 2^{:.1} from the twin (moved 2^{:.1})",
        e.log2(),
        moved.log2()
    );
    assert!(e < TOL, "{e}");
    assert!(moved > 16.0 * TOL);
}

/// Catches duplicate AA handoff bootstraps, the wrong micro-batch/layer accessor, stale
/// activations after an update, lost optimizer state and noncanonical boundary packing.
fn check_aa_handoffs(stage_dims: &[[usize; 3]], placements: &[BootPlacement], additional: u64) {
    let m = mini_fx();
    let ev = m.ev.fork();
    let mut r = rng(24);
    let init = init_stages(stage_dims, &mut r, 0.2);
    let optim = NiOptimizer::Nag {
        lr: LR,
        momentum: MU,
    };
    let mut mlps = Vec::new();
    let mut states = Vec::new();
    for (st, &placement) in init.iter().zip(placements) {
        let (mlp, state) = NonInteractiveTrainer::encrypt_network(
            &ev,
            &m.bs,
            st,
            acts(),
            &m.layout,
            placement,
            &m.enc,
            &m.encoder,
            &mut r,
        )
        .unwrap();
        mlps.push(mlp);
        states.push(state);
    }
    let output = mlps.pop().unwrap();
    let mut net = BlockMlp::new(mlps, output).unwrap();
    let dims = net.stage_dims();
    let keys = ni_keys(&dims);
    let stages = states
        .into_iter()
        .enumerate()
        .map(|(s, st)| {
            NonInteractiveTrainer::new(&ev, &m.bs, optim, m.layout, placements[s], net.stage(s), st)
                .unwrap()
                .with_transpose_keys(keys[s].clone())
        })
        .collect();
    let mut t = BlockNiTrainer::new(&ev, &m.bs, stages)
        .expect("AA stages must reuse their existing activation bootstrap at a block boundary");
    let mut want = init.clone();
    let mut vels = zero_vels(&want);
    let dec = Decryptor::new(Arc::clone(&m.params), m.sk.clone());
    let acts: Vec<Vec<ActCost>> = dims
        .iter()
        .map(|_| vec![ActCost::QUADRATIC, ActCost::IDENTITY])
        .collect();
    let mut expected_handoffs = 0;
    // Two updates carry the nonzero NAG velocity; two distinct micro-batches catch [0][0]
    // reuse for every micro. Both one-block and two-block graphs are exercised below.
    for update in 0..2 {
        let specs: Vec<_> = (0..dims.len())
            .map(|s| t.stages()[s].batch_spec(net.stage(s)).unwrap())
            .collect();
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut xcs = Vec::new();
        let mut ycs = Vec::new();
        for _ in 0..2 {
            let x = Array2::from_shape_fn((32, stage_dims[0][0]), |_| 0.5 * uni(&mut r));
            let y = one_hot(32, 4, &mut r);
            let enc_at = |a: &Array2<f64>, scale, level, r: &mut Rng| {
                m.enc.encrypt(
                    &m.encoder
                        .encode(&pack_activations(a, &m.layout), scale, level)
                        .unwrap(),
                    r,
                )
            };
            xcs.push(enc_at(&x, specs[0].x_scale, specs[0].x_level, &mut r));
            ycs.push(
                specs
                    .iter()
                    .map(|sp| enc_at(&y, sp.y_scale, sp.y_level, &mut r))
                    .collect::<Vec<_>>(),
            );
            xs.push(x);
            ys.push(y);
        }
        let refs: Vec<Vec<_>> = ycs.iter().map(|ys| ys.iter().collect()).collect();
        let batches: Vec<_> = xcs
            .iter()
            .zip(&refs)
            .map(|(x, ys)| (x, &ys[..], 32))
            .collect();
        // Independent plaintext old-weight activations, before advancing any stage.
        let mut hidden = Vec::new();
        let mut inputs = xs.clone();
        for st in &want[..want.len() - 1] {
            inputs = inputs.iter().map(|x| stage_hidden(st, x)).collect();
            hidden.push(inputs.clone());
        }
        let before = ev.counter();
        let before_act: u64 = t.stages().iter().map(|s| s.activation_bootstraps()).sum();
        let before_w: u64 = t.stages().iter().map(|s| s.weight_bootstraps()).sum();
        t.set_next_batch_full(Some(true));
        net.train_step_ni(&mut t, &batches).unwrap();
        let c = &ev.counter() - &before;
        assert_eq!(t.last_handoff_bootstraps().len(), dims.len() - 1);
        for (boundary, handoffs) in t.last_handoff_bootstraps().iter().enumerate() {
            assert_eq!(handoffs.len(), 2);
            for (micro, h) in handoffs.iter().enumerate() {
                assert_eq!(h.scale(), m.params.scale(), "canonical Delta handoff");
                assert_eq!(
                    h.level(),
                    m.bs.output_level(),
                    "no needless handoff mod-switch"
                );
                assert!(h.level() >= t.stages()[boundary + 1].start_level());
                if placements[boundary] == BootPlacement::AfterActivation {
                    let aa = &t.stages()[boundary].last_activation_bootstraps()[micro][0];
                    assert_eq!(h.len(), aa.len());
                    assert_eq!((h.scale(), h.level()), (aa.scale(), aa.level()));
                    for (p, q) in h.polys().iter().zip(aa.polys()) {
                        assert_eq!(
                            p.limbs(),
                            q.limbs(),
                            "handoff must reuse recorded AA output"
                        );
                    }
                }
                let got = m.encoder.decode(&dec.decrypt(h));
                let packed = pack_activations(&hidden[boundary][micro], &m.layout);
                let e = got
                    .iter()
                    .zip(&packed)
                    .map(|(g, w)| (g - w).abs())
                    .fold(0.0, f64::max);
                assert!(
                    e < TOL,
                    "boundary {boundary}, micro {micro}, all canonical slots: {e}"
                );
            }
        }
        // Homogeneous formula is checked as a whole. A mixed schedule is summed per stage,
        // with the literal boundary count below: AA then AU has exactly two new boots.
        let mut predictions = Vec::new();
        for s in 0..dims.len() {
            predictions.push(
                predict_block_update_with(
                    &dims[s..s + 1],
                    &acts[s..s + 1],
                    &m.layout,
                    &keys[s..s + 1],
                    placements[s],
                    2,
                    true,
                    &[t.stages()[s].last_update_booted()],
                    EvalConfig::default(),
                )
                .unwrap(),
            );
        }
        if additional == 0 {
            let booted: Vec<_> = t.stages().iter().map(|s| s.last_update_booted()).collect();
            let p = predict_block_update_with(
                &dims,
                &acts,
                &m.layout,
                &keys,
                BootPlacement::AfterActivation,
                2,
                true,
                &booted,
                EvalConfig::default(),
            )
            .unwrap();
            assert_eq!(p.handoff_bootstraps, 0, "AA reuses stage work");
            assert_eq!(c.bootstraps, p.bootstraps());
            assert_eq!(
                t.bootstrap_seconds(),
                t.stages()
                    .iter()
                    .map(|s| s.bootstrap_seconds())
                    .sum::<f64>(),
                "AA reuse adds zero handoff bootstrap time"
            );
        }
        expected_handoffs += additional;
        assert_eq!(t.handoff_bootstraps(), expected_handoffs);
        let nb = predictions.iter().map(|p| p.bootstraps()).sum::<u64>() + additional;
        let per = m.bs.cost_per_boot();
        assert_eq!(c.bootstraps, nb);
        assert_eq!(
            c.rotations,
            predictions.iter().map(|p| p.rotations()).sum::<u64>() + nb * per.rotations
        );
        assert_eq!(
            c.relins,
            predictions.iter().map(|p| p.relins()).sum::<u64>() + nb * per.relins
        );
        assert_eq!(
            c.key_switches(),
            predictions
                .iter()
                .map(|p| p.key_switches(m.bs.key_switches_per_boot()))
                .sum::<u64>()
                + additional * m.bs.key_switches_per_boot()
        );
        let act: u64 = t.stages().iter().map(|s| s.activation_bootstraps()).sum();
        let w: u64 = t.stages().iter().map(|s| s.weight_bootstraps()).sum();
        assert_eq!(
            c.bootstraps,
            act - before_act + w - before_w + additional,
            "every existing stage bootstrap counted exactly once"
        );
        let x_all = ndarray::concatenate(
            ndarray::Axis(0),
            &xs.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let y_all = ndarray::concatenate(
            ndarray::Axis(0),
            &ys.iter().map(|y| y.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        block_step(&mut want, &mut vels, &x_all, &vec![&y_all; dims.len()]);
        let got = decrypt_all(&net, &dec, &m.encoder, &m.layout);
        let e = max_err(&got, &want);
        eprintln!(
            "AA blocks={}, update={update}: plaintext weight error 2^{:.2}",
            dims.len() - 1,
            e.log2()
        );
        assert!(e < TOL, "update {update}: {e}");
    }
    assert!(
        max_err(&want, &init) > 16.0 * TOL,
        "every test must actually train"
    );
}

#[test]
fn ni_aa_one_block_reuses_bootstraps_across_microbatches_and_updates() {
    check_aa_handoffs(&NI_STAGES, &[BootPlacement::AfterActivation; 2], 0);
}

#[test]
fn ni_aa_two_blocks_preserve_local_training_and_canonical_handoffs() {
    check_aa_handoffs(
        &[[16, 16, 4], [16, 8, 4], [8, 8, 4]],
        &[BootPlacement::AfterActivation; 3],
        0,
    );
}

#[test]
fn ni_aa_mixed_placements_bootstrap_only_non_aa_boundaries() {
    check_aa_handoffs(
        &[[16, 16, 4], [16, 8, 4], [8, 8, 4]],
        &[
            BootPlacement::AfterActivation,
            BootPlacement::AfterUpdate,
            BootPlacement::WhenNeeded,
        ],
        2,
    );
}

// ---------------------------------------------------------------------------------------------
// Non-interactive: the hand-off outside |m| <= 1, and stage independence.
// ---------------------------------------------------------------------------------------------

/// The block trainer on `mini` with WhenNeeded and NAG, for `init`.
fn ni_setup(init: &[Stage], r: &mut Rng) -> (BlockMlp, BlockNiTrainer<'static>) {
    let m = mini_fx();
    let placement = BootPlacement::WhenNeeded;
    let (net, states) = BlockNiTrainer::encrypt_network(
        &m.ev,
        &m.bs,
        init,
        (0..init.len()).map(|_| acts()).collect(),
        &m.layout,
        placement,
        &m.enc,
        &m.encoder,
        r,
    )
    .unwrap();
    let optim = NiOptimizer::Nag {
        lr: LR,
        momentum: MU,
    };
    let stages = states
        .into_iter()
        .enumerate()
        .map(|(s, st)| {
            NonInteractiveTrainer::new(&m.ev, &m.bs, optim, m.layout, placement, net.stage(s), st)
                .unwrap()
        })
        .collect();
    (net, BlockNiTrainer::new(&m.ev, &m.bs, stages).unwrap())
}

/// One update on `x` with labels `ys[s]` per stage.
fn ni_update(
    net: &mut BlockMlp,
    t: &mut BlockNiTrainer<'static>,
    x: &Array2<f64>,
    ys: &[&Array2<f64>],
    r: &mut Rng,
) {
    let m = mini_fx();
    let specs: Vec<_> = (0..net.num_stages())
        .map(|s| t.stages()[s].batch_spec(net.stage(s)).unwrap())
        .collect();
    let enc_at = |a: &Array2<f64>, scale: f64, level: usize, r: &mut Rng| {
        m.enc.encrypt(
            &m.encoder
                .encode(&pack_activations(a, &m.layout), scale, level)
                .unwrap(),
            r,
        )
    };
    let xc = enc_at(x, specs[0].x_scale, specs[0].x_level, r);
    let ycs: Vec<_> = specs
        .iter()
        .zip(ys)
        .map(|(sp, y)| enc_at(y, sp.y_scale, sp.y_level, r))
        .collect();
    let yr: Vec<_> = ycs.iter().collect();
    t.set_next_batch_full(Some(true));
    net.train_step_ni(t, &[(&xc, &yr[..], x.nrows())]).unwrap();
}

/// The max |hand-off - twin activation| of the last update (block 0, 16 wide, `n` rows).
fn handoff_err(t: &BlockNiTrainer<'static>, want: &Array2<f64>) -> f64 {
    let m = mini_fx();
    let dec = Decryptor::new(Arc::clone(&m.params), m.sk.clone());
    let h = &t.last_handoff_bootstraps()[0][0];
    assert_eq!(h.scale(), m.params.scale(), "the hand-off is at Delta");
    let got = unpack_activations(&m.encoder.decode(&dec.decrypt(h)), &m.layout, want.ncols());
    want.iter()
        .zip(got.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
}

/// Review I-1: a hand-off whose input exceeds the bootstrap's `|m| <= 1` precondition (here
/// `max |a|` in [2, 4], as eMLP-2/3's 2.0-2.3) goes through a real bootstrap with a bounded
/// error. Bound: `2^-17`, i.e. no worse than Task 38's n16_boot measurement at `|m| = 4`
/// (17.38 bits absolute, `docs/bootstrap.md` §11.4); `mini` (N = 2^10) is more precise than
/// n16_boot, so this leaves margin while still catching an EvalMod range failure (which gives
/// O(1) errors). The in-range hand-off of the same network is printed for comparison.
#[test]
fn ni_handoff_above_one_is_bounded() {
    let mut r = rng(11);
    let mut init = init_stages(&NI_STAGES, &mut r, 0.2);
    let x = Array2::from_shape_fn((32, 16), |_| uni(&mut r));
    let y = one_hot(32, 4, &mut r);
    // Scale block 0's weights so that max z = 1.5 on x: max (z^2 + z) = 3.75.
    let z = affine(&x, &init[0][0].0, &init[0][0].1);
    let zmax = z.iter().copied().fold(f64::MIN, f64::max);
    let c = 1.5 / zmax;
    init[0][0].0.mapv_inplace(|w| w * c);
    init[0][0].1.iter_mut().for_each(|b| *b *= c);
    let mut errs = Vec::new();
    for (case, scale) in [("in range", 0.25), ("above 1", 1.0)] {
        let mut st = init.clone();
        st[0][0].0.mapv_inplace(|w| w * scale);
        st[0][0].1.iter_mut().for_each(|b| *b *= scale);
        let want = stage_hidden(&st[0], &x);
        let amax = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let (mut net, mut t) = ni_setup(&st, &mut r);
        ni_update(&mut net, &mut t, &x, &[&y, &y], &mut r);
        let e = handoff_err(&t, &want);
        eprintln!(
            "hand-off {case}: max |a| = {amax:.3}, max |boot(a) - a| = 2^{:.2}",
            e.log2()
        );
        if case == "above 1" {
            assert!((2.0..=4.0).contains(&amax), "max |a| = {amax}");
        } else {
            assert!(amax <= 1.0, "max |a| = {amax}");
        }
        errs.push(e);
    }
    assert!(
        errs[1] < 2f64.powi(-17),
        "above-range hand-off 2^{}",
        errs[1].log2()
    );
    assert!(errs[0] < 2f64.powi(-17));
}

/// Review M-4: the non-interactive stages are independent too: the output stack's labels do
/// not reach block 0 (bit for bit), but do reach the output stack.
#[test]
fn ni_block0_ignores_output_stack_labels() {
    let mut r0 = rng(12);
    let init = init_stages(&NI_STAGES, &mut r0, 0.2);
    let x = Array2::from_shape_fn((32, 16), |_| 0.5 * uni(&mut r0));
    let y = one_hot(32, 4, &mut r0);
    let y_other = one_hot(32, 4, &mut r0);
    assert_ne!(y, y_other);
    let m = mini_fx();
    let dec = Decryptor::new(Arc::clone(&m.params), m.sk.clone());
    let run = |yo: &Array2<f64>| {
        let mut r = rng(13);
        let (mut net, mut t) = ni_setup(&init, &mut r);
        ni_update(&mut net, &mut t, &x, &[&y, yo], &mut r);
        decrypt_all(&net, &dec, &m.encoder, &m.layout)
    };
    let a = run(&y);
    let b = run(&y_other);
    for ((wa, ba), (wb, bb)) in a[0].iter().zip(&b[0]) {
        assert_eq!(wa, wb);
        assert_eq!(ba, bb);
    }
    assert!(max_err(&a[1..], &b[1..]) > 1e-3);
}
