//! Task 52: ReBoot's eMLP-2/3 block structure (`hekit_nn::block`).
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

use std::sync::{Arc, OnceLock};

use hekit_ckks::bootstrap::{BootParams, Bootstrapper};
use hekit_ckks::keys::SecretKey;
use hekit_ckks::{CkksParams, Decryptor, Encoder, Encryptor, EvalConfig, Evaluator, KeyGenerator};
use hekit_core::error::HekitError;
use hekit_core::sampler::Rng;
use hekit_nn::activation::{Activation, Identity, PolyRelu2};
use hekit_nn::block::{
    BlockMlp, BlockNiTrainer, BlockTrainer, block_rotation_steps, ni_block_rotation_steps,
    predict_block_update_with,
};
use hekit_nn::dense::TransposeKeys;
use hekit_nn::layout::{Layout, pack_activations, unpack_activations};
use hekit_nn::optim::Nag;
use hekit_nn::refresh::ClientRefresher;
use hekit_nn::trainer::{LayerOptimizer, Mlp, Trainer};
use hekit_nn::trainer_ni::{ActCost, BootPlacement, NiOptimizer, NonInteractiveTrainer};
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

fn acts<E: hekit_ckks::HeOps>() -> Vec<Box<dyn Activation<E>>> {
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
    let (q, p) = hekit_ckks::params::n16_boot_bits();
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
    let mut r = rng(4);
    let init = init_stages(&NI_STAGES, &mut r, 0.2);
    let placement = BootPlacement::WhenNeeded;
    let (mut net, states) = BlockNiTrainer::encrypt_network(
        &m.ev,
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
            NonInteractiveTrainer::new(&m.ev, &m.bs, optim, m.layout, placement, net.stage(s), st)
                .unwrap()
                .with_transpose_keys(keys[s].clone())
        })
        .collect();
    let mut t = BlockNiTrainer::new(&m.ev, &m.bs, stages).unwrap();
    let specs: Vec<_> = (0..net.num_stages())
        .map(|s| t.stages()[s].batch_spec(net.stage(s)).unwrap())
        .collect();
    let per_boot = m.bs.cost_per_boot();
    let boot_ks = m.bs.key_switches_per_boot();
    let act_costs: Vec<Vec<ActCost>> = dims
        .iter()
        .map(|_| vec![ActCost::QUADRATIC, ActCost::IDENTITY])
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
        let before = m.ev.counter();
        net.train_step_ni(&mut t, &[(&xc, &yr[..], 32)]).unwrap();
        let c = &m.ev.counter() - &before;
        let booted: Vec<bool> = t.stages().iter().map(|s| s.last_update_booted()).collect();
        booted_any |= booted.iter().any(|&b| b);
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

#[test]
fn ni_block_trainer_refuses_after_activation() {
    let m = mini_fx();
    let mut r = rng(4);
    let init = init_stages(&NI_STAGES, &mut r, 0.2);
    let placement = BootPlacement::AfterActivation;
    let (net, states) = BlockNiTrainer::encrypt_network(
        &m.ev,
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
    assert!(matches!(
        BlockNiTrainer::new(&m.ev, &m.bs, stages),
        Err(HekitError::InvalidParams(_))
    ));
}
