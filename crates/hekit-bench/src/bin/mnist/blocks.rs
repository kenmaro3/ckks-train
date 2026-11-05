//! Task 52: ReBoot's eMLP-2/3 in the `mnist` bench (`--blocks W1[,W2,...]`).
//!
//! The network is ReBoot's block structure as recorded in R-REB-5
//! (`results/quadruped/raw/reboot_emlp23_full/config_values.json`, `structure_summary`; see
//! [`hekit_nn::block`]): local blocks `in -> W_i` (`z^2 + z`) with local heads
//! `W_i -> --out-width`, then the output stack `W_last -> --hidden` (`z^2 + z`)
//! `-> --out-width`. eMLP-2 is `--blocks 64 --hidden 32`, eMLP-3 `--blocks 128,64 --hidden 32`
//! (`configs/reboot_emlp{2,3}.json`). Every stage has its own loss (`delta = out - y`) and
//! every layer its own NAG velocity at the shared lr; the accuracy and the per-step metrics are
//! those of the output stack.
//!
//! - `--mode plain`: the plaintext twin ([`block_twin_step`]); with `--compare-steps FILE`
//!   (ReBoot's `twin_steps_*.tsv`: `step epoch batch lr batch_loss batch_acc`) the run records,
//!   per step, the relative batch-loss difference and whether the batch accuracy (argmax over
//!   all outputs, ReBoot's printed definition) is equal.
//! - `--mode noninteractive` (CPU): [`BlockNiTrainer`], one non-interactive trainer per stage
//!   and one hand-off bootstrap per micro-batch at every stage boundary; every update's counter
//!   is checked against [`predict_block_update_with`] with each stage's a-priori bootstrap
//!   schedule ([`boot_schedule`]). `--plan` prints the prediction without keys.
//! - `--init-weights DIR` reads ReBoot's per-layer files in its trainable order:
//!   `linear_fwd_{i:02}.npy` and `linear_lrn_{i:02}.npy` for block `i`, then
//!   `linear_{nb:02}.npy`, `linear_{nb+1:02}.npy` for the output stack (`nb` blocks), each
//!   `(in, out)` and transposed (a square one needs `in x out` / `in_out` after its name in the
//!   manifest); SHA-256 against `--init-weights-manifest` (an entry `<dir name>/<file>`).
//!   Biases: zero with `--no-bias` (ReBoot has none), else drawn.

use super::*;

use hekit_bench::twin::{block_forward_plain, block_hidden, block_twin_step};
use hekit_ckks::Ciphertext;
use hekit_nn::block::{
    BlockMicroBatch, BlockNiTrainer, ni_block_rotation_steps, predict_block_update_with,
};
use hekit_nn::trainer_ni::levels_per_iter_of;

/// The stages' dimensions: blocks `[in, W_i, out]`, then the output stack
/// `[W_last, hidden, out]`.
pub(super) fn stage_dims(input: usize, args: &Args) -> Vec<[usize; 3]> {
    let mut v = Vec::with_capacity(args.blocks.len() + 1);
    let mut d_in = input;
    for &w in &args.blocks {
        v.push([d_in, w, args.out_width]);
        d_in = w;
    }
    v.push([d_in, args.hidden, args.out_width]);
    v
}

/// ReBoot's layer names in its trainable order (R-REB-5 `trainable_layer_order`).
pub(super) fn reboot_layer_names(n_blocks: usize) -> Vec<String> {
    let mut v = Vec::new();
    for i in 0..n_blocks {
        v.push(format!("linear_fwd_{i:02}"));
        v.push(format!("linear_lrn_{i:02}"));
    }
    v.push(format!("linear_{n_blocks:02}"));
    v.push(format!("linear_{:02}", n_blocks + 1));
    v
}

/// The initial stages: [`init_net`] per stage from one stream, then, with `--init-weights`,
/// every weight replaced by ReBoot's file (module docs).
fn initial_stages(
    dims: &[[usize; 3]],
    r: &mut Rng,
    args: &Args,
) -> Result<Vec<Vec<Layer>>, String> {
    let mut stages: Vec<Vec<Layer>> = dims
        .iter()
        .map(|d| init_net(d, r, &args.init, args.no_bias))
        .collect();
    let Some(dir) = &args.init_weights else {
        return Ok(stages);
    };
    let meta = args
        .init_weights_manifest
        .as_ref()
        .map(|p| std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display())))
        .transpose()?
        .unwrap_or_default();
    let dir_name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let names = reboot_layer_names(args.blocks.len());
    for (layer, name) in stages.iter_mut().flatten().zip(&names) {
        let path = dir.join(format!("{name}.npy"));
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        match manifest_sha256(&meta, &format!("{dir_name}/{name}.npy"))
            .or_else(|| manifest_sha256(&meta, &format!("{name}.npy")))
        {
            Some(want) => {
                let got = sha256_hex(&bytes);
                if got != want {
                    return Err(format!("{}: sha256 {got}, manifest {want}", path.display()));
                }
            }
            None => eprintln!(
                "mnist: {}: no sha256 in the manifest, not checked",
                path.display()
            ),
        }
        let m = parse_npy_f64(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let (d_out, d_in) = layer.w.dim();
        layer.w = if d_out == d_in {
            let at = meta.find(&format!("{name}.npy")).ok_or_else(|| {
                format!("{name} is square: its orientation must be in the manifest")
            })?;
            let tail = &meta[at..meta.len().min(at + 400)];
            if tail.contains("in x out") || tail.contains("in_out") {
                m.t().to_owned()
            } else {
                return Err(format!("{name} is square and not listed as in x out"));
            }
        } else if m.dim() == (d_in, d_out) {
            m.t().to_owned()
        } else if m.dim() == (d_out, d_in) {
            m
        } else {
            return Err(format!(
                "{name} has shape {:?}, the layer is {d_out} x {d_in}",
                m.dim()
            ));
        };
    }
    Ok(stages)
}

/// ReBoot's per-step twin log: `(lr, batch_loss, batch_acc)` per step (`#` lines and the
/// header skipped).
fn read_twin_steps(path: &std::path::Path) -> Result<Vec<(f64, f64, f64)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("step") && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() < 6 {
                return Err(format!("{}: short line {l:?}", path.display()));
            }
            let p = |s: &str| s.trim().parse::<f64>().map_err(|e| format!("{s}: {e}"));
            Ok((p(f[3])?, p(f[4])?, p(f[5])?))
        })
        .collect()
}

fn dims_json(dims: &[[usize; 3]]) -> J {
    J::A(
        dims.iter()
            .map(|d| J::A(d.iter().map(|&v| J::I(v as i64)).collect()))
            .collect(),
    )
}

fn zero_vels(stages: &[Vec<Layer>]) -> Vec<Vec<Array2<f64>>> {
    stages
        .iter()
        .map(|st| st.iter().map(|l| Array2::zeros(l.w.dim())).collect())
        .collect()
}

/// The max |activation| each block hands on (the hand-off bootstrap's input), on `x`.
fn handoff_max(stages: &[Vec<Layer>], x: &Array2<f64>) -> Vec<f64> {
    let mut a = x.clone();
    stages[..stages.len() - 1]
        .iter()
        .map(|st| {
            a = block_hidden(st, &a);
            max_abs(a.iter().copied())
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Plain.
// ---------------------------------------------------------------------------------------------

pub(super) fn run_plain_blocks(args: &Args) -> Result<(), String> {
    let load_start = loadavg();
    let started = unix_now();
    let (train, test) = load(args)?;
    let dims = stage_dims(train.features(), args);
    let mut r = rng(args.seed, 2);
    let init = initial_stages(&dims, &mut r, args)?;
    let mut twin = init.clone();
    let mut vels = zero_vels(&twin);
    let y_train = one_hot(&train.labels, args.out_width);
    let n = train.len();
    let schedule = args
        .lr_schedule
        .as_deref()
        .map(|p| read_lr_schedule(p, args.lr_key.as_deref()))
        .transpose()?;
    let orders = args
        .epoch_orders
        .as_deref()
        .map(|p| read_epoch_orders(p, n))
        .transpose()?;
    let schedule = schedule
        .map(|v| per_update_schedule(v, &orders, n, args))
        .transpose()?;
    check_schedule_inputs(&schedule, &orders, n, args)?;
    let reference = args
        .compare_steps
        .as_deref()
        .map(read_twin_steps)
        .transpose()?;
    let (mut step_loss, mut step_acc_all, mut lrs) = (Vec::new(), Vec::new(), Vec::new());
    let mut handoff_abs = vec![0.0f64; args.blocks.len()];
    let mut epochs = Vec::new();
    let mut updates = 0usize;
    let stages_n = dims.len();
    for epoch in 0..args.epochs {
        let order = epoch_order(&orders, epoch, n)?;
        for start in (0..order.len()).step_by(args.batch) {
            let end = (start + args.batch).min(order.len());
            let idx = &order[start..end];
            let x = train.images.select(Axis(0), idx);
            let y = y_train.select(Axis(0), idx);
            // ReBoot's printed per-step metrics of the model output, before the update.
            let o = block_forward_plain(&twin, &x);
            let labels: Vec<u8> = idx.iter().map(|&i| train.labels[i]).collect();
            step_loss.push((&o - &y).mapv(|v| v * v).sum());
            step_acc_all.push(accuracy_over(&o, &labels, o.ncols()));
            for (m, v) in handoff_abs.iter_mut().zip(handoff_max(&twin, &x)) {
                *m = m.max(v);
            }
            let lr = step_lr(&schedule, updates, args)?;
            lrs.push(lr);
            let ys = vec![&y; stages_n];
            let blr = if args.no_bias { 0.0 } else { lr };
            block_twin_step(
                &mut twin,
                &mut vels,
                &x,
                &ys,
                lr,
                args.momentum,
                blr,
                WeightDecay::NONE,
            );
            updates += 1;
        }
        let out = block_forward_plain(&twin, &test.images);
        eprintln!(
            "mnist plain blocks: epoch {epoch}: test acc {:.4}",
            accuracy(&out, &test.labels)
        );
        epochs.push(obj(vec![
            ("epoch", J::I(epoch as i64)),
            ("test_acc", J::N(accuracy(&out, &test.labels))),
            (
                "test_acc_all_outputs",
                J::N(accuracy_over(&out, &test.labels, out.ncols())),
            ),
        ]));
    }
    let out = block_forward_plain(&twin, &test.images);
    let test_acc = accuracy(&out, &test.labels);
    let comparison = match &reference {
        None => J::S("none (--compare-steps not given)".into()),
        Some(rf) => {
            if rf.len() != updates {
                return Err(format!(
                    "--compare-steps has {} steps, the run {updates}",
                    rf.len()
                ));
            }
            let rel: Vec<f64> = rf
                .iter()
                .zip(&step_loss)
                .map(|(r, l)| (l - r.1).abs() / r.1.abs())
                .collect();
            let acc_equal = rf
                .iter()
                .zip(&step_acc_all)
                .filter(|(r, a)| (*a - r.2).abs() < 1e-6)
                .count();
            let lr_rel = rf
                .iter()
                .zip(&lrs)
                .map(|(r, l)| (l / args.batch as f64 - r.0).abs() / r.0)
                .fold(0.0, f64::max);
            let max_rel = rel.iter().copied().fold(0.0, f64::max);
            eprintln!(
                "mnist plain blocks: vs {}: max rel batch-loss diff {max_rel:.3e}, batch acc \
                 equal {acc_equal}/{updates}, max rel lr diff {lr_rel:.2e}",
                args.compare_steps.as_ref().expect("given").display()
            );
            obj(vec![
                (
                    "file",
                    J::S(
                        args.compare_steps
                            .as_ref()
                            .expect("given")
                            .display()
                            .to_string(),
                    ),
                ),
                ("steps", J::I(updates as i64)),
                ("max_rel_batch_loss_diff", J::N(max_rel)),
                (
                    "min_rel_batch_loss_diff",
                    J::N(rel.iter().copied().fold(f64::INFINITY, f64::min)),
                ),
                ("batch_acc_equal", J::I(acc_equal as i64)),
                ("max_rel_lr_diff", J::N(lr_rel)),
                (
                    "note",
                    J::S(
                        "batch loss = sum over the batch and all outputs of (y - y_hat)^2 of \
                          the output stack, before the update; batch acc = argmax over all \
                          outputs (ReBoot's printed definitions)"
                            .into(),
                    ),
                ),
            ])
        }
    };
    let result = obj(vec![
        ("provenance", provenance(&load_start, started)),
        (
            "task",
            J::S("Task 52: ReBoot eMLP-2/3 block structure, plaintext twin".into()),
        ),
        (
            "config",
            obj(vec![
                ("stage_dims", dims_json(&dims)),
                (
                    "structure",
                    J::S(
                        "local blocks (linear, z^2 + z, local head; own loss, no gradient \
                          below) then the output stack (linear, z^2 + z, linear; main loss)"
                            .into(),
                    ),
                ),
                ("batch", J::I(args.batch as i64)),
                ("lr_on_batch_mean_gradient", J::N(args.lr)),
                ("lr_sum", args.lr_sum.map_or(J::N(f64::NAN), J::N)),
                ("momentum", J::N(args.momentum)),
                ("init", J::S(args.init.clone())),
                ("biases", J::B(!args.no_bias)),
                ("output_width", J::I(args.out_width as i64)),
                (
                    "init_weights",
                    J::S(
                        args.init_weights
                            .as_ref()
                            .map_or_else(|| "drawn (--seed)".into(), |p| p.display().to_string()),
                    ),
                ),
                (
                    "lr_schedule",
                    J::S(
                        args.lr_schedule
                            .as_ref()
                            .map_or_else(|| "constant".into(), |p| p.display().to_string()),
                    ),
                ),
                (
                    "indices",
                    J::S(args.indices.as_ref().map_or_else(
                        || "training set in order".into(),
                        |p| p.display().to_string(),
                    )),
                ),
                (
                    "dataset",
                    J::S(args.dataset_npy.as_ref().map_or_else(
                        || format!("IDX files in {}", args.data.display()),
                        |p| format!("npy arrays in {} (sha256-checked)", p.display()),
                    )),
                ),
                ("seed", J::I(args.seed as i64)),
                ("train_samples", J::I(n as i64)),
                ("epochs", J::I(args.epochs as i64)),
                ("updates", J::I(updates as i64)),
            ]),
        ),
        ("epochs", J::A(epochs)),
        ("test_acc", J::N(test_acc)),
        (
            "test_acc_all_outputs",
            J::N(accuracy_over(&out, &test.labels, out.ncols())),
        ),
        ("test_samples", J::I(test.len() as i64)),
        (
            "max_abs_handoff_activation_per_block",
            J::A(handoff_abs.iter().map(|&v| J::N(v)).collect()),
        ),
        ("comparison_with_reboot", comparison),
        (
            "step_batch_loss_sum",
            J::A(step_loss.iter().map(|&v| J::N(v)).collect()),
        ),
        (
            "step_batch_acc_all_outputs",
            J::A(step_acc_all.iter().map(|&v| J::N(v)).collect()),
        ),
        (
            "step_lr_on_batch_mean_gradient",
            J::A(lrs.iter().map(|&v| J::N(v)).collect()),
        ),
    ]);
    let mut text = String::new();
    result.write(&mut text, 0);
    text.push('\n');
    let path = args.out.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "results/mnist_blocks_{}_plain_b{}.json",
            args.blocks
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join("-"),
            args.batch
        ))
    });
    write_out(&path, &text)
}

// ---------------------------------------------------------------------------------------------
// Non-interactive.
// ---------------------------------------------------------------------------------------------

/// The run's prediction: per update the closed form with each stage's planned bootstraps.
struct Plan {
    /// Per stage, the a-priori bootstrap schedule.
    schedules: Vec<Vec<bool>>,
    /// Per update, the closed-form key switches (lr velocity-reference shift included).
    per_update_ks: Vec<u64>,
    /// The closed form of a steady update without weight bootstraps.
    nonboot_ks: u64,
    /// Per stage, its rotations + relins per update (no bootstraps).
    stage_nonboot_ks: Vec<u64>,
    /// Weight bootstraps per update, averaged.
    weight_boots_avg: f64,
    /// Hand-off bootstraps per update.
    handoff_boots: u64,
    ref_shift: u64,
}

#[allow(clippy::too_many_arguments)]
fn plan(
    dims: &[Vec<usize>],
    layout: &Layout,
    keys: &[TransposeKeys],
    placement: BootPlacement,
    micro: usize,
    full: &[bool],
    start: usize,
    out: usize,
    lrs: &[Option<f64>],
    lr_ref: f64,
    cfg: EvalConfig,
    boot_ks: u64,
) -> Result<Plan, String> {
    let e2s = |e: hekit_core::error::HekitError| e.to_string();
    let act_objs: [&dyn Activation; 2] = [&PolyRelu2, &Identity];
    let schedules = dims
        .iter()
        .map(|d| boot_schedule(d, &act_objs, layout, placement, true, start, out, full))
        .collect::<Result<Vec<_>, _>>()
        .map_err(e2s)?;
    let acts: Vec<Vec<ActCost>> = dims.iter().map(|_| lib_acts(&[true, false])).collect();
    let ref_shift: u64 = dims
        .iter()
        .map(|d| ref_shift_rotations(d, layout))
        .sum::<Result<u64, _>>()
        .map_err(e2s)?;
    let base = predict_block_update_with(
        dims,
        &acts,
        layout,
        keys,
        placement,
        micro,
        true,
        &vec![false; dims.len()],
        cfg,
    )
    .map_err(e2s)?;
    let mut per_update_ks = Vec::with_capacity(full.len());
    let mut wb = 0u64;
    for u in 0..full.len() {
        let booted: Vec<bool> = schedules.iter().map(|s| s[u]).collect();
        let p = predict_block_update_with(
            dims, &acts, layout, keys, placement, micro, true, &booted, cfg,
        )
        .map_err(e2s)?;
        wb += p.stages.iter().map(|c| c.weight_bootstraps).sum::<u64>();
        let shift = match lrs.get(u).copied().flatten() {
            Some(lr) if lr != lr_ref => ref_shift,
            _ => 0,
        };
        per_update_ks.push(p.key_switches(boot_ks) + shift);
    }
    Ok(Plan {
        stage_nonboot_ks: base
            .stages
            .iter()
            .map(|c| c.rotations() + c.relins())
            .collect(),
        nonboot_ks: base.rotations() + base.relins(),
        weight_boots_avg: wb as f64 / full.len().max(1) as f64,
        handoff_boots: base.handoff_bootstraps,
        schedules,
        per_update_ks,
        ref_shift,
    })
}

pub(super) fn run_ni_blocks(args: &Args) -> Result<(), String> {
    let started_unix = unix_now();
    let load_start = loadavg();
    let t_all = Instant::now();
    if args.be.backend != Backend::Cpu {
        return Err("--blocks runs on the CPU backend only".into());
    }
    let placement = placement_named(&args.placement)?;
    if placement == BootPlacement::AfterActivation {
        return Err(
            "--blocks: after_activation would bootstrap the block activation twice; use \
             when_needed or after_update"
                .into(),
        );
    }
    let loaded = load(args);
    let features = match (&loaded, args.plan) {
        (Ok((train, _)), _) => train.features(),
        (Err(e), true) => {
            eprintln!(
                "mnist: --plan: data not loaded ({e}); {} inputs (--input-dim)",
                args.input_dim
            );
            args.input_dim
        }
        (Err(e), false) => return Err(e.clone()),
    };
    let params = Arc::new(params_named(&args.params)?);
    let sdims = stage_dims(features, args);
    let dims: Vec<Vec<usize>> = sdims.iter().map(|d| d.to_vec()).collect();
    let d = sdims
        .iter()
        .flatten()
        .map(|x| x.next_power_of_two())
        .max()
        .expect("dims");
    let lanes = args.batch.next_power_of_two().min(params.slots() / d);
    let micro = args.batch.div_ceil(lanes);
    if micro > 1 && micro * lanes != args.batch {
        return Err(format!(
            "--batch {} is not a whole number of {lanes}-lane ciphertexts",
            args.batch
        ));
    }
    let layout = Layout::new(params.slots(), d, lanes).map_err(|e| e.to_string())?;
    let keys = dims
        .iter()
        .map(|st| match args.transpose.as_str() {
            "reuse" => mlp_key_pool(st, &layout)
                .map(TransposeKeys::Reuse)
                .map_err(|e| e.to_string()),
            "dedicated" => Ok(TransposeKeys::Dedicated),
            t => Err(format!("unknown --transpose {t} (reuse | dedicated)")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let steps = ni_block_rotation_steps(&dims, &layout, &keys).map_err(|e| e.to_string())?;
    let limbs = params.q().len() + params.p().len();
    let key_bytes = params.dnum() * 2 * limbs * params.n() * 8;
    let bp = BootParams {
        log_slots: params.slots().trailing_zeros(),
        ..BootParams::default_n16()
    };
    let out_level = hekit_ckks::bootstrap::levels_after_boot(&params, &bp);
    let start = out_level - 1;
    let act_objs: [&dyn Activation; 2] = [&PolyRelu2, &Identity];
    let lpi = levels_per_iter_of(&act_objs, placement);
    let b = args.batch as f64;
    eprintln!(
        "mnist: noninteractive blocks {:?} {} on {} (N = 2^{}), d = {d}, {lanes} lanes x {micro} \
         micro-batches: {} training Galois keys ({:.1} GB); levels per stage step {lpi} of \
         {start}",
        sdims,
        placement.name(),
        args.params,
        params.log_n(),
        steps.len(),
        ((steps.len() + 1) * key_bytes) as f64 / 1e9
    );

    // Per-update lr (in our mean units) and the velocity reference.
    let n_plan = match &loaded {
        Ok((tr, _)) => tr.len(),
        Err(_) => args.subset,
    };
    let lr_sched = args
        .lr_schedule
        .as_deref()
        .map(|p| read_lr_schedule(p, args.lr_key.as_deref()))
        .transpose()?;
    let orders = match &loaded {
        Ok((tr, _)) => args
            .epoch_orders
            .as_deref()
            .map(|p| read_epoch_orders(p, tr.len()))
            .transpose()?,
        Err(_) => None,
    };
    let lr_sched = lr_sched
        .map(|v| per_update_schedule(v, &orders, n_plan, args))
        .transpose()?;
    check_schedule_inputs(&lr_sched, &orders, n_plan, args)?;
    let lr_ref = lr_sched
        .as_ref()
        .map_or(args.lr, |v| v.iter().copied().fold(0.0, f64::max) * b);
    let full = run_fullness(args, lanes, n_plan);
    let lrs: Vec<Option<f64>> = (0..full.len())
        .map(|u| {
            lr_sched
                .as_ref()
                .map(|_| step_lr(&lr_sched, u, args))
                .transpose()
        })
        .collect::<Result<_, _>>()?;
    // n16_boot's measured bootstrap count (the plan has no bootstrapper).
    let plan_boot_ks = HEKIT_BOOT_KS;
    let pl = plan(
        &dims,
        &layout,
        &keys,
        placement,
        micro,
        &full,
        start,
        out_level,
        &lrs,
        lr_ref,
        args.be.config,
        plan_boot_ks,
    )?;
    let samples_planned = (n_plan * args.epochs) as f64;
    let pred_per_sample = pl.per_update_ks.iter().sum::<u64>() as f64 / samples_planned;
    let plan_json = obj(vec![
        ("stage_dims", dims_json(&sdims)),
        ("layout_d", J::I(d as i64)),
        ("lanes", J::I(lanes as i64)),
        ("micro_batches_per_update", J::I(micro as i64)),
        ("placement", J::S(placement.name().into())),
        ("levels_per_stage_step", J::I(lpi as i64)),
        ("start_level", J::I(start as i64)),
        ("bootstrap_output_level", J::I(out_level as i64)),
        (
            "stage_nonboot_ks_per_update",
            J::A(
                pl.stage_nonboot_ks
                    .iter()
                    .map(|&v| J::I(v as i64))
                    .collect(),
            ),
        ),
        ("nonboot_ks_per_update", J::I(pl.nonboot_ks as i64)),
        (
            "weight_bootstraps_per_update_avg",
            J::N(pl.weight_boots_avg),
        ),
        (
            "handoff_bootstraps_per_update",
            J::I(pl.handoff_boots as i64),
        ),
        (
            "velocity_reference_shift_rotations",
            J::I(pl.ref_shift as i64),
        ),
        ("boot_ks_assumed", J::I(plan_boot_ks as i64)),
        ("updates", J::I(full.len() as i64)),
        ("predicted_ks_per_sample", J::N(pred_per_sample)),
        (
            "boot_schedules",
            J::A(
                pl.schedules
                    .iter()
                    .map(|s| J::A(s.iter().map(|&x| J::I(i64::from(x))).collect()))
                    .collect(),
            ),
        ),
        ("galois_keys", J::I(steps.len() as i64)),
        (
            "training_key_bytes",
            J::I(((steps.len() + 1) * key_bytes) as i64),
        ),
    ]);
    if args.plan {
        let mut s = String::new();
        plan_json.write(&mut s, 0);
        println!("{s}");
        return Ok(());
    }
    let (train, test) = loaded?;
    let n = train.len();

    // Keys (client side; the server gets evaluation keys only).
    let t_keys = Instant::now();
    let mut kg =
        KeyGenerator::new(Arc::clone(&params), rng(args.seed, 1)).map_err(|e| e.to_string())?;
    let encoder = Encoder::new(Arc::clone(&params));
    let bs =
        Bootstrapper::new(Arc::clone(&params), bp, &encoder, &mut kg).map_err(|e| e.to_string())?;
    let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
    let sk = kg.secret_key();
    let ev = Evaluator::with_config(
        Arc::clone(&params),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&steps)),
        args.be.config,
    );
    drop(kg);
    let keygen_s = t_keys.elapsed().as_secs_f64();
    let per_boot = bs.cost_per_boot();
    let boot_ks = bs.key_switches_per_boot();
    let e2s = |e: hekit_core::error::HekitError| e.to_string();

    // Network (client side) and trainers.
    let mut r = rng(args.seed, 2);
    let init = initial_stages(&sdims, &mut r, args)?;
    let plain: Vec<Vec<(Array2<f64>, Vec<f64>)>> = init
        .iter()
        .map(|st| st.iter().map(|l| (l.w.clone(), l.b.clone())).collect())
        .collect();
    let acts: Vec<Vec<Box<dyn Activation>>> = dims
        .iter()
        .map(|_| {
            vec![
                Box::new(PolyRelu2) as Box<dyn Activation>,
                Box::new(Identity),
            ]
        })
        .collect();
    let (mut net, states) = BlockNiTrainer::encrypt_network(
        &ev, &bs, &plain, acts, &layout, placement, &enc, &encoder, &mut r,
    )
    .map_err(e2s)?;
    let optim = NiOptimizer::Nag {
        lr: lr_ref,
        momentum: args.momentum,
    };
    let stage_trainers = states
        .into_iter()
        .enumerate()
        .map(|(s, st)| {
            let t =
                NonInteractiveTrainer::new(&ev, &bs, optim, layout, placement, net.stage(s), st)?
                    .with_transpose_keys(keys[s].clone())
                    .with_micro_batches(micro);
            Ok(if args.no_bias {
                t.with_frozen_biases()
            } else {
                t
            })
        })
        .collect::<hekit_core::error::Result<Vec<_>>>()
        .map_err(e2s)?;
    let mut trainer = BlockNiTrainer::new(&ev, &bs, stage_trainers).map_err(e2s)?;
    let specs = (0..net.num_stages())
        .map(|s| trainer.stages()[s].batch_spec(net.stage(s)))
        .collect::<hekit_core::error::Result<Vec<_>>>()
        .map_err(e2s)?;
    // The run's own plan with the real bootstrap count.
    let lrs_run: Vec<Option<f64>> = lrs.clone();
    let pl = plan(
        &dims,
        &layout,
        &keys,
        placement,
        micro,
        &full,
        start,
        out_level,
        &lrs_run,
        lr_ref,
        args.be.config,
        boot_ks,
    )?;
    let pred_per_sample = pl.per_update_ks.iter().sum::<u64>() as f64 / (n * args.epochs) as f64;
    eprintln!(
        "mnist: keys {keygen_s:.1} s, {boot_ks} KS per bootstrap; predicted {pred_per_sample:.3} \
         KS/sample ({} non-boot KS per update, {:.2} weight + {} hand-off bootstraps)",
        pl.nonboot_ks, pl.weight_boots_avg, pl.handoff_boots
    );

    let mut twin = init.clone();
    let mut vels = zero_vels(&twin);
    let y_train = one_hot(&train.labels, args.out_width);
    let stages_n = dims.len();
    let acts_cost: Vec<Vec<ActCost>> = dims.iter().map(|_| lib_acts(&[true, false])).collect();
    let mut matches = true;
    let mut total = OpCounter::default();
    let mut update_s = Vec::new();
    let mut per_update = Vec::new();
    let mut handoff_abs = vec![0.0f64; args.blocks.len()];
    let mut samples = 0usize;
    let mut u = 0usize;
    let t_train = Instant::now();
    for epoch in 0..args.epochs {
        let order = epoch_order(&orders, epoch, n)?;
        for st0 in (0..order.len()).step_by(args.batch) {
            let end = (st0 + args.batch).min(order.len());
            trainer.set_next_batch_full(full.get(u + 1).copied());
            let lr = step_lr(&lr_sched, u, args)?;
            if lr_sched.is_some() {
                trainer.set_lr(lr).map_err(e2s)?;
            }
            // Client: one x per micro-batch (stage 0's spec), its labels at every stage's spec.
            let mut cts = Vec::new();
            for sub in (st0..end).step_by(lanes) {
                let se = (sub + lanes).min(end);
                let x = train.images.select(Axis(0), &order[sub..se]);
                let y = y_train.select(Axis(0), &order[sub..se]);
                let (xc, _) = specs[0]
                    .encrypt(&enc, &encoder, &layout, &x, &y, &mut r)
                    .map_err(e2s)?;
                let ys = specs
                    .iter()
                    .map(|sp| {
                        sp.encrypt(&enc, &encoder, &layout, &x, &y, &mut r)
                            .map(|c| c.1)
                    })
                    .collect::<hekit_core::error::Result<Vec<_>>>()
                    .map_err(e2s)?;
                cts.push((xc, ys, se - sub));
            }
            let yrefs: Vec<Vec<&Ciphertext>> = cts.iter().map(|c| c.1.iter().collect()).collect();
            let mbs: Vec<BlockMicroBatch<'_, Ciphertext>> = cts
                .iter()
                .zip(&yrefs)
                .map(|(c, y)| (&c.0, &y[..], c.2))
                .collect();
            let before = ev.counter();
            let t1 = Instant::now();
            net.train_step_ni(&mut trainer, &mbs)
                .map_err(|e| format!("update {u}: {e}"))?;
            let secs = t1.elapsed().as_secs_f64();
            let c = &ev.counter() - &before;
            // Twin.
            let x = train.images.select(Axis(0), &order[st0..end]);
            let y = y_train.select(Axis(0), &order[st0..end]);
            for (m, v) in handoff_abs.iter_mut().zip(handoff_max(&twin, &x)) {
                *m = m.max(v);
            }
            let ys = vec![&y; stages_n];
            let blr = if args.no_bias { 0.0 } else { lr };
            block_twin_step(
                &mut twin,
                &mut vels,
                &x,
                &ys,
                lr,
                args.momentum,
                blr,
                WeightDecay::NONE,
            );
            // The closed form with the planned schedule, against the counter.
            let booted: Vec<bool> = trainer
                .stages()
                .iter()
                .map(|t| t.last_update_booted())
                .collect();
            let planned: Vec<bool> = pl.schedules.iter().map(|s| s[u]).collect();
            let p = predict_block_update_with(
                &dims,
                &acts_cost,
                &layout,
                &keys,
                placement,
                mbs.len(),
                true,
                &planned,
                args.be.config,
            )
            .map_err(e2s)?;
            let rescaled = lr != lr_ref;
            let shift = if rescaled { pl.ref_shift } else { 0 };
            let nb = p.bootstraps();
            let ok = booted == planned
                && trainer
                    .stages()
                    .iter()
                    .all(|t| t.last_update_rescaled() == rescaled)
                && c.bootstraps == nb
                && c.rotations == p.rotations() + shift + nb * per_boot.rotations
                && c.relins == p.relins() + nb * per_boot.relins
                && c.key_switches() == p.key_switches(boot_ks) + shift;
            matches &= ok;
            eprintln!(
                "mnist: epoch {epoch} update {u} ({} samples): {secs:.1} s, {} KS ({} bootstraps, \
                 stages booted {booted:?}), matches prediction {ok}; elapsed {:.0} s, load {}",
                end - st0,
                c.key_switches(),
                c.bootstraps,
                t_all.elapsed().as_secs_f64(),
                loadavg()
            );
            total += &c;
            per_update.push(c);
            update_s.push(secs);
            samples += end - st0;
            u += 1;
        }
    }
    let train_s = t_train.elapsed().as_secs_f64();

    // Results (the client decrypts only now).
    let dec = Decryptor::new(Arc::clone(&params), sk);
    let got: Vec<Vec<Layer>> = (0..net.num_stages())
        .map(|s| {
            net.stage(s)
                .layers
                .iter()
                .map(|(dn, _)| Layer {
                    w: dn.w.decrypt(&dec, &encoder, &layout),
                    b: unpack_activations(
                        &encoder.decode(&dec.decrypt(&dn.b)),
                        &layout,
                        dn.w.d_out(),
                    )
                    .row(0)
                    .to_vec(),
                })
                .collect()
        })
        .collect();
    let max_diff = got
        .iter()
        .flatten()
        .zip(twin.iter().flatten())
        .map(|(g, t)| {
            max_abs(g.w.iter().zip(&t.w).map(|(a, b)| a - b))
                .max(max_abs(g.b.iter().zip(&t.b).map(|(a, b)| a - b)))
        })
        .fold(0.0, f64::max);
    let out_he = block_forward_plain(&got, &test.images);
    let out_twin = block_forward_plain(&twin, &test.images);
    let ks = total.key_switches();
    let pred_run: u64 = pl.per_update_ks[..u].iter().sum();
    let sf = samples as f64;
    eprintln!(
        "mnist: {u} updates, {samples} samples in {train_s:.1} s; {ks} KS = {:.3}/sample \
         (predicted {:.3}); matches {matches}; weights 2^{:.1} from the twin",
        ks as f64 / sf,
        pred_run as f64 / sf,
        max_diff.log2()
    );
    let result = obj(vec![
        ("provenance", provenance(&load_start, started_unix)),
        (
            "task",
            J::S(
                "Task 52: ReBoot eMLP-2/3 block structure, non-interactive (bootstrapping)".into(),
            ),
        ),
        ("plan", plan_json),
        (
            "setup",
            obj(vec![
                ("params", J::S(args.params.clone())),
                ("log_n", J::I(params.log_n() as i64)),
                ("batch", J::I(args.batch as i64)),
                ("transpose_keys", J::S(args.transpose.clone())),
                ("boot_ks", J::I(boot_ks as i64)),
                ("eval_config", J::R(config_json(&args.be.config))),
                ("lr_reference_on_mean_gradient", J::N(lr_ref)),
                (
                    "lr_schedule",
                    J::S(
                        args.lr_schedule
                            .as_ref()
                            .map_or_else(|| "constant".into(), |p| p.display().to_string()),
                    ),
                ),
                (
                    "init_weights",
                    J::S(
                        args.init_weights
                            .as_ref()
                            .map_or_else(|| "drawn (--seed)".into(), |p| p.display().to_string()),
                    ),
                ),
                ("biases", J::B(!args.no_bias)),
                ("seed", J::I(args.seed as i64)),
            ]),
        ),
        (
            "measured",
            obj(vec![
                ("updates", J::I(u as i64)),
                ("samples", J::I(samples as i64)),
                ("total", ni_counter_json(&total)),
                ("ks_per_sample_incl_bootstraps", J::N(ks as f64 / sf)),
                ("predicted_ks_this_run", J::I(pred_run as i64)),
                ("measured_over_predicted", J::N(ks as f64 / pred_run as f64)),
                ("every_update_equals_prediction", J::B(matches)),
                (
                    "handoff_bootstraps",
                    J::I(trainer.handoff_bootstraps() as i64),
                ),
                ("per_update_first", ni_counter_json(&per_update[0])),
            ]),
        ),
        (
            "time_s",
            obj(vec![
                ("keygen_total", J::N(keygen_s)),
                ("train_total", J::N(train_s)),
                ("per_update", stats(&update_s)),
                ("bootstrap_total", J::N(trainer.bootstrap_seconds())),
                ("wall_total", J::N(t_all.elapsed().as_secs_f64())),
            ]),
        ),
        (
            "twin",
            obj(vec![
                ("max_abs_diff", J::N(max_diff)),
                ("log2_max_abs_diff", J::N(max_diff.log2())),
                (
                    "max_abs_handoff_activation_per_block",
                    J::A(handoff_abs.iter().map(|&v| J::N(v)).collect()),
                ),
                (
                    "handoff_note",
                    J::S(
                        "the hand-off bootstraps the block activation; above 1 it violates the \
                          bootstrap's |m| <= 1 precondition (not clipped)"
                            .into(),
                    ),
                ),
            ]),
        ),
        (
            "evaluation",
            obj(vec![
                ("test_samples", J::I(test.len() as i64)),
                (
                    "test_acc_encrypted_trained",
                    J::N(accuracy(&out_he, &test.labels)),
                ),
                ("test_acc_twin", J::N(accuracy(&out_twin, &test.labels))),
            ]),
        ),
        (
            "memory",
            obj(vec![(
                "peak_rss_bytes",
                peak_rss_bytes().map_or(J::N(f64::NAN), |b| J::I(b as i64)),
            )]),
        ),
    ]);
    let mut text = String::new();
    result.write(&mut text, 0);
    text.push('\n');
    let path = args.out.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "results/mnist_blocks_{}_ni_{}_b{}.json",
            args.blocks
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join("-"),
            placement.name(),
            args.batch
        ))
    });
    write_out(&path, &text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `configs/reboot_emlp{2,3}.json` (R-REB-5) and the flags they list give the same network
    /// and hyperparameters.
    #[test]
    fn block_configs_match_the_flags() {
        for (name, widths) in [("reboot_emlp2", vec![64]), ("reboot_emlp3", vec![128, 64])] {
            let path = format!("{}/configs/{name}.json", env!("CARGO_MANIFEST_DIR"));
            let c: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let flags = c["mnist_flags"].as_str().unwrap();
            let blocks_flag = flags
                .split_whitespace()
                .skip_while(|f| *f != "--blocks")
                .nth(1)
                .unwrap();
            let hidden_flag = flags
                .split_whitespace()
                .skip_while(|f| *f != "--hidden")
                .nth(1)
                .unwrap();
            let a = parse_args_from(
                &[
                    "--mode",
                    "plain",
                    "--faithful",
                    "--blocks",
                    blocks_flag,
                    "--hidden",
                    hidden_flag,
                ]
                .map(String::from),
            )
            .unwrap();
            assert_eq!(a.blocks, widths, "{name}");
            let input = c["input"].as_u64().unwrap() as usize;
            let dims = stage_dims(input, &a);
            let blocks = c["blocks"].as_array().unwrap();
            assert_eq!(blocks.len(), widths.len());
            for (b, d) in blocks.iter().zip(&dims) {
                assert_eq!(b["in"].as_u64().unwrap() as usize, d[0]);
                assert_eq!(b["width"].as_u64().unwrap() as usize, d[1]);
                assert_eq!(b["head"].as_u64().unwrap() as usize, d[2]);
                assert_eq!(b["bias"], serde_json::json!(false));
            }
            let out = c["output_stack"].as_array().unwrap();
            let last = dims.last().unwrap();
            assert_eq!(out[0]["in"].as_u64().unwrap() as usize, last[0]);
            assert_eq!(out[0]["out"].as_u64().unwrap() as usize, last[1]);
            assert_eq!(out[1]["out"].as_u64().unwrap() as usize, last[2]);
            assert_eq!(c["output_width"].as_u64().unwrap() as usize, a.out_width);
            assert_eq!(c["lr_sum"].as_f64().unwrap(), FAITHFUL_LR_SUM);
            assert_eq!(c["momentum"].as_f64().unwrap(), a.momentum);
            assert_eq!(c["weight_decay"].as_f64().unwrap(), 0.0);
            assert!(a.no_bias);
            // ReBoot's layer names in its trainable order.
            let names: Vec<String> = blocks
                .iter()
                .flat_map(|b| b["reboot_layers"].as_array().unwrap().clone())
                .chain(out.iter().map(|l| l["reboot_layer"].clone()))
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            assert_eq!(names, reboot_layer_names(widths.len()));
        }
    }

    #[test]
    fn blocks_flags_are_checked() {
        let p = |v: &[&str]| parse_args_from(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert!(p(&["--blocks", "64", "--mode", "interactive"]).is_err());
        assert!(p(&["--blocks", "0", "--mode", "plain"]).is_err());
        assert!(p(&["--compare-steps", "x.tsv", "--mode", "plain"]).is_err());
        assert_eq!(
            p(&["--blocks", "128,64", "--mode", "noninteractive"])
                .unwrap()
                .blocks,
            vec![128, 64]
        );
    }
}
