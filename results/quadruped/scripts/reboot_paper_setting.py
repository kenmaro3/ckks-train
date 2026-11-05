"""R-REB-3: ReBoot eMLP-1 at the paper's Table 4 setting, plaintext only, with ReBoot's released plaintext path.
Lives at ~/reboot/reboot_paper_setting.py on quadruped-gpu. Contains no ReBoot code: it imports ReBoot modules, reads
precision_analysis.py's configuration via runpy (its loop does not run), overrides lr / weight decay / epochs, and uses
ReBoot's own data loader, shuffles, model constructor, plaintext train step and lr scheduler.

Setting (requests.md R-REB-3): all 60,000 training images resized to 14x14 by ReBoot's loader, min-max scaled over the
resized 60k (ReBoot's pipeline), full 10k test; lr 0.001, weight decay 0.001 (both of ReBoot's decay knobs, as the
script maps its single --weight-decay to them), batch 48, Nesterov momentum 0.9 (ReBoot's plaintext Nesterov, as its
twin uses), seeds 0-9, 50 epochs, two lr schedules: const (0.001 throughout) and sched (ReBoot's scheduler as
precision_analysis.py builds it, base 0.001, constant for 5 epochs = 5 * steps_per_epoch steps, then cosine restarts
every 50 steps, stepped once per batch). Test accuracy (argmax over the 10 classes) after every epoch; no early stop.

RNG sequence as precision_analysis.py: seed S before loading (preprocessing shuffle uses seed 42), seed S + plain model
(the twin, trained here), seed S + encrypted model's draws (same shapes), then one global-RNG shuffle per epoch. No other
numpy draws happen in the training path, so the per-epoch orders are the same for both schedules.

Usage: python reboot_paper_setting.py RAW_OUT DATA_OUT WORKERS
"""
import hashlib
import json
import multiprocessing as mp
import os
import runpy
import sys
import time

import numpy as np

SRC = os.environ["REBOOT_SRC"]
sys.path.insert(0, SRC)
EXP = os.path.join(SRC, "experiments", "3_training_encrypted")
raw_out, data_out, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(raw_out, exist_ok=True)
os.makedirs(data_out, exist_ok=True)

LR, WD, B = 0.001, 0.001, 48
EPOCHS = int(os.environ.get("RREB3_EPOCHS", "50"))                      # smoke-test overrides only
SEEDS = [int(x) for x in os.environ.get("RREB3_SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]

import lib.utils.data as data_mod  # noqa: E402
from lib.models.local_loss_models import LocalLossMLP  # noqa: E402
from lib.optim.schedulers import CosineLR  # noqa: E402
from lib.utils.data import load_float_dataset, shuffle_dataset  # noqa: E402
from lib.utils.enums import NonLinearity, OptimizerName  # noqa: E402
from lib.utils.train import train_step_plain  # noqa: E402

SCALING = {}
_orig_pre = data_mod._preprocess_float_dataset


def _capture(train_data, *a, **k):
    SCALING.update({"train_min_before_scaling": float(train_data.min()), "train_max_before_scaling": float(train_data.max()),
                    "shape_before_scaling": list(train_data.shape)})
    return _orig_pre(train_data, *a, **k)


data_mod._preprocess_float_dataset = _capture


def configs_for(seed):
    argv = ["MNIST", "--num-layers", "2", "--num-hidden", "32", "--batch-size", str(B), "--num-epochs", str(EPOCHS),
            "--seed", str(seed)]
    sys.argv = [os.path.join(EXP, "precision_analysis.py")] + argv
    cwd = os.getcwd(); os.chdir(EXP)
    c = dict(runpy.run_path(sys.argv[0], run_name="reboot_config_only")["configs"])
    os.chdir(cwd)
    c.update(subsample=None, lr=LR, fwd_decay=WD, lrn_decay=WD, num_epochs=EPOCHS)
    return argv, c


def make_model(c, opt, n_in, n_out):
    return LocalLossMLP(
        cc=None, num_fc_layers=c["num_fc_layers"], num_fc_hidden=c["num_fc_hidden"], num_classes=n_out, input_dim=n_in,
        dropout_rate=c["dropout_rate"], non_linearity=NonLinearity[c["non_linearity"]], fwd_decay_inv=c["fwd_decay"],
        subnet_decay_inv=c["lrn_decay"], bias=c["bias"], debug=False, optimizer=OptimizerName[opt],
        momentum=c["momentum"], instantiate_last_block=c["instantiate_last_block"], subnet_trainable=c["subnet_trainable"],
        output_non_linearity=c["output_non_linearity"], encrypted=False)


# ---- data (identical for all seeds: preprocessing uses its own fixed seed); load once, forked workers share it
argv0, c0 = configs_for(0)
np.random.seed(0)
X, _, Xt, Y, _, Yt = load_float_dataset(c0, show_images=False, show_log=False)
K = X.shape[0]
X = X.reshape(K, -1); Xt = Xt.reshape(Xt.shape[0], -1)
idx = np.arange(K).reshape(-1, 1)
perm1, _ = shuffle_dataset(idx, idx, seed=42); perm1 = perm1.reshape(-1)
NB = K // B


def orders_and_init(seed):
    """Per-epoch training orders (as original MNIST indices) and initial weights, following the script's RNG use."""
    _, c = configs_for(seed)
    # (the script's data load ends with ReBoot's seed-42 shuffle, and the script re-seeds with S before building
    # models, so the load does not affect the orders and is not repeated here)
    np.random.seed(c["seed"]); twin = make_model(c, "PLAIN_NESTEROV_SGD", X.shape[1], Y.shape[1])
    np.random.seed(c["seed"]); make_model(c, c["optimizer"], X.shape[1], Y.shape[1])
    pos = np.arange(K); orders = np.empty((EPOCHS, K), dtype=np.uint16)
    for e in range(EPOCHS):
        p, _ = shuffle_dataset(np.arange(K).reshape(-1, 1), np.arange(K).reshape(-1, 1))
        pos = pos[p.reshape(-1)]
        orders[e] = perm1[pos]
    init = [np.asarray(l.weights).copy() for l in twin.get_trainable_layers()]
    return c, orders, init


def run(task):
    seed, sched_name = task
    os.environ["OMP_NUM_THREADS"] = "1"
    c, orders, init = orders_and_init(seed)
    np.random.seed(c["seed"])
    model = make_model(c, "PLAIN_NESTEROV_SGD", X.shape[1], Y.shape[1])
    for l, w in zip(model.get_trainable_layers(), init):
        assert np.array_equal(np.asarray(l.weights), w)
    sched = CosineLR(base_lr=c["lr"], steps_before_restart=50, from_step=5 * NB)
    rowof = np.empty(K, dtype=np.int64); rowof[perm1] = np.arange(K)   # original index -> row of X
    yt = np.argmax(Yt[:, :10], axis=1)
    rows_out, t0 = [], time.time()
    for e in range(EPOCHS):
        model.train()
        ls, accs = [], []
        for b in range(NB):
            if sched_name == "sched":
                sched.step(); lr = float(sched.get_lr())
            else:
                lr = LR
            r = rowof[orders[e, b * B:(b + 1) * B].astype(np.int64)]
            loss, acc, _, _ = train_step_plain(model=model, X_train=X[r], y_train=Y[r], lr=lr)
            ls.append(float(loss)); accs.append(float(acc))
        model.eval()
        pred = np.asarray(model.forward(Xt))
        test_acc = float(np.mean(np.argmax(pred[:, :10], axis=1) == yt))
        finite = bool(np.all(np.isfinite(pred)))
        rows_out.append((e + 1, test_acc, float(np.mean(ls)), float(np.mean(accs)), lr, finite))
    with open(os.path.join(raw_out, f"acc_curves_seed{seed}_{sched_name}.tsv"), "w") as f:
        f.write("# ReBoot plaintext path, paper setting (lr 0.001, wd 0.001, batch 48, 60k at 14x14), schedule "
                f"{sched_name}, seed {seed}; test_acc = argmax over 10 classes on the 10k test set after each epoch\n")
        f.write("epoch\ttest_acc\ttrain_batch_loss_mean\ttrain_batch_acc_mean\tlr_last_step\toutputs_finite\n")
        for r in rows_out:
            f.write("%d\t%.4f\t%.6g\t%.6g\t%.10g\t%s\n" % r)
    return seed, sched_name, [r[1] for r in rows_out], time.time() - t0


if __name__ == "__main__":
    T0 = time.time()
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    manifest = {"setting": "R-REB-3 paper setting (60k train at 14x14, 10k test)", "files": []}
    yo = np.empty(K, dtype=np.uint8); Xo = np.empty_like(X)
    Xo[perm1] = X; yo[perm1] = np.argmax(Y[:, :10], axis=1).astype(np.uint8)
    for name, arr in [("train_images_14x14.npy", Xo), ("train_labels.npy", yo),
                      ("test_images_14x14.npy", Xt), ("test_labels.npy", np.argmax(Yt[:, :10], axis=1).astype(np.uint8))]:
        p = os.path.join(data_out, name); np.save(p, arr)
        manifest["files"].append({"file": name, "shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": sha(p)})
    for seed in SEEDS:
        c, orders, init = orders_and_init(seed)
        p = os.path.join(data_out, f"epoch_orders_seed{seed}.npy"); np.save(p, orders)
        manifest["files"].append({"file": f"epoch_orders_seed{seed}.npy", "shape": list(orders.shape), "dtype": "uint16",
                                  "meaning": f"row e = original MNIST training indices in training order for epoch e+1; step b of that epoch uses columns b*{B}..b*{B}+{B - 1}",
                                  "sha256": sha(p)})
        for i, w in enumerate(init, 1):
            p = os.path.join(data_out, f"init_weights_seed{seed}_W{i}.npy"); np.save(p, w.astype(np.float64))
            manifest["files"].append({"file": f"init_weights_seed{seed}_W{i}.npy", "shape": list(w.shape),
                                      "orientation": "in x out", "dtype": "float64 (exact upcast of float32)", "sha256": sha(p)})
    manifest["pipeline"] = ["MNIST uint8 -> float32 0..255, no padding", f"all {K} training images, all {Xt.shape[0]} test images",
                            "bicubic resize to 14x14 by ReBoot's loader", "min-max scaling AFTER the resize over the resized training set (constants in `scaling`); test uses the training min/max",
                            "no mean/std normalization", "row-major flatten to 196 features"]
    manifest["scaling"] = dict(SCALING)
    manifest["row_order"] = "train_images row i = MNIST training image i"
    json.dump(manifest, open(os.path.join(data_out, "manifest.json"), "w"), indent=2)

    tasks = [(s, n) for s in SEEDS for n in ("const", "sched")]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(run, tasks, chunksize=1)
    curves = {(s, n): a for s, n, a, _ in res}
    secs = {f"{s}_{n}": t for s, n, _, t in res}

    def ms(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "std_sample": float(v.std(ddof=1)) if v.size > 1 else None,
                "min": float(v.min()), "max": float(v.max())}
    summary = {"seeds": SEEDS, "epochs": EPOCHS, "per_schedule": {}}
    for n in ("const", "sched"):
        best = [max(curves[(s, n)]) for s in SEEDS]
        best_ep = [int(np.argmax(curves[(s, n)])) + 1 for s in SEEDS]
        summary["per_schedule"][n] = {
            "best_epoch_test_acc": ms(best), "best_epoch_per_seed": best_ep,
            "at_epoch": {str(e): ms([curves[(s, n)][e - 1] for s in SEEDS]) for e in (10, 20, 50) if e <= EPOCHS},
            "mean_curve": [float(np.mean([curves[(s, n)][e] for s in SEEDS])) for e in range(EPOCHS)],
        }
    summary["paper_table4"] = "94.61 +/- 0.27 (as transcribed in requests.md from arXiv:2506.19693 Table 4; not read by this script)"
    summary["worker_seconds"] = secs
    summary["wall_s"] = time.time() - T0
    json.dump(summary, open(os.path.join(raw_out, "summary.json"), "w"), indent=2)
    json.dump({
        "base_command_of_config": "precision_analysis.py " + " ".join(argv0) + " (configuration only; overrides below)",
        "overrides": {"subsample": None, "lr": LR, "fwd_decay": WD, "lrn_decay": WD, "num_epochs": EPOCHS},
        "batch_size": B, "momentum": c0["momentum"], "optimizer_trained": "PLAIN_NESTEROV_SGD (ReBoot's plaintext Nesterov, as its twin)",
        "input_resize": c0["resize"], "train_samples": K, "test_samples": int(Xt.shape[0]), "steps_per_epoch": NB,
        "samples_not_trained_per_epoch": K - NB * B, "seeds": SEEDS,
        "schedules": {"const": "lr 0.001 every step",
                      "sched": f"ReBoot's scheduler as precision_analysis.py builds it: base 0.001, constant for 5*{NB} = {5 * NB} steps, then cosine annealing with warm restarts every 50 steps down to 1e-7, stepped once per batch"},
        "weight_decay_note": "applied by ReBoot's optimizer as gradient + wd * W on both the forward and the local-classifier layers (fwd_decay = lrn_decay = 0.001, as training_encrypted.py maps --weight-decay; ReBoot's model argument is named *_decay_inv but is passed through unchanged as the optimizer's weight decay)",
        "eval": "test accuracy = argmax over the first 10 of 16 outputs vs the label, after every epoch, model in eval mode",
        "scaling": dict(SCALING), "data_dir": "results/quadruped/data/reboot_emlp1_paper/ (manifest.json)",
        "workers": workers,
    }, open(os.path.join(raw_out, "config_values.json"), "w"), indent=2)
    print("R-REB-3 done", json.dumps({n: summary["per_schedule"][n]["best_epoch_test_acc"] for n in ("const", "sched")}))
