"""R-REB-2: ReBoot eMLP-1 full training setting as data (no training). Lives at ~/reboot/reboot_full_setting.py on
quadruped-gpu. Contains no ReBoot code: it imports ReBoot modules, reads the configuration of
experiments/3_training_encrypted/precision_analysis.py via runpy (its loop is under `if __name__ == "__main__"`, so it
does not run), and calls ReBoot's own data loader, shuffles, model constructor and lr scheduler.

The full setting is the MLP-1 line of upstream experiments/3_training_encrypted/experiments.sh (precision_analysis.py
MNIST --subsample 4800 --num-layers 2 --num-hidden 32 --batch-size 48 --num-epochs 10 --seed S, S in 0, 1, 2).

Per-step indices replay the RNG sequence that script follows: preprocessing shuffles the first K images with seed 42;
then np.random is seeded with S, the plain model and (after re-seeding) the encrypted model are built (same draws as
two plain constructions with the same shapes), and each epoch shuffles once with the global RNG; no other numpy random
draws happen in the training path (only Linear init and the data shuffles draw). Same method as the subset run, where a
plaintext replay matched all 21 encrypted steps. lr per step comes from ReBoot's own scheduler object, stepped once per
batch as that script does.

Usage: python reboot_full_setting.py OUT_DIR
"""
import glob
import hashlib
import json
import os
import re
import runpy
import sys

import numpy as np

SRC = os.environ["REBOOT_SRC"]
sys.path.insert(0, SRC)
EXP = os.path.join(SRC, "experiments", "3_training_encrypted")
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

sh = open(os.path.join(EXP, "experiments.sh")).read().replace("\r", "")
mlp1 = [l.strip() for l in sh.split("# MLP-2")[0].splitlines() if l.strip().startswith("python")]
argv0 = mlp1[0].split()[2:]          # drop "python3 precision_analysis.py"
listed_seeds = [int(re.search(r"--seed (\d+)", l).group(1)) for l in mlp1]
# The paper's Table 4 averages 10 runs; experiments.sh lists seeds 0-2. Seeds 3-9 use the same mechanism (--seed S).
seeds = listed_seeds + [x for x in range(10) if x not in listed_seeds]

from lib.models.local_loss_models import LocalLossMLP  # noqa: E402
from lib.optim.schedulers import CosineLR  # noqa: E402
import lib.utils.data as data_mod  # noqa: E402
from lib.utils.data import load_float_dataset, shuffle_dataset  # noqa: E402

SCALING = {}
_orig_pre = data_mod._preprocess_float_dataset


def _capture(train_data, *a, **k):
    # record the training-set min/max that ReBoot's min-max scaling uses (computed after the resize)
    SCALING.update({"train_min_before_scaling": float(train_data.min()), "train_max_before_scaling": float(train_data.max()),
                    "dtype_before_scaling": str(train_data.dtype), "shape_before_scaling": list(train_data.shape)})
    return _orig_pre(train_data, *a, **k)


data_mod._preprocess_float_dataset = _capture
from lib.utils.enums import NonLinearity, OptimizerName  # noqa: E402
from lib.utils.train import train_step_plain  # noqa: E402

result = {"source_command_lines": mlp1, "script": "experiments/3_training_encrypted/precision_analysis.py",
          "seeds": seeds, "seeds_listed_in_experiments_sh": listed_seeds,
          "seeds_note": "seeds not listed in experiments.sh are produced with the same command and --seed S; ReBoot seeds only through np.random.seed(S), so they are as faithful as the listed ones",
          "per_seed": {}}
for seed in seeds:
    argv = [a if not (i > 0 and argv0[i - 1] == "--seed") else str(seed) for i, a in enumerate(argv0)]
    sys.argv = [os.path.join(EXP, "precision_analysis.py")] + argv
    os.chdir(EXP)
    c = runpy.run_path(sys.argv[0], run_name="reboot_config_only")["configs"]
    K, B, E = c["subsample"], c["batch_size"], c["num_epochs"]

    np.random.seed(c["seed"])
    X_train, _, X_test, y_train, _, y_test = load_float_dataset(c, show_images=False, show_log=False)
    X_train = X_train.reshape(K, -1)
    idx = np.arange(K).reshape(-1, 1)
    perm1, _ = shuffle_dataset(idx, idx, seed=42)
    perm1 = perm1.reshape(-1)
    # Model inputs exactly as ReBoot feeds them, re-ordered by original MNIST index (row i = training image i).
    Xo = np.empty_like(X_train); Xo[perm1] = X_train
    yo = np.empty(K, dtype=np.uint8); yo[perm1] = np.argmax(y_train[:, :10], axis=1).astype(np.uint8)
    Xt = X_test.reshape(X_test.shape[0], -1)
    yt = np.argmax(y_test[:, :10], axis=1).astype(np.uint8)
    arrays = {"train_images_14x14.npy": Xo, "train_labels.npy": yo, "test_images_14x14.npy": Xt, "test_labels.npy": yt}
    digest = {n: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for n, a in arrays.items()}
    if "data" not in result:
        ddir = os.path.join(out_dir, "data")
        os.makedirs(ddir, exist_ok=True)
        files = []
        for n, a in arrays.items():
            path = os.path.join(ddir, n)
            np.save(path, a)
            files.append({"file": n, "shape": list(a.shape), "dtype": str(a.dtype),
                          "sha256_npy_file": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                          "sha256_raw_array_bytes": digest[n]})
        result["data"] = {
            "files": files, "scaling": dict(SCALING), "array_digests_seed0": digest, "identical_for_seeds": [c["seed"]],
            "pipeline": [
                "MNIST uint8 images (torchvision MNIST from the IDX files), cast to float32 with values 0..255, no padding",
                f"training set: first K = {K} images (IDX order); test set: all 10000 images",
                f"resize to {c['resize']} x {c['resize']} with bicubic interpolation (torchvision Resize on the float32 0..255 images, so values may overshoot 0..255)",
                "min-max scaling AFTER the resize: x' = (x - min) / (max - min), with min and max taken over the resized K training images (values in `scaling`); the test images use the same training min and max, so they can fall slightly outside [0, 1]",
                "no mean/std normalization (normalize = False)",
                "flattened row-major to resize*resize features, as the network sees them",
            ],
            "row_order": "train_images row i = MNIST training image i (i < K); test rows in IDX order",
            "labels": "uint8 class 0..9 (ReBoot one-hot pads them to 16 columns; padding columns are always 0)",
        }
    else:
        assert digest == result["data"]["array_digests_seed0"], "data differ across seeds"
        result["data"]["identical_for_seeds"].append(c["seed"])

    def make_model(opt):
        return LocalLossMLP(
            cc=None, num_fc_layers=c["num_fc_layers"], num_fc_hidden=c["num_fc_hidden"],
            num_classes=y_train.shape[1], input_dim=X_train.shape[1], dropout_rate=c["dropout_rate"],
            non_linearity=NonLinearity[c["non_linearity"]], fwd_decay_inv=c["fwd_decay"],
            subnet_decay_inv=c["lrn_decay"], bias=c["bias"], debug=False, optimizer=OptimizerName[opt],
            momentum=c["momentum"], instantiate_last_block=c["instantiate_last_block"],
            subnet_trainable=c["subnet_trainable"], output_non_linearity=c["output_non_linearity"], encrypted=False)

    np.random.seed(c["seed"])
    make_model("PLAIN_NESTEROV_SGD")                       # plain twin
    np.random.seed(c["seed"])
    enc_init = make_model(c["optimizer"])                  # encrypted model's draws (same shapes)
    nb = K // B
    order_pos = np.arange(K)
    steps, lrs = [], []
    sched = CosineLR(base_lr=c["lr"], steps_before_restart=50, from_step=5 * nb)
    for e in range(E):
        p, _ = shuffle_dataset(np.arange(K).reshape(-1, 1), np.arange(K).reshape(-1, 1))
        order_pos = order_pos[p.reshape(-1)]
        for b in range(nb):
            sched.step()
            lrs.append(float(sched.get_lr()))
            steps.append((e, b, [int(perm1[i]) for i in order_pos[b * B:(b + 1) * B]]))
        dropped = [int(perm1[i]) for i in order_pos[nb * B:]]

    fname = "train_indices.txt" if seed == seeds[0] else f"train_indices_seed{seed}.txt"
    with open(os.path.join(out_dir, fname), "w") as f:
        f.write(f"# ReBoot full setting (precision_analysis.py {' '.join(argv)}): MNIST training-set indices\n")
        f.write(f"# (0-based, IDX/torchvision order; all < {K}) per step, in order. {E} epochs x {nb} steps.\n")
        f.write("# Reconstructed from ReBoot's own shuffles without training (method checked on the subset run).\n")
        f.write("# step<TAB>epoch<TAB>batch<TAB>48 indices separated by spaces (step and batch 1-based, epoch 1-based)\n")
        for s, (e, b, ix) in enumerate(steps, 1):
            f.write(f"{s}\t{e + 1}\t{b + 1}\t" + " ".join(map(str, ix)) + "\n")
        f.write(f"# not trained in each epoch: {K - nb * B} samples\n")

    wdir = os.path.join(out_dir, f"init_weights_seed{seed}")
    os.makedirs(wdir, exist_ok=True)
    wfiles = []
    for i, layer in enumerate(enc_init.get_trainable_layers(), 1):
        w = np.asarray(layer.weights)
        path = os.path.join(wdir, f"W{i}.npy")
        np.save(path, w.astype(np.float64))
        wfiles.append({"file": f"init_weights_seed{seed}/W{i}.npy", "shape": list(w.shape), "orientation": "in x out",
                       "dtype_in_reboot": str(w.dtype), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()})

    # Plaintext twin, as precision_analysis.py trains it: plain model (PLAIN_NESTEROV_SGD) built right after
    # np.random.seed(S) (same initial weights as the encrypted model), ReBoot's plaintext step on each batch with the
    # scheduler's lr. precision_analysis.py does not write these per-step losses or any test accuracy to disk; they are
    # recomputed here by running that same plaintext path.
    pos = np.empty(K, dtype=int); pos[perm1] = np.arange(K)
    np.random.seed(c["seed"])
    twin = make_model("PLAIN_NESTEROV_SGD")
    tw = []
    for s_i, ((e, b, ix), lr) in enumerate(zip(steps, lrs), 1):
        rows = pos[ix]
        loss, acc, _, _ = train_step_plain(model=twin, X_train=X_train[rows], y_train=y_train[rows], lr=lr)
        tw.append((s_i, e + 1, b + 1, lr, float(loss), float(acc)))
    twin.eval()
    pred = np.asarray(twin.forward(X_test.reshape(X_test.shape[0], -1)))
    test_acc = float(np.mean(np.argmax(pred[:, :10], axis=1) == np.argmax(y_test[:, :10], axis=1)))
    test_acc_all = float(np.mean(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1)))
    tname = f"twin_steps_seed{seed}.tsv"
    with open(os.path.join(out_dir, tname), "w") as f:
        f.write("# ReBoot plaintext twin (precision_analysis.py setting), recomputed with ReBoot's plaintext step; batch loss = sum over the batch of per-sample sums of squared errors over 16 outputs\n")
        f.write("step\tepoch\tbatch\tlr\tbatch_loss\tbatch_acc\n")
        for r in tw:
            f.write("%d\t%d\t%d\t%.10g\t%.9g\t%.9g\n" % r)

    lr_changes = [{"step": s + 1, "lr": lrs[s]} for s in range(len(lrs)) if s == 0 or lrs[s] != lrs[s - 1]]
    result["per_seed"][str(seed)] = {
        "argv": argv, "indices_file": fname, "init_weights": wfiles,
        "steps": len(steps), "epochs": E, "steps_per_epoch": nb, "batch_size": B,
        "train_samples_drawn": K, "subsample_rule": "first K MNIST training images (IDX order)",
        "samples_not_trained_per_epoch": K - nb * B,
        "input_features": int(X_train.shape[1]), "input_resize": c["resize"],
        "output_width": int(y_train.shape[1]), "test_samples": int(X_test.shape[0]),
        "lr_per_step": lrs, "lr_first": lrs[0], "lr_last": lrs[-1], "lr_min": min(lrs), "lr_max": max(lrs),
        "lr_constant_steps": sum(1 for x in lrs if x == lrs[0]), "lr_change_points_first_10": lr_changes[:10],
        "seeds": {"model_and_epoch_shuffle_seed": c["seed"], "subsample_shuffle_seed": 42},
        "listed_in_experiments_sh": seed in listed_seeds,
        "twin": {"steps_file": tname, "test_acc_argmax10": test_acc, "test_acc_argmax_all_outputs": test_acc_all,
                 "final_step_batch_loss": tw[-1][4], "epoch_mean_train_acc_last_epoch": float(np.mean([r[5] for r in tw if r[1] == E])),
                 "source": "recomputed (ReBoot's plaintext step on the reconstructed batches); precision_analysis.py logs no per-step twin loss to disk and no test accuracy"},
    }
    c0 = c

result["config_values"] = {
    "lr_base": c0["lr"], "lr_note": "set in precision_analysis.py's configuration (the --lr argument is not used there)",
    "lr_schedule": "constant for the first 5 epochs (500 steps), then cosine annealing with warm restarts every 50 steps down to 1e-7, stepped once per batch; exact per-step values in per_seed[*].lr_per_step",
    "momentum": c0["momentum"], "optimizer": "Nesterov momentum SGD (encrypted variant; the plaintext twin uses the plaintext variant)",
    "weight_decay": {"fwd": c0["fwd_decay"], "local": c0["lrn_decay"]},
    "batch_size": c0["batch_size"], "epochs": c0["num_epochs"], "subsample_K": c0["subsample"],
    "input_resize": c0["resize"], "input_resize_note": "images resized to resize x resize with bicubic interpolation before flattening",
    "normalize": c0["normalize"], "bias": c0["bias"], "hidden": c0["num_fc_hidden"],
    "activation": "x^2 + x", "pad_labels": c0["pad_labels"],
    "differs_from_subset_run": "the subset run used training_encrypted.py (no resize: 784 inputs; lr from --lr default 5e-4; scheduler stepped once per epoch); the full setting uses precision_analysis.py (14x14 = 196 inputs; scheduler stepped once per batch; trains a plaintext twin alongside)",
    "wandb": c0["wandb"],
}
accs = [v["twin"]["test_acc_argmax10"] for v in result["per_seed"].values()]
result["twin_test_acc_over_seeds"] = {"seeds": seeds, "values": accs, "mean": float(np.mean(accs)),
                                      "std_population": float(np.std(accs)), "std_sample": float(np.std(accs, ddof=1)) if len(accs) > 1 else None}
# Reported accuracy in the repository's text files (README, notebooks, csv/json/txt data), if any.
hits = []
for path in glob.glob(os.path.join(SRC, "**", "*"), recursive=True):
    if os.path.isfile(path) and path.endswith((".md", ".txt", ".csv", ".json", ".ipynb", ".yaml", ".yml")):
        txt = open(path, errors="ignore").read()
        for ln, line in enumerate(txt.splitlines(), 1):
            if re.search(r"accuracy|Accuracy", line):
                nums = re.findall(r"\b\d{2}\.\d{1,2}\b", line)
                if nums:  # record only where and which numbers, not the text
                    hits.append({"file": os.path.relpath(path, SRC), "line": ln, "numbers": nums[:20]})
result["paper_reported_accuracy_in_repo"] = hits if hits else "not found"
json.dump(result, open(os.path.join(out_dir, "config_values.json"), "w"), indent=2)
print("R-REB-2 twin test acc", result["twin_test_acc_over_seeds"], "seeds", seeds, "steps", {s: v["steps"] for s, v in result["per_seed"].items()},
      "inputs", result["per_seed"][str(seeds[0])]["input_features"], "acc hits", len(hits) if hits else 0)
