"""R-REB-5: ReBoot eMLP-2 / eMLP-3 at the full released setting, as data. Lives at ~/reboot/reboot_emlp23.py on
quadruped-gpu. Contains no ReBoot code: it imports ReBoot modules, reads precision_analysis.py's configuration via runpy
(its training loop does not run) with the MLP-2 / MLP-3 argument lines of experiments.sh, builds the models with ReBoot's
constructor, introspects them (layer names, classes, shapes, decay rates, which layers pass gradients down), saves the
initial weights, rebuilds the per-step orders from ReBoot's shuffles, and replays ReBoot's plaintext twin.

Usage: python reboot_emlp23.py OUT_DIR EMLP1_INDICES_DIR
"""
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
out_dir, emlp1_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

from lib.models.local_loss_models import LocalLossMLP  # noqa: E402
from lib.optim.schedulers import CosineLR  # noqa: E402
from lib.utils.data import load_float_dataset, shuffle_dataset  # noqa: E402
from lib.utils.enums import NonLinearity, OptimizerName  # noqa: E402
from lib.utils.train import train_step_plain  # noqa: E402

sh = open(os.path.join(EXP, "experiments.sh")).read().replace("\r", "")
sections = {"emlp2": sh.split("# MLP-2")[1].split("# MLP-3")[0], "emlp3": sh.split("# MLP-3")[1]}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def read_emlp1_order(seed):
    name = "train_indices.txt" if seed == 0 else f"train_indices_seed{seed}.txt"
    rows = [l.split("\t") for l in open(os.path.join(emlp1_dir, name)) if l[:1].isdigit()]
    return [list(map(int, r[3].split())) for r in rows]


def describe_layer(l):
    d = {"name": l.name, "class": type(l).__name__}
    for a in ("in_features", "out_features", "propagate_backward", "decay_rate", "momentum", "trainable"):
        if hasattr(l, a):
            v = getattr(l, a)
            d[a] = v if isinstance(v, (bool, int, float, str)) or v is None else str(v)
    if hasattr(l, "optimizer") and l.optimizer is not None:
        d["optimizer"] = type(l.optimizer).__name__
        if hasattr(l.optimizer, "weight_decay"):
            d["optimizer_weight_decay"] = float(l.optimizer.weight_decay)
    if hasattr(l, "weights") and l.weights is not None:
        d["weight_shape"] = list(np.shape(l.weights))
    return d


result = {"source": "experiments.sh MLP-2 / MLP-3 lines, precision_analysis.py configuration", "models": {}}
for key, sec in sections.items():
    lines = [l.strip() for l in sec.splitlines() if l.strip().startswith("python")]
    seeds = [int(re.search(r"--seed (\d+)", l).group(1)) for l in lines]
    argv0 = lines[0].split()[2:]
    M = {"command_lines": lines, "seeds_listed": seeds, "per_seed": {}}
    for seed in seeds:
        argv = [a if not (i > 0 and argv0[i - 1] == "--seed") else str(seed) for i, a in enumerate(argv0)]
        sys.argv = [os.path.join(EXP, "precision_analysis.py")] + argv
        cwd = os.getcwd(); os.chdir(EXP)
        c = runpy.run_path(sys.argv[0], run_name="reboot_config_only")["configs"]
        os.chdir(cwd)
        K, B, E = c["subsample"], c["batch_size"], c["num_epochs"]
        np.random.seed(c["seed"])
        X, _, Xt, Y, _, Yt = load_float_dataset(c, show_images=False, show_log=False)
        X = X.reshape(K, -1); Xt = Xt.reshape(Xt.shape[0], -1)
        idx = np.arange(K).reshape(-1, 1)
        perm1, _ = shuffle_dataset(idx, idx, seed=42); perm1 = perm1.reshape(-1)

        def make(opt):
            return LocalLossMLP(
                cc=None, num_fc_layers=c["num_fc_layers"], num_fc_hidden=c["num_fc_hidden"], num_classes=Y.shape[1],
                input_dim=X.shape[1], dropout_rate=c["dropout_rate"], non_linearity=NonLinearity[c["non_linearity"]],
                fwd_decay_inv=c["fwd_decay"], subnet_decay_inv=c["lrn_decay"], bias=c["bias"], debug=False,
                optimizer=OptimizerName[opt], momentum=c["momentum"], instantiate_last_block=c["instantiate_last_block"],
                subnet_trainable=c["subnet_trainable"], output_non_linearity=c["output_non_linearity"], encrypted=False)

        np.random.seed(c["seed"]); twin = make("PLAIN_NESTEROV_SGD")
        np.random.seed(c["seed"]); enc_init = make(c["optimizer"])
        nb = K // B
        pos = np.arange(K); steps, lrs = [], []
        sched = CosineLR(base_lr=c["lr"], steps_before_restart=50, from_step=5 * nb)
        for e in range(E):
            p, _ = shuffle_dataset(np.arange(K).reshape(-1, 1), np.arange(K).reshape(-1, 1))
            pos = pos[p.reshape(-1)]
            for b in range(nb):
                sched.step(); lrs.append(float(sched.get_lr()))
                steps.append((e, b, [int(perm1[i]) for i in pos[b * B:(b + 1) * B]]))
        e1 = read_emlp1_order(seed)
        same_as_emlp1 = [s[2] for s in steps] == e1
        fname = f"train_indices_{key}_seed{seed}.txt"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(f"# ReBoot {key} full setting (precision_analysis.py {' '.join(argv)}): MNIST training-set indices per step\n")
            f.write("# step<TAB>epoch<TAB>batch<TAB>48 indices (0-based IDX order; step/epoch/batch 1-based)\n")
            for s_i, (e, b, ix) in enumerate(steps, 1):
                f.write(f"{s_i}\t{e + 1}\t{b + 1}\t" + " ".join(map(str, ix)) + "\n")

        # structure as data (first seed only; identical across seeds)
        if "structure" not in M:
            M["structure"] = {
                "blocks": [{"block_name": blk.name, "class": type(blk).__name__,
                            "forward_layers": [describe_layer(l) for l in blk.layers.layers] if hasattr(blk.layers, "layers") else str(blk.layers),
                            "local_head": [describe_layer(l) for l in blk.pred_loss_net.layers] if hasattr(blk.pred_loss_net, "layers") else str(blk.pred_loss_net)}
                           for blk in enc_init.blocks],
                "output_stack": [describe_layer(l) for l in enc_init.output_layers.layers],
                "trainable_layer_order": [l.name for l in enc_init.get_layers_with_parameters()],
            }
            M["config"] = {"argv": argv, "lr": c["lr"], "momentum": c["momentum"], "fwd_decay": c["fwd_decay"],
                           "lrn_decay": c["lrn_decay"], "batch_size": B, "epochs": E, "steps_per_epoch": nb,
                           "steps": len(steps), "subsample_K": K, "input_resize": c["resize"],
                           "input_features": int(X.shape[1]), "output_width": int(Y.shape[1]), "hidden": c["num_fc_hidden"],
                           "num_fc_layers": c["num_fc_layers"], "instantiate_last_block": c["instantiate_last_block"],
                           "bias": c["bias"], "optimizer": c["optimizer"], "activation": "x^2 + x",
                           "lr_first": lrs[0], "lr_last": lrs[-1],
                           "lr_schedule": "constant for 5 epochs (500 steps), then cosine annealing with warm restarts every 50 steps, stepped once per batch (ReBoot's scheduler)"}
        wdir = os.path.join(out_dir, f"init_weights_{key}_seed{seed}")
        os.makedirs(wdir, exist_ok=True)
        wf = []
        for l in enc_init.get_layers_with_parameters():
            w = np.asarray(l.weights); p_ = os.path.join(wdir, f"{l.name}.npy"); np.save(p_, w.astype(np.float64))
            wf.append({"file": f"init_weights_{key}_seed{seed}/{l.name}.npy", "layer": l.name, "shape": list(w.shape),
                       "orientation": "in x out", "dtype_in_reboot": str(w.dtype), "sha256": sha(p_)})
        entry = {"argv": argv, "indices_file": fname, "indices_identical_to_emlp1_same_seed": same_as_emlp1,
                 "init_weights": wf, "lr_per_step": lrs}
        if seed == 0:
            rowof = np.empty(K, dtype=int); rowof[perm1] = np.arange(K)
            tw = []
            for s_i, ((e, b, ix), lr) in enumerate(zip(steps, lrs), 1):
                r = rowof[ix]
                loss, acc, _, _ = train_step_plain(model=twin, X_train=X[r], y_train=Y[r], lr=lr)
                tw.append((s_i, e + 1, b + 1, lr, float(loss), float(acc)))
            twin.eval()
            pred = np.asarray(twin.forward(Xt))
            tname = f"twin_steps_{key}_seed0.tsv"
            with open(os.path.join(out_dir, tname), "w") as f:
                f.write(f"# ReBoot plaintext twin ({key}, full setting, seed 0), recomputed with ReBoot's plaintext step; batch loss/acc as ReBoot's step returns them (main output)\n")
                f.write("step\tepoch\tbatch\tlr\tbatch_loss\tbatch_acc\n")
                for r_ in tw:
                    f.write("%d\t%d\t%d\t%.10g\t%.9g\t%.9g\n" % r_)
            entry["twin"] = {"steps_file": tname,
                             "test_acc_argmax10": float(np.mean(np.argmax(pred[:, :10], 1) == np.argmax(Yt[:, :10], 1))),
                             "test_acc_argmax_all_outputs": float(np.mean(np.argmax(pred, 1) == np.argmax(Yt, 1))),
                             "test_eval": "model output (output stack), not the local heads",
                             "final_step_batch_loss": tw[-1][4]}
        M["per_seed"][str(seed)] = entry
    result["models"][key] = M

json.dump(result, open(os.path.join(out_dir, "config_values.json"), "w"), indent=2, default=str)
print("R-REB-5", {k: {"twin_seed0": v["per_seed"]["0"]["twin"]["test_acc_argmax10"],
                      "same_indices_as_emlp1": {s: e["indices_identical_to_emlp1_same_seed"] for s, e in v["per_seed"].items()}}
                  for k, v in result["models"].items()})
