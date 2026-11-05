"""Recover which MNIST training images ReBoot's subset run used in each step, and check that by replaying the run in
plaintext. Lives at ~/reboot/reboot_indices.py on quadruped-gpu. Contains no ReBoot code: it imports ReBoot's modules,
loads training_encrypted.py's configuration via runpy (its training loop is under `if __name__ == "__main__"`, so it
does not run), and calls ReBoot's own data loader, shuffles, model class and plaintext train step.

Index reconstruction (same RNG sequence as the script, which shuffles index lists whose length is all that matters):
  1. load_float_dataset keeps the first K training images, then shuffles them with seed 42  -> perm1
  2. the script seeds np.random with `seed`, builds the plain model, re-seeds, builds the encrypted model (the same
     number of draws as the plain model is assumed here), then shuffles once per epoch with the global RNG -> perm2
Check: a plaintext model with the same initial weights is trained with ReBoot's plaintext step on the reconstructed
batches; its per-step batch loss and accuracy must match the encrypted run's log (stats.json).

Usage: python reboot_indices.py STATS_JSON OUT_TXT OUT_JSON  (argv of the run is read from STATS_JSON)
"""
import json
import os
import runpy
import sys

import numpy as np

SRC = os.environ["REBOOT_SRC"]
sys.path.insert(0, SRC)
EXP = os.path.join(SRC, "experiments", "3_training_encrypted")
stats_path, out_txt, out_json = sys.argv[1:4]
st = json.load(open(stats_path))
run_argv = st["wrapper"]["argv"]

sys.argv = [os.path.join(EXP, "training_encrypted.py")] + run_argv
os.chdir(EXP)
g = runpy.run_path(sys.argv[0], run_name="reboot_config_only")
configs = g["configs"]
from lib.models.local_loss_models import LocalLossMLP  # noqa: E402
from lib.utils.data import load_float_dataset, shuffle_dataset  # noqa: E402
from lib.utils.enums import NonLinearity, OptimizerName  # noqa: E402
from lib.utils.train import train_step_plain  # noqa: E402

K, B, E = configs["subsample"], configs["batch_size"], configs["num_epochs"]
assert E == 1, "index reconstruction implemented for 1 epoch"

np.random.seed(configs["seed"])
X_train, _, X_test, y_train, _, y_test = load_float_dataset(configs, show_images=False, show_log=False)
idx = np.arange(K).reshape(-1, 1)
perm1, _ = shuffle_dataset(idx, idx, seed=42)
perm1 = perm1.reshape(-1)


def make_model(optimizer=None):
    return LocalLossMLP(
        cc=None, num_fc_layers=configs["num_fc_layers"], num_fc_hidden=configs["num_fc_hidden"],
        num_classes=y_train.shape[1], input_dim=int(np.prod(X_train.shape[1:])),
        dropout_rate=configs["dropout_rate"], non_linearity=NonLinearity[configs["non_linearity"]],
        fwd_decay_inv=configs["fwd_decay"], subnet_decay_inv=configs["lrn_decay"], bias=configs["bias"],
        debug=False, optimizer=optimizer or OptimizerName[configs["optimizer"]], momentum=configs["momentum"],
        instantiate_last_block=configs["instantiate_last_block"], subnet_trainable=configs["subnet_trainable"],
        output_non_linearity=configs["output_non_linearity"], encrypted=False)


# Same RNG consumption as the script before the epoch shuffle.
np.random.seed(configs["seed"])
make_model()                      # plain_model
np.random.seed(configs["seed"])
make_model()                      # stands in for encrypted_model's draws (same shapes)
perm2, _ = shuffle_dataset(np.arange(K).reshape(-1, 1), np.arange(K).reshape(-1, 1))
perm2 = perm2.reshape(-1)
order = perm1[perm2]              # position in the first-K MNIST training images, in training order

# Check 1: the reconstructed order reproduces ReBoot's preprocessed X_train.
from torchvision.datasets import MNIST  # noqa: E402
raw = MNIST(os.environ["REBOOT_DATA"], train=True, download=False).data.numpy()[:K].reshape(K, -1) / 255.0
xf = X_train.reshape(K, -1)
data_check = float(np.max(np.abs(xf - raw[perm1])))

# Check 2: plaintext replay with ReBoot's plaintext step on the reconstructed batches.
# ReBoot's encrypted Nesterov optimizer only accepts ciphertexts, so the replay uses its plaintext Nesterov
# optimizer (PLAIN_NESTEROV_SGD); a matching per-step loss also checks that the two optimizers agree.
np.random.seed(configs["seed"])
model = make_model(OptimizerName.PLAIN_NESTEROV_SGD)
Xs, ys = xf[perm2], y_train[perm2]
nb = K // B
replay = []
for b in range(nb):
    loss, acc, _, _ = train_step_plain(model=model, X_train=Xs[b * B:(b + 1) * B], y_train=ys[b * B:(b + 1) * B],
                                       lr=st["steps"][b]["lr"])
    replay.append((float(loss), float(acc)))
enc = [(s["batch_loss"], s["batch_train_acc"]) for s in st["steps"]]
rel = [abs(p[0] - e[0]) / abs(e[0]) for p, e in zip(replay, enc)]
acc_eq = sum(abs(p[1] - e[1]) < 1e-9 for p, e in zip(replay, enc))

with open(out_txt, "w") as f:
    f.write("# ReBoot subset run (training_encrypted.py " + " ".join(run_argv) + "): MNIST training-set indices\n")
    f.write("# (0-based, in the order of the torchvision/IDX training file; all < %d) used in each step, in order.\n" % K)
    f.write("# Reconstructed from ReBoot's own shuffles (raw/reboot_emlp1_official_subset/reboot_indices.json has the checks).\n")
    f.write("# step<TAB>48 indices separated by spaces\n")
    for b in range(nb):
        f.write(f"{b + 1}\t" + " ".join(str(int(i)) for i in order[b * B:(b + 1) * B]) + "\n")
    f.write("# dropped (not trained, K - B*floor(K/B) = %d): " % (K - nb * B) + " ".join(str(int(i)) for i in order[nb * B:]) + "\n")

json.dump({
    "run_argv": run_argv, "replay_optimizer": "PLAIN_NESTEROV_SGD (ReBoot plaintext counterpart of ENCRYPTED_NESTEROV_SGD)", "subsample_K": K, "batch": B, "steps": nb, "trained_samples": nb * B,
    "dropped": int(K - nb * B),
    "check_preprocessed_data_vs_raw_mnist_max_abs": data_check,
    "check_plaintext_replay": {
        "batch_loss_rel_diff_max": max(rel), "batch_loss_rel_diff_per_step": rel,
        "batch_acc_equal_steps": int(acc_eq), "steps": nb,
        "replay_loss_acc": replay, "encrypted_loss_acc": enc,
    },
}, open(out_json, "w"), indent=2)
print("data_check", data_check, "max rel loss diff", max(rel), "acc equal", acc_eq, "/", nb)
