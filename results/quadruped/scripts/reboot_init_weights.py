"""R-REB-1: save ReBoot eMLP-1's initial weights (subset run setup, seed 0) as .npy data and check them by replaying
step 1. Lives at ~/reboot/reboot_init_weights.py on quadruped-gpu. Contains no ReBoot code: it imports ReBoot's modules,
reads training_encrypted.py's configuration via runpy (the training loop does not run), and uses ReBoot's own model
class, data loader and plaintext train step.

The model is built exactly as the script builds it before encryption: np.random.seed(seed), then the LocalLossMLP
constructor; the trainable layers' weight matrices are saved before any step, as stored by ReBoot ((in, out), float32),
upcast to float64 (exact).
Check: a fresh model with these weights loaded runs ReBoot's plaintext Nesterov step on step 1's 48 samples (taken from
reboot_subsample_indices.txt); its batch loss and accuracy are compared with reboot_indices.json (plaintext replay) and
with the encrypted run's step 1 (stats.json).

Usage: python reboot_init_weights.py STATS_JSON INDICES_TXT INDICES_JSON OUT_DIR
"""
import hashlib
import json
import os
import runpy
import sys

import numpy as np

SRC = os.environ["REBOOT_SRC"]
sys.path.insert(0, SRC)
EXP = os.path.join(SRC, "experiments", "3_training_encrypted")
stats_path, idx_txt, idx_json, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
st = json.load(open(stats_path))
run_argv = st["wrapper"]["argv"]

sys.argv = [os.path.join(EXP, "training_encrypted.py")] + run_argv
os.chdir(EXP)
configs = runpy.run_path(sys.argv[0], run_name="reboot_config_only")["configs"]
from lib.models.local_loss_models import LocalLossMLP  # noqa: E402
from lib.utils.data import load_float_dataset, shuffle_dataset  # noqa: E402
from lib.utils.enums import NonLinearity, OptimizerName  # noqa: E402
from lib.utils.train import train_step_plain  # noqa: E402

K, B = configs["subsample"], configs["batch_size"]
np.random.seed(configs["seed"])
X_train, _, _, y_train, _, _ = load_float_dataset(configs, show_images=False, show_log=False)
xf = X_train.reshape(K, -1)


def make_model(optimizer=None):
    return LocalLossMLP(
        cc=None, num_fc_layers=configs["num_fc_layers"], num_fc_hidden=configs["num_fc_hidden"],
        num_classes=y_train.shape[1], input_dim=xf.shape[1],
        dropout_rate=configs["dropout_rate"], non_linearity=NonLinearity[configs["non_linearity"]],
        fwd_decay_inv=configs["fwd_decay"], subnet_decay_inv=configs["lrn_decay"], bias=configs["bias"],
        debug=False, optimizer=optimizer or OptimizerName[configs["optimizer"]], momentum=configs["momentum"],
        instantiate_last_block=configs["instantiate_last_block"], subnet_trainable=configs["subnet_trainable"],
        output_non_linearity=configs["output_non_linearity"], encrypted=False)


# Initial weights: same seeding and constructor call as the script's model setup.
np.random.seed(configs["seed"])
init = make_model()
layers = init.get_trainable_layers()
files = []
for i, layer in enumerate(layers, 1):
    w = np.asarray(layer.weights)
    path = os.path.join(out_dir, f"W{i}.npy")
    np.save(path, w.astype(np.float64))
    files.append({"file": f"W{i}.npy", "layer_name": layer.name, "shape": list(w.shape),
                  "orientation": "in x out (rows = input features, columns = output units); forward is x @ W",
                  "in_features": int(layer.in_features), "out_features": int(layer.out_features),
                  "dtype_in_reboot": str(w.dtype), "dtype_saved": "float64 (exact upcast)",
                  "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                  "min": float(w.min()), "max": float(w.max())})

# Step-1 batch from the indices file (original MNIST indices) mapped onto ReBoot's preprocessed arrays.
idx = np.arange(K).reshape(-1, 1)
perm1, _ = shuffle_dataset(idx, idx, seed=42)
perm1 = perm1.reshape(-1)
pos = np.empty(K, dtype=int)
pos[perm1] = np.arange(K)              # original index -> row in ReBoot's X_train
step1 = [int(t) for t in next(l for l in open(idx_txt) if l.startswith("1\t")).split("\t")[1].split()]
rows = pos[step1]

np.random.seed(configs["seed"])
model = make_model(OptimizerName.PLAIN_NESTEROV_SGD)
for layer, f in zip(model.get_trainable_layers(), files):
    layer.weights = np.load(os.path.join(out_dir, f["file"])).astype(np.float32)
loss, acc, _, _ = train_step_plain(model=model, X_train=xf[rows], y_train=y_train[rows], lr=st["steps"][0]["lr"])
ref = json.load(open(idx_json))["check_plaintext_replay"]
replay1, enc1 = ref["replay_loss_acc"][0], ref["encrypted_loss_acc"][0]
meta = {
    "request": "R-REB-1 (results/quadruped/requests.md)",
    "run_argv_of_subset_run": run_argv, "seed": configs["seed"],
    "construction": "np.random.seed(seed) then ReBoot's LocalLossMLP constructor with the subset run's configuration (as the script does before encrypting); weights read before any step",
    "files": files,
    "check_step1": {
        "batch_indices_source": "reboot_subsample_indices.txt step 1",
        "replay_from_npy": {"batch_loss": float(loss), "batch_acc": float(acc)},
        "reboot_indices_json_replay": {"batch_loss": replay1[0], "batch_acc": replay1[1]},
        "encrypted_run_step1": {"batch_loss": enc1[0], "batch_acc": enc1[1]},
        "abs_diff_vs_replay": abs(float(loss) - replay1[0]),
        "rel_diff_vs_encrypted": abs(float(loss) - enc1[0]) / abs(enc1[0]),
        "acc_equal_replay": abs(float(acc) - replay1[1]) < 1e-12,
        "acc_equal_encrypted": abs(float(acc) - enc1[1]) < 1e-12,
        "lr": st["steps"][0]["lr"], "optimizer": "PLAIN_NESTEROV_SGD (ReBoot plaintext counterpart of the encrypted Nesterov step)",
    },
}
json.dump(meta, open(os.path.join(out_dir, "init_weights.json"), "w"), indent=2)
print("R-REB-1", json.dumps(meta["check_step1"]))
