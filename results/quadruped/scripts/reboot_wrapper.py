"""Run ReBoot's unmodified experiments/3_training_encrypted/training_encrypted.py with external instrumentation.

Lives at ~/reboot/reboot_wrapper.py on quadruped-gpu (outside ~/hekit). Contains no ReBoot code: it imports
ReBoot modules and wraps three functions at runtime, then executes the training script with runpy.

  * os.cpu_count -> REBOOT_THREADS (default 12): the script calls reboot_cpp.set_num_threads(os.cpu_count()),
    which is 12 on the paper/Mac setup and would be 32 on this server.
  * reboot_cpp.bootstrap_array: counts calls, ciphertexts and the num_iterations argument, and times each call.
  * lib.utils.train.train_step_encrypted: records ReBoot's own step_time per batch plus wall time per call.
  * Model.encrypt: snapshots the plaintext initial weights and checks that our decoder reproduces them from the
    freshly encrypted weights (max abs error), which validates the decoder used for the test evaluation.

After training, the encrypted weights are decrypted, decoded into plain (in, out) matrices, loaded into the
script's own plain_model (same seed and architecture), and evaluated on the script's own X_test / y_test
(10,000 MNIST test images). ReBoot's script itself prints training metrics only.

Usage: REBOOT_SRC=~/reboot/src/ReBoot REBOOT_STATS_OUT=stats.json python reboot_wrapper.py MNIST --subsample ...
"""
import json
import os
import runpy
import sys
import time

THREADS = int(os.environ.get("REBOOT_THREADS", "12"))
os.cpu_count = lambda: THREADS
SRC = os.environ["REBOOT_SRC"]
STATS_OUT = os.environ["REBOOT_STATS_OUT"]
sys.path.insert(0, SRC)
EXP_DIR = os.path.join(SRC, "experiments", "3_training_encrypted")
SCRIPT = os.path.join(EXP_DIR, "training_encrypted.py")

import numpy as np  # noqa: E402
import reboot_cpp  # noqa: E402
import lib.utils.train as train_mod  # noqa: E402
from lib.models.models import Model  # noqa: E402

T0 = time.perf_counter()
STATS = {
    "wrapper": {"threads_os_cpu_count_override": THREADS, "script": SCRIPT, "argv": sys.argv[1:]},
    "bootstrap": {"bootstrap_array_calls": 0, "ciphertexts_bootstrapped": 0, "num_iterations_args": [],
                  "precision_args": [], "seconds_per_call": []},
    "steps": [],
}


def dump():
    STATS["wrapper"]["elapsed_s"] = time.perf_counter() - T0
    with open(STATS_OUT, "w") as f:
        json.dump(STATS, f, indent=2, default=float)


_orig_bs = reboot_cpp.bootstrap_array


def bootstrap_array(array, *args, **kwargs):
    b = STATS["bootstrap"]
    b["bootstrap_array_calls"] += 1
    b["ciphertexts_bootstrapped"] += len(array)
    b["num_iterations_args"].append(kwargs.get("num_iterations", args[0] if args else None))
    b["precision_args"].append(kwargs.get("precision", args[1] if len(args) > 1 else None))
    t = time.perf_counter()
    r = _orig_bs(array, *args, **kwargs)
    b["seconds_per_call"].append(time.perf_counter() - t)
    return r


reboot_cpp.bootstrap_array = bootstrap_array

_orig_step = train_mod.train_step_encrypted


def train_step_encrypted(*args, **kwargs):
    t_start = time.time()
    t = time.perf_counter()
    loss, acc, step_time, pred = _orig_step(*args, **kwargs)
    STATS["steps"].append({"index": len(STATS["steps"]), "unix_start": t_start,
                           "reboot_step_time_s": float(step_time), "wall_s": time.perf_counter() - t,
                           "batch_loss": float(loss), "batch_train_acc": float(acc),
                           "lr": float(kwargs.get("lr", float("nan")))})
    dump()
    return loss, acc, step_time, pred


train_mod.train_step_encrypted = train_step_encrypted


def decode(layer, cc):
    """Decrypt a CKKSPackedLinear weight ciphertext into its logical (in_features, out_features) matrix."""
    n = layer.row_size * layer.col_size
    v = np.asarray(cc.decrypt_array(layer.weights, num_slots=n), dtype=np.float64).reshape(-1)[:n]
    m = v.reshape(layer.row_size, layer.col_size)
    if not layer.row_packing:
        m = m.T
    return m[: layer.in_features, : layer.out_features]


_orig_encrypt = Model.encrypt


def encrypt(self, cc):
    plain = [np.array(layer.weights, dtype=np.float64).copy() for layer in self.get_trainable_layers()]
    _orig_encrypt(self, cc)
    errs = [float(np.max(np.abs(decode(layer, cc) - w)))
            for layer, w in zip(self.get_trainable_layers(), plain)]
    STATS["decoder_check"] = {"max_abs_err_per_layer_at_encryption": errs,
                              "shapes": [list(w.shape) for w in plain]}
    STATS["timeline"] = {"encrypt_done_s": time.perf_counter() - T0}
    dump()


Model.encrypt = encrypt

# Run the unmodified script (cwd = its own dir: it reads ../config/fhe_config_mlp1.yaml).
sys.argv = [SCRIPT] + sys.argv[1:]
os.chdir(EXP_DIR)
t_train = time.perf_counter()
g = runpy.run_path(SCRIPT, run_name="__main__")
STATS["timeline"]["script_done_s"] = time.perf_counter() - T0
STATS["timeline"]["script_wall_s"] = time.perf_counter() - t_train

# Test evaluation with the decrypted trained weights.
cc, enc, plain_model = g["cc"], g["encrypted_model"], g["plain_model"]
X_test, y_test = g["X_test"], g["y_test"]
X_test = X_test.reshape(X_test.shape[0], -1)
plain_model.eval()
init_pred = np.asarray(plain_model.forward(X_test))
enc_layers, pl_layers = enc.get_trainable_layers(), plain_model.get_trainable_layers()
assert len(enc_layers) == len(pl_layers)
Ws = [decode(layer, cc) for layer in enc_layers]
for pl, W in zip(pl_layers, Ws):
    assert tuple(np.shape(pl.weights)) == W.shape, (np.shape(pl.weights), W.shape)
    pl.weights = W.astype(np.float32)
pred = np.asarray(plain_model.forward(X_test))
ncls = 10
yt = np.argmax(y_test[:, :ncls], axis=1)
STATS["evaluation"] = {
    "test_samples": int(X_test.shape[0]),
    "test_acc_encrypted_trained_argmax10": float(np.mean(np.argmax(pred[:, :ncls], axis=1) == yt)),
    "test_acc_encrypted_trained_argmax_all_outputs": float(np.mean(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))),
    "test_acc_init_argmax10": float(np.mean(np.argmax(init_pred[:, :ncls], axis=1) == yt)),
    "test_mse_encrypted_trained": float(np.mean((pred - y_test) ** 2)),
    "output_dim": int(pred.shape[1]),
    "train_samples": int(g["X_train"].shape[0]),
    "batches_per_epoch": int(g["n_train_batches"]),
    "final_train_loss_reported": float(g["loss_history_enc"][-1]),
    "final_train_acc_reported": float(g["acc_history_enc"][-1]),
    "weight_abs_max_per_layer": [float(np.max(np.abs(W))) for W in Ws],
    "method": "decrypt trained weights, decode, load into the script's plain_model, plaintext forward on X_test",
}
STATS["crypto"] = {k: getattr(cc, k, None) for k in ("row_size", "col_size", "valid_slots", "num_slots",
                                                       "bs_levels", "bs_iterations", "bs_precision")}
dump()
print("WRAPPER_DONE", json.dumps(STATS["evaluation"]))
