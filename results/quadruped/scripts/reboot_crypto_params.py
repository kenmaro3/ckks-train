"""R-REB-4: ReBoot eMLP-1's CKKS parameters as data. Lives at ~/reboot/reboot_crypto_params.py on quadruped-gpu.
Contains no ReBoot code: it imports ReBoot's CryptoContext class and builds it exactly as training_encrypted.py does for
the subset run (same config file, same plain model shape), but intercepts openfhe.GenCryptoContext: the CCParams object
ReBoot passes is captured, the real context is generated from it, and the constructor is then stopped before any key
generation or bootstrapping setup (those do not change the modulus chain, which GenCryptoContext fixes).

Usage: python reboot_crypto_params.py OUT_JSON [--probe]
"""
import json
import os
import runpy
import sys

import openfhe as fhe

SRC = os.environ["REBOOT_SRC"]
sys.path.insert(0, SRC)
EXP = os.path.join(SRC, "experiments", "3_training_encrypted")
out_json = sys.argv[1]
probe = "--probe" in sys.argv

CAP = {}
_real_gen = fhe.GenCryptoContext


class _Stop(Exception):
    pass


def _gen(params):
    CAP["params"] = params
    CAP["cc"] = _real_gen(params)
    raise _Stop()


fhe.GenCryptoContext = _gen

sys.argv = [os.path.join(EXP, "training_encrypted.py"), "MNIST", "--subsample", "1024", "--num-layers", "2",
            "--num-hidden", "32", "--batch-size", "48", "--num-epochs", "1", "--seed", "0"]
os.chdir(EXP)
c = runpy.run_path(sys.argv[0], run_name="reboot_config_only")["configs"]
from lib.cryptocontext import CryptoContext  # noqa: E402
from lib.models.local_loss_models import LocalLossMLP  # noqa: E402
from lib.utils.enums import NonLinearity, OptimizerName  # noqa: E402

plain = LocalLossMLP(
    cc=None, num_fc_layers=c["num_fc_layers"], num_fc_hidden=c["num_fc_hidden"], num_classes=16, input_dim=784,
    dropout_rate=c["dropout_rate"], non_linearity=NonLinearity[c["non_linearity"]], fwd_decay_inv=c["fwd_decay"],
    subnet_decay_inv=c["lrn_decay"], bias=c["bias"], debug=False, optimizer=OptimizerName["PLAIN_NESTEROV_SGD"],
    momentum=c["momentum"], instantiate_last_block=c["instantiate_last_block"], subnet_trainable=c["subnet_trainable"],
    output_non_linearity=c["output_non_linearity"], encrypted=False)
try:
    CryptoContext("../config/fhe_config_mlp1.yaml", model=plain)
except _Stop:
    pass
params, cc = CAP["params"], CAP["cc"]

if probe:
    print("PARAMS_STR\n" + str(params))
    print("PARAMS_DIR", [a for a in dir(params) if a.startswith("Get")])
    print("CC_DIR", [a for a in dir(cc) if a.startswith("Get")])
    sys.exit(0)

# Exact CCParams values ReBoot passed (data), plus what the Python API exposes on the generated context.
g = {k: getattr(params, k)() for k in ["GetRingDim", "GetMultiplicativeDepth", "GetScalingModSize", "GetFirstModSize",
                                        "GetNumLargeDigits", "GetBatchSize", "GetStandardDeviation", "GetDigitSize",
                                        "GetMaxRelinSkDeg"]}
e = {k: str(getattr(params, k)()) for k in ["GetSecretKeyDist", "GetKeySwitchTechnique", "GetScalingTechnique",
                                             "GetSecurityLevel", "GetMultiplicationTechnique", "GetEncryptionTechnique"]}
py_cc = {"GetRingDimension": cc.GetRingDimension(), "GetCyclotomicOrder": cc.GetCyclotomicOrder()}
try:
    import math
    q = cc.GetModulus()
    py_cc["GetModulus_log2"] = math.log2(int(q)) if not isinstance(q, float) else math.log2(q)
except Exception as ex:  # noqa: BLE001
    py_cc["GetModulus_error"] = str(ex)
json.dump({"ccparams_str": str(params), "ccparams_numeric": g, "ccparams_enums": e, "python_cc": py_cc,
           "construction": "ReBoot's CryptoContext('../config/fhe_config_mlp1.yaml', model=<784-32-16 plain eMLP-1>) as in the subset run; openfhe.GenCryptoContext intercepted, real context generated, constructor stopped before keygen/bootstrap setup"},
          open(out_json, "w"), indent=2, default=str)
print("wrote", out_json)
