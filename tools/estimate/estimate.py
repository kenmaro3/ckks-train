"""Run malb/lattice-estimator on the hekit-ckks presets (dev tool; needs SageMath).

Usage (from this directory):
    _ext/sage-env/bin/python estimate.py [--rough] [--cost-model bdgl16|default] [params.json]

The attacker's modulus for RLWE key-switching keys is Q*P, so q = 2^log_pq (a power-of-two
modulus is used; the estimate depends on q essentially only through log q).
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_ext" / "lattice-estimator"))

from estimator import LWE, ND, RC  # noqa: E402


# Only the three attack families asked for: uSVP, BDD and (hybrid) dual. The remaining
# estimator attacks (Arora-GB, BKW, hybrid/MITM variants) are skipped: slow and never the minimum
# for these parameters.
DENY = ("arora-gb", "bkw", "bdd_hybrid", "bdd_mitm_hybrid", "dual_mitm_hybrid")


def secret_dist(spec, n):
    if spec == "ternary_uniform":
        return ND.Uniform(-1, 1)
    h = int(spec)
    return ND.SparseTernary(n, h // 2, h - h // 2)  # h/2 ones, rest minus ones; weight h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params", nargs="?", default=str(HERE / "params.json"))
    ap.add_argument("--rough", action="store_true")
    ap.add_argument("--cost-model", default="bdgl16", choices=["bdgl16", "default"])
    args = ap.parse_args()
    presets = json.load(open(args.params))["presets"]
    kw = {} if args.cost_model == "default" else {"red_cost_model": RC.BDGL16}
    summary = {}
    for p in presets:
        n, log_pq = p["n"], p["log_pq"]
        q = 2 ** int(round(log_pq))
        params = LWE.Parameters(n=n, q=q, Xs=secret_dist(p["secret"], n),
                                Xe=ND.DiscreteGaussian(p["sigma"]), m=n)
        print(f"=== {p['name']}: n={n} log2(q)={log_pq:.2f} secret={p['secret']}", flush=True)
        print(params, flush=True)
        if args.rough:
            res = LWE.estimate.rough(params)
        else:
            res = LWE.estimate(params, deny_list=DENY, **kw)
        print(res, flush=True)
        bits = {k: math.log2(float(v["rop"])) for k, v in res.items()}
        summary[p["name"]] = bits
        print("min rop (bits):", {k: round(v, 1) for k, v in bits.items()},
              "->", round(min(bits.values()), 1), flush=True)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
