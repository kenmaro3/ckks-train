"""Run malb/lattice-estimator on the hekit-ckks presets (dev tool; needs SageMath).

Usage (from this directory):
    _ext/sage-env/bin/python estimate.py [--rough] [--cost-model bdgl16|matzov|adps16]
                                         [--m n|oo] [--only NAME[,NAME...]] [params.json]

Modelling choices:
  * n = N, q = 2^ceil(log PQ): the attacker's modulus for RLWE key-switching keys is Q*P (a
    power-of-two stand-in; the estimate depends on q essentially only through log q).
  * secret "ternary_default" = hekit's sampler (-1/0/1 with prob 1/4, 1/2, 1/4), modelled as
    SparseTernary(n/4, n/4, n); an integer secret = exact Hamming weight h (h/2 of each sign).
  * error = discrete Gaussian, sigma 3.2; m = n samples (or unbounded with --m oo).
  * cost models: bdgl16 = asymptotic sieve 8d*2^(0.292*beta+16.4) (HE Standard "BKZ.sieve"),
    matzov = estimator default, adps16 = Core-SVP (2^(0.292*beta), no lower-order terms).
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_ext" / "lattice-estimator"))

from estimator import LWE, ND, RC  # noqa: E402
from estimator.lwe_dual import dual_hybrid as dual_hybrid_ejk20  # noqa: E402
from sage.all import oo  # noqa: E402

# Attack families reported: uSVP, BDD and (hybrid) dual. Arora-GB and BKW are skipped (assumed not
# competitive at these dimensions, not checked here, and very slow); the hybrid/MITM variants are skipped for the
# dense default secret but enabled when a Hamming weight is set (sparse secrets favour them).
# LWE.estimate at the pinned commit has no "dual_mitm_hybrid" entry: its "dual_hybrid" is MATZOV's dual
# (LWE.dual_hybrid = lwe_dual.matzov). For sparse secrets the EJK20 dual hybrid with the MITM guessing
# step, lwe_dual.dual_hybrid(mitm_optimization=True), is run separately as "dual_mitm_hybrid".
ALL_ATTACKS = ("usvp", "bdd", "bdd_hybrid", "bdd_mitm_hybrid", "dual", "dual_hybrid", "dual_mitm_hybrid")
DENY_DENSE = ("arora-gb", "bkw", "bdd_hybrid", "bdd_mitm_hybrid", "dual_mitm_hybrid")
DENY_SPARSE = ("arora-gb", "bkw")


def secret_dist(spec, n):
    if spec == "ternary_default":
        return ND.SparseTernary(n // 4, n // 4, n)
    h = int(spec)
    return ND.SparseTernary(h // 2, h - h // 2, n)


def cost_model(name):
    return {"bdgl16": RC.BDGL16, "matzov": RC.MATZOV, "adps16": RC.ADPS16}[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params", nargs="?", default=str(HERE / "params.json"))
    ap.add_argument("--rough", action="store_true")
    ap.add_argument("--cost-model", default="bdgl16", choices=["bdgl16", "matzov", "adps16"])
    ap.add_argument("--m", default="n", choices=["n", "oo"])
    ap.add_argument("--hamming-weight", type=int, help="override the secret with weight h")
    ap.add_argument("--only", help="comma-separated preset names to run (default: all)")
    ap.add_argument("--dual-opt-step", type=int, default=8,
                    help="opt_step of the dual MITM hybrid optimizer (estimator default 8; smaller is "
                    "more robust and slower)")
    ap.add_argument("--attacks", help=f"comma-separated subset of {','.join(ALL_ATTACKS)} "
                    "(default: all that apply to the secret)")
    args = ap.parse_args()
    presets = json.load(open(args.params))["presets"]
    if args.only:
        names = args.only.split(",")
        unknown = set(names) - {p["name"] for p in presets}
        if unknown:
            sys.exit(f"unknown preset(s): {sorted(unknown)}")
        presets = [p for p in presets if p["name"] in names]
    summary = {}
    for p in presets:
        n, log_pq = p["n"], p["log_pq"]
        spec = args.hamming_weight if args.hamming_weight else p["secret"]
        q = 2 ** math.ceil(log_pq)
        params = LWE.Parameters(n=n, q=q, Xs=secret_dist(spec, n),
                                Xe=ND.DiscreteGaussian(p["sigma"]),
                                m=n if args.m == "n" else oo)
        print(f"=== {p['name']}: n={n} log2(q)={math.ceil(log_pq)} secret={spec} m={args.m} "
              f"cost={'rough' if args.rough else args.cost_model}", flush=True)
        if args.rough:
            res = LWE.estimate.rough(params)
        else:
            deny = DENY_DENSE if spec == "ternary_default" else DENY_SPARSE
            if args.attacks:
                keep = args.attacks.split(",")
                deny = tuple(set(deny) | (set(ALL_ATTACKS) - set(keep)))
            res = {}
            if set(ALL_ATTACKS) - {"dual_mitm_hybrid"} - set(deny):
                res = LWE.estimate(params, red_cost_model=cost_model(args.cost_model),
                                   deny_list=deny + ("dual_mitm_hybrid",))
            if "dual_mitm_hybrid" not in deny:
                res["dual_mitm_hybrid"] = dual_hybrid_ejk20(
                    params, red_cost_model=cost_model(args.cost_model), mitm_optimization=True,
                    opt_step=args.dual_opt_step)
        for k, v in res.items():
            print(f"{k:20s}:: {v}", flush=True)
        bits = {k: math.log2(float(v["rop"])) for k, v in res.items()}
        summary[p["name"]] = bits
        print("min rop (bits):", {k: round(v, 1) for k, v in bits.items()},
              "->", round(min(bits.values()), 1), flush=True)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
