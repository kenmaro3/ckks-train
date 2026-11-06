"""Run malb/lattice-estimator on the ckks-train-ckks presets (dev tool; needs SageMath).

Usage (from this directory):
    _ext/sage-env/bin/python estimate.py [--rough] [--cost-model bdgl16|matzov|adps16]
                                         [--m n|oo] [--only NAME[,NAME...]] [--attacks A,...]
                                         [--dense-hybrids] [--primal-hybrid-zeta Z,...]
                                         [--mitm-opt analytical|numerical] [params.json]

Modelling choices:
  * n = N, q = 2^ceil(log PQ): the attacker's modulus for RLWE key-switching keys is Q*P (a
    power-of-two stand-in; the estimate depends on q essentially only through log q).
  * secret "ternary_default" = hekit's sampler (-1/0/1 with prob 1/4, 1/2, 1/4), modelled as
    SparseTernary(n/4, n/4, n); an integer secret = exact Hamming weight h (h/2 of each sign);
    a pair [p, m] = exactly p coefficients +1 and m coefficients -1 (reboot_params.json).
  * error = discrete Gaussian, sigma from params.json (ckks_train_ckks::keys::ERROR_STD_DEV = 3.2);
    m = n samples (or unbounded with --m oo).
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
# competitive at these dimensions, not checked here, and very slow).
#
# At the pinned commit LWE.estimate's "dual_hybrid" is MATZOV's dual (LWE.dual_hybrid =
# lwe_dual.matzov), which does not guess secret coordinates. The EJK20 dual hybrid, which guesses
# zeta coordinates (h1 of them nonzero for a sparse secret), is run separately in two variants:
#   "dual_hybrid_ejk20": lwe_dual.dual_hybrid(mitm_optimization=False), exhaustive search;
#   "dual_mitm_hybrid":  lwe_dual.dual_hybrid(mitm_optimization=True), MITM search
#                        (--mitm-opt analytical|numerical, the estimator's two MITM solvers).
# The estimator's dual-hybrid optimisers are not reliable at n = 2^16 (they depend on opt_step);
# for a binding sparse figure use dual_mitm_scan.py, which scans zeta and h1 on a grid.
#
# The hybrids (bdd_hybrid, bdd_mitm_hybrid and the two EJK20 duals) run by default only for a
# Hamming-weight secret. --dense-hybrids enables them for the dense default secret too. The full
# primal_hybrid optimiser does not finish within 30 minutes at log PQ 1732; --primal-hybrid-zeta
# evaluates both primal hybrids at fixed guessing dimensions instead.
ESTIMATE_ATTACKS = ("usvp", "bdd", "bdd_hybrid", "bdd_mitm_hybrid", "dual", "dual_hybrid")
EJK20_ATTACKS = ("dual_hybrid_ejk20", "dual_mitm_hybrid")
ALL_ATTACKS = ESTIMATE_ATTACKS + EJK20_ATTACKS
HYBRIDS = ("bdd_hybrid", "bdd_mitm_hybrid") + EJK20_ATTACKS
SKIPPED = ("arora-gb", "bkw")


def secret_dist(spec, n):
    if spec == "ternary_default":
        return ND.SparseTernary(n // 4, n // 4, n)
    if isinstance(spec, list):  # [#(+1), #(-1)], an exact sign split (reboot_params.json)
        return ND.SparseTernary(int(spec[0]), int(spec[1]), n)
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
                    help="opt_step of the EJK20 dual hybrid optimisers (estimator default 8; per the "
                    "estimator docstring a higher value is more robust and slower)")
    ap.add_argument("--mitm-opt", default="analytical", choices=["analytical", "numerical"],
                    help="MITM solver of dual_mitm_hybrid (estimator default analytical)")
    ap.add_argument("--dense-hybrids", action="store_true",
                    help="also run the hybrid attacks on the dense default secret")
    ap.add_argument("--primal-hybrid-zeta",
                    help="comma-separated fixed guessing dimensions: report bdd_hybrid@Z and "
                    "bdd_mitm_hybrid@Z (LWE.primal_hybrid(zeta=Z)) instead of optimising zeta")
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
            run = set(ALL_ATTACKS)
            if spec == "ternary_default" and not args.dense_hybrids:
                run -= set(HYBRIDS)
            if args.attacks:
                keep = set(args.attacks.split(","))
                bad = keep - set(ALL_ATTACKS)
                if bad:
                    sys.exit(f"unknown attack(s): {sorted(bad)}")
                run &= keep
            rc = cost_model(args.cost_model)
            zetas = ([int(z) for z in args.primal_hybrid_zeta.split(",")]
                     if args.primal_hybrid_zeta else None)
            in_estimate = run & set(ESTIMATE_ATTACKS)
            if zetas is not None:
                in_estimate -= {"bdd_hybrid", "bdd_mitm_hybrid"}
            res = {}
            if in_estimate:
                deny = SKIPPED + tuple(sorted(set(ESTIMATE_ATTACKS) - in_estimate))
                res = LWE.estimate(params, red_cost_model=rc, deny_list=deny)
            if zetas is not None:
                for z in zetas:
                    for name, babai, mitm in (("bdd_hybrid", False, False),
                                              ("bdd_mitm_hybrid", True, True)):
                        if name in run:
                            res[f"{name}@{z}"] = LWE.primal_hybrid(
                                params, zeta=z, babai=babai, mitm=mitm, red_cost_model=rc)
            if "dual_hybrid_ejk20" in run:
                res["dual_hybrid_ejk20"] = dual_hybrid_ejk20(
                    params, red_cost_model=rc, mitm_optimization=False,
                    opt_step=args.dual_opt_step)
            if "dual_mitm_hybrid" in run:
                res["dual_mitm_hybrid"] = dual_hybrid_ejk20(
                    params, red_cost_model=rc, mitm_optimization=args.mitm_opt,
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
