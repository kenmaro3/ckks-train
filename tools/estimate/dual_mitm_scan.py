"""Grid scan of the EJK20 dual hybrid for a sparse-secret entry (dev tool; needs SageMath).

`lwe_dual.dual_hybrid` searches the splitting dimension zeta with a local-minimum search whose
result depends on `opt_step` (the cost is not unimodal in zeta). This script evaluates the same
inner optimisation (`DH.optimize_blocksize` for every h1) on an explicit zeta grid, then refines
around the best point, and prints the minimum. The guessing solver is either the MITM search
(`--solver mitm`, the "dual_mitm_hybrid" of estimate.py, with `--mitm-opt analytical|numerical`)
or exhaustive search (`--solver exhaustive`, the "dual_hybrid_ejk20" of estimate.py).

Usage (from this directory):
    _ext/sage-env/bin/python dual_mitm_scan.py NAME [--cost-model bdgl16|matzov|adps16]
        [--solver mitm|exhaustive] [--mitm-opt analytical|numerical]
        [--zeta-min A] [--zeta-max B] [--step S] [--refine R] [params.json]
"""
import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_ext" / "lattice-estimator"))

from estimator import LWE, ND  # noqa: E402
from estimator.cost import Cost  # noqa: E402
from estimator.conf import mitm_opt as mitm_opt_default  # noqa: E402
from estimator.lwe_dual import DH  # noqa: E402
from estimator.lwe_guess import exhaustive_search, mitm  # noqa: E402

from estimate import cost_model, secret_dist  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("params", nargs="?", default=str(HERE / "params.json"))
    ap.add_argument("--cost-model", default="bdgl16", choices=["bdgl16", "matzov", "adps16"])
    ap.add_argument("--zeta-min", type=int, default=40000)
    ap.add_argument("--zeta-max", type=int, default=65000)
    ap.add_argument("--step", type=int, default=1000)
    ap.add_argument("--refine", type=int, default=50)
    ap.add_argument("--h1-max", type=int, default=None, help="largest h1 tried (default h)")
    ap.add_argument("--solver", default="mitm", choices=["mitm", "exhaustive"])
    ap.add_argument("--mitm-opt", default=mitm_opt_default, choices=["analytical", "numerical"],
                    help=f"MITM solver mode (estimator default {mitm_opt_default})")
    args = ap.parse_args()
    p = next(p for p in json.load(open(args.params))["presets"] if p["name"] == args.name)
    n = p["n"]
    params = LWE.Parameters(n=n, q=2 ** math.ceil(p["log_pq"]), Xs=secret_dist(p["secret"], n),
                            Xe=ND.DiscreteGaussian(p["sigma"]), m=n).normalize()
    h = params.Xs.hamming_weight
    # As DualHybrid.__call__ does before optimising.
    Cost.register_impermanent(rop=True, mem=False, red=True, beta=False, delta=False, m=True,
                              d=False, zeta=False, t=False, h1=False)
    if args.solver == "mitm":
        solver = partial(mitm, optimization=args.mitm_opt)
        # h1 = 0 (every guessed coordinate zero) has no MITM split: split_balanced divides by
        # zero. That case is the exhaustive solver's h1 = 0 point (--solver exhaustive).
        h1_lo = 1
    else:
        solver = exhaustive_search
        h1_lo = 0
    rc = cost_model(args.cost_model)

    def at(zeta):
        best = None
        # Points the estimator cannot cost are skipped.
        for h1 in range(max(h1_lo, h - (n - zeta)), min(zeta, h, args.h1_max or h) + 1):
            try:
                c = DH.optimize_blocksize(solver=solver, params=params, zeta=zeta, h1=h1,
                                          success_probability=0.99, red_cost_model=rc,
                                          log_level=5)
            except (ZeroDivisionError, ValueError, KeyError):
                continue
            if best is None or c["rop"] < best["rop"]:
                best = c
        return best

    def show(zeta, c):
        if c is None:
            print(f"zeta {zeta:6d}: not costed", flush=True)
            return
        print(f"zeta {zeta:6d}: log2 rop {math.log2(float(c['rop'])):7.2f}  beta {c.get('beta')}"
              f"  h1 {c.get('h1')}  d {c.get('d')}", flush=True)

    print(f"=== {args.name}: n={n} log2(q)={math.ceil(p['log_pq'])} secret={p['secret']} "
          f"cost={args.cost_model} solver={args.solver}"
          f"{'/' + args.mitm_opt if args.solver == 'mitm' else ''} zeta grid {args.zeta_min}..{args.zeta_max} step {args.step}",
          flush=True)
    t = time.time()
    grid = {}
    for zeta in range(args.zeta_min, args.zeta_max + 1, args.step):
        grid[zeta] = at(zeta)
        show(zeta, grid[zeta])
    grid = {z: c for z, c in grid.items() if c is not None}
    z0 = min(grid, key=lambda z: grid[z]["rop"])
    lo, hi = max(0, z0 - args.step), min(n, z0 + args.step)
    print(f"--- refine around {z0} in [{lo}, {hi}] step {args.refine}", flush=True)
    for zeta in range(lo, hi + 1, args.refine):
        if zeta not in grid:
            c = at(zeta)
            show(zeta, c)
            if c is not None:
                grid[zeta] = c
    z = min(grid, key=lambda z: grid[z]["rop"])
    print(f"MIN zeta {z}: log2 rop {math.log2(float(grid[z]['rop'])):.2f} ({time.time() - t:.0f} s)",
          flush=True)
    print(repr(grid[z]))


if __name__ == "__main__":
    main()
