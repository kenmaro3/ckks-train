# Security estimate (dev tool)

Runs [malb/lattice-estimator](https://github.com/malb/lattice-estimator) on the ckks-train-ckks presets.
It needs SageMath, is a development tool only, and `_ext/` (Sage, estimator) is not committed
(`_ext/` is gitignored).

Licences: lattice-estimator is LGPLv3+ and SageMath is GPLv3. Both are run-time development
tools only. They live in the gitignored `_ext/`, are never committed, vendored or linked into any
hekit crate, and the scripts here only import the estimator when run.

## Setup (macOS arm64, no system Sage)

    mkdir -p _ext/bin && curl -sL https://micro.mamba.pm/api/micromamba/osx-arm64/latest -o _ext/mm.tar.bz2
    tar -xjf _ext/mm.tar.bz2 -C _ext bin/micromamba
    MAMBA_ROOT_PREFIX=$PWD/_ext/mamba _ext/bin/micromamba create -y -p _ext/sage-env \
        -c conda-forge --override-channels sage python=3.12      # ~20 min
    git -C _ext clone https://github.com/malb/lattice-estimator.git
    git -C _ext/lattice-estimator checkout ae45d740ceb138af87a4f92edc592a0ab0d4f43f

## Run

    cargo run -p ckks-train-ckks --example export_params > params.json
    _ext/sage-env/bin/python estimate.py                       # BDGL16 cost model
    _ext/sage-env/bin/python estimate.py --cost-model matzov   # estimator default
    _ext/sage-env/bin/python estimate.py --cost-model adps16   # Core-SVP
    _ext/sage-env/bin/python estimate.py --rough               # crude screening only

Results and interpretation: `docs/security.md`.
Options:
- `--m oo`: unbounded samples.
- `--hamming-weight h`: sparse secret; enables the hybrid attacks, which are slow.
- `--only n16_boot,n16_boot_sse_to_sparse`: run a subset of the exported entries.
- `--attacks usvp,bdd_mitm_hybrid,...`: run a subset of the attacks. Besides `LWE.estimate`'s
  six, there are the EJK20 dual hybrids `dual_hybrid_ejk20` (exhaustive guessing) and
  `dual_mitm_hybrid` (MITM guessing).
- `--mitm-opt analytical|numerical`: the MITM solver of `dual_mitm_hybrid`.
- `--dense-hybrids`: also run the hybrids on the dense default secret.
- `--primal-hybrid-zeta 0,20,...`: evaluate `bdd_hybrid` and `bdd_mitm_hybrid` at fixed
  guessing dimensions. The full primal-hybrid optimiser does not finish at log PQ 1732.
- `--dual-opt-step k`: `opt_step` of the EJK20 optimisers. Higher is more robust and slower.

`reboot_params.json` holds ReBoot's captured CKKS parameters (R-REB-4) for the same models. Its
secret is a pair `[p, m]`, exactly p coefficients +1 and m coefficients −1. Run it with
`estimate.py reboot_params.json ...` and `dual_mitm_scan.py reboot_main reboot_params.json ...`.

`params.json` also holds `n16_boot_sse_to_sparse`: the SSE switching key s -> s' of n16_boot, an
RLWE sample under the h' = 32 ephemeral key at modulus q_0 * P (docs/security.md, Task 39).

`dual_mitm_scan.py NAME --cost-model M` grid-scans the EJK20 dual hybrid over the splitting
dimension and h1. The estimator's own optimiser for it depends on `opt_step`, so binding sparse
figures come from this scan (docs/security.md, Task 39). Options:
- `--solver mitm|exhaustive`;
- `--mitm-opt analytical|numerical`;
- the grid: `--zeta-min`, `--zeta-max`, `--step`, `--refine`.

For example:

    _ext/sage-env/bin/python dual_mitm_scan.py n16_boot_sse_to_sparse --cost-model bdgl16 \
        --mitm-opt numerical --zeta-min 56000 --zeta-max 61500 --step 250 --refine 50
