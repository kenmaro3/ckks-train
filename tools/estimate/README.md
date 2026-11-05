# Security estimate (dev tool)

Runs [malb/lattice-estimator](https://github.com/malb/lattice-estimator) on the hekit-ckks presets.
It needs SageMath, is a development tool only, and `_ext/` (Sage, estimator) is not committed
(`_ext/` is gitignored).

## Setup (macOS arm64, no system Sage)

    mkdir -p _ext/bin && curl -sL https://micro.mamba.pm/api/micromamba/osx-arm64/latest -o _ext/mm.tar.bz2
    tar -xjf _ext/mm.tar.bz2 -C _ext bin/micromamba
    MAMBA_ROOT_PREFIX=$PWD/_ext/mamba _ext/bin/micromamba create -y -p _ext/sage-env \
        -c conda-forge --override-channels sage python=3.12      # ~20 min
    git -C _ext clone https://github.com/malb/lattice-estimator.git
    git -C _ext/lattice-estimator checkout ae45d740ceb138af87a4f92edc592a0ab0d4f43f

## Run

    cargo run -p hekit-ckks --example export_params > params.json
    _ext/sage-env/bin/python estimate.py                       # BDGL16 cost model
    _ext/sage-env/bin/python estimate.py --cost-model matzov   # estimator default
    _ext/sage-env/bin/python estimate.py --cost-model adps16   # Core-SVP
    _ext/sage-env/bin/python estimate.py --rough               # crude screening only

Results and interpretation: `docs/security.md`.
Options: `--m oo`, `--hamming-weight h` (sparse secret; enables hybrid attacks, slow),
`--only n16_boot,n16_boot_sse_to_sparse` (run a subset of the exported entries),
`--attacks usvp,bdd_mitm_hybrid,...` (run a subset of the attacks).

`params.json` also holds `n16_boot_sse_to_sparse`: the SSE switching key s -> s' of n16_boot, an
RLWE sample under the h' = 32 ephemeral key at modulus q_0 * P (docs/security.md, Task 39).

`dual_mitm_scan.py NAME --cost-model M` grid-scans the EJK20 dual MITM hybrid over the splitting
dimension (the estimator's own optimiser for it depends on `opt_step`; docs/security.md, Task 39):

    _ext/sage-env/bin/python dual_mitm_scan.py n16_boot_sse_to_sparse --cost-model bdgl16 \
        --zeta-min 40000 --zeta-max 65000 --step 500 --refine 25
