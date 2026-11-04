# Security estimate (dev tool)

Runs [malb/lattice-estimator](https://github.com/malb/lattice-estimator) on the hekit-ckks presets.
It needs SageMath, is a development tool only, and nothing from it is vendored or committed
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
    _ext/sage-env/bin/python estimate.py --cost-model default  # estimator default (lower = more conservative)
    _ext/sage-env/bin/python estimate.py --rough               # crude screening only

Results and interpretation: `docs/security.md`.
