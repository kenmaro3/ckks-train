# Data files committed with hekit-bench

## breast_cancer.csv: Breast Cancer Wisconsin (Diagnostic), UCI WDBC

- Creators: William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (University of
  Wisconsin); donor W. Nick Street (1995).
- Source: UCI Machine Learning Repository, dataset 17,
  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
  (download: https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip,
  sha256 `bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af`, fetched
  2026-11-06).
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/legalcode
- Citation: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer
  Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5DW2B
- Changes: none. The file is `wdbc.data` from the archive above, byte for byte, renamed to
  `breast_cancer.csv`; sha256
  `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a` (checked by
  `hekit_bench::data::load_wdbc`). Format: 569 records, no header; per record the patient id,
  the diagnosis (`M` malignant, 212; `B` benign, 357) and 30 real-valued features.

Fashion-MNIST and MNIST are not committed: `scripts/fetch_fashion_mnist.sh` and
`scripts/fetch_mnist.sh` download them into the gitignored `data/` directory.
