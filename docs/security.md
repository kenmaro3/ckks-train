# Security estimates for the CKKS presets

Route: (b) SageMath 10.9 from conda-forge via micromamba (osx-arm64, Python 3.12.16), plus
malb/lattice-estimator at commit `ae45d740ceb138af87a4f92edc592a0ab0d4f43f` (2026-11-04).
Dev tool only (`tools/estimate/`); `_ext/` (Sage, estimator) is gitignored and not committed.

## Model

LWE with n = N, q = 2^ceil(log PQ) (power-of-two stand-in; the attacker's modulus for
key-switching keys is Q*P). Secret: hekit's default sampler draws -1/0/1 with probability
1/4, 1/2, 1/4, modelled as `ND.SparseTernary(n/4, n/4, n)` (this is not the uniform-ternary
distribution `ND.Uniform(-1,1)`; both give beta = 369 for n16_train). Error:
`ND.DiscreteGaussian(3.2)`. Attacks: usvp, bdd, dual, dual_hybrid (Arora-GB, BKW and the
hybrid/MITM variants skipped for the dense secret). m = n samples; rerunning with m = oo gave
identical numbers (BDGL16 and MATZOV), so m = n is not binding. Values are log2(rop), the
minimum over the four attacks (per-attack numbers in the raw output).

## Results by cost model

| preset | N | log PQ | BDGL16 (= HE Standard sieve) | MATZOV (estimator default) | Core-SVP (ADPS16, 2^0.292b) |
|---|---|---|---|---|---|
| seal_2020 | 2^15 | 480 | 252.9 | 245.6 | 217.8 |
| n16_train | 2^16 | 1592 | 143.9 | 141.2 | 107.7 (< 128) |

Core-SVP is the same primal-uSVP attack at the same beta = 369 with no lower-order terms; it
gives 107.7 bits for n16_train, below 128. The HE Standard explicitly chooses the sieve cost
model over ADPS16, so the preset constants are kept: n16_train is >= 128 under the
standard's methodology (BKZ.sieve) and under the estimator's current default (MATZOV). Anyone
requiring Core-SVP >= 128 for N = 2^16 would need a smaller log PQ (not computed here).

## Consistency with the HE Standard

Homomorphic Encryption Security Standard v1.1 (2018),
https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf :
Table 1 (BKZ.sieve, secret (-1,1), 128-bit classical, n = 32768, p. 26) gives max log Q = 881;
seal_2020 uses log PQ = 480. The standard tabulates only n = 2^10..2^15, so there is no 2018
table value for N = 2^16. The standard's BKZ.sieve cost, 8d * 2^(0.292 beta + 16.4), matches
`RC.BDGL16` in the installed estimator (reduction.py: `svp_repeat = 8d`, `2^(0.292*beta+16.4)`;
I did not check the specific estimator commit the standard used). Hence n16_train is >= 128
under the standard's methodology (143.9 bits), but the 2018 tables do not cover N = 2^16.

## Hamming-weight secrets

`estimate.py --hamming-weight h` (or an integer `secret` in params.json) uses
`ND.SparseTernary(h//2, h - h//2, n)` and enables the hybrid/MITM attacks. Smoke test with
h = 192, rough model only (the full run with the hybrid attacks did not finish in a few
minutes): seal_2020 215.2, n16_train 107.2 (primal beta = 367). The rough model is Core-SVP-like,
so this is not a 128-bit claim; a full sparse-secret estimate is needed when M3 adds sparse
bootstrapping keys. M3 Task 39 did that (next section): `n16_boot_sparse` sets h = 192 and is
**below 128 bits, not used in the paper**.

## Bootstrapping presets (M3, Task 39)

Same route and estimator as above: SageMath 10.9, malb/lattice-estimator commit
`ae45d740ceb138af87a4f92edc592a0ab0d4f43f` (unchanged since M1), `tools/estimate/estimate.py`,
`params.json` from `cargo run -p hekit-ckks --example export_params`. One Sage process at a
time, while Task 33's baseline job ran on the Mac.

### Settings

| setting | N | q primes (L = max level) | k (special primes) | dnum | log PQ | main key |
|---|---|---|---|---|---|---|
| `n16_boot` | 2^16 | 32: [54, 42×16, 34×4, 58×7, 56×4] (L = 31) | 4 × 60 | 8 | 1732 (exact 1731.998) | dense ternary (hekit default) |
| `n16_boot_sparse` | 2^16 | same | same | same | 1732 | ternary, exactly h = 192 nonzero |

Both bootstrap with SSE (h' = 32, `BootParams::default_n16` / `BootParams::sparse_n16`, K = 13)
and leave 16 levels after bootstrapping.

### Which RLWE instances the keys expose

- **(a) Main key, dense, modulus Q·P** (log 1732). The public key, the relinearization key, the
  33 Galois keys and the SSE key s' → s (`SseKeys::to_dense`) are all RLWE samples under the
  dense s at modulus Q·P (or a divisor of it). The s' → s key has the form
  b_i = −a_i·s + e_i + [P]_{q_i}·s' (mod Q·P), one sample per digit (dnum = 8 samples). s' sits
  only in the messages, so as an RLWE instance these are 8 more samples under the dense s, and
  **(a) covers them**. The m = oo runs cover the sample count: all keys together expose about 290
  RLWE samples under s.
- **(b) SSE key s → s' (`SseKeys::to_sparse`), modulus q_0·P** (log 294 = 54 + 4×60,
  exported as `n16_boot_sse_to_sparse`). One part, so one RLWE sample (m = n): b = −a·s' + e + P·s
  (mod q_0·P), under the ephemeral s' with h' = 32. It is modelled as plain RLWE under s' at
  modulus q_0·P, which **is conservative**: it is the same instance with the term P·s given to
  the attacker. (Modulo P alone that term vanishes, and what is left is RLWE under s' at the
  smaller modulus P, which is harder.) s' is dropped after key generation, and nothing else is
  published under s'.
- **(c) `n16_boot_sparse` main key, h = 192, modulus Q·P** (log 1732). This is an alternative
  only. Its SSE key is the same instance (b).

Secret models: dense = `ND.SparseTernary(n/4, n/4, n)` as in M1; a weight h or h' =
`ND.SparseTernary(h/2, h/2, n)`. hekit's fixed-weight sampler (`Rng::ternary_poly` with
`Some(h)`) draws exactly h nonzero positions with independent signs, so the split between +1 and
−1 is binomial instead of exactly h/2. This is the usual modelling. If anything, it favours the
attacker. For a typical hekit key, the guessed part with h1 nonzeros has 2^h1 sign patterns
against the model's C(h1, h1/2). At the binding h1 of 9 or 10 that is about 2 bits more search,
or 1 bit after the MITM square root. Keys with strongly unbalanced signs (|#(+1) − 16| ≥ 8) occur
with probability about 0.7%, and even for them the MITM search shrinks by at most about 1 bit.
Error: σ = 3.2 (`keys::ERROR_STD_DEV`), m = n.

**Attacks.**
- **`LWE.estimate`'s six attacks.** These are usvp, bdd, dual and dual_hybrid, plus the primal
  hybrids bdd_hybrid and bdd_mitm_hybrid. At this commit `dual_hybrid` is MATZOV's dual, which
  does not guess secret coordinates.
- **The EJK20 dual hybrid, which guesses ζ coordinates.** `LWE.estimate` does not include it,
  so `estimate.py` runs it separately in two variants:
  - `dual_hybrid_ejk20`: `lwe_dual.dual_hybrid(mitm_optimization=False)`, exhaustive search over
    the guessed part;
  - `dual_mitm_hybrid`: `mitm_optimization` set to `"analytical"` (the estimator default) or
    `"numerical"`, a MITM search over the guessed part.
- **Which secret gets which attacks.** Sparse (b) and (c) run all eight. Dense (a) runs the four
  non-hybrid ones by default, as in M1. With `--dense-hybrids` it also runs the four hybrids;
  see "Hybrid attacks on the dense key".
- **Skipped.** Arora-GB and BKW are skipped, as in M1.

**Hybrid attacks on the dense key (a).** (a) has the smallest margin, 1.3 bits under MATZOV, so
the hybrids were checked on it as well (`estimate.py --dense-hybrids`). The full `primal_hybrid`
optimiser does not finish within 30 minutes at log PQ 1732, so the primal hybrids were evaluated
at fixed guessing dimensions instead (`--primal-hybrid-zeta`):

| attack on (a) | MATZOV | BDGL16 |
|---|---|---|
| bdd_mitm_hybrid (Babai + MITM), ζ = 0 / 5 / 10 / 20 / 40 | 129.6 / 146.4 / 145.8 / 144.5 / 142.4 | 131.9 / – / – / 146.8 / – |
| bdd_mitm_hybrid, ζ = 60 / 80 / 100 / 120 / 130 / 140 / 150 / 160 | 140.6 / 139.1 / 137.9 / **137.1** / 138.5 / 142.3 / 147.8 / 154.1 | – |
| bdd_mitm_hybrid, ζ = 240 / 320 / 480 / 640 / 1000 | 209.1 / 264.9 / 377.8 / 491.1 / 746.5 | – |
| bdd_hybrid (no Babai, no MITM), ζ = 0 / 5 / 10 / 20 / 40 | 129.3 / 129.6 / 129.6 / 129.6 / 130.6 | 131.5 / – / – / 131.8 / – |
| dual_hybrid_ejk20 (exhaustive) | 130.4 (ζ 0, β 326) | 131.6 |
| dual_mitm_hybrid | ∞ | ∞ |

**None of them lowers the reported 129.3 (MATZOV) or 131.5 (BDGL16).**
- **Exhaustive guessing.** For bdd_hybrid and dual_hybrid_ejk20 the optimum is ζ = 0, which is
  the BDD attack (respectively the plain dual) already reported. Guessing a dense ternary
  coordinate (entropy 1.5 bits) costs more than the dimension it removes.
- **MITM guessing.** The primal MITM hybrid has a local minimum of 137.1 near ζ = 120, 7.8 bits
  above the BDD figure. Past ζ ≈ 140 it rises steeply. The EJK20 dual MITM hybrid returns ∞.

The binding attacks on (a) stay usvp and bdd.

**The estimator's dual-hybrid optimisers are not reliable at n = 2^16.** Their search over the
splitting dimension ζ is a local-minimum search whose result depends on `opt_step`; per the
estimator's docstring, a higher step is more robust and slower. For the dual MITM hybrid on (b):

| optimiser setting | BDGL16 | MATZOV | ADPS16 |
|---|---|---|---|
| step 8 (default) | 136.4 | 136.6 | 179.5 |
| step 16 | 134.6 | 135.0 | – |
| step 2 | 136.4 | 206.7 | 122.3 |
| step 1 | fails inside the estimator | fails | – |

**A default `LWE.estimate`-style result for a sparse key is therefore not trustworthy at this
commit without a grid.** `tools/estimate/dual_mitm_scan.py` evaluates the same inner optimisation
(`DH.optimize_blocksize` for every h1) on an explicit ζ grid and refines around the best point.
**The grid minima are the (b) figures below.** The scans:

- **MITM, analytical solver** (`mitm_opt = "analytical"`, the estimator default). h1 ≥ 1, since
  h1 = 0 has no MITM split; that case is the exhaustive solver's h1 = 0 point. ζ = 40000..65000 in
  steps of 500, refined in steps of 25. Minima: **134.4** (BDGL16, ζ 59000, h1 9, β 153),
  **135.0** (MATZOV, ζ 58500, h1 10, β 173) and 122.0 (ADPS16). The curve is smooth around the
  minimum (±0.1 bit over ±100 in ζ) and rises to both edges.
- **MITM, numerical solver** (`mitm_opt = "numerical"`, the estimator's other mode). About one
  minute per ζ point, so the grid is ζ = 56000..61500 in steps of 250, refined in steps of 50.
  That range holds every solver's minimum (ζ 58450–59450). The costs at its edges are 141–144
  bits, well above the minimum. The numerical solver is noisy: it jumps by up to 2.2 bits between
  neighbouring grid points, and by up to 1.5 bits between steps of 50 near the minimum. Minima:
  **133.8** (BDGL16, ζ 58450, h1 10, β 173) and **134.2** (MATZOV, ζ 58550, h1 10, β 169).
- **Exhaustive solver** (`dual_hybrid_ejk20`, zero-guessing without MITM). h1 ≥ 0,
  ζ = 40000..64000 in steps of 500, refined in steps of 25. Minima: 155.0 (BDGL16, ζ 59275) and
  156.1 (MATZOV, ζ 59450). The default optimiser gives 159.2 / 156.1. It does not bind: about
  20 bits above the MITM variant.

**The conservative (b) figure is the minimum over both MITM solvers: 133.8 (BDGL16) and 134.2
(MATZOV), both from the numerical solver.** The analytical figures alone are 134.4 / 135.0. The
primal hybrids and the (c) runs use the estimator's own optimisers. The reviewer scanned the (b) primal MITM hybrid on a ζ grid
(50000..64000, step 1000) and found 144.5, against the optimiser's 144.2, so the primal optimiser
is not the weak spot for (b). For (c) a grid could only lower a figure that is already below 128.

**`inf` entries.** For (b), usvp, bdd, dual and dual_hybrid return ∞: no block size up to the
estimator's cap β = 1754 (`conf.max_beta`) succeeds. That means at least 0.292 × 1754 = 512 bits
even under Core-SVP, so they do not bind. The binding attacks on (b) guess about 2^15.8 of the
65536 secret coordinates. The primal hybrids guess about 57000 coordinates, which leaves a
dimension of about 8500. The binding dual MITM point (BDGL16, ζ = 59000) leaves n − ζ = 6536
coordinates in a lattice of dimension 13334.

### Results (log2 rop, minimum over attacks; binding attack in brackets)

| key / instance | log modulus | BDGL16 (= HE Standard sieve) | MATZOV (estimator default) | Core-SVP (ADPS16; information only) |
|---|---|---|---|---|
| (a) `n16_boot` dense main key | 1732 | **131.5** [usvp/bdd, β 326] | **129.3** [usvp/bdd, β 326] | 95.2 |
| (b) SSE key s → s', h' = 32 | 294 | **133.8** [dual_mitm_hybrid, numerical MITM grid: ζ 58450, h1 10, β 173; analytical 134.4] | **134.2** [dual_mitm_hybrid, numerical MITM grid: ζ 58550, h1 10, β 169; analytical 135.0] | 122.0 [dual_mitm_hybrid, analytical grid] |
| (c) `n16_boot_sparse` main key, h = 192 | 1732 | **122.2** [dual_mitm_hybrid, β 269] | **121.7** [dual_mitm_hybrid, β 269] | 90.2 |

The rows in more detail:
- (a) at m = oo: identical (131.5 / 129.3), as for n16_train in M1.
- (b), other attacks: bdd_mitm_hybrid gives 144.2 (BDGL16) / 144.4 (MATZOV) / 133.2 (ADPS16),
  and bdd_hybrid 162.9 / 164.0 / 143.2. dual_hybrid_ejk20 (exhaustive) gives 155.0 / 156.1 on its
  grid. The rough model gives ∞. At m = oo every attack is
  unchanged (estimator optimiser: 136.4 / 136.6). For comparison, exhaustive search over the
  fixed-weight secrets costs log2(C(65536, 32)·2^32) = 426.3 bits.
- (c) under BDGL16, per attack: usvp 131.2, bdd 131.0, dual 131.3, dual_hybrid 131.0,
  bdd_hybrid 127.9, bdd_mitm_hybrid 123.4, dual_mitm_hybrid 122.2. Under MATZOV: usvp 129.0, bdd
  128.9, dual 129.9, dual_hybrid 129.8, bdd_hybrid 126.1, bdd_mitm_hybrid 122.1, dual_mitm_hybrid
  121.7. The hybrid attacks take 8.8 bits (BDGL16) and 7.2 bits (MATZOV) off the best
  non-hybrid attack. The rough model gives 94.9.

Core-SVP is reported for information only, as in M1. It is below 128 for every key here: 95.2
for (a), 122.0 for (b) and 90.2 for (c), as it was for n16_train (107.7). The HE Standard uses
the sieve model.

### Verdict

- **`n16_boot` (dense main key + SSE h' = 32) is ≥ 128 bits under BDGL16 and MATZOV for both
  of its key types**: (a) 131.5 / 129.3 and (b) 133.8 / 134.2. The (b) figures are the
  conservative numerical-MITM grid minima; the analytical ones are 134.4 / 135.0. The MATZOV margin
  of (a) is 1.3 bits, and none of the hybrid attacks lowers it. Both are at least 128, so the 58-bit EvalMod primes are kept (no trim), and the chain,
  log PQ 1732, levels_after_boot 16 and the measured precision (17.46 bits, docs/bootstrap.md
  §11.4) stand. **`n16_boot` may be used in the paper.**
- **`n16_boot_sparse` (h = 192) is below 128 bits** (122.2 under BDGL16, 121.7 under
  MATZOV), because of the hybrid attacks. Following the rule "a setting that misses 128 bits is
  not used in the paper" (128 bit を満たさない設定は論文に使わない), **`n16_boot_sparse` is not
  used in the paper**. It stays in the library as a measured alternative: its bootstrap passes
  the gate with 21.12 bits (model 21.0; `n16_boot_keys_bootstrap_sparse`), 3.7 bits more than
  the dense key, because a smaller main key lowers the rescale and key-switch rounding noise.

### ReBoot's parameters under the same models (Task 39 follow-up)

ReBoot's CKKS parameters, as captured from its own OpenFHE 1.2.1 CryptoContext
(`results/quadruped/raw/reboot_crypto_params/params.json`, request R-REB-4):
- N = 2^16;
- Q: 26 towers (54 + 24 × 49 + one 19.58-bit tower that FLEXIBLEAUTOEXT adds), log Q = 1249.585;
- P: 8 × 60, log P = 480, log PQ = 1729.585;
- HYBRID key switching, dnum 3;
- secret SPARSE_TERNARY with h = 192 (97 × +1, 95 × −1), σ = 3.19. The weight 192 is our
  empirical count on one generated key (R-REB-4). It was not confirmed from an OpenFHE constant;
  the sign split does not matter (a (96, 96) split gives the same figure to 0.04 bit);
- requested security level `HEStd_128_classic`.

They were estimated with exactly the tooling and models of (a)–(c):
- **The instance.** It is the RLWE instance at modulus P·Q, where OpenFHE's HYBRID
  key-switching keys live. P = 480 bits covers a digit of α = 9 towers (at most about 446 bits),
  so P·Q is the largest modulus exposed. HYBRID dnum 3 changes only the number of samples (3 per
  key), not the modulus or the secret.
- **The parameters.** q = 2^ceil(log PQ) = 2^1730, the secret is `ND.SparseTernary(97, 95, n)`
  (`tools/estimate/reboot_params.json`), σ = 3.19 and m = n.
- **Which way the modelling choices lean.**
  - Rounding q up to 2^1730 leans *toward* the below-128 claim. The effect is at most 0.07 bit:
    the review's spot-check at the binding point gives 121.84 with q = 2^1730 and 121.88 with the
    exact P·Q.
  - m = n leans *against* the claim. The real attacker has more samples (3 per key and many
    keys), and more samples can only lower the cost.
- **The attacks.** All eight, as for (b) and (c), plus:
  - the primal hybrids at fixed ζ from 0 to 12000;
  - the dual MITM hybrid on a ζ grid with both MITM solvers, with ζ = 2000..16000 (step 1000,
    refined in steps of 250) for the analytical solver and ζ = 6000..12000 (step 500, refined in
    steps of 100) for the numerical one;
  - the EJK20 exhaustive dual on a ζ grid, ζ = 1000..16000.
- **Grid checks.** Every grid minimum is interior: the edges are at least 2.3 bits higher. The
  best h1 (at most 18 for MITM, at most 10 for exhaustive) is far below the h1 caps of the scans
  (40–96 and 48).

| attack on ReBoot's key | BDGL16 | MATZOV |
|---|---|---|
| usvp / bdd / dual / dual_hybrid (MATZOV dual) | 131.3 / 131.3 / 131.3 / 131.3 | 129.1 / 129.0 / 129.9 / 130.1 |
| bdd_hybrid, estimator / fixed-ζ minimum | 128.0 / 128.3 (ζ 3000) | 126.2 / 126.5 (ζ 3000) |
| bdd_mitm_hybrid, estimator / fixed-ζ minimum | 123.5 / 123.4 (ζ 6500) | 122.2 / 122.1 (ζ 6000) |
| dual_hybrid_ejk20 (exhaustive), estimator / grid | 127.9 / 127.8 (ζ 5750, h1 9) | 153.3 / 127.2 (ζ 5500, h1 9) |
| dual_mitm_hybrid, estimator (analytical) | 122.3 | 121.9 |
| dual_mitm_hybrid, analytical grid | 122.1 (ζ 9000, h1 17, β 264) | 121.8 (ζ 9000, h1 17, β 264) |
| dual_mitm_hybrid, numerical grid | **121.8** (ζ 8400, h1 18, β 268) | **121.5** (ζ 8400, h1 18, β 268) |
| **minimum** | **121.8** | **121.5** |

The estimator's own optimiser gives 153.3 for the EJK20 dual under MATZOV, against 127.2 on the
grid: another instance of the unreliable dual-hybrid optimisers.

**Pipeline consistency check against hekit's (c)** (a near-identical instance, so this is not
independent validation). hekit's (c) is the same kind of key, h = 192 at log PQ 1732, σ 3.2. The estimator's optimisers gave it 122.2 / 121.7. On ReBoot's key they give
122.3 / 121.9, which agrees to 0.2 bit: ReBoot's modulus is 2 bits smaller and its σ is
marginally smaller. (c) was not grid-scanned. On ReBoot the grids lower the figure by a further
0.4–0.5 bit, and (c) would move the same way, but (c) is already rejected.

**hekit against ReBoot** (log2 rop, the minimum over all attacks; Core-SVP for information only):

| key | N | log PQ | secret distribution | BDGL16 | MATZOV | Core-SVP |
|---|---|---|---|---|---|---|
| hekit (a): `n16_boot` main key | 2^16 | 1732 (1731.998) | dense ternary (P(±1) = 1/4 each, h ≈ N/2) | **131.5** | **129.3** | 95.2 |
| hekit (b): SSE key s → s' | 2^16 | 294 (q_0·P) | sparse ternary, h' = 32 | **133.8** | **134.2** | 122.0 |
| ReBoot main key | 2^16 | 1730 (1729.585) | sparse ternary, h = 192 (97 / 95) | **121.8** | **121.5** | 90.3 |

The ReBoot row is the MITM tier: operations only, with about 2^104–2^110 memory. Without MITM,
ReBoot is at 127.8 / 126.2 (see the conclusion).

**Conclusion.** Under the same lattice-estimator models used for hekit, **ReBoot's parameters
are below 128 bits under both BDGL16 and MATZOV.** There are two tiers (BDGL16 / MATZOV):
- **Non-hybrid attacks** (usvp, bdd, dual, MATZOV dual): at least 131.3 / 129.0.
- **Hybrids without MITM** (bdd_hybrid, EJK20 exhaustive dual): **127.8 / 126.2**. The BDGL16
  figure is from the EJK20 grid; the bdd_hybrid figures are 128.0 / 126.2. ReBoot is therefore
  already below 128 before any MITM attack: clearly under MATZOV, and just under the boundary
  under BDGL16. The estimator reports memory of 2^113.8 / 2^111.6 for the EJK20 grid minima and
  no memory figure for bdd_hybrid.
- **Hybrids with MITM** (dual MITM, primal MITM): **121.8 / 121.5**, about 6.5 bits short. These
  are operation counts (rop) only. The binding dual MITM needs about 2^104 (analytical solver) to
  2^110 (numerical solver) memory. A reader who discounts memory-heavy MITM attacks still has the
  previous tier.

**Where the shortfall comes from: the configuration and OpenFHE's level check.** The cause is
the configuration, OpenFHE's `SPARSE_TERNARY` secret (h = 192) together with
`HEStd_128_classic`. It is not a security claim by ReBoot's authors.

The coordinator checked how the level check works in OpenFHE v1.2.1's own source (BSD-2):
- `HEStd_128_classic` is a lookup table of the maximum log Q per ring dimension
  (`StdLatticeParm::StandardLatticeParmSets`), with one column per distribution: HEStd_uniform,
  HEStd_error and HEStd_ternary.
- At N = 65536 the 128-bit classic bounds are 1747 for HEStd_ternary and 1749 for HEStd_error.
- `ParamsGenCKKSRNS` maps any non-Gaussian secret, `SPARSE_TERNARY` included, to the HEStd_ternary
  column, which is the dense-ternary one. The check does not model the Hamming weight.
- With HYBRID key switching the bound is applied to log(QP).

ReBoot's log PQ 1729.6 therefore passes the dense-ternary bound (≤ 1747), although its secret is
sparse with h = 192. This is a known limitation of table-based parameter checks when the secret
is sparse.

Suggested wording for the paper: "ReBoot's configuration (OpenFHE `SPARSE_TERNARY`, h = 192,
with `HEStd_128_classic`) estimates at about 122 bits under the same models used for hekit; the
level check does not model the sparse secret."

**Comparison with hekit.** The same h = 192 at hekit's own log PQ 1732 is (c), which is
rejected for the paper for the same reason. hekit's paper settings, (a) and (b), are both at
least 128 under both models.

**Uncertainty.** This is an estimate under the stated models. The lattice-estimator caveats
above, including the rop-only MITM costing, apply to both sides equally.

The raw output is under "Task 39 follow-up (ReBoot), raw output".

## Raw output

### BDGL16, full estimate

```
=== seal_2020: n=32768 log2(q)=480 secret=ternary_default m=n cost=bdgl16
usvp                 :: rop: ≈2^253.5, red: ≈2^253.5, δ: 1.002542, β: 747, d: 64801, tag: usvp
bdd                  :: rop: ≈2^253.5, red: ≈2^253.5, svp: ≈2^244.2, β: 747, η: 779, d: 63812, tag: bdd
dual                 :: rop: ≈2^253.8, mem: ≈2^164.8, m: ≈2^15.0, β: 748, d: 65409, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^252.9, red: ≈2^252.9, guess: ≈2^225.3, β: 745, p: 3, ζ: 0, t: 130, β': 745, N: ≈2^139.7, m: ≈2^15.0
usvp                ::          rop: ≈2^253.5
         red: ≈2^253.5
           δ: 1.002542
           β:      747
           d:    64801
         tag:     usvp
bdd                 ::          rop: ≈2^253.5
         red: ≈2^253.5
         svp: ≈2^244.2
           β:      747
           η:      779
           d:    63812
         tag:      bdd
dual                ::          rop: ≈2^253.8
         mem: ≈2^164.8
           m: ≈2^15.0
           β:      748
           d:    65409
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^252.9
         red: ≈2^252.9
       guess: ≈2^225.3
           β:      745
           p:        3
           ζ:        0
           t:      130
          β':      745
           N: ≈2^139.7
           m: ≈2^15.0
min rop (bits): {'usvp': 253.5, 'bdd': 253.5, 'dual': 253.8, 'dual_hybrid': 252.9} -> 252.9
=== n16_train: n=65536 log2(q)=1592 secret=ternary_default m=n cost=bdgl16
usvp                 :: rop: ≈2^144.1, red: ≈2^144.1, δ: 1.004210, β: 369, d: 126160, tag: usvp
bdd                  :: rop: ≈2^143.9, red: ≈2^143.9, svp: ≈2^138.5, β: 368, η: 417, d: 130704, tag: bdd
dual                 :: rop: ≈2^144.1, mem: ≈2^85.1, m: ≈2^16.0, β: 369, d: 131072, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^143.9, red: ≈2^143.9, guess: ≈2^129.3, β: 368, p: 3, ζ: 0, t: 70, β': 368, N: ≈2^69.5, m: ≈2^16.0
usvp                ::          rop: ≈2^144.1
         red: ≈2^144.1
           δ: 1.004210
           β:      369
           d:   126160
         tag:     usvp
bdd                 ::          rop: ≈2^143.9
         red: ≈2^143.9
         svp: ≈2^138.5
           β:      368
           η:      417
           d:   130704
         tag:      bdd
dual                ::          rop: ≈2^144.1
         mem: ≈2^85.1
           m: ≈2^16.0
           β:      369
           d:   131072
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^143.9
         red: ≈2^143.9
       guess: ≈2^129.3
           β:      368
           p:        3
           ζ:        0
           t:       70
          β':      368
           N: ≈2^69.5
           m: ≈2^16.0
min rop (bits): {'usvp': 144.1, 'bdd': 143.9, 'dual': 144.1, 'dual_hybrid': 143.9} -> 143.9
SUMMARY {"seal_2020": {"usvp": 253.50772845629993, "bdd": 253.48778695508926, "dual": 253.81320153762812, "dual_hybrid": 252.93713537352465}, "n16_train": {"usvp": 144.09289503967787, "bdd": 143.88580686952648, "dual": 144.148, "dual_hybrid": 143.8552902141802}}
```

### MATZOV (estimator default), full estimate

```
=== seal_2020: n=32768 log2(q)=480 secret=ternary_default m=n cost=matzov
usvp                 :: rop: ≈2^245.6, red: ≈2^245.6, δ: 1.002542, β: 747, d: 64801, tag: usvp
bdd                  :: rop: ≈2^245.6, red: ≈2^245.6, svp: ≈2^238.6, β: 747, η: 778, d: 63832, tag: bdd
dual                 :: rop: ≈2^247.0, mem: ≈2^166.0, m: ≈2^15.0, β: 748, d: 65409, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^246.1, red: ≈2^246.1, guess: ≈2^225.3, β: 745, p: 3, ζ: 0, t: 130, β': 751, N: ≈2^139.7, m: ≈2^15.0
usvp                ::          rop: ≈2^245.6
         red: ≈2^245.6
           δ: 1.002542
           β:      747
           d:    64801
         tag:     usvp
bdd                 ::          rop: ≈2^245.6
         red: ≈2^245.6
         svp: ≈2^238.6
           β:      747
           η:      778
           d:    63832
         tag:      bdd
dual                ::          rop: ≈2^247.0
         mem: ≈2^166.0
           m: ≈2^15.0
           β:      748
           d:    65409
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^246.1
         red: ≈2^246.1
       guess: ≈2^225.3
           β:      745
           p:        3
           ζ:        0
           t:      130
          β':      751
           N: ≈2^139.7
           m: ≈2^15.0
min rop (bits): {'usvp': 245.6, 'bdd': 245.6, 'dual': 247.0, 'dual_hybrid': 246.1} -> 245.6
=== n16_train: n=65536 log2(q)=1592 secret=ternary_default m=n cost=matzov
usvp                 :: rop: ≈2^141.2, red: ≈2^141.2, δ: 1.004210, β: 369, d: 126160, tag: usvp
bdd                  :: rop: ≈2^141.2, red: ≈2^141.0, svp: ≈2^137.9, β: 368, η: 417, d: 130704, tag: bdd
dual                 :: rop: ≈2^142.3, mem: ≈2^91.6, m: ≈2^16.0, β: 369, d: 131072, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^142.0, red: ≈2^142.0, guess: ≈2^113.3, β: 368, p: 3, ζ: 0, t: 60, β': 399, N: ≈2^74.9, m: ≈2^16.0
usvp                ::          rop: ≈2^141.2
         red: ≈2^141.2
           δ: 1.004210
           β:      369
           d:   126160
         tag:     usvp
bdd                 ::          rop: ≈2^141.2
         red: ≈2^141.0
         svp: ≈2^137.9
           β:      368
           η:      417
           d:   130704
         tag:      bdd
dual                ::          rop: ≈2^142.3
         mem: ≈2^91.6
           m: ≈2^16.0
           β:      369
           d:   131072
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^142.0
         red: ≈2^142.0
       guess: ≈2^113.3
           β:      368
           p:        3
           ζ:        0
           t:       60
          β':      399
           N: ≈2^74.9
           m: ≈2^16.0
min rop (bits): {'usvp': 141.2, 'bdd': 141.2, 'dual': 142.3, 'dual_hybrid': 142.0} -> 141.2
SUMMARY {"seal_2020": {"usvp": 245.58390424280523, "bdd": 245.57316022934438, "dual": 246.99480438813944, "dual_hybrid": 246.12953771074197}, "n16_train": {"usvp": 141.2322724360252, "bdd": 141.1641155681762, "dual": 142.27972042406688, "dual_hybrid": 141.99269002795725}}
```

### Core-SVP ADPS16, full estimate

```
=== seal_2020: n=32768 log2(q)=480 secret=ternary_default m=n cost=adps16
usvp                 :: rop: ≈2^218.1, red: ≈2^218.1, δ: 1.002542, β: 747, d: 64801, tag: usvp
bdd                  :: rop: ≈2^218.6, red: ≈2^218.1, svp: ≈2^216.7, β: 747, η: 741, d: 65219, tag: bdd
dual                 :: rop: ≈2^218.4, mem: ≈2^164.8, m: ≈2^15.0, β: 748, d: 65409, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^217.8, red: ≈2^217.8, guess: ≈2^159.1, β: 746, p: 4, ζ: 0, t: 70, β': 746, N: ≈2^146.8, m: ≈2^15.0
usvp                ::          rop: ≈2^218.1
         red: ≈2^218.1
           δ: 1.002542
           β:      747
           d:    64801
         tag:     usvp
bdd                 ::          rop: ≈2^218.6
         red: ≈2^218.1
         svp: ≈2^216.7
           β:      747
           η:      741
           d:    65219
         tag:      bdd
dual                ::          rop: ≈2^218.4
         mem: ≈2^164.8
           m: ≈2^15.0
           β:      748
           d:    65409
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^217.8
         red: ≈2^217.8
       guess: ≈2^159.1
           β:      746
           p:        4
           ζ:        0
           t:       70
          β':      746
           N: ≈2^146.8
           m: ≈2^15.0
min rop (bits): {'usvp': 218.1, 'bdd': 218.6, 'dual': 218.4, 'dual_hybrid': 217.8} -> 217.8
=== n16_train: n=65536 log2(q)=1592 secret=ternary_default m=n cost=adps16
usvp                 :: rop: ≈2^107.7, red: ≈2^107.7, δ: 1.004210, β: 369, d: 126160, tag: usvp
bdd                  :: rop: ≈2^107.7, red: ≈2^107.7, svp: ≈2^53.7, β: 369, η: 183, d: 129941, tag: bdd
dual                 :: rop: ≈2^107.7, mem: ≈2^85.1, m: ≈2^16.0, β: 369, d: 131072, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^107.7, red: ≈2^107.7, guess: ≈2^28.2, β: 369, p: 13, ζ: 0, t: 0, β': 369, N: ≈2^18.3, m: ≈2^16.0
usvp                ::          rop: ≈2^107.7
         red: ≈2^107.7
           δ: 1.004210
           β:      369
           d:   126160
         tag:     usvp
bdd                 ::          rop: ≈2^107.7
         red: ≈2^107.7
         svp: ≈2^53.7
           β:      369
           η:      183
           d:   129941
         tag:      bdd
dual                ::          rop: ≈2^107.7
         mem: ≈2^85.1
           m: ≈2^16.0
           β:      369
           d:   131072
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^107.7
         red: ≈2^107.7
       guess: ≈2^28.2
           β:      369
           p:       13
           ζ:        0
           t:        0
          β':      369
           N: ≈2^18.3
           m: ≈2^16.0
min rop (bits): {'usvp': 107.7, 'bdd': 107.7, 'dual': 107.7, 'dual_hybrid': 107.7} -> 107.7
SUMMARY {"seal_2020": {"usvp": 218.124, "bdd": 218.5713074305015, "dual": 218.416, "dual_hybrid": 217.832}, "n16_train": {"usvp": 107.74799999999999, "bdd": 107.74799999999999, "dual": 107.74799999999999, "dual_hybrid": 107.74799999999999}}
```

### LWE.estimate.rough (default model)

```
=== seal_2020: n=32768 log2(q)=480 secret=ternary_default m=n cost=rough
usvp                 :: rop: ≈2^218.1, red: ≈2^218.1, δ: 1.002542, β: 747, d: 64801, tag: usvp
dual_hybrid          :: rop: ≈2^217.8, red: ≈2^217.8, guess: ≈2^159.1, β: 746, p: 4, ζ: 0, t: 70, β': 746, N: ≈2^146.8, m: ≈2^15.0
usvp                ::          rop: ≈2^218.1
         red: ≈2^218.1
           δ: 1.002542
           β:      747
           d:    64801
         tag:     usvp
dual_hybrid         ::          rop: ≈2^217.8
         red: ≈2^217.8
       guess: ≈2^159.1
           β:      746
           p:        4
           ζ:        0
           t:       70
          β':      746
           N: ≈2^146.8
           m: ≈2^15.0
min rop (bits): {'usvp': 218.1, 'dual_hybrid': 217.8} -> 217.8
=== n16_train: n=65536 log2(q)=1592 secret=ternary_default m=n cost=rough
usvp                 :: rop: ≈2^107.7, red: ≈2^107.7, δ: 1.004210, β: 369, d: 126160, tag: usvp
dual_hybrid          :: rop: ≈2^107.7, red: ≈2^107.7, guess: ≈2^28.2, β: 369, p: 13, ζ: 0, t: 0, β': 369, N: ≈2^18.3, m: ≈2^16.0
usvp                ::          rop: ≈2^107.7
         red: ≈2^107.7
           δ: 1.004210
           β:      369
           d:   126160
         tag:     usvp
dual_hybrid         ::          rop: ≈2^107.7
         red: ≈2^107.7
       guess: ≈2^28.2
           β:      369
           p:       13
           ζ:        0
           t:        0
          β':      369
           N: ≈2^18.3
           m: ≈2^16.0
min rop (bits): {'usvp': 107.7, 'dual_hybrid': 107.7} -> 107.7
SUMMARY {"seal_2020": {"usvp": 218.124, "dual_hybrid": 217.832}, "n16_train": {"usvp": 107.74799999999999, "dual_hybrid": 107.74799999999999}}
```

### m = oo, MATZOV

```
=== seal_2020: n=32768 log2(q)=480 secret=ternary_default m=oo cost=matzov
usvp                 :: rop: ≈2^245.6, red: ≈2^245.6, δ: 1.002542, β: 747, d: 64801, tag: usvp
bdd                  :: rop: ≈2^245.6, red: ≈2^245.6, svp: ≈2^238.6, β: 747, η: 778, d: 63832, tag: bdd
dual                 :: rop: ≈2^247.0, mem: ≈2^166.0, m: ≈2^15.0, β: 748, d: 65409, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^246.1, red: ≈2^246.1, guess: ≈2^225.3, β: 745, p: 3, ζ: 0, t: 130, β': 751, N: ≈2^139.7, m: ≈2^15.0
usvp                ::          rop: ≈2^245.6
         red: ≈2^245.6
           δ: 1.002542
           β:      747
           d:    64801
         tag:     usvp
bdd                 ::          rop: ≈2^245.6
         red: ≈2^245.6
         svp: ≈2^238.6
           β:      747
           η:      778
           d:    63832
         tag:      bdd
dual                ::          rop: ≈2^247.0
         mem: ≈2^166.0
           m: ≈2^15.0
           β:      748
           d:    65409
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^246.1
         red: ≈2^246.1
       guess: ≈2^225.3
           β:      745
           p:        3
           ζ:        0
           t:      130
          β':      751
           N: ≈2^139.7
           m: ≈2^15.0
min rop (bits): {'usvp': 245.6, 'bdd': 245.6, 'dual': 247.0, 'dual_hybrid': 246.1} -> 245.6
=== n16_train: n=65536 log2(q)=1592 secret=ternary_default m=oo cost=matzov
usvp                 :: rop: ≈2^141.2, red: ≈2^141.2, δ: 1.004210, β: 369, d: 126160, tag: usvp
bdd                  :: rop: ≈2^141.2, red: ≈2^141.0, svp: ≈2^137.9, β: 368, η: 417, d: 130704, tag: bdd
dual                 :: rop: ≈2^142.3, mem: ≈2^91.6, m: ≈2^16.0, β: 369, d: 131113, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^142.0, red: ≈2^142.0, guess: ≈2^113.3, β: 368, p: 3, ζ: 0, t: 60, β': 399, N: ≈2^74.9, m: ≈2^16.0
usvp                ::          rop: ≈2^141.2
         red: ≈2^141.2
           δ: 1.004210
           β:      369
           d:   126160
         tag:     usvp
bdd                 ::          rop: ≈2^141.2
         red: ≈2^141.0
         svp: ≈2^137.9
           β:      368
           η:      417
           d:   130704
         tag:      bdd
dual                ::          rop: ≈2^142.3
         mem: ≈2^91.6
           m: ≈2^16.0
           β:      369
           d:   131113
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^142.0
         red: ≈2^142.0
       guess: ≈2^113.3
           β:      368
           p:        3
           ζ:        0
           t:       60
          β':      399
           N: ≈2^74.9
           m: ≈2^16.0
min rop (bits): {'usvp': 141.2, 'bdd': 141.2, 'dual': 142.3, 'dual_hybrid': 142.0} -> 141.2
SUMMARY {"seal_2020": {"usvp": 245.58390424280523, "bdd": 245.57316022934438, "dual": 246.99480438813944, "dual_hybrid": 246.12953771074197}, "n16_train": {"usvp": 141.2322724360252, "bdd": 141.1641155681762, "dual": 142.27994791354885, "dual_hybrid": 141.99269002795725}}
```

### m = oo, BDGL16

```
=== seal_2020: n=32768 log2(q)=480 secret=ternary_default m=oo cost=bdgl16
usvp                 :: rop: ≈2^253.5, red: ≈2^253.5, δ: 1.002542, β: 747, d: 64801, tag: usvp
bdd                  :: rop: ≈2^253.5, red: ≈2^253.5, svp: ≈2^244.2, β: 747, η: 779, d: 63812, tag: bdd
dual                 :: rop: ≈2^253.8, mem: ≈2^164.8, m: ≈2^15.0, β: 748, d: 65409, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^252.9, red: ≈2^252.9, guess: ≈2^225.3, β: 745, p: 3, ζ: 0, t: 130, β': 745, N: ≈2^139.7, m: ≈2^15.0
usvp                ::          rop: ≈2^253.5
         red: ≈2^253.5
           δ: 1.002542
           β:      747
           d:    64801
         tag:     usvp
bdd                 ::          rop: ≈2^253.5
         red: ≈2^253.5
         svp: ≈2^244.2
           β:      747
           η:      779
           d:    63812
         tag:      bdd
dual                ::          rop: ≈2^253.8
         mem: ≈2^164.8
           m: ≈2^15.0
           β:      748
           d:    65409
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^252.9
         red: ≈2^252.9
       guess: ≈2^225.3
           β:      745
           p:        3
           ζ:        0
           t:      130
          β':      745
           N: ≈2^139.7
           m: ≈2^15.0
min rop (bits): {'usvp': 253.5, 'bdd': 253.5, 'dual': 253.8, 'dual_hybrid': 252.9} -> 252.9
=== n16_train: n=65536 log2(q)=1592 secret=ternary_default m=oo cost=bdgl16
usvp                 :: rop: ≈2^144.1, red: ≈2^144.1, δ: 1.004210, β: 369, d: 126160, tag: usvp
bdd                  :: rop: ≈2^143.9, red: ≈2^143.9, svp: ≈2^138.5, β: 368, η: 417, d: 130704, tag: bdd
dual                 :: rop: ≈2^144.1, mem: ≈2^85.1, m: ≈2^16.0, β: 369, d: 131113, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^143.9, red: ≈2^143.9, guess: ≈2^129.3, β: 368, p: 3, ζ: 0, t: 70, β': 368, N: ≈2^69.5, m: ≈2^16.0
usvp                ::          rop: ≈2^144.1
         red: ≈2^144.1
           δ: 1.004210
           β:      369
           d:   126160
         tag:     usvp
bdd                 ::          rop: ≈2^143.9
         red: ≈2^143.9
         svp: ≈2^138.5
           β:      368
           η:      417
           d:   130704
         tag:      bdd
dual                ::          rop: ≈2^144.1
         mem: ≈2^85.1
           m: ≈2^16.0
           β:      369
           d:   131113
           ↻:        1
         tag:     dual
dual_hybrid         ::          rop: ≈2^143.9
         red: ≈2^143.9
       guess: ≈2^129.3
           β:      368
           p:        3
           ζ:        0
           t:       70
          β':      368
           N: ≈2^69.5
           m: ≈2^16.0
min rop (bits): {'usvp': 144.1, 'bdd': 143.9, 'dual': 144.1, 'dual_hybrid': 143.9} -> 143.9
SUMMARY {"seal_2020": {"usvp": 253.50772845629993, "bdd": 253.48778695508926, "dual": 253.81320153762812, "dual_hybrid": 252.93713537352465}, "n16_train": {"usvp": 144.09289503967787, "bdd": 143.88580686952648, "dual": 144.14845121190882, "dual_hybrid": 143.8552902141802}}
```

### Hamming weight 192, rough

```
=== seal_2020: n=32768 log2(q)=480 secret=192 m=n cost=rough
usvp                 :: rop: ≈2^216.4, red: ≈2^216.4, δ: 1.002558, β: 741, d: 63123, tag: usvp
dual_hybrid          :: rop: ≈2^215.2, red: ≈2^215.2, guess: ≈2^158.4, β: 737, p: 2, ζ: 0, t: 140, β': 737, N: ≈2^146.5, m: ≈2^15.0
usvp                ::          rop: ≈2^216.4
         red: ≈2^216.4
           δ: 1.002558
           β:      741
           d:    63123
         tag:     usvp
dual_hybrid         ::          rop: ≈2^215.2
         red: ≈2^215.2
       guess: ≈2^158.4
           β:      737
           p:        2
           ζ:        0
           t:      140
          β':      737
           N: ≈2^146.5
           m: ≈2^15.0
min rop (bits): {'usvp': 216.4, 'dual_hybrid': 215.2} -> 215.2
=== n16_train: n=65536 log2(q)=1592 secret=192 m=n cost=rough
usvp                 :: rop: ≈2^107.2, red: ≈2^107.2, δ: 1.004226, β: 367, d: 129457, tag: usvp
dual_hybrid          :: rop: ≈2^107.2, red: ≈2^107.2, guess: ≈2^59.0, β: 367, p: 2, ζ: 0, t: 40, β': 367, N: ≈2^49.4, m: ≈2^16.0
usvp                ::          rop: ≈2^107.2
         red: ≈2^107.2
           δ: 1.004226
           β:      367
           d:   129457
         tag:     usvp
dual_hybrid         ::          rop: ≈2^107.2
         red: ≈2^107.2
       guess: ≈2^59.0
           β:      367
           p:        2
           ζ:        0
           t:       40
          β':      367
           N: ≈2^49.4
           m: ≈2^16.0
min rop (bits): {'usvp': 107.2, 'dual_hybrid': 107.2} -> 107.2
SUMMARY {"seal_2020": {"usvp": 216.37199999999999, "dual_hybrid": 215.20399999999998}, "n16_train": {"usvp": 107.16399999999999, "dual_hybrid": 107.16399999999999}}
```

### Task 39 raw output

Header, one-line result per attack, and the minimum (the expanded per-attack dictionaries are omitted; for dual_mitm_hybrid, which `LWE.estimate` does not print, the `rop` line of its dictionary is kept). Wall-clock time per run in brackets.

#### (a) n16_boot, BDGL16 [27 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=bdgl16
usvp                 :: rop: ≈2^131.5, red: ≈2^131.5, δ: 1.004580, β: 326, d: 125952, tag: usvp
bdd                  :: rop: ≈2^131.5, red: ≈2^131.3, svp: ≈2^128.5, β: 325, η: 383, d: 130661, tag: bdd
dual                 :: rop: ≈2^131.6, mem: ≈2^76.0, m: ≈2^16.0, β: 326, d: 131072, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^131.6, red: ≈2^131.6, guess: ≈2^108.2, β: 326, p: 2, ζ: 0, t: 90, β': 326, N: ≈2^61.7, m: ≈2^16.0
min rop (bits): {'usvp': 131.5, 'bdd': 131.5, 'dual': 131.6, 'dual_hybrid': 131.6} -> 131.5
```

#### (a) n16_boot, MATZOV [23 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=matzov
usvp                 :: rop: ≈2^129.3, red: ≈2^129.3, δ: 1.004580, β: 326, d: 125952, tag: usvp
bdd                  :: rop: ≈2^129.3, red: ≈2^129.3, svp: ≈2^118.5, β: 326, η: 347, d: 125696, tag: bdd
dual                 :: rop: ≈2^130.2, mem: 330, m: ≈2^16.0, β: 329, d: 131072, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^130.1, red: ≈2^130.1, guess: ≈2^113.3, β: 325, p: 3, ζ: 0, t: 60, β': 359, N: ≈2^70.0, m: ≈2^16.0
min rop (bits): {'usvp': 129.3, 'bdd': 129.3, 'dual': 130.2, 'dual_hybrid': 130.1} -> 129.3
```

#### (a) n16_boot, Core-SVP ADPS16 [29 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=adps16
usvp                 :: rop: ≈2^95.2, red: ≈2^95.2, δ: 1.004580, β: 326, d: 125952, tag: usvp
bdd                  :: rop: ≈2^95.2, red: ≈2^95.2, svp: ≈2^41.8, β: 326, η: 142, d: 129122, tag: bdd
dual                 :: rop: ≈2^95.2, mem: ≈2^76.0, m: ≈2^16.0, β: 326, d: 131072, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^95.2, red: ≈2^95.2, guess: ≈2^21.6, β: 326, p: 6, ζ: 0, t: 0, β': 326, N: ≈2^11.6, m: ≈2^16.0
min rop (bits): {'usvp': 95.2, 'bdd': 95.2, 'dual': 95.2, 'dual_hybrid': 95.2} -> 95.2
```

#### (a) n16_boot, BDGL16, m = oo [34 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=oo cost=bdgl16
usvp                 :: rop: ≈2^131.5, red: ≈2^131.5, δ: 1.004580, β: 326, d: 125952, tag: usvp
bdd                  :: rop: ≈2^131.5, red: ≈2^131.3, svp: ≈2^128.5, β: 325, η: 383, d: 130661, tag: bdd
dual                 :: rop: ≈2^131.6, mem: ≈2^76.0, m: ≈2^16.0, β: 326, d: 131133, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^131.6, red: ≈2^131.6, guess: ≈2^108.2, β: 326, p: 2, ζ: 0, t: 90, β': 326, N: ≈2^61.7, m: ≈2^16.0
min rop (bits): {'usvp': 131.5, 'bdd': 131.5, 'dual': 131.6, 'dual_hybrid': 131.6} -> 131.5
```

#### (a) n16_boot, MATZOV, m = oo [35 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=oo cost=matzov
usvp                 :: rop: ≈2^129.3, red: ≈2^129.3, δ: 1.004580, β: 326, d: 125952, tag: usvp
bdd                  :: rop: ≈2^129.3, red: ≈2^129.3, svp: ≈2^118.7, β: 326, η: 348, d: 125685, tag: bdd
dual                 :: rop: ≈2^130.2, mem: 330, m: ≈2^16.0, β: 329, d: 131538, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^130.1, red: ≈2^130.1, guess: ≈2^113.3, β: 325, p: 3, ζ: 0, t: 60, β': 359, N: ≈2^70.0, m: ≈2^16.0
min rop (bits): {'usvp': 129.3, 'bdd': 129.3, 'dual': 130.2, 'dual_hybrid': 130.1} -> 129.3
```

#### (b) SSE key, rough [4 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=rough
min rop (bits): {'usvp': inf, 'dual_hybrid': inf} -> inf
```

#### (b) SSE key, BDGL16 [348 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=bdgl16
bdd_hybrid           :: rop: ≈2^162.9, red: ≈2^162.6, svp: ≈2^160.6, β: 99, η: 26, ζ: ≈2^15.9, |S|: ≈2^32.7, d: 10877, prob: ≈2^-98.6, ↻: ≈2^100.8, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^144.2, red: ≈2^142.7, svp: ≈2^143.6, β: 182, η: 2, ζ: ≈2^15.8, |S|: ≈2^119.3, d: 14856, prob: ≈2^-54.1, ↻: ≈2^56.3, tag: hybrid
min rop (bits): {'usvp': inf, 'bdd': inf, 'bdd_hybrid': 162.9, 'bdd_mitm_hybrid': 144.2, 'dual': inf, 'dual_hybrid': inf} -> 144.2
```

#### (b) SSE key, BDGL16, dual_mitm_hybrid [30 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=bdgl16
dual_mitm_hybrid :: rop: ≈2^136.4
min rop (bits): {'dual_mitm_hybrid': 136.4} -> 136.4
```

#### (b) SSE key, MATZOV [350 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=matzov
bdd_hybrid           :: rop: ≈2^164.0, red: ≈2^163.7, svp: ≈2^161.8, β: 93, η: 20, ζ: ≈2^15.9, |S|: ≈2^32.8, d: 10555, prob: ≈2^-100.0, ↻: ≈2^102.2, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^144.4, red: ≈2^142.8, svp: ≈2^143.8, β: 207, η: 2, ζ: ≈2^15.8, |S|: ≈2^132.9, d: 16059, prob: ≈2^-47.2, ↻: ≈2^49.4, tag: hybrid
min rop (bits): {'usvp': inf, 'bdd': inf, 'bdd_hybrid': 164.0, 'bdd_mitm_hybrid': 144.4, 'dual': inf, 'dual_hybrid': inf} -> 144.4
```

#### (b) SSE key, MATZOV, dual_mitm_hybrid [38 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=matzov
dual_mitm_hybrid :: rop: ≈2^136.6
min rop (bits): {'dual_mitm_hybrid': 136.6} -> 136.6
```

#### (b) SSE key, Core-SVP ADPS16 [388 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=adps16
bdd_hybrid           :: rop: ≈2^143.2, red: ≈2^142.1, svp: ≈2^142.3, β: 151, η: 74, ζ: ≈2^15.8, |S|: ≈2^16.8, d: 13632, prob: ≈2^-95.8, ↻: ≈2^98.0, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^133.2, red: ≈2^131.5, svp: ≈2^132.7, β: 249, η: 2, ζ: ≈2^15.8, |S|: ≈2^91.2, d: 18048, prob: ≈2^-56.6, ↻: ≈2^58.8, tag: hybrid
dual_mitm_hybrid :: rop: ≈2^179.5
min rop (bits): {'usvp': inf, 'bdd': inf, 'bdd_hybrid': 143.2, 'bdd_mitm_hybrid': 133.2, 'dual': inf, 'dual_hybrid': inf, 'dual_mitm_hybrid': 179.5} -> 133.2
```

#### (b) SSE key, dual_mitm_hybrid, BDGL16, opt_step 2 [24 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=bdgl16
dual_mitm_hybrid :: rop: ≈2^136.4
min rop (bits): {'dual_mitm_hybrid': 136.4} -> 136.4
```

#### (b) SSE key, dual_mitm_hybrid, BDGL16, opt_step 4 [27 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=bdgl16
dual_mitm_hybrid :: rop: ≈2^136.4
min rop (bits): {'dual_mitm_hybrid': 136.4} -> 136.4
```

#### (b) SSE key, dual_mitm_hybrid, BDGL16, opt_step 16 [55 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=bdgl16
dual_mitm_hybrid :: rop: ≈2^134.6
min rop (bits): {'dual_mitm_hybrid': 134.6} -> 134.6
```

#### (b) SSE key, dual_mitm_hybrid, MATZOV, opt_step 2 [12 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=matzov
dual_mitm_hybrid :: rop: ≈2^206.7
min rop (bits): {'dual_mitm_hybrid': 206.7} -> 206.7
```

#### (b) SSE key, dual_mitm_hybrid, MATZOV, opt_step 4 [28 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=matzov
dual_mitm_hybrid :: rop: ≈2^136.6
min rop (bits): {'dual_mitm_hybrid': 136.6} -> 136.6
```

#### (b) SSE key, dual_mitm_hybrid, MATZOV, opt_step 16 [63 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=matzov
dual_mitm_hybrid :: rop: ≈2^135.0
min rop (bits): {'dual_mitm_hybrid': 135.0} -> 135.0
```

#### (b) SSE key, dual_mitm_hybrid, ADPS16, opt_step 2 [26 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=adps16
dual_mitm_hybrid :: rop: ≈2^122.3
min rop (bits): {'dual_mitm_hybrid': 122.3} -> 122.3
```

#### (b) SSE key, dual_mitm_scan.py grid, BDGL16

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 cost=bdgl16 zeta grid 40000..65000 step 500
--- refine around 59000 in [58500, 59500] step 25
MIN zeta 59000: log2 rop 134.38 (329 s)
rop: ≈2^134.4, mem: ≈2^73.6, m: ≈2^12.7, k: ≈2^14.9, ↻: ≈2^60.6, β: 153, d: 13334, ζ: ≈2^15.8, h1: 9
```

#### (b) SSE key, dual_mitm_scan.py grid, MATZOV

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 cost=matzov zeta grid 40000..65000 step 500
--- refine around 58500 in [58000, 59000] step 25
MIN zeta 58500: log2 rop 134.98 (429 s)
rop: ≈2^135.0, mem: ≈2^80.6, m: ≈2^12.8, k: ≈2^14.9, ↻: ≈2^54.3, β: 173, d: 14338, ζ: ≈2^15.8, h1: 10
```

#### (b) SSE key, dual_mitm_scan.py grid, ADPS16

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 cost=adps16 zeta grid 40000..65000 step 500
--- refine around 57500 in [57000, 58000] step 25
MIN zeta 57475: log2 rop 122.02 (343 s)
rop: ≈2^122.0, mem: ≈2^59.5, m: ≈2^13.0, k: ≈2^14.8, ↻: ≈2^62.4, β: 216, d: 16421, ζ: ≈2^15.8, h1: 7
```

#### (b) SSE key, BDGL16, m = oo [812 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=oo cost=bdgl16
bdd_hybrid           :: rop: ≈2^162.9, red: ≈2^162.6, svp: ≈2^160.6, β: 99, η: 26, ζ: ≈2^15.9, |S|: ≈2^32.7, d: 10877, prob: ≈2^-98.6, ↻: ≈2^100.8, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^144.2, red: ≈2^142.7, svp: ≈2^143.6, β: 182, η: 2, ζ: ≈2^15.8, |S|: ≈2^119.3, d: 14856, prob: ≈2^-54.1, ↻: ≈2^56.3, tag: hybrid
dual_mitm_hybrid :: rop: ≈2^136.4
min rop (bits): {'usvp': inf, 'bdd': inf, 'bdd_hybrid': 162.9, 'bdd_mitm_hybrid': 144.2, 'dual': inf, 'dual_hybrid': inf, 'dual_mitm_hybrid': 136.4} -> 136.4
```

#### (b) SSE key, MATZOV, m = oo [1048 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=oo cost=matzov
bdd_hybrid           :: rop: ≈2^164.0, red: ≈2^163.7, svp: ≈2^161.8, β: 93, η: 20, ζ: ≈2^15.9, |S|: ≈2^32.8, d: 10555, prob: ≈2^-100.0, ↻: ≈2^102.2, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^144.4, red: ≈2^142.8, svp: ≈2^143.8, β: 207, η: 2, ζ: ≈2^15.8, |S|: ≈2^132.9, d: 16059, prob: ≈2^-47.2, ↻: ≈2^49.4, tag: hybrid
dual_mitm_hybrid :: rop: ≈2^136.6
min rop (bits): {'usvp': inf, 'bdd': inf, 'bdd_hybrid': 164.0, 'bdd_mitm_hybrid': 144.4, 'dual': inf, 'dual_hybrid': inf, 'dual_mitm_hybrid': 136.6} -> 136.6
```

#### (c) n16_boot_sparse h = 192, rough [6 s]

```
=== n16_boot_sparse: n=65536 log2(q)=1732 secret=192 m=n cost=rough
usvp                 :: rop: ≈2^94.9, red: ≈2^94.9, δ: 1.004590, β: 325, d: 125568, tag: usvp
dual_hybrid          :: rop: ≈2^94.9, red: ≈2^94.9, guess: ≈2^19.8, β: 325, p: 6, ζ: 0, t: 0, β': 325, N: 1474.820, m: ≈2^16.0
min rop (bits): {'usvp': 94.9, 'dual_hybrid': 94.9} -> 94.9
```

#### (c) n16_boot_sparse h = 192, BDGL16 [483 s]

```
=== n16_boot_sparse: n=65536 log2(q)=1732 secret=192 m=n cost=bdgl16
usvp                 :: rop: ≈2^131.2, red: ≈2^131.2, δ: 1.004590, β: 325, d: 125568, tag: usvp
bdd                  :: rop: ≈2^131.0, red: ≈2^131.0, svp: ≈2^125.3, β: 324, η: 372, d: 130486, tag: bdd
bdd_hybrid           :: rop: ≈2^127.9, red: ≈2^127.6, svp: ≈2^125.1, β: 299, η: 1, ζ: ≈2^11.8, |S|: ≈2^87.3, d: 123683, prob: 0.250, ↻: 16, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^123.4, red: ≈2^123.1, svp: ≈2^120.9, β: 281, η: 2, ζ: ≈2^12.7, |S|: ≈2^164.9, d: 118287, prob: 0.158, ↻: 27, tag: hybrid
dual                 :: rop: ≈2^131.3, mem: ≈2^75.8, m: ≈2^16.0, β: 325, d: 130857, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^131.0, red: ≈2^131.0, guess: ≈2^107.5, β: 324, p: 2, ζ: 0, t: 90, β': 324, N: ≈2^31.9, m: ≈2^16.0
dual_mitm_hybrid :: rop: ≈2^122.2
min rop (bits): {'usvp': 131.2, 'bdd': 131.0, 'bdd_hybrid': 127.9, 'bdd_mitm_hybrid': 123.4, 'dual': 131.3, 'dual_hybrid': 131.0, 'dual_mitm_hybrid': 122.2} -> 122.2
```

#### (c) n16_boot_sparse h = 192, MATZOV [527 s]

```
=== n16_boot_sparse: n=65536 log2(q)=1732 secret=192 m=n cost=matzov
usvp                 :: rop: ≈2^129.0, red: ≈2^129.0, δ: 1.004590, β: 325, d: 125568, tag: usvp
bdd                  :: rop: ≈2^128.9, red: ≈2^128.8, svp: ≈2^125.4, β: 324, η: 372, d: 130486, tag: bdd
bdd_hybrid           :: rop: ≈2^126.1, red: ≈2^126.1, svp: ≈2^120.8, β: 305, η: 1, ζ: ≈2^11.5, |S|: ≈2^84.3, d: 125332, prob: 0.536, ↻: 6, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^122.1, red: ≈2^122.0, svp: ≈2^117.5, β: 279, η: 2, ζ: ≈2^12.7, |S|: ≈2^156.0, d: 117725, prob: 0.078, ↻: 57, tag: hybrid
dual                 :: rop: ≈2^129.9, mem: 329, m: ≈2^16.0, β: 328, d: 131072, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^129.8, red: ≈2^129.8, guess: ≈2^107.5, β: 324, p: 2, ζ: 0, t: 90, β': 358, N: ≈2^31.9, m: ≈2^16.0
dual_mitm_hybrid :: rop: ≈2^121.7
min rop (bits): {'usvp': 129.0, 'bdd': 128.9, 'bdd_hybrid': 126.1, 'bdd_mitm_hybrid': 122.1, 'dual': 129.9, 'dual_hybrid': 129.8, 'dual_mitm_hybrid': 121.7} -> 121.7
```

#### (c) n16_boot_sparse h = 192, Core-SVP ADPS16 [544 s]

```
=== n16_boot_sparse: n=65536 log2(q)=1732 secret=192 m=n cost=adps16
usvp                 :: rop: ≈2^94.9, red: ≈2^94.9, δ: 1.004590, β: 325, d: 125568, tag: usvp
bdd                  :: rop: ≈2^94.9, red: ≈2^94.9, svp: ≈2^68.3, β: 325, η: 233, d: 126762, tag: bdd
bdd_hybrid           :: rop: ≈2^94.0, red: ≈2^94.0, svp: ≈2^69.2, β: 322, η: 117, ζ: 398, |S|: ≈2^33.9, d: 130054, prob: 0.990, ↻: 1, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^91.8, red: ≈2^91.7, svp: ≈2^87.9, β: 298, η: 2, ζ: ≈2^12.0, |S|: ≈2^98.9, d: 122970, prob: 0.168, ↻: 25, tag: hybrid
dual                 :: rop: ≈2^94.9, mem: ≈2^75.8, m: ≈2^16.0, β: 325, d: 130857, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^94.9, red: ≈2^94.9, guess: ≈2^19.8, β: 325, p: 6, ζ: 0, t: 0, β': 325, N: 1474.820, m: ≈2^16.0
dual_mitm_hybrid :: rop: ≈2^90.2
min rop (bits): {'usvp': 94.9, 'bdd': 94.9, 'bdd_hybrid': 94.0, 'bdd_mitm_hybrid': 91.8, 'dual': 94.9, 'dual_hybrid': 94.9, 'dual_mitm_hybrid': 90.2} -> 90.2
```

### Task 39 fix round 1, raw output

Same format as above: header, one line per attack (or the scan's refine, MIN and final cost lines), and the minimum. Wall-clock time per run in brackets.

#### (a) n16_boot, EJK20 dual hybrids, MATZOV [184 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=matzov
dual_mitm_hybrid :: rop: ≈2^inf
min rop (bits): {'dual_hybrid_ejk20': 130.4, 'dual_mitm_hybrid': inf} -> 130.4
```

#### (a) n16_boot, EJK20 dual hybrids, BDGL16 [164 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=bdgl16
dual_mitm_hybrid :: rop: ≈2^inf
min rop (bits): {'dual_hybrid_ejk20': 131.6, 'dual_mitm_hybrid': inf} -> 131.6
```

#### (a) n16_boot, primal hybrids at fixed ζ 0, 20, MATZOV [90 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=matzov
bdd_hybrid@0 :: rop: ≈2^129.3
bdd_mitm_hybrid@0 :: rop: ≈2^129.6
bdd_hybrid@20 :: rop: ≈2^129.6
bdd_mitm_hybrid@20 :: rop: ≈2^144.5
min rop (bits): {'bdd_hybrid@0': 129.3, 'bdd_mitm_hybrid@0': 129.6, 'bdd_hybrid@20': 129.6, 'bdd_mitm_hybrid@20': 144.5} -> 129.3
```

#### (a) n16_boot, primal hybrids at fixed ζ 0, 20, BDGL16 [121 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=bdgl16
bdd_hybrid@0 :: rop: ≈2^131.5
bdd_mitm_hybrid@0 :: rop: ≈2^131.9
bdd_hybrid@20 :: rop: ≈2^131.8
bdd_mitm_hybrid@20 :: rop: ≈2^146.8
min rop (bits): {'bdd_hybrid@0': 131.5, 'bdd_mitm_hybrid@0': 131.9, 'bdd_hybrid@20': 131.8, 'bdd_mitm_hybrid@20': 146.8} -> 131.5
```

#### (a) n16_boot, primal hybrids at fixed ζ 5, 10, 40, MATZOV [191 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=matzov
bdd_hybrid@5 :: rop: ≈2^129.6
bdd_mitm_hybrid@5 :: rop: ≈2^146.4
bdd_hybrid@10 :: rop: ≈2^129.6
bdd_mitm_hybrid@10 :: rop: ≈2^145.8
bdd_hybrid@40 :: rop: ≈2^130.6
bdd_mitm_hybrid@40 :: rop: ≈2^142.4
min rop (bits): {'bdd_hybrid@5': 129.6, 'bdd_mitm_hybrid@5': 146.4, 'bdd_hybrid@10': 129.6, 'bdd_mitm_hybrid@10': 145.8, 'bdd_hybrid@40': 130.6, 'bdd_mitm_hybrid@40': 142.4} -> 129.6
```

#### (a) n16_boot, primal MITM hybrid at fixed ζ 60..1000, MATZOV [300 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=matzov
bdd_mitm_hybrid@60 :: rop: ≈2^140.6
bdd_mitm_hybrid@80 :: rop: ≈2^139.1
bdd_mitm_hybrid@120 :: rop: ≈2^137.1
bdd_mitm_hybrid@160 :: rop: ≈2^154.1
bdd_mitm_hybrid@240 :: rop: ≈2^209.1
bdd_mitm_hybrid@320 :: rop: ≈2^264.9
bdd_mitm_hybrid@480 :: rop: ≈2^377.8
bdd_mitm_hybrid@640 :: rop: ≈2^491.1
bdd_mitm_hybrid@1000:: rop: ≈2^746.5
min rop (bits): {'bdd_mitm_hybrid@60': 140.6, 'bdd_mitm_hybrid@80': 139.1, 'bdd_mitm_hybrid@120': 137.1, 'bdd_mitm_hybrid@160': 154.1, 'bdd_mitm_hybrid@240': 209.1, 'bdd_mitm_hybrid@320': 264.9, 'bdd_mitm_hybrid@480': 377.8, 'bdd_mitm_hybrid@640': 491.1, 'bdd_mitm_hybrid@1000': 746.5} -> 137.1
```

#### (a) n16_boot, primal MITM hybrid at fixed ζ 100..150, MATZOV [116 s]

```
=== n16_boot: n=65536 log2(q)=1732 secret=ternary_default m=n cost=matzov
bdd_mitm_hybrid@100 :: rop: ≈2^137.9
bdd_mitm_hybrid@130 :: rop: ≈2^138.5
bdd_mitm_hybrid@140 :: rop: ≈2^142.3
bdd_mitm_hybrid@150 :: rop: ≈2^147.8
min rop (bits): {'bdd_mitm_hybrid@100': 137.9, 'bdd_mitm_hybrid@130': 138.5, 'bdd_mitm_hybrid@140': 142.3, 'bdd_mitm_hybrid@150': 147.8} -> 137.9
```

#### (b) SSE key, dual_hybrid_ejk20 (exhaustive), default optimiser, BDGL16 [27 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=bdgl16
min rop (bits): {'dual_hybrid_ejk20': 159.2} -> 159.2
```

#### (b) SSE key, dual_hybrid_ejk20 (exhaustive), default optimiser, MATZOV [36 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 m=n cost=matzov
min rop (bits): {'dual_hybrid_ejk20': 156.1} -> 156.1
```

#### (b) SSE key, dual_mitm_scan.py --solver exhaustive, BDGL16 [315 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 cost=bdgl16 solver=exhaustive zeta grid 40000..64000 step 500
--- refine around 59000 in [58500, 59500] step 25
MIN zeta 59275: log2 rop 155.02 (310 s)
rop: ≈2^155.0, mem: ≈2^70.3, m: ≈2^12.6, β: 132, d: 12525, ↻: ≈2^82.6, ζ: ≈2^15.9, h1: 4
```

#### (b) SSE key, dual_mitm_scan.py --solver exhaustive, MATZOV [454 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 cost=matzov solver=exhaustive zeta grid 40000..64000 step 500
--- refine around 59500 in [59000, 60000] step 25
MIN zeta 59450: log2 rop 156.07 (448 s)
rop: ≈2^156.1, mem: ≈2^70.4, m: ≈2^12.6, β: 125, d: 12168, ↻: ≈2^83.7, ζ: ≈2^15.9, h1: 4
```

#### (b) SSE key, dual_mitm_scan.py --mitm-opt numerical, BDGL16 [1277 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 cost=bdgl16 solver=mitm/numerical zeta grid 56000..61500 step 250
--- refine around 58500 in [58250, 58750] step 50
MIN zeta 58450: log2 rop 133.76 (1267 s)
rop: ≈2^133.8, m: ≈2^12.8, k: ≈2^14.9, mem: ≈2^86.4, ↻: ≈2^54.0, β: 173, d: 14389, ζ: ≈2^15.8, h1: 10
```

#### (b) SSE key, dual_mitm_scan.py --mitm-opt numerical, MATZOV [1988 s]

```
=== n16_boot_sse_to_sparse: n=65536 log2(q)=294 secret=32 cost=matzov solver=mitm/numerical zeta grid 56000..61500 step 250
--- refine around 58500 in [58250, 58750] step 50
MIN zeta 58550: log2 rop 134.16 (1972 s)
rop: ≈2^134.2, m: ≈2^12.8, k: ≈2^14.9, mem: ≈2^86.4, ↻: ≈2^54.5, β: 169, d: 14188, ζ: ≈2^15.8, h1: 10
```


### Task 39 follow-up (ReBoot), raw output

Header, the one-line results `LWE.estimate` prints, and the `min rop` line, which holds every attack (including the EJK20 duals and the fixed-ζ primal hybrids). For the scans: the refine, MIN and final cost lines. Params from `tools/estimate/reboot_params.json`. Wall-clock time in brackets.

#### ReBoot, full estimate, BDGL16 [513 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] m=n cost=bdgl16
usvp                 :: rop: ≈2^131.3, red: ≈2^131.3, δ: 1.004590, β: 325, d: 127747, tag: usvp
bdd                  :: rop: ≈2^131.3, red: ≈2^131.3, svp: ≈2^120.6, β: 325, η: 356, d: 127143, tag: bdd
bdd_hybrid           :: rop: ≈2^128.0, red: ≈2^128.0, svp: ≈2^122.7, β: 303, η: 1, ζ: ≈2^11.6, |S|: ≈2^85.7, d: 124642, prob: 0.401, ↻: 9, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^123.5, red: ≈2^123.2, svp: ≈2^121.1, β: 281, η: 2, ζ: ≈2^12.7, |S|: ≈2^165.1, d: 118156, prob: 0.147, ↻: 29, tag: hybrid
dual                 :: rop: ≈2^131.3, mem: ≈2^75.8, m: ≈2^16.0, β: 325, d: 130782, ↻: 1, tag: dual
min rop (bits): {'usvp': 131.3, 'bdd': 131.3, 'bdd_hybrid': 128.0, 'bdd_mitm_hybrid': 123.5, 'dual': 131.3, 'dual_hybrid': 131.3, 'dual_hybrid_ejk20': 127.9, 'dual_mitm_hybrid': 122.3} -> 122.3
```

#### ReBoot, full estimate, MATZOV [628 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] m=n cost=matzov
usvp                 :: rop: ≈2^129.1, red: ≈2^129.1, δ: 1.004590, β: 325, d: 127747, tag: usvp
bdd                  :: rop: ≈2^129.0, red: ≈2^129.0, svp: ≈2^119.9, β: 325, η: 352, d: 127213, tag: bdd
bdd_hybrid           :: rop: ≈2^126.2, red: ≈2^126.1, svp: ≈2^122.7, β: 303, η: 1, ζ: ≈2^11.6, |S|: ≈2^85.7, d: 124642, prob: 0.401, ↻: 9, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^122.2, red: ≈2^122.2, svp: ≈2^117.7, β: 279, η: 2, ζ: ≈2^12.8, |S|: ≈2^156.2, d: 117586, prob: 0.072, ↻: 62, tag: hybrid
dual                 :: rop: ≈2^129.9, mem: 329, m: ≈2^16.0, β: 328, d: 131072, ↻: 1, tag: dual
min rop (bits): {'usvp': 129.1, 'bdd': 129.0, 'bdd_hybrid': 126.2, 'bdd_mitm_hybrid': 122.2, 'dual': 129.9, 'dual_hybrid': 130.1, 'dual_hybrid_ejk20': 153.3, 'dual_mitm_hybrid': 121.9} -> 121.9
```

#### ReBoot, full estimate, Core-SVP ADPS16 [491 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] m=n cost=adps16
usvp                 :: rop: ≈2^94.9, red: ≈2^94.9, δ: 1.004590, β: 325, d: 127747, tag: usvp
bdd                  :: rop: ≈2^94.9, red: ≈2^94.9, svp: ≈2^73.9, β: 325, η: 252, d: 130527, tag: bdd
bdd_hybrid           :: rop: ≈2^93.9, red: ≈2^93.7, svp: ≈2^91.1, β: 309, η: 93, ζ: ≈2^11.1, |S|: ≈2^53.8, d: 126360, prob: 0.346, ↻: 11, tag: hybrid
bdd_mitm_hybrid      :: rop: ≈2^91.9, red: ≈2^91.8, svp: ≈2^88.2, β: 298, η: 2, ζ: ≈2^12.1, |S|: ≈2^99.2, d: 122801, prob: 0.157, ↻: 27, tag: hybrid
dual                 :: rop: ≈2^94.9, mem: ≈2^75.8, m: ≈2^16.0, β: 325, d: 130782, ↻: 1, tag: dual
min rop (bits): {'usvp': 94.9, 'bdd': 94.9, 'bdd_hybrid': 93.9, 'bdd_mitm_hybrid': 91.9, 'dual': 94.9, 'dual_hybrid': 94.9, 'dual_hybrid_ejk20': 93.9, 'dual_mitm_hybrid': 90.3} -> 90.3
```

#### ReBoot, primal hybrids at fixed ζ, BDGL16 [687 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] m=n cost=bdgl16
min rop (bits): {'bdd_hybrid@0': 131.3, 'bdd_mitm_hybrid@0': 192.4, 'bdd_hybrid@1000': 130.2, 'bdd_mitm_hybrid@1000': 131.3, 'bdd_hybrid@2000': 128.5, 'bdd_mitm_hybrid@2000': 130.2, 'bdd_hybrid@3000': 128.3, 'bdd_mitm_hybrid@3000': 128.1, 'bdd_hybrid@4000': 128.4, 'bdd_mitm_hybrid@4000': 126.0, 'bdd_hybrid@5000': 128.9, 'bdd_mitm_hybrid@5000': 124.2, 'bdd_hybrid@5500': 129.2, 'bdd_mitm_hybrid@5500': 124.0, 'bdd_hybrid@6000': 129.8, 'bdd_mitm_hybrid@6000': 123.6, 'bdd_hybrid@6500': 130.6, 'bdd_mitm_hybrid@6500': 123.4, 'bdd_hybrid@7000': 132.1, 'bdd_mitm_hybrid@7000': 123.6, 'bdd_hybrid@7500': 132.9, 'bdd_mitm_hybrid@7500': 123.6, 'bdd_hybrid@8000': 133.6, 'bdd_mitm_hybrid@8000': 123.8, 'bdd_hybrid@9000': 135.4, 'bdd_mitm_hybrid@9000': 124.9, 'bdd_hybrid@10000': 137.8, 'bdd_mitm_hybrid@10000': 126.0, 'bdd_hybrid@12000': 144.1, 'bdd_mitm_hybrid@12000': 129.8} -> 123.4
```

#### ReBoot, primal hybrids at fixed ζ, MATZOV [697 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] m=n cost=matzov
min rop (bits): {'bdd_hybrid@0': 129.0, 'bdd_mitm_hybrid@0': 190.2, 'bdd_hybrid@1000': 128.1, 'bdd_mitm_hybrid@1000': 129.0, 'bdd_hybrid@2000': 126.6, 'bdd_mitm_hybrid@2000': 128.1, 'bdd_hybrid@3000': 126.5, 'bdd_mitm_hybrid@3000': 126.1, 'bdd_hybrid@4000': 126.8, 'bdd_mitm_hybrid@4000': 124.2, 'bdd_hybrid@5000': 127.3, 'bdd_mitm_hybrid@5000': 122.7, 'bdd_hybrid@5500': 127.7, 'bdd_mitm_hybrid@5500': 122.4, 'bdd_hybrid@6000': 128.6, 'bdd_mitm_hybrid@6000': 122.1, 'bdd_hybrid@6500': 129.9, 'bdd_mitm_hybrid@6500': 122.3, 'bdd_hybrid@7000': 130.9, 'bdd_mitm_hybrid@7000': 122.2, 'bdd_hybrid@7500': 131.5, 'bdd_mitm_hybrid@7500': 122.3, 'bdd_hybrid@8000': 132.3, 'bdd_mitm_hybrid@8000': 122.8, 'bdd_hybrid@9000': 134.3, 'bdd_mitm_hybrid@9000': 123.7, 'bdd_hybrid@10000': 137.2, 'bdd_mitm_hybrid@10000': 125.3, 'bdd_hybrid@12000': 143.2, 'bdd_mitm_hybrid@12000': 129.3} -> 122.1
```

#### ReBoot, dual_mitm_scan.py analytical, BDGL16 [243 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] cost=bdgl16 solver=mitm/analytical zeta grid 2000..16000 step 1000
--- refine around 9000 in [8000, 10000] step 250
MIN zeta 9000: log2 rop 122.06 (238 s)
rop: ≈2^122.1, mem: ≈2^104.3, m: ≈2^15.8, k: ≈2^12.2, ↻: ≈2^14.7, β: 264, d: 113404, ζ: ≈2^13.1, h1: 17
```

#### ReBoot, dual_mitm_scan.py analytical, MATZOV [309 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] cost=matzov solver=mitm/analytical zeta grid 2000..16000 step 1000
--- refine around 9000 in [8000, 10000] step 250
MIN zeta 9000: log2 rop 121.79 (305 s)
rop: ≈2^121.8, mem: ≈2^104.2, m: ≈2^15.8, k: ≈2^12.2, ↻: ≈2^14.7, β: 264, d: 113404, ζ: ≈2^13.1, h1: 17
```

#### ReBoot, dual_mitm_scan.py analytical, ADPS16 [252 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] cost=adps16 solver=mitm/analytical zeta grid 2000..16000 step 1000
--- refine around 6000 in [5000, 7000] step 250
MIN zeta 6250: log2 rop 90.27 (247 s)
rop: ≈2^90.3, mem: ≈2^74.4, m: ≈2^15.9, k: ≈2^11.6, ↻: ≈2^12.7, β: 284, d: 118934, ζ: ≈2^12.6, h1: 12
```

#### ReBoot, dual_mitm_scan.py numerical, BDGL16 [818 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] cost=bdgl16 solver=mitm/numerical zeta grid 6000..12000 step 500
--- refine around 8500 in [8000, 9000] step 100
MIN zeta 8400: log2 rop 121.84 (813 s)
rop: ≈2^121.8, m: ≈2^15.8, k: ≈2^12.1, mem: ≈2^109.9, ↻: ≈2^13.3, β: 268, d: 114562, ζ: ≈2^13.0, h1: 18
```

#### ReBoot, dual_mitm_scan.py numerical, MATZOV [845 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] cost=matzov solver=mitm/numerical zeta grid 6000..12000 step 500
--- refine around 8500 in [8000, 9000] step 100
MIN zeta 8400: log2 rop 121.45 (834 s)
rop: ≈2^121.5, m: ≈2^15.8, k: ≈2^12.1, mem: ≈2^109.9, ↻: ≈2^13.3, β: 268, d: 114562, ζ: ≈2^13.0, h1: 18
```

#### ReBoot, dual_mitm_scan.py --solver exhaustive, BDGL16 [77 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] cost=bdgl16 solver=exhaustive zeta grid 1000..16000 step 1000
--- refine around 6000 in [5000, 7000] step 250
MIN zeta 5750: log2 rop 127.78 (72 s)
rop: ≈2^127.8, mem: ≈2^113.8, m: ≈2^15.9, β: 284, d: 119435, ↻: 368, ζ: ≈2^12.5, h1: 9
```

#### ReBoot, dual_mitm_scan.py --solver exhaustive, MATZOV [111 s]

```
=== reboot_main: n=65536 log2(q)=1730 secret=[97, 95] cost=matzov solver=exhaustive zeta grid 1000..16000 step 1000
--- refine around 5000 in [4000, 6000] step 250
MIN zeta 5500: log2 rop 127.18 (106 s)
rop: ≈2^127.2, mem: ≈2^111.6, m: ≈2^15.9, β: 286, d: 119961, ↻: 255, ζ: ≈2^12.4, h1: 9
```

