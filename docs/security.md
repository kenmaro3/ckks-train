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
  b = −a·s + e + P·s' (mod Q·P): s' sits only in the message, so as an RLWE instance it is one
  more sample under the dense s, and **(a) covers it**.
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
−1 is binomial instead of exactly h/2. This is the usual modelling and it is not refined here.
Error: σ = 3.2, m = n. Attacks: dense (a) uses usvp, bdd, dual and dual_hybrid, as in M1. Sparse
(b) and (c) use those plus bdd_hybrid, bdd_mitm_hybrid and dual_mitm_hybrid. At this commit
`LWE.estimate` has no dual MITM entry (its `dual_hybrid` is MATZOV's dual), so `estimate.py`
runs the EJK20 dual hybrid with MITM guessing, `lwe_dual.dual_hybrid(mitm_optimization=True)`,
separately and reports it as `dual_mitm_hybrid`. Arora-GB and BKW are skipped, as in M1.

**The dual MITM optimiser is not robust.** Its search over the splitting dimension ζ is a
local-minimum search whose result depends on `opt_step`. On (b) it gave 136.4 / 136.6
(BDGL16 / MATZOV) at the default step 8. With step 16 it gave 134.6 / 135.0; with step 2,
136.4 / 206.7; under ADPS16, 179.5 at step 8 and 122.3 at step 2 (step 1 fails inside the
estimator). `tools/estimate/dual_mitm_scan.py` therefore evaluates the same inner optimisation
(`DH.optimize_blocksize` with the MITM solver, every h1 ≥ 1) on an explicit grid,
ζ = 40000..65000 in steps of 500, and refines in steps of 25 around the best point. **The grid
minima are the (b) figures below.** They are smooth around the minimum (±0.1 bit over ±100 in
ζ). The primal hybrids and the (c) runs use the estimator's own optimiser. For (c) a grid could
only lower a figure that is already below 128.

**`inf` entries.** For (b), usvp, bdd, dual and dual_hybrid return ∞: no block size up to the
estimator's cap β = 1754 (`conf.max_beta`) succeeds. That means at least 0.292 × 1754 = 512 bits
even under Core-SVP, so they do not bind. The binding attacks on (b) guess about 2^15.8 of the
65536 secret coordinates as zero and attack the remaining dimension of about 8500.

### Results (log2 rop, minimum over attacks; binding attack in brackets)

| key / instance | log modulus | BDGL16 (= HE Standard sieve) | MATZOV (estimator default) | Core-SVP (ADPS16; information only) |
|---|---|---|---|---|
| (a) `n16_boot` dense main key | 1732 | **131.5** [usvp/bdd, β 326] | **129.3** [usvp/bdd, β 326] | 95.2 |
| (b) SSE key s → s', h' = 32 | 294 | **134.4** [dual_mitm_hybrid, grid: ζ 59000, h1 9, β 153] | **135.0** [dual_mitm_hybrid, grid: ζ 58500, h1 10, β 173] | 122.0 [dual_mitm_hybrid, grid] |
| (c) `n16_boot_sparse` main key, h = 192 | 1732 | **122.2** [dual_mitm_hybrid, β 269] | **121.7** [dual_mitm_hybrid, β 269] | 90.2 |

The rows in more detail:
- (a) at m = oo: identical (131.5 / 129.3), as for n16_train in M1.
- (b), other attacks: bdd_mitm_hybrid gives 144.2 (BDGL16) / 144.4 (MATZOV) / 133.2 (ADPS16),
  and bdd_hybrid 162.9 / 164.0 / 143.2. The rough model gives ∞. At m = oo every attack is
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
  of its key types**: (a) 131.5 / 129.3 and (b) 134.4 / 135.0. The MATZOV margin of (a) is 1.3
  bits. Both are at least 128, so the 58-bit EvalMod primes are kept (no trim), and the chain,
  log PQ 1732, levels_after_boot 16 and the measured precision (17.46 bits, docs/bootstrap.md
  §11.4) stand. **`n16_boot` may be used in the paper.**
- **`n16_boot_sparse` (h = 192) is below 128 bits** (122.2 under BDGL16, 121.7 under
  MATZOV), because of the hybrid attacks. Following the rule "a setting that misses 128 bits is
  not used in the paper" (128 bit を満たさない設定は論文に使わない), **`n16_boot_sparse` is not
  used in the paper**. It stays in the library as a measured alternative: its bootstrap passes
  the gate with 21.12 bits (model 21.0; `n16_boot_keys_bootstrap_sparse`), 3.7 bits more than
  the dense key, because a smaller main key lowers the rescale and key-switch rounding noise.

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

