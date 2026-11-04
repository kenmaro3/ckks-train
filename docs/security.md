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
bootstrapping keys. No preset sets a Hamming weight.

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
