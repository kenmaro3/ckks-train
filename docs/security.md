# Security estimates for the CKKS presets

Route: (b) SageMath 10.9 from conda-forge via micromamba (osx-arm64, Python 3.12.16), plus
malb/lattice-estimator at commit `ae45d740ceb138af87a4f92edc592a0ab0d4f43f` (2026-11-04).
Dev tool only (`tools/estimate/`); nothing from it is committed.

Model: LWE with n = N, q = 2^round(log PQ) (power-of-two stand-in; the attacker's modulus for
key-switching keys is Q*P), secret uniform ternary (`ND.Uniform(-1,1)`), error
`ND.DiscreteGaussian(3.2)`, m = n. Attacks: usvp, bdd, dual, dual_hybrid (Arora-GB, BKW and
the hybrid/MITM variants skipped via deny_list). Values are log2(rop).

| preset | N | log PQ | usvp | bdd | dual | dual_hybrid | min |
|---|---|---|---|---|---|---|---|
| seal_2020, BDGL16 | 2^15 | 480 | 253.8 | 253.5 | 254.1 | 253.2 | 253.2 |
| seal_2020, default cost model | 2^15 | 480 | 245.8 | 245.6 | 247.3 | 246.4 | 245.6 |
| n16_train, BDGL16 | 2^16 | 1592 | 144.1 | 144.1 | 144.1 | 144.1 | 144.1 |
| n16_train, default cost model | 2^16 | 1592 | 141.2 | 141.2 | 142.3 | 142.0 | 141.2 |

Both presets are >= 128 bits under the most conservative result (141.2 for n16_train).
The provisional n16_train constants (L = 31, k = 4, dnum = 8) are kept: confirmed by Task 22.

Caveat: `LWE.estimate.rough` (a crude 0.292*beta sieve cost with no lower-order terms) gives
107.7 bits for n16_train (beta = 369) and 217.8 for seal_2020. It is a screening heuristic, not the
decision value; the full estimates above are used. A reader wanting the rough-model margin should
note n16_train would not reach 128 under it. For cross-checking, the HomomorphicEncryption.org
standard tables give a max log Q for 128-bit classical (ternary) of 881 at N = 2^15 and 1772 at
N = 2^16 (quoted from memory of the 2018 standard, not re-verified here); 480 and 1592 are below.

Secrets with a Hamming weight (`with_hamming_weight`) are NOT covered; estimate.py supports it via
`SparseTernary` but no preset uses it.

## Raw output (BDGL16)

```
=== seal_2020: n=32768 log2(q)=480.00 secret=ternary_uniform
usvp                 :: rop: ≈2^253.8, red: ≈2^253.8, δ: 1.002540, β: 748, d: 63792, tag: usvp
bdd                  :: rop: ≈2^253.5, red: ≈2^253.5, svp: ≈2^246.2, β: 747, η: 786, d: 64346, tag: bdd
dual                 :: rop: ≈2^254.1, mem: ≈2^165.0, m: ≈2^15.0, β: 749, d: 65455, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^253.2, red: ≈2^253.2, guess: ≈2^225.6, β: 746, p: 3, ζ: 0, t: 130, β': 746, N: ≈2^134.4, m: ≈2^15.0
min rop (bits): {'usvp': 253.8, 'bdd': 253.5, 'dual': 254.1, 'dual_hybrid': 253.2} -> 253.2
=== n16_train: n=65536 log2(q)=1592.00 secret=ternary_uniform
usvp                 :: rop: ≈2^144.1, red: ≈2^144.1, δ: 1.004210, β: 369, d: 126387, tag: usvp
bdd                  :: rop: ≈2^144.1, red: ≈2^144.1, svp: ≈2^134.4, β: 369, η: 403, d: 125949, tag: bdd
dual                 :: rop: ≈2^144.1, mem: ≈2^85.1, m: ≈2^16.0, β: 369, d: 131072, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^144.1, red: ≈2^144.1, guess: ≈2^118.6, β: 369, p: 2, ζ: 10, t: 80, β': 369, N: ≈2^74.3, m: ≈2^16.0
min rop (bits): {'usvp': 144.1, 'bdd': 144.1, 'dual': 144.1, 'dual_hybrid': 144.1} -> 144.1
SUMMARY {"seal_2020": {"usvp": 253.77708788999735, "bdd": 253.50672954450047, "dual": 254.10621578103525, "dual_hybrid": 253.22913537337467}, "n16_train": {"usvp": 144.0954885518754, "bdd": 144.09218685093657, "dual": 144.148, "dual_hybrid": 144.14700906863825}}
```

## Raw output (default cost model)

```
=== seal_2020: n=32768 log2(q)=480.00 secret=ternary_uniform
usvp                 :: rop: ≈2^245.8, red: ≈2^245.8, δ: 1.002540, β: 748, d: 63792, tag: usvp
bdd                  :: rop: ≈2^245.6, red: ≈2^245.6, svp: ≈2^238.0, β: 747, η: 776, d: 64720, tag: bdd
dual                 :: rop: ≈2^247.3, mem: ≈2^166.2, m: ≈2^15.0, β: 749, d: 65455, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^246.4, red: ≈2^246.4, guess: ≈2^225.6, β: 746, p: 3, ζ: 0, t: 130, β': 752, N: ≈2^134.4, m: ≈2^15.0
min rop (bits): {'usvp': 245.8, 'bdd': 245.6, 'dual': 247.3, 'dual_hybrid': 246.4} -> 245.6
=== n16_train: n=65536 log2(q)=1592.00 secret=ternary_uniform
usvp                 :: rop: ≈2^141.2, red: ≈2^141.2, δ: 1.004210, β: 369, d: 126387, tag: usvp
bdd                  :: rop: ≈2^141.2, red: ≈2^141.2, svp: ≈2^134.0, β: 369, η: 403, d: 125949, tag: bdd
dual                 :: rop: ≈2^142.3, mem: ≈2^91.6, m: ≈2^16.0, β: 369, d: 131072, ↻: 1, tag: dual
dual_hybrid          :: rop: ≈2^142.0, red: ≈2^142.0, guess: ≈2^129.6, β: 368, p: 3, ζ: 0, t: 70, β': 399, N: ≈2^82.6, m: ≈2^16.0
min rop (bits): {'usvp': 141.2, 'bdd': 141.2, 'dual': 142.3, 'dual_hybrid': 142.0} -> 141.2
SUMMARY {"seal_2020": {"usvp": 245.84053105498958, "bdd": 245.5896152745565, "dual": 247.2837848168276, "dual_hybrid": 246.41790059389658}, "n16_train": {"usvp": 141.2348735492737, "bdd": 141.23947496455423, "dual": 142.27972042406688, "dual_hybrid": 141.99289948189048}}
```
