# Pure CKKS exact-scale metadata (Task 54 P1a)

Status: implemented, locally verified and independently reviewed, 2026-11-06.
The concrete API and rationale below were recorded before production implementation.
The independently reviewed `study-diagonal-admission-design.md` supplies the P1 contract.
This package covers exact-target scalar metadata only; bootstrap descriptors remain P1b.

## Public API

The `ckks_train_ckks::metadata` module exposes:

```rust,ignore
CtMeta::new(params: &CkksParams, level: usize, scale: f64) -> Result<CtMeta>
CtMeta::{level, scale, scale_bits}(&self)
exact_product_scale_at(params: &CkksParams, level: usize, target: f64) -> Result<f64>
target_plain_scale_at(params: &CkksParams, level: usize, source_scale: f64, target: f64)
    -> Result<f64>
scalar_metadata_at(params: &CkksParams, source: CtMeta, value: f64, target: f64)
    -> Result<ScalarProductMeta>
ScalarProductMeta::{pre_rescale, post_rescale}(&self) -> CtMeta
ScalarProductMeta::{plain_scale, rounded_integer, represented_value}(&self) -> f64
```

`CtMeta` has private level and exact F64 scale-bit fields; its constructor accepts level0
but rejects levels above the supplied parameters and nonpositive/nonfinite scales.
Metadata is not a parameter/context identity: scalar calls recheck its level against their
parameters. Equal dimensions or matching labels do not prove basis, key or content validity.

`ScalarProductMeta` privately stores the exact pre-rescale label and level, the computed
plaintext scale, the rounded integer as F64 (including signed zero), and the exact target at
level minus1. `represented_value` returns the actual F64 `rounded_integer / plain_scale`,
not the requested ideal coefficient. No plaintext-range or decryption-error bound is claimed.
Pure calls allocate no keys, ciphertexts, Encoder, evaluator or RNG, and change no counters.
`CkksParams` may contain actual secure-preset RNS tables; those are parameters, not keys.

## Executable arithmetic and errors

Keep one existing exact-label implementation: nearest-first search at `target * (q[level]
as f64)`, testing offsets0,+1,-1,...,+8,-8 with F64 division equality. There is no tolerance,
scale relabel, reordered expression, approximate prime or guessed Delta recurrence.
`exact_product_scale_at` delegates directly and preserves its validation order: level range,
positive finite target, level0, finite product, then reachability. Existing `HekitError`
variants report `InvalidParams`, `NoLevelsLeft` and `ScaleMismatch`.

The checked plaintext-scale call validates source level/scale, computes the exact label,
then `plain_scale = label / source_scale`, rejects nonfinite intermediates, and enforces
runtime's inclusive `MIN_PLAIN_SCALE = 2^20` minimum. Scalar metadata first revalidates
source metadata, then uses the shared legacy scalar calculation in its original order:
finite value, exact label, minimum plaintext scale, `round(value * (label / source_scale))`,
strict `abs(k) < 2^62`. It also explicitly rejects nonfinite plaintext scales/integers.
Zero and negative scalars are valid; a nonzero value that rounds to zero is valid. This API
does not impose the stricter learning-factor policy of other admission packages.

The runtime scalar arithmetic will share a narrow helper; its basis checks remain before
that helper and its polynomial work/counting remain after it. No general fake HeOps
interpreter or new execution path is needed. Existing target helpers retain their behavior.

## Deliberate boundary with legacy runtime

Inspection found a finite source scale as small as `f64::from_bits(1)` can make
`label / source_scale` infinite. Legacy `target_plain_scale` returns infinity because its
minimum check only compares `< 2^20`. With value0, legacy scalar arithmetic rounds
`0 * infinity` to NaN, whose magnitude comparison does not reject it; the integer cast
becomes zero. New pure metadata rejects such unsupported intermediates. This extraction
must not silently harden CPU/CUDA semantics or claim equivalence on that edge. A focused
small runtime test preserves this counterexample; runtime hardening needs a separate
review and decision. Root explicitly ratified this narrow boundary before production code. Normal finite-intermediate scalar outcomes/error order stay shared.

## Alternatives rejected and evidence scope

Do not duplicate the ±8 search, use all-q-as-2^42 approximations, compare with tolerances,
encode dummy ciphertexts for admission, or move minimum-scale checks onto ordinary
activation/mul_plain paths. Private fields avoid unchecked public receipts; F64 bits avoid
lossy logarithmic scales. Recording actual `k / s_p` avoids describing tiny rounded-zero
coefficients as their ideal nonzero value.

Verification covers actual n16_train/n16_boot parameter-only recurrence, independent
neighbor-scan and fixed unreachable fixtures, invalid finite/level/minimum/integer bounds,
and a small insecure CPU fixture against Evaluator/HeOps labels, coefficient value and
counts. No secure key generation, bootstrap/training execution, performance, precision,
security, physical-device, CUDA execution or planner/admission claim follows from P1a.


## Local verification

The missing-module test first failed with E0432. After finite-domain positive behavior
passed, explicit overflow tests failed with `Ok(inf)` and a NaN-bearing scalar receipt;
adding only the pure API's finite-intermediate checks made them pass. Final metadata suite:
9 passed, including the separately named legacy edge characterization and N=16 encrypted
runtime scalar parity. Existing pure-label search, scalar errors, minimum-scale rejection
and HeOps tests: 5 passed. Targeted Clippy with `-D warnings`, owned Rust formatting and
NN consumer compile passed. The ±32-neighbor independent oracle includes fixed unreachable
target bits `0x4135eb851eb8aeb9` at actual `q[1] = 974849` for the coarse-prime fixture.

All builds used `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8`; tests used
`--test-threads=8`. Existing large-key scalar-success tests were not run; the new N=16
fixture supplies bounded success/value/count evidence. No full workspace suite or secure
key generation was run. No HeOps trait/default changed, so CUDA compile/execution was not
required or performed. Full commands, RED/GREEN details and scope are in the P1a report.
