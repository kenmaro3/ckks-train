# Task54 P4 CPU source fix3 and runtime1 boundary review

2026-11-07. **SPEC: PASS for the scoped fix3 correction. QUALITY: PASS.** The source change removes the exact recorded synthetic-rate policy failure while preserving canonical policy and fixed recipe/reference values. No new finding in the assigned delta. Runtime2 has not run and is not released by this review.

Reviewed the exact fix3 release, complete8058-byte source/docs delta, appended report, all8 source pins, final no-run/Clippy records, immutable runtime1 raw/terminal records, unchanged canonical rate validation/codec, and relevant synthetic build/test ordering. Only reads/hashes and this review-file write were performed; no runtime, tests, Cargo, constructors, source edits, network, Git mutations or subagents.

## Frozen evidence

Independently verified:

- Report SHA256 `2b5bbeb8f248bd92e892afff9f741e742aab50c8a4a4997f126d5dc2773784ea`.
- Exact8058-byte delta SHA256 `d9b7bff545fb64ed5fc5748f6b8afc7d77e835e745f5b45bd7f886ae4fd29a3f`.
- All8 source pins JSON SHA256 `cba30c1d4ea8bc8e746d8f5e0454c3d6722c772ad776aad4c77026f814b3ca13`; current all8 files match. Only the two assigned cfg(test) files differ from fix2; the other six remain byte-identical, including canonical P3, client/session, NN diagnostic/training and Signal oracle.
- Root's release SHA256 `bd82713b9aaa477e7939a28816e02b967d81b48c5eeb7317e3d901fa34941b02`; reviewed design SHA256 `63fc5ad6ca06973417f0445a168ee0cd8b3f40c62490b6ccd18c503e5ab766df`.
- Recorded `02-compile-final` and `03-clippy` exit0, with clean raw outputs and matching before/after8-source inventories equal to reviewed pins. These compile/static checks do not execute the regression or HE matrix.
- Runtime1 raw is exactly1965 bytes, SHA256 `f1d14696c5cc7e4daeee089db83a34017b8b306310bd8535b759d32d8d4ef242`. Terminal test/controller exits are101; before/after source inventories match each other and all8 fix2 pins. Failure evidence remains separate from fix3 source/static evidence.

## Actual runtime1 boundary

One actual test ran and failed:0passed,1failed,190filtered. The raw log shows two parameter reservations, one DFT proof, one bootstrap descriptor, and six completed Square plan rows. The next plan fails with exact typed `RateConflict` / `full maximum reference` before keys/spy/setup/session work.

The source ordering independently establishes the more precise partial planner count: the seventh PolyC3 I plan completed geometry and initial-state construction before calling canonical rates(), which rejected its single.04 kernel against explicit.08 reference. It had not entered compose, so update planners remain6. Thus actual runtime1 partial totals are **2parameters/1proof/1descriptor/7geometry/7initial/6updates**, with **0keys/0Encoders/0bootstrapper/0Engine/0trainer/0HE evaluation/0factory spy/0sessions**. Only6/6/6 was printed in completed-plan rows;7geometry/7initial is a justified source-order inference, correctly labelled in the report.

This is a recorded fixture-policy failure, not a failed CKKS numerical comparison, a smoke numerical pass or a canonical/D7 pre-key proof. Earlier compile/source reviews did not establish this runtime path passed. No retry has been performed as part of fix3 or this review.

## Cause and correction

Unchanged canonical `rates()` requires finite positive source/mean/delta/kernel fields, authoritative folded kernel bits, and reference equal to the full scheduled kernel maximum before delegating to `encode_rate_schedule_fields`. That canonical full-plan-maximum rule is appropriate and remains byte-identical.

The private synthetic fixture intentionally differs: its already-frozen closed Poly smoke recipe has one consumed.04 rate and independent explicit.08 reference. Calling the canonical policy validator in this distinct fixture caused the exact runtime1 error. The actual builder now calls private `encode_synthetic_rates`, a one-operation wrapper delegating to the existing field encoder. This binds the exact same typed fields and F64 bits without copying schema or arithmetic and without asserting canonical admission. The closed recipe and actual existing P2 initial/update validators retain configuration/numerical validity ownership.

The change does not lower reference, raise current rate, append a fabricated future rate, change planning/update count, modify NN math or loosen production validation. It alters no public constructor or execution capability. Root's added design rationale states these distinctions accurately. The previous NI helper-versus-literal-mask boundary remains intact and consistent with source.

## Regression and work inventory

The existing single ignored test now calls `explicit_reference_policy_regression` before `run_contract_matrix` and therefore before all parameter/setup constructors. It uses the actual closed PolyC3 recipe, asserts one.04 kernel and.08 reference bit pattern, and:

1. Calls unchanged canonical policy once, requiring exact RateConflict/full-maximum rejection and no bytes written.
2. Calls the exact private synthetic wrapper used by the builder once, requiring both original bit patterns in existing-codec output.

A local guard and final assertion bound those two policy/encoding invocations. No independent schema offsets/decoder, optimizer formula, extra test group, parameter, geometry, initial/update plan, Engine, trainer or HE operation is introduced. The bit-presence assertions are narrow shared-codec regression evidence; they do not claim a general transcript decoder. The thin wrapper and unchanged codec make that boundary appropriate for this specific fix.

All expensive per-run matrix caps and fixed I1e-5/NI4e-4 tolerances remain unchanged, as do54encrypted-state captures,268unchanged assertions,322configuration collections,68client inspect/decrypt endpoints and122encrypted model snapshot copies. The corrected future matrix is still prospective. Runtime1's actual partial setup/planner work must remain counted separately and in any cumulative execution inventory; a later full run must not erase those operations or the failed run.

## Disposition

The recorded cause is corrected at its real synthetic-builder callsite, canonical policy is preserved, and the assigned delta introduces no new important/minor finding. Source/static acceptance for this scoped correction is complete. Root may separately decide and freeze a finite runtime2 release with its per-run and cumulative counts.

No numerical parity, runtime regression pass, actual full cap completion, production/canonical D7 acceptance, P4/Task54 completion, or scientific precision/range/security/resource/campaign evidence follows from this review.
