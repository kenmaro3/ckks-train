# Task54 D4c2 NI first-primary outcome — independent audit

**COVERAGE: PASS. RESULT: PASS for the registered finite N1024 Lazy fixture. INTEGRITY: PASS for the retained evidence and supplied source/executable identities.** No discrepancy or failed numerical gate was found. This closes the outcome audit of attempt1 only; it does not authorize a rerun, publication, further campaign execution or production admission.

The root assignment supersedes the outcome brief's historical “prepared, unassigned” heading. I inspected the fixed validation plan, original implementation report/review, FIX1 report/re-review, exact numerical driver/helpers and relevant stage/trainer/refresh metadata, counting and hidden-selection paths. I independently audited existing bytes with a new standard-library Python reader. No build, test executable, metadata executable, key generation, HE, decryption, network/SSH, Git mutation or subagent was used. Read-only `git show <supplied-commit>:<path>` checked the 13 source-manifest files against that commit. Only the new audit artifacts named below were written.

## Attempt and evidence identity

Directory: `task-54-d4c2-ni-primary-attempt-1/`, relative to this ledger. It contains exactly six payload files and `SHA256SUMS`; every entry independently verifies. The manifest SHA256 is `f160b411c13e6db6278259b256c199ba6a6ba7561a155bd77ed7213519212127`.

| Retained file | SHA256 |
|---|---|
| `frozen-test-executable` | `9ee26bbbd7ea239b67d99e8de7d7c67aeb1ecf6764d65cf98a013f90c6c7eb5b` |
| `stdout.txt` | `f26da2e6a3ce29439aebea7311c078bd5bbe3461d5372de88fc9a700d543e116` |
| `stderr.txt` | `698d8338ac326352bb4c44140c4f9267e1cf91ebbc8721d1d0b2241fe175e159` |
| `runner.txt` | `a878c7d9e598618d217d4775c6075ab8d3713b066063263d011ad34da75fdec5` |
| `runner_program.py` | `dfc6d01b109aa7b5dea41c744f49c65f251403b93b154e52e28839f73a413c54` |
| `source-SHA256SUMS` | `75f07083391da0b449204e976d7035886ccd860b204bbe9e5f591333aaf96a8d` |

All 13 source entries match both current bytes and source commit `f53f34f0396e05ecae3d670fee21140454123349`. The copied manifest equals the separately frozen FIX1 source manifest. The supplied scoped FIX1 review independently hashes to `36198aab83efa7b7ba46d661fe4c1426ba8e17b0a7181c9614699156a215fdc2`. Executable before/after hashes in the runner both equal the frozen executable's independently recomputed hash. Runner metadata records unchanged sources. Those checks bind the supplied artifacts; no reproducible binary rebuild, code-signing attestation or fresh remote/push verification is claimed.

The saved runner program uses exclusive output creation and an attempt directory with `exist_ok=False`, snapshots the executable, checks approved source/review identities before launch and sources/executable again after return. It launches one child and performs no retry. Scientific stdout/stderr are direct child streams in separate files; the runner's metadata is separately labeled. The runner reports local Mac execution **2026-11-07 00:19:19.919358–00:20:38.735245 JST**, return code0. `stderr.txt` has only Cargo's finished-profile and selected-target execution lines, with no compilation line or error. Wall-clock intervals and the harness's 78.24s identify this attempt; they are not a server performance benchmark.

The recorded exact command is:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --test packed_training_ni packed_ni_primary_12_cases_max_228_calls -- --ignored --exact --nocapture --test-threads=1
```

It matches the registered ignored, exact-name, serial primary selection. The guard is in `tests/packed_training_ni.rs:273–276`. There is one primary start, one complete ordered case sequence, one PASS final summary (`stdout.txt:2405`) and one harness result (`stdout.txt:2408`): one passed, zero failed, 12 filtered out. No construction/execution failure, duplicate/restarted case, skipped expected diagnostic or extra unclassified log line was found. The frozen record supports one retained attempt; it cannot establish that no unrecorded external execution ever occurred.

## Coverage, schedule and counters

`stdout.txt:4` reports N1024/log_n10, exact Delta scale bits4796333603149578240, dnum8, the full32-Q/four-P n16_boot chain, log_slots9 and default n16 BootParams otherwise, held-out key seed153, and sorted training steps `[-2,-1,1,2,4,8,16,32,64,128,256]`. Its Q/P bit patterns are `[54,42×16,34×4,58×7,56×4]` and `[60×4]`; every BOOT output reports level16. Source `primary.rs:55–97` preserves the fixed constructor/key order and level16 fresh encryption; this is the full chain at small N, not secure-size N2^16.

Both ordinals0/1 each execute the following four-update schedules, giving all12 Square/Poly × ordinal × AU/WN/AA cases:

| Mode | Placement | Actual calls per update (zero-based0–3) | State-refresh updates | Per-case total |
|---|---|---|---|---:|
| Square | AU | 4,4,4,4 | 0,1,2,3 | 16 |
| Square | WN | 4,4,4,4 | 0,1,2,3 | 16 |
| Square | AA | 7,5,6,7 | 0,1,2,3 | 25 |
| Poly | AU | 4,4,4,4 | 0,1,2,3 | 16 |
| Poly | WN | 4,4,4,4 | 0,1,2,3 | 16 |
| Poly | AA | 3,5,2,7 | 1,3 | 17 |

There are **48 BEGIN rows,48 PREPARED rows,212 BOOT rows,176 PROJECTION rows and1,916 PARITY rows**. Every expected row was consumed once in exact source-derived order; all2,400 diagnostic rows are contiguous at stdout lines5–2404. BEGIN attempted/completed totals advance by precisely the preceding update's events and validated-update totals progress0–47. Final actual attempted212/completed212 and validated48/missing0 are reconciled with these rows, rather than inferred from the212 forecast alone. The cap228 was never reached. There are no unexecuted registered cases.

The four valid-ID sequences are exactly `[0,1,2]`, `[0]`, `[0,2]`, `[0,1,2]`; update2 retains record2, and sentinel3 never appears in any event or forward diagnostic. Every AA update starts with its valid-record hidden events in that order. AU/WN have no hidden events. State events then follow actual layer ordinals: Weight(2o), Bias(2o), Weight(2o+1), Bias(2o+1) for Square; Weight(2o), Velocity(2o), Weight(2o+1), Velocity(2o+1) for Poly. Totals are88 weight,48 bias,40 velocity and36 hidden calls. State events have exactly one support projection; hidden events have none. Source role order is `refresh.rs:185–219`; original record identity is retained by `stage.rs:419,584–589` and `trainer.rs:580–594`.

All2,496 `SCHEDULE`/`TRACE` lines in original and FIX1 forecasts are byte-identical. Every measured PREPARED seven-field counter equals its matching frozen SCHEDULE counter; the audit evidence retains all48 raw/forecast line crosswalks. This is observed-vs-prospective equality, not replacement of observed work with a cost model.

Counter field order throughout is `(rotations,relins,ct_ct_tensor,ct_pt_mults,rescales,sse_switches,bootstraps)`:

- Each of212 observed nested BOOT deltas equals **(37,40,46,207,56,2,1)**, matching the constructed Bootstrapper's separately reported configured cost. Inclusive key switches are79/call in this N1024 context; no N2^16 figure is substituted.
- Summed48 observed inclusive update deltas are **(9626,9224,10616,45122,13854,424,212)**.
- Their nested bootstrap subintervals sum to **(7844,8480,9752,43884,11872,424,212)**. These are already included and must not be added again.

## Numerical result and exact worst observations

Every saved PARITY row's full complex error, active error, padding error, replica spread and imaginary leakage is finite, nonnegative and ≤ **E=2^-12=0.000244140625**. Full error equals max(active,padding) in every row. Replica spread measures differences between replicas and can exceed the corresponding full error; both remain below E. Every BOOT incremental error is ≤ **epsilon_boot=2^-20=9.5367431640625e-7**. No percentile, averaged or final-only gate was substituted.

Locations below refer to the unchanged `task-54-d4c2-ni-primary-attempt-1/stdout.txt`. Updates, events and slots are zero-based. When a maximum is repeated, the table gives its first raw occurrence. The TSV preserves every entire cited record and additional per-snapshot maxima.

| Diagnostic | Exact printed maximum | Raw location and case | Metric-specific worst slot |
|---|---:|---|---|
| Full complex error / active error | `2.53557821117368795e-8` | line577; Square ordinal0 AA update3 event0 Hidden(stage0,record0), boot_output | 184 |
| Padding error | `1.26475082515420745e-8` | line591; Square ordinal0 AA update3 event3 Weight(0), boot_output | 428, also full-error maximum in this row |
| Replica spread | `3.06720765902612449e-8` | line807; Square ordinal1 AU update3 event3 Bias(3), boot_output | not emitted for replica spread |
| Imaginary leakage | `2.27929197722204910e-8` | line191; Square ordinal0 AU update3 record0 hidden.pre | not emitted for imaginary leakage |
| Projection drift | `1.21086388574046290e-8` | line81; Square ordinal0 AU update1 event3 Bias(1) | 380 |
| Incremental bootstrap complex error | `1.26730185407340126e-8` | line592; Square ordinal0 AA update3 event3 Weight(0) | 428 |

`worst_slot` on PARITY rows is the full-error location. It is not a saved argmax for each diagnostic column; no replica/imaginary argmax is invented above. Projection drift has no separate registered epsilon gate: the pre/post projection parities are independently gated by E. Its finite maximum is retained without introducing a new acceptance threshold.

All212 actual bootstrap input norms pass their role-specific bound and the universal≤1 contract:

| Role family | Exact largest reported norm | Bound | Raw location and case | Worst slot |
|---|---:|---:|---|---:|
| Weight | `2.74428858451710611e-1` | 0.4 | line1986; Poly ordinal1 AU update3 event0 Weight(2) | 0 |
| Square bias | `2.62329525847428968e-1` | 0.4 | line1240; Square ordinal1 AA update3 event6 Bias(3) | 372 |
| NAG U | `3.94056167230180110e-2` | 0.1 | line2380; Poly ordinal1 AA update3 event6 Velocity(3) | 0 |
| Hidden / overall | `3.36142541618032420e-1` | 0.5 | line2348; Poly ordinal1 AA update3 event0 Hidden(stage1,record0) | 1 |

For all212 BOOT rows, input level0/output level16 and both scale-bit values4796333603149578240 match exactly. Exact source requires context ownership/two components and expected output level/scale before retaining a successful event (`trainer.rs:539–567`, `types.rs:6–18`); input construction validates exact Delta and level0 (`refresh.rs:582–622,671–684`). These checks are distinct from slot range/precision. Component/context/NTT flags are not individually serialized in the saved BOOT rows, so the external audit does not claim independent reconstruction of those absent flags. In particular, `types::check_ct` does not itself inspect an NTT flag.

The full live graph trace is not printed by this primary. Source `stage.rs:564–566` compares the actual trace and all seven graph counts to preflight; `trainer.rs:499–513` separately compares committed histories and inclusive counts. Successful control flow supports those program checks. Every serialized BOOT input/output level/scale and every serialized observed count was independently compared; no external replay of unsaved graph histories is claimed.

## Fixture fidelity and hidden-selection evidence

The source retains the literal W/b/X/Y arrays, original masks and IDs, rates `[0.04,0.01,0.02,0.08]`, Lazy only, Square output3/classes3 and loss2/3, Poly output3/classes2/no bias, mu0.9 and immutable reference0.08. Initial encryption seeds200–203 and record seeds160+8*update+2*record+role are fixed. Each case resets its initial deterministic RNGs. These are correlated reproducible cases, not twelve independent random-key samples. The held-out seed153 is used once to construct the main context; the immutable84-call calibration was neither selected nor rerun.

The independent literal packer implements all512 slots with the registered `s=4*i+j` orientations/supports and includes active channel2, padding and full vector replicas. Scalar forward/backward uses old W/z1 and independent unscaled V, with no production scalar-oracle call (`packed_ni_helpers/mod.rs:6–170`). A separate ordinary f64 calculation from the fixed plan literals confirms final learning visibility above4E: Square W movements approximately0.00156116594/0.00279636966 and bias movements0.00212190562/0.01232951274; Poly W movements0.02442884901/0.03459904160 and U magnitudes0.02758264782/0.03940561430. These derived plaintext values are labeled separately in audit evidence; they are not new encrypted measurements or exact Rust-f64-bit attestations.

For every update, saved metric tuples agree across old/old_retained, candidate/event pre_projection, state-event boot_output/committed_candidate, and carried candidate/committed_candidate when AA Poly does not refresh. Every AA hidden event source matches hidden.pre, its output matches hidden.post, and hidden.selected matches post; AU/WN selected matches pre. Both hidden snapshots are compared against the old-state scalar hidden. Outputs are compared against old-state scalar outputs on every valid record.

Numerical equality of diagnostic tuples alone does not prove object identity. Source `stage.rs:431–491` selects the returned AA post object before affine2, stores that same selected hidden for outer1 and receipt.hidden, and passes the earlier old z1 to derivative1. This is source corroboration, not a new run of the separate historical operand-witness control. No fresh encryption of carried model state, dummy U/bias, secret callback, client-refresh fallback, plaintext repair, seed/rate/threshold retuning or retry appears in the primary path.

The source detects range/precision errors **after synchronous prepare_update returns and before commit/next update** (`primary.rs:464–584`). All returned event metric rows are emitted before ordered event acceptance. A later call in the same update could already have completed before the first externally detected gate failure; this PASS is not evidence of immediate per-call stopping. Runtime exceptions/invalid metadata would have separate failure/count accounting and do not count as successful retained events. None occurred in this attempt.

## Limits and retained audit artifacts

The exact driver computes max_error over all512 complex slots, rejecting nonfinite real/imaginary values (`primary.rs:152–199`), and computes each actual-input norm and incremental error from externally decoded retained ciphertext snapshots (`primary.rs:264–377`). Thus the **program's** checks cover all512 slots, including padding and replicas. The saved data are aggregate maxima/worst-slot diagnostics, not all decoded arrays or ciphertexts. This audit independently validates every reported aggregate and its gate, but cannot independently redecrypt or recompute every slot-level diagnostic. It also does not establish absence of an unseen implementation defect in dependencies.

This is finite empirical correctness evidence for these12 Lazy cases/four updates, small N1024 and this deterministic context. It does not prove universal bootstrap precision, complete shared Encoder-free P1 admission, encoded Q/2/internal EvalMod feasibility for other histories, secret/key provenance, cryptographic security, secure N2^16 execution, PerSample numerical parity, production memory/liveness, GPU/backend/Linux operational support, c3 block composition, D4d profile bridge, D6c2 sessions, full registered campaign/profile coverage, adoption or production Ready. Snapshot retention consumes real memory; no memory-fit measurement is available. Root retains publication and future-work decisions.

New files, all in this ignored ledger:

- `task-54-d4c2-ni-main-outcome-review.md` — this independent report.
- `task-54-d4c2-ni-main-outcome-audit.py` — standard-library reader, provenance checks and independent scalar arithmetic. It does not invoke the saved runner or executable.
- `task-54-d4c2-ni-main-outcome-evidence.txt` — all input/source hashes,48 measured/forecast counter crosswalks, audit results and exact worst raw rows.
- `task-54-d4c2-ni-main-outcome-maxima.tsv` — global/role/snapshot maxima with original raw lines.
- `task-54-d4c2-ni-main-outcome-audit-SHA256SUMS` — manifest of these four report/reader/evidence payloads.

No original attempt, source, public document, raw log, scientific-result JSON or Git state was changed.
