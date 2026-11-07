# Task54 P4 CPU runtime/evidence review

2026-11-07. **SPEC: PASS for the bounded synthetic CPU contract. QUALITY: PASS within the reviewed source/runtime scope. EVIDENCE: PASS for custody, actual counts and the derived parity table.** No new finding. This closes this finite CPU validation package, not canonical/D7 admission, all P4/Task54 work or a scientific experiment.

The reviewer read the runtime report, runtime2 release/authorization/controller and immutable raw/terminal records; independently parsed every guard/plan/parity/result row, checked current source pins and authorization hashes, joined all32 derived parity rows to raw session/update context and reviewed the reachable success/failure assertion paths. Runtime1 failure evidence was checked again. Only read/hash/text-analysis operations and this review-file write occurred. No tests, Cargo, HE/Engine/planner/constructors, source edits, network, Git mutations or subagents were invoked.

## Frozen custody and actual selected test

Verified hashes:

- Runtime report: `034639939efa12e1a72fe6c2a8601d025e8066e8d67752cace9e3d502abbe276`.
- Runtime2 raw:228,070 bytes, `0cc1cc336f129af8c58c7a6aabb8cdc02c185cbc75c76c891b199d407415efd2`.
- Derived parity.tsv: `e50975c7ed329a09b76701cca0729034270a55915849b89574977ad42be60a8a`.
- Runtime1 raw remains1965 bytes, `f1d14696c5cc7e4daeee089db83a34017b8b306310bd8535b759d32d8d4ef242`.

Runtime2 authorization binds the exact independently reviewed fix3 review, release, all8 source pins and source report by hash; all bindings still match. Terminal custody binds that authorization, exact command/environment, disk check, controller/Cargo PIDs, start/end, source-before/after, raw size/hash and exit. The controller makes one subprocess call, writes fresh evidence with exclusive creation, and has no automatic retry path. Its recorded test/controller exits are0. The root-held tool handle69253 is reported separately; local controller PID6642/Cargo PID6643 and terminal custody agree with the report.

All8 runtime2 before/after source hashes are identical to reviewed fix3 pins and to the current files. The selected command uses `--ignored --exact --nocapture --test-threads=1` for the one reviewed ignored lib test. Raw output explicitly says `running 1 test`, names `study_diagonal::p4_tests::p4_cpu_bridge_contract`, and ends `1 passed; 0 failed; 0 ignored; 190 filtered out`. Thus the selected ignored test actually executed; this is not no-run or an ignored-without-execution result.

The recorded29.76s test duration and controller timestamps establish execution custody only. They are not a comparative benchmark, server-update timing or scientific performance measurement.

## Independently checked actual counts

Every raw `P4_RESERVE` sequence is contiguous from before0, increases exactly once per operation, and stays at/below its reviewed cap. Totals independently extracted from all raw lines:

| Guarded actual operation | Runtime2 |
|---|---:|
| CkksParams / DFT proof / BootstrapMetadata |2 /1 /1 |
| KeyGenerator / Bootstrapper / Encoder |2 /1 /22 |
| Sessions / SignalOracle creations |20 /18 |
| Engine::train / trainer attempts |32 /34 |
| Actual bootstrap helper invocations |71 |
| Encrypted-state captures / unchanged comparisons |54 /268 |
| Positive terminal factory spy |1 |

There are20 completed-plan log rows. Their cumulative geometry/initial counters progress1 through20, update increments match records1 for smoke,4 for scheduled,2 for first-reference and1 for poison, and their final usage is20geometry/20initial/34updates/0handoffs/0wrappers. `P4_FINAL` agrees: successes32, attempts34, actual_boots71, F64=32, same planner usage. The final assertions are reached before that line, followed by the actual test PASS.

Session grouping independently confirms the fixed assembly:12smoke sessions each have1trainer call and1parity row; sessions13–16 have4each; sessions17–18 have2each; sessions19–20 each have1trainer attempt and no oracle/parity row. The first18 sessions account for32 successes and the last2 for the intentional failure paths. Modes are joined from the immutable reviewed assembly:20sessions I9/NI11; successful comparisons I12/NI20. The71 boots are all NI, including1 in the final poison session; no I bootstrap was logged.

These counters are reservations at the source-reviewed actual call sites, rather than a planned count summed after the fact. On the completed test paths, setup/trainer constructors succeeded, accepted steps returned success, and the bootstrap forwarding wrapper called the real implementation. The injected second refusal occurs before a second real operation. Source guards plus reached assertions support the stated constructor/helper inventory; they are not general-purpose allocator instrumentation.

The2 pre-parameter rate-policy log rows show the intentional canonical rejection and actual synthetic wrapper encoding. Their local two-call guard/final assertion completed. The16 negative preparation cases do not each print a dedicated outcome row, but their exact-error and zero-factory-call assertions, positive terminal control and final negative count are unconditionally reached before key setup; the completed test establishes those outcomes. This remains local synthetic shared-factory evidence, not a canonical validator/D7/start proof.

## Parity values and derivative joins

The raw contains exactly32 `P4_PARITY` rows. For every row, the derived TSV preserves the session ordinal, update index, exact printed maximum-error token and exact frozen-cap token. Family/mode labels agree with the reviewed closed test assembly and raw session boundaries; raw Engine context names independently agree with the family join. No row is omitted, duplicated, reordered or rounded in the TSV.

| Backend | Successful comparisons | Largest raw maximum_abs token | Frozen cap |
|---|---:|---:|---:|
| I |12 |7.64926433305390674e-9 |1e-5 |
| NI |20 |1.89993002996935445e-8 |4e-4 |

The I maximum is session13/update3 (PolyFourRate I); NI maximum is session14/update3 (PolyFourRate NI-WN). Every raw error is below its printed cap. Cap tokens encode the reviewed fixed F64 values; no tolerance changed after execution.

These maxima are computed by the actual client snapshot comparison over all logical W/b values, with full dimension/cardinality checks and mandatory finite assertions, against the existing actual Signal baseline Engine snapshots. All logical padded output channels are included. Frozen pure bias is checked exactly zero; encrypted bias uses the same fixed numerical tolerance. The successful raw32 rows therefore support this bounded all-entry CPU numerical result. They do not measure secureN16 precision, dataset accuracy, range/security/resource admission or general CKKS error bounds.

## Reachable acceptance assertions and limits of observation

The immutable source has no early successful return bypassing the contract matrix/final assertions. Each accepted step selects plan-bound rates/lookahead, checks entering state before setters, invokes the existing trainer once, compares all seven inclusive OpCounter fields directly to the bound plan, compares refresh/bootstrap/branch decisions, checks available hidden/AA diagnostics and all exposed post-state entries/cardinalities/scale bits/configuration, probes BatchSpec in isolated counters, and commits progress only after postchecks. Each success is unwrapped by the orchestrated test before its numerical comparison. Test PASS thus establishes those checked properties for these actual32 steps, not merely that parity lines were printed.

The268 complete unchanged-state assertions include232 first-update malformed-packet comparisons,2 later changing-rate rejects with velocity present,32 replay/exhaustion rejects and2 poisoned retries. Their54 snapshot guard records agree with the corrected source inventory. They compare raw optimizer configuration/factors and all encrypted model/velocity/compact limbs plus progress/counters/failure/trace state, without new training work. The valid scheduled update follows each later-rate rejection and passed.

The2 configured poison sessions reuse real training and permit one actual refresh/bootstrap before the second operation is refused. The reached assertions require trainer-phase failure at successful index0, unchanged successful progress, exact nonzero partial measured delta, one successful real forwarding operation and unchanged retained state/failure on rejected retry. I's nonempty current attempted trace is checked; NI's empty precommit trace is explicitly checked because its unchanged trainer publishes trace only at commit. No rollback or hidden partial NI trace is claimed.

The full typed P3 trace and literal internal kappa/mask constants remain unobserved. NI reference factors use the approved existing helper from actual configured state. Foreign-basis mutation is I-only, with the shared NI guard inspected but no separate NI foreign-basis negative asserted. Wrong-owner is a private association-token mutation, not an additional real-key experiment. The report states these boundaries accurately.

The52 BatchSpec probes,36 invalid label/cardinality requests,68 client inspect/decrypt endpoints,122 encrypted model snapshot copies and322 configuration collections are source-derived call inventories on a completed test. They are not separately logged event totals. The report correctly distinguishes them from logged guards and P4_FINAL. No scientific memory/RSS or endpoint-disclosure claim follows from those copy counts.

## Runtime1 preservation and cumulative execution

Runtime1 remains a separate failed actual invocation, test/controller exit101, with unchanged raw hash and source-before/after fix2 custody. Its failure is canonical full-maximum validation mistakenly applied to the distinct synthetic.04/reference.08 fixture. The raw shows6 completed plans; inspected error position after initial-state creation establishes7geometry/7initial but only6updates. Actual runtime1 setup was2parameters/1proof/1descriptor; all keys/HE/Engine/trainer/spy/sessions were0.

Adding that preserved partial work to completed runtime2 gives:

| Operation across runtime1+runtime2 | Actual total |
|---|---:|
| Parameters / proofs / metadata descriptors |4 /2 /2 |
| Geometry / initial / update plans / handoffs |27 /27 /40 /0 |
| Keys / Bootstrapper / Encoders |2 /1 /22 |
| Sessions / paired oracles / Engine updates |20 /18 /32 |
| Trainer attempts / successful updates / intended midwork failures |34 /32 /2 |
| Actual bootstrap invocations |71 |

The runtime report retains the first failure and includes its consumed setup/planner work; it does not substitute the successful run's counts for cumulative work. Earlier pure sensitivity Engine work is a separate historical validation phase; the32 paired Engine calls verified here are not a rerun of its mutant campaign.

## Verdict

The finite synthetic CPU contract has actual runtime acceptance and faithful evidence under the reviewed source. No correction or rerun is needed for this package. The runtime report accurately separates printed guards/results, assertion-backed outcomes, source-derived inventory and remaining unexecuted gates.

Canonical materialization/owned D7 gates/durable start, secure-profile resource/range/security/precision, production run_owned, P5/block, GPU, full/quick campaigns and scientific performance remain unaccepted. This review does not mark full P4/P3/D7/Task54/M4/M5 complete or authorize a publication/deployment action.
