## Spec Compliance

**PASS for this source-only dependency. QUALITY: Approved. No critical or important finding.** No runtime acceptance or new execution release is given.

- The candidate implements the required actual path: `study_runtime.rs:267` retains the existing owner/context, selects the RegisteredCase from that root's registry, resolves that case's actual profile and checked seed, materializes once, calls capture/finish/component encoding in the same frame, and still returns `MissingExecutionDependency`. Errors remain explicit; no HE factory, scientific publisher, gate or admission/start capability is added.
- `study_provenance/train_capture.rs:8–21` defines distinct private-field capture/rechecked types borrowing the exact Train context, registered case and immutable materialization. Capture is available through `VerifiedStudyTrainContext`; finish consumes the first value. `train_capture.rs:72–110,139–173` checks all three exact pointers and current Train role/receipt/runtime recipe before repeating case/input/runtime checks. There is no parsed-claim, JSON, Clone, Default, Deserialize or public constructor path into either type.
- `train_capture.rs:44–71,111–173` checks canonical/shared input preparation, profile/seed/logical batch, backend/recipe and registered canonical-byte digest, reuses the existing selected-input facts/recheck, retains cwd/argv/thread environment/start executable, and compares invocation/executable and retained worker/Host correspondence at finish. The final artifact recheck preserves source/build/executable/approval/receipt validation.
- `study_runtime.rs:98` projects workers from the retained `HostExecutor`, after its recheck. `CaseRuntimeLease::facts` uses its own retained Host and brackets the projection with full lease rechecks. It does not use Reference scalar1, environment99, requested count, global-pool discovery or caller-supplied worker facts. The existing executor recheck retains exact owner/pool and queried worker-count checks; stable host identity remains cpu_model/OS/hostname, without converting thread telemetry into identity.
- `study_reference/document.rs:1969` connects only the rechecked type to the document producer. `document/producer.rs:79` obtains Train-checked current build/source facts and the retained Host/actual workers; `document/case_provenance.rs:35–71` reuses the existing decoder/validation/canonical encoding with bounded output bytes and hash. It produces an inspection component, not authority. The original Case schema/tags/order/caps and Reference producer/capture paths are preserved.
- Ownership is within the corrected release: eight existing files and two focused new files. The actual `study_provenance.rs` reexport wiring follows the append-only root correction; the nonexistent `study_provenance/mod.rs` is not introduced. No dependency, P4/P5 or unrelated numerical-source change appears in the complete diff.

**Cannot verify from this diff / current source-phase evidence:** actual approved Linux Train capture and the installed positive `run_owned` flow, live process/CUDA observations and full producer behavior with such a context remain unmeasured. The proposed 31 parent tests are not a PASS and do not create that evidence. Complete correctness of unchanged registry/materializer, input identity derivation, subprocess snapshot verifier, process/CUDA rechecks and nested document codecs is not established by this task review. These dependencies retain their existing validation obligations.

## Strengths

- `train_capture.rs:8–21,72–92,139–173`: explicit lifetime-bound owners plus exact pointer rechecks prevent replacing a same-shaped input/case/context with another object. The shared generic pointer predicate permits structural unit coverage without fabricating a verified context.
- `study_runtime.rs:98` and the lease fact view retain actual executor/Host provenance; the new `study_runtime/tests.rs:162` projection tests and the existing isolated environment99 test distinguish real pool2 from scalar1/environment99 and reject outside/foreign same-sized pools or changed owner.
- `document/producer.rs:79` adds a separate Train producer while leaving the Reference role guard and captured producer unchanged. The reusable JSON-to-component helper is a private formatting boundary; accepting JSON there does not reconstruct a live capture.
- `train_capture_tests.rs:8–49` explicitly separates compile-only actual-API wiring/negative trait checks from behavioral tests. `train_capture_tests.rs:50–138` reuses exactly two canonical Iris materializations for selection/source/receipt/owner predicates and labels sentinel tests honestly. The new codec test is explicitly synthetic and complements the existing exact raw-negative/golden regressions rather than claiming their transport coverage itself.

## Issues

### Critical (Must Fix)

None found in the bounded diff.

### Important (Should Fix)

None found in the bounded diff.

### Minor (Nice to Have)

No additional source change requested. The following evidence distinctions must survive later execution/publication:

- `train_capture_tests.rs:8–21` typechecks actual capture/finish/producer APIs but is not invoked by any proposed parent test. The canonical Iris test exercises shared predicates and owner sentinels, not an approved Train capture; the codec test exercises synthetic JSON, not the actual `captured_case` adapter. This is consistent with the released source phase, but a passing future pure batch cannot be promoted to Linux production integration.
- `document/case_provenance_tests.rs:167–190` deliberately checks only semantic negatives surviving Value roundtrip and its existing component predicate. Keep the existing 67 exact raw-negative regression selected; the new test alone does not cover duplicate-key/transport literals. The report already makes this distinction.

## Checks and evidence

The task-reviewer prompt was followed: the 1,193-line complete diff was read once in three consecutive chunks; no Git command or source edit was used. Outside-diff reads were restricted to these named risks:

1. **Retained source/executable/receipt validation might be bypassed by cached facts.** Inspected unchanged `context.rs:893–1068,1247–1297`: artifact recheck verifies runtime, environment, approval/receipt bytes, executable identity/hash and source snapshot; selected-input recheck enforces exact artifact context and recomputes contract/input/materialized/source/receipt facts. Capture/finish calls reach these checks. The Reference capture/finish role guards remain present.
2. **Actual worker/Host projection might accept a different executor or mistake telemetry for identity.** Inspected unchanged `study_runtime.rs:138–153,241–263`: exact pool-specific membership, owner thread, retained/current count, process and stable Host tuple are rechecked. CUDA remains delegated to its existing retained binding; its internals were not broadened or independently audited.
3. **The proposed test inventory might hide extra pools/threads/subprocesses or source work.** Inspected the relevant unchanged runtime regression bodies, `Host::observe` (`context.rs:1156`) and the `study_inputs::materialize` entry (`study_inputs.rs:454`). Host observation uses bounded local reads/direct libc observations, not a spawned uname process. The new canonical test has exactly two materialize callsites and no learning. Existing materializer/data-loader internals remain an unchanged dependency, not a new integration result.

Verified artifact identities:

| Artifact | SHA256 |
|---|---|
| Corrected source release | `a28ad4c1d6bdbc76f624330bd7d48e87b5603ab37bd941a7a1db2cf539674d28` |
| Source report | `54c2183d01d39bb0b0daf663f3763be40387d19f91e812d911bf1b332cc98388` |
| Complete review diff | `065d0f17bb2bb36b55cdcecfea305cd681d13bd1ad01202afda5ccd454e224db` |

All ten current source hashes match `task-54-d7-train-capture-source-validation/source-pins.json`, against reported BASE `9aba62f0a2a2b565466216511238dec3fd7a244f`. The unrelated prepared-request commit is outside the candidate. The diff's introductory tracked-only stat lists eight files; the complete body and pin inventory correctly include both new files as well.

Retained command metadata/raw evidence supports a genuine compile RED for the absent capture/facts/codec APIs (`01-red`,10 compiler errors), final parallel no-run GREEN (`12-no-run-final`), strict clippy GREEN (`11-clippy-final`), serial check GREEN (`05-check-serial`) and parallel+CUDA check GREEN (`06-check-cuda`). Earlier redundant-closure/needless-borrow lint failures are preserved; the final callable cleanup is in the diff and final clippy output is clean. These are build/static checks, not executed unit tests or native CUDA availability evidence. No check was rerun by this reviewer.

The retained preservation script/result checks 256 outside-owned tracked bench/Cargo paths and 82 Case golden files against BASE, plus byte-preserved Reference capture/finish/helpers and Reference producer, and unchanged preexisting Case/document prefixes. This is preservation evidence, not proof that every unchanged dependency is correct. Its `runtime_tests_executed:0` field is a recorded source-phase statement, not an executed assertion. The source-phase command inventory contains no test execution.

The proposed manifest has31unique exact parent names:5new/26regressions. Its independently summed planned work is2canonical materializations (Iris0/7 once each),11pools with workers `[1,1,2,2,1,1,2,2,2,2,2]`,10HostExecutor objects,2explicit std threads and1subprocess. The environment99 parent requires the child key unset and its single child creates pool2. The separate proposed doctest filter has8compile-fail/1compile-pass source examples; no doctest or parent unit test ran here. Broad module filters or additionally selecting the explicitly excluded materializing regressions would exceed the stated inventory.

## Assessment

**Task quality: Approved for source review.** The actual retained-owner path and private inspection boundary match the requirements without broadening Reference authority or inventing readiness. Runtime acceptance remains a separate root decision, with actual Linux integration still required.

The full scope remains all21profiles/full2680+20+7/quick268, distinct Reference-only publication, unsupported backend/route states and existing campaign holds. Physical/runtime/execution descriptors, five live gates, target/reference request plumbing, admission/start/durable receipts, pre-key consuming handoff, HE continuation and scientific outputs remain missing dependencies. No full Task54, HE/gate/start/security/resource/scientific acceptance or subsequent execution is claimed.
