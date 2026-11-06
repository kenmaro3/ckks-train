# Frozen branch map

All entries refer to literal files in `expected/`; canonical component files are named explicitly. A codec vector is not approval, reference execution, registered case membership, gate readiness, or a live capability. All domains/markers are test-only except exact auxiliary hash domains and explicitly negative files.

| New branch/field | Literal vector(s) | Classification |
|---|---|---|
| target_lock=0, he_admission=1, he_start=2; complete=0 | lock, initial, start-initial | synthetic transport |
| synthetic=true=01 | all 14 primary envelopes | synthetic transport |
| synthetic=false=00 | test-header-false.negative; production-header-false.negative | explicitly untrusted negative, never approved |
| Case origin family=0 / supplement=1 | branch-0 / branch-1 | tags only |
| scope full=0 | branch-0 through branch-4 | tags only |
| role main=0 / baseline_a=1 / baseline_b=2 / baseline_c=3 / main_ablation=4 | branch-0 / 1 / 2 / 3 / 4 | tags only |
| backend cpu=0 / cuda=1 / external_cpu=2 | branch-0 / 1 / 2 | tags only |
| mode interactive=0 / noninteractive=1 | branch-0 / 1 | tags only |
| placement client_refresh=0 / when_needed=1 / after_update=2 / after_activation=3 / upstream_verified=4 | branch-0 / 1 / 2 / 3 / 4 | tags only |
| packing batched_diagonal=0 / row_2020=1 / diagonal_2020=2 / alternating_reboot_row_column=3 / official_upstream=4 | branch-0 / 1 / 2 / 3 / 4 | tags only |
| security n16_train=0 / n16_boot=1 / seal_2020=2 / upstream_verified=3 / reduced_level_supplement=4 | branch-0 / 1 / 2 / 3 / 4 | tags only |
| Case eval_config null=00 / present=01 | branch-0 / branch-1 | tags only |
| Case Eval hoist false/true; bsgs false/true; lazy_relin false/true | branch-2/1; branch-1/2; branch-1/4 | tags only |
| implementation main_diagonal=0 / matched_row2020=1 / matched_diagonal2020=2 / packed_c=3 / external_b=4 | branch-0 / 1 / 2 / 3 / 4 | tags only |
| bootstrap none=0 / host=1 / native_cuda=2 / upstream=3 | branch-0 / 1 / 2 / 3 | tags only |
| bootstrap_descriptor null/present | branch-0 / branch-1 | tags only |
| Execution Eval hoist false/true; bsgs false/true; lazy_relin false/true | branch-0/1; branch-0/2; branch-0/4 | tags only |
| Named §8 backend CUDA=01 / NI=01 / after_activation=03 / batched_diagonal=00 | branch-5; case-5.canonical.hex | exact four bytes at field offsets, synthetic only |
| All five Gate fields: inputs, implementation, layout, resource, user (each revision + evidence) | gates-0.canonical.hex; initial | fictional claims, no gate capability |
| predecessor null | initial; predecessor-null.canonical.hex (00) | synthetic initial branch |
| predecessor present, prior start absent | retry-no-start; predecessor-no-start.canonical.hex | synthetic snapshot branch |
| predecessor present, prior start present | retry-with-start; predecessor-with-start.canonical.hex | synthetic snapshot branch |
| Reference captured=0; case captured=0 / role study_train=1 | lock / initial | fictional provenance; unavailable top-level not legal and no positive vector |
| All five auxiliary hash domains | profile-slot, lock-id, registered-case, case-binding, admission-gates canonical.hex | exact ratified `hekit-study-*` domain bytes, fictional inputs; not capabilities |
| Clock wall unix_system_time=0 / mono std_instant=0 | lock / initial / start-initial | fictional integer time claims |
| Complete ordered endpoint + counts, boundary list, Accuracy | reference, lock | tiny fictional values only |
| Endpoint exact boundary false / unavailable update and eval timing | lock-endpoint-negative | semantic negative (must not verify as lock) |
| Endpoint exact boundary true / available update and eval timing | lock | synthetic transport only |
| Accuracy exact ten literal binary64 cases and Target clamp/subtract | binary64-components.json + binary64-vectors.json | arithmetic only |

## Inherited variants and byte preservation

`initial` and `reference` retain raw Unix encoding tag0, UTF8 `/fictional/exe`, an empty argv element, non-UTF8 `ff00`, UTF8 `c3a9`, and non-UTF8 absolute source root `/fictional/ff`. No OS path is opened. Raw bytes are not normalized or lossy-decoded. `branch-0/1` cover build.debug false/true. `branch-0/1/2` tracked covers clean0/dirty1/unavailable2; untracked covers dirty1/unavailable2/clean0. These dirtiness/unavailability branches are semantic negatives for production provenance readiness.

Commit Observation available0/unavailable1 appears in branch-0/1. Every host optional field (`cpu_model,os,hostname,available_cpus,process_id,observed_threads`) is available in branch-0 and unavailable in branch-1. Host execution_threads is positive2 in case provenance and1 in reference provenance; neither is measured. Every thread-environment key is present in sorted order; available/unavailable value branches both appear. Features and source manifest arrays preserve sorted UTF8 ordering. Provenance top-level unavailable is forbidden for target/receipt positive types; the package does not invent that branch.

The linked source is a separate private `hekit-test-reference-document-v1` fixture with synthetic=true. Its opaque contract/seed0 bytes are identical to the lock's retained contract/receipt digest; seed7 has different explicitly authored receipt bytes and the same contract. This demonstrates byte linkage only, not a canonical materialization or seed derivation. Retry-no-start and retry-with-start are alternative observed fixture snapshots of initial attempt state, not one executed filesystem chronology. Later initial start has time50; no-start retry captures time45; with-start retry captures time60.

## Explicit rejection data

`expected/error-expectations.json` lists each exact mutated JSON file, expected layer/category, file SHA256, mutation recipe and `executed_against_production:false`. It includes escaped duplicate key, unknown field, unsupported domain/hash, wrong payload/identity digest, wrong exact source link, mismatched nested source schema, unchanged value/changed bits, coherent one-ULP target, prior-start digest mismatch, escaped large hex, production-header/true, test-header/false, and production-header/false. Categories are independently specified expectations, not observations of API return values or claims about error precedence.

`binary64-vectors.json` additionally gives literal out-of-range counts, negative zero, both one-ULP neighbors, F32 rounding, the discriminating exact-rational 1/199 minus 1/200 substitution (3efa589c04bbec09 versus ratified 3efa589c04bbec00), and nonfinite bits. Negative data does not imply a production parser was run. Branch admissions deliberately mix axes to cover tags; registry membership is not asserted and several combinations must fail future semantic admission even if correctly encoded and rehashed.

FIX1 classification clarification: `production-header-false.negative.json` retains the private nested `reference.schema`; it is already production-schema-invalid and also contains unapproved fictional claims. Its expectation is nested-schema rejection, not isolation or evidence of an approved-context check. No literal bytes were changed for this correction. The superseded 199/200 rational label is preserved only in archive/pre-fix1.
