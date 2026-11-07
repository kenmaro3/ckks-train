# Task54 D7 custody inventory delivery report

Status: DONE as a proposed inventory; ready for independent review, not yet ratified or executed.

Read the parent Task54 brief, complete custody-test inventory brief, ratified gates design §§6–7, complete custody proposal, all of publication.rs/publication_tests.rs and target.rs/target_tests.rs, and the actual Full target/predecessor/start helper bodies. Existing implementation is in publication.rs; no native.rs/protocol.rs was assumed. HEAD at initial inspection was `1e65598b73a3ff2fd407daa362ae1beaa423e02c`.

Deliverable: `task-54-d7-custody-test-inventory.md`. Its source-pin table and final drift note identify exact read bytes. Main stable source pins:

- publication.rs: `8cfabba4dcf7daae93e90ce40a0bc1bcf13c59c713bb5c9d65964b97a4a2c004`
- publication_tests.rs: `10f640a0961da262ddcd15627a5a1c59455b1fddbee7cc5c9a4711b6eff33fd6`
- target.rs: `1c54c5918d0b73dabcf213bf43dacdaa3682951f471dd48a0c6de7579ad49d1b`
- target_tests.rs: `7a3c34222b70a9ec8cf02a46033d995c120d7fe18bbdceb0ea96aba6895806e3`

C01–C10 now have named future test filters, explicit fixture sets, small buffer/read accounting, seven-stage publish traces including cleanup, exact directory creation/sync actions, deterministic fault/crash indices, recovery sync vectors, ownership/concurrency/drop obligations, and separate native Linux requirements. C1 composes current and prior admission target verification against the same retained target/reference/policy before predecessor checks, with changed-prior-lock/reference/policy refusal cases. No operation counts are mislabeled as actual OS syscall counts.

Minimum missing private seams are explicit: ordinal test recorder, retained-FD read helper, exclusive named-attempt directory creation, fixed multi-file recovery sync sequence, and narrow native internal fault boundaries. Existing publisher pre-link/visible-uncertain semantics and temp ownership are preserved; in particular a successful link side effect must not be injected as a pre-link Ops::link error.

Concrete concern: new Full evidence vectors do not form a linked bundle with old Full admission goldens. The inventory separates filesystem-only synthetic ordering controls from actual existing pure codec composition. Combined codec+protocol acceptance needs a separately frozen coherent independent Full fixture; do not invent linkage or modify old goldens. This is a fixture duty, not an architecture/authority change. Quick's accepted linked FIX1 fixture is already available.

Commands performed were read-only cat/sed/rg/wc, `git rev-parse HEAD`, `git diff -- crates/ckks-train-bench/src/study_reference/document/admission.rs`, and `shasum -a 256` of named files; exit0 on final inspections/pin checks. One stdlib file-edit command corrected draft arithmetic/wording only, no executable fixture generation. No Cargo, tests, --list, runtime, generators, native observer, SSH, HE, real parameters/P1/planner/materializer/registry, subagents, source/public-doc edits, commit or push. Only these two ledger inventory/report artifacts were written. The concurrent codec source change was observed read-only and remains its owner's separate responsibility.

All numbers in the inventory are future harness contracts or read-only fixture lengths, not tests passed, real gates, resource fit, native lifecycle success or scientific results. Existing matrix/holds/baselines/security/Task5/license boundaries remain unchanged.
