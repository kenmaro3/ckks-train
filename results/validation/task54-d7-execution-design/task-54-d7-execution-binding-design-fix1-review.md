# D7 execution-binding design FIX1 — scoped independent re-review

2026-11-07. **SPEC PASS / QUALITY PASS for the scoped design FIX1. Important I1 and I2 are addressed. No new Critical or Important breakage was found in the changed material.** This is a design/vector review result, not root ratification, source authorization, native/runtime evidence, gate clearance or an execution release.

## Reviewed scope and pins

Read the Task 54 brief and execution-binding design brief first, then the original review, FIX1 report, complete FIX1 diff, revised proposal and vector contents. The scope is the original I1/I2 resolutions and new breakage in their changes. Untouched design areas were not reopened; original Minor M1/M2 remain source-brief carry-forward rather than separate approval tasks.

| Artifact | SHA256 |
|---|---|
| Original review | `6fd9e31175b729012660c9aa22dcc31e22ceac713c54b7625eea6876af960f0d` |
| FIX1 report | `5b784df466718d711731f299332a97e95041a296492b74c31a10bf53b0bfd798` |
| Revised proposal | `0b4c92c2f3723589ec9006089e0999255b6b209b8286ef097e4aee74abe289fe` |
| Vector `SHA256SUMS` | `899b45991113581638a5758bd9bdaec441df768e771fa706a16549faf93b287e` |

The last three pins were checked by read-only hashing here. Root reports the preserved original proposal matches `25d88851549d4ad85989572b7a541745dd2f240545f434b490bae7fb0cedebe0`, and that its read-only length/digest check passed all 70 saved fixture images. Those repetitive artifact checks were not rerun. The 19-file checksum inventory remains the reviewed vector-package boundary; it is not an execution evidence identity.

## I1 — addressed: owned returned driver, local borrowed sessions

Revised proposal lines 85-107 supply the missing concrete ownership handoff while retaining the ratified `begin_checked_he<'s,'c:'s,'p:'s>(StartedHeCase<'s,'c,'p>) -> HeSession<'c,'p>` signature:

- `HeSession` is an owned driver carrying a live frame plus independently owned crypto/initial packets. It does not return a P4/P5 session borrowing one of its own fields (lines 95-97).
- Context and input/setup-owner borrows have separate `'c` and `'p` lifetimes. The named current API supports the claimed separation: `CheckedMaterializedInput` stores a context borrow and owned facts, not an input borrow (`crates/ckks-train-bench/src/study_provenance/context.rs:953-979,1021-1024`). Therefore retaining it for `'c` does not require inventing `'c:'p` or extending the single-lifetime TrainCapture.
- Begin retains the store borrow/held handles through rechecks and checked ownership transfer, explicitly calls `validate_plan_binding` after durable start and before keys, accounts its temporary allocations, releases the store only after transfer, and rechecks the retained live frame immediately before crypto allocation (lines 99-103). This resolves the previous omitted call and preserves the consuming failure semantics of target-design sections 2 and 7.
- Private consuming `execute_cpu` creates the refresher, P5 facade and borrowed P4/P5 sessions inside nested local scopes, then drops them before their crypto owners and live leases (line 105). No borrowed session escapes through the returned driver or the installed executor's plain outcome. This fits the actual borrowed constructors in `study_diagonal.rs:372-395` and `study_blocks.rs:205-221,323-343` without a self-reference, public callback, leaked owner or allocator permit.
- The missing real gates/start constructor remains explicitly unavailable (line 107); disconnected fixed helper bodies cannot create a fake live driver. P5 snapshot remains success-immediately only (line 105).

This is a viable architecture sketch, not a claim that the future Rust ownership code compiles. Its source acceptance must enforce the stated drop-order invariant also when a returned driver is dropped without execution and when setup fails. In particular, source field/declaration order or explicit destruction must keep crypto teardown before live resource/user lease release; the prose listing `live` and `setup` is not permission to rely on an incompatible default field-drop order. This carries the resolved I1 lifetime requirement into implementation, rather than adding another design gate.

## I2 — addressed: complete independent byte oracle and sequencing

Revised proposal lines 115-119 now explicitly require independent vector review/freeze **before** hash-producing encoder source. The four unchanged domain/field tables at lines 29-73 have complete saved byte examples and declared failure specimens. The standalone generator was read, not executed or imported; it uses only Python stdlib and literal values (`design-vectors/generate.py:1-20,148-209`). No repository encoder, parameter constructor, planner or source test was involved in this review.

Independent content inspection used a separate read-only decoder, written inline from the field tables, with bounded reads, explicit tags, lowercase Hash validation and exact-end checks. It decoded the saved `.hex` files directly, without importing the generator or consuming its offsets as the parse specification. **All 14 complete positive images decoded to exactly their full `inputs.json` field values.** This checks actual field meaning/order beyond matching lengths or digests.

| Domain | Inspected positive images | Representation coverage and result |
|---|---|---|
| I parameters | `parameter_dense`, `parameter_fixed` | q/p and bit-vector order, exact IEEE scale bits, dnum, None/Some main weight. Both match. |
| Layout | `layout_i_single`, `layout_ni_two`, `layout_i_three` | Full/Quick, I/NI, 1/2/3 stages, ordered negative I32 chains, each mask's step lists/Bool, compact absent/present, container layer None/Some, partial partitions. All match the unchanged P3 body order. |
| Key policy | I/NI × Dedicated/Reuse | Ordered reuse pool, exact training union, separate generated/executed bootstrap sets, bootstrap option and flag combinations. All four match. |
| Security | I dense/fixed; NI absent/fixed/dense ephemeral | Secret versus Option tags, None versus Some(DenseTernary), assurance0, optional scope0 and all nine limitation tags. All five match. |

The read-only decoder also inspected **all 56 saved complete negative images**. Every malformed-wire specimen rejected under its designated structural rule; every semantic/source-binding specimen remained structurally readable and changed the declared field(s) relative to its base. For replacement-input specimens, decoded fields exactly matched the saved replacement input. The generator's mutation declarations were checked against those outcomes (`generate.py:220-290`):

- Header/domain, text/list length, unsupported Bool/Option/enum, stage cardinality, uppercase Hash and trailing-byte failures are representation failures.
- Empty/mismatched parameter chains, invalid dnum/log N/scale/weights, invalid or discontinuous parts, rotation zero/duplicate/superset, generated/executed substitution and missing NI descriptor all have the stated invariant or exact-source mismatch.
- Ordered transpose-chain/reuse-pool changes, an otherwise valid changed ephemeral weight, reordered/omitted limitations and None versus Some(DenseTernary) are comparisons against unchanged retained source facts. They are not forbidden solely because a digest changed. A fresh source object with different permitted fields would require its own binding. The package explicitly fixes this distinction.

The additional families at `generate.py:290-297` precisely specify every proper prefix, A_MAX−1/A_MAX/A_MAX+1/u64::MAX size inputs, and overflowing add/multiply/32-bit-usize conversion. These are reviewed rejection/size-guard expectations; this review did not allocate oversized fillers or execute a repository parser. Size-guard acceptance at/below A_MAX does not imply a valid descriptor.

All positive vectors remain synthetic transport oracles. Literal `a`/`b` digests are placeholders, hand-written geometries do not prove a StageGeometry, fixed main/non-default ephemeral examples do not satisfy registered production policy, and Some(DenseTernary) ephemeral is not produced by the current P1 mapping. These distinctions are explicit in inputs/README and FIX1 report lines 39-52. Future source tests must separate pure wire examples from actual-owned producer acceptance; these vectors cannot mint checked preparation or readiness.

The independent content inspection found no domain, primitive width, tag, nesting/order or body mismatch. No further representation constant or complete fixture is required to close I2 within the reviewed scope. The vector family is appropriate for root freeze before source.

## QUALITY, carry-forward and evidence limits

The complete diff changes the requested ownership, post-start revalidation and before-source vector sequencing; it preserves the original four encoding bodies and complete coverage/hold statements. Reuse of `artifact_context().source_commit()` resolves the redundant-accessor direction. The requested-payload clarification is consistent with preserving old ReceiptLimits semantics; M1's allocation table and source implementation remain work for the source brief. The accounting draft received no separate independent approval verdict in this review.

No new unrelated finding is entered from this scoped pass. No source/Cargo mutation, repository test, constructor, HE/native execution, network/Git operation or subagent was used. The only executed inspection beyond reads/hashes was the independent read-only literal-byte decoder described above; it wrote nothing and proves no Rust/runtime behavior. Only this review report was written.

Source review, bounded source tests, actual retained producer/route integration, live gates/resource/user leases, durable admission/start, native Linux evidence and scientific output remain separate requirements. Existing P1/P3 limitations and full encrypted-block user hold remain. Preserve all 21 profiles, full 2680 family rows + 20 exact-Iris prospective rows + seven supplement entries and 268 quick projections, with unsupported CUDA/native/external/matched/packed adapters still visible. This PASS closes only the scoped design-review I1/I2 findings.
