# D7 phase2 preparation/accounting FIX1 report

Read the complete independent review, verified SHA256 `1fa646266d31593ef5ad0b8b0760617a88a5be2fb2ddd908d440724efb374b6f`, and inspected the cited current source plus all current prepare_then_make/prepare_blocks_then_make call sites. Original proposal bytes were verified and preserved unchanged as `task-54-d7-preparation-accounting-proposal-before-fix1.md`, SHA256 `e51db6786477a1271183f3c54f85cd9b02e5d7ece179bb41573ca389fa0df55e`.

## Important I1 disposition: corrected design, pending independent review

The original sentence keeping accounting only through a generic callback was insufficient. `prepare_initial` passes `Ok`, returns CheckedPreparation, and encrypts later. More deeply, `encrypt_initial` moves Description into EncryptedInitialState (`study_client.rs:502–509`); block encryption clones descriptions/classes into BlockEncryptionBinding and moves originals into EncryptedBlockInitial (`study_client/blocks.rs:328–345`). Session construction subsequently moves descriptions into the session (`study_diagonal.rs:405`, `study_blocks.rs:419`). Encryption is therefore not geometry destruction. FIX1 does not claim otherwise.

The revised ownership/signature section separates two independently parameterized lifetimes:

```rust
CanonicalPreparationInput<'input, 'params>
RetainedPreparation<'input, CheckedPreparation<'params>>
RetainedPreparation<'input, CheckedBlockPreparation<'params>>
```

No `'input: 'params` or reverse relation is imposed. A private shared checker for single/block takes the input, partition and `Option<&mut PreparationAccounting>` and returns CheckedPreparation borrowing only `'params`. It factors the existing validation/extraction rather than rerunning geometry, cloning it, or resetting the receipt.

Existing private callback wrappers keep their old signatures and call this checker with None. They preserve the existing legacy P3/allocation scope and never unwrap a new D7 retained owner. New retained preparation always seeds its original plan charge once, calls with Some, then returns checked state, input/case/request borrows and that same accounting owner together. Only this retained owner is accepted by the new descriptor collector. This is explicit compatible factoring, not an assertion that legacy outputs acquire D7 retention accounting.

Move prepare_initial's existing array preparation and encrypt_initial call inside its callback, returning EncryptedInitialState. Likewise remove the two synthetic Ok-return paths below. This audit prevents current callbacks from returning CheckedPreparation; the closure's private generic type is not claimed to provide a type-level nonescape guarantee. Future call sites must preserve this audited rule.

For the new D7 path, phase2 returns a private PreparedCanonicalBundle with descriptors declared before RetainedPreparation (whose accounting field is last), enforcing descriptor/checked-before-accounting drop order, or drops that complete bundle; it has no standalone checked extraction API. The fixed phase3 begin consumes that bundle inside the already-required genuine live frame: move the existing accounting value and descriptor ownership into LiveCaseFrame before passing checked fields to encryption. The enclosing owned driver retains both live frame and CpuOwnedSetup. Its setup/crypto must drop before live; descriptors inside live must drop before accounting. On setup error, live remains outside nested checked/crypto locals. execute_cpu similarly keeps live outside the nested borrowed session, binding and owned crypto scopes. This covers descriptions surviving inside initial state and session, and the separate block binding.

The block copy in current encrypt_block_initial requires reservation of both description nested payload and classes before cloning on the eventual D7 path; a fixed private accounted encryption helper may share the existing body with the legacy entry. It must not route the new owner through the unaccounted wrapper. Post-encryption geometry copies or later factory arrays retain the ratified separate stage accounting requirements. No change to public legacy encrypted return types, generic lease framework, receipt reset, geometry rebuild or readiness capability is needed for this correction. Phase3 implementation remains separately reviewed and is not authorized here.

## Complete current callback-result audit

| Caller | Current callback/result | Required handling |
|---|---|---|
| study_client.rs from_key_generator (~310) | Creates material; consumes checked via encrypt_initial; returns client/context/initial | Legacy wrapper contract remains; already consumes checked inside callback. Caller-owned keys already exist; no pre-key refusal claim. |
| study_client.rs prepare_initial (~419) | `Ok` returns checked, then encryption outside | Move existing arrays/encryption into callback. Return encrypted initial only. |
| study_client.rs prepare_owned_canonical (~805) | make_owned borrows checked, then encrypt_initial consumes it; returns client/context/initial | Existing disconnected legacy continuation remains; future D7 continuation consumes retained bundle directly with live accounting transfer. |
| study_client/blocks.rs prepare_block_initial (~226) | encrypt_block_initial consumes checked; returns initial/binding | Legacy unchanged; do not mistake these escaped descriptions for destroyed geometry. |
| study_client/blocks.rs prepare_block_owned_canonical (~585) | make_blocks_owned borrows; encryption consumes; returns client/context/initial/binding | Existing disconnected legacy path; fixed D7 body must retain accounting and reserve binding copy. |
| study_diagonal/p4_tests.rs spy_contract (~111,131) | Marker or unit; negatives assert zero callback calls | No checked escape; keep synthetic-only status. |
| study_diagonal/p4_tests.rs prepare_initial helper (~197) | `Ok` returns checked before encryption | Move existing encryption into callback. No test/crypto execution in this fix. |
| study_diagonal/p4_tests.rs setup closure (~676) | make_owned borrows checked; returns client/context | Checked geometry drops in callback scope; legacy synthetic scope only. |
| study_blocks/p5_tests.rs setup (~116) | make_blocks_owned borrows checked; returns client/context | No checked escape; legacy synthetic scope only. |
| study_blocks/p5_tests.rs spy_contract (~133,153) | Marker/unit after inspection | No checked escape. |
| study_blocks/p5_tests.rs per-case initial (~1005) | `Ok` returns checked, then encrypt_block_initial outside | Move existing arrays/encryption into callback; initial/binding retain their legacy ownership. |

The source package must compile a small lifetime-use check with independently short input and longer parameter borrow, plus unchanged wrappers. This is a compile requirement, not a claim that compilation was performed. The runtime inventory remains frozen only after source review.

## Minor M1/M2 carried into source brief

M1: binding workspace inventory now explicitly includes secure_parameters preset vectors and temporary extensions, the final validate_plan_binding plan-digest String after canonical_binding returns, and four retained 64-byte identity digest strings (contract digest clone and materialized digest included). Their actual live intervals determine the peak, even if some earlier maxima are dominated. Stable-sort internal library scratch is excluded from the stated application requested-payload convention, never called zero; pointer-vector payload remains counted. The successful size pass must allocate none of these. Narrow byte-preserving removal of allocations is permissible after source review; guessing an upper multiplier is not.

M2: DescriptorDigests and `_bytes_with` operations are module-private to execution_descriptors.rs and reachable only from its fixed collector deriving hashes from the same facts. They are not crate-visible caller-supplied source APIs. Existing source helpers still derive their own hashes. Size-only placeholders cannot reach output encoding. This confinement is not a new identity/capability framework.

## Scope, writes and verification

Changed proposal sections: independent lifetime signatures and shared-checker factoring; compatibility/new-owner boundary; concrete future live-frame transfer/drop order; confined digest cache; complete binding inventory notes; compile-only lifetime verification requirement. Original old-P2/new-retention distinction, exact-size approach, six-case conditional inventory, original limits and full Task54 scope remain.

Only these three named documents were written: unchanged original backup, revised proposal, this report. No source/Cargo/tests, constructor, planner, materialization, crypto/native operation, network, Git operation or subagent was run. Checks were text/source inspection and document SHA256/word-count verification only. No source/runtime release or observed six-case success is claimed; MissingExecutionDependency, genuine gates/start, production holds and full21/full2680+20+7/quick268 remain unchanged.
