# Task54 fixture custody and source-selection repair

This archive records a reviewed source prerequisite, not a native build, encrypted training result, or Task54 completion. The final authority is [FIX1 scoped review](ledger/task-54-d7-fixture-custody-fix1-review.md), read together with the [amended report](ledger/task-54-d7-fixture-custody-fix1-amended-report.md), [pre-edit disposition](ledger/task-54-d7-fixture-custody-fix1-brief.md) and [eight-file amendment](ledger/task-54-d7-fixture-custody-fix1-amendment.md). Specification and quality passed with the historical limitations below explicitly retained.

## Change and current evidence

The two existing fixture families retain all 158 payloads byte-for-byte in two JSON transport bundles, with their original adoption notes retained. A private test-only accessor and six consumer files use those exact strings. The selected source inventory decreases from 565 to 410 records; the production cap stays 512. All 49 existing consumer tests and the new all-member accessor test passed; strict Clippy and formatting passed on the unchanged Rust source. No HE, context, dataset, native execution or new dependency is included.

Eight crate LICENSE/THIRD_PARTY_NOTICES symlinks are now regular 0644 copies of the exact root notice bytes. This repairs the existing no-symlink source rule without excluding files or weakening policy. It changes Git blobs/types and requires synchronization when root notices change; it does not change license terms.

Fresh attempt02 made one pure preflight call over all 410 selected rows, accepted a 54,531-byte manifest under the unchanged 8 MiB bound, and propagated actual exit 0. Attempt01 stopped with exit 1 before preflight on the first license symlink and produced no result. Both attempts and full outputs remain separate. Root independently verified all 442 final handoff pins and the 410 regular files. The staged and committed source membership is checked separately at publication.

## Historical failures remain failures

The original handoff incorrectly called preflight/physical-tree checks GREEN: its retained 786,035-byte log is a traceback, and its earlier result JSON is stale relative to that driver. These artifacts and false initial claims are preserved for audit, not adopted as successes. The new bounded selected-row check does not establish equality with every file in the current Mac tree. A genuine clean fixed-recipe source_inventory/build remains unperformed.

FC-2 remains historically unmet: the initial accessor RED was an unconditional placeholder panic; final direct-accessor routing is GREEN-only. Root accepted current independent payload custody, actual cap/missing-bundle RED, preserved consumer assertions and independent source review after recording reasons and alternatives before correction. No retrospective test replay repairs that chronology.

FC-3 remains unavailable: the complete original E0618 compiler error output was overwritten. Partial author recollections and later successful compilation cannot restore it. Original reports, stale outputs, the pre-fix snapshot, both new attempts and the initial/scoped reviews are all retained unchanged. JSON here is test or operational evidence, never scientific measurement JSON.

## Custody

`archive-map.json` maps every copied artifact to its original ledger path and byte pin; `SHA256SUMS` covers all archive files except itself. Raw paths and historical status fields remain as emitted. The source base is ec2d2656f21e9139d3dde12abdce13d770ea2757; intervening 598ae07 changed observation documentation only. The full source diff and the narrower eight-type-change diff are both retained.

Remaining work includes the actual CPU supervisor/broker and build identity, retained input and same-owner gates, actual training/FP32/timing/publication/runner integration, native validation and the full registered matrix. FULL-2 and the existing queue order are unchanged.
