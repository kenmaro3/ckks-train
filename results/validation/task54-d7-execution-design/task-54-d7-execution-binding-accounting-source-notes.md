# D7 source-brief accounting clarifications (draft, no source release)

Current source d0cf114; this note resolves how to carry design-review Minor M1/M2 into the implementation brief. Read together with the reviewed fixed design, once accepted. It does not ratify pending I1/I2 or authorize execution.

## M1: one additive preparation owner

Use one private preparation accounting value scoped to the actual prepared CPU route. Seed it once with the retained exact DiagonalPlan::charges().combined(); this existing number remains requested transcript+metadata payload, as documented by ReceiptLimits and ReceiptCharges. An Arc/borrow of the same plan/P1/context bytes does not charge a second payload copy. Newly allocated copies do. Keep original request max_bytes and max_records unchanged; do not reset or increase them to admit descriptors.

Track additional retained requested payload and maximum live temporary requested payload separately, checking their sum with the seeded plan charge before reservations/copies. Checked add/multiply must fail before allocation on overflow. New descriptor lengths/digest storage, checked geometry nested vectors, transpose lists, descriptions/activation vectors and per-stage arrays must be enumerated by the source report, with source-owned helper calculations. A size pass borrows real structures; it must not clone geometry/partitions before checking a budget. Temporary allowance is released at its actual scope end, not merely when logical construction advances. This budget measures requested payload, not allocator capacity or RSS: use vector element counts times size_of for in-memory retained metadata (including nested vector headers where newly owned), and exact bytes for serialized buffers. Retain allocator capacities as diagnostic observations if useful; never retroactively relabel the existing P3 requested-length budget as a capacity or process-memory bound. This explicit consistent requested-payload convention supersedes the proposal's ambiguous sentence requiring nested capacities as though existing P3 used that convention. Separate live resource admission still must cover physical memory.

The source brief must give a call-site allocation table for pre-key preparation and post-start validate_plan_binding: retained or transient, owner, checked size computation, reservation point and release point. Existing materialization/context bounds and parameter/P1 construction bounds remain their own mechanisms; identify their source owners and unresolved coverage honestly. No claim that the preparation receipt alone bounds these operations, target verification's independent materialization, encrypted array copies, key allocation or process RSS. Do not change existing canonical bytes or add a new scientific cap to solve accounting.

## M2: existing source commit accessor

Use context.artifact_context().source_commit(), already available in study_provenance/context.rs. No redundant commit accessor. If needed, add only a narrow borrowed current-Train-receipt source Manifest canonical-bytes view, retaining existing current-receipt and live rechecks. Do not select any historical Reference receipt.

## Scope and sequencing

Four new encodings require independently frozen design vectors before source. Pure representation and actual retained preparation/continuation are separate implementation layers of the same full D7 objective. A codec alone is not the production ExecutionBinding producer; a checked pre-key route is not a StartedHeCase; an owned post-start driver does not clear gates. Source phases may be separated for review, but the remaining concrete producer, route, gates/admission/start, scientific runner and full matrix remain open. Do not use descriptor completion to shrink Task54.
