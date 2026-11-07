# Independent linked Full synthetic oracle

This is an unratified test-data delivery for independent review, using the accepted Full wire and FIX1 evidence schema. No production encoder, parser, genuine reader, capability, resource observation, registry, materializer, HE or Cargo was invoked. All runtime/build/authority/scientific claims are fictional. There is no production allow-synthetic switch.

`inputs.json` and `full_grammar.py` were frozen before generation; `input-freeze.json` pins them. Grammar tables were copied from the previously accepted independent Full oracle, without its encoder. The newly authored `generate.py` emits primitives and offsets; `verify.py` uses a separate canonical interpreter and cross-record checks. Both share accepted grammar data, so verification is not an independent review of that grammar. New evidence bytes use the accepted ordered literal leaves, not a production encoder or prospective implementation output.

The family contains one seed0 reference/lock, one registered Full seed7 CPU interactive main case/execution, shared request/consumption, and three positive attempt scopes:

| Scope | Current admission/start | Prior admission/start in selected history |
| --- | --- | --- |
| A Initial | A/admission.json at40; A/start.json at50 | None |
| B Retry without prior start | B/admission.json at45; B/start.json at46 | A/admission.json; no A start in this snapshot |
| C Retry with prior start | C/admission.json at60; C/start.json at61 | A/admission.json and A/start.json |

B and C are separate history snapshots. An integration harness must select the snapshot, rather than enumerate every file in this delivery directory. In particular, the A/start file exists as later data but is absent from B's selected history. This package's folder structure is an offline fixture container, not a live attempt namespace.

Each scope has exactly `inputs.evidence.bin`, `implementation.evidence.bin`, `layout.evidence.bin`, `resource.evidence.bin`, `user.evidence.bin`. These five whole-file hashes populate the admission Gates, its gates hash is independently encoded, and start links that exact admission file/payload. Full start does not acquire new evidence fields. Authority scope/current records and claim records are retained source fixtures. They do not create a live lease, root authority, source-currentness proof or hidden filesystem token.

The reference's synthetic final accuracy is1/3, with two completed updates; the linked target is `max(0, 1/3 - 0.005)`, bits `3fd50369d0369d03`. Every Full evidence threshold uses those target bits. The selected receipt/consumption is seed7; the reference receipt remains seed0. The request permits up to four updates; consumed Full plan has the fixture's two complete updates. Input order/materialization and actual trainer execution are not performed.

`dependency-graph.json` lists record dependencies and the three history snapshots. Its edges include exact predecessor admission/start, authority/current, all five evidence, Gates and every FileRef retained leaf. The identity preimages under `retained/linked/` bind the existing small synthetic contract, receipts and materialized identities. Those accepted Full literals are opaque `FICTIONAL-*` identities, not valid real profile/seed/input payloads for a genuine materializer. Their equality and hashes are the supported private test-domain semantics here.

`negative/c1-prior-{lock,reference,policy}/` each contains a changed prior A admission, its rehashed five evidence files and start, and a B retry with its five evidence files/authority/predecessor links rehashed to that prior. Only the intended prior-target claim differs from the baseline target. The selected C1 history uses A/admission and B/admission/B/start, with no prior start. Prior start artifacts are retained for integrity inspection but not selected for this C1 test. The Python verifier establishes valid self-integrity, successful predecessor-only comparison, successful current-target comparison, then the one expected prior-target mismatch. A production helper test must independently reach that rejection rather than reject synthetic input earlier and call it C1 coverage.

The three other negatives distinguish exact file hash from payload hash and malformed integrity: whitespace-only admission JSON preserves its payload but breaks B's exact file link; self-rehashed start retains the correct admission file hash but a wrong payload hash; stale-self-hash admission changes the attempt without recomputing integrity. These are different expected rejection stages.

`manifest.json` pins generated raw files and98 canonical records; `resolved-literal-inputs.json` contains7,660 primitive paths/values/offsets/lengths. Every canonical record has `.bin` and `.hex`;21 supported document records also have exact `.json`. `run1-archive.zip` preserves all313 files from the first generation/verification before local claim-text and graph improvements. `invocations.json` is the cumulative author command ledger; per-run logs are not cumulative counters.

All78 old Full golden files and137 old gate-vector files are pinned in `preserved-before.json` and byte-identical `preserved-after.json`. This package adds integration coherence, not replacements for their enum/branch coverage. The existing six Full roles and ten evidence formats retain their separate coverage.

All buffers are small; no maximum-size allocation was attempted. Full C/R/A and record-specific limits remain unchanged. Verification checks all actual scalar/list/byte limits in the copied Full tables, actual raw/outer budgets, and new evidence A. It does not implement a hardened streaming parser or claim adversarial malformed-input coverage.

Before integration acceptance, independent review must approve these data. Prospective production codecs must compare exact bytes and enforce production synthetic rejection; private tests must compose current/prior target checks, evidence sidecar links, publication faults/custody and capability lifetimes. Genuine source-specific correctness/security readers, exhaustive resource modeling and current capacity/claim ownership, origin-current root authority, real input/case/plan readers and source/build/executable provenance remain separate obligations. The retained accepted stubs intentionally establish none of these. No resource-fit, gate-availability, scientific result or experiment authorization follows.
