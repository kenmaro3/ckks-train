# Task54 D7 resource FORMAT FIX1 — scoped independent rereview

SPEC: **PASS (single-finding format rereview)**.

QUALITY: **PASS (single-finding format rereview)**.

Finding 1 in review `dac82a73` is **ADDRESSED**. No new blocking issue was found in the changed text. R3's remaining supplied-input hash-role gap is closed; the prior R1/R2/R4/R5/R6/R7 dispositions remain unchanged. This is not a broad architecture rereview or authority to generate vectors, implement source, validate a model or execute HE/runtime work. Root ratification/resumption remains separate.

## Exact inputs and verification

Read the original review first, then the complete scoped diff, changed contract paragraph, appended closure report and fix pins. Independently verified the following:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| Final format contract | 30832 | `a7c947fceca6cf7bb7415114838de0de87c7bbe21657efef4da792b4bd392429` |
| Appended closure report | 13775 | `77e2fc1de6f9f7cfa36c89ff6475f72ea5ff81ee4c9c0e6e14c2e4266ece33c9` |
| `task-54-d7-resource-format-fix1.diff` | 12048 | `35adb51a8288b21aa52cf0be5729e0773ba444b43a28cf0e8d236ac23f5af9dd` |
| `task-54-d7-resource-format-fix1-pins.json` | — | `c2bcc34e7d5aa05da8e6a46a9157dac86781c4dee27500205a4d819de8470192` |
| Original format review | 11817 | `dac82a73edf6dfa800ac01dce63ec70b0e5b75ccd020cfba4be43ff08ee5f43e` |

Read-only document bookkeeping verified all nine artifact length/hash entries in the fix pins, including immutable proposal e72b99a7, normalization report6c8bb195 and accepted map31504e7c. The preserved contract/report snapshots match `7f9130f56d629247ec08fe0ed0739e984168caae0eaecdfe77e51f2898318044` and `5298ef3a3203f1b140683ca3bc481c75179aeb4cf90bb3fd3d7ae665f23775d2`. Independently reconstructed both revised files from all four diff hunks, checking old/context bytes and counts. The original report is an unchanged byte prefix of the appended report. This was document-only verification, not a project test, helper or corpus generator.

## Finding 1 disposition

Final contract lines 102 and 104 now select exactly the supplied retained `MaterializedProfile.identity().digest()`, the existing D6a combined materialized identity. Some is derived only from the actual SAME live owner. Once its selected receipt is derived, the value must equal both that owner identity and `supplied_receipt.materialized_digest()`. The text expressly excludes the receipt's complete canonical-byte digest and the shared contract digest, removing the ambiguity that blocked independent literal expectations.

The source relationship is real, not a new identity definition. At the same fixed BASE `3a65f42dac83a17112c84dc3f23d3eb0aae7d0a8`, read-only `git show` of `crates/ckks-train-bench/src/study_reference.rs:738–758` confirms that `derive_identities` stores `p.identity().digest()` in the selected receipt's materialized field. The original review's fixed-source accessors (`study_inputs.rs:285–289,388–390`, `study_reference.rs:95–103,114–133`) distinguish the three meanings; no moving source was used.

The fix also preserves the important surrounding checks:

- Independent actual/selected reconstruction and complete `actual_receipt.canonical_bytes() == supplied_receipt.canonical_bytes()` comparison remain mandatory. Materialized-hash equality cannot replace them.
- Some still requires an already-existing input under the SAME owner at selection. Future construction remains None, is modeled before allocation, and does not rewrite the immutable claim afterward.
- Target lock remains seed0 with null target and supplied_input. Other operation/option predicates are unchanged.
- Future oracle distinctions name D6a versus receipt digest, contract digest and a wrong digest, but the prose does not claim those vectors were generated or checked.

This is the bounded remedy requested by the original review. It reuses an existing accessor/identity and adds one explicit equality obligation. It introduces no new field, preimage, digest algorithm, outer wire or authority.

## Changed-text scope and limits

The contract delta is confined to FIX1 status, this one field's exact meaning/equality and source evidence. The report delta is append-only, preserves the original submission and accurately keeps acceptance pending independent review. No change was introduced to existing outer Resource/FileRef/Full/Quick formats, caps, bootstrap arithmetic, model obligations, thresholds, original holds or all21/full2680+20+7/quick268 scope. No broader investigation or alternative architecture was needed.

Actual physical model/bootstrap bounds, host/current observations, helper lifetime, same-owner typed wiring and CPU/GPU/external/baseline execution remain outside this format PASS. Missing evidence remains missing. The independent oracle and source work need their existing separate release steps.

Only `task-54-d7-resource-format-fix1-review.md` was written. No source/Cargo/publicdocs change, project helper/generator/test/--list, native/runtime/HE execution, SSH, data/process scan, subagent, installation, spending, commit or push occurred. No literal vector was created. Reviewed inputs, snapshots, proposal and earlier reviews were left unchanged.
