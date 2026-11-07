# Task54 D7 gate schema FIX1 / 5 report

Status: ready for scoped independent re-review; I1 addressed in proposal/oracle only, not yet ratified.

Read the full independent review `task-54-d7-gate-schema-review.md`, SHA256 `d5f729d360344c86df35dd04e3dbc4d8a6e173d917117f539dcbefc2dabfd50e`. Exactly its Important I1 was addressed. No other architecture, policy, permission or science decision was reopened. Root review/diff originals were not modified.

## Preservation before edits

Before the first change, copied the original proposal plus the entire135-file vector directory to `task-54-d7-gate-schema-before-fix1/`. All136 copied files were byte-compared with originals, and rechecked after FIX1 against the archival preservation manifest. No archived source was edited. Original proposal SHA256 `3c0f489a1129cffbd37894daa0e2cb35d38328fb1fe3f43eeba90f68624767c3`; original vector manifest `8dd99cc4f068b2f3209bd497a06cc31a772227a5f49d6ae1111bfb7d30ff88bb`. Archive manifest SHA256 `a79fed34baaeda2fcb626bdd8166dcc1f4402da2597c0d706d622b3d19cba688`.

## Exact I1 correction

- Quick preflight now encodes argv as `List<Bytes>` with unchanged count≤256 and decoded argument≤4096 bytes, NUL rejection and exact retained capture ordering/equality, including argv0. NonUTF8 bytes are preserved; no UTF8-only restriction, normalization or replacement conversion is introduced. Empty captured arguments remain representable.
- The primary synthetic preflight is the positive representation oracle: its third argument is raw `--receipt=/synthetic/` followed by byte ff then `receipt.json`. `captured-argv.json` retains the exact fictional expected bytes; this is not a real process observation. The first two ASCII arguments also now have Bytes literal types; their length/raw primitive encoding is unchanged.
- Added complete `negative/argv-byte-tampering.bin`: only ff becomes fe, with identical argument/file lengths. Its expected failure is mismatch against retained captured bytes even when local payload digest is recomputed. The verifier confirms the single changed byte and inequality; no production parser/gate rejection was executed.
- Refreshed canonical preflight bytes/hash and raw JSON hash, dependent start preflight links, dependent end preflight/start links, their complete binary/hex/JSON files and derived event mutation vectors. Identity and all ten evidence vectors stay byte-identical, as do the original78 Full golden files.
- Refreshed literal types/offsets, manifests, generation/verification reports, README and checksum list. The post-generation verifier report is consistently excluded from generation checksums on repeat generation so a stale prior report is not pinned before overwrite; original report remains archived. This is regeneration bookkeeping, not an evidence deletion.

## Actually executed bounded commands and counts

Cwd `/Users/marconi/Documents/research/hekit`:

```sh
python3 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d7-gate-schema-vectors/generate.py
python3 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d7-gate-schema-vectors/verify_literals.py
```

Both exit0; each was invoked exactly once in FIX1. Original author invocations were generator2/verifier1; corrected cumulative author totals are generator3/verifier2. The independent review's separate read-only inline inspection1 is not a supplied-script invocation. Supporting stdlib archival, edits and report/hash reads are file-only work; no project encoder imports or source/runtime execution occurred. The generation report's `python_stdlib_generator_invocations:1` remains per invocation, not cumulative.

Current verifier:25 literal records,1,135 reconstructed primitive spans,131 generation-manifest file identities,135 checksum entries and three exact event-link sets; one explicit invalid-UTF8 positive representation and one single-byte mismatch oracle. Current inventory14 core vectors,22 concrete negative files and16 additional specifications. All78 preexisting Full goldens match frozen hashes. Boundary arithmetic stays allocation-free at the prior A/K limits.

Cargo/Rust/native/SSH/HE/real parameters/P1/planner/materializer/registry/observer/live gate constructors/production positives/scientific outputs: all0. No public-doc/source/old-fixture edits, subagents, commit or push. No gate/resource/science availability assertions. Existing21/full2680+20+7/quick268, holds, baselines, Task5/security/license obligations remain unchanged.

## Regenerated file inventory and exact pins

Compared with the original archive: 22 changed existing vector files, 2 added vector files, 113 unchanged vector files, zero removed. The proposal is additionally revised.

| Changed or added vector file | Status | Bytes | Current SHA256 |
|---|---|---:|---|
| `README.md` | changed | 4361 | `85d827e228b3925071bc18ac4b9333e980c13789e2cc942a538caf6e37b3fda3` |
| `SHA256SUMS` | changed | 12947 | `472b762defd7ca16ccd2f87cc5412c3f71cb2307544c122a9397cef2dcc33a7b` |
| `generate.py` | changed | 32046 | `e058ccebf6ff6831c2a3ba1837aa806715cce3ce45fc27621e51b7fa36c9229c` |
| `generation-report.json` | changed | 3144 | `616aff82bfc64f07531124b14f9155b3f3389dbe47d55f14bca4662035c3b639` |
| `literal-inputs.json` | changed | 278638 | `8cd5a3b684713c422b0fab01827272ac00744da6c84d7991dc5b85464ef62243` |
| `manifest.json` | changed | 207564 | `5e765a3ecce98341939dcb33078729ad71766a0b6912783ce579ead8f431ffe4` |
| `mutation-specs.json` | changed | 13976 | `0b69ae003b2a6688f1f18c5cd10f2701c20fdd5e1bfe55f76415b9e3f3ecf8df` |
| `negative/end-incomplete.bin` | changed | 954 | `67314c2f788e8c9c06217b70ebdb62bed90dd7a273947df71269b1728c4be5a7` |
| `negative/end-link.bin` | changed | 954 | `adeb0508ff17e9ad71e79d9eafaf201822850c98fe0bf33e10d6ae2e31b4c35f` |
| `negative/event-evidence.bin` | changed | 3411 | `c8b9896cde03faab0a9988ddeb6c866728a4917c04c966decf3c0884e5c78e3d` |
| `negative/event-link.bin` | changed | 802 | `1a750670c765c1905f4386014281e6133bd98fe3d868b6f7db160cfd61a367ce` |
| `quick.end.bin` | changed | 954 | `6c724624e846f7870da407641ff1f6e6d07357ec5e1aac087208a7ac0db0805d` |
| `quick.end.hex` | changed | 1909 | `c5ddbdd31e5ac22b59b36137eab147bb42fbcbcb8e299b4f5820ce261fb6a3c1` |
| `quick.end.json` | changed | 2163 | `b1c9df83963fcefe842b7ea1ada0468a02737bc700da2dceb65342a18f2313df` |
| `quick.preflight.bin` | changed | 3411 | `6fb9de4d3f3e8eed8cc09fdc1df420ec6ffba3bb0fc18aab411bda5879c0d941` |
| `quick.preflight.hex` | changed | 6823 | `f1bb0fbcd7b0667e9882b9b1a10d364e8dced17adcb231af20b92cae3a592b0b` |
| `quick.preflight.json` | changed | 7089 | `ac5b4fd71e3079fd0f42cf89510ebd2b3e8aff2e8bcd2e6da633efb0f7aa91b4` |
| `quick.start.bin` | changed | 802 | `d9c6619e1bdcae253d728ad7979dc10fef670b9602b4a6b0a0dd9c4273d7983e` |
| `quick.start.hex` | changed | 1605 | `ba7f307a2639e3aaa69d2eb88058523cb7e45a911e678890f0e7794eb96baf38` |
| `quick.start.json` | changed | 1863 | `f73899f149b74e0527c389d895a3f49ce29d2b692fc3b570b28f16785fd628df` |
| `verification-report.json` | changed | 455 | `e31746b7759e65c19c4226d9c34351eba327241634aad96699e9976c9dd2445e` |
| `verify_literals.py` | changed | 7275 | `80bf3ba242139bf579161409a24a7e5046319f0a5d2a122e3d4187ab7b44636b` |
| `captured-argv.json` | added | 506 | `25efd39ab02adb04acbbb5933461ec45fac13214acfd66a12371354e49a36856` |
| `negative/argv-byte-tampering.bin` | added | 3411 | `3ead0349443b51dc4cb86315dffc25573e3173c44f1103dd42edd79e9de4d686` |

| Main artifact | Current SHA256 |
|---|---|
| `task-54-d7-gate-schema-proposal.md` | `1e54707536ee52c481cfa789c308d9f4605f096a1a720649c2eebe7a1dc6f50b` |
| `manifest.json` | `5e765a3ecce98341939dcb33078729ad71766a0b6912783ce579ead8f431ffe4` |
| `generate.py` | `e058ccebf6ff6831c2a3ba1837aa806715cce3ce45fc27621e51b7fa36c9229c` |
| `verify_literals.py` | `80bf3ba242139bf579161409a24a7e5046319f0a5d2a122e3d4187ab7b44636b` |
| `verification-report.json` | `e31746b7759e65c19c4226d9c34351eba327241634aad96699e9976c9dd2445e` |
| `SHA256SUMS` | `472b762defd7ca16ccd2f87cc5412c3f71cb2307544c122a9397cef2dcc33a7b` |

Complete core vector and event sizes/hashes are in proposal §10 and the current manifest. Ready for scoped I1 re-review; no source or runtime ratification is inferred.
