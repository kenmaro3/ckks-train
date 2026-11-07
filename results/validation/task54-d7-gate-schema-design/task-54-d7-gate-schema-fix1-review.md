# Task54 D7 gate schema FIX1 independent re-review

Reviewed 2026-11-07. **SPEC: PASS. QUALITY: PASS. I1: RESOLVED.** No remaining findings in the scoped correction. This approves the design/oracle correction for root ratification; it is not production source, genuine evidence applicability, live authority or a scientific result.

The original independent review's sole Important I1 is the entire re-review scope. Unaffected architecture, scientific policies, user authority and evidence obligations were not reopened. Read the complete FIX1 report and changed proposal/generator/verifier/README sections; independently inspected all changed bytes and the complete diff against preserved before/current artifacts.

Exact reviewed pins:

- FIX1 report: `326742ba143543cfd4e24afbd6e037bf4098a28cc65d21ae0eccf93e0b950c34`.
- Complete FIX1 diff: `1c5e8e5c51b4ab44e30f4f7c3d9fb1e7ff6d18f72963f6486338647232f5730d`.
- Current proposal: `1e54707536ee52c481cfa789c308d9f4605f096a1a720649c2eebe7a1dc6f50b`.
- Current vector manifest: `5e765a3ecce98341939dcb33078729ad71766a0b6912783ce579ead8f431ffe4`.

## I1 disposition

Proposal §6 now specifies `argv:List<Bytes≤4096>` and explicitly preserves count≤256, no NUL, exact argument order, argv0 and captured nonUTF8 bytes. It permits empty captured arguments and forbids UTF8 validation, normalization or lossy conversion. This corrects the incompatibility with existing `args_os`/`Vec<Vec<u8>>` capture without narrowing accepted invocation transport. Existing A/K caps are unchanged.

The generator's `CAPTURE_ARGV` and `event` now use `r.blob` for every argument. The first two ASCII arguments retain the same length/raw bytes primitive representation, while the explicit third fictional argument contains invalid UTF8 byte ff. `captured-argv.json`, literal Bytes values, binary preflight and hex JSON transport preserve that byte exactly. This is clearly labeled synthetic representation, not an actual process observation or accepted frontend command.

`negative/argv-byte-tampering.bin` changes exactly one byte, at absolute offset3044, from ff to fe. Its argument length prefix and whole-file length are unchanged. The mutation remains well-formed Bytes but differs from the retained fictional capture; the frozen expected refusal is exact argv mismatch even if a local payload hash is recomputed. Negative SHA256 is `3ead0349443b51dc4cb86315dffc25573e3173c44f1103dd42edd79e9de4d686`. Neither the author nor this reviewer claims a production parser/gate rejection was executed.

Preflight payload/file links, dependent start links and end preflight/start links are refreshed consistently. All ten evidence binaries and the separate Quick identity are byte-identical to the preserved originals. The correction introduces no new identity dependency or hash cycle.

## Independent validation actually performed

Executed one read-only inline Python stdlib inspection command, exit0. Did not import or run the supplied generator/verifier and did not rerun old tests. Checks included all literal primitive values/types/offsets/lengths, binary/hex equality and hashes, positive argv byte preservation and invalid UTF8, the exact negative changed byte, event JSON/payload/sidecar links, all manifest/checksum entries, preserved archive and old Full hashes. Every text hunk was compared to both archived/current file lines; every diff file's Git blob identity was checked against its complete retained bytes, including binary entries.

| Check | Result |
|---|---:|
| Preserved original proposal/vector files | 136 unchanged |
| Changed / added / unchanged current vector files | 22 / 2 / 113 |
| Removed vector files | 0 |
| Complete diff files / text hunks verified | 25 / 35 |
| Literal records / primitive spans | 25 / 1,135 |
| Manifest file identities / checksum entries | 131 / 135 |
| Existing Full golden hashes | 78 unchanged |
| New nonUTF8 positive / single-byte mismatch | 1 / 1 |
| Exact Quick event link sets | 3 |

Preflight canonical/raw lengths are3411/7089; start802/1863; end954/2163. All remain below existing caps. The checksum exclusion for post-generation `verification-report.json` is explicit and consistent on regeneration; its old version is retained in the136-file archive and its current SHA256 is pinned in the FIX1 report/proposal. No evidence was removed to conceal a regeneration.

Invocation accounting is stated consistently: preserved original author history generator2/verifier1 plus reported FIX1 generator1/verifier1 yields cumulative generator3/verifier2. The per-invocation generation report still correctly says1. Original review inspection1 and this re-review inspection1 are separate read-only reviewer commands and are not supplied-script executions. Counts of author invocations are the explicit preserved/report history, not a fact inferable from final file bytes alone; independently reproduced byte/count/link results agree with the reported current verifier results.

No source/public-doc edits, supplied-script reruns, Cargo/Rust/native/HE/SSH/subagents, production parser/gate calls or scientific output occurred. Only this review was written. Current accepted authority and all21/full2680+20+7/quick268, holds, baseline/security/Task5/license boundaries remain unchanged. Genuine evidence readers/applicability, complete resource model, actual currentness channel, live custody and later source/native validation remain the previously documented obligations.
