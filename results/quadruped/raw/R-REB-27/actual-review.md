# Independent R27 actual staging evidence review

2026-11-07 JST. Review of retained actual evidence only, following the independent source review `af819cb0f19d476b1b1596827e4d9035eec0289b5e20636a14d8c1ac87c747a9`. No SSH, network, test rerun, Git operation/mutation, source modification, target/scientific import or FULL observation was performed in this review. Only this review document was written. The original controller streams/metadata are unchanged.

**SPEC verdict: PASS for the released R27 checkout/raw staging acceptance scope. QUALITY verdict: PASS. No blocking actual-evidence discrepancy found.** All three phases have actual terminal rc0, and the final retained output is `staging_verified`. This accepts the finite checkout/input/preservation observations below; it does not accept context/capture/native closure or any scientific outcome. Root alone integrates/releases subsequent work.

## Release, controller inputs and original streams

Read `task-54-linux-checkout-staging-execution-release.md`, SHA256 `a80bed3c25dcb22d09bdd6bea4032563c47a47c14077e0877f56fc71faf065fa`. It explicitly accepts the source-review qualifications before execution: remote path/Git pins are in the fixed external release/script/commands, and early runtime/receive failures may rely on controller evidence rather than uniform script JSON. The generator's omitted descriptive budget metadata remains a deferred minor; no regeneration occurred or is requested.

Read all three original `.meta.json`, `.stdout`, `.stderr` sets, exact `commands.json` and `manifest.json`. Recomputed every original stream digest and matched its controller metadata. Every actual SSH argv, stdin path, stdin byte count and stdin hash matches its exact frozen phase command. The current local stdin artifacts still match those frozen identities. All metadata bind the above execution-release hash and report `stdin_identity_stable:true`.

| Phase | Terminal rc | Controller elapsed seconds | Original stdout SHA256 |
|---|---:|---:|---|
| preflight | 0 | 9.997 | `e91526fca65f300ce0c42e867048a2db673c175b518762acfb2976432ce41a34` |
| receive | 0 | 25.487 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| create-readback | 0 | 12.031 | `2f4894b5dfd0123db5dc531cdea04efe68f68ba2ba08aaa6c8c72a579743e7cd` |

All three top-level stderr files are empty and have SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. Git command stderr, including normal init hints, bundle verification and checkout messages, is separately preserved inside the final JSON; it was not discarded.

Controller metadata hashes are preflight `90ff9e7473c175baeae49480eba8246311c7f4bb3ba8389203125d54b0a4a98a`, receive `26fc6baf5418abdb2b4278dc6fcbfa296024ffc77751a9571ed41c619957bf4d`, and create-readback `d4069f685043a7e96df5ff046131332c96679b693811e1583557db17e97bacc4`. The receive metadata's prior-preflight stdout hash and create metadata's prior-receive metadata hash match their original files. Actual controller finish/start timestamps establish nonoverlapping preflight -> receive -> create-readback order. Elapsed values are operational phase observations, not scientific or training performance measurements.

`commands.json` remains SHA256 `eaf2cca7833ab3e6f888090eba09dfc95c14104963373b5c29303b770fdde9af`; data manifest remains `302ed14d0a443d4f23cee23ee60d5c75bb5090b48f0cf8c80aa6e36964eadee5`. Preflight/create stdin is the fixed 16,745-byte script `d8f3e95a01fe2acc45b3732f135e50293a6958f23500f4460b13045994e06d1f`; receive stdin is the 112,650,240-byte package `373efd0a284fdcfe7997162bd5d8ef1db633a238c068c740ea6e11bbb2959cc0`.

**Nonblocking evidence precision finding:** The controller metadata retain a stable-input boolean, size and hash; they do not retain the raw before/after fd/path stat structures described in the execution release. The boolean cannot be forensically replayed from those missing raw structures. Do not claim raw-stat replay or invent them afterward. The exact controller input hashes, original stream digests and independent destination package whole-hash/readback remain consistent. This is a recording limitation, not an observed input mismatch; originals must remain untouched.

## Actual preflight and package/readback identity

Preflight output is `preflight_complete`, release ID `task54-checkout-010bb3c-v1-373efd0a284fdcfe`, with 27,298,234,368 free bytes recorded, above the released 824,604,661-byte initial gate. The same release ID is present in the final `staging_verified` output. The pinned script's successful gates establish the released Linux/uid/nice/environment/Git-pin and exclusive-root checks; these are script-observed checks, not new observations by this reviewer.

The final package observation is the exact fixed path `/home/kmihara/hekit-stage-task54-010bb3c-v1/package.tar.part`, 112,650,240 bytes, SHA256 `373efd0a284fdcfe7997162bd5d8ef1db633a238c068c740ea6e11bbb2959cc0`, regular uid1000 mode0600. This is destination whole-package validation in the create phase, in addition to transport exit0. The fixed script only parses/extracts after this check. Its success includes the remaining free-space and full member checks reviewed previously.

Accepted checkout is `/home/kmihara/hekit-exporter-010bb3c-task54-v1`. Actual archive path is `/home/kmihara/hekit-exporter-010bb3c-task54-v1/results/quadruped/raw/reboot_emlp23_full`. Actual new data root is `/home/kmihara/hekit-stage-task54-010bb3c-v1/payload/data`. These exactly match the frozen external release; the producer archive and old baseline raw root are not substituted.

## Actual Git observations

Read and decoded every one of the 15 actual Git command records. Independently recomputed all binary stdout/stderr SHA256 values, checked exact argv/environment/returncode and chronological command order, and matched each command to the reviewed fixed command shape. All records use the exact eight-entry `GIT_ENV`; the six preparation commands additionally have explicit pack/index threads1 and maintenance/gc controls.

1. Template-free init returns0 and reports the new independent `.git` path.
2. Bundle verify returns0 and records complete history, SHA1 object format and the frozen advertised HEAD.
3. Bundle list-heads returns exactly `1220bd0d76225e312bbaca9e78d1a22093abc3b5 HEAD`.
4. Fetch returns0 from the exact received local bundle, with protocol restrictions, no tags and no submodules.
5. Target ancestry under FETCH_HEAD returns0.
6. Detached checkout returns0 and names target010bb3c.
7. Exact consumer config command returns0 with 189 actual NUL-delimited bytes. Its SHA256 is `8af1efd8f9a96581ed737d72d2f6c088d2318975d934fb8614b7eb69e3c6c2ae`. Parsed actual keys contain no filter/include/includeif prefix and no core.worktree.
8. Exact consumer rev-parse returns `010bb3c82627a0b8809e0308cdcbf76615bf9b39`.
9. Exact exporter-scoped status returns0 with empty stdout/stderr.
10. Exact exporter-scoped binary diff returns0 with empty stdout/stderr.
11. Symbolic-ref returns the expected1 with empty stdout/stderr, establishing detached HEAD rather than hiding a nonzero result.
12. Tree rev-parse returns `509078781b1d7933f644c515957da6c54659dfed`.
13. Full ls-tree returns0 and 149,532 actual stdout bytes. Parsed entries exactly equal the entire frozen manifest inventory: 1,293 blobs / 62,447,344 declared bytes; inventory SHA256 `67ac11ddadfdeeb8736cd2a550d305322a6ea2b98262824a4d81f051c2843312`.
14. Separate full repository status returns0 with empty stdout/stderr.
15. Separate full repository binary diff returns0 with empty stdout/stderr.

The successful fixed readback also checks canonical D, real independent `.git`, no alternates/commondir and exact detached `.git/HEAD`; no additional standalone remote lstat/HEAD-file evidence is claimed beyond those successful script checks and retained Git observations.

## Actual selected files, raw8 and preservation

All 38 actual selected-file records have exactly the manifest's path set, byte lengths and SHA256 values, with paths under the exact new D. Each retained stat records a regular uid1000 file and a matching size. Combined selected bytes are 3,008,543: exporter61,243 + config193,960 + archive36 2,753,340. Exporter is `0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce`; fixed config is `76c493e28692531d07d54337dfcf89a21d8d78fa2979e02b20dccad47fb303c5`. These are actual remote readback hashes matched to the previously independently checked committed-blob manifest. No claim of loading/executing the files is made.

All eight raw records have exactly the released names, byte lengths and SHA256 values, paths under the new `payload/data/MNIST/raw`, and regular uid1000 stat records. They total 66,544,770 bytes, including all four original gzip files. No recompression, baseline filling or scientific dataset loading follows from this observation.

All 24 static sentinel records are exactly equal in preflight, immediately before creation/readback, and after it. The set is the nine existing root identities, six small old `.git/HEAD`/config hash+metadata records, producer config identity and eight baseline raw presence/metadata records. Producer remains191,369B / `9d6a21484856b3a5761f64fa139c1da226531d31e901eec2f5d72493ad0ce444`. Baseline uncompressed four metadata identities remain unchanged and its four gzip names remain absent. Old config bodies are not present in the output.

This supports unchanged selected static sentinels during staging. It does not establish whole-server bit-for-bit immutability, unchanged growing job outputs, or installed-package/native/reference closure. No FULL process/output evidence is added by this review. Actual R27 registration and the preceding root-owned FULL observer/timing gate are outside the bounded evidence files reviewed here; this document does not replace root's records for those gates.

## Accepted boundary

Retained actual evidence supports a real fresh detached010bb3c checkout plus all eight exact raw inputs in the released new roots, with the finite static preservation checks unchanged. It retains the two explicitly accepted source qualifications and the additional controller raw-stat recording limitation. No rerun or evidence rewrite is needed for this staging verdict. Context provenance/admission, full installed-package/native/canonical/reference identities, helper-source acceptance, compatibility capture, HE execution, training, performance, adoption and scientific acceptance remain separate gates.
