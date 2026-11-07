# Resource source implementation seams — read-only preparation

Current HEAD 52f10dad592cf0a6def8b3ddf9b8061203f91a50 inspected; no source edit, runtime, materializer, project helper or corpus execution. This is a bounded source observation, not a replacement design or implementation release. Independent corpus author remains the sole artifact writer for its corpus.

- document.rs defines private exact-field Fields/FieldSeed over borrowed RawValue, rejects unknown and duplicate decoded keys, requires fields when consumed, and has existing lexical U64, Hash/Text, Option and bounded List decoders. These are useful existing primitives; do not replace this with an unrelated generic schema framework.
- document.rs DocumentLayout currently has Reference/TargetLock/Admission/Start/Quick only. Scan.string grants enlarged unescaped hex limits only at specific existing structural paths. All other value strings are decoded as String with a4096-byte decoded cap (apart from named existing provenance exceptions). New ClaimWork.registration/request and RunMetadata.argv have different exact paths. Passing the new six sources through an old layout is therefore not evidence of full-format support. New source preflight must explicitly cover the ratified bytes paths, lexical forms/counts/aggregate arithmetic before decoding, without broadening legacy limits. Keys versus array elements are structurally distinguished already.
- evidence.rs is explicitly an inert private codec without live readers/constructors. It holds existing Mode/outer branches and retains original bytes. quick.rs demonstrates separate JSON/binary source identity, production versus test domains and strict wrapper fields; its layout and outer bytes are frozen. Parsing a resource source cannot create a live resource owner or accepted gate.
- study_runtime.rs run_owned still captures the runtime/context, resolves the registry/case and materializes input before planning. study_profile.rs prepare_dataset still preflights, read_locked-verifies files and reopens via profile_data::load_dataset. The accepted actual-owner/same-retained-bytes integration is still required; format implementation alone cannot close it. These latter facts match the prior accepted integration map, not new evidence of completion.

Next production-source brief must cite the frozen independently reviewed corpus once available, test actual positive bytes plus reachable negatives against real Rust decoding and binary bootstrap encoding, and preserve source-vs-live authority separation. The six JSON records are raw original-file identities, not an invented canonical-JSON wire. There is no new source/runtime permission in this note. No request to relax caps, seed membership or physical-model obligations.

Read-only command caveat: one rg command included nonexistent document/parse.rs and returnedexit2 while displaying valid existing paths. It made no changes; actual parsing code is document.rs. No test failure occurred.

| Read source | Bytes | SHA256 |
|---|---:|---|
| `crates/ckks-train-bench/src/study_reference/document.rs` | 65310 | `30c77a7181866d1aaacf2151f3a3f94cb64d594e3dfa643f885114c8ea60f7de` |
| `crates/ckks-train-bench/src/study_reference/document/evidence.rs` | 15901 | `4f727106cdbccd765a0cb5b3712fc81ec658efc170dddb86192e8ecabb83ae82` |
| `crates/ckks-train-bench/src/study_reference/document/evidence/quick.rs` | 23187 | `444b46d5546127486c419b568c119975ab7a519485b68d2103441a00d88d8cbb` |
| `crates/ckks-train-bench/src/study_runtime.rs` | 15808 | `975e0a2d95c04abb42787e477a9e9ce324869e9c6e15216edaade688cd648f95` |
| `crates/ckks-train-bench/src/study_profile.rs` | 79141 | `eaaf3ad085f1a347b664099f720a1ad2fb068ada66939baf60c3c6bbefe3670f` |
