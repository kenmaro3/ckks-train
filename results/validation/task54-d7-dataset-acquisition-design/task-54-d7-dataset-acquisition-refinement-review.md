# Task54 D7 closed dataset acquisition refinement — independent review

## SPEC verdict: PASS

## QUALITY verdict: APPROVED as a design refinement

No blocking design issue was found. All 11 enumerated local files match the proposed exact raw sizes and their already-locked whole-file SHA256 identities. Only Fashion training images exceed 8MiB. The closed role-selected size policy resolves that concrete conflict without raising external report limits, changing scientific inputs, or weakening immutable-byte/held-FD custody.

This is design acceptance evidence for root's ratification decision, not source implementation, a live gate, materializer validation, resource-fit evidence or runtime authorization. Cargo/tests/`--list`/project helpers/materializers/SSH/native/subagent runs: **0**. No prior tests were rerun. Only this review report was written; unfinished publication source was neither inspected nor edited.

## Inputs and fixed source pins

`L` below is `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5`. All source/registry inspection used read-only `git show` at **2da02ab7c861c7d85ff25c4ee5bf67f4d86a6c72**, not the concurrently edited worktree.

| Reviewed input | Bytes | SHA256 |
|---|---:|---|
| `L/task-54-d7-dataset-acquisition-refinement.md` | 8304 | `83a29bb6dbad4cd93bb5d50ef8424b3cc5d2a3e52597d9783e72b73b717db840` |
| `L/task-54-d7-retained-reader-integration-map.md` | 46547 | `4266520bb4a4237c1e68ee7c373ae7001b14aaa35aaa62cddd0e84dc663ec1a1` |
| `L/task-54-d7-dataset-acquisition-refinement-review.diff` | 8585 | `0f8d06051302bef1d809a1f04ec5875647f7723d4341873dbee113004646b7fb` |
| `L/task-54-d7-gate-schema-proposal.md` | — | `1e54707536ee52c481cfa789c308d9f4605f096a1a720649c2eebe7a1dc6f50b` |

The integration map's original 30,490-byte prefix independently hashes to `d4835c1e167ad8959f1485e9a6d7120e49200596add00585b291e2894c255d9e`. Read the complete map/addendum and proposal/diff, with ratified schema §5 as the custody/limit requirement.

| Fixed BASE source of dataset authority | Bytes | SHA256 |
|---|---:|---|
| `crates/ckks-train-bench/configs/experiment_matrix.json` | 96937 | `164dbc67fddb87bed814bfa9bcbad1f97508a4a2de265b7a8ca899d8fab1b4cc` |
| `crates/ckks-train-bench/configs/experiment_input_locks.json` | 3696 | `cc04f6f7451b0ec99416e10d14c2bf6b2eaf5f703c8de43d500cac3e40da126b` |
| `results/quadruped/raw/reboot_emlp1_full/config_values.json` | 270294 | `b89c9129177f5831cf044c775a2228fecace8214479aa54b1ad71b2d10a8ad18` |
| `results/quadruped/data/reboot_emlp1_full/data_manifest.json` | 2904 | `0ddec87635b5c0e6326c49932a79e0cf8a035d25705b22d6f7d176fb1ee2dd98` |
| `scripts/fetch_fashion_mnist.sh` | 2345 | `1ecc5cfa25ea1d16d93c8df533e24ead408cf35fdee2eee9a18527a11753c3c0` |

These are hashes of fixed BASE blobs. The captured config matches the existing portable input-lock entry; the BASE manifest's `files/pipeline/row_order/scaling` equal the locked capture's corresponding data records. The Fashion script matches `FASHION_SOURCE_SHA256` in BASE `study_profile.rs:35–36`. Dataset expectations below therefore come from existing locked authority, not from the just-observed local hashes.

## Independently observed dataset size/hash evidence

Read exactly the 11 proposal paths locally, once each, with Python stdlib file I/O and SHA256 in 1MiB chunks. Final files were opened O_RDONLY/O_NOFOLLOW and checked regular; bytes were counted while hashing, rather than inferring content from stat. Pre/post FD and final path device/inode/size/mtime/ctime stamps agreed for every read. Parsed only bounded NPY/IDX headers, with no project loader, array materialization, preprocessing, RNG or HE. This is a local hash snapshot; it is not a replacement for the future retained safe-ancestor/read-custody implementation or adversarial-race tests.

**All 11 rows MATCH both required size and existing whole-file SHA256; total bytes streamed: 66,696,124.**

| Exact repository-relative path | Counted raw bytes | Observed whole-file SHA256 = existing expected SHA256 |
|---|---:|---|
| `crates/ckks-train-bench/data/iris.csv` | 2756 | `874d28c2148c94ac8bed1b98ab1d93d27a473f72b2a712b06833f49b3259e54d` |
| `crates/ckks-train-bench/data/iris_split_seed2020.txt` | 705 | `9da646715cfa3f2bd0e800680ff987fe066a4e0403cc2acf25ca90b8cf02cf12` |
| `crates/ckks-train-bench/data/breast_cancer.csv` | 124103 | `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a` |
| `results/quadruped/data/reboot_emlp1_full/train_images_14x14.npy` | 3763328 | `7b1590d8884172b853d8037063b290d20b41112f6c0aa5309769639552b9276c` |
| `results/quadruped/data/reboot_emlp1_full/train_labels.npy` | 4928 | `1d63491b1c54835583c5ca6886305d7b83d7c9ca0c3ab24433c431b1f7e84856` |
| `results/quadruped/data/reboot_emlp1_full/test_images_14x14.npy` | 7840128 | `9464f6e1544cd65d7387036bab2e71e3826b121b108d2bb4f1360e9bfb0d6d87` |
| `results/quadruped/data/reboot_emlp1_full/test_labels.npy` | 10128 | `ff7e84b144c037e7215dfa787d6773550c5db83029d9a4e7bae6e90f605f081d` |
| `data/fashion_mnist/train-images-idx3-ubyte` | 47040016 | `c59f468a2f672dc815687fe0f83887768d799fd8a3f3276145d20f83aa44d888` |
| `data/fashion_mnist/train-labels-idx1-ubyte` | 60008 | `bad3541b69d912435c50bb6ba87bec294ff4f6a2e1246121d8633921760443d9` |
| `data/fashion_mnist/t10k-images-idx3-ubyte` | 7840016 | `5b4141f0afbad91edebe8549f8fcffe087ea10ca49f1dbef5c9a5cd8815ce37b` |
| `data/fashion_mnist/t10k-labels-idx1-ubyte` | 10008 | `0402a96d92fd2663957122ceb108a494c5af83dab82d92729df917d7dec38c34` |

Expected-hash provenance: Iris constants at BASE `study_profile.rs:33–34`; WDBC constant at `data.rs:720`; MNIST `sha256_npy_file` from the locked capture's `/data/files` and matching manifest; Fashion unpacked-file SHA256 values in the pinned fetch script's final LIST. `study_profile.rs:1763–1775,1777–1807,1809–1855` uses these same representations. Gzip MD5 values and MNIST raw-array digests are **not** substituted for whole-file identities.

All four NPY headers are version1 with a 128-byte total prefix/header, C order. Images are `<f4` with shapes `(4800,196)` and `(10000,196)`; labels are `|u1` with shapes `(4800,)` and `(10000,)`. Exact header+shape×width equals each counted file length. During the same reads, independently hashed each post-header array payload; all four equal the capture's `sha256_raw_array_bytes`:

| NPY basename | Observed raw-array SHA256 = locked raw-array SHA256 |
|---|---|
| `train_images_14x14.npy` | `c0501c3eb43788f57f823a0e93586192291fe4104976cefafb1d65540681381f` |
| `train_labels.npy` | `d94e8472b108cd7ee460b811c01a6416bb1cf8b1e85aee979a31e48f0ef8426c` |
| `test_images_14x14.npy` | `a1f684430826b571df70c8f73fae41080fa76cb43b257c2a1e013a28aeb547e9` |
| `test_labels.npy` | `ddeff807876a9661a1110d45c266c86239a3a1b7d37da0c3716a7a683c852ff5` |

Fashion headers are unsigned-byte IDX with exact shapes `(60000,28,28)`, `(60000,)`, `(10000,28,28)` and `(10000,)`. Header plus dimension product agrees with every raw size. Only train images exceed 8,388,608, by 38,651,408 bytes. MNIST test images are below it by 548,480 bytes. No dataset regeneration or scientific result edit was needed.

## Design checks and strengths

1. **Exact current scope and representations:** Proposal `:7–29` correctly targets the fixed 11-file union while requiring only the complete per-selected-dataset inventory at acquisition. BASE registry `/datasets` declares 4,800×196 MNIST training inputs and 10,000 test rows, not 60,000×784. Static JSON expansion, without registry runtime, confirms 21 profiles, 268 Quick family projections, 2,680 Full family rows, seven supplements; affected Fashion counts are main_i8/80, main_ni24/240, a_matched2/20, c_i4/40, c_ni12/120, b_external1/10, totaling **51/510**. These are registrations, not runnable/admissible outcomes. The exact-Iris20 and seven supplement obligations remain outside any claimed completion here.
2. **The pre-read gap is real:** BASE `study_profile.rs:111–126,1478–1502` whole-reads and hashes before invoking the typed loader. `profile_data.rs:113–118,273–307,323–334,373–384,423–425` performs additional whole reads/reopens; final declared shapes are checked at `:453–458`. `data.rs:103–106,354–355,559–570` reads whole IDX/NPY files before typed validation. These are not finite role-specific acquisition budgets. Proposal `:33–37,43` explicitly requires inserting bounded acquisition before those reads and feeding the same checked bytes into the real typed path. A post-materialization audit would not satisfy this design.
3. **Narrow limit change:** Ratified schema §5 (`task-54-d7-gate-schema-proposal.md:108–110`) already requires fixed role selection, held File/stamp/immutable bytes and bounded external evidence. Proposal `:9,29,33–39` supplies a closed exact-size DatasetInput rule, not a caller-selected cap. Only the one actual oversized registered representation needs the larger acquisition. External evidence stays8MiB; C/R/A/K and FileRef wire fields are unchanged. Non-dataset source/config/order/init/manifest files receive no new exemption.
4. **Same bytes and custody:** Proposal `:34–36` requires pre-allocation size refusal, bounded short/growth handling, whole-file hash and typed validation over retained immutable bytes, original File/stamp/ancestor custody, and refusal of equal-byte inode replacement. It forbids pathname reopening for a separate typed load. This preserves the schema requirement rather than replacing source custody with a manifest or streaming digest.
5. **Resource ordering:** Proposal `:37` requires budget/model/capacity support before acquisition and explicitly accounts for raw-retained plus decoded/temporary coexistence. This prevents a completed data gate from retroactively authorizing earlier allocation. The reference to47,040,016 retained raw bytes is a buffer cost; per the same paragraph's existing resident-versus-incremental rule it must not be treated as measured RSS or automatically credited as already charged resident_bytes. Actual allocation overlap/accounting remains a later source/model obligation.
6. **Verification requirements:** Proposal `:43` covers exact-size acceptance, ±1 before allocation, size-changing reads, same-size wrong digest, typed header/shape/payload errors, equal-byte replacement and safe-root/path refusal. Full and Quick retain complete canonical train/test source inventory; two consumed updates do not justify dropping raw sources or training-wide preprocessing. Generic small negatives and any real47MB acquisition have separate explicit runtime inventories; this review released neither.
7. **No scope or approval bypass:** Proposal `:29,39,47–54` leaves supplement availability, scientific thresholds, Full/Quick split, full encrypted-block holds and actual gate requirements intact. The design preserves permissive-license boundaries and introduces no baseline code, source runtime, fixture/vector mutation or new dependency.

## Findings

- Critical: none.
- Important: none.
- Minor requiring a design correction: none.
- Required size/hash precondition at proposal `:27`: **satisfied by the independent read-only evidence above for the existing fixed representations**. This does not waive revalidation in the future production reader.

## Exact unresolved implementation/evidence obligations

These are explicitly retained source-release gates, not defects in this acquisition-policy proposal:

- Root ratification and a concrete source brief must select the closed role/size/hash mapping and pass the SAME acquired bytes/FD owners through all relevant resolver/preflight/materializer checks. Existing unbounded dataset reads and repeated path opens remain until changed and reviewed; this design alone fixes none of them.
- The acquisition itself needs actual reviewed pre-allocation accounting/model/capacity checks, including all selected raw inputs, parsed IDX/NPY buffers, decoded arrays, preprocessing overlap and reader/preparation metadata. No memory-fit or measured residency conclusion follows from the hash pass or finite sizes.
- Retained safe-root/ancestor/file identity and equal-byte-replacement behavior need implementation plus bounded fault/native evidence. This review's stable local FD stamps are not that proof.
- Source tests must retain all existing typed semantics and error distinctions, exercise the proposal's negative inventory, and separately bound any real large acquisition before execution. No project test or materializer has run in this review.
- Actual report applicability, authority/currentness channels, resource model validation, five gate producers, C1 same-target Retry composition, consuming runtime and scientific campaign results remain separate Task54 dependencies. Accepted publication design/source work belongs to its separate writer and was not inspected here.

No additional design clarification or size/digest correction is needed before root decides this narrow refinement. The requested21/full2680+20+7/quick268 scope, original scientific gates/holds and unchanged C/R/A/K remain mandatory.
