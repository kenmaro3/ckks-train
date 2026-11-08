# Paper cutoff readiness review — 2026-11-08

Reviewed HEAD: `47d0035dd167d63b90f7552f8a394510f3d2deb5`.
Scope: current interim manuscript, all eight included sections, bibliography, seven generated tables, included TeX figure, paper README/metadata, numerical source artifacts, M3/M4/M5 reports. Read-only review; the only written file is this report. No tests, generation, TeX/native build, SSH, installation, HE execution, or unrelated Rust WIP inspection was performed. The already-reviewed primary HEonGPU citation was checked for internal consistency only; no new full primary retrieval was repeated.

Release context: `v1.0.0-preview.1` at fixed `6fba46c` is public and immutable. This report concerns the current source for the separate first final preprint, evidence cutoff 2026-11-09 18:00 JST and GitHub/arXiv target 2026-11-10 09:00 JST. It does not authorize or claim an arXiv submission, server build, new measurement, or gate change.

## Verdict

**SPEC: PASS for the current interim-paper scientific claim scope.** The central claims remain conditional and accurately distinguish completed FULL-1 B=64 from pending FULL-2 B=48, plaintext reference from encrypted baseline, the four derived KS conventions, model denominators from measured numerators, subset timing from missing full-setting timing, security-different comparison, GPU operators from unexecuted training/ablation/bootstrap, and implementation from the unexecuted whole matrix. No reviewed statement promotes the current results to an equal-security, accuracy-matched research success. Missing original research gates are not prerequisites for publishing a candid first preprint.

**QUALITY: NEEDS FIX before freezing the next paper artifact.** F1 is a concrete contradictory scope statement; F2 conceals the magnitude/definition of a specifically discussed final-parameter diagnostic; F3 leaves demonstrably stale author/submission guidance in the paper. F4 is a compact reproducibility clarification recommended with those fixes. These are content/traceability corrections using existing evidence, not new research scope. A source/PDF/package rereview after correction and after the cutoff remains necessary; this review is not visual QA or a TeX/arXiv build result.

## Findings and precise fixes

### F1 — Scope weight-decay omission to the encrypted trainer (medium)

Locations: `paper/sections/noninteractive.tex:28`; `paper/sections/discussion.tex:19`; affected interpretation of `paper/tables/plain_datasets.tex:12`.

Problem: “Weight decay is not implemented” and “Weight decay ... [is] omitted” are unqualified. The reported plaintext MNIST paper-setting reproduction does implement/use coupled weight decay. A reader can reasonably infer that the reported 95.00% paper-setting reproduction also omitted the baseline regularizer, when its retained source says otherwise.

Evidence: `results/plain_mnist_emlp1_paper.json:55,90-92` records weight decay 0.001 in the ReBoot convention and 0.000020833333333333333 on the mean-gradient implementation; `effective.weight_decay_form` explicitly says coupled PyTorch-SGD-style decay before momentum/Nesterov. FULL-1 source `training.weight_decay=0` (`results/quadruped/raw/R-41-FULL-1/mnist196_ni_when_needed_b64_full_seed0.json:58`). `docs/m3-report.md:267` independently identifies 0.001 decay in the separate R-REB-3 reconstruction. The numerical table values themselves are consistent with their sources.

Replace the last two sentences of noninteractive:28 with:

> Biases use SGD; the faithful ReBoot configuration freezes their encrypted zero values while retaining the implementation's bias arithmetic. The encrypted trainer omits weight decay; the captured faithful encrypted-comparison setting uses zero decay. The separate plaintext paper-setting reproduction includes its specified coupled weight decay.

Replace only the opening three sentences of discussion:19 with:

> The quartic activation's claimed accuracy benefit is unmeasured. The encrypted trainer omits weight decay and optional softmax/cross-entropy; the separate plaintext paper-setting reproduction includes its specified weight decay. Bias momentum is not matched to NAG, and faithful bias freezing preserves extra encrypted arithmetic.

No threshold, result, or baseline protocol should change.

### F2 — Expose the actual FULL-1 final-parameter discrepancy through the generator (medium)

Locations: `paper/sections/evaluation.tex:31`; `paper/sections/discussion.tex:10`; `paper/tables/accuracy_vs_reboot.tex:16`. Source extraction path: `tools/paper_tables.py:453-491`.

Problem: the paper explicitly says FULL-1 records accumulated weight/bias error and that it is larger than the subset error, but never gives the measured magnitude or defines the diagnostic. The accuracy table gives 90.36% vs 90.37%, which alone makes it easy to underestimate the numerical accumulation. This is a missing explanation of an already-discussed negative observation, not evidence that any new threshold was violated.

Evidence: retained FULL-1 JSON `twin.max_abs_w_diff_per_layer=[0.004549795297114266,0.006826572783266538]`, `twin.max_abs_b_diff_per_layer=[0.0002538453491169957,0.00011036939865718823]`, `twin.max_abs_diff=0.006826572783266538`, `twin.log2_max_abs_diff=-7.194622815887531` (`.../mnist196_ni_when_needed_b64_full_seed0.json:277-283`). This is the final-model maximum absolute difference over weights/biases versus the same-B plaintext twin after 750 updates. It is distinct from the CPU standalone per-bootstrap random-input precision observation and from accuracy percentage-point differences. The historical FULL-1 is not to be retroactively judged against an unrelated new NI1/256 comparison condition.

Recommended implementation: in `table_accuracy`, add a generated note for each completed FULL row from its `twin.max_abs_diff` and optionally `twin.log2_max_abs_diff`, retaining field provenance. Intended generated FULL-1 note:

> FULL-1 final-model maximum absolute error over weights and biases versus its same-B plaintext twin: 0.006827 (approximately 2^-7.19), after 750 updates. This is an accumulated final-parameter diagnostic, not per-bootstrap precision or an independent accuracy gate.

Use source fields for the value and update count rather than hand-entering a performance result into authored TeX. Missing FULL-2 must remain pending, not receive FULL-1's value. For evaluation:31, replace the final four sentences beginning “FULL-1 extends” with:

> FULL-1 extends exact schedule agreement to its complete update sequence. Table~\ref{tab:accuracy_vs_reboot} reports its final-model maximum absolute error over weights and biases against the same-batch plaintext twin. This accumulated discrepancy is larger than in the earlier subset checks; close final accuracy neither bounds parameter error nor measures standalone bootstrap precision. FULL-2 and guarantees across seeds or arbitrary trajectories remain unmeasured.

“Complete registered update sequence” can remain if desired; it must not imply completion of the separate FULL-2 experiment or Task54 whole matrix.

### F3 — Replace current-paper placeholder/anonymity/submission guidance (low, definite stale text)

Location: `paper/sections/appendix.tex:28`.

Problem: the manuscript still promises instructions for “author-placeholder variants,” asks for authorship review before submission as though identity remains undetermined, and groups “all pending research gates” into a blanket before-submission review statement. The public named preprint author is confirmed; pending gates may be reported as pending in the first arXiv version. Conference anonymity and future WAHC rules are separate concerns.

Evidence: `paper/main.tex:35-38` names sole author Kentaro Mihara, author note Independent Researcher and the confirmed email; `paper/README.md:3-8` records the public named preview and no arXiv submission; `paper/arxiv-metadata.md` consistently marks submission draft, deferred account/endorsement verification, the selected non-exclusive arXiv license, and untested server build. No invented location/company affiliation is present. The author-note mechanism is intentional and is not an affiliation defect. The historical 2026-11-07 README references to actual earlier placeholder builds need not be erased.

Replace appendix:28 with:

> Anonymous mode omits the author block; the named preprint identifies Kentaro Mihara as an Independent Researcher. ReBoot and OpenFHE are executed as external baselines; no implementation source is copied into this artifact. The estimator and TeX compiler are isolated development tools, not library dependencies. The library's dependency policy, including the specific character-table exception disclosed in Section~\ref{sec:limits}, is separate from the licenses of those external research tools. Instructions for rebuilding anonymous, named-review, and named-preprint variants accompany the source. Before arXiv submission, the final source, metadata, and server-generated PDF require checking against the retained evidence; unavailable measurements remain explicitly unavailable. Any later conference submission requires a separate anonymity and venue-rules review; the WAHC~2026 limit is only a provisional formatting constraint~\cite{wahc2026}.

This also narrows the unsupported blanket claim “The anonymous draft omits repository/account identifiers”: current generated table notes include a local shared-host alias, while anonymous mode is principally an author-block switch. Do not claim full artifact anonymity from that switch alone.

### F4 — Define the released “full setting” and link the evidence artifact in the PDF (recommended, compact)

Locations: `paper/sections/evaluation.tex:6`; `paper/sections/appendix.tex:26-28`. The manuscript currently relies on repository-only docs for the actual eMLP-1 hidden width/protocol and contains no public repository URL in its body/bibliography. arXiv metadata Comments includes the URL, but that is not guaranteed to accompany a detached PDF. “Full setting” means the complete released 4,800-example/10-pass experiment, not all 60,000 MNIST training examples.

Evidence: generated `rq1_ks.tex:12`, `accuracy_vs_reboot.tex:12`, retained FULL-1 JSON `network`, `training`, and provenance command; `docs/m3-report.md:212-214,247-259`. The source uses flattened 48,000 sample indices with `epochs=1`; the paper correctly reports the scientific 4,800 x 10-pass setting rather than the literal CLI epoch field. FULL-1 initial-weight seed 0 and encryption RNG seed 32 are distinct.

Replace the first two sentences of evaluation:6 with a compact protocol declaration:

> The headline uses MNIST reduced to 196 inputs and the complete released encrypted-comparison setting: 4,800 training examples over ten passes, a 196-to-32-to-16 eMLP-1 with $z^2+z$ hidden activation, frozen zero biases, sum-loss NAG with momentum 0.9, and the released sample order and learning-rate schedule. This is distinct from the 60,000-example paper-setting reconstruction. FULL-1 has completed at effective batch 64 with one initial-weight seed; FULL-2 at the released batch 48 in 64 physical lanes remains pending.

To keep the PDF compact, do not duplicate all per-seed paths or a full hyperparameter table. Preserve the existing subsequent explanation of output padding/legacy/partial-batch differences. In an `ArxivDraft` conditional in the appendix, add:

```tex
\ifdefined\ArxivDraft
Code, retained results, and field-level provenance are available at
\url{https://github.com/kenmaro3/ckks-train}.
The published \texttt{v1.0.0-preview.1} remains a fixed interim artifact;
the source and evidence identity for this preprint are recorded separately.
\fi
```

At final freeze, name the actual final source/evidence identity once known. Do not relabel the immutable preview as the final artifact. This recommendation need not expand the provisional 12-page limit or erase anonymous conference mode.

## Numerical source consistency observed

- FULL-1: 877040 KS / 48000 processed examples = 18.2716667; H=1500, measured 111 KS/bootstrap; R=37 full-setting model, r=4, baseline B=48. The four generated ratios 0.232/0.387/0.395/0.400 agree with those inputs and are derived conventions, not four executions. 72050.794763016 s gives 1.501 s/example; no full-setting measured baseline denominator is inserted. Accuracy 0.9036, twin 0.9037, seed-zero released plaintext reference 0.9041 are correctly kept separate from 10-seed 0.90648/sample-SD 0.00271.
- F1/F2: raw K/H/S values 33756/68/1008 and 18028/36/1008, in-library Lazy reference 35.0625, measured official subset 84 bootstrap ciphertexts / 21 updates. Generated KS ratios, figure values, update timings and 75.77/73.58% accuracies are consistent. B64's -1.48 pt subset accuracy difference remains disclosed. Legacy AU/WN/AA counts and time values agree with retained R-41-1..5 sources; their accuracy is correctly not used as faithful-baseline evidence. Interactive 20256/1024=19.78125 is a separate client-refresh experiment.
- ReBoot full-setting seed 0..9 plaintext twins and the ten hekit faithful plaintext results agree. eMLP2/3 seed-zero plaintext 0.9156/0.9175 agree with their separate released twins. R-REB-3 fixed-epoch/optimistic test-selected rows agree with `summary.json`; 94.61% is a published plaintext reference, not a missing encrypted execution.
- Task51 fixed-final/test-best means and sample standard deviations agree with the five `results/plain_*.json` sources. Breast Cancer final miss and 10/10 z-score divergence remain explicit; no encrypted Fashion/Breast Cancer result is manufactured. F1 addresses the specific regularization-description conflict.
- GPU operator table values agree with retained R-47-2 Criterion `median.point_estimate` values (ns converted to ms); the 34.994/41.673 ms pre-fix stream comparison is not retold as a post-fix win. Top-level hoisted/matvec rows remain skipped. No RQ3 server timing is inferred from the insecure eight-config count-only smoke. Its 233/285/311/324 and repeated no-hoist counts agree with retained smoke metadata. The HEonGPU 2581.8 us reference/caveat and nominal 1.81 ratio agree with `results/reference/literature.json` and current M4 report; no matched-security/normalized latency claim is made.
- Security table agrees with `docs/security.md:281-283` (131.5/129.3, 133.8/134.2, 121.8/121.5). The manuscript preserves the bounded attack search, secret-distribution proxy, one-key ReBoot sign split, extreme memory demands, lower Core-SVP estimates, and absence of a security proof. It correctly does not infer equal security from equal N.
- `tools/tables.py` is only a compatibility wrapper; `tools/paper_tables.py` is the current authoritative generator. This review inspected the source extraction paths and retained JSON/doc/criterion inputs; it did not execute `--check`, table generation, or a training rerun, and therefore is not a fresh generator/test verdict.

## Cutoff-dependent edits and measurements that may remain unavailable

1. FULL-2: only replace pending after final raw JSON, producer/input provenance, complete update sequence and terminal status have been accepted. Compare its encrypted accuracy with released full-setting seed-zero **0.9041**, using the existing ±0.5 percentage-point rule. Never substitute FULL-1 B64 or the 10-seed mean. Regenerate KS four ways, timing, accuracy, and any accumulated-error note from the accepted source. Update abstract/introduction/evaluation/discussion and arXiv abstract together. If unavailable at cutoff, keep pending; no synthetic completion or threshold relaxation.
2. Full-setting ReBoot same-CPU timing remains unavailable. Retain that fact even if FULL-2 completes; neither the 47.6 h extrapolation nor the 784-input official subset is a measured full-setting denominator. Equal-security failure remains independent of FULL-2 accuracy.
3. R-47-3, R-48 parity/ablation, R-52 encrypted subsets: accept only rows actually executed with the required identity, optimizer, level, layout, batch, and measurement scope. A NAG-to-SGD fallback needs its corresponding CPU SGD comparison. A skip remains unmeasured; OOM remains failure. If nothing completes by cutoff, current pending/skip wording is a valid first-preprint outcome.
4. Physical GPU bootstrap precision/latency/allocation/peak-memory remains unavailable. The current benchmark still uses the host adapter. No queued operator/training result implicitly validates the compiled device bootstrap or all-GPU training.
5. Task54's 21-profile whole matrix and 10-seed execution remain incomplete. Existing pipeline implementation/tests/table regeneration must not become an executed-matrix claim. The current paper does not make that claim; keep that distinction in any added artifact paragraph.
6. After evidence freeze, rebuild/inspect the final PDF, regenerate/read back the separate source ZIP, refresh metadata abstract/hash/page count, and inspect the actual arXiv server log/PDF during authorized submission. Current metadata honestly says not submitted and account/endorsement checks deferred. Local TeX Live 2026 success does not prove arXiv's server toolchain. No new account/endorsement requirement is imposed by this content review.

## Frozen reviewed paper inputs (SHA-256)

The following hashes freeze the committed reviewed baseline at HEAD `47d0035dd167d63b90f7552f8a394510f3d2deb5`; a changed file requires scoped rereview. The root began the generated error-note fix concurrently after the source reads, so these are explicitly the baseline git-object hashes rather than a later worktree hash. They do not certify a compiled artifact.

| File | SHA-256 |
|---|---|
| `paper/main.tex` | `ccaf2e223bb55ecae6e9d011cf91d1d5a39b4c5f67d44ca07f56d63b1a5ed68e` |
| `paper/sections/appendix.tex` | `d649882d25c3b499d9cdbaaec2c92d6a38c64b0f948ca8cf58d36daf664f6b60` |
| `paper/sections/discussion.tex` | `41ac193e793d86be3cdf869cef828a2b93fa89ff98b3cacc0fffa4f41e05e06e` |
| `paper/sections/evaluation.tex` | `b97f48ae9ba15947b00a7046a67fc755347e0c731a56c71e7e2d4b28731a3181` |
| `paper/sections/gpu.tex` | `578b315eb8f1960a4caa4dcf5fa718ce000f2da1a77158dedc99cf979adb5588` |
| `paper/sections/introduction.tex` | `0705dd6848677d0c3f33b9fddfb57ffb39aecbd776beba5d1a545bcde652824f` |
| `paper/sections/noninteractive.tex` | `e398b3ed02b2b153f4a717d6689b06e55d8c0259859291ff8bdafca9728a86c8` |
| `paper/sections/packing.tex` | `51747a73c24cb897bd3ca631d9be13c4f4f62201e31cd0e5eb8f986f2b1be136` |
| `paper/sections/preliminaries.tex` | `3ac6b82f79f43f9bb13f38bf2f61c0d1c58461a193bd3c6b1a5443daade79c4f` |
| `paper/refs.bib` | `16fd704def307c0cd833735c55d650389e8e6bfc0b0e94fd9b87df215fda67f7` |
| `paper/tables/accuracy_vs_reboot.tex` | `b566fb557e2bff0c42ddbbea4463a7841d8691e7309fb0628f6887327a2ac751` |
| `paper/tables/gpu_vs_cpu_ops.tex` | `c9104367559251ee43275272c219c2e632b59225c1d56eda7170422afe3043cf` |
| `paper/tables/plain_datasets.tex` | `1667a75790194a9dc0fef7b36af3fc6cc5aeda22ac87b395ef19d82d9db7a6fe` |
| `paper/tables/rq1_ks.tex` | `d34c3779aca556ad5e9e1254e4e36040c6d127c729d3dade41428f582a17372b` |
| `paper/tables/rq3_ablation.tex` | `de38d3727aecf170b18a3ce0ed4db28f5984bf6938934081236e53d7f6db7cba` |
| `paper/tables/security.tex` | `3b30d20da49c757d274b6b30501bb524ab448150301d0f153b3e2d60d31c019e` |
| `paper/tables/wallclock_same_cpu.tex` | `bfb3f1555da2829afd17fa1971b2d804a9150698e077b1ba29c6f17ab45e4ade` |
| `paper/figs/rq1_ratios.tex` | `92eb4b1b82fa2cfec2b016c862ebe510bdfeaf382f40fa2f777148478df341de` |
| `paper/figs/rq1_ratios.sources.json` | `45089db8d5c73412357877c05645288415038aef19b73e1c6e04b8d8f428e124` |
| `paper/README.md` | `deff8de9e7bdaf01ff189e8af7ce3cf005ed196170fe4fb31be5e53657e27724` |
| `paper/arxiv-metadata.md` | `199e6a5fcefee6e3534d58f8d1e18fad73d6055abe5008572f16ab3f31acd5d9` |

The memory quick pass was used only for evidence-boundary guidance, then superseded by current repository artifacts: `MEMORY.md:205-242`, source rollout ID `01a1a696-d169-7a82-bb69-cda112c781c9`. No current numerical or milestone claim rests on old memory.
