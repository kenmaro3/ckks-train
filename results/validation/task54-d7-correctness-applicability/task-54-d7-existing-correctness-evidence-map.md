# Existing correctness evidence — read-only applicability map

Prepared after99b8e88 at current docs-onlyHEAD0254dc3. This locates existing evidence for the future genuine implementation gate; it is not a gate, policy change, current-source attestation or new measurement. No test/HE/observer/SSH/estimator was run. Whole-file hashes below establish current local identity only, not original execution provenance. Paired security map remains task-54-d7-existing-security-evidence-map.md.

| Existing evidence | What can be located | What cannot be inferred |
|---|---|---|
| docs/m1-report.md §2–3 and current seal_kat.rs/precision.rs | Historical SEAL4.1.2 bidirectional KAT and precision policies, historical commits and exact documented exceptions; ignored kat/seal_2020/values.json remains present | Markdown statement is not a machine-verifiable raw run/current source linkage. M1 KAT seal_2020 coverage is not identical n16_train hybrid decomposition or bootstrapping coverage. Do not rerun KAT merely to populate metadata without concrete gap and runtime inventory. |
| docs/bootstrap.md §11.4–11.6 | Historical secure-n16 bootstrap precision gate, its preconditions, ignored heavy test names; trainer_ni n16_boot_keys_three_updates source exists | Existing ignored test declaration is not a fresh executed pass. Source currently differs in names/refactors; need exact applicable raw/source/config lineage before positive evidence reader. Original error thresholds remain; do not substitute small-fixture tolerance. |
| bootstrap-linux-20261106 | Original focused GNU/Linux test/release logs and historical CI metadata; conditional structural metadata accepted for that environment | Does not prove secure-size HE precision/security/resource, all Linux/toolchains, current native context or whole CI. Current process verifier still required. |
| task54-p4-cpu and task54-p5-cpu | Reviewed bounded synthetic I/NI bridge behavior; P5 includes changed-source P4 regression, original raw parity/counter and selected-source pins | Not canonical secure-N campaign precision, exact full physical memory, GPU or full encrypted-block release. Selected source pins are not exhaustive transitive executable attestation. P5 source-derived internal-stage counts and unobserved diagnostic allocations remain limited. |
| quadruped/raw/bootstrap_bench | Real original server bench raw+meta, recorded exit0, source sync commit plus dirty-files/diff marker; setup and observed runtime/counters | Timing benchmark alone not precision KAT. Dirty source marker is not a reconstructable complete approved manifest by itself. Historical RSS not a complete future CPU driver liveness/capacity claim. |
| quadruped/raw/R-41-FULL-1 | Actual legacy full NI seed0 output, runner provenance/exit and archived source values; M3 comparison already accepted | Cannot be relabeled as a prospective D7 started run or assigned missing new lock/admission/start links. Legacy/full1 result is not arbitrary new model/seed/layout/factory coverage; FULL2 designated accuracy remains distinct. |

Concrete next evidence-reader work: identify each mandatory original correctness check and retained actual parameter/route/source applicability; hold/read original evidence bytes through the reviewed context; compare exact algorithms/parameters/key roles/mode/source interfaces, record applicability decisions and their reviewed evidence. Where original logs/source capture are missing or coverage is narrower, return missing/failed applicability rather than a scalar PASS. Constructor/use checks after durable start still enforce actual key association, range/scale/levels before affected operations; final accuracy/time/KS are measured outcomes, not circular prerequisites. No new numeric tolerance or all-profile rerun requirement is established here.

Read scope was named files and selected sections, not a full repository audit. A search for secure-n16 test names/PASS in the selected results/validation log extensions found no matching entries; this does NOT establish absence from all logs, ignored ledger, remote history or old commits. Renamed crate paths make a path-limited37f9502..HEAD diff appear all-new; any future source applicability comparison must map historical hekit-* paths to current ckks-train-* and inspect substantive changes rather than use that misleading statistic. No acceptance/rejection is inferred from it.

## Current local identity inventory

|Path|Bytes|SHA256|
|---|---:|---|
|`docs/m1-report.md`|28369|`4932c6781288d93ebe5991657e425db1b1178b8c93b3b6123e28afed995271ab`|
|`docs/bootstrap.md`|57035|`9b69db665449b936e4c970b3656c909bb207f8f48ef4049a7f6f5dd226c7408d`|
|`docs/m3-report.md`|63011|`93dbcfd97fd3c82cf9b2ffcb76fa44027f0306a2e59f18d719c4475f8aa59434`|
|`docs/study-gates-design.md`|11148|`2d1601ad66512e98ce287a5d595e53d8e7dd77f439477c3070d3c5ad056356e3`|
|`results/validation/bootstrap-linux-20261106/README.md`|10275|`4b4270ee3f0f038e90c3e56e0257d0a6969205c7bf1470e14f6b7e0b3452df64`|
|`results/validation/task54-p4-cpu/README.md`|3356|`42ad240e027c40239d79aa149131dda8d66deadbb78e0022afa4ad548237d8d2`|
|`results/validation/task54-p5-cpu/README.md`|6093|`3fe32cc9d84be3bdd1b7932a5d873bdcdcb2566709f91c68072ea95a590fc7f2`|
|`results/validation/task54-p5-cpu/ledger/task-54-p5-cpu-runtime1-review.md`|13413|`45210ea628162407bc335036e62d6d498c80c91ea6645798e10b0177b43fd4bd`|
|`results/validation/task54-p5-cpu/ledger/task-54-p5-p4-regression-runtime1-review.md`|8328|`bc9ff9af4bfb4ed8834fdf539324a354a74a5edbf4d50835a9460c3e7afb17b6`|
|`results/quadruped/raw/bootstrap_bench/meta.json`|4023|`e8018db1641db28c1e82af60323e57c23c1442413fc98527d4f93faf4f92df6f`|
|`results/quadruped/raw/bootstrap_bench/program.log`|1223|`697d61194089c33a73ff46c1cd20a8c8b864c233fa3daccdfda047499da92578`|
|`results/quadruped/raw/bootstrap_bench/synced-commit.txt`|156|`921148f862990505818bd53679348baacd100de587625b075680b886d766d233`|
|`results/quadruped/raw/R-41-FULL-1/README.md`|2861|`34f94aac164f8073272fb3f8e981274ba42478f19e2ac09f2ca5d38f2fb147fa`|
|`results/quadruped/raw/R-41-FULL-1/mnist196_ni_when_needed_b64_full_seed0.json`|17657|`c18c7e217c710342164aad6d255f82f78089a306e7d8c05f2fc7c2778cd68acf`|
|`results/quadruped/raw/R-41-FULL-1/meta.json`|5158|`dcfbe1bb203ce25ce1123c84be68a117d9bfbd5fedc369782ae11f89f9d27b96`|
|`crates/ckks-train-ckks/tests/seal_kat.rs`|15427|`8686ad83eefaf3b8a205c4da3f40c18307199a9bd9f5b3615bf49503617f775c`|
|`crates/ckks-train-ckks/tests/precision.rs`|1444|`6837cb281ff23629a69d665bed8ccd684ad9fc251d3da47da570171ba8e8b632`|
|`crates/ckks-train-nn/tests/trainer_ni.rs`|43647|`de237a1b0f0b8a67e520412419e62f75b56387ec87995eaf933ccfa24e60257c`|
