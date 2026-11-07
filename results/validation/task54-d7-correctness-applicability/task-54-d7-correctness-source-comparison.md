# Rename-aware historical source comparison

Read-only comparison against committed263c830e3568ccfa4bbec6bc746c08a4971aa197, not the concurrently edited worktree. This addresses the prior map warning that a path-only diff made renamed code appear all-new. No test, Cargo, estimator, HE, native context or SSH invocation. No positive applicability decision or current-source correctness claim.

The exact two byte substitutions from scripts/rename.sh normalize old crate package/import names only. No whitespace/algorithm normalization. Scope is tracked core/ckks/nn src/tests and Cargo.toml; it omits bench, Cargo.lock, compiler/toolchain, transitive dependencies and generated/native code. Current additions are listed separately. Historical M1 documented KAT commit and actual legacy FULL1 producer commit are compared; historical docs/raw results remain unchanged.

## m1-kat

Historical commit: `37f9502281576edb96afde41f42843f6d917161f`. Counts: {"changed-after-rename": 14, "rename-only": 11, "byte-identical": 4}; new paths=66.

Changed after exact rename normalization:

- `crates/ckks-train-ckks/Cargo.toml`

- `crates/ckks-train-ckks/src/counter.rs`

- `crates/ckks-train-ckks/src/evaluator.rs`

- `crates/ckks-train-ckks/src/keys.rs`

- `crates/ckks-train-ckks/src/keyswitch.rs`

- `crates/ckks-train-ckks/src/lib.rs`

- `crates/ckks-train-ckks/src/linalg.rs`

- `crates/ckks-train-ckks/src/params.rs`

- `crates/ckks-train-core/Cargo.toml`

- `crates/ckks-train-core/src/baseconv.rs`

- `crates/ckks-train-core/src/error.rs`

- `crates/ckks-train-core/src/poly.rs`

- `crates/ckks-train-nn/Cargo.toml`

- `crates/ckks-train-nn/src/lib.rs`


These are syntactic differences, not a conclusion that every algorithm changed. The normalized diff is retained for targeted semantic inspection. Unchanged leaf bytes alone do not establish equivalent build, dispatch, parameters, dependencies or test execution.

## legacy-full1

Historical commit: `ff2c2ebc9d7946490bcf63f8b6be9f45fe67d49f`. Counts: {"changed-after-rename": 24, "rename-only": 28, "byte-identical": 8}; new paths=35.

Changed after exact rename normalization:

- `crates/ckks-train-ckks/Cargo.toml`

- `crates/ckks-train-ckks/src/bootstrap/bootstrapper.rs`

- `crates/ckks-train-ckks/src/bootstrap/evalmod.rs`

- `crates/ckks-train-ckks/src/bootstrap/linear.rs`

- `crates/ckks-train-ckks/src/bootstrap/mod.rs`

- `crates/ckks-train-ckks/src/bootstrap/poly.rs`

- `crates/ckks-train-ckks/src/evaluator.rs`

- `crates/ckks-train-ckks/src/lib.rs`

- `crates/ckks-train-ckks/src/linalg.rs`

- `crates/ckks-train-core/Cargo.toml`

- `crates/ckks-train-nn/Cargo.toml`

- `crates/ckks-train-nn/src/activation.rs`

- `crates/ckks-train-nn/src/dense.rs`

- `crates/ckks-train-nn/src/layout_reboot.rs`

- `crates/ckks-train-nn/src/lib.rs`

- `crates/ckks-train-nn/src/optim.rs`

- `crates/ckks-train-nn/src/refresh.rs`

- `crates/ckks-train-nn/src/trainer.rs`

- `crates/ckks-train-nn/src/trainer_ni.rs`

- `crates/ckks-train-nn/tests/backward.rs`

- `crates/ckks-train-nn/tests/dense.rs`

- `crates/ckks-train-nn/tests/grad.rs`

- `crates/ckks-train-nn/tests/trainer.rs`

- `crates/ckks-train-nn/tests/trainer_ni.rs`


These are syntactic differences, not a conclusion that every algorithm changed. The normalized diff is retained for targeted semantic inspection. Unchanged leaf bytes alone do not establish equivalent build, dispatch, parameters, dependencies or test execution.

## Next action and limit

Do not attach a blanket source-equivalence claim to either old result. The current correctness reader must select an applicable source-specific criterion and inspect actual changed algorithms/route dependencies; otherwise applicability remains missing. This comparison narrows that inspection to real changes and separates untouched leaf tests from newly introduced paths. It does not authorize broad reruns or invent new numeric gates. No full semantic review of the retained diffs was done in this pass.

## Exact artifacts

- `legacy-full1.diff`: 192964 bytes, SHA256 `433fbe5bf905a2553d74367b98d4cabb9cb1be776cb18f1e2e00526b4f88c6e5`

- `legacy-full1.json`: 28483 bytes, SHA256 `0ebe5b599d39666bcda6db07c7bd29e04648501b17861afb1e2181b4cea24343`

- `m1-kat.diff`: 129286 bytes, SHA256 `ad50bf929a8c2e28c8a575c131594bacbc61b65a6d1a3e8af57a5d2f5b82544a`

- `m1-kat.json`: 16458 bytes, SHA256 `768ec5e1f61d8ae2da6b90896d5175bc7b18d28125ba8e1baa67ec7131b7bad8`
