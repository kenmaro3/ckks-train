"""Read-only source/preservation evidence. No Cargo or test executable invocation."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path.cwd()
OUT = pathlib.Path(__file__).resolve().parent
BASE = "9aba62f0a2a2b565466216511238dec3fd7a244f"
PATHS = [
    "crates/ckks-train-bench/src/study_runtime.rs",
    "crates/ckks-train-bench/src/study_runtime/tests.rs",
    "crates/ckks-train-bench/src/study_provenance.rs",
    "crates/ckks-train-bench/src/study_provenance/context.rs",
    "crates/ckks-train-bench/src/study_provenance/train_capture.rs",
    "crates/ckks-train-bench/src/study_provenance/train_capture_tests.rs",
    "crates/ckks-train-bench/src/study_reference/document.rs",
    "crates/ckks-train-bench/src/study_reference/document/case_provenance.rs",
    "crates/ckks-train-bench/src/study_reference/document/case_provenance_tests.rs",
    "crates/ckks-train-bench/src/study_reference/document/producer.rs",
]


def git(*args):
    return subprocess.check_output(["git", *args])


def old(path):
    return git("show", f"{BASE}:{path}")


def sha(value):
    return hashlib.sha256(value).hexdigest()


changed = git("diff", BASE, "--name-only", "--", "crates", "Cargo.toml", "Cargo.lock").decode().splitlines()
assert set(changed).issubset(PATHS), changed
tracked = git("ls-tree", "-r", "--name-only", BASE, "--", "crates/ckks-train-bench/src", "Cargo.toml", "Cargo.lock").decode().splitlines()
unchanged = [path for path in tracked if path not in PATHS]
for path in unchanged:
    assert (ROOT / path).read_bytes() == old(path), path

context = "crates/ckks-train-bench/src/study_provenance/context.rs"
begin = b"    pub(crate) fn capture(&self, p: &MaterializedProfile)"
end = b"// Non-authoritative inspection values:"
assert old(context).split(begin, 1)[1].split(end, 1)[0] == (ROOT / context).read_bytes().split(begin, 1)[1].split(end, 1)[0]
producer = "crates/ckks-train-bench/src/study_reference/document/producer.rs"
begin = b"pub(super) fn captured(v:"
assert old(producer).split(begin, 1)[1].split(b"\nfn common_value(", 1)[0] == (ROOT / producer).read_bytes().split(begin, 1)[1].split(b"\npub(super) fn captured_case(", 1)[0]
for path in [
    "crates/ckks-train-bench/src/study_reference/document/case_provenance_tests.rs",
    "crates/ckks-train-bench/src/study_reference/document.rs",
]:
    assert (ROOT / path).read_bytes().startswith(old(path)), path

pins = {path: sha((ROOT / path).read_bytes()) for path in PATHS}
(OUT / "source-pins.json").write_text(json.dumps({
    "source_base": BASE,
    "head_at_static_capture": git("rev-parse", "HEAD").decode().strip(),
    "files": pins,
}, indent=2) + "\n")
result = {
    "changed_tracked_source_paths": changed,
    "allowed_focused_new_paths": [path for path in PATHS if path not in tracked],
    "unchanged_tracked_bench_source_and_workspace_cargo_count": len(unchanged),
    "unchanged_case_golden_count": len([path for path in unchanged if "/case_provenance_goldens/" in path]),
    "reference_capture_finish_and_helpers_byte_identical": True,
    "reference_captured_producer_byte_identical": True,
    "preexisting_case_tests_and_document_source_prefix_byte_identical": True,
    "unowned_results_present_sha256": {path: sha((ROOT / path).read_bytes()) for path in [
        "results/mnist_blocks_128-64_plain_b48.json",
        "results/mnist_blocks_64_plain_b48.json",
    ]},
    "unowned_vscode_exists": (ROOT / ".vscode").is_dir(),
    "runtime_tests_executed": 0,
    "production_run_owned_executed": 0,
}
(OUT / "static-preservation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
