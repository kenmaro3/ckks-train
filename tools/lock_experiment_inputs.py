#!/usr/bin/env python3
"""Lock reviewed captured input blobs; manually invoked, never called by preflight."""
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_COMMIT = "8eabcdd4cd46b3144c5a222ddb0ce533ee627b49"
SOURCES = (
    "results/quadruped/raw/reboot_emlp1_full/config_values.json",
    "results/quadruped/raw/reboot_emlp23_full/config_values.json",
)


def build(root):
    """Return a lock from historical blobs, rejecting changed working bytes."""
    entries = {}

    def blob(path):
        value = subprocess.run(
            ["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=root,
            check=True, capture_output=True,
        ).stdout
        if (root / path).read_bytes() != value:
            raise ValueError(f"working input differs from reviewed blob: {path}")
        entries[path] = {"path": path, "sha256": hashlib.sha256(value).hexdigest()}
        return value

    for source in SOURCES:
        config = json.loads(blob(source))
        models = config.get("models", {"emlp1": config})
        for model in models.values():
            for seed in model["per_seed"].values():
                blob(str(Path(source).parent / seed["indices_file"]))
    script = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "artifact_type": "input_integrity_lock",
        "source_commit": SOURCE_COMMIT,
        "scope": "reviewed repository snapshot; not upstream attestation or measurement",
        "generator": {"path": "tools/lock_experiment_inputs.py", "sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "method": "git show <source_commit>:<path>; compare exact working bytes; SHA256 of blob bytes"},
        "files": [entries[k] for k in sorted(entries)],
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lock = build(root)
    target = root / "crates/ckks-train-bench/configs/experiment_input_locks.json"
    target.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"locked {len(lock['files'])} reviewed blob files at {SOURCE_COMMIT}")
