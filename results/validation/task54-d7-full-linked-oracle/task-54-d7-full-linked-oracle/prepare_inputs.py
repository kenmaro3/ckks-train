#!/usr/bin/env python3
"""Freeze accepted small literal inputs BEFORE new independent generation.

Read-only old families; writes only this new directory. No project imports.
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE.parent
ROOT = HERE.parents[3]
OLD_FULL = ROOT / "crates/ckks-train-bench/src/study_reference/target_goldens"
OLD_GATE = LEDGER / "task-54-d7-gate-schema-vectors"
OLD_ORACLE = LEDGER / "d6c3-independent-goldens/oracle.py"
sha = lambda b: hashlib.sha256(b).hexdigest()
assert not (HERE / "inputs.json").exists(), "Do not overwrite a frozen input ledger"

preserved = {}
for directory in (OLD_FULL, OLD_GATE):
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            b = p.read_bytes()
            preserved[str(p.relative_to(ROOT))] = {"bytes": len(b), "sha256": sha(b)}

full = {n: json.loads((OLD_FULL / (n + ".json")).read_text())
        for n in ("reference", "lock", "initial", "start-initial")}
gate = json.loads((OLD_GATE / "literal-inputs.json").read_text())["records"]
names = ["full." + n + ".evidence" for n in ("inputs", "implementation", "layout", "resource", "user")]
names += ["full.request", "full.consumption", "full.authority.scope", "full.authority.current"]
leafs = {name: [{k: f[k] for k in ("path", "type", "value")} for f in gate[name]["fields"]] for name in names}
for p in sorted((OLD_GATE / "retained").rglob("*")):
    if p.is_file():
        out = HERE / "retained" / p.relative_to(OLD_GATE / "retained")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)

inputs = {
    "classification": "SYNTHETIC INDEPENDENT FULL LINK ORACLE; NO GENUINE EVIDENCE OR LIVE AUTHORITY",
    "basis": "Existing accepted Full tables/literals plus accepted FIX1 evidence leaf tables; no Rust encoder or unfinished source",
    "full_templates": full,
    "evidence_leaf_templates": leafs,
    "attempts": [
        {"name": "A", "attempt_id": "LINKED-INITIAL", "admitted": 40, "started": 50, "observation": 38, "prior": None},
        {"name": "B", "attempt_id": "LINKED-RETRY-NO-START", "admitted": 45, "started": 46, "observation": 43, "prior": {"attempt": "A", "with_start": False}},
        {"name": "C", "attempt_id": "LINKED-RETRY-WITH-START", "admitted": 60, "started": 61, "observation": 58, "prior": {"attempt": "A", "with_start": True}}
    ],
    "settings": {"seed": 7, "full_updates": 2, "reference_seed": 0,
        "source_commit": "1111111111111111111111111111111111111111",
        "instruction": "SYNTHETIC existing scope: A Initial; B Retry from A before start; C Retry from A with start; explicit overlap for B/C. No actual user authority.",
        "non_utf8_argv_hex": ["2f66696374696f6e616c2f657865", "", "ff", "c3a9"]},
    "negative_settings": {
        "prior_lock": "2" * 64, "prior_reference": "3" * 64,
        "prior_policy_approval": "4" * 64,
        "classification": "Rehash prior self-integrity and dependent current predecessor links; same-target/policy composition must still reject"
    }
}
(HERE / "inputs.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False) + "\n")
(HERE / "preserved-before.json").write_text(json.dumps(preserved, indent=2) + "\n")
# Reuse only the already independently accepted immutable Full grammar DATA.
# The new generator below uses a separate encoder with offsets; no old encoder runs.
grammar = OLD_ORACLE.read_text().split("def sha(b):", 1)[0]
grammar = '# Accepted independent Full grammar tables, copied for standalone use.\n# Original oracle SHA256: ' + sha(OLD_ORACLE.read_bytes()) + '\n' + grammar
(HERE / "full_grammar.py").write_text(grammar)
freeze = {"inputs_sha256": sha((HERE / "inputs.json").read_bytes()),
          "full_grammar_sha256": sha((HERE / "full_grammar.py").read_bytes()),
          "preserved_before_sha256": sha((HERE / "preserved-before.json").read_bytes()),
          "old_full_files": sum(p.startswith(str(OLD_FULL.relative_to(ROOT))) for p in preserved),
          "old_gate_files": sum(p.startswith(str(OLD_GATE.relative_to(ROOT))) for p in preserved),
          "generation_has_not_started": True, "preparation_invocations": 1}
(HERE / "input-freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
print(json.dumps(freeze, sort_keys=True))
