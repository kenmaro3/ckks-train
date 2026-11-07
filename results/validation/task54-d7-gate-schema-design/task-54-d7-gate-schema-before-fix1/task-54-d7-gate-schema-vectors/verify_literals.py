#!/usr/bin/env python3
"""Read-only fixture verification, except its own new verification-report.json.

No imports from generator/project, no production parser or real gate calls.
"""
import hashlib
import json
import struct
from pathlib import Path

P = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
inputs = json.loads((P / "literal-inputs.json").read_text())["records"]
manifest = json.loads((P / "manifest.json").read_text())
count = 0
for name, record in inputs.items():
    whole = (P / (name + ".bin")).read_bytes()
    pos = 0
    for item in record["fields"]:
        t, value = item["type"], item["value"]
        if t in ("Text", "Hash"):
            b = value.encode()
            encoded = len(b).to_bytes(8, "little") + b
        elif t == "Bytes":
            b = bytes.fromhex(value)
            encoded = len(b).to_bytes(8, "little") + b
        elif t == "tag":
            encoded = value.to_bytes(1, "little")
        elif t == "U64":
            encoded = value.to_bytes(8, "little")
        elif t == "I32":
            encoded = value.to_bytes(4, "little", signed=True)
        else:
            raise AssertionError(t)
        assert item["offset"] == pos
        assert item["length"] == len(encoded)
        assert whole[pos:pos+len(encoded)] == encoded, (name, item["path"])
        pos += len(encoded)
        count += 1
    assert pos == len(whole)
    assert bytes.fromhex((P / (name + ".hex")).read_text()) == whole
    assert manifest["records"][name]["canonical_sha256"] == sha(whole)
    assert len(whole) <= 2097152

for name, item in manifest["files"].items():
    b = (P / name).read_bytes()
    assert len(b) == item["bytes"] and sha(b) == item["sha256"], name

def values(name):
    return {f["path"]: f["value"] for f in inputs[name]["fields"]}

for branch in ("full", "quick"):
    for gate in ("inputs", "implementation", "layout", "resource", "user"):
        name = branch + "." + gate + ".evidence"
        v = values(name)
        assert v["header.synthetic"] == 1
        assert v["header.domain"] == "hekit-test-gate-" + gate + "-evidence-v1"
        assert v["scope"] == (branch == "quick")
        assert sha(bytes.fromhex(v["common.execution"])) == sha((P / (branch + ".execution.bin")).read_bytes())
        if branch == "quick":
            assert not any(k.startswith("full.") or "independent_target" in k for k in v)
            assert sha(bytes.fromhex(v["quick.operational_identity"])) == v["quick.operational_id"]
            assert v["quick.seed"] == 0 and v["quick.consumption"] == 1
    r = values(branch + ".resource.evidence")
    total = [0] * 4
    additional = [0] * 4
    assert r["resource.requirements.components.count"] == 22
    for i in range(22):
        prefix = "resource.requirements.components[%d]" % i
        assert r[prefix + ".kind"] == i
        mask = r[prefix + ".phase_mask"]
        resident = r[prefix + ".resident_bytes"]
        extra = r[prefix + ".additional_bytes"]
        if mask == 0:
            assert resident == extra == 0
        for phase in range(4):
            if mask & (1 << phase):
                total[phase] += resident + extra
                additional[phase] += extra
    assert max(total) == r["resource.requirements.peak_host_bytes"]
    assert max(additional) == r["resource.requirements.incremental_host_bytes"]
    assert max(total) > max(additional)  # literal demonstrates resident/additional distinction
    u = values(branch + ".user.evidence")
    assert sha(bytes.fromhex(u["user.authority_bytes"])) == u["user.authority_sha256"]
    assert sha(bytes.fromhex(u["user.current_bytes"])) == u["user.current_sha256"]

q = values("quick.identity")
for field, digest in (("registration", "registration_id"), ("execution", "execution_sha256"),
                      ("consumed_plan", "consumed_plan_sha256"), ("request", "request_sha256")):
    assert sha(bytes.fromhex(q[field])) == q[digest]

events = {}
for kind in ("preflight", "start", "end"):
    name = "quick." + kind
    raw = (P / (name + ".json")).read_bytes()
    doc = json.loads(raw)
    canonical = bytes.fromhex(doc["canonical_bytes"])
    assert canonical == (P / (name + ".bin")).read_bytes()
    assert sha(canonical) == doc["integrity"]["payload_sha256"]
    assert len(raw) <= 2097152 and len(canonical) <= 1048000
    assert doc["synthetic"] is True
    v = values(name)
    assert v["identity_sha256"] == sha((P / "quick.identity.bin").read_bytes())
    for gate in ("inputs", "implementation", "layout", "resource", "user"):
        assert v["gates." + gate + ".evidence_sha256"] == sha((P / ("quick." + gate + ".evidence.bin")).read_bytes())
    if kind != "preflight":
        assert (v["preflight.file_sha256"], v["preflight.payload_sha256"]) == events["preflight"]
    if kind == "end":
        assert (v["start.file_sha256"], v["start.payload_sha256"]) == events["start"]
        assert v["outcome"] == 0 and v["completed_updates"] == 2 and v["failure.present"] == 0
    events[kind] = (sha(raw), sha(canonical))

checksum_count = 0
for line in (P / "SHA256SUMS").read_text().splitlines():
    digest, path = line.split("  ", 1)
    assert sha((P / path).read_bytes()) == digest, path
    checksum_count += 1

result = {"classification": "SYNTHETIC BYTE CONSISTENCY ONLY; NO PRODUCTION SEMANTIC VALIDATION",
          "exit": 0, "literal_records": len(inputs), "primitive_spans_verified": count,
          "generation_manifest_files_verified": len(manifest["files"]),
          "checksum_entries_verified": checksum_count,
          "quick_exact_event_links_verified": 3, "source_or_HE_calls": 0,
          "mutation_rejections_executed": 0}
(P / "verification-report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, sort_keys=True))
