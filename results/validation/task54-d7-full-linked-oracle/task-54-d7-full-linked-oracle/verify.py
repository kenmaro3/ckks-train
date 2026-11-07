#!/usr/bin/env python3
"""Independent verification of the new synthetic oracle, never production code.

Uses copied accepted grammar DATA but no generator functions/import. Rebuilds
JSON canonical bytes independently, checks primitive spans and cross-record
semantics, and demonstrates why rehashed C1 cases need same-target checking.
"""
import hashlib
import json
import struct
from pathlib import Path
import full_grammar as schema

P = Path(__file__).resolve().parent
ROOT = P.parents[3]
digest = lambda b: hashlib.sha256(b).hexdigest()
u64 = lambda n: int(n).to_bytes(8, "little")
def text(s):
    raw = s.encode("utf-8")
    return u64(len(raw)) + raw
def blob(s):
    raw = bytes.fromhex(s)
    return u64(len(raw)) + raw


def canonical_value(t, v):
    k = t[0]
    if k in ("text", "hash", "git"):
        if k == "text": assert len(v.encode()) <= t[1]
        if k == "hash": assert len(v) == 64 and all(c in "0123456789abcdef" for c in v)
        if k == "git": assert len(v) == 40 and all(c in "0123456789abcdef" for c in v)
        return text(v)
    if k == "u64": return u64(v)
    if k == "bool": return bytes([bool(v)])
    if k == "enum": return bytes([t[1][v]])
    if k == "bytes":
        assert len(bytes.fromhex(v)) <= t[1]
        return blob(v)
    if k == "f64": return struct.pack("<d", v)
    if k == "record":
        assert set(v) == {name for name, _ in t[1]}
        return b"".join(canonical_value(ty, v[name]) for name, ty in t[1])
    if k == "list":
        assert len(v) <= t[2]
        return u64(len(v)) + b"".join(canonical_value(t[1], x) for x in v)
    if k == "option": return b"\x00" if v is None else b"\x01" + canonical_value(t[1], v)
    if k == "captured":
        assert v["status"] == "captured"
        return b"\x00" + canonical_value(t[1], v["value"])
    if k == "observation":
        return b"\x00" + canonical_value(t[1], v["value"]) if v["status"] == "available" else b"\x01" + text(v["reason"])
    if k in ("target", "accuracy"):
        assert struct.pack(">d", v["value"]).hex() == v["bits"]
        return (u64(v["correct"]) + u64(v["test_count"]) if k == "accuracy" else b"") + struct.pack("<d", v["value"])
    if k == "integrity": return text(v["encoding"]) + text(v["hash"])
    raise AssertionError(k)


def canonical_document(kind, v):
    return text(schema.ENC) + text(v["schema"]) + canonical_value(schema.SCHEMAS[kind], v)


manifest = json.loads((P / "manifest.json").read_text())
leafs = json.loads((P / "resolved-literal-inputs.json").read_text())["records"]
inp = json.loads((P / "inputs.json").read_text())
docs = {}
links = {}
span_count = 0
file_refs = 0
for name, record in manifest["records"].items():
    raw = (P / (name + ".bin")).read_bytes()
    assert len(raw) == record["bytes"] and digest(raw) == record["canonical_sha256"]
    assert bytes.fromhex((P / (name + ".hex")).read_text()) == raw
    pos = 0
    for f in leafs[name]:
        v, typ = f["value"], f["type"]
        b = text(v) if typ in ("Text", "Hash") else blob(v) if typ == "Bytes" else u64(v) if typ == "U64" else int(v).to_bytes(4, "little", signed=True) if typ == "I32" else bytes([v])
        assert pos == f["offset"] and len(b) == f["length"]
        assert raw[pos:pos+len(b)] == b, (name, f["path"])
        pos += len(b); span_count += 1
    assert pos == len(raw)
    if record["kind"] in schema.SCHEMAS:
        file = (P / (name + ".json")).read_bytes()
        v = json.loads(file)
        assert v["synthetic"] is True and v["schema"].startswith("hekit-test-")
        assert canonical_document(record["kind"], v) == raw
        assert v["integrity"]["payload_sha256"] == digest(raw)
        assert len(file) == record["file_bytes"] and digest(file) == record["file_sha256"]
        docs[name] = v
        links[name] = {"file_sha256": digest(file), "payload_sha256": digest(raw)}
        limit = 70492488 if record["kind"] in ("admission", "reference") else 7324204 if record["kind"] == "lock" else 2097152
        assert len(file) <= limit
        if record["kind"] in ("admission", "reference", "lock"):
            holder = v["identity"] if record["kind"] == "reference" else v
            cb = len(bytes.fromhex(holder["contract"]["canonical_bytes"]))
            receipt = v["selected_input"]["seed_receipt"] if record["kind"] == "admission" else holder["seed_receipt"] if record["kind"] == "reference" else None
            rb = len(bytes.fromhex(receipt["canonical_bytes"])) if receipt else 0
            assert cb <= 2613526 and rb <= 31584142
            assert len(raw) - cb - rb <= 2097152 and len(file) - 2*cb - 2*rb <= 2097152
    else:
        assert len(raw) <= 2097152

for name, record in manifest["raw_files"].items():
    b = (P / name).read_bytes()
    assert len(b) == record["bytes"] and digest(b) == record["sha256"], name


def vfields(name):
    return {f["path"]: f["value"] for f in leafs[name]}


ref, lock = docs["reference"], docs["lock"]
assert ref["identity"]["seed"] == 0
assert lock["reference"]["link"] == links["reference"]
assert lock["contract"] == ref["identity"]["contract"]
for k in ("seed_receipt_sha256", "materialized_sha256"):
    expected = ref["identity"]["seed_receipt"]["sha256"] if k == "seed_receipt_sha256" else ref["identity"][k]
    assert lock["reference"][k] == expected
assert lock["reference"]["final_observation"] == ref["final_observation"]
for k, val in ref["completion"].items(): assert lock["reference"]["completion"][k] == val
assert lock["reference"]["completion"]["final_endpoint"] == ref["endpoints"][-1]
target = max(0.0, ref["final_observation"]["value"] - 0.005)
assert struct.pack(">d", target).hex() == lock["target"]["bits"] and target > 0

case_raw = (P / "aux/registered-case.bin").read_bytes()
exec_raw = (P / "aux/execution.bin").read_bytes()
con_raw = (P / "aux/consumption.bin").read_bytes()
req_raw = (P / "aux/request.bin").read_bytes()
cv = vfields("aux/consumption")
assert cv["seed"] == 7 and cv["consumption"] == 0 and cv["total_updates"] == 2
assert cv["request_sha256"] == digest(req_raw)
assert cv["total_updates"] <= vfields("aux/request")["work_limits.updates"]
for seed in (0, 7):
    identity = ref["identity"]["seed_receipt"] if seed == 0 else docs["A/admission"]["selected_input"]["seed_receipt"]
    assert (P / ("retained/linked/receipt-seed%d.bin" % seed)).read_bytes() == bytes.fromhex(identity["canonical_bytes"])
    assert digest(bytes.fromhex(identity["canonical_bytes"])) == identity["sha256"]
    materialized = ref["identity"]["materialized_sha256"] if seed == 0 else docs["A/admission"]["selected_input"]["materialized_sha256"]
    assert digest((P / ("retained/linked/materialized-seed%d.bin" % seed)).read_bytes()) == materialized
assert digest(bytes.fromhex(lock["contract"]["canonical_bytes"])) == lock["contract"]["sha256"]
evidence_checks = 0
start_checks = 0
for name, a in docs.items():
    if manifest["records"][name]["kind"] != "admission": continue
    prefix = name.rsplit("/", 1)[0]
    assert a["case_id"] == digest(case_raw)
    assert a["execution_binding_sha256"] == digest(exec_raw)
    assert a["case"]["registry_sha256"] == a["policy"]["registry_sha256"]
    assert a["case"]["seed"] == a["selected_input"]["seed"] == 7
    assert a["contract"] == lock["contract"]
    assert cv["selected_receipt_sha256"] == a["selected_input"]["seed_receipt"]["sha256"]
    assert cv["trainer_sha256"] == a["execution"]["trainer_plan_sha256"]
    assert digest((P / (prefix + "/gates.bin")).read_bytes()) == a["gates_sha256"]
    provenance = a["provenance"]["value"]
    for gate in ("inputs", "implementation", "layout", "resource", "user"):
        ename = prefix + "/" + gate + ".evidence"
        f = vfields(ename)
        assert digest((P / (ename + ".bin")).read_bytes()) == a["gates"][gate]["evidence_sha256"]
        assert f["header.synthetic"] == 1 and f["header.domain"].startswith("hekit-test-")
        assert f["scope"] == 0
        assert f["policy_revision"] == a["gates"][gate]["policy_revision"]
        assert bytes.fromhex(f["common.execution"]) == exec_raw
        assert bytes.fromhex(f["full.case"]) == case_raw
        assert f["full.case_id"] == a["case_id"] and f["full.attempt_id"] == a["attempt_id"]
        assert f["full.threshold_bits"] == int(lock["target"]["bits"], 16)
        assert f["common.contract_sha256"] == a["contract"]["sha256"]
        assert f["common.selected_receipt_sha256"] == a["selected_input"]["seed_receipt"]["sha256"]
        assert f["common.materialized_sha256"] == a["selected_input"]["materialized_sha256"]
        assert f["common.approval_sha256"] == a["policy"]["approval_sha256"]
        assert f["common.source_revision"] == provenance["source"]["commit"]["value"] == a["execution"]["implementation_revision"]
        assert f["common.source_manifest_sha256"] == provenance["build"]["source_manifest_sha256"] == provenance["recheck"]["source_manifest_sha256"] == a["execution"]["implementation_manifest_sha256"]
        assert f["common.train_receipt_sha256"] == provenance["build"]["receipt_file_sha256"] == provenance["recheck"]["build_receipt_file_sha256"]
        assert f["common.executable_sha256"] == provenance["executable"]["start_sha256"] == provenance["executable"]["end_sha256"]
        assert f["common.campaign_id"] == a["campaign_id"] and f["common.profile_id"] == a["profile_id"]
        if gate in ("implementation", "resource"):
            ap = gate + ".applicability."
            for key, ex in (("implementation_manifest", "implementation_manifest_sha256"), ("implementation_revision", "implementation_revision"), ("parameters", "parameter_identity_sha256"), ("keys", "key_policy_sha256"), ("security", "security_policy_sha256"), ("layout", "layout_plan_sha256"), ("trainer", "trainer_plan_sha256"), ("runtime", "runtime_policy_sha256")):
                assert f[ap + key] == a["execution"][ex]
                if gate == "implementation": assert f["implementation.correctness[0].applicability." + key] == f[ap + key]
            assert f[ap + "implementation"] == f[ap + "backend"] == f[ap + "mode"] == f[ap + "consumption"] == 0
            assert f[ap + "bootstrap.present"] == 0 and a["execution"]["bootstrap_descriptor_sha256"] is None
        for key in ("file_sha256", "payload_sha256"):
            assert f["full.lock." + key] == a["lock"][key]
            assert f["full.reference." + key] == a["reference"][key]
        pred = a["predecessor"]
        assert f["full.predecessor.present"] == int(pred is not None)
        if pred:
            assert f["full.predecessor.attempt_id"] == pred["attempt_id"]
            for key in ("file_sha256", "payload_sha256"):
                assert f["full.predecessor.admission." + key] == pred["admission"][key]
            assert f["full.predecessor.start.present"] == int(pred["start"] is not None)
            if pred["start"]:
                for key in ("file_sha256", "payload_sha256"):
                    assert f["full.predecessor.start." + key] == pred["start"][key]
        if gate == "inputs":
            assert f["inputs.contract.sha256"] == a["contract"]["sha256"]
            assert f["inputs.contract.canonical_length"] == len(bytes.fromhex(a["contract"]["canonical_bytes"]))
            assert f["inputs.materialized_sha256"] == a["selected_input"]["materialized_sha256"]
            for key in ("selected_receipt", "independent_target_receipt"):
                assert f["inputs." + key + ".sha256"] == a["selected_input"]["seed_receipt"]["sha256"]
                assert f["inputs." + key + ".canonical_length"] == len(bytes.fromhex(a["selected_input"]["seed_receipt"]["canonical_bytes"]))
            assert f["inputs.independent_materialized_sha256"] == a["selected_input"]["materialized_sha256"]
        if gate in ("inputs", "layout"):
            assert bytes.fromhex(f[gate + ".consumed_plan"]) == con_raw
            assert f[gate + ".consumed_plan_sha256"] == digest(con_raw)
        if gate == "layout":
            assert bytes.fromhex(f["layout.request"]) == req_raw and f["layout.request_sha256"] == digest(req_raw)
            for key, ex in (("parameters", "parameter_identity_sha256"), ("layout", "layout_plan_sha256"), ("trainer", "trainer_plan_sha256"), ("keys", "key_policy_sha256")):
                assert f["layout." + key] == a["execution"][ex]
        if gate == "resource":
            claim = (P / "retained" / f["resource.claim.claim_record.path"]).read_text()
            assert "\nclaim=" + f["resource.claim.claim_id"] + "\n" in claim
            assert "\nattempt=" + a["attempt_id"] + "\n" in claim
            assert f["resource.claim.overlap"] == int(pred is not None)
            assert f["resource.physical"] == f["resource.observation.physical"] == a["execution"]["physical_binding_sha256"]
            assert f["resource.requirements.components.count"] == 22
            totals, increments = [], []
            for phase in (1, 2, 4, 8):
                total = inc = 0
                for i in range(22):
                    base = "resource.requirements.components[%d]." % i
                    assert f[base + "kind"] == i
                    if f[base + "phase_mask"] & phase:
                        inc += f[base + "additional_bytes"]
                        total += f[base + "resident_bytes"] + f[base + "additional_bytes"]
                totals.append(total); increments.append(inc)
            assert max(totals) == f["resource.requirements.peak_host_bytes"]
            assert max(increments) == f["resource.requirements.incremental_host_bytes"]
        if gate == "user":
            scope = (P / (prefix + "/authority.scope.bin")).read_bytes()
            current = (P / (prefix + "/authority.current.bin")).read_bytes()
            assert bytes.fromhex(f["user.authority_bytes"]) == scope and f["user.authority_sha256"] == digest(scope)
            assert bytes.fromhex(f["user.current_bytes"]) == current and f["user.current_sha256"] == digest(current)
            av = vfields(prefix + "/authority.scope")
            assert av["scopes[0].attempt"] == a["attempt_id"] and av["scopes[0].registration_id"] == a["case_id"]
            assert av["scopes[0].retry"] == av["scopes[0].overlap"] == int(pred is not None)
            if pred:
                assert av["scopes[0].predecessor.attempt_id"] == pred["attempt_id"]
                for key in ("file_sha256", "payload_sha256"):
                    assert av["scopes[0].predecessor.admission." + key] == pred["admission"][key]
                assert av["scopes[0].predecessor.start.present"] == int(pred["start"] is not None)
                if pred["start"]:
                    for key in ("file_sha256", "payload_sha256"):
                        assert av["scopes[0].predecessor.start." + key] == pred["start"][key]
            assert vfields(prefix + "/authority.current")["authority_sha256"] == digest(scope)
        for path, value in f.items():
            if path.endswith(".root") and path[:-5] + ".path" in f:
                root = path[:-5]
                target = P / "retained" / f[root + ".path"]
                b = target.read_bytes()
                assert len(b) == f[root + ".length"] and digest(b) == f[root + ".sha256"], (ename, root)
                file_refs += 1
        evidence_checks += 1
    s = docs[prefix + "/start"]
    assert s["admission"] == links[name]
    for key in ("policy", "lock", "reference", "case_id", "execution_binding_sha256", "attempt_id"):
        assert s[key] == a[key]
    assert s["seed"] == a["selected_input"]["seed"]
    assert s["seed_receipt_sha256"] == a["selected_input"]["seed_receipt"]["sha256"]
    assert s["materialized_sha256"] == a["selected_input"]["materialized_sha256"]
    assert s["started"]["unix_ns"] >= a["admitted"]["unix_ns"]
    start_checks += 1


def target_failures(a):
    errors = []
    if a["lock"] != links["lock"]: errors.append("lock")
    if a["reference"] != links["reference"]: errors.append("reference")
    if a["policy"] != lock["policy"]: errors.append("policy")
    if a["contract"] != lock["contract"]: errors.append("contract")
    assert a["campaign_id"] == lock["campaign_id"] and a["profile_id"] == lock["profile_id"] and a["lock_id"] == lock["lock_id"]
    assert a["admitted"]["unix_ns"] >= lock["created"]["unix_ns"]
    return errors


def predecessor_only(current_name, prior_name, start_name=None):
    current, prior = docs[current_name], docs[prior_name]
    p = current["predecessor"]
    for key in ("campaign_id", "profile_id", "case_id", "contract", "selected_input"):
        assert current[key] == prior[key]
    assert p["attempt_id"] == prior["attempt_id"] != current["attempt_id"]
    assert p["admission"] == links[prior_name]
    assert prior["admitted"]["unix_ns"] <= current["admitted"]["unix_ns"]
    assert p["start"] == (links[start_name] if start_name else None)
    if start_name:
        assert docs[start_name]["admission"] == links[prior_name]
        assert docs[start_name]["started"]["unix_ns"] <= current["admitted"]["unix_ns"]
    return True


assert not target_failures(docs["A/admission"])
assert not target_failures(docs["B/admission"])
assert not target_failures(docs["C/admission"])
assert predecessor_only("B/admission", "A/admission")
assert predecessor_only("C/admission", "A/admission", "A/start")
c1_results = []
for field in ("lock", "reference", "policy"):
    root = "negative/c1-prior-" + field
    current_name, prior_name = root + "/B/admission", root + "/A/admission"
    assert predecessor_only(current_name, prior_name)
    assert not target_failures(docs[current_name])
    assert target_failures(docs[prior_name]) == [field]
    c1_results.append({"mutation": field, "self_integrity": "valid", "predecessor_only": "passes",
                       "current_same_target": "passes", "prior_same_target": "rejects " + field,
                       "production_parser_or_gate_executed": False})
ws = (P / "negative/admission-whitespace.json").read_bytes()
assert canonical_document("admission", json.loads(ws)) == (P / "A/admission.bin").read_bytes()
assert digest(ws) != links["A/admission"]["file_sha256"]
assert docs["negative/start-wrong-payload-link"]["admission"]["file_sha256"] == links["B/admission"]["file_sha256"]
assert docs["negative/start-wrong-payload-link"]["admission"]["payload_sha256"] != links["B/admission"]["payload_sha256"]
bad = json.loads((P / "negative/admission-stale-self-hash.json").read_bytes())
assert digest(canonical_document("admission", bad)) != bad["integrity"]["payload_sha256"]

graph = json.loads((P / "dependency-graph.json").read_text())["records"]
done = set()
def visit(name, active):
    assert name not in active, ("cycle", name)
    if name in done: return
    if name not in graph:
        assert name.startswith("retained/") and (P / name).is_file()
    else:
        assert graph[name] == manifest["records"][name]["dependencies"]
        for dependency in graph[name]: visit(dependency, active | {name})
    done.add(name)
for name in graph: visit(name, set())

before = json.loads((P / "preserved-before.json").read_text())
assert before == json.loads((P / "preserved-after.json").read_text())
for name, v in before.items():
    b = (ROOT / name).read_bytes()
    assert len(b) == v["bytes"] and digest(b) == v["sha256"]
result = {"classification": "INDEPENDENT SYNTHETIC CONSISTENCY AND C1 ORACLE CHECKS ONLY",
          "exit": 0, "records": len(leafs), "primitive_spans": span_count,
          "json_canonical_rebuilds": len(docs), "evidence_admission_links": evidence_checks,
          "retained_file_refs": file_refs, "start_admission_links": start_checks,
          "dependency_nodes_including_retained": len(done),
          "c1_results": c1_results, "old_files_unchanged": len(before),
          "production_parser_native_HE_Cargo_calls": 0}
(P / "verification-log.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, sort_keys=True))
