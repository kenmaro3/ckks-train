#!/usr/bin/env python3
"""Standalone new synthetic Full oracle; no production encoders imported.

full_grammar contains copied ACCEPTED independent table data only. This new
encoder emits primitive offsets and derives the entire dependency chain.
"""
import copy
import hashlib
import json
import struct
from pathlib import Path
import full_grammar as g

P = Path(__file__).resolve().parent
ROOT = P.parents[3]
ENC = "hekit-study-reference-encoding-v1"
LABEL = "SYNTHETIC FULL INTEGRATION ORACLE; NO LIVE CAPABILITY OR SCIENTIFIC RESULT"
sha = lambda b: hashlib.sha256(b).hexdigest()
I = json.loads((P / "inputs.json").read_text())
FREEZE = json.loads((P / "input-freeze.json").read_text())
assert sha((P / "inputs.json").read_bytes()) == FREEZE["inputs_sha256"]
assert sha((P / "full_grammar.py").read_bytes()) == FREEZE["full_grammar_sha256"]
M = {"classification": LABEL, "records": {}, "raw_files": {}, "negative_cases": []}
LEAVES = {}
DOCS = {}
LINKS = {}
NAMES = ("inputs", "implementation", "layout", "resource", "user")


def write(name, b):
    p = P / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)
    M["raw_files"][name] = {"bytes": len(b), "sha256": sha(b)}


def json_bytes(v):
    return (json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def jwrite(name, v):
    write(name, json_bytes(v))


class Encoder:
    def __init__(self):
        self.b = bytearray()
        self.fields = []

    def leaf(self, path, typ, value):
        if typ in ("Text", "Hash"):
            raw = value.encode()
            b = struct.pack("<Q", len(raw)) + raw
        elif typ == "Bytes":
            raw = bytes.fromhex(value)
            b = struct.pack("<Q", len(raw)) + raw
        elif typ == "U64":
            b = struct.pack("<Q", value)
        elif typ == "I32":
            b = struct.pack("<i", value)
        elif typ == "tag":
            b = bytes([int(value)])
        else:
            raise AssertionError(typ)
        self.fields.append({"path": path, "type": typ, "value": value,
                            "offset": len(self.b), "length": len(b)})
        self.b.extend(b)

    def typed(self, typ, value, path):
        tag = typ[0]
        if tag in ("text", "hash", "git"):
            self.leaf(path, "Hash" if tag == "hash" else "Text", value)
        elif tag in ("u64", "bool", "enum"):
            v = typ[1][value] if tag == "enum" else int(value)
            self.leaf(path, "U64" if tag == "u64" else "tag", v)
        elif tag == "bytes":
            self.leaf(path, "Bytes", value)
        elif tag == "f64":
            self.leaf(path, "U64", struct.unpack("<Q", struct.pack("<d", value))[0])
        elif tag == "record":
            assert set(value) == {k for k, _ in typ[1]}, (path, set(value))
            for k, t in typ[1]:
                self.typed(t, value[k], path + "." + k)
        elif tag == "list":
            assert len(value) <= typ[2]
            self.leaf(path + ".count", "U64", len(value))
            for i, v in enumerate(value):
                self.typed(typ[1], v, path + "[%d]" % i)
        elif tag == "option":
            self.leaf(path + ".present", "tag", value is not None)
            if value is not None:
                self.typed(typ[1], value, path + ".value")
        elif tag in ("accuracy", "target"):
            assert struct.pack(">d", value["value"]).hex() == value["bits"]
            if tag == "accuracy":
                self.typed(g.U, value["correct"], path + ".correct")
                self.typed(g.U, value["test_count"], path + ".test_count")
            self.typed(g.F, value["value"], path + ".value_bits")
        elif tag == "observation":
            available = value["status"] == "available"
            self.leaf(path + ".status", "tag", 0 if available else 1)
            self.typed(typ[1] if available else g.T(256), value["value" if available else "reason"],
                       path + (".value" if available else ".reason"))
        elif tag == "captured":
            assert value["status"] == "captured"
            self.leaf(path + ".status", "tag", 0)
            self.typed(typ[1], value["value"], path + ".value")
        elif tag == "integrity":
            self.leaf(path + ".encoding", "Text", value["encoding"])
            self.leaf(path + ".hash", "Text", value["hash"])
        else:
            raise AssertionError(tag)

    def save(self, name, kind="aux", deps=()):
        b = bytes(self.b)
        LEAVES[name] = self.fields
        write(name + ".bin", b)
        write(name + ".hex", (b.hex() + "\n").encode())
        M["records"][name] = {"kind": kind, "bytes": len(b), "canonical_sha256": sha(b),
                              "dependencies": list(deps), "primitive_count": len(self.fields)}
        return b


def emit_leafs(name, fields, deps=()):
    e = Encoder()
    for f in fields:
        e.leaf(f["path"], f["type"], f["value"])
    assert len(e.b) <= 2097152
    return e.save(name, "evidence" if name.endswith(".evidence") else "aux", deps)


def component(name, domain, typ, value):
    e = Encoder()
    e.leaf("header.encoding", "Text", ENC)
    e.leaf("header.domain", "Text", "hekit-study-" + domain + "-v1")
    e.typed(typ, value, "body")
    return e.save(name)


def emit_doc(name, kind, value, deps=()):
    value = copy.deepcopy(value)
    e = Encoder()
    e.leaf("header.encoding", "Text", ENC)
    e.leaf("header.domain", "Text", value["schema"])
    e.typed(g.SCHEMAS[kind], value, "body")
    can = e.save(name, kind, deps)
    value["integrity"]["payload_sha256"] = sha(can)
    raw = json_bytes(value)
    write(name + ".json", raw)
    M["records"][name].update(file_bytes=len(raw), file_sha256=sha(raw))
    DOCS[name] = value
    LINKS[name] = {"file_sha256": sha(raw), "payload_sha256": sha(can)}
    return value


def template(name):
    return copy.deepcopy(I["evidence_leaf_templates"][name])


def values(fields):
    return {f["path"]: f["value"] for f in fields}


def setv(fields, path, value):
    matches = [f for f in fields if f["path"] == path]
    assert len(matches) == 1, path
    matches[0]["value"] = value


def replace_values(fields, mapping):
    for f in fields:
        if isinstance(f["value"], str) and f["value"] in mapping:
            f["value"] = mapping[f["value"]]


def link_fields(prefix, link):
    return [{"path": prefix + "." + k, "type": "Hash", "value": link[k]}
            for k in ("file_sha256", "payload_sha256")]


def prior_fields(prefix, prior):
    out = [{"path": prefix + ".present", "type": "tag", "value": int(prior is not None)}]
    if prior:
        out += [{"path": prefix + ".attempt_id", "type": "Text", "value": prior["attempt_id"]}]
        out += link_fields(prefix + ".admission", prior["admission"])
        out += [{"path": prefix + ".start.present", "type": "tag", "value": int(prior["start"] is not None)}]
        if prior["start"]:
            out += link_fields(prefix + ".start", prior["start"])
    return out


def replace_field_range(fields, path, replacement):
    i = next(i for i, f in enumerate(fields) if f["path"] == path)
    fields[i:i+1] = replacement


def retained(path, b):
    write("retained/" + path, b)
    return {"path": path, "length": len(b), "sha256": sha(b)}


def fileref_set(fields, prefix, ref):
    for k in ("path", "length", "sha256"):
        setv(fields, prefix + "." + k, ref[k])


base = I["full_templates"]
old_ev = values(template("full.inputs.evidence"))
old_impl = values(template("full.implementation.evidence"))
SOURCE = I["settings"]["source_commit"]
CONTRACT = base["initial"]["contract"]
RECEIPT = base["initial"]["selected_input"]["seed_receipt"]
MATERIALIZED = base["initial"]["selected_input"]["materialized_sha256"]
POLICY = copy.deepcopy(base["lock"]["policy"])
POLICY["approval_sha256"] = old_ev["common.approval_sha256"]
CASE = copy.deepcopy(base["initial"]["case"])
EXEC = copy.deepcopy(base["initial"]["execution"])
EXEC["implementation_revision"] = SOURCE
EXEC["implementation_manifest_sha256"] = old_ev["common.source_manifest_sha256"]
for k, ev in (("parameter_identity_sha256", "parameters"), ("layout_plan_sha256", "layout"),
              ("trainer_plan_sha256", "trainer"), ("key_policy_sha256", "keys"),
              ("security_policy_sha256", "security"), ("runtime_policy_sha256", "runtime")):
    EXEC[k] = old_impl["implementation.applicability." + ev]
EXEC["physical_binding_sha256"] = values(template("full.resource.evidence"))["resource.physical"]
CASE_BYTES = component("aux/registered-case", "registered-case", g.Case, CASE)
EXEC_BYTES = component("aux/execution", "case-binding", g.Execution, EXEC)
LOCK_ID_BYTES = component("aux/lock-id", "target-lock-id",
    g.S(campaign_id=g.T(128), profile_id=g.T(128)),
    {"campaign_id": base["initial"]["campaign_id"], "profile_id": base["initial"]["profile_id"]})
assert sha(LOCK_ID_BYTES) == base["lock"]["lock_id"]


def context(value):
    p = value["provenance"]["value"]
    p["source"]["commit"] = {"status": "available", "value": SOURCE}
    p["build"]["source_manifest_sha256"] = old_ev["common.source_manifest_sha256"]
    p["recheck"]["source_manifest_sha256"] = old_ev["common.source_manifest_sha256"]
    p["build"]["receipt_file_sha256"] = old_ev["common.train_receipt_sha256"]
    p["recheck"]["build_receipt_file_sha256"] = old_ev["common.train_receipt_sha256"]
    p["build"]["cargo_lock_sha256"] = POLICY["portable_lock_sha256"]
    p["executable"]["start_sha256"] = old_ev["common.executable_sha256"]
    p["executable"]["end_sha256"] = old_ev["common.executable_sha256"]
    p["invocation"]["argv"] = [{"encoding": "unix_bytes", "bytes": b}
                                for b in I["settings"]["non_utf8_argv_hex"]]


reference = copy.deepcopy(base["reference"])
context(reference)
reference = emit_doc("reference", "reference", reference)
lock = copy.deepcopy(base["lock"])
lock["policy"] = POLICY
lock["reference"]["link"] = LINKS["reference"]
context(lock)
lock = emit_doc("lock", "lock", lock, ["reference"])
assert reference["identity"]["contract"] == lock["contract"] == CONTRACT
assert lock["target"]["bits"] == struct.pack(">d", max(0.0, reference["final_observation"]["value"] - 0.005)).hex()
THRESHOLD = int(lock["target"]["bits"], 16)
assert THRESHOLD != 0

REQUEST = emit_leafs("aux/request", template("full.request"))
con = template("full.consumption")
setv(con, "seed", I["settings"]["seed"])
setv(con, "request_sha256", sha(REQUEST))
setv(con, "trainer_sha256", EXEC["trainer_plan_sha256"])
setv(con, "selected_receipt_sha256", RECEIPT["sha256"])
setv(con, "total_updates", I["settings"]["full_updates"])
CONSUMPTION = emit_leafs("aux/consumption", con, ["aux/request", "aux/execution"])
assert I["settings"]["full_updates"] == reference["completion"]["planned_updates"]
retained("linked/contract.bin", bytes.fromhex(CONTRACT["canonical_bytes"]))
retained("linked/receipt-seed0.bin", bytes.fromhex(reference["identity"]["seed_receipt"]["canonical_bytes"]))
retained("linked/receipt-seed7.bin", bytes.fromhex(RECEIPT["canonical_bytes"]))
retained("linked/materialized-seed7.bin", b"FICTIONAL:materialized7")
assert sha(b"FICTIONAL:materialized7") == MATERIALIZED
retained("linked/materialized-seed0.bin", b"FICTIONAL:materialized0")
assert sha(b"FICTIONAL:materialized0") == reference["identity"]["materialized_sha256"]


def predecessor_deps(prior):
    if prior is None:
        return []
    result = []
    for key in ("admission", "start"):
        if prior[key] is not None:
            names = [name for name, link in LINKS.items() if link == prior[key]]
            assert len(names) == 1
            result += names
    return result


def authority(prefix, attempt, prior):
    f = template("full.authority.scope")
    replacements = {old_ev["common.campaign_id"]: base["initial"]["campaign_id"],
        old_ev["common.profile_id"]: base["initial"]["profile_id"],
        old_ev["full.case_id"]: sha(CASE_BYTES), old_ev["full.attempt_id"]: attempt["attempt_id"]}
    replace_values(f, replacements)
    setv(f, "instruction_bytes", I["settings"]["instruction"].encode().hex())
    setv(f, "scopes[0].retry", int(prior is not None))
    setv(f, "scopes[0].overlap", int(prior is not None))
    if prior:
        i = next(i for i, x in enumerate(f) if x["path"] == "scopes[0].overlap")
        nested = prior_fields("scopes[0].predecessor", prior)[1:]
        f[i:i] = nested
    scope = emit_leafs(prefix + "/authority.scope", f, predecessor_deps(prior))
    c = template("full.authority.current")
    setv(c, "authority_sha256", sha(scope))
    setv(c, "observed.unix_ns", attempt["observation"])
    current = emit_leafs(prefix + "/authority.current", c, [prefix + "/authority.scope"])
    sr = retained(prefix + "/authority.scope.bin", scope)
    cr = retained(prefix + "/authority.current.bin", current)
    return scope, current, sr, cr


def make_attempt(prefix, attempt, prior, mutated_prior=None):
    # A negative prior changes its claim and all corresponding evidence fields;
    # the current admission remains against the original target.
    a = copy.deepcopy(base["initial"])
    a["policy"] = copy.deepcopy(POLICY)
    a["lock"] = copy.deepcopy(LINKS["lock"])
    a["reference"] = copy.deepcopy(LINKS["reference"])
    a["case"], a["case_id"] = copy.deepcopy(CASE), sha(CASE_BYTES)
    a["execution"], a["execution_binding_sha256"] = copy.deepcopy(EXEC), sha(EXEC_BYTES)
    a["attempt_id"], a["predecessor"] = attempt["attempt_id"], copy.deepcopy(prior)
    a["admitted"]["unix_ns"] = attempt["admitted"]
    context(a)
    if mutated_prior == "lock":
        a["lock"]["file_sha256"] = I["negative_settings"]["prior_lock"]
    elif mutated_prior == "reference":
        a["reference"]["payload_sha256"] = I["negative_settings"]["prior_reference"]
    elif mutated_prior == "policy":
        a["policy"]["approval_sha256"] = I["negative_settings"]["prior_policy_approval"]
    scope, current, sr, cr = authority(prefix, attempt, prior)
    claim = retained(prefix + "/claim-record.txt", (LABEL + "\nclaim=CLAIM-" + attempt["name"] + "\nattempt=" + attempt["attempt_id"] + "\n").encode())
    for gate in NAMES:
        f = template("full." + gate + ".evidence")
        mapping = {old_ev["common.campaign_id"]: a["campaign_id"], old_ev["common.profile_id"]: a["profile_id"],
                   old_ev["common.source_revision"]: SOURCE,
                   old_ev["common.contract_sha256"]: CONTRACT["sha256"],
                   old_ev["common.selected_receipt_sha256"]: RECEIPT["sha256"],
                   old_ev["common.materialized_sha256"]: MATERIALIZED,
                   old_ev["full.case_id"]: sha(CASE_BYTES), old_ev["full.attempt_id"]: a["attempt_id"]}
        replace_values(f, mapping)
        setv(f, "common.execution", EXEC_BYTES.hex())
        setv(f, "common.approval_sha256", a["policy"]["approval_sha256"])
        setv(f, "full.case", CASE_BYTES.hex())
        setv(f, "full.lock_id", a["lock_id"])
        setv(f, "full.threshold_bits", THRESHOLD)
        for key in ("file_sha256", "payload_sha256"):
            setv(f, "full.lock." + key, a["lock"][key])
            setv(f, "full.reference." + key, a["reference"][key])
        replace_field_range(f, "full.predecessor.present", prior_fields("full.predecessor", prior))
        if gate in ("inputs", "layout"):
            setv(f, gate + ".consumed_plan", CONSUMPTION.hex())
            setv(f, gate + ".consumed_plan_sha256", sha(CONSUMPTION))
        if gate == "inputs":
            for k, ident in (("contract", CONTRACT), ("selected_receipt", RECEIPT), ("independent_target_receipt", RECEIPT)):
                setv(f, "inputs." + k + ".canonical_length", len(bytes.fromhex(ident["canonical_bytes"])))
        if gate == "layout":
            setv(f, "layout.request", REQUEST.hex())
            setv(f, "layout.request_sha256", sha(REQUEST))
        if gate == "resource":
            setv(f, "resource.observation.clock.unix_ns", attempt["observation"])
            setv(f, "resource.claim.claim_id", "CLAIM-" + attempt["name"])
            setv(f, "resource.claim.overlap", int(prior is not None))
            fileref_set(f, "resource.claim.claim_record", claim)
        if gate == "user":
            setv(f, "user.authority_bytes", scope.hex())
            setv(f, "user.authority_sha256", sha(scope))
            setv(f, "user.current_bytes", current.hex())
            setv(f, "user.current_sha256", sha(current))
            fileref_set(f, "user.authority_file", sr)
            fileref_set(f, "user.current_file", cr)
            setv(f, "user.observed.unix_ns", attempt["observation"])
        deps = ["reference", "lock", "aux/execution", "aux/registered-case", "aux/consumption"] + predecessor_deps(prior)
        if gate == "layout":
            deps += ["aux/request"]
        if gate == "user":
            deps += [prefix + "/authority.scope", prefix + "/authority.current"]
        for field, value in values(f).items():
            if field.endswith(".root"):
                path = values(f).get(field[:-5] + ".path")
                if path is not None:
                    deps += ["retained/" + path]
        data = emit_leafs(prefix + "/" + gate + ".evidence", f, deps)
        a["gates"][gate] = {"policy_revision": "test-d7-" + gate + "-v1", "evidence_sha256": sha(data)}
    gb = component(prefix + "/gates", "admission-gates", g.Gates, a["gates"])
    M["records"][prefix + "/gates"]["dependencies"] = [prefix + "/" + n + ".evidence" for n in NAMES]
    a["gates_sha256"] = sha(gb)
    emit_doc(prefix + "/admission", "admission", a, ["reference", "lock", "aux/execution", "aux/registered-case", prefix + "/gates"] + predecessor_deps(prior) + [prefix + "/" + n + ".evidence" for n in NAMES])
    s = copy.deepcopy(base["start-initial"])
    for key in ("policy", "lock", "reference", "case_id", "execution_binding_sha256", "attempt_id"):
        s[key] = copy.deepcopy(a[key])
    s["admission"] = LINKS[prefix + "/admission"]
    s["started"]["unix_ns"] = attempt["started"]
    context(s)
    emit_doc(prefix + "/start", "start", s, [prefix + "/admission"])
    return a


attempts = {x["name"]: x for x in I["attempts"]}
for name in ("A", "B", "C"):
    x = attempts[name]
    prior = None
    if x["prior"]:
        prior = {"attempt_id": attempts["A"]["attempt_id"], "admission": LINKS["A/admission"],
                 "start": LINKS["A/start"] if x["prior"]["with_start"] else None}
    make_attempt(name, x, prior)

for field in ("lock", "reference", "policy"):
    root = "negative/c1-prior-" + field
    make_attempt(root + "/A", attempts["A"], None, field)
    prior = {"attempt_id": attempts["A"]["attempt_id"], "admission": LINKS[root + "/A/admission"], "start": None}
    make_attempt(root + "/B", attempts["B"], prior)
    M["negative_cases"].append({"name": "c1-prior-" + field, "classification": "internally rehashed cross-record target inconsistency",
        "prior": root + "/A/admission", "current": root + "/B/admission", "target": "lock", "reference": "reference",
        "required_failure": "check_target_claims(prior,same retained target/reference/policy)",
        "predecessor_only_expected": "passes", "self_integrity_expected": "passes",
        "all_evidence_hashes_recomputed": True, "production_execution": False})

# A raw-file-only change preserves canonical payload; an unchanged dependent Link fails.
raw = (P / "A/admission.json").read_bytes() + b"\n"
write("negative/admission-whitespace.json", raw)
M["negative_cases"].append({"name": "admission-whitespace", "classification": "valid payload, wrong exact file Link",
    "file": "negative/admission-whitespace.json", "payload_sha256": LINKS["A/admission"]["payload_sha256"],
    "expected_file_sha256": LINKS["A/admission"]["file_sha256"], "actual_file_sha256": sha(raw),
    "dependent": "B/admission", "production_execution": False})
s = copy.deepcopy(DOCS["B/start"])
s["admission"]["payload_sha256"] = "5" * 64
emit_doc("negative/start-wrong-payload-link", "start", s, ["B/admission"])
M["negative_cases"].append({"name": "start-wrong-payload-link", "classification": "self-rehashed wrong admission payload Link",
    "record": "negative/start-wrong-payload-link", "required_failure": "check_start_claims", "production_execution": False})
bad = copy.deepcopy(DOCS["A/admission"])
bad["attempt_id"] = "MALFORMED-INTEGRITY"
write("negative/admission-stale-self-hash.json", json_bytes(bad))
M["negative_cases"].append({"name": "admission-stale-self-hash", "classification": "malformed integrity; self-hash intentionally stale",
    "file": "negative/admission-stale-self-hash.json", "required_failure": "payload integrity before cross-links", "production_execution": False})

# Existing copies remain immutable; record the same inventory after generation.
before = json.loads((P / "preserved-before.json").read_text())
for path, v in before.items():
    b = (ROOT / path).read_bytes()
    assert len(b) == v["bytes"] and sha(b) == v["sha256"], path
jwrite("preserved-after.json", before)
jwrite("resolved-literal-inputs.json", {"classification": LABEL, "records": LEAVES})
jwrite("dependency-graph.json", {"classification": LABEL,
    "records": {name: item["dependencies"] for name, item in M["records"].items()},
    "history_snapshots": {"Initial": ["A/admission", "A/start"],
       "Retry_no_prior_start": ["A/admission", "B/admission", "B/start"],
       "Retry_with_prior_start": ["A/admission", "A/start", "C/admission", "C/start"]},
    "note": "B admission45/start46 precede A start50. C admission60 follows it. These are different inspected history snapshots, not one static predecessor-absence assertion."})
# Register all retained preimages, including accepted stubs copied at input freeze.
for p in sorted((P / "retained").rglob("*")):
    if p.is_file():
        b = p.read_bytes()
        M["raw_files"][str(p.relative_to(P))] = {"bytes": len(b), "sha256": sha(b)}
jwrite("manifest.json", M)
log = {"classification": LABEL, "generator_invocations_this_run": 1, "exit": 0,
       "records": len(M["records"]), "evidence_records": sum(v["kind"] == "evidence" for v in M["records"].values()),
       "negative_cases": len(M["negative_cases"]), "old_files_preserved": len(before),
       "input_freeze_sha256": sha((P / "input-freeze.json").read_bytes()),
       "manifest_sha256": sha((P / "manifest.json").read_bytes()),
       "project_encoder_imports": 0, "production_parser_calls": 0, "native_HE_Cargo_registry_materializer_calls": 0}
(P / "generation-log.json").write_bytes(json_bytes(log))
print(json.dumps(log, sort_keys=True))
