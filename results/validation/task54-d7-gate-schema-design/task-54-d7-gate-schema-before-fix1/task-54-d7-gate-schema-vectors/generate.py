#!/usr/bin/env python3
"""Independent SYNTHETIC literal oracle. Python stdlib only; no project imports.

This script is a design artifact, not a production encoder or gate. It writes
only its own directory. It never invokes a subprocess, observer, HE or registry.
"""
import hashlib
import json
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
A, K = 2097152, 1048000
ENC = "hekit-study-reference-encoding-v1"
LABEL = "SYNTHETIC TEST ORACLE ONLY; NO LIVE CAPABILITY OR SCIENTIFIC/RESOURCE CLAIM"
RECORDS = {}
RETAINED = {}
INPUTS = {}
MANIFEST = {"classification": LABEL, "files": {}, "records": {}, "mutations": []}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dump(name, b):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)
    MANIFEST["files"][name] = {"bytes": len(b), "sha256": sha(b)}


def js(name, value):
    dump(name, (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode())


class Record:
    def __init__(self, name, domain, synthetic=True):
        self.name = name
        self.parts = []
        self.fields = []
        self.text("header.encoding", ENC)
        self.text("header.domain", domain)
        if synthetic is not None:
            self.tag("header.synthetic", int(synthetic))

    def put(self, path, typ, value, b):
        off = sum(map(len, self.parts))
        self.fields.append({"path": path, "type": typ, "value": value,
                            "offset": off, "length": len(b)})
        self.parts.append(b)

    def text(self, path, value):
        b = value.encode("utf-8")
        assert len(b) <= 4096
        self.put(path, "Text", value, struct.pack("<Q", len(b)) + b)

    def h(self, path, value):
        assert len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        self.text(path, value)
        self.fields[-1]["type"] = "Hash"

    def u(self, path, value):
        assert 0 <= value < 2**64
        self.put(path, "U64", value, struct.pack("<Q", value))

    def tag(self, path, value):
        assert 0 <= value <= 255
        self.put(path, "tag", value, bytes([value]))

    def blob(self, path, value):
        self.put(path, "Bytes", value.hex(), struct.pack("<Q", len(value)) + value)

    def i32(self, path, value):
        self.put(path, "I32", value, struct.pack("<i", value))

    def data(self):
        return b"".join(self.parts)

    def finish(self, core=False):
        b = self.data()
        assert len(b) <= A
        RECORDS[self.name] = b
        INPUTS[self.name] = {"classification": LABEL, "fields": self.fields}
        dump(self.name + ".bin", b)
        dump(self.name + ".hex", (b.hex() + "\n").encode())
        MANIFEST["records"][self.name] = {
            "classification": LABEL, "core_vector": core,
            "bytes": len(b), "canonical_sha256": sha(b),
            "raw_file_sha256": sha(b),
            "offsets": [{k: f[k] for k in ("path", "type", "offset", "length")}
                        for f in self.fields]}
        return b


def new(name, suffix):
    return Record(name, "hekit-test-" + suffix + "-v1")


def retained(name, root=2, content=None):
    path = "synthetic/" + name
    if path not in RETAINED:
        b = (LABEL + "\n" + name + "\n").encode() if content is None else content
        RETAINED[path] = (root, b)
        dump("retained/" + path, b)
    else:
        assert RETAINED[path][0] == root
    return path


def rawhash(path):
    return sha(RETAINED[path][1])


def fileref(r, prefix, path):
    root, b = RETAINED[path]
    r.tag(prefix + ".root", root)
    r.text(prefix + ".path", path)
    r.u(prefix + ".length", len(b))
    r.h(prefix + ".sha256", sha(b))


def clock(r, prefix, n):
    r.tag(prefix + ".wall_source", 0)
    r.u(prefix + ".unix_ns", 1000000 + n)
    r.tag(prefix + ".monotonic_source", 0)
    r.u(prefix + ".elapsed_since_operation_start_ns", n)


def link(r, prefix, value):
    r.h(prefix + ".file_sha256", value[0])
    r.h(prefix + ".payload_sha256", value[1])


def identityref(r, prefix, kind, b):
    schema = {"contract": "hekit-study-profile-contract-v1",
              "receipt": "hekit-study-seed-input-receipt-v1"}[kind]
    r.text(prefix + ".schema", schema)
    r.text(prefix + ".encoding", ENC)
    r.u(prefix + ".canonical_length", len(b))
    r.h(prefix + ".sha256", sha(b))


NAMES = ["inputs", "implementation", "layout", "resource", "user"]
CAMPAIGN, PROFILE, ATTEMPT = "SYNTHETIC-CAMPAIGN", "SYNTHETIC-PROFILE", "SYNTHETIC-ATTEMPT"
REV = sha(b"SYNTHETIC source revision")[:40]
SOURCE = retained("source-manifest.txt", 1)
APPROVAL = retained("approval.txt", 3)
TRAIN = retained("train-receipt.txt", 2)
EXE = retained("executable-identity.txt", 2)
REGISTRY = retained("registry.txt", 3)
PARAM = retained("parameter-descriptor.txt", 2)
KEYS = retained("key-descriptor.txt", 2)
SECURITY = retained("security-descriptor.txt", 2)
PHYSICAL = retained("physical-descriptor.txt", 5)
RUNTIME = retained("runtime-descriptor.txt", 5)
MATERIALIZED = retained("materialized.txt", 0)
CONTRACT = RETAINED[retained("contract-canonical.txt", 0)][1]
RECEIPT = RETAINED[retained("receipt-canonical.txt", 0)][1]
INPUT_FILES = [retained("data.tsv", 0), retained("initial.tsv", 0),
               retained("order.tsv", 0)]
LOCK_FILE = retained("full-lock.json", 2)
LOCK_PAYLOAD = retained("full-lock-canonical.txt", 2)
REF_FILE = retained("full-reference.json", 2)
REF_PAYLOAD = retained("full-reference-canonical.txt", 2)
LOCK_LINK = (rawhash(LOCK_FILE), rawhash(LOCK_PAYLOAD))
REF_LINK = (rawhash(REF_FILE), rawhash(REF_PAYLOAD))
LOCK_ID = sha(b"SYNTHETIC lock id")


def registration():
    # Frozen CaseIdentity grammar hand-transcribed from target-design section5.
    r = Record("registration", "hekit-study-registered-case-v1", None)
    r.h("registry_sha256", rawhash(REGISTRY))
    r.tag("origin", 0)
    r.text("registration_id", "SYNTHETIC-REGISTRATION")
    r.text("profile_id", PROFILE)
    r.u("seed", 0)
    for key in ("scope", "role", "backend", "mode", "placement", "packing", "security_policy"):
        r.tag(key, 0)
    r.u("logical_batch", 2)
    r.tag("eval_config.present", 0)
    return r.finish()


REG = registration()


def execution(branch):
    r = Record(branch + ".execution", "hekit-study-case-binding-v1", None)
    r.tag("implementation", 0)
    r.text("implementation_revision", REV)
    r.h("implementation_manifest_sha256", rawhash(SOURCE))
    r.h("parameter_identity_sha256", rawhash(PARAM))
    r.tag("bootstrap_descriptor_sha256.present", 0)
    r.h("layout_plan_sha256", rawhash(retained(branch + ".layout-descriptor.txt")))
    r.h("trainer_plan_sha256", rawhash(retained(branch + ".trainer-descriptor.txt")))
    for k in ("hoist", "bsgs", "lazy_relin"):
        r.tag("eval_config." + k, 0)
    r.tag("bootstrap_engine", 0)
    for key, p in (("key_policy_sha256", KEYS), ("security_policy_sha256", SECURITY),
                   ("physical_binding_sha256", PHYSICAL), ("runtime_policy_sha256", RUNTIME)):
        r.h(key, rawhash(p))
    return r.finish()


EXEC = {b: execution(b) for b in ("full", "quick")}


def request(branch):
    r = new(branch + ".request", "gate-request")
    r.tag("consumption", int(branch == "quick"))
    r.tag("physical_lanes.present", 1)
    r.u("physical_lanes.value", 2)
    r.tag("transpose", 0)
    for k in ("hoist", "bsgs", "lazy_relin"):
        r.tag("eval_config." + k, 0)
    r.u("limits.max_bytes", 65536)
    r.u("limits.max_records", 64)
    for k, v in zip(("geometries", "initial_states", "updates", "handoffs", "wrapper_calls"),
                    (1, 1, 4, 0, 3)):
        r.u("work_limits." + k, v)
    return r.finish()


REQUEST = {b: request(b) for b in ("full", "quick")}


def consumption(branch):
    r = new(branch + ".consumption", "gate-consumption")
    r.tag("consumption", int(branch == "quick"))
    r.u("seed", 0)
    r.h("request_sha256", sha(REQUEST[branch]))
    r.h("trainer_sha256", rawhash("synthetic/" + branch + ".trainer-descriptor.txt"))
    r.h("selected_receipt_sha256", sha(RECEIPT))
    if branch == "full":
        r.u("total_updates", 4)
    else:
        r.u("updates.count", 2)
        for i, indices in enumerate(((2, 0), (3, 1))):
            p = "updates[%d]" % i
            r.u(p + ".index", i)
            r.u(p + ".epoch", 0)
            r.u(p + ".batch", i)
            r.u(p + ".sample_indices.count", len(indices))
            for j, index in enumerate(indices):
                r.u(p + ".sample_indices[%d]" % j, index)
            r.u(p + ".parts.count", 1)
            r.u(p + ".parts[0].start", 0)
            r.u(p + ".parts[0].valid", 2)
    return r.finish()


CONSUMED = {b: consumption(b) for b in ("full", "quick")}


def quick_identity():
    r = new("quick.identity", "quick-identity")
    r.blob("registration", REG)
    r.h("registration_id", sha(REG))
    r.blob("execution", EXEC["quick"])
    r.h("execution_sha256", sha(EXEC["quick"]))
    r.h("contract_sha256", sha(CONTRACT))
    r.h("selected_receipt_sha256", sha(RECEIPT))
    r.h("materialized_sha256", rawhash(MATERIALIZED))
    r.text("campaign_id", CAMPAIGN)
    r.text("profile_id", PROFILE)
    r.u("seed", 0)
    r.tag("consumption", 1)
    r.blob("consumed_plan", CONSUMED["quick"])
    r.h("consumed_plan_sha256", sha(CONSUMED["quick"]))
    r.blob("request", REQUEST["quick"])
    r.h("request_sha256", sha(REQUEST["quick"]))
    r.h("trainer_sha256", rawhash("synthetic/quick.trainer-descriptor.txt"))
    r.text("operational_attempt", ATTEMPT)
    return r.finish(core=True)


QID = quick_identity()


def applicability(r, p, branch):
    r.h(p + ".implementation_manifest", rawhash(SOURCE))
    r.text(p + ".implementation_revision", REV)
    r.h(p + ".parameters", rawhash(PARAM))
    r.tag(p + ".bootstrap.present", 0)
    for k, v in (("keys", KEYS), ("security", SECURITY),
                 ("layout", "synthetic/" + branch + ".layout-descriptor.txt"),
                 ("trainer", "synthetic/" + branch + ".trainer-descriptor.txt"),
                 ("runtime", RUNTIME)):
        r.h(p + "." + k, rawhash(v))
    for k in ("implementation", "backend", "mode"):
        r.tag(p + "." + k, 0)
    r.tag(p + ".consumption", int(branch == "quick"))


def scope(r, branch):
    r.tag("scope", int(branch == "quick"))
    r.blob("common.execution", EXEC[branch])
    r.h("common.contract_sha256", sha(CONTRACT))
    r.h("common.selected_receipt_sha256", sha(RECEIPT))
    r.h("common.materialized_sha256", rawhash(MATERIALIZED))
    r.text("common.campaign_id", CAMPAIGN)
    r.text("common.profile_id", PROFILE)
    for k, f in (("approval_sha256", APPROVAL), ("train_receipt_sha256", TRAIN),
                 ("executable_sha256", EXE), ("source_manifest_sha256", SOURCE)):
        r.h("common." + k, rawhash(f))
    r.text("common.source_revision", REV)
    if branch == "full":
        r.blob("full.case", REG)
        r.h("full.case_id", sha(REG))
        link(r, "full.lock", LOCK_LINK)
        link(r, "full.reference", REF_LINK)
        r.h("full.lock_id", LOCK_ID)
        r.u("full.threshold_bits", 0)
        r.text("full.attempt_id", ATTEMPT)
        r.tag("full.predecessor.present", 0)
    else:
        r.blob("quick.registration", REG)
        r.h("quick.registration_id", sha(REG))
        r.blob("quick.operational_identity", QID)
        r.h("quick.operational_id", sha(QID))
        r.u("quick.seed", 0)
        r.tag("quick.consumption", 1)
        r.h("quick.consumed_plan_sha256", sha(CONSUMED[branch]))
        r.h("quick.request_sha256", sha(REQUEST[branch]))
        r.text("quick.operational_attempt", ATTEMPT)


def tail_inputs(r, branch):
    r.u("inputs.source_files.count", len(INPUT_FILES))
    for i, p in enumerate(INPUT_FILES):
        fileref(r, "inputs.source_files[%d]" % i, p)
    identityref(r, "inputs.contract", "contract", CONTRACT)
    identityref(r, "inputs.selected_receipt", "receipt", RECEIPT)
    r.h("inputs.materialized_sha256", rawhash(MATERIALIZED))
    r.blob("inputs.consumed_plan", CONSUMED[branch])
    r.h("inputs.consumed_plan_sha256", sha(CONSUMED[branch]))
    if branch == "full":
        identityref(r, "inputs.independent_target_receipt", "receipt", RECEIPT)
        r.h("inputs.independent_materialized_sha256", rawhash(MATERIALIZED))


def tail_implementation(r, branch):
    applicability(r, "implementation.applicability", branch)
    r.u("implementation.correctness.count", 1)
    p = "implementation.correctness[0]"
    r.tag(p + ".kind", 0)
    for key, name, root in (("report", branch + ".kat-report.txt", 2),
                            ("command", "kat-command.txt", 2),
                            ("source_manifest", "source-manifest.txt", 1)):
        fileref(r, p + "." + key, retained(name, root))
    r.text(p + ".source_revision", REV)
    fileref(r, p + ".criterion_policy", retained("kat-policy.txt", 3))
    applicability(r, p + ".applicability", branch)
    p = "implementation.security"
    fileref(r, p + ".policy", retained("security-policy.txt", 3))
    fileref(r, p + ".parameter_export", retained("security-parameters.txt"))
    r.text(p + ".estimator_revision", sha(b"SYNTHETIC estimator revision")[:40])
    r.u(p + ".estimator_sources.count", 1)
    fileref(r, p + ".estimator_sources[0]", retained("estimator-source.txt"))
    for key in ("environment", "command", "model_config"):
        fileref(r, p + "." + key, retained("security-" + key + ".txt"))
    r.u(p + ".instances.count", 1)
    r.tag(p + ".instances[0].role", 0)
    for key in ("problem", "bdgl16_raw", "matzov_raw"):
        fileref(r, p + ".instances[0]." + key, retained("security-main-" + key + ".txt"))
    r.u("implementation.construction_obligations.count", 4)
    for i in range(4):
        r.tag("implementation.construction_obligations[%d]" % i, i)


def tail_layout(r, branch):
    r.h("layout.parameters", rawhash(PARAM))
    r.tag("layout.bootstrap.present", 0)
    for k, path in (("layout", "synthetic/" + branch + ".layout-descriptor.txt"),
                    ("trainer", "synthetic/" + branch + ".trainer-descriptor.txt"), ("keys", KEYS)):
        r.h("layout." + k, rawhash(path))
    r.blob("layout.consumed_plan", CONSUMED[branch])
    r.h("layout.consumed_plan_sha256", sha(CONSUMED[branch]))
    r.blob("layout.request", REQUEST[branch])
    r.h("layout.request_sha256", sha(REQUEST[branch]))


def tail_resource(r, branch):
    applicability(r, "resource.applicability", branch)
    r.h("resource.physical", rawhash(PHYSICAL))
    fileref(r, "resource.model", retained(branch + ".resource-model.txt"))
    fileref(r, "resource.model_validation", retained(branch + ".resource-validation.txt"))
    for key, val in zip(("max_bytes", "max_records", "plan_bytes", "retained_bytes",
                         "live_temporary_bytes", "peak_temporary_bytes"), (65536, 64, 800, 200, 0, 400)):
        r.u("resource.requested." + key, val)
    r.u("resource.requirements.components.count", 22)
    # Pure fictional arithmetic; inapplicable NI/block/NAG terms zero.
    peak = 0
    incremental = 0
    for i in range(22):
        mask, n = (0, 0) if i in (4, 9, 11, 12) else (15, 100 + i)
        p = "resource.requirements.components[%d]" % i
        r.tag(p + ".kind", i)
        r.tag(p + ".phase_mask", mask)
        resident = min(n, 20) if i in (14, 16, 18) else 0
        r.u(p + ".resident_bytes", resident)
        r.u(p + ".additional_bytes", n - resident)
        peak += n
        incremental += n - resident
    r.u("resource.requirements.peak_host_bytes", peak)
    r.u("resource.requirements.incremental_host_bytes", incremental)
    r.u("resource.requirements.required_disk_bytes", 4096)
    r.h("resource.observation.physical", rawhash(PHYSICAL))
    clock(r, "resource.observation.clock", 1)
    r.u("resource.observation.host_mem_available", 100000)
    r.u("resource.observation.cgroups.count", 1)
    p = "resource.observation.cgroups[0]"
    r.tag(p + ".version", 1)
    r.text(p + ".path", "/SYNTHETIC-cgroup")
    r.tag(p + ".limit.present", 1)
    r.u(p + ".limit.value", 50000)
    r.u(p + ".current", 1000)
    for key in ("limit_raw", "current_raw"):
        fileref(r, p + "." + key, retained("cgroup-" + key + ".txt", 5))
    r.u("resource.observation.disk_available", 100000)
    for key in ("host_raw", "disk_raw"):
        fileref(r, "resource.observation." + key, retained(key + ".txt", 5))
    r.tag("resource.claim.mechanism", 0)
    r.text("resource.claim.claim_id", "SYNTHETIC-CLAIM")
    fileref(r, "resource.claim.claim_record", retained("claim-record.txt", 5))
    r.tag("resource.claim.overlap", 0)
    r.u("resource.claim.other_claimed_host_bytes", 0)
    r.u("resource.claim.other_claimed_disk_bytes", 0)


def authority(branch):
    r = new(branch + ".authority.scope", "user-scope")
    r.tag("origin", 0)
    r.text("conversation_id", "SYNTHETIC-CONVERSATION")
    r.text("message_id", "SYNTHETIC-MESSAGE")
    r.blob("instruction_bytes", b"SYNTHETIC TEST ORACLE: not an actual user instruction or approval")
    r.u("scope_generation", 1)
    r.u("scopes.count", 1)
    r.tag("scopes[0].branch", int(branch == "quick"))
    r.text("scopes[0].campaign", CAMPAIGN)
    r.text("scopes[0].profile", PROFILE)
    r.h("scopes[0].registration_id", sha(REG))
    r.text("scopes[0].attempt", ATTEMPT)
    r.tag("scopes[0].retry", 0)
    r.tag("scopes[0].overlap", 0)
    r.u("holds.count", 1)
    r.tag("holds[0].branch", 0)
    r.h("holds[0].registration_id", sha(b"SYNTHETIC OTHER HELD BLOCK REGISTRATION"))
    r.tag("holds[0].reason", 0)
    scopebytes = r.finish()
    r = new(branch + ".authority.current", "user-current")
    r.text("conversation_id", "SYNTHETIC-CONVERSATION")
    r.text("source_head_message_id", "SYNTHETIC-MESSAGE")
    r.u("scope_generation", 1)
    r.h("authority_sha256", sha(scopebytes))
    r.u("revoked_scopes.count", 0)
    r.tag("revoked_all", 0)
    clock(r, "observed", 1)
    currentbytes = r.finish()
    retained(branch + "/authority.scope.bin", 4, scopebytes)
    retained(branch + "/authority.current.bin", 4, currentbytes)
    return scopebytes, currentbytes


AUTH = {b: authority(b) for b in ("full", "quick")}


def tail_user(r, branch):
    for key, b in zip(("authority", "current"), AUTH[branch]):
        r.blob("user." + key + "_bytes", b)
        r.h("user." + key + "_sha256", sha(b))
    fileref(r, "user.authority_file", "synthetic/" + branch + "/authority.scope.bin")
    fileref(r, "user.current_file", "synthetic/" + branch + "/authority.current.bin")
    r.u("user.selected_scope_index", 0)
    clock(r, "user.observed", 1)


TAILS = [tail_inputs, tail_implementation, tail_layout, tail_resource, tail_user]
for branch in ("full", "quick"):
    for name, tail in zip(NAMES, TAILS):
        r = new(branch + "." + name + ".evidence", "gate-" + name + "-evidence")
        r.text("policy_revision", "test-d7-" + name + "-v1")
        scope(r, branch)
        tail(r, branch)
        r.finish(core=True)


def gates(r):
    for name in NAMES:
        r.text("gates." + name + ".policy_revision", "test-d7-" + name + "-v1")
        r.h("gates." + name + ".evidence_sha256", sha(RECORDS["quick." + name + ".evidence"]))


EVENT_LINKS = {}


def event(kind):
    r = new("quick." + kind, "quick-" + kind)
    if kind == "preflight":
        r.blob("identity", QID)
    r.h("identity_sha256", sha(QID))
    gates(r)
    if kind == "preflight":
        r.u("argv.count", 2)
        r.text("argv[0]", "SYNTHETIC-study-train")
        r.text("argv[1]", "--quick")
        r.h("source_manifest_sha256", rawhash(SOURCE))
        r.text("source_revision", REV)
        for key, p in (("approval_sha256", APPROVAL), ("train_receipt_sha256", TRAIN), ("executable_sha256", EXE)):
            r.h(key, rawhash(p))
        clock(r, "observed", 2)
    else:
        link(r, "preflight", EVENT_LINKS["preflight"])
        if kind == "end":
            link(r, "start", EVENT_LINKS["start"])
            r.tag("outcome", 0)
            r.u("completed_updates", 2)
            r.tag("failure.present", 0)
        clock(r, "started" if kind == "start" else "ended", 3 if kind == "start" else 4)
    canonical = r.finish(core=True)
    assert len(canonical) <= K
    doc = {"schema": "hekit-test-quick-" + kind + "-v1", "kind": "quick_" + kind,
           "synthetic": True, "canonical_bytes": canonical.hex(),
           "integrity": {"encoding": ENC, "hash": "sha256", "payload_sha256": sha(canonical)}}
    raw = (json.dumps(doc, separators=(",", ":")) + "\n").encode()
    assert len(raw) <= A
    dump("quick." + kind + ".json", raw)
    EVENT_LINKS[kind] = (sha(raw), sha(canonical))
    MANIFEST["records"]["quick." + kind]["json"] = {
        "raw_file_bytes": len(raw), "raw_file_sha256": sha(raw),
        "payload_sha256": sha(canonical), "envelope_bytes_excluding_hex": len(raw) - 2 * len(canonical)}


for kind in ("preflight", "start", "end"):
    event(kind)


def mutation(name, base, field, replacement, reason):
    f = next(x for x in INPUTS[base]["fields"] if x["path"] == field)
    raw = RECORDS[base]
    if f["type"] == "tag":
        patch = bytes([replacement])
    elif f["type"] == "U64":
        patch = struct.pack("<Q", replacement)
    elif f["type"] == "Hash":
        patch = struct.pack("<Q", 64) + replacement.encode()
    else:
        raise AssertionError("Only explicit fixed-size mutations here")
    assert len(patch) == f["length"]
    out = raw[:f["offset"]] + patch + raw[f["offset"] + f["length"]:]
    dump("negative/" + name + ".bin", out)
    MANIFEST["mutations"].append({"name": name, "base": base + ".bin", "field": field,
        "offset": f["offset"], "length": f["length"], "replacement": replacement,
        "file": "negative/" + name + ".bin", "expected": reason,
        "sha256": sha(out), "bytes": len(out), "validation_executed": False})


mutation("branch-tag", "full.inputs.evidence", "scope", 1, "Quick branch body mismatch; no scope promotion")
mutation("unknown-tag", "quick.inputs.evidence", "scope", 2, "Unknown closed scope tag")
mutation("synthetic-bit", "quick.inputs.evidence", "header.synthetic", 0, "Test-domain/synthetic mismatch; production always rejects test domain")
mutation("identity", "quick.inputs.evidence", "quick.operational_id", "0" * 64, "Operational identity hash mismatch")
mutation("input-receipt", "full.inputs.evidence", "inputs.independent_target_receipt.sha256", "0" * 64, "Independent target receipt mismatch")
mutation("target-link", "full.inputs.evidence", "full.lock.file_sha256", "0" * 64, "Verified target link mismatch; zero is not authority")
mutation("threshold-placeholder", "full.inputs.evidence", "full.threshold_bits", 1, "Threshold differs from independently verified target")
mutation("quick-seed", "quick.consumption", "seed", 1, "Quick seed must be zero")
mutation("quick-one-update", "quick.consumption", "updates.count", 1, "Quick count must be exactly two; trailing update remains")
mutation("quick-three-updates", "quick.consumption", "updates.count", 3, "Quick count must be exactly two; missing third update")
mutation("order", "quick.consumption", "updates[0].sample_indices[0]", 3, "Canonical first-update sample order mismatch")
mutation("partition", "quick.consumption", "updates[0].parts[0].start", 1, "Partition gap; first start must be zero")
mutation("request", "quick.request", "limits.max_bytes", 65537, "Original request mismatch even if enclosing hashes recomputed")
mutation("full-consumption", "quick.consumption", "consumption", 0, "Full/Quick consumed-plan branch mismatch")
mutation("event-evidence", "quick.preflight", "gates.layout.evidence_sha256", "0" * 64, "Retained sidecar bytes/hash mismatch")
mutation("event-link", "quick.start", "preflight.file_sha256", "0" * 64, "Exact preflight whole-file link mismatch")
mutation("end-link", "quick.end", "start.payload_sha256", "0" * 64, "Exact start canonical link mismatch")
mutation("end-incomplete", "quick.end", "completed_updates", 1, "Complete outcome requires exactly two completed updates")
mutation("resource-peak", "quick.resource.evidence", "resource.requirements.peak_host_bytes", 1, "Requirement phase sum mismatch; no fabricated capacity pass")
mutation("authority-revoked", "quick.authority.current", "revoked_all", 1, "Current authority revoked; hashes cannot override")
mutation("authority-generation", "quick.authority.current", "scope_generation", 2, "Authority/current source generation mismatch")

# Structural mutation specs avoid generating large boundary fixtures.
specs = [
 {"name": "placeholder-tail", "base": "quick.inputs.evidence.bin", "operation": "append 00", "expected": "Trailing Full target placeholder forbidden"},
 {"name": "different-operational-attempt", "base": "quick.identity.bin", "operation": "Replace SYNTHETIC-ATTEMPT by SYNTHETIC-ATTEMPX in operational_attempt; recompute local identity hash", "expected": "Scope/authority/held attempt mismatch"},
 {"name": "gate-policy", "base": "quick.preflight.bin", "operation": "Change final v1 to v2 in gates.inputs.policy_revision", "expected": "Unknown exact compiled policy revision; no fallback"},
 {"name": "raw-json-whitespace-cap", "base": "quick.start.json", "operation": "Use bounded-reader advertised raw length A+1; no allocation", "expected": "RawFileCap"},
 {"name": "binary-cap", "base": "quick.inputs.evidence.bin", "operation": "Bounded reader length A+1", "expected": "EvidenceFileCap before allocation"},
 {"name": "quick-canonical-cap", "base": "quick.start.json", "operation": "Canonical hex decoded length K+1 in length-only harness", "expected": "CanonicalCap before allocation"},
 {"name": "u64-length-overflow", "base": "quick.inputs.evidence.bin", "operation": "Replace execution Bytes length by 18446744073709551615", "expected": "Checked length/cap refusal before allocation"},
 {"name": "unknown-source-record", "base": "quick.inputs.evidence.bin", "operation": "Change source_files[0].path to same-size unregistered relative path", "expected": "Expected canonical source inventory mismatch; never open artifact path"},
 {"name": "missing-verified-target", "base": "full.inputs.evidence.bin", "operation": "Retain exact bytes but omit private verified-target owner in harness", "expected": "Missing target; bytes alone cannot form Full scope"},
 {"name": "zero-threshold-without-origin", "base": "full.inputs.evidence.bin", "operation": "Retain existing numeric zero threshold but supply no independently verified target", "expected": "No origin; numeric equality not authority"},
 {"name": "mixed-gates", "base": "quick.preflight.bin", "operation": "For each of five gate slots substitute corresponding full evidence digest and bytes", "expected": "Five separate scope mismatch refusals"},
 {"name": "stale-current-source", "base": "quick.user.evidence.bin", "operation": "Keep all hashes and old bytes, origin source head revokes chosen scope", "expected": "Live source-currentness refusal; old record hash insufficient"},
 {"name": "production-parser", "base": "all core vectors", "operation": "Offer each synthetic vector to production parser", "expected": "All fourteen refuse test domain/synthetic marker"},
 {"name": "missing-resource-component", "base": "quick.resource.evidence.bin", "operation": "Remove component21 and decrement list count to21", "expected": "Incomplete required fixed22 component inventory"},
 {"name": "json-duplicate-key", "base": "quick.start.json", "operation": "Insert duplicate decoded schema key using escaped alias", "expected": "Duplicate decoded key"},
 {"name": "escaped-hex", "base": "quick.start.json", "operation": "Escape one canonical_bytes character as unicode", "expected": "Raw lowercase unescaped hex required"},
]
js("mutation-specs.json", {"classification": LABEL, "concrete": MANIFEST["mutations"], "specifications": specs})

# Verify old Full goldens strictly read-only against an existing frozen manifest.
frozen_path = HERE.parent / "task-54-d7-admission-codec-frozen-goldens.txt"
old = []
for line in frozen_path.read_text().splitlines():
    digest, filename = line.split(None, 1)
    filename = filename.strip()
    b = (ROOT / filename).read_bytes()
    assert sha(b) == digest, filename
    old.append({"path": filename, "bytes": len(b), "sha256": digest, "unchanged": True})
js("full-frozen-preservation.json", {"classification": LABEL, "manifest_source": str(frozen_path.relative_to(ROOT)),
    "source_sha256": sha(frozen_path.read_bytes()), "file_count": len(old), "files": old})

boundary = {"A": A, "K": K, "binary_A_allowed_length": A,
            "binary_A_plus_1_rejected_length": A + 1,
            "quick_K_allowed_length": K, "quick_K_plus_1_rejected_length": K + 1,
            "large_allocations": 0, "quick_envelope": {}}
for kind in ("preflight", "start", "end"):
    e = MANIFEST["records"]["quick." + kind]["json"]["envelope_bytes_excluding_hex"]
    # Production domain has one additional character versus test; false vs true +1.
    eprod = e + 2
    assert 2 * K + eprod <= A
    boundary["quick_envelope"][kind] = {"test_overhead": e, "production_overhead": eprod,
       "production_raw_at_K": 2 * K + eprod, "remaining_raw_whitespace_budget": A - 2 * K - eprod}
assert 5 * A + 70492488 + A == 83075400
assert 8 * A == 16777216
js("boundary-arithmetic.json", boundary)
js("literal-inputs.json", {"classification": LABEL, "records": INPUTS})
core = {k: {x: v[x] for x in ("bytes", "canonical_sha256")} for k, v in MANIFEST["records"].items() if v["core_vector"]}
assert len(core) == 14
report = {"classification": LABEL, "command": "python3 " + str(Path(__file__).relative_to(ROOT)),
          "expected_exit": 0, "core_vectors": core,
          "counts": {"binary_evidence": 10, "quick_identity": 1, "quick_events": 3,
                     "concrete_mutations": len(MANIFEST["mutations"]), "mutation_specs": len(specs),
                     "old_full_files_unchanged": len(old), "all_literal_records": len(RECORDS)},
          "source_free_calls": {"python_stdlib_generator_invocations": 1, "project_encoder_imports": 0,
               "Rust": 0, "Cargo": 0, "HE": 0, "native_observer": 0, "SSH": 0,
               "parameters": 0, "P1": 0, "materializer": 0, "registry": 0, "live_gate_constructors": 0,
               "production_positive_records": 0, "scientific_outputs": 0},
          "validation": "assertions check byte lengths, scalar ranges, arithmetic, link construction, old file hashes; semantic mutation refusals are future tests, not executed here"}
js("generation-report.json", report)
js("manifest.json", MANIFEST)
entries = []
for p in sorted(HERE.rglob("*")):
    if p.is_file() and p.name not in ("SHA256SUMS",):
        entries.append(sha(p.read_bytes()) + "  " + str(p.relative_to(HERE)))
(HERE / "SHA256SUMS").write_text("\n".join(entries) + "\n")
print(json.dumps({"core_vectors": len(core), "literal_records": len(RECORDS),
                  "mutations": len(MANIFEST["mutations"]), "old_full_unchanged": len(old),
                  "manifest_sha256": sha((HERE / "manifest.json").read_bytes())}, sort_keys=True))
