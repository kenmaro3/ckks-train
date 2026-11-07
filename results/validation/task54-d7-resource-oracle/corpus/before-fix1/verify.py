"""Independent bounded inspection of literal artifacts; no generator imports.

This is a private structural oracle, never a production/native evidence reader.
"""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
E = json.loads((HERE / "literal-expectations.json").read_text())
CASES = json.loads((HERE / "negative-cases.json").read_text())
MAX = 18446744073709551615
HEX = re.compile(r"[0-9a-f]{64}\Z")
IDENT = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")

class Refusal(Exception):
    pass

def need(condition, reason):
    if not condition:
        raise Refusal(reason)

def sha(b):
    return hashlib.sha256(b).hexdigest()

def integer(n, low=0, high=MAX):
    need(type(n) is int and low <= n <= high, "integer")

def tag(n, values):
    integer(n)
    need(n in values, "tag")

def keys(o, names):
    need(type(o) is dict and set(o) == set(names.split()), "keys")

def array(a, limit=65536):
    need(type(a) is list and len(a) <= limit, "list")

def text(s, identifier=False):
    need(type(s) is str, "text")
    try:
        b = s.encode("utf8")
    except UnicodeEncodeError:
        raise Refusal("unicode")
    need(bool(IDENT.fullmatch(s)) if identifier else len(b) <= 4096, "text")

def hash_value(s):
    need(type(s) is str and bool(HEX.fullmatch(s)), "hash")

def uuid(s):
    need(type(s) is str and bool(UUID.fullmatch(s)), "uuid")

def ordered_unique(a, reason):
    need(a == sorted(a) and len(set(a)) == len(a), reason)

def add(a, b):
    integer(a); integer(b)
    need(a <= MAX - b, "arithmetic-overflow")
    return a + b

def multiply(a, b):
    integer(a); integer(b)
    need(b == 0 or a <= MAX // b, "arithmetic-overflow")
    return a * b

def token_int(token):
    need(bool(re.fullmatch(r"0|-?[1-9][0-9]*", token)), "integer")
    return int(token)

def token_float(token):
    raise Refusal("integer")

def object_pairs(pairs):
    d = {}
    for k, value in pairs:
        need(k not in d, "duplicate-key")
        d[k] = value
    return d

STRING = re.compile(r'"(?:[^"\\]|\\.)*"')

def source_json(b):
    need(len(b) <= 8388608, "raw-cap")
    try:
        s = b.decode("utf8")
    except UnicodeDecodeError:
        raise Refusal("utf8")
    need(not s.startswith("\ufeff"), "json")
    try:
        obj = json.loads(s, object_pairs_hook=object_pairs, parse_int=token_int,
                         parse_float=token_float, parse_constant=token_float)
    except (ValueError, json.JSONDecodeError):
        raise Refusal("json")
    def unicode_walk(v):
        if isinstance(v, str):
            try: v.encode("utf8")
            except UnicodeEncodeError: raise Refusal("unicode")
        elif isinstance(v, list):
            for x in v: unicode_walk(x)
        elif isinstance(v, dict):
            for k, x in v.items(): unicode_walk(k); unicode_walk(x)
    unicode_walk(obj)
    # Token-level Bytes check includes escaped key names. It does not normalize
    # strings then claim that the original transport was unescaped.
    tokens = list(STRING.finditer(s))
    for idx, m in enumerate(tokens):
        key = json.loads(m.group())
        tail = s[m.end():].lstrip()
        if not tail.startswith(":"):
            continue
        if key in ("registration", "request"):
            need(idx + 1 < len(tokens) and bool(re.fullmatch(r'"(?:[0-9a-f]{2})*"', tokens[idx + 1].group())), "bytes-spelling")
        elif key == "argv":
            start = s.index("[", m.end())
            stop = s.index("]", start)
            for arg in STRING.finditer(s, start, stop):
                need(bool(re.fullmatch(r'"(?:[0-9a-f]{2})*"', arg.group())), "bytes-spelling")
    return obj

RAW = {name: value.encode() for name, value in E["raw_literals"].items()}
OLD = HERE.parent / "task-54-d7-gate-schema-vectors"
NESTED = {name: (OLD / name).read_bytes() for name in E["nested_input_pins"]}
for name, b in NESTED.items():
    need([len(b), sha(b)] == E["nested_input_pins"][name], "nested-literal-pin")
for name, b in RAW.items():
    need((HERE / "raw" / name).read_bytes() == b, "raw-literal")

def raw_ref(name, root=2):
    return {"root": root, "path": "fictional/" + name, "length": len(RAW[name]), "sha256": sha(RAW[name])}

def fr(o, folder):
    keys(o, "root path length sha256")
    tag(o["root"], range(6)); text(o["path"]); integer(o["length"]); hash_value(o["sha256"])
    p = o["path"]
    need(bool(p) and not p.startswith("/") and "\\" not in p and "\0" not in p and all(x not in ("", ".", "..") for x in p.split("/")), "fileref-path")
    if p.startswith("fictional/") and p[10:] in RAW:
        b = RAW[p[10:]]
    elif p in ("resource/linux-cpu-v1/model.json", "resource/linux-cpu-v1/validation.json"):
        need(o["root"] == 2, "file-link")
        b = (HERE / folder / p.rsplit("/", 1)[1]).read_bytes()
    else:
        raise Refusal("unexpected-fileref")
    need(o["length"] == len(b) and o["sha256"] == sha(b), "file-link")

def clock(o):
    keys(o, "wall_source unix_ns monotonic_source elapsed_since_operation_start_ns")
    tag(o["wall_source"], [0]); tag(o["monotonic_source"], [0])
    integer(o["unix_ns"]); integer(o["elapsed_since_operation_start_ns"])

def clocks(o):
    clock(o["started"]); clock(o["finished"])
    need(o["started"]["elapsed_since_operation_start_ns"] <= o["finished"]["elapsed_since_operation_start_ns"], "clock-order")

def identity(o):
    keys(o, "device mount_id type")
    for x in o.values(): integer(x)

def root_roles(a):
    array(a)
    need(bool(a), "root-order")
    for x in a: tag(x, range(6))
    ordered_unique(a, "root-order")

def namespace(o, folder):
    keys(o, "boot_id mount_ns cgroup_ns pid_ns user_ns installation_record")
    uuid(o["boot_id"])
    for field in ["mount_ns", "cgroup_ns", "pid_ns", "user_ns"]:
        keys(o[field], "device inode")
        integer(o[field]["device"]); integer(o[field]["inode"])
    fr(o["installation_record"], folder)

def allocator(o, folder):
    keys(o, "implementation library_files configuration derivation")
    tag(o["implementation"], [0]); array(o["library_files"], 512)
    seen = []
    for item in o["library_files"]:
        fr(item, folder); seen.append((item["root"], item["path"]))
    need(len(set(seen)) == len(seen), "library-duplicate")
    fr(o["configuration"], folder); fr(o["derivation"], folder)

def kernel(o, folder):
    keys(o, "release version page_size source configuration")
    for field in ["release", "version"]:
        text(o[field]); need(bool(o[field]), "text")
    integer(o["page_size"]); need(o["page_size"] > 0, "page-size")
    fr(o["source"], folder); fr(o["configuration"], folder)

def run_ids(a):
    array(a)
    for x in a: text(x, True)
    ordered_unique(a, "run-id-order")

def environment(o, folder):
    keys(o, "target toolchain cargo_lock build_receipt allocator kernel workers bootstrap")
    text(o["target"], True)
    for field in ["toolchain", "cargo_lock", "build_receipt"]: fr(o[field], folder)
    allocator(o["allocator"], folder); kernel(o["kernel"], folder); integer(o["workers"])
    keys(o["bootstrap"], "source_derivation validation_runs")
    fr(o["bootstrap"]["source_derivation"], folder); run_ids(o["bootstrap"]["validation_runs"])

def source_identity(o):
    hash_value(o["source_manifest"])
    need(type(o["source_revision"]) is str and bool(re.fullmatch("[0-9a-f]{40}", o["source_revision"])), "git40")
    need(o["source_manifest"] == sha(RAW["manifest.txt"]) and o["source_revision"] == E["source_revision"], "source-identity")

def component_order(items):
    array(items)
    need(len(items) == 22 and [x.get("kind") if type(x) is dict else None for x in items] == list(range(22)), "component-order")

def requirements(o):
    keys(o, "components peak_host_bytes incremental_host_bytes required_disk_bytes")
    component_order(o["components"])
    sums, extra = [0] * 4, [0] * 4
    for c in o["components"]:
        keys(c, "kind phase_mask resident_bytes additional_bytes")
        tag(c["kind"], range(22)); tag(c["phase_mask"], range(16))
        integer(c["resident_bytes"]); integer(c["additional_bytes"])
        need(c["phase_mask"] != 0 or c["resident_bytes"] == c["additional_bytes"] == 0, "inapplicable-bytes")
        total = add(c["resident_bytes"], c["additional_bytes"])
        for j, bit in enumerate([1, 2, 4, 8]):
            if c["phase_mask"] & bit:
                sums[j] = add(sums[j], total); extra[j] = add(extra[j], c["additional_bytes"])
    for field in ["peak_host_bytes", "incremental_host_bytes", "required_disk_bytes"]: integer(o[field])
    need(o["peak_host_bytes"] == max(sums), "peak-identity")
    need(o["incremental_host_bytes"] == max(extra), "incremental-identity")
    return sums, extra

def supported(o):
    array(o)
    need(bool(o), "work-order")
    sort_keys, selector_keys = [], []
    for w in o:
        need(type(w) is dict and "kind" in w, "keys"); tag(w["kind"], [0, 1, 2])
        if w["kind"] == 0:
            keys(w, "kind registration_id consumption"); hash_value(w["registration_id"]); tag(w["consumption"], [0, 1])
            key = (0, w["registration_id"], w["consumption"])
            selector_keys.append(key); sort_keys.append(key)
        else:
            keys(w, "kind operation profile_id seeds"); text(w["profile_id"], True)
            tag(w["operation"], [0, 1, 2] if w["kind"] == 1 else [0, 1]); array(w["seeds"])
            for s in w["seeds"]: integer(s)
            need(bool(w["seeds"]), "work-seeds"); ordered_unique(w["seeds"], "work-seeds")
            if w["kind"] == 1 or w["operation"] == 1: need(w["seeds"] == [0], "work-seeds")
            selector_keys.append((w["kind"], w["operation"], w["profile_id"]))
            sort_keys.append((w["kind"], w["operation"], w["profile_id"], tuple(w["seeds"])))
    need(len(set(selector_keys)) == len(selector_keys), "work-duplicate")
    need(sort_keys == sorted(sort_keys), "work-order")

def metadata(o, folder):
    keys(o, "source_manifest source_revision executable environment argv exit_code timed_out started finished measurements")
    source_identity(o); hash_value(o["executable"]); environment(o["environment"], folder)
    array(o["argv"], 256); need(bool(o["argv"]), "argv")
    for x in o["argv"]:
        need(type(x) is str and bool(re.fullmatch("(?:[0-9a-f]{2})*", x)), "bytes-spelling")
        b = bytes.fromhex(x); need(len(b) <= 4096 and b"\0" not in b, "argv")
    if o["exit_code"] is not None: integer(o["exit_code"], -2147483648, 2147483647)
    need(type(o["timed_out"]) is bool, "bool"); clocks(o); array(o["measurements"])
    for m in o["measurements"]:
        keys(m, "kind scope bytes raw"); tag(m["kind"], [0, 1, 2]); tag(m["scope"], [0, 1, 2]); integer(m["bytes"]); fr(m["raw"], folder)

def binary_text(b, pos, prefix_reason):
    need(pos + 8 <= len(b), prefix_reason)
    count = int.from_bytes(b[pos:pos + 8], "little")
    need(count <= 4096 and pos + 8 + count <= len(b), prefix_reason)
    try: s = b[pos + 8:pos + 8 + count].decode("ascii")
    except UnicodeDecodeError: raise Refusal(prefix_reason)
    return s, pos + 8 + count

def request_bytes(s, folder):
    need(type(s) is str and bool(re.fullmatch("(?:[0-9a-f]{2})*", s)), "bytes-spelling")
    b = bytes.fromhex(s)
    head, pos = binary_text(b, 0, "request-domain")
    domain, pos = binary_text(b, pos, "request-domain")
    test = folder == "synthetic"
    need(head == "hekit-study-reference-encoding-v1" and domain == ("hekit-test-" if test else "hekit-study-") + "gate-request-v1" and b[pos:pos + 1] == bytes([int(test)]), "request-domain")
    body = b[pos + 1:]
    need(bool(body) and body[0] in [0, 1], "request-body")
    old = NESTED["full.request.bin" if body[0] == 0 else "quick.request.bin"]
    _, p = binary_text(old, 0, "request-body"); _, p = binary_text(old, p, "request-body")
    need(body == old[p + 1:], "request-body")
    return body[0]

def claim_work(w, folder, model):
    need(type(w) is dict and "kind" in w, "keys"); tag(w["kind"], [0, 1, 2])
    if w["kind"] == 0:
        keys(w, "kind registration request")
        need(type(w["registration"]) is str and bool(re.fullmatch("(?:[0-9a-f]{2})*", w["registration"])), "bytes-spelling")
        reg = bytes.fromhex(w["registration"])
        need(reg == NESTED["registration.bin"], "registration-literal")
        consumption = request_bytes(w["request"], folder)
        need(any(x == {"kind": 0, "registration_id": sha(reg), "consumption": consumption} for x in model["supported_work"]), "work-membership")
        return
    if w["kind"] == 1:
        keys(w, "kind operation reference_id expected_file profile_id seed document")
        tag(w["operation"], [0, 1, 2]); text(w["reference_id"], True); text(w["profile_id"], True); integer(w["seed"]); fr(w["document"], folder)
        if w["expected_file"] is not None: hash_value(w["expected_file"])
        need((w["expected_file"] is not None) == (w["operation"] == 1), "reference-option")
        if w["operation"] == 1: need(w["expected_file"] == w["document"]["sha256"], "reference-hash")
        need(w["seed"] == 0, "reference-seed")
        seed = w["seed"]
    else:
        keys(w, "kind operation campaign_id profile_id selected_seed target reference supplied_input")
        tag(w["operation"], [0, 1]); text(w["campaign_id"], True); text(w["profile_id"], True); integer(w["selected_seed"]); fr(w["reference"], folder)
        if w["target"] is not None: fr(w["target"], folder)
        if w["supplied_input"] is not None: hash_value(w["supplied_input"])
        need((w["target"] is not None) == (w["operation"] == 0), "target-option")
        if w["operation"] == 1:
            need(w["selected_seed"] == 0 and w["supplied_input"] is None, "lock-options")
        elif w["supplied_input"] is not None:
            need(w["supplied_input"] == sha(E["synthetic_context"]["materialized_preimage"].encode()), "supplied-materialized")
        seed = w["selected_seed"]
    need(any(x["kind"] == w["kind"] and x.get("operation") == w["operation"] and x.get("profile_id") == w["profile_id"] and seed in x.get("seeds", []) for x in model["supported_work"]), "work-membership")

def filesystem_profile(o, folder):
    keys(o, "identity roots mount_evidence quota_evidence derivation")
    identity(o["identity"]); root_roles(o["roots"])
    for f in ["mount_evidence", "quota_evidence", "derivation"]: fr(o[f], folder)

def fs_order(a):
    ordered_unique([(x["identity"]["device"], x["identity"]["mount_id"], x["identity"]["type"]) for x in a], "filesystem-order")

def topology(o, folder):
    keys(o, "membership mountinfo controllers namespaces")
    for f in ["membership", "mountinfo", "controllers"]: fr(o[f], folder)
    namespace(o["namespaces"], folder)

def validate_record(b, base, folder):
    o = source_json(b)
    domain_name = "claim" if base.startswith("claim-") else base
    need(type(o) is dict and "schema" in o and "synthetic" in o, "keys")
    need(o["schema"] == ("hekit-test-" if folder == "synthetic" else "hekit-study-") + "resource-" + domain_name + "-v1" and type(o["synthetic"]) is bool and o["synthetic"] == (folder == "synthetic"), "domain")
    model = source_json((HERE / folder / "model.json").read_bytes())
    validation = source_json((HERE / folder / "validation.json").read_bytes())
    policy = source_json((HERE / folder / "host-policy.json").read_bytes())
    if base == "model":
        keys(o, "schema synthetic revision source_manifest source_revision source_files registry build_environment rule_revision supported_work components bootstrap_derivation")
        text(o["revision"], True); source_identity(o); hash_value(o["registry"])
        array(o["source_files"], 512); need(bool(o["source_files"]), "source-order")
        for f in o["source_files"]: fr(f, folder)
        ordered_unique([(f["root"], f["path"]) for f in o["source_files"]], "source-order")
        fr(o["build_environment"], folder); text(o["rule_revision"], True); need(o["rule_revision"] == "d7-cpu-allocation-v1", "rule-revision")
        supported(o["supported_work"]); component_order(o["components"])
        for c in o["components"]:
            keys(c, "kind rule derivation"); tag(c["kind"], range(22)); tag(c["rule"], range(22)); need(c["kind"] == c["rule"], "rule-identity"); fr(c["derivation"], folder)
        fr(o["bootstrap_derivation"], folder)
    elif base == "validation":
        keys(o, "schema synthetic model source_manifest source_revision environment coverage runs")
        fr(o["model"], folder); source_identity(o); environment(o["environment"], folder)
        need(o["environment"]["build_receipt"] == raw_ref("executable.txt"), "build-identity")
        component_order(o["coverage"])
        for c in o["coverage"]:
            keys(c, "kind source_derivation validation_runs"); tag(c["kind"], range(22)); fr(c["source_derivation"], folder); run_ids(c["validation_runs"])
        array(o["runs"])
        for r in o["runs"]:
            keys(r, "id command stdout stderr metadata covered_components"); text(r["id"], True)
            for f in ["command", "stdout", "stderr"]: fr(r[f], folder)
            metadata(r["metadata"], folder)
            need(r["metadata"]["environment"] == o["environment"], "environment-equality")
            array(r["covered_components"])
            for k in r["covered_components"]: tag(k, range(22))
            ordered_unique(r["covered_components"], "covered-order")
        ids = [r["id"] for r in o["runs"]]; ordered_unique(ids, "run-order")
        by_id = {r["id"]: r for r in o["runs"]}
        linked = set(o["environment"]["bootstrap"]["validation_runs"])
        need(linked <= set(ids), "run-reference")
        for c in o["coverage"]:
            need(set(c["validation_runs"]) <= set(ids), "run-reference")
            linked.update(c["validation_runs"])
            for rid in c["validation_runs"]: need(c["kind"] in by_id[rid]["covered_components"], "reciprocal-coverage")
        for r in o["runs"]:
            for k in r["covered_components"]: need(r["id"] in o["coverage"][k]["validation_runs"], "reciprocal-coverage")
        need(linked == set(ids), "unreferenced-run")
    elif base == "host-policy":
        keys(o, "schema synthetic boot_id physical_host host_namespace_origin allocator_environment kernel_environment filesystem_environment claim_domain model validation")
        uuid(o["boot_id"]); hash_value(o["physical_host"]); namespace(o["host_namespace_origin"], folder)
        allocator(o["allocator_environment"], folder); kernel(o["kernel_environment"], folder); array(o["filesystem_environment"])
        for f in o["filesystem_environment"]: filesystem_profile(f, folder)
        fs_order(o["filesystem_environment"])
        need(o["claim_domain"] == "task54-host-serial-v1", "claim-domain"); fr(o["model"], folder); fr(o["validation"], folder)
        need(o["boot_id"] == o["host_namespace_origin"]["boot_id"], "policy-boot")
        need(o["allocator_environment"] == validation["environment"]["allocator"] and o["kernel_environment"] == validation["environment"]["kernel"], "policy-environment")
    elif base.startswith("claim-"):
        keys(o, "schema synthetic claim_id claim_domain physical_host boot_id model validation work approval source_manifest source_revision created modeled_envelope")
        text(o["claim_id"], True); need(o["claim_domain"] == "task54-host-serial-v1", "claim-domain")
        hash_value(o["physical_host"]); uuid(o["boot_id"]); fr(o["model"], folder); fr(o["validation"], folder); hash_value(o["approval"]); source_identity(o); clock(o["created"])
        need(o["physical_host"] == policy["physical_host"] and o["boot_id"] == policy["boot_id"], "claim-host")
        requirements(o["modeled_envelope"]); claim_work(o["work"], folder, model)
    elif base == "host-observation":
        keys(o, "schema synthetic claim_id sequence physical started finished topology_before topology_after meminfo_before meminfo_after controllers structural_root")
        text(o["claim_id"], True); integer(o["sequence"]); hash_value(o["physical"]); clocks(o)
        topology(o["topology_before"], folder); topology(o["topology_after"], folder)
        fr(o["meminfo_before"], folder); fr(o["meminfo_after"], folder); array(o["controllers"])
        seen = []
        for c in o["controllers"]:
            keys(c, "version path limit_raw current_raw"); tag(c["version"], [0, 1]); text(c["path"]); need(bool(c["path"]), "text"); fr(c["limit_raw"], folder); fr(c["current_raw"], folder); seen.append((c["version"], c["path"]))
        need(len(set(seen)) == len(seen), "controller-duplicate")
        r = o["structural_root"]
        keys(r, "version domain_path mount_identity disposition controller_raw"); tag(r["version"], [0, 1]); text(r["domain_path"]); need(bool(r["domain_path"]), "text"); identity(r["mount_identity"]); tag(r["disposition"], [0, 1]); array(r["controller_raw"])
        for f in r["controller_raw"]: fr(f, folder)
        need(len({(f["root"], f["path"]) for f in r["controller_raw"]}) == len(r["controller_raw"]), "root-evidence-duplicate")
        need(r["disposition"] != 1 or r["version"] == 1, "structural-root")
        need(not (r["disposition"] == 1 and any(c["path"] == r["domain_path"] for c in o["controllers"])), "structural-root-row")
        need(o["claim_id"] == E["synthetic_context"]["claim_id"], "claim-link")
        need(o["physical"] == sha(E["synthetic_context"]["physical_preimage"].encode()), "physical-link")
        need(o["topology_before"]["namespaces"] == policy["host_namespace_origin"] == o["topology_after"]["namespaces"], "topology-origin")
        need([c["path"] for c in o["controllers"]] == ["/child", "/parent"], "controller-order")
    elif base == "disk-observation":
        keys(o, "schema synthetic claim_id sequence started finished filesystems")
        text(o["claim_id"], True); integer(o["sequence"]); clocks(o); array(o["filesystems"])
        total = 0
        for f in o["filesystems"]:
            keys(f, "identity roots frsize bavail favail required_bytes required_files quota_evidence")
            identity(f["identity"]); root_roles(f["roots"])
            for k in ["frsize", "bavail", "favail", "required_bytes", "required_files"]: integer(f[k])
            fr(f["quota_evidence"], folder); need(f["frsize"] > 0, "disk-frsize")
            available = multiply(f["frsize"], f["bavail"]); total = add(total, f["required_bytes"])
            need(f["required_bytes"] <= available, "disk-bytes"); need(f["required_files"] <= f["favail"], "disk-inodes")
            need(any(f["identity"] == p["identity"] and f["roots"] == p["roots"] and f["quota_evidence"] == p["quota_evidence"] for p in policy["filesystem_environment"]), "disk-scope")
        fs_order(o["filesystems"])
        need(o["claim_id"] == E["synthetic_context"]["claim_id"], "claim-link")
        need(o["sequence"] == 1, "observation-sequence")
        need(total == E["required_disk_bytes"], "disk-total")
    else:
        raise Refusal("unknown-record")
    return o

def bootstrap(b, folder):
    test = folder == "synthetic"
    need(len(b) == (703 if test else 704), "bootstrap-length")
    h, pos = binary_text(b, 0, "bootstrap-domain"); d, pos = binary_text(b, pos, "bootstrap-domain")
    need(h == "hekit-study-reference-encoding-v1" and d == ("hekit-test-" if test else "hekit-study-") + "resource-bootstrap-v1" and b[pos] == int(test), "bootstrap-domain")
    pos += 1
    fields = {}
    def number():
        nonlocal pos
        value = int.from_bytes(b[pos:pos + 8], "little"); pos += 8
        return value
    def fixed_hash():
        nonlocal pos
        need(number() == 64, "bootstrap-hash-prefix")
        value = b[pos:pos + 64].decode("ascii"); pos += 64; hash_value(value)
        return value
    fields["generation"] = number()
    fields["origin"] = fixed_hash(); fields["policy"] = fixed_hash()
    need(number() == 36, "bootstrap-uuid-length")
    boot = b[pos:pos + 36].decode("ascii"); pos += 36
    uid = number(); hostkey = fixed_hash()
    namespaces = [[number(), number()] for _ in range(4)]
    proc = [number() for _ in range(4)]; cgroup = [number() for _ in range(4)]
    profiles = [fixed_hash() for _ in range(3)]
    need(pos == len(b) - 1 and b[pos] == 0, "bootstrap-claim-domain")
    policy = source_json((HERE / folder / "host-policy.json").read_bytes())
    need(fields["origin"] == policy["host_namespace_origin"]["installation_record"]["sha256"], "bootstrap-origin")
    need(fields["policy"] == sha((HERE / folder / "host-policy.json").read_bytes()), "bootstrap-policy")
    need(boot == policy["boot_id"] == policy["host_namespace_origin"]["boot_id"], "bootstrap-boot")
    need(namespaces == E["namespace_pairs"], "bootstrap-namespaces")
    need(profiles == [sha(("fictional " + name + " selector\n").encode()) for name in ["kernel", "allocator", "filesystem"]], "bootstrap-profile")
    need(fields["generation"] == 3 and uid == 1000 and hostkey == sha(b"fictional stable host key\n") and proc == [31, 41, 51, 40864] and cgroup == [32, 42, 52, 1667723888], "bootstrap-anchor-literal")
    # Independent literal offsets, not offsets reported by the generator.
    delta = -1 if test else 0
    offsets = E["bootstrap_production_offsets"]
    need(int.from_bytes(b[offsets["generation"] + delta:offsets["generation"] + delta + 8], "little") == 3, "literal-offset")
    need(b[offsets["claim_domain"] + delta] == 0, "literal-offset")
    return fields

def witness(o):
    """Illustrative private context only; deliberately no source-format fields."""
    k = o["kind"]
    if k == "owner-input":
        need(not o["some"] or o["already_exists"], "input-not-existing")
        need(not o["some"] or o["same_owner"], "input-other-owner")
    elif k == "finalize": need(o["work_kind"] == 0, "preparation-promotion")
    elif k == "ancestry":
        need(o["origin_complete"], "hidden-ancestor")
        need(o["paths"] == ["/child", "/parent"], "missing-ancestor")
    elif k == "sum":
        total = 0
        for v in o["values"]: total = add(total, v)
    elif k == "raw-cap":
        integer(o["length"]); integer(o["cap"]); need(o["length"] <= o["cap"], "raw-cap")
    elif k == "huge-finite": need(o["normalized"] == o["raw"], "finite-not-unlimited")
    elif k == "other-claims": need(o["other"] != 0 or o["serial_domain_established"], "serial-domain")
    elif k == "finite-headroom":
        available = min(o["host"], max(0, o["limit"] - o["current"]))
        need(max(0, available - o["other"]) == o["expected"], "headroom")
    elif k == "opaque-profile-not-json-hash": need(o["selectors_are_opaque"], "profile-contract")
    else: raise Refusal("unknown-witness")

def main():
    # HERE is repo/.superpowers/sdd/ledger/oracle; parents[3] is repo.
    contract = HERE.parents[3] / "results/validation/task54-d7-resource-format/task-54-d7-resource-format-contract.md"
    need(sha(contract.read_bytes()) == E["contract_sha256"], "contract-pin")
    manifest = source_json((HERE / "manifest.json").read_bytes())
    need(len(manifest) == 123, "manifest-count")
    for name, pin in manifest.items():
        need(not name.startswith("/") and ".." not in name.split("/"), "manifest-path")
        b = (HERE / name).read_bytes()
        need(len(b) == pin["length"] and sha(b) == pin["sha256"], "manifest-pin")
    positive = []
    names = ["model", "validation", "host-policy", "claim-train-full", "claim-train-quick", "claim-reference-read", "claim-reference-recover", "claim-reference-snapshot", "claim-target-read-future", "claim-target-read-supplied", "claim-target-lock", "host-observation", "disk-observation"]
    for folder in ["synthetic", "inert-production"]:
        for name in names:
            o = validate_record((HERE / folder / (name + ".json")).read_bytes(), name, folder)
            if name.startswith("claim-"):
                sums, extras = requirements(o["modeled_envelope"])
                need(sums == [6, 3, 0, 0] == E["phase_total"] and extras == [5, 3, 0, 0] == E["phase_additional"], "literal-phase-sums")
            positive.append(folder + "/" + name)
        bootstrap((HERE / folder / "bootstrap.bin").read_bytes(), folder)
        positive.append(folder + "/bootstrap")
    for w in E["positive_witnesses"]: witness(w)
    outcomes = []
    need(len(CASES) == 84, "negative-count")
    for case in CASES:
        ext = ".bin" if case["base"] == "bootstrap" else ".json"
        b = (HERE / "negative" / (case["id"] + ext)).read_bytes()
        observed = "ACCEPTED"
        try:
            if case["base"] == "bootstrap": bootstrap(b, "synthetic")
            elif case["base"] == "witness": witness(source_json(b))
            else: validate_record(b, case["base"], "synthetic")
        except Refusal as error:
            observed = str(error)
        category = "private-context-witness" if case["base"] == "witness" else "bootstrap-structural" if case["base"] == "bootstrap" else "source-json-structural"
        outcomes.append(dict(id=case["id"], category=category, expected=case["expect"], observed=observed, first_refusal_matches=observed == case["expect"]))
    result = dict(notice=E["notice"], positive_source_json=26, positive_bootstrap=2, positive_private_witnesses=5, negatives_source_json=66, negatives_bootstrap=9, negatives_private_witnesses=9, positive_paths=positive, outcomes=outcomes, all_expected=all(x["first_refusal_matches"] for x in outcomes), no_project_or_native_execution=True, limitation="Structural checks and private context illustrations only; native origin/lifetime, physical model/source validity, actual complete receipt semantics and experiment applicability remain absent.")
    (HERE / "verification-result.json").write_text(json.dumps(result, indent=2) + "\n")
    mismatches = [x for x in outcomes if not x["first_refusal_matches"]]
    print(json.dumps(dict(positive=len(positive), negatives=len(outcomes), mismatches=mismatches), sort_keys=True))
    need(not mismatches, "first-refusal-mismatch")

if __name__ == "__main__":
    main()
