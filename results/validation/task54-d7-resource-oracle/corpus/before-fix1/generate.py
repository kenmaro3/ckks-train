"""Finite fictional resource-format artifacts. No project imports or descendants."""
from pathlib import Path
import copy
import hashlib
import json
import struct

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "task-54-d7-gate-schema-vectors"
E = json.loads((HERE / "literal-expectations.json").read_text())
CASES = json.loads((HERE / "negative-cases.json").read_text())
assert len(CASES) == 84
OUT = {}

def digest(b):
    return hashlib.sha256(b).hexdigest()

def emit(name, b):
    assert name not in OUT and len(b) <= 65536
    OUT[name] = b

def encoded(obj):
    return (json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n").encode()

RAW = {k: v.encode() for k, v in E["raw_literals"].items()}
for name, b in RAW.items():
    assert len(b) <= 256
    emit("raw/" + name, b)

def raw_fr(name, root=2):
    return dict(root=root, path="fictional/" + name, length=len(RAW[name]), sha256=digest(RAW[name]))

def u64(n):
    return struct.pack("<Q", n)

def text(s):
    b = s.encode()
    return u64(len(b)) + b

def old_text(b, pos):
    n = int.from_bytes(b[pos:pos + 8], "little")
    return b[pos + 8:pos + 8 + n].decode(), pos + 8 + n

NESTED = {}
for name, (length, sha) in E["nested_input_pins"].items():
    b = (OLD / name).read_bytes()
    assert len(b) == length and digest(b) == sha
    NESTED[name] = b

def request(name, synthetic):
    b = NESTED[name]
    header, p = old_text(b, 0)
    domain, p = old_text(b, p)
    assert header == E["primitive_header"] and domain == "hekit-test-gate-request-v1" and b[p] == 1
    # Preserve frozen request body exactly; legacy registration is never transformed.
    domain = ("hekit-test-" if synthetic else "hekit-study-") + "gate-request-v1"
    return text(header) + text(domain) + bytes([int(synthetic)]) + b[p + 1:]

CLOCK = dict(wall_source=0, unix_ns=1, monotonic_source=0, elapsed_since_operation_start_ns=0)
FINISH = dict(wall_source=0, unix_ns=2, monotonic_source=0, elapsed_since_operation_start_ns=1)
FACT = raw_fr("facts.txt")
NS = dict(boot_id=E["boot_uuid"], **{name: dict(device=d, inode=i) for name, (d, i) in zip(
    ["mount_ns", "cgroup_ns", "pid_ns", "user_ns"], E["namespace_pairs"])}, installation_record=copy.deepcopy(FACT))
ALLOCATOR = dict(implementation=0, library_files=[raw_fr("executable.txt")], configuration=copy.deepcopy(FACT), derivation=copy.deepcopy(FACT))
KERNEL = dict(release="fictional-kernel", version="fictional version; no observed host", page_size=4, source=copy.deepcopy(FACT), configuration=copy.deepcopy(FACT))
ENV = dict(target="fictional-linux", toolchain=copy.deepcopy(FACT), cargo_lock=copy.deepcopy(FACT),
           build_receipt=raw_fr("executable.txt"), allocator=ALLOCATOR, kernel=KERNEL, workers=1,
           bootstrap=dict(source_derivation=copy.deepcopy(FACT), validation_runs=["run-a", "run-b"]))
FSID = dict(device=3, mount_id=4, type=5)
FSPROFILE = dict(identity=copy.deepcopy(FSID), roots=[0, 1, 2, 3, 4, 5], mount_evidence=copy.deepcopy(FACT), quota_evidence=copy.deepcopy(FACT), derivation=copy.deepcopy(FACT))
REQUIREMENTS = dict(components=[dict(kind=i, phase_mask=E["requirements_masks"][i], resident_bytes=E["requirements_resident"][i], additional_bytes=E["requirements_additional"][i]) for i in range(22)], peak_host_bytes=6, incremental_host_bytes=5, required_disk_bytes=12)
PHYSICAL = digest(E["synthetic_context"]["physical_preimage"].encode())
HOST = digest(E["synthetic_context"]["physical_host_preimage"].encode())
SELECTORS = [digest(("fictional " + x + " selector\n").encode()) for x in ["kernel", "allocator", "filesystem"]]
PACKAGES = {}

for folder, synthetic in [("synthetic", True), ("inert-production", False)]:
    prefix = "hekit-test-" if synthetic else "hekit-study-"
    records = {}

    def record(label, **fields):
        return dict(schema=prefix + "resource-" + label + "-v1", synthetic=synthetic, **fields)

    def package_file(name, obj):
        b = encoded(obj)
        assert len(b) <= 32768
        records[name] = obj
        emit(folder + "/" + name + ".json", b)
        return b

    def ref(name):
        key = folder + "/" + name + ".json"
        assert key in OUT
        return dict(root=2, path="resource/linux-cpu-v1/" + name + ".json", length=len(OUT[key]), sha256=digest(OUT[key]))

    reg = NESTED["registration.bin"]
    supported = [dict(kind=0, registration_id=digest(reg), consumption=c) for c in [0, 1]]
    supported += [dict(kind=1, operation=op, profile_id="fixture-profile", seeds=[0]) for op in [0, 1, 2]]
    supported += [dict(kind=2, operation=op, profile_id="fixture-profile", seeds=[0, 7] if op == 0 else [0]) for op in [0, 1]]
    package_file("model", record("model", revision="fixture-revision", source_manifest=digest(RAW["manifest.txt"]), source_revision=E["source_revision"], source_files=[raw_fr("manifest.txt", 1), raw_fr("source.txt", 1)], registry=digest(RAW["registry.txt"]), build_environment=raw_fr("executable.txt"), rule_revision="d7-cpu-allocation-v1", supported_work=supported, components=[dict(kind=i, rule=i, derivation=copy.deepcopy(FACT)) for i in range(22)], bootstrap_derivation=copy.deepcopy(FACT)))
    runs = []
    for ident, exit_code, timed_out, covered in [("run-a", 0, False, list(range(22))), ("run-b", None, True, [])]:
        md = dict(source_manifest=digest(RAW["manifest.txt"]), source_revision=E["source_revision"], executable=digest(RAW["executable.txt"]), environment=copy.deepcopy(ENV), argv=[b"fictional-validate".hex(), "ff", ""], exit_code=exit_code, timed_out=timed_out, started=copy.deepcopy(CLOCK), finished=copy.deepcopy(FINISH), measurements=[dict(kind=k, scope=k, bytes=k + 3, raw=copy.deepcopy(FACT)) for k in range(3)] if not timed_out else [])
        runs.append(dict(id=ident, command=copy.deepcopy(FACT), stdout=copy.deepcopy(FACT), stderr=raw_fr("empty.bin"), metadata=md, covered_components=covered))
    package_file("validation", record("validation", model=ref("model"), source_manifest=digest(RAW["manifest.txt"]), source_revision=E["source_revision"], environment=copy.deepcopy(ENV), coverage=[dict(kind=i, source_derivation=copy.deepcopy(FACT), validation_runs=["run-a"]) for i in range(22)], runs=runs))
    policy = record("host-policy", boot_id=E["boot_uuid"], physical_host=HOST, host_namespace_origin=copy.deepcopy(NS), allocator_environment=copy.deepcopy(ALLOCATOR), kernel_environment=copy.deepcopy(KERNEL), filesystem_environment=[copy.deepcopy(FSPROFILE)], claim_domain="task54-host-serial-v1", model=ref("model"), validation=ref("validation"))
    package_file("host-policy", policy)
    works = {}
    for label, c in [("full", "full.request.bin"), ("quick", "quick.request.bin")]:
        works["train-" + label] = dict(kind=0, registration=reg.hex(), request=request(c, synthetic).hex())
    for label, op in [("read", 0), ("recover", 1), ("snapshot", 2)]:
        works["reference-" + label] = dict(kind=1, operation=op, reference_id="fictional-reference", expected_file=digest(RAW["reference.json"]) if op == 1 else None, profile_id="fixture-profile", seed=0, document=raw_fr("reference.json", 0))
    for label, supplied in [("read-future", False), ("read-supplied", True)]:
        works["target-" + label] = dict(kind=2, operation=0, campaign_id="fictional-campaign", profile_id="fixture-profile", selected_seed=7, target=raw_fr("target.json", 0), reference=raw_fr("reference.json", 0), supplied_input=digest(RAW["materialized.txt"]) if supplied else None)
    works["target-lock"] = dict(kind=2, operation=1, campaign_id="fictional-campaign", profile_id="fixture-profile", selected_seed=0, target=None, reference=raw_fr("reference.json", 0), supplied_input=None)
    for label, work in works.items():
        package_file("claim-" + label, record("claim", claim_id="fictional-claim", claim_domain="task54-host-serial-v1", physical_host=HOST, boot_id=E["boot_uuid"], model=ref("model"), validation=ref("validation"), work=work, approval=digest(b"fictional approval; no authority\n"), source_manifest=digest(RAW["manifest.txt"]), source_revision=E["source_revision"], created=copy.deepcopy(CLOCK), modeled_envelope=copy.deepcopy(REQUIREMENTS)))
    topology = dict(membership=copy.deepcopy(FACT), mountinfo=copy.deepcopy(FACT), controllers=copy.deepcopy(FACT), namespaces=copy.deepcopy(NS))
    package_file("host-observation", record("host-observation", claim_id="fictional-claim", sequence=1, physical=PHYSICAL, started=copy.deepcopy(CLOCK), finished=copy.deepcopy(FINISH), topology_before=copy.deepcopy(topology), topology_after=copy.deepcopy(topology), meminfo_before=copy.deepcopy(FACT), meminfo_after=copy.deepcopy(FACT), controllers=[dict(version=1, path=p, limit_raw=copy.deepcopy(FACT), current_raw=copy.deepcopy(FACT)) for p in ["/child", "/parent"]], structural_root=dict(version=1, domain_path="/", mount_identity=copy.deepcopy(FSID), disposition=1, controller_raw=[copy.deepcopy(FACT)])))
    package_file("disk-observation", record("disk-observation", claim_id="fictional-claim", sequence=1, started=copy.deepcopy(CLOCK), finished=copy.deepcopy(FINISH), filesystems=[dict(identity=copy.deepcopy(FSID), roots=[0, 1, 2, 3, 4, 5], frsize=4, bavail=8, favail=3, required_bytes=12, required_files=2, quota_evidence=copy.deepcopy(FACT))]))
    b = text(E["primitive_header"]) + text(prefix + "resource-bootstrap-v1") + bytes([int(synthetic)])
    b += u64(3) + text(FACT["sha256"]) + text(digest(OUT[folder + "/host-policy.json"]))
    b += u64(36) + E["boot_uuid"].encode() + u64(1000) + text(digest(b"fictional stable host key\n"))
    for pair in E["namespace_pairs"]:
        for value in pair:
            b += u64(value)
    for value in E["proc_root"] + E["cgroup_root"]:
        b += u64(value)
    b += b"".join(text(s) for s in SELECTORS) + b"\x00"
    assert len(b) == E["bootstrap_lengths"][folder]
    emit(folder + "/bootstrap.bin", b)
    PACKAGES[folder] = records

for case in CASES:
    base, op = case["base"], case["op"]
    ext = ".bin" if base == "bootstrap" else ".json"
    if base == "bootstrap":
        b = bytearray(OUT["synthetic/bootstrap.bin"])
        if op == "truncate": b = b[:-1]
        elif op == "append-byte": b.append(0)
        elif op == "byte":
            i, value = case["offset"], case["value"]
            b[i] = value if b[i] != value else value + 1
        else: raise AssertionError(op)
        emit("negative/" + case["id"] + ext, bytes(b))
        continue
    if base == "witness":
        emit("negative/" + case["id"] + ext, encoded(case["value"]))
        continue
    obj = copy.deepcopy(PACKAGES["synthetic"][base])
    if "path" in case:
        node = obj
        for k in case["path"][:-1]: node = node[k]
        k = case["path"][-1]
    if op == "set": node[k] = case["value"]
    elif op == "drop":
        if isinstance(node, list): node.pop(k)
        else: del node[k]
    elif op == "swap": node[k][0], node[k][1] = node[k][1], node[k][0]
    elif op == "duplicate": node[k].append(copy.deepcopy(node[k][0]))
    elif op == "copy-field":
        value = obj
        for key in case["from"]: value = value[key]
        node[k] = copy.deepcopy(value)
    elif op == "hash": node[k] = digest(RAW[case["hash_of"]])
    elif op == "overlap-work": obj["supported_work"].insert(2, copy.deepcopy(obj["supported_work"][2]))
    elif op == "unreferenced-run":
        value = copy.deepcopy(obj["runs"][-1]); value["id"] = "run-z"; obj["runs"].append(value)
    elif op == "target-some": obj["work"]["target"] = raw_fr("target.json", 0)
    elif op == "wrong-materialized":
        s = obj["work"]["supplied_input"]; obj["work"]["supplied_input"] = ("0" if s[0] != "0" else "1") + s[1:]
    elif op == "request-domain":
        b = bytearray.fromhex(obj["work"]["request"]); b[49] = ord("x"); obj["work"]["request"] = b.hex()
    elif op not in ["raw", "duplicate-key", "lex", "invalid-utf8", "append", "bom", "escaped-bytes"]:
        raise AssertionError(op)
    b = encoded(obj)
    if op == "raw": b = case["value"].encode()
    elif op == "duplicate-key": b = b[:1] + b'"\\u0073chema":"alias",' + b[1:]
    elif op == "lex": b = b.replace(b'"sequence":1,', ('"sequence":' + case["value"] + ',').encode(), 1)
    elif op == "invalid-utf8": b = b + b"\xff"
    elif op == "append": b += case["value"].encode()
    elif op == "bom": b = b"\xef\xbb\xbf" + b
    elif op == "escaped-bytes":
        marker = b'"registration":"'; i = b.index(marker) + len(marker)
        b = b[:i] + b"\\u00" + format(b[i], "02x").encode() + b[i + 1:]
    emit("negative/" + case["id"] + ext, b)

manifest = {name: dict(length=len(b), sha256=digest(b)) for name, b in sorted(OUT.items())}
emit("manifest.json", encoded(manifest))
emit("generation-result.json", encoded(dict(notice=E["notice"], input_pins=E["nested_input_pins"], package_json=26, bootstrap_binary=2, raw_files=11, negative_json=75, negative_binary=9, output_count=125, generated_bytes_before_bookkeeping=sum(x["length"] for x in manifest.values()), no_descendants=True)))
assert len(OUT) == 125 and sum(map(len, OUT.values())) <= 3145728
for name in OUT:
    assert not (HERE / name).exists(), "Refuse overwrite: " + name
for name, b in OUT.items():
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)
print(json.dumps(dict(output_count=len(OUT), aggregate_bytes=sum(map(len, OUT.values())), largest_bytes=max(map(len, OUT.values())), notice=E["notice"]), sort_keys=True))
