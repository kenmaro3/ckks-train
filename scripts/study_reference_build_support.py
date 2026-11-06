"""Private, bounded primitives for the fixed study-reference build recipe.

These encodings are transport, not approval. Production policy is enforced by
validate_recipe_capsule and the explicit recipe entry point. No scientific run.
"""

import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess

LIMIT = 8 * 1024 * 1024
CAPSULE_LIMIT = 2 * 1024 * 1024
TEXT = 4096
COUNT = 65536
RECIPE = "study-reference-linux-clean-v1"
ALGORITHM = "study-reference-session-v1"
TARGET = "x86_64-unknown-linux-gnu"
DESIGNS = (
    "docs/study-reference-session-design.md",
    "docs/study-reference-document-design.md",
    "docs/study-reference-design.md",
    "docs/experiment-matrix.md",
    "docs/study-training-design.md",
)
DOMAINS = {
    k: "hekit-study-" + v + "-v1"
    for k, v in [
        ("manifest", "build-files"),
        ("environment", "build-environment"),
        ("capsule", "compile-capsule"),
        ("receipt", "build-receipt"),
        ("approval", "local-approval"),
    ]
}


class Invalid(ValueError):
    """Invalid, changed, oversized or unsupported provenance input."""


def require(condition, reason):
    if not condition:
        raise Invalid(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_hash(value):
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "sha256",
    )
    return value


class Writer:
    def __init__(self, domain, limit=LIMIT):
        self.data = bytearray()
        self.limit = limit
        self.t("hekit-study-reference-encoding-v1")
        self.t(DOMAINS[domain])

    def add(self, b):
        require(len(b) <= self.limit - len(self.data), "encoded byte cap")
        self.data.extend(b)

    def n(self, n):
        require(type(n) is int and 0 <= n < 2**64, "u64")
        self.add(struct.pack("<Q", n))

    def q(self, n):
        require(type(n) is int and 0 <= n <= 255, "tag")
        self.add(bytes([n]))

    def b(self, b, cap=TEXT):
        require(isinstance(b, bytes) and len(b) <= cap, "byte field cap")
        self.n(len(b))
        self.add(b)

    def t(self, t, cap=TEXT):
        require(isinstance(t, str), "text")
        self.b(t.encode("utf8"), cap)

    def h(self, h):
        self.t(valid_hash(h))

    def count(self, items, cap=COUNT):
        require(len(items) <= cap, "count cap")
        self.n(len(items))

    def done(self):
        return bytes(self.data)


class Reader:
    def __init__(self, data, domain, cap=LIMIT):
        require(isinstance(data, bytes) and len(data) <= cap, "byte cap")
        self.data = data
        self.i = 0
        require(
            self.t() == "hekit-study-reference-encoding-v1"
            and self.t() == DOMAINS[domain],
            "domain",
        )

    def take(self, n):
        require(0 <= n <= len(self.data) - self.i, "truncated")
        r = self.data[self.i : self.i + n]
        self.i += n
        return r

    def n(self):
        return struct.unpack("<Q", self.take(8))[0]

    def q(self, tags):
        q = self.take(1)[0]
        require(q in tags, "tag")
        return q

    def b(self, cap=TEXT):
        n = self.n()
        require(n <= cap, "byte cap")
        return self.take(n)

    def t(self, cap=TEXT):
        try:
            return self.b(cap).decode("utf8")
        except UnicodeError as e:
            raise Invalid("utf8") from e

    def h(self):
        return valid_hash(self.t())

    def count(self, cap=COUNT):
        n = self.n()
        require(n <= cap and n <= len(self.data) - self.i, "count cap")
        return range(n)


def unique_sorted(rows, key=lambda x: x):
    rows = sorted(rows, key=key)
    require(all(key(a) != key(b) for a, b in zip(rows, rows[1:])), "duplicate key")
    return rows


def manifest(role, records):
    require(role in range(4), "role")
    w = Writer("manifest")
    w.q(role)
    records = unique_sorted(records, lambda x: x[0])
    w.count(records, 512 if role == 0 else COUNT)
    for path, tag, payload in records:
        w.b(path)
        require(tag in (0, 1, 2), "file tag")
        w.q(tag)
        if tag == 1:
            w.h(payload)
        elif tag == 2:
            w.b(payload)
        else:
            require(payload is None, "absent payload")
    return w.done()


def environment(records):
    w = Writer("environment")
    records = unique_sorted(records, lambda x: x[0].encode())
    w.count(records)
    for key, value in records:
        require(key.isascii() and key and "\0" not in key, "environment key")
        w.t(key)
        w.q(0 if value is None else 1)
        if value is not None:
            w.b(value)
    return w.done()


def capsule(c):
    preflight_capsule(c)
    w = Writer("capsule", CAPSULE_LIMIT)
    w.t(c["recipe"])
    w.t(c["algorithm"])
    require(len(c["manifests"]) == 4, "manifest count")
    for h in c["manifests"]:
        w.h(h)
    w.h(c["environment"])
    w.h(c["lock"])
    w.t(c["commit"])
    w.t(c["compiler"], 16384)
    w.t(c["cargo"])
    for k in ("target", "host", "profile", "opt"):
        w.t(c[k])
    require(c["debug"] in (0, 1), "debug")
    w.q(c["debug"])
    features = unique_sorted(c["features"])
    w.count(features, 64)
    for x in features:
        w.t(x)
    w.t(c["version"])
    w.h(c["metadata"])
    packages = unique_sorted(c["packages"], lambda x: x[0].encode())
    w.count(packages)
    for ident, path, tag, source in packages:
        w.t(ident)
        w.b(path)
        require(tag in (0, 1), "package tag")
        w.q(tag)
        w.t(source)
    for k in ("metadata_argv", "build_argv"):
        w.count(c[k], 256)
        for x in c[k]:
            w.b(x)
    w.b(c["cwd"])
    return w.done()


def emitted_key(r):
    return (
        r[0].encode(),
        r[2].encode(),
        r[3],
        r[1],
        tuple(x.encode() for x in r[4]),
        (0, b"") if r[5] is None else (1, r[5]),
    )


def receipt(r):
    preflight_receipt(
        len(r["capsule"]),
        [len(b) for b in r["files"]],
        len(r["environment"]),
        r["executable"],
        r["emitted"],
        r["argv"],
    )
    c = decode("capsule", r["capsule"])
    require(len(r["files"]) == 4 and len(r["post"]) == 5, "receipt lists")
    for role, b in enumerate(r["files"]):
        m = decode("manifest", b)
        require(
            m["role"] == role and sha(b) == c["manifests"][role],
            "embedded manifest link",
        )
    decode("environment", r["environment"])
    require(sha(r["environment"]) == c["environment"], "environment link")
    w = Writer("receipt")
    w.b(r["capsule"], CAPSULE_LIMIT)
    for x in r["files"]:
        w.b(x, LIMIT)
    w.b(r["environment"], LIMIT)
    w.b(r["executable"])
    w.h(r["executable_hash"])
    for x in r["post"]:
        w.h(x)
    w.h(r["metadata"])
    emitted = set(
        (a, b, c, d, tuple(sorted(k)), e) for a, b, c, d, k, e in r["emitted"]
    )
    emitted = sorted(emitted, key=emitted_key)
    w.count(emitted)
    for ident, path, name, src, kinds, exe in emitted:
        w.t(ident)
        w.b(path)
        w.t(name)
        w.b(src)
        w.count(kinds, COUNT)
        require(len(set(kinds)) == len(kinds), "duplicate kind")
        for k in kinds:
            w.t(k)
        w.q(0 if exe is None else 1)
        if exe is not None:
            w.b(exe)
    w.count(r["argv"], 256)
    for x in r["argv"]:
        w.b(x)
    return w.done()


def approval(a):
    w = Writer("approval")
    w.t(a["id"])
    w.t(a["recipe"])
    w.t(a["algorithm"])
    designs = unique_sorted(a["designs"], lambda x: x[0].encode())
    w.count(designs, 32)
    for path, h in designs:
        w.t(path)
        w.h(h)
    w.t(a["commit"])
    w.h(a["source"])
    builds = unique_sorted(a["builds"], lambda x: x[0])
    w.count(builds, 64)
    for triple in builds:
        require(len(triple) == 3, "build triple")
        for h in triple:
            w.h(h)
    return w.done()


def decode(domain, data):
    r = Reader(data, domain, CAPSULE_LIMIT if domain == "capsule" else LIMIT)
    if domain == "manifest":
        role = r.q(range(4))
        rows = []
        for _ in r.count(512 if role == 0 else COUNT):
            path = r.b()
            tag = r.q(range(3))
            rows.append((path, tag, None if tag == 0 else r.h() if tag == 1 else r.b()))
        out = dict(role=role, records=rows)
    elif domain == "environment":
        out = []
        for _ in r.count():
            key = r.t()
            tag = r.q((0, 1))
            out.append((key, None if tag == 0 else r.b()))
    elif domain == "capsule":
        out = dict(
            recipe=r.t(),
            algorithm=r.t(),
            manifests=[r.h() for _ in range(4)],
            environment=r.h(),
            lock=r.h(),
            commit=r.t(),
            compiler=r.t(16384),
            cargo=r.t(),
        )
        for k in ("target", "host", "profile", "opt"):
            out[k] = r.t()
        out["debug"] = r.q((0, 1))
        out["features"] = [r.t() for _ in r.count(64)]
        out["version"] = r.t()
        out["metadata"] = r.h()
        out["packages"] = [(r.t(), r.b(), r.q((0, 1)), r.t()) for _ in r.count()]
        for k in ("metadata_argv", "build_argv"):
            out[k] = [r.b() for _ in r.count(256)]
        out["cwd"] = r.b()
    elif domain == "receipt":
        out = dict(
            capsule=r.b(CAPSULE_LIMIT),
            files=[r.b(LIMIT) for _ in range(4)],
            environment=r.b(LIMIT),
            executable=r.b(),
            executable_hash=r.h(),
            post=[r.h() for _ in range(5)],
            metadata=r.h(),
        )
        rows = []
        for _ in r.count():
            ident = r.t()
            path = r.b()
            name = r.t()
            src = r.b()
            kinds = tuple(r.t() for _ in r.count())
            tag = r.q((0, 1))
            rows.append((ident, path, name, src, kinds, None if tag == 0 else r.b()))
        out["emitted"] = rows
        out["argv"] = [r.b() for _ in r.count(256)]
    elif domain == "approval_v2":
        out = dict(
            id=r.t(),
            designs=[(r.t(), r.h()) for _ in r.count(32)],
            commit=r.t(),
            source=r.h(),
            builds=[
                (r.q((0, 1)), r.t(), r.t(), r.h(), r.h(), r.h()) for _ in r.count(64)
            ],
        )
    elif domain == "approval":
        out = dict(
            id=r.t(),
            recipe=r.t(),
            algorithm=r.t(),
            designs=[(r.t(), r.h()) for _ in r.count(32)],
            commit=r.t(),
            source=r.h(),
            builds=[(r.h(), r.h(), r.h()) for _ in r.count(64)],
        )
    else:
        raise Invalid("domain")
    require(r.i == len(data), "trailing bytes")
    encoded = manifest(**out) if domain == "manifest" else globals()[domain](out)
    require(encoded == data, "noncanonical encoding")
    return out


def canonical(path):
    p = Path(path)
    try:
        resolved = p.resolve(strict=True)
    except OSError as e:
        raise Invalid("canonical path unavailable") from e
    require(
        p.is_absolute() and str(p) == str(resolved), "canonical absolute path required"
    )
    return p


def contained(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def source_path(path):
    try:
        s = path.decode("utf8")
    except UnicodeError as e:
        raise Invalid("source utf8") from e
    require(
        s
        and not s.startswith("/")
        and "\\" not in s
        and "\0" not in s
        and all(x not in ("", ".", "..") for x in s.split("/")),
        "source path",
    )
    return s


def enumerate_tree(root, allow_links, paths=None, max_records=COUNT):
    """Metadata only; expand links before subsequent components, including '..'."""
    root = canonical(root)
    found = {}
    active = set()
    link_targets = {}

    def checked_path(p):
        raw = os.fsencode(p)
        require(len(raw) <= TEXT, "path cap")
        require(contained(p, root), "path escape")
        return raw

    def record(raw, tag, payload):
        require(len(found) < max_records, "inventory count cap")
        found[raw] = (raw, tag, payload)

    def absolute_parts(target):
        # Root itself is canonical. Consume only its actual prefix; never
        # collapse '..' or resolve an outside-root alias to enter this root.
        parts = target.split(b"/")
        index = 0
        for expected in root.parts[1:]:
            while index < len(parts) and parts[index] in (b"", b"."):
                index += 1
            require(
                index < len(parts) and parts[index] == os.fsencode(expected),
                "link escape",
            )
            index += 1
        return parts[index:]

    def resolve_parts(current, parts):
        for part in parts:
            # Even a later '.'/'..'/trailing slash requires a directory. Do not
            # turn a nonexistent or regular-file component into a valid path.
            require(
                stat.S_ISDIR(current.lstat().st_mode), "non-directory link component"
            )
            if part in (b"", b"."):
                continue
            if part == b"..":
                require(current != root, "link escape")
                current = current.parent
                continue
            candidate = current / os.fsdecode(part)
            checked_path(candidate)
            if stat.S_ISLNK(candidate.lstat().st_mode):
                # visit records this link's OWN lexical key/raw target and its
                # terminal closure, returning the expanded location. Only then
                # may a following '..' move to that location's parent.
                current = visit(candidate)
            else:
                current = candidate
        return current

    def visit(p):
        raw = checked_path(p)
        require(raw not in active, "link cycle")
        if raw in found:
            return link_targets.get(raw, p)
        st = p.lstat()
        if stat.S_ISLNK(st.st_mode):
            require(allow_links, "symlink forbidden")
            target = os.readlink(raw)
            require(len(target) <= TEXT, "link cap")
            record(raw, 2, target)
            active.add(raw)
            if target.startswith(b"/"):
                terminal = resolve_parts(root, absolute_parts(target))
            else:
                terminal = resolve_parts(p.parent, target.split(b"/"))
            visit(terminal)
            link_targets[raw] = terminal
            active.remove(raw)
            return terminal
        if stat.S_ISREG(st.st_mode):
            record(raw, 1, None)
        elif stat.S_ISDIR(st.st_mode):
            active.add(raw)
            with os.scandir(p) as it:
                for entry in it:
                    visit(p / entry.name)
            active.remove(raw)
        else:
            raise Invalid("nonregular inventory entry")
        return p

    for selected in paths if paths is not None else [root]:
        selected = Path(selected)
        checked_path(selected)
        parts = [os.fsencode(part) for part in selected.relative_to(root).parts]
        visit(resolve_parts(root, parts))
    return sorted(found.values())


def preflight_manifest(role, rows, max_records=None):
    require(
        len(rows)
        <= (max_records if max_records is not None else 512 if role == 0 else COUNT),
        "manifest count cap",
    )
    n = (
        8
        + len("hekit-study-reference-encoding-v1")
        + 8
        + len(DOMAINS["manifest"])
        + 1
        + 8
    )
    last = None
    for path, tag, payload in sorted(rows):
        require(
            isinstance(path, bytes) and 0 < len(path) <= TEXT and path != last,
            "manifest path",
        )
        last = path
        require(tag in (0, 1, 2), "tag")
        if role == 0:
            source_path(path)
        extra = 72 if tag == 1 else 8 + len(payload) if tag == 2 else 0
        if tag == 2:
            require(len(payload) <= TEXT, "link cap")
        n += 8 + len(path) + 1 + extra
        require(n <= LIMIT, "manifest size cap")
    return n


def regular_hash(path):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as f:
        before = os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode), "nonregular hash")
        h = hashlib.sha256()
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
        after = os.fstat(f.fileno())
        require(
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "file changed",
        )
        return h.hexdigest()


def hash_inventory(role, rows, root=None):
    preflight_manifest(role, rows)
    out = []
    for path, tag, payload in rows:
        p = os.path.join(os.fsencode(root), path) if root is not None else path
        if tag == 0:
            require(not os.path.lexists(p), "absent became present")
        if tag == 2:
            require(os.readlink(p) == payload, "link changed")
        out.append((path, tag, regular_hash(p) if tag == 1 else payload))
    return manifest(role, out)


def config_inventory(root, cargo_home):
    rows = {}
    for p in [root, *Path(root).parents]:
        for name in ("config", "config.toml"):
            rows[os.fsencode(p / ".cargo" / name)] = (
                os.fsencode(p / ".cargo" / name),
                0,
                None,
            )
    for name in ("config", "config.toml"):
        rows[os.fsencode(Path(cargo_home) / name)] = (
            os.fsencode(Path(cargo_home) / name),
            0,
            None,
        )
    for p in rows:
        require(not os.path.lexists(p), "Cargo config present")
    result = sorted(rows.values())
    preflight_manifest(1, result)
    return result


def build_environment(o):
    require(type(o["jobs"]) is int and 1 <= o["jobs"] <= 4, "jobs1..4")
    for k in (
        "root",
        "cargo",
        "rustc",
        "python",
        "git",
        "linker",
        "cargo_home",
        "target",
        "home",
        "tmp",
        "request",
    ):
        require(
            os.path.isabs(o[k]) and "\0" not in o[k] and len(os.fsencode(o[k])) <= TEXT,
            "absolute option path",
        )
    require("\x1f" not in o["linker"], "linker unit separator")
    return dict(
        HOME=o["home"],
        CARGO_HOME=o["cargo_home"],
        CARGO_TARGET_DIR=o["target"],
        RUSTC=o["rustc"],
        PATH=os.pathsep.join(
            sorted(
                {
                    os.path.dirname(o[k])
                    for k in ("cargo", "rustc", "python", "git", "linker")
                },
                key=os.fsencode,
            )
        ),
        LANG="C",
        LC_ALL="C",
        TZ="UTC",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_OPTIONAL_LOCKS="0",
        TMPDIR=o["tmp"],
        CARGO_BUILD_JOBS=str(o["jobs"]),
        CARGO_INCREMENTAL="0",
        CARGO_NET_OFFLINE="true",
        CARGO_ENCODED_RUSTFLAGS="-Clinker=" + o["linker"],
        HEKIT_STUDY_CAPSULE_REQUEST=o["request"],
        HEKIT_STUDY_PYTHON=o["python"],
    )


def recipe_argv(o):
    meta = [
        o["cargo"],
        "metadata",
        "--format-version",
        "1",
        "--locked",
        "--offline",
        "--no-default-features",
        "--filter-platform",
        TARGET,
        "--manifest-path",
        o["root"] + "/Cargo.toml",
    ]
    build = [
        o["cargo"],
        "build",
        "--locked",
        "--offline",
        "--release",
        "-p",
        "ckks-train-bench",
        "--no-default-features",
        "--bin",
        "study_reference",
        "--target",
        TARGET,
        "--target-dir",
        o["target"],
        "-j",
        str(o["jobs"]),
        "--message-format=json-render-diagnostics",
    ]
    return meta, build


def packages(metadata, root, cargo_home):
    require(metadata.get("version") == 1, "metadata version")
    entries = metadata.get("packages")
    require(isinstance(entries, list) and len(entries) <= COUNT, "package cap")
    members = metadata.get("workspace_members")
    require(isinstance(members, list), "workspace members")
    result = []
    for p in entries:
        ident = p["id"]
        path = Path(p["manifest_path"])
        source = p["source"]
        require(isinstance(ident, str) and len(ident.encode()) <= TEXT, "package id")
        canonical(path)
        require(path.name == "Cargo.toml", "manifest name")
        if source is None:
            require(
                ident in members and contained(path, root), "outside workspace package"
            )
            tag = 0
            source = ""
        else:
            require(
                isinstance(source, str)
                and source.startswith("registry+")
                and contained(path, Path(cargo_home) / "registry/src"),
                "unsupported dependency source",
            )
            relative = path.relative_to(Path(cargo_home) / "registry/src")
            require(len(relative.parts) == 3, "registry package root")
            tag = 1
            require(
                (path.parent / ".cargo-checksum.json").is_file(),
                "missing package cache checksum",
            )
        require(len(source.encode()) <= TEXT, "source cap")
        result.append((ident, os.fsencode(path), tag, source))
    return unique_sorted(result, lambda x: x[0].encode())


def artifacts(messages, inventory, target):
    mapping = {x[0]: x for x in inventory}
    rows = set()
    for m in messages:
        if m.get("reason") != "compiler-artifact":
            continue
        ident = m["package_id"]
        require(ident in mapping, "unknown emitted package")
        pkg = mapping[ident]
        path = os.fsencode(m["manifest_path"])
        require(path == pkg[1], "emitted manifest mismatch")
        t = m["target"]
        src = os.fsencode(t["src_path"])
        require(
            len(src) <= TEXT and len(t["name"].encode()) <= TEXT,
            "artifact text/path cap",
        )
        require(
            contained(Path(os.fsdecode(src)).resolve(), Path(os.fsdecode(path)).parent),
            "target source escape",
        )
        kinds = t["kind"]
        require(
            isinstance(kinds, list)
            and len(kinds) <= COUNT
            and all(isinstance(k, str) and len(k.encode()) <= TEXT for k in kinds),
            "target kinds",
        )
        exe = m["executable"]
        outputs = m["filenames"]
        require(isinstance(outputs, list) and len(outputs) <= COUNT, "outputs cap")
        for p in outputs + ([] if exe is None else [exe]):
            require(
                len(os.fsencode(p)) <= TEXT
                and os.path.isabs(p)
                and contained(Path(p).resolve(), target),
                "output escape",
            )
        rows.add(
            (
                ident,
                path,
                t["name"],
                src,
                tuple(unique_sorted(kinds)),
                None if exe is None else os.fsencode(exe),
            )
        )
        require(len(rows) <= COUNT, "artifact count")
    return sorted(rows, key=emitted_key)


def bounded_run(argv, cwd, env, cap=LIMIT):
    """Bound stdout while streaming; never accumulate stderr or use a shell."""
    with subprocess.Popen(
        argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE
    ) as p:
        try:
            data = p.stdout.read(cap + 1)
            require(len(data) <= cap, "command output cap")
            require(p.wait() == 0, "command failed")
            return data
        except BaseException:
            p.kill()
            p.wait()
            raise


def git_state(root, git, env):
    prefix = [
        git,
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(root),
    ]
    require(
        bounded_run(prefix + ["rev-parse", "--show-toplevel"], root, env).rstrip(b"\n")
        == os.fsencode(root),
        "Git root",
    )
    commit = (
        bounded_run(prefix + ["rev-parse", "HEAD"], root, env).decode("ascii").strip()
    )
    require(
        len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
        "Git revision",
    )
    bounded_run(prefix + ["diff", "--no-ext-diff", "--quiet", "HEAD", "--"], root, env)
    require(
        not bounded_run(
            prefix + ["ls-files", "--others", "--exclude-standard", "-z"], root, env
        ),
        "untracked checkout",
    )
    return commit, prefix


def source_inventory(root, git, env, designs=DESIGNS):
    root = canonical(root)
    commit, prefix = git_state(root, git, env)
    tracked = bounded_run(prefix + ["ls-files", "--stage", "-z"], root, env)
    entries = {}
    for entry in tracked.split(b"\0"):
        if not entry:
            continue
        facts, path = entry.split(b"\t", 1)
        mode, _, stage = facts.split(b" ")
        require(stage == b"0" and mode != b"160000", "unmerged/submodule")
        entries[path] = mode
    roots = [root / x for x in ("crates", "scripts", "tools")]
    fixed = [b"Cargo.toml", b"Cargo.lock"] + [os.fsencode(x) for x in designs]
    fixed += [
        os.fsencode(x)
        for x in ("build.rs", "rust-toolchain", "rust-toolchain.toml")
        if os.path.lexists(root / x)
    ]
    selected = {
        p
        for p in entries
        if p.startswith((b"crates/", b"scripts/", b"tools/")) or p in fixed
    }
    require(set(fixed) <= selected, "required source missing")
    rows = []
    for path, tag, payload in enumerate_tree(
        root, False, roots + [root / os.fsdecode(x) for x in fixed], max_records=512
    ):
        rel = os.fsencode(Path(os.fsdecode(path)).relative_to(root))
        source_path(rel)
        rows.append((rel, tag, payload))
    require({r[0] for r in rows} == selected, "source tracked/filesystem mismatch")
    preflight_manifest(0, rows)
    return commit, sorted(rows)


def toolchain_inventory(sysroot, host, target, tools):
    root = canonical(sysroot)
    require(host == target == TARGET, "native Linux target")
    require(
        Path(tools["rustc"]) == root / "bin/rustc"
        and Path(tools["cargo"]) == root / "bin/cargo",
        "direct toolchain paths",
    )
    paths = [root / "bin/rustc", root / "bin/cargo"]
    lib = root / "lib"
    with os.scandir(lib) as entries:
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                require(e.name == "rustlib", "unsupported top-level lib directory")
            else:
                paths.append(lib / e.name)
    for triple in sorted({host, target}):
        for part in ("lib", "bin"):
            p = lib / "rustlib" / triple / part
            require(p.is_dir(), "missing native toolchain tree")
            paths.append(p)
    rows = enumerate_tree(root, True, paths)
    for k in ("python", "git", "linker"):
        p = canonical(tools[k])
        rows += enumerate_tree(p.parent, True, [p])
    result = sorted({r[0]: r for r in rows}.values())
    preflight_manifest(3, result)
    return result


def validate_recipe_capsule(c):
    require(c["recipe"] == RECIPE and c["algorithm"] == ALGORITHM, "recipe version")
    require(
        c["target"] == c["host"] == TARGET
        and c["profile"] == "release"
        and c["opt"] == "3"
        and c["debug"] == 0
        and not c["features"],
        "compile facts",
    )
    require(
        len(c["commit"]) == 40 and all(x in "0123456789abcdef" for x in c["commit"]),
        "commit",
    )


def header_size(domain):
    return 8 + len("hekit-study-reference-encoding-v1") + 8 + len(DOMAINS[domain])


def blob_size(b):
    return 8 + len(b)


def text_size(t):
    return blob_size(t.encode())


def argv_size(v):
    require(len(v) <= 256 and all(len(b) <= TEXT for b in v), "argv cap")
    return 8 + sum(blob_size(b) for b in v)


def preflight_capsule(c):
    for key in (
        "recipe",
        "algorithm",
        "commit",
        "cargo",
        "target",
        "host",
        "profile",
        "opt",
        "version",
    ):
        require(
            isinstance(c[key], str) and len(c[key].encode()) <= TEXT, "capsule text cap"
        )
    require(
        len(c["compiler"].encode()) <= 16384 and len(c["cwd"]) <= TEXT,
        "compiler/cwd cap",
    )
    require(
        len(c["features"]) <= 64
        and all(len(x.encode()) <= TEXT for x in c["features"]),
        "feature cap",
    )
    require(len(c["packages"]) <= COUNT, "package cap")
    for ident, path, tag, source in c["packages"]:
        require(
            len(ident.encode()) <= TEXT
            and len(path) <= TEXT
            and len(source.encode()) <= TEXT
            and tag in (0, 1),
            "package field cap",
        )
    # All digest payloads are 72 bytes, irrespective of unknown content.
    n = (
        header_size("capsule")
        + text_size(c["recipe"])
        + text_size(c["algorithm"])
        + 6 * 72
        + text_size(c["commit"])
        + text_size(c["compiler"])
        + text_size(c["cargo"])
    )
    n += (
        sum(text_size(c[k]) for k in ("target", "host", "profile", "opt", "version"))
        + 1
        + 8
        + sum(text_size(x) for x in c["features"])
        + 72
        + 8
    )
    require(len(c["packages"]) <= COUNT, "package cap")
    n += sum(
        text_size(i) + blob_size(p) + 1 + text_size(src)
        for i, p, _, src in c["packages"]
    )
    n += (
        argv_size(c["metadata_argv"]) + argv_size(c["build_argv"]) + blob_size(c["cwd"])
    )
    _checked_size_add(0, n)
    require(n <= CAPSULE_LIMIT, "capsule preflight cap")
    return n


def preflight_receipt(capsule_size, manifest_sizes, env_size, exe, emitted, argv):
    require(
        0 <= capsule_size <= CAPSULE_LIMIT
        and 0 <= env_size <= LIMIT
        and all(0 <= x <= LIMIT for x in manifest_sizes),
        "embedded byte caps",
    )
    require(len(exe) <= TEXT, "executable path cap")
    require(len(manifest_sizes) == 4 and len(emitted) <= COUNT, "receipt counts")
    n = (
        header_size("receipt")
        + 8
        + capsule_size
        + sum(8 + x for x in manifest_sizes)
        + 8
        + env_size
        + blob_size(exe)
        + 7 * 72
        + 8
        + argv_size(argv)
    )
    for ident, path, name, src, kinds, executable in emitted:
        require(len(kinds) <= COUNT, "kind cap")
        for b in [
            ident.encode(),
            path,
            name.encode(),
            src,
            *[x.encode() for x in kinds],
            *([] if executable is None else [executable]),
        ]:
            require(len(b) <= TEXT, "artifact field cap")
        n += (
            text_size(ident)
            + blob_size(path)
            + text_size(name)
            + blob_size(src)
            + 8
            + sum(text_size(x) for x in kinds)
            + 1
            + (0 if executable is None else blob_size(executable))
        )
    _checked_size_add(0, n)
    require(n <= LIMIT, "receipt preflight cap")
    return n


def env_size(env):
    require(len(env) <= COUNT, "environment count")
    n = header_size("environment") + 8
    for key, value in env.items():
        require(
            len(key.encode()) <= TEXT and len(os.fsencode(value)) <= TEXT,
            "environment field cap",
        )
        n += text_size(key) + 1 + blob_size(os.fsencode(value))
    require(n <= LIMIT, "environment cap")
    return n


def dependency_inventory(inventory):
    rows = {}
    for _, manifest_path, tag, _ in inventory:
        if tag == 1:
            root = Path(os.fsdecode(manifest_path)).parent
            for row in enumerate_tree(root, True):
                rows[row[0]] = row
            require(len(rows) <= COUNT, "dependency count")
    out = sorted(rows.values())
    preflight_manifest(2, out)
    return out


def check_lock_checksums(root, inventory):
    # Python >=3.11 is an explicit prerequisite; TOML parses the actual lock.
    import tomllib

    lock = tomllib.loads(read_regular(Path(root) / "Cargo.lock", LIMIT).decode("utf8"))
    checks = {
        (p["name"], p["version"], p.get("source")): p.get("checksum")
        for p in lock["package"]
    }
    for _, manifest_path, tag, source in inventory:
        if tag != 1:
            continue
        folder = Path(os.fsdecode(manifest_path)).parent
        checksum = json.loads(read_regular(folder / ".cargo-checksum.json", LIMIT))
        # Match registry source plus package checksum, not a guessed folder suffix.
        package_checksum = checksum.get("package")
        valid_hash(package_checksum)
        require(
            any(
                src == source and chk == package_checksum
                for (_, _, src), chk in checks.items()
            ),
            "lock/cache package checksum",
        )


def read_regular(path, cap=LIMIT):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as f:
        before = os.fstat(f.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= cap,
            "bounded regular file",
        )
        data = f.read(cap + 1)
        after = os.fstat(f.fileno())
        require(
            len(data) == before.st_size
            and len(data) <= cap
            and (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "file changed",
        )
        return data


def exclusive_write(path, data):
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def verify_runtime_snapshot(root, receipt_bytes, git):
    r = decode("receipt", receipt_bytes)
    c = decode("capsule", r["capsule"])
    _validate_linkage(r, c)
    require(os.fsencode(root) == c["cwd"], "source root mismatch")
    recipe = _closed_capsule(c)
    options = _helper_options(r, c, recipe)
    require(os.fsencode(git) == options[b"--git"], "snapshot selected Git")
    env = dict(
        (k, os.fsdecode(v))
        for k, v in decode("environment", r["environment"])
        if v is not None
    )
    commit, rows = source_inventory(root, git, env)
    configs = config_inventory(Path(root), Path(env["CARGO_HOME"]))
    require(commit == c["commit"], "source commit changed")
    sizes = [preflight_manifest(0, rows), preflight_manifest(1, configs)]
    require(sum(sizes) <= LIMIT, "runtime inventory cap")
    require(
        hash_inventory(0, rows, root) == r["files"][0]
        and hash_inventory(1, configs) == r["files"][1],
        "runtime source/config changed",
    )
    return c["manifests"][0]


def reject_caller_environment(env):
    for key in env:
        require(
            not key.startswith(("CARGO_", "RUST", "GIT_", "LD_", "DYLD_"))
            and key
            not in (
                "CUDARC_CUDA_VERSION",
                "CUDA_HOME",
                "CUDA_PATH",
                "CUDA_ROOT",
                "CUDA_TOOLKIT_ROOT_DIR",
                "CUDNN_LIB",
            ),
            "forbidden caller compiler/loader/Git override",
        )


def unchanged_metadata(before, after):
    require(
        len(before) <= LIMIT and len(after) <= LIMIT and before == after,
        "post metadata changed",
    )


def fresh_destination(p):
    p = Path(p)
    require(
        p.is_absolute()
        and p.parent == p.parent.resolve(strict=True)
        and not os.path.lexists(p),
        "fresh destination",
    )


# Exact administrative recipes. A classification is not a build/context capability.
from dataclasses import dataclass


@dataclass(frozen=True)
class _Recipe:
    name: str
    algorithm: str
    role: int
    features: tuple
    feature: str | None
    binary: str
    helper: str
    subcommand: str


_REFERENCE = _Recipe(
    RECIPE,
    ALGORITHM,
    0,
    (),
    None,
    "study_reference",
    "scripts/study_reference_build.py",
    "build",
)
_TRAIN = tuple(
    _Recipe(
        "study-train-" + mode + "-linux-clean-v1",
        "study-train-session-v1",
        1,
        features,
        feature,
        "study_train",
        "scripts/study_train_build.py",
        mode,
    )
    for mode, features, feature in (
        ("cpu-serial", (), None),
        ("cpu-rayon", ("parallel",), "parallel"),
        ("cuda-host", ("cuda", "parallel"), "cuda"),
    )
)
_RECIPES = (_REFERENCE, *_TRAIN)
DOMAINS["approval_v2"] = "hekit-study-local-approval-v2"


def _classify(recipe, algorithm):
    found = [r for r in _RECIPES if r.name == recipe]
    require(len(found) == 1, "unsupported recipe")
    require(found[0].algorithm == algorithm, "unsupported algorithm")
    return found[0]


def _closed_capsule(c):
    recipe = _classify(c["recipe"], c["algorithm"])
    require(
        c["target"] == c["host"] == TARGET
        and c["profile"] == "release"
        and c["opt"] == "3"
        and c["debug"] == 0
        and tuple(c["features"]) == recipe.features,
        "compile facts",
    )
    require(
        [line[6:] for line in c["compiler"].splitlines() if line.startswith("host: ")]
        == [TARGET],
        "compiler host",
    )
    require(
        len(c["commit"]) == 40 and all(x in "0123456789abcdef" for x in c["commit"]),
        "commit",
    )
    return recipe


def _recipe_argv(o, recipe):
    require(recipe in _RECIPES, "closed recipe")
    meta, build = recipe_argv(o)
    build[9] = recipe.binary
    if recipe.feature:
        meta[7:7] = ["--features", "ckks-train-bench/" + recipe.feature]
        build[8:8] = ["--features", recipe.feature]
    return meta, build


def approval_v2(a):
    require(len(a["designs"]) <= 32 and len(a["builds"]) <= 64, "approval caps")
    w = Writer("approval_v2")
    w.t(a["id"])
    designs = unique_sorted(a["designs"], lambda x: x[0].encode())
    w.count(designs, 32)
    for path, h in designs:
        w.t(path)
        w.h(h)
    w.t(a["commit"])
    w.h(a["source"])
    builds = unique_sorted(a["builds"], lambda x: x[3])
    w.count(builds, 64)
    for role, recipe, algorithm, r, c, e in builds:
        require(role in (0, 1), "approval role")
        w.q(role)
        w.t(recipe)
        w.t(algorithm)
        for h in (r, c, e):
            w.h(h)
    return w.done()


def _approval_policy(a, version):
    require(
        a["id"]
        and len(a["commit"]) == 40
        and all(x in "0123456789abcdef" for x in a["commit"]),
        "approval policy",
    )
    require([p for p, _ in a["designs"]] == sorted(DESIGNS), "approved design set")
    if version == 1:
        require(
            a["recipe"] == RECIPE and a["algorithm"] == ALGORITHM,
            "approval v1 Reference only",
        )
    else:
        for role, recipe, algorithm, *_ in a["builds"]:
            require(
                role == _classify(recipe, algorithm).role,
                "approval role/recipe mismatch",
            )


def _approved_receipt(a, version, data, r, c):
    require(
        c["commit"] == a["commit"] and c["manifests"][0] == a["source"],
        "approved source/version mismatch",
    )
    triple = (sha(data), sha(r["capsule"]), r["executable_hash"])
    if version == 1:
        require(
            c["recipe"] == a["recipe"]
            and c["algorithm"] == a["algorithm"]
            and triple in a["builds"],
            "receipt/capsule/executable not approved",
        )
    else:
        recipe = _classify(c["recipe"], c["algorithm"])
        require(
            (recipe.role, c["recipe"], c["algorithm"], *triple) in a["builds"],
            "receipt/capsule/executable not approved",
        )


def _receipt_manifest_links(r, c):
    for role, data in enumerate(r["files"]):
        m = decode("manifest", data)
        require(
            m["role"] == role and sha(data) == c["manifests"][role],
            "embedded manifest link",
        )
    require(sha(r["environment"]) == c["environment"], "environment link")


def _receipt_post(r, c):
    _receipt_manifest_links(r, c)
    require(
        r["post"] == c["manifests"] + [c["environment"]]
        and r["metadata"] == c["metadata"],
        "pre/post mismatch",
    )


def _claim_path(raw):
    """Check untrusted raw Unix bytes before Path can normalize components."""
    require(
        isinstance(raw, bytes)
        and len(raw) <= TEXT
        and raw.startswith(b"/")
        and b"\0" not in raw
        and not any(part in (b".", b"..") for part in raw.split(b"/")),
        "raw lexical path",
    )
    return Path(os.fsdecode(raw))


def _helper_options(r, c, recipe):
    argv = r["argv"]
    root = _claim_path(c["cwd"])
    require(
        len(argv) >= 2
        and argv[:2]
        == [
            os.fsencode(root / recipe.helper),
            recipe.subcommand.encode(),
        ],
        "fixed helper dispatch",
    )
    require((len(argv) - 2) % 2 == 0, "helper option pairs")
    options = {}
    for k, v in zip(argv[2::2], argv[3::2]):
        require(k not in options, "duplicate helper option")
        options[k] = v
    required = {
        b"--" + k.encode()
        for k in (
            "root",
            "cargo",
            "rustc",
            "python",
            "git",
            "linker",
            "cargo-home",
            "target",
            "home",
            "tmp",
            "request",
            "receipt",
            "jobs",
        )
    }
    optional = {b"--rust-toolchain-sha256", b"--rust-toolchain-toml-sha256"}
    require(
        required <= options.keys() and options.keys() <= required | optional,
        "unsupported/missing helper option",
    )
    for k, v in options.items():
        if k in optional:
            valid_hash(v.decode())
        elif k != b"--jobs":
            _claim_path(v)
        else:
            require(b"\0" not in v, "helper NUL")
    return options


def _recipe_vectors(r, c, env, recipe):
    root = _claim_path(c["cwd"])
    target = _claim_path(env["CARGO_TARGET_DIR"])
    options = _helper_options(r, c, recipe)
    o = {
        "cargo": os.fsdecode(options[b"--cargo"]),
        "root": os.fsdecode(c["cwd"]),
        "target": os.fsdecode(env["CARGO_TARGET_DIR"]),
        "jobs": os.fsdecode(env["CARGO_BUILD_JOBS"]),
    }
    meta, build = _recipe_argv(o, recipe)
    require(c["metadata_argv"] == list(map(os.fsencode, meta)), "metadata argv")
    require(c["build_argv"] == list(map(os.fsencode, build)), "build argv")
    selected = [
        a
        for a in r["emitted"]
        if a[2] == recipe.binary and a[4] == ("bin",) and a[5] == r["executable"]
    ]
    require(
        len(selected) == 1
        and selected[0][1] == os.fsencode(root / "crates/ckks-train-bench/Cargo.toml")
        and selected[0][3]
        == os.fsencode(
            root / ("crates/ckks-train-bench/src/bin/" + recipe.binary + ".rs")
        ),
        "unique writer binary linkage",
    )
    require(
        r["executable"] == os.fsencode(target / TARGET / "release" / recipe.binary),
        "receipt executable path",
    )
    return options


def _validate_linkage(r, c):
    recipe = _closed_capsule(c)
    _receipt_post(r, c)
    env = dict(decode("environment", r["environment"]))
    options = _recipe_vectors(r, c, env, recipe)
    o = {
        k: os.fsdecode(options[("--" + k.replace("_", "-")).encode()])
        for k in (
            "root",
            "cargo",
            "rustc",
            "python",
            "git",
            "linker",
            "cargo_home",
            "target",
            "home",
            "tmp",
            "request",
        )
    }
    require(options[b"--jobs"] in (b"1", b"2", b"3", b"4"), "recipe jobs")
    o["jobs"] = int(options[b"--jobs"])
    require(
        env == {k: os.fsencode(v) for k, v in build_environment(o).items()}
        and c["cwd"] == options[b"--root"],
        "recipe helper/environment linkage",
    )
    files = [decode("manifest", data) for data in r["files"]]
    source = dict((p, (tag, value)) for p, tag, value in files[0]["records"])
    for m in files:
        for path, tag, _ in m["records"]:
            require(path and b"\0" not in path, "manifest path")
            if m["role"] == 0:
                source_path(path)
                require(tag == 1, "source manifest policy")
            else:
                require(
                    path.startswith(b"/")
                    and not any(x in (b".", b"..") for x in path.split(b"/")),
                    "absolute lexical path",
                )
            if m["role"] == 1:
                require(tag == 0, "config must be absent")
    for name in (
        "scripts/study_reference_build.py",
        "scripts/study_reference_build_support.py",
        recipe.helper,
        "crates/ckks-train-bench/build.rs",
    ):
        require(
            source.get(name.encode(), (None,))[0] == 1, "build helper source uncovered"
        )
    require(
        source.get(
            ("crates/ckks-train-bench/src/bin/" + recipe.binary + ".rs").encode(),
            (None,),
        )[0]
        == 1,
        "binary source uncovered",
    )
    for name, option in (
        ("rust-toolchain", b"--rust-toolchain-sha256"),
        ("rust-toolchain.toml", b"--rust-toolchain-toml-sha256"),
    ):
        present = source.get(name.encode())
        pin = options.get(option)
        require(
            (present is None and pin is None)
            or (
                present is not None and pin is not None and present == (1, pin.decode())
            ),
            "toolchain source pin linkage",
        )
    for key in (b"--target", b"--request", b"--receipt"):
        require(
            not _claim_path(options[key]).is_relative_to(_claim_path(c["cwd"])),
            "helper output inside source",
        )
    require(source.get(b"Cargo.lock") == (1, c["lock"]), "Cargo.lock source linkage")
    tools = {p for p, tag, _ in files[3]["records"] if tag == 1}
    require(
        all(
            options[("--" + k).encode()] in tools
            for k in ("cargo", "rustc", "python", "git", "linker")
        ),
        "tool absent from inventory",
    )
    dependencies = {p for p, tag, _ in files[2]["records"] if tag == 1}
    packages_by_id = {p[0]: p for p in c["packages"]}
    root = _claim_path(c["cwd"])
    target = _claim_path(env["CARGO_TARGET_DIR"])
    for _, manifest, tag, package_source in c["packages"]:
        path = _claim_path(manifest)
        require(
            path.is_absolute() and path.name == "Cargo.toml", "package manifest path"
        )
        if tag == 0:
            require(
                package_source == "" and path.is_relative_to(root), "workspace source"
            )
            require(
                source.get(os.fsencode(path.relative_to(root)), (None,))[0] == 1,
                "workspace package source uncovered",
            )
        else:
            registry = _claim_path(options[b"--cargo-home"]) / "registry/src"
            require(
                package_source.startswith("registry+")
                and path.is_relative_to(registry)
                and len(path.relative_to(registry).parts) == 3,
                "registry package root",
            )
            require(
                manifest in dependencies
                and os.fsencode(path.with_name(".cargo-checksum.json")) in dependencies,
                "registry package source uncovered",
            )
    for ident, manifest, _, src, _, executable in r["emitted"]:
        manifest_path = _claim_path(manifest)
        emitted_source_path = _claim_path(src)
        executable_path = None if executable is None else _claim_path(executable)
        require(ident in packages_by_id, "unknown emitted package")
        require(
            manifest == packages_by_id[ident][1]
            and emitted_source_path.is_relative_to(manifest_path.parent),
            "emitted source linkage",
        )
        require(
            executable_path is None or executable_path.is_relative_to(target),
            "emitted executable escape",
        )
    return recipe


def _resolved_features(metadata, recipe):
    """Actual Linux-filtered Cargo metadata; normal selected edges, not presence."""
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    require(
        len(nodes) == len(metadata["resolve"]["nodes"]), "duplicate resolution node"
    )
    packages_by_id = {p["id"]: p for p in metadata["packages"]}

    def package(name):
        found = [
            p for p in metadata["packages"] if p["name"] == name and p["id"] in nodes
        ]
        require(len(found) == 1, "unique resolved package: " + name)
        return found[0]

    def facts(name, features, version=None):
        p = package(name)
        require(
            nodes[p["id"]]["features"] == list(features), "resolved features: " + name
        )
        require(version is None or p["version"] == version, "resolved version: " + name)
        return p["id"]

    def edges(ident):
        return {
            d["name"]: d["pkg"]
            for d in nodes[ident]["deps"]
            if any(k["kind"] is None for k in d["dep_kinds"])
        }

    bench = facts("ckks-train-bench", recipe.features)
    ckks = facts("ckks-train-ckks", ("parallel",) if recipe.features else ())
    require(edges(bench).get("ckks_train_ckks") == ckks, "bench CKKS edge")
    relevant = {bench, ckks}
    if recipe.features:
        rayon = facts("rayon", (), "1.12.0")
        core = facts("rayon-core", (), "1.13.0")
        require(
            edges(bench).get("rayon") == edges(ckks).get("rayon") == rayon
            and edges(rayon).get("rayon_core") == core,
            "shared Rayon edges",
        )
        relevant.update((rayon, core))
    else:
        require(
            "rayon" not in edges(bench) and "rayon" not in edges(ckks),
            "serial Rayon edge",
        )
    if recipe.subcommand == "cuda-host":
        cuda = facts("ckks-train-cuda", ("cuda",))
        cudarc = facts(
            "cudarc",
            ("cuda-13010", "driver", "dynamic-loading", "nvrtc", "std"),
            "0.19.0",
        )
        require(
            edges(bench).get("ckks_train_cuda") == cuda
            and edges(cuda).get("cudarc") == cudarc
            and edges(cuda).get("ckks_train_ckks") == ckks,
            "CUDA dependency edges",
        )
        relevant.update((cuda, cudarc))
    else:
        require("ckks_train_cuda" not in edges(bench), "CPU CUDA edge")
    require(relevant <= packages_by_id.keys(), "resolved package mapping")
    # Workspace metadata conservatively includes dev-feature unification (e.g.
    # serde derive). Only the closed arithmetic/runtime feature owners have an
    # exact production policy here; do not equate metadata-only dev features
    # with compiler features for unrelated dependencies.
    return {ident: nodes[ident]["features"] for ident in relevant}


def _actual_features(recipe, features):
    require(
        features == sorted(set(features)) and tuple(features) == recipe.features,
        "actual Cargo features",
    )


def _design_digest(expected, actual):
    require(expected == actual, "design bytes changed")


def _checked_size_add(a, b):
    require(
        type(a) is int
        and type(b) is int
        and 0 <= a <= 2**64 - 1
        and 0 <= b <= 2**64 - 1 - a,
        "size arithmetic",
    )
    return a + b


def _emitted_feature_facts(message, resolved, bench, recipe):
    ident = message.get("package_id")
    if ident in resolved:
        require(message.get("features") == resolved[ident], "emitted resolved features")
    if ident == bench and message.get("target", {}).get("name") == recipe.binary:
        profile = message.get("profile", {})
        require(
            profile.get("opt_level") == "3"
            and profile.get("debuginfo") in (None, 0)
            and profile.get("test") is False,
            "emitted compile profile",
        )
