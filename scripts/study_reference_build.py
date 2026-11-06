#!/usr/bin/env python3
"""Explicit offline native Linux study-reference build; never runs the binary.

Use Python 3.11+ by an explicitly selected canonical path. `capture` is called
only by bench build.rs; `snapshot` is a read-only runtime source recheck.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import study_reference_build_support as s


def encode_request(c, env, source_paths):
    return json.dumps(
        dict(
            capsule=s.capsule(c).hex(),
            environment=s.environment(
                [(k, os.fsencode(v)) for k, v in env.items()]
            ).hex(),
            source_paths=[os.fsdecode(x) for x in source_paths],
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def capture(args):
    request = json.loads(s.read_regular(args.request))
    c = s.decode("capsule", bytes.fromhex(request["capsule"]))
    recipe = s._closed_capsule(c)
    s._actual_features(recipe, args.feature)
    observed = []
    for key, value in os.environ.items():
        if key.startswith("CARGO_FEATURE_"):
            s.require(
                key in ("CARGO_FEATURE_PARALLEL", "CARGO_FEATURE_CUDA")
                and value == "1",
                "unknown Cargo feature fact",
            )
            observed.append(key[len("CARGO_FEATURE_") :].lower())
    s.require(sorted(observed) == args.feature, "actual Cargo feature forwarding")
    env = s.decode("environment", bytes.fromhex(request["environment"]))
    s.require(
        s.sha(bytes.fromhex(request["environment"])) == c["environment"],
        "request environment link",
    )
    # Cargo's generated variables are checked independently by build.rs.
    for k, v in env:
        s.require(
            os.environb.get(k.encode()) == v, "capture caller environment changed"
        )
    for key, actual in [
        ("compiler", args.compiler),
        ("target", args.target),
        ("host", args.host),
        ("profile", args.profile),
        ("opt", args.opt),
        ("version", args.version),
    ]:
        s.require(c[key] == actual, "compile fact mismatch: " + key)
    s.require(args.debug == "false" and c["debug"] == 0, "compile debug")
    for path in request["source_paths"]:
        s.require(
            isinstance(path, str)
            and len(os.fsencode(path)) <= s.TEXT
            and "\n" not in path
            and "\r" not in path,
            "Cargo rerun path",
        )
    s.require(len(request["source_paths"]) <= s.COUNT, "Cargo rerun count")
    out = Path(args.outdir)
    s.exclusive_write(out / "study_capsule.bin", s.capsule(c))
    s.exclusive_write(
        out / "study_capsule.rs",
        b'pub(super) const EMBEDDED_CAPSULE: Option<&[u8]> = Some(include_bytes!(concat!(env!("OUT_DIR"), "/study_capsule.bin")));\n',
    )
    # Each candidate is explicit; Cargo accepts paths linewise, so reject LF/CR.
    for path in request["source_paths"]:
        s.require("\n" not in path and "\r" not in path, "Cargo rerun path newline")
        print("cargo:rerun-if-changed=" + path)


def build(args, recipe=s._REFERENCE):
    s.require(recipe in s._RECIPES, "closed recipe")
    s.require(sys.version_info >= (3, 11), "Python3.11 required")
    s.reject_caller_environment(os.environ)
    o = vars(args).copy()
    o.pop("mode")
    o.pop("rust_toolchain_sha256")
    o.pop("rust_toolchain_toml_sha256")
    # No option provides overrides/features/extra Cargo argv.
    for k in (
        "root",
        "cargo",
        "rustc",
        "python",
        "git",
        "linker",
        "cargo_home",
        "home",
        "tmp",
    ):
        s.canonical(o[k])
    s.require(
        Path(sys.executable).resolve() == Path(o["python"]), "selected Python differs"
    )
    for k in ("cargo", "rustc", "python", "git", "linker"):
        p = Path(o[k])
        s.require(
            p.is_file() and os.access(p, os.X_OK) and p.name != "rustup",
            "direct executable",
        )
    for k in ("home", "tmp"):
        st = Path(o[k]).stat()
        s.require(
            stat_mode(st) == 0o700 and st.st_uid == os.geteuid(),
            "private controlled directory",
        )
    for k in ("target", "request", "receipt"):
        p = Path(o[k])
        s.fresh_destination(p)
        s.require(
            not s.contained(p, Path(o["root"])), "build output outside source checkout"
        )
    env = s.build_environment(o)
    meta_argv, build_argv = s._recipe_argv(o, recipe)
    # Reject Cargo configuration and dirty/extra source before Cargo discovery.
    config = s.config_inventory(Path(o["root"]), Path(o["cargo_home"]))
    commit, source = s.source_inventory(o["root"], o["git"], env)
    # Exclusive target creation: never reuse an existing Cargo cache.
    os.mkdir(o["target"], 0o700)
    compiler = s.bounded_run([o["rustc"], "-vV"], o["root"], env, 16384).decode()
    hosts = [line[6:] for line in compiler.splitlines() if line.startswith("host: ")]
    s.require(hosts == [s.TARGET], "native compiler host")
    sysroot = (
        s.bounded_run([o["rustc"], "--print", "sysroot"], o["root"], env, 4096)
        .decode()
        .strip()
    )
    cargo_v = s.bounded_run([o["cargo"], "-V"], o["root"], env, 4096).decode()
    metadata_bytes = s.bounded_run(meta_argv, o["root"], env)
    metadata = json.loads(metadata_bytes)
    resolved_features = s._resolved_features(metadata, recipe)
    inventory = s.packages(metadata, Path(o["root"]), Path(o["cargo_home"]))
    dependencies = s.dependency_inventory(inventory)
    toolchain = s.toolchain_inventory(sysroot, hosts[0], s.TARGET, o)
    inventories = [source, config, dependencies, toolchain]
    sizes = [s.preflight_manifest(role, rows) for role, rows in enumerate(inventories)]
    bench = [
        p
        for p in metadata["packages"]
        if p.get("name") == "ckks-train-bench" and p["source"] is None
    ]
    s.require(len(bench) == 1, "unique bench package")
    c = dict(
        recipe=recipe.name,
        algorithm=recipe.algorithm,
        manifests=["0" * 64] * 4,
        environment="0" * 64,
        lock="0" * 64,
        commit=commit,
        compiler=compiler,
        cargo=cargo_v,
        target=s.TARGET,
        host=s.TARGET,
        profile="release",
        opt="3",
        debug=0,
        features=list(recipe.features),
        version=bench[0]["version"],
        metadata="0" * 64,
        packages=inventory,
        metadata_argv=[os.fsencode(x) for x in meta_argv],
        build_argv=[os.fsencode(x) for x in build_argv],
        cwd=os.fsencode(o["root"]),
    )
    cap_size = s.preflight_capsule(c)
    env_size = s.env_size(env)
    argv = [os.fsencode(x) for x in sys.argv]
    expected_exe = os.fsencode(Path(o["target"]) / s.TARGET / "release" / recipe.binary)
    s.preflight_receipt(
        cap_size, sizes, env_size, expected_exe, [], argv
    )  # preliminary lower bound only
    # No helper content hash occurs above this complete metadata preflight.
    for filename, pin in [
        ("rust-toolchain", args.rust_toolchain_sha256),
        ("rust-toolchain.toml", args.rust_toolchain_toml_sha256),
    ]:
        p = Path(o["root"]) / filename
        if p.exists():
            s.require(
                pin is not None and s.regular_hash(p) == s.valid_hash(pin),
                "unapproved rust-toolchain bytes",
            )
        else:
            s.require(pin is None, "toolchain pin without file")
    s.check_lock_checksums(o["root"], inventory)
    files = [
        s.hash_inventory(role, rows, o["root"] if role == 0 else None)
        for role, rows in enumerate(inventories)
    ]
    environment = s.environment([(k, os.fsencode(v)) for k, v in env.items()])
    c.update(
        metadata=s.sha(metadata_bytes),
        manifests=[s.sha(x) for x in files],
        environment=s.sha(environment),
        lock=s.regular_hash(Path(o["root"]) / "Cargo.lock"),
    )
    request_bytes = encode_request(
        c,
        env,
        [os.fsencode(Path(o["root"]) / os.fsdecode(x[0])) for x in source]
        + [x[0] for x in config]
        + [os.fsencode(o["request"])],
    )
    s.exclusive_write(o["request"], request_bytes)
    emitted = []
    observed_feature_packages = set()
    finished = False
    capsule_outputs = []
    with subprocess.Popen(
        build_argv,
        cwd=o["root"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    ) as p:
        try:
            while True:
                line = p.stdout.readline(s.LIMIT + 1)
                if not line:
                    break
                s.require(len(line) <= s.LIMIT, "Cargo message cap")
                m = json.loads(line)
                if m.get("reason") == "build-finished":
                    s.require(
                        not finished and m.get("success") is True,
                        "unsuccessful/duplicate build-finished",
                    )
                    finished = True
                if (
                    m.get("reason") == "build-script-executed"
                    and m.get("package_id") == bench[0]["id"]
                ):
                    out = Path(m["out_dir"])
                    s.require(
                        out.is_absolute()
                        and s.contained(out.resolve(), Path(o["target"])),
                        "build script output escape",
                    )
                    capsule_outputs.append(out / "study_capsule.bin")
                    s.require(len(capsule_outputs) <= 1, "multiple bench build scripts")
                if m.get("reason") == "compiler-artifact":
                    s._emitted_feature_facts(
                        m, resolved_features, bench[0]["id"], recipe
                    )
                    if m.get("package_id") in resolved_features:
                        observed_feature_packages.add(m["package_id"])
                    # Normalize immediately; never accumulate full messages/logs.
                    emitted = s.unique_sorted(
                        list(
                            set(
                                emitted + s.artifacts([m], inventory, Path(o["target"]))
                            )
                        ),
                        s.emitted_key,
                    )
                    s.require(len(emitted) <= s.COUNT, "artifact cap")
                    s.preflight_receipt(
                        cap_size, sizes, env_size, expected_exe, emitted, argv
                    )
            s.require(p.wait() == 0 and finished, "Cargo build incomplete")
        except BaseException:
            p.kill()
            p.wait()
            raise
    s.require(
        set(resolved_features) <= observed_feature_packages,
        "missing emitted feature facts",
    )
    selected = [
        x
        for x in emitted
        if x[0] == bench[0]["id"]
        and x[2] == recipe.binary
        and x[4] == ("bin",)
        and x[5] is not None
    ]
    s.require(
        len(selected) == 1
        and selected[0][5] == expected_exe
        and selected[0][3]
        == os.fsencode(
            Path(o["root"])
            / ("crates/ckks-train-bench/src/bin/" + recipe.binary + ".rs")
        ),
        "unique binary linkage",
    )
    post_metadata = s.bounded_run(meta_argv, o["root"], env)
    s.unchanged_metadata(metadata_bytes, post_metadata)
    s.require(
        s.packages(json.loads(post_metadata), Path(o["root"]), Path(o["cargo_home"]))
        == inventory,
        "post package mapping changed",
    )
    post_commit, post_source = s.source_inventory(o["root"], o["git"], env)
    post_rows = [
        post_source,
        s.config_inventory(Path(o["root"]), Path(o["cargo_home"])),
        s.dependency_inventory(inventory),
        s.toolchain_inventory(sysroot, hosts[0], s.TARGET, o),
    ]
    post_sizes = [s.preflight_manifest(i, rows) for i, rows in enumerate(post_rows)]
    s.preflight_receipt(cap_size, post_sizes, env_size, expected_exe, emitted, argv)
    # Final exact receipt cap is checked before ANY postbuild content hashing.
    post_files = [
        s.hash_inventory(i, rows, o["root"] if i == 0 else None)
        for i, rows in enumerate(post_rows)
    ]
    s.require(
        post_commit == commit and post_files == files,
        "build source/config/dependency/toolchain changed",
    )
    s.require(
        s.environment([(k, os.fsencode(v)) for k, v in s.build_environment(o).items()])
        == environment,
        "build environment changed",
    )
    cap = s.capsule(c)
    s.require(
        s.read_regular(o["request"]) == request_bytes
        and len(capsule_outputs) == 1
        and s.read_regular(capsule_outputs[0], s.CAPSULE_LIMIT) == cap,
        "changed request/compiled capsule",
    )
    r = dict(
        capsule=cap,
        files=files,
        environment=environment,
        executable=expected_exe,
        executable_hash=s.regular_hash(expected_exe),
        post=[s.sha(x) for x in post_files] + [s.sha(environment)],
        metadata=s.sha(post_metadata),
        emitted=emitted,
        argv=argv,
    )
    s._validate_linkage(r, c)
    s.exclusive_write(o["receipt"], s.receipt(r))


def stat_mode(st):
    return st.st_mode & 0o7777


def build_options(b):
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
    ):
        b.add_argument("--" + k, required=True)
    b.add_argument("--jobs", type=int, required=True)
    b.add_argument("--rust-toolchain-sha256")
    b.add_argument("--rust-toolchain-toml-sha256")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="mode", required=True)
    b = subs.add_parser("build")
    build_options(b)
    c = subs.add_parser("capture")
    for k in (
        "request",
        "outdir",
        "compiler",
        "target",
        "host",
        "profile",
        "opt",
        "debug",
        "version",
    ):
        c.add_argument("--" + k, required=True)
    c.add_argument("--feature", action="append", default=[])
    h = subs.add_parser("hash-file")
    h.add_argument("--path", required=True)
    r = subs.add_parser("snapshot")
    for k in ("root", "receipt", "git"):
        r.add_argument("--" + k, required=True)
    a = p.parse_args()
    if a.mode == "build":
        build(a)
    elif a.mode == "capture":
        capture(a)
    elif a.mode == "hash-file":
        print(s.regular_hash(a.path))
    else:
        print(
            s.verify_runtime_snapshot(
                s.canonical(a.root), s.read_regular(a.receipt), a.git
            )
        )


if __name__ == "__main__":
    try:
        main()
    except (s.Invalid, OSError, ValueError, KeyError, TypeError) as e:
        print("study-reference build rejected: " + str(e), file=sys.stderr)
        sys.exit(1)
