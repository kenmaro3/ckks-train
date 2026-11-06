"""Frozen independent transport and private predicates; no approved contexts."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest import mock
import study_reference_build as build
import study_reference_build_support as s

FIX = (
    Path(__file__).resolve().parent.parent
    / "crates/ckks-train-bench/src/study_provenance/fixtures/roles"
)
LITERALS = json.loads((FIX / "literals.json").read_text())
INDEX = json.loads((FIX / "index.json").read_text())


def compile_build_script(source, compiler, directory):
    """Test-only host compile; synthetic study compiler facts come later."""
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    copied = directory / "build.rs"
    output = directory / "build-script-build"
    assert not output.exists(), "fresh compiler output required"
    copied.write_bytes(source_bytes)
    assert copied.read_bytes() == source_bytes, "compiler input byte mismatch"
    env = dict(os.environ, RUSTUP_AUTO_INSTALL="0")
    for key in (
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTC_BOOTSTRAP",
    ):
        env.pop(key, None)
    version = subprocess.run(
        [str(compiler), "-vV"], capture_output=True, timeout=10, env=env
    )
    assert version.returncode == 0, "selected rustc version unavailable"
    command = [
        str(compiler),
        "--edition=2024",
        "--crate-name",
        "study_build_harness",
        "--crate-type",
        "bin",
        "-C",
        "debuginfo=0",
        str(copied),
        "-o",
        str(output),
    ]
    identity = {
        "source": str(source),
        "source_sha256": source_hash,
        "source_bytes": len(source_bytes),
        "copied_source": str(copied),
        "compiler": str(compiler),
        "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "compiler_version": version.stdout.decode(errors="replace"),
        "command": command,
    }
    print("BUILD_SCRIPT_COMPILE " + json.dumps(identity), flush=True)
    try:
        result = subprocess.run(
            command, cwd=directory, env=env, capture_output=True, timeout=60
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        raise AssertionError(
            f"compile failed: source_sha256={source_hash} command={command!r} "
            f"{str(error)[:4096]}"
        ) from error
    assert result.returncode == 0, (
        f"compile failed: source_sha256={source_hash} command={command!r} "
        f"exit={result.returncode} {result.stderr[:4096].decode(errors='replace')}"
    )
    assert copied.read_bytes() == source_bytes, "compiled copy changed"
    assert source.read_bytes() == source_bytes, "reviewed source changed"
    assert output.is_file() and os.access(output, os.X_OK), "compiler output missing"
    print(
        "BUILD_SCRIPT_OUTPUT "
        + json.dumps(
            {
                "path": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "source_copy_byte_equal": True,
                "source_unchanged": True,
            }
        ),
        flush=True,
    )
    return output


class BuildScriptOwnershipTests(unittest.TestCase):
    def test_exact_source_and_owned_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, compiler = root / "original.rs", root / "rustc"
            content = b"fn main() {}\n// exact bytes\n"
            source.write_bytes(content)
            compiler.write_bytes(b"MOCK compiler identity only")
            output = root / "build-script-build"

            def run(argv, **kwargs):
                if argv[1:] == ["-vV"]:
                    return subprocess.CompletedProcess(argv, 0, b"MOCK version", b"")
                self.assertEqual(Path(argv[-3]).read_bytes(), content)
                self.assertEqual(Path(argv[-1]), output)
                self.assertEqual(kwargs["cwd"], root)
                self.assertEqual(kwargs["timeout"], 60)
                self.assertFalse(output.exists())
                output.write_bytes(b"MOCK executable ownership only")
                output.chmod(0o700)
                return subprocess.CompletedProcess(argv, 0, b"", b"")

            with mock.patch.object(subprocess, "run", side_effect=run), mock.patch(
                "builtins.print"
            ):
                self.assertEqual(compile_build_script(source, compiler, root), output)
            self.assertEqual(source.read_bytes(), content)
            with mock.patch.object(subprocess, "run") as run, self.assertRaisesRegex(
                AssertionError, "fresh compiler output required"
            ):
                compile_build_script(source, compiler, root)
            run.assert_not_called()

    def test_compile_failure_and_timeout_have_no_artifact_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, compiler = root / "original.rs", root / "rustc"
            source.write_bytes(b"fn main() {}\n")
            compiler.write_bytes(b"MOCK compiler identity only")
            for failure in (
                subprocess.CompletedProcess([], 7, b"", b"failure" * 2000),
                subprocess.TimeoutExpired(["MOCK compiler"], 60),
            ):
                with self.subTest(failure=type(failure).__name__), mock.patch.object(
                    subprocess,
                    "run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0, b"MOCK version", b""),
                        failure,
                    ],
                ), mock.patch("builtins.print"), self.assertRaises(
                    AssertionError
                ) as caught:
                    compile_build_script(source, compiler, root)
                message = str(caught.exception)
                self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), message)
                self.assertIn(str(compiler), message)
                self.assertIn("compile failed", message)
                self.assertLess(len(message), 5500)
                self.assertFalse((root / "build-script-build").exists())


def literal(name):
    return bytes.fromhex(LITERALS[name]["text"])


def domain(name):
    kind = INDEX[name]["kind"]
    if kind == "historical_transport":
        return (
            "capsule"
            if name.endswith("capsule")
            else "approval" if name.endswith("approval") else "receipt"
        )
    return {"files": "manifest", "approval_v1": "approval"}.get(kind, kind)


def decoded(name):
    return s.decode(domain(name), literal(name))


def receipt_predicates(name, capsule_override=None):
    r = decoded(name)
    c = (
        s.decode("capsule", r["capsule"])
        if capsule_override is None
        else capsule_override
    )
    recipe = s._closed_capsule(c)
    s._receipt_post(r, c)
    s._recipe_vectors(r, c, dict(s.decode("environment", r["environment"])), recipe)
    return r, c


def expected_error(name):
    pairs = [
        (("unknown_recipe", "unknown_approval_recipe"), "unsupported recipe"),
        (("unknown_algorithm", "unknown_approval_algorithm"), "unsupported algorithm"),
        (("metadata_feature",), "metadata argv"),
        (("feature_order",), "noncanonical encoding"),
        (
            (
                "serial_parallel",
                "missing_feature",
                "missing_parallel",
                "unknown_feature",
                "wrong_target",
            ),
            "compile facts",
        ),
        (("compiler_host",), "compiler host"),
        (("wrong_source", "wrong_emitted_target"), "unique writer binary linkage"),
        (("wrong_output",), "receipt executable path"),
        (("helper_extra_features",), "unsupported/missing helper option"),
        (("helper_script",), "fixed helper dispatch"),
        (("build_bin",), "build argv"),
        (("post_",), "pre/post mismatch"),
        (("capsule_source_digest",), "embedded manifest link"),
        (("role_recipe_mismatch",), "approval role/recipe mismatch"),
        (("triple_", "v1_hidden_train"), "receipt/capsule/executable not approved"),
        (("empty_id",), "approval policy"),
        (("wrong_design_path",), "approved design set"),
        (("design_digest",), "design bytes changed"),
        (("negative_commit", "approval_source"), "approved source/version mismatch"),
        (("negative_v1_",), "approval v1 Reference only"),
    ]
    for tokens, reason in pairs:
        if any(t in name for t in tokens):
            return reason
    raise AssertionError("unmapped obligation " + name)


class LiteralPolicyTests(unittest.TestCase):
    def test_exact_literal_hashes_and_parser_negatives(self):
        rejects = 0
        for name, item in INDEX.items():
            with self.subTest(name=name):
                entry = LITERALS[name]
                self.assertEqual(s.sha(entry["text"].encode()), entry["file_sha256"])
                self.assertEqual(s.sha(literal(name)), item["sha256"])
                self.assertEqual(len(literal(name)), item["length"])
                if item["expected"] == "parser_reject":
                    rejects += 1
                    with self.assertRaises(s.Invalid):
                        decoded(name)
                elif "obligation" not in item["expected"]:
                    decoded(name)
        self.assertEqual(rejects, 29)
        self.assertEqual(len(INDEX), 141)

    def test_four_recipes_roundtrip_and_reference_policy_stays_closed(self):
        for name in ("reference", "cpu_serial", "cpu_rayon", "cuda_host"):
            with self.subTest(name=name):
                r, c = receipt_predicates(name + "_receipt")
                self.assertEqual(s.capsule(c), literal(name + "_capsule"))
                self.assertEqual(s.receipt(r), literal(name + "_receipt"))
                if name != "reference":
                    with self.assertRaises(s.Invalid):
                        s.validate_recipe_capsule(c)
                else:
                    s.validate_recipe_capsule(c)
                # Partial fictional source closure cannot issue a checked build.
                with self.assertRaisesRegex(s.Invalid, "source uncovered"):
                    s._validate_linkage(r, c)

    def test_exact_five_v2_memberships_and_v1_unchanged(self):
        expected = {
            "reference_only": ["reference"],
            "reference_cpu_serial": ["reference", "cpu_serial"],
            "reference_cpu_rayon": ["reference", "cpu_rayon"],
            "reference_cuda_host": ["reference", "cuda_host"],
            "multiple_builds": [
                "reference",
                "reference_second",
                "cpu_serial",
                "cpu_rayon",
                "cuda_host",
            ],
        }
        for group, names in expected.items():
            a = decoded("approval_v2_" + group)
            s._approval_policy(a, 2)
            members = []
            for name in names:
                r = decoded(name + "_receipt")
                c = s.decode("capsule", r["capsule"])
                recipe = s._closed_capsule(c)
                members.append(
                    (
                        recipe.role,
                        c["recipe"],
                        c["algorithm"],
                        s.sha(literal(name + "_receipt")),
                        s.sha(r["capsule"]),
                        r["executable_hash"],
                    )
                )
                s._approved_receipt(a, 2, literal(name + "_receipt"), r, c)
            self.assertEqual(a["builds"], sorted(members, key=lambda b: b[3]))
            self.assertEqual(s.approval_v2(a), literal("approval_v2_" + group))
        a = decoded("approval_v1_reference")
        s._approval_policy(a, 1)
        self.assertEqual(s.approval(a), literal("approval_v1_reference"))

    def test_all_74_semantic_predicates_and_24_linked_bundles(self):
        count = bundles = 0
        for name, item in INDEX.items():
            if "obligation" not in item["expected"]:
                continue
            count += 1
            with self.subTest(name=name):
                kind = item["kind"]

                def predicate():
                    nonlocal bundles
                    if kind == "capsule":
                        c = decoded(name)
                        if "capsule_source_digest" in name:
                            s._receipt_manifest_links(decoded("cpu_serial_receipt"), c)
                        elif any(x in name for x in ("metadata_feature", "build_bin")):
                            r = decoded(
                                "cpu_rayon_receipt"
                                if "metadata" in name
                                else "cpu_serial_receipt"
                            )
                            s._recipe_vectors(
                                r,
                                c,
                                dict(s.decode("environment", r["environment"])),
                                s._closed_capsule(c),
                            )
                        else:
                            s._closed_capsule(c)
                    elif kind == "receipt":
                        receipt_predicates(name)
                    else:
                        a = decoded(name)
                        version = 1 if kind == "approval_v1" else 2
                        s._approval_policy(a, version)
                        if item["expected"] == "linked_bundle_obligation":
                            bundles += 1
                            rname = name.removesuffix("_approval") + "_receipt"
                            r, c = receipt_predicates(rname)
                            s._approved_receipt(a, version, literal(rname), r, c)
                        elif name == "negative_design_digest":
                            expected = decoded("approval_v2_reference_cpu_serial")[
                                "designs"
                            ]
                            for (path, h), (_, actual) in zip(a["designs"], expected):
                                s._design_digest(h, actual)
                        else:
                            r = decoded("cpu_serial_receipt")
                            c = s.decode("capsule", r["capsule"])
                            s._approved_receipt(
                                a, version, literal("cpu_serial_receipt"), r, c
                            )

                with self.assertRaises(s.Invalid) as caught:
                    predicate()
                self.assertEqual(str(caught.exception), expected_error(name))
                print("POLICY", name, str(caught.exception))
        self.assertEqual(count, 74)
        # Some bundles reject at their approval's own closed role/recipe policy.
        self.assertEqual(
            sum(v["expected"] == "linked_bundle_obligation" for v in INDEX.values()), 24
        )

    def test_all_bundle_nested_hashes_independent_of_semantic_acceptance(self):
        # Decode raw bounded receipt fields here only to inspect links on the
        # capsule-source mismatch; production receipt codec rejects that link.
        seen = 0
        for name, item in INDEX.items():
            if item["expected"] != "linked_bundle_obligation":
                continue
            seen += 1
            a = decoded(name)
            rname = name.removesuffix("_approval") + "_receipt"
            data = literal(rname)
            reader = s.Reader(data, "receipt")
            cap = reader.b(s.CAPSULE_LIMIT)
            for _ in range(5):
                reader.b(s.LIMIT)
            reader.b()
            executable = reader.h()
            self.assertIn(
                (s.sha(data), s.sha(cap), executable),
                [tuple(b[3:]) for b in a["builds"]],
            )
        self.assertEqual(seen, 24)


class BuildFactsTests(unittest.TestCase):
    def test_capture_actual_features_and_fail_before_write(self):
        for name in ("reference", "cpu_serial", "cpu_rayon", "cuda_host"):
            c = decoded(name + "_capsule")
            env = decoded(name + "_environment")
            args = types.SimpleNamespace(
                request="unused",
                outdir="unused",
                compiler=c["compiler"],
                target=c["target"],
                host=c["host"],
                profile=c["profile"],
                opt=c["opt"],
                debug="false",
                version=c["version"],
                feature=c["features"],
            )
            request = json.dumps(
                dict(
                    capsule=literal(name + "_capsule").hex(),
                    environment=literal(name + "_environment").hex(),
                    source_paths=[],
                )
            ).encode()
            actual = dict((k.encode(), v) for k, v in env)
            actual.update(
                {("CARGO_FEATURE_" + f.upper()).encode(): b"1" for f in c["features"]}
            )
            with mock.patch.object(
                s, "read_regular", return_value=request
            ), mock.patch.dict(os.environb, actual, clear=True), mock.patch.object(
                s, "exclusive_write"
            ) as write:
                build.capture(args)
                self.assertEqual(write.call_count, 2)
                write.reset_mock()
                for changed in (
                    ["unknown"],
                    (
                        list(reversed(c["features"]))
                        if len(c["features"]) > 1
                        else ["parallel", "cuda"]
                    ),
                ):
                    args.feature = changed
                    with self.assertRaises(s.Invalid):
                        build.capture(args)
                    write.assert_not_called()
                args.feature = c["features"]
                with mock.patch.dict(os.environb, {b"CARGO_FEATURE_UNKNOWN": b"1"}):
                    with self.assertRaises(s.Invalid):
                        build.capture(args)
                    write.assert_not_called()

    def test_observed_metadata_edges_and_feature_substitution(self):
        projection = json.loads((FIX / "metadata.json").read_text())["modes"]
        for mode, recipe in [
            ("serial", s._REFERENCE),
            ("serial", s._TRAIN[0]),
            ("parallel", s._TRAIN[1]),
            ("cuda", s._TRAIN[2]),
        ]:
            metadata = projection[mode]["metadata"]
            checked = s._resolved_features(metadata, recipe)
            self.assertTrue(checked)
            critical = {
                p["id"]
                for p in metadata["packages"]
                if p["name"]
                in (
                    "ckks-train-bench",
                    "ckks-train-ckks",
                    "rayon",
                    "rayon-core",
                    "ckks-train-cuda",
                    "cudarc",
                )
            }
            for ident in checked.keys() & critical:
                changed = copy.deepcopy(metadata)
                node = next(n for n in changed["resolve"]["nodes"] if n["id"] == ident)
                node["features"].append("unexpected")
                with self.assertRaises(s.Invalid):
                    s._resolved_features(changed, recipe)
            changed = copy.deepcopy(metadata)
            bench = next(
                p["id"] for p in changed["packages"] if p["name"] == "ckks-train-bench"
            )
            node = next(n for n in changed["resolve"]["nodes"] if n["id"] == bench)
            node["deps"] = [d for d in node["deps"] if d["name"] != "ckks_train_ckks"]
            with self.assertRaises(s.Invalid):
                s._resolved_features(changed, recipe)

    def test_build_rs_actual_feature_forwarding_synthetic_compiler(self):
        # Actual compiled build.rs dispatch, SYNTHETIC compiler and capsule
        # facts. No real native build/approval/runtime is asserted.
        import subprocess, sys

        root_dir = Path(__file__).resolve().parent.parent
        selected = os.environ.get("RUSTC")
        if selected is None:
            rustup = shutil.which("rustup")
            self.assertIsNotNone(rustup, "installed rustup or absolute RUSTC required")
            selection = subprocess.run(
                [rustup, "which", "rustc"],
                cwd=root_dir,
                env=dict(os.environ, RUSTUP_AUTO_INSTALL="0"),
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(selection.returncode, 0, selection.stderr)
            selected = os.fsdecode(selection.stdout.strip())
        host_compiler = Path(selected)
        self.assertTrue(
            host_compiler.is_absolute()
            and host_compiler.is_file()
            and os.access(host_compiler, os.X_OK),
            "selected installed absolute rustc required; no artifact fallback",
        )
        source = root_dir / "crates/ckks-train-bench/build.rs"
        with tempfile.TemporaryDirectory() as compilation, mock.patch.object(
            Path,
            "glob",
            side_effect=AssertionError("stale target-artifact discovery forbidden"),
        ):
            compile_dir = Path(compilation).resolve()
            before = shutil.disk_usage(compile_dir).free
            self.assertGreaterEqual(before, 805306368, "0.75 GiB compile floor")
            print("BUILD_SCRIPT_DISK_BEFORE " + str(before), flush=True)
            script = compile_build_script(source, host_compiler, compile_dir)
            after = shutil.disk_usage(compile_dir).free
            print("BUILD_SCRIPT_DISK_AFTER " + str(after), flush=True)
            self.assertGreaterEqual(after, 805306368, "0.75 GiB compile floor")
            for name in ("reference", "cpu_serial", "cpu_rayon", "cuda_host"):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp).resolve()
                    compiler = root / "rustc"
                    compiler.write_text(
                        "#!"
                        + str(Path(sys.executable).resolve())
                        + "\nprint("
                        + repr("SYNTHETIC compiler\nhost: " + s.TARGET + "\n")
                        + ",end='')\n"
                    )
                    compiler.chmod(0o700)
                    opts = {
                        k: str(root / k)
                        for k in (
                            "root",
                            "cargo",
                            "git",
                            "linker",
                            "cargo_home",
                            "target",
                            "home",
                            "tmp",
                            "request",
                        )
                    }
                    opts.update(
                        rustc=str(compiler),
                        python=str(Path(sys.executable).resolve()),
                        jobs=1,
                    )
                    env = s.build_environment(opts)
                    c = decoded(name + "_capsule")
                    environment = s.environment(
                        [(k, os.fsencode(v)) for k, v in env.items()]
                    )
                    c.update(
                        compiler="SYNTHETIC compiler\nhost: " + s.TARGET + "\n",
                        environment=s.sha(environment),
                    )
                    Path(opts["request"]).write_bytes(build.encode_request(c, env, []))
                    base = dict(
                        env,
                        TARGET=s.TARGET,
                        HOST=s.TARGET,
                        PROFILE="release",
                        OPT_LEVEL="3",
                        DEBUG="false",
                        CARGO_PKG_VERSION=c["version"],
                        CARGO_MANIFEST_DIR=str(root_dir / "crates/ckks-train-bench"),
                    )
                    actual = {"CARGO_FEATURE_" + f.upper(): "1" for f in c["features"]}
                    for number, features in enumerate(
                        (
                            actual,
                            {**actual, "CARGO_FEATURE_UNKNOWN": "1"},
                            {"CARGO_FEATURE_CUDA": "1"},
                        )
                    ):
                        out = root / str(number)
                        out.mkdir()
                        result = subprocess.run(
                            [str(script)],
                            env=dict(base, OUT_DIR=str(out), **features),
                            capture_output=True,
                            timeout=10,
                        )
                        print(
                            "BUILD_SCRIPT_SYNTHETIC_CASE "
                            + json.dumps(
                                dict(
                                    recipe=name,
                                    case=number,
                                    script=str(script),
                                    returncode=result.returncode,
                                )
                            ),
                            flush=True,
                        )
                        if number == 0:
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertEqual(
                                (out / "study_capsule.bin").read_bytes(), s.capsule(c)
                            )
                        else:
                            self.assertNotEqual(result.returncode, 0)
                            self.assertFalse((out / "study_capsule.bin").exists())
                            self.assertFalse((out / "study_capsule.rs").exists())

    def test_closed_options_and_cuda_overrides(self):
        import subprocess, sys

        script = Path(__file__).with_name("study_train_build.py")
        for args in (
            ["build"],
            ["cpu-serial", "--features", "parallel"],
            ["cpu-rayon", "--bin", "fake"],
            ["cuda-host", "--target-triple", "native"],
        ):
            p = subprocess.run(
                [sys.executable, "-B", str(script), *args], capture_output=True
            )
            self.assertEqual(p.returncode, 2)
        for key in (
            "CUDARC_CUDA_VERSION",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_ROOT",
            "CUDA_TOOLKIT_ROOT_DIR",
            "CUDNN_LIB",
        ):
            with self.assertRaises(s.Invalid):
                s.reject_caller_environment({key: "x"})

    def test_emitted_facts_match_resolved_graph_and_compile_profile(self):
        m = json.loads((FIX / "metadata.json").read_text())["modes"]["cuda"]["metadata"]
        recipe = s._TRAIN[2]
        resolved = s._resolved_features(m, recipe)
        bench = next(p["id"] for p in m["packages"] if p["name"] == "ckks-train-bench")
        good = dict(
            package_id=bench,
            features=["cuda", "parallel"],
            target={"name": "study_train"},
            profile={"opt_level": "3", "debuginfo": 0, "test": False},
        )
        s._emitted_feature_facts(good, resolved, bench, recipe)
        for key, value in [
            ("features", []),
            ("profile", {"opt_level": "0", "debuginfo": 0, "test": False}),
            ("profile", {"opt_level": "3", "debuginfo": 2, "test": False}),
            ("profile", {"opt_level": "3", "debuginfo": 0, "test": True}),
        ]:
            changed = copy.deepcopy(good)
            changed[key] = value
            with self.assertRaises(s.Invalid):
                s._emitted_feature_facts(changed, resolved, bench, recipe)

    def test_canonical_exact_capsule_receipt_totals(self):
        c = decoded("reference_capsule")
        c.update(packages=[], cwd=b"")
        remaining = s.CAPSULE_LIMIT - s.preflight_capsule(c)
        i = 0
        while remaining > s.TEXT:
            ident = f"p{i:06}"
            overhead = 8 + len(ident) + 8 + 1 + 8
            n = min(s.TEXT, remaining - overhead)
            c["packages"].append((ident, b"x" * n, 0, ""))
            remaining -= overhead + n
            i += 1
        c["cwd"] = b"x" * remaining
        data = s.capsule(c)
        self.assertEqual(len(data), s.CAPSULE_LIMIT)
        s.decode("capsule", data)
        c["cwd"] += b"x"
        with self.assertRaises(s.Invalid):
            s.capsule(c)
        with self.assertRaises(s.Invalid):
            s.decode("capsule", data + b"x")
        r = decoded("reference_receipt")
        r.update(emitted=[], argv=[], executable=b"")
        remaining = s.LIMIT - s.preflight_receipt(
            len(r["capsule"]),
            list(map(len, r["files"])),
            len(r["environment"]),
            r["executable"],
            r["emitted"],
            r["argv"],
        )
        i = 0
        while remaining > s.TEXT:
            ident = f"p{i:06}"
            overhead = 8 + len(ident) + 8 + 8 + 8 + 8 + 1
            n = min(s.TEXT, remaining - overhead)
            r["emitted"].append((ident, b"x" * n, "", b"", (), None))
            remaining -= overhead + n
            i += 1
        r["executable"] = b"x" * remaining
        data = s.receipt(r)
        self.assertEqual(len(data), s.LIMIT)
        s.decode("receipt", data)
        r["executable"] += b"x"
        with mock.patch.object(
            s, "sha", side_effect=AssertionError("hash before preflight")
        ):
            with self.assertRaises(s.Invalid):
                s.receipt(r)
        with self.assertRaises(s.Invalid):
            s.decode("receipt", data + b"x")

    def test_checked_sizes_preflight_before_hash(self):
        self.assertEqual(s._checked_size_add(2**64 - 2, 1), 2**64 - 1)
        with self.assertRaisesRegex(s.Invalid, "size arithmetic"):
            s._checked_size_add(2**64 - 1, 1)
        with self.assertRaisesRegex(s.Invalid, "size arithmetic"):
            s._checked_size_add(-1, 1)
        c = decoded("cpu_serial_capsule")
        c["compiler"] = "x" * 16385
        with mock.patch.object(
            s, "sha", side_effect=AssertionError("hash before preflight")
        ):
            with self.assertRaises(s.Invalid):
                s.preflight_capsule(c)
        for n in (s.LIMIT, s.CAPSULE_LIMIT):
            w = s.Writer("capsule", limit=n)
            w.add(b"x" * (n - len(w.data)))
            self.assertEqual(len(w.done()), n)
            with self.assertRaises(s.Invalid):
                w.add(b"x")
        for n in (4096, 16384):
            w = s.Writer("capsule")
            w.b(b"x" * n, n)
            with self.assertRaises(s.Invalid):
                w.b(b"x" * (n + 1), n)


class LinkageRawPathFix1Tests(unittest.TestCase):
    """Synthetic private predicate only; no fixture/approval/context is created."""

    def baseline(self):
        r = decoded("cpu_serial_receipt")
        c = s.decode("capsule", r["capsule"])
        manifest = s.decode("manifest", r["files"][0])
        manifest["records"].append(
            (
                b"scripts/study_reference_build_support.py",
                1,
                s.sha(b"SYNTHETIC PRIVATE PREDICATE ONLY"),
            )
        )
        r["files"][0] = s.manifest(0, manifest["records"])
        c["manifests"][0] = s.sha(r["files"][0])
        r["capsule"] = s.capsule(c)
        r["post"] = c["manifests"] + [c["environment"]]
        r = s.decode("receipt", s.receipt(r))
        c = s.decode("capsule", r["capsule"])
        self.assertEqual(s._validate_linkage(r, c), s._TRAIN[0])
        return r, c

    def test_canonical_nonselected_artifact_source_and_output_escapes(self):
        for field, raw in (
            (3, b"/synthetic/role-root/crates/ckks-train-bench/../../../../outside.rs"),
            (5, b"/synthetic/fresh-cpu_serial/../outside-executable"),
        ):
            with self.subTest(field=field, raw=raw):
                r, _ = self.baseline()
                auxiliary = list(r["emitted"][0])
                auxiliary[2], auxiliary[4] = "synthetic_auxiliary", ("custom-build",)
                auxiliary[field] = raw
                r["emitted"].append(tuple(auxiliary))
                encoded = s.receipt(r)
                r = s.decode("receipt", encoded)
                self.assertEqual(s.receipt(r), encoded)
                c = s.decode("capsule", r["capsule"])
                with self.assertRaisesRegex(s.Invalid, "raw lexical path"):
                    s._validate_linkage(r, c)

    def test_nonselected_dot_nul_relative_and_overlong_paths(self):
        for field in (1, 3, 5):
            for raw in (
                b"/synthetic/./auxiliary",
                b"/synthetic/a\0b",
                b"relative/auxiliary",
            ):
                with self.subTest(field=field, raw=raw):
                    r, _ = self.baseline()
                    auxiliary = list(r["emitted"][0])
                    auxiliary[2], auxiliary[4] = "synthetic_auxiliary", (
                        "custom-build",
                    )
                    auxiliary[field] = raw
                    r["emitted"].append(tuple(auxiliary))
                    r = s.decode("receipt", s.receipt(r))
                    c = s.decode("capsule", r["capsule"])
                    with self.assertRaisesRegex(s.Invalid, "raw lexical path"):
                        s._validate_linkage(r, c)
        # Overlong bytes are rejected by the transport before linkage; do not
        # mislabel that as a successful raw-path predicate test.
        r, _ = self.baseline()
        auxiliary = list(r["emitted"][0])
        auxiliary[3] = b"/" + b"x" * s.TEXT
        r["emitted"].append(tuple(auxiliary))
        with self.assertRaises(s.Invalid):
            s.receipt(r)

    def test_raw_validator_preserves_bounds_bytes_and_rejects_before_path(self):
        for raw in (
            b"/",
            b"/synthetic/.hidden",
            b"/synthetic/\xff",
            b"/" + b"x" * (s.TEXT - 1),
        ):
            self.assertEqual(os.fsencode(s._claim_path(raw)), raw)
        for raw in (
            b"",
            b"relative",
            b"/a/.",
            b"/a/../b",
            b"/a\0b",
            b"/" + b"x" * s.TEXT,
        ):
            with self.subTest(raw=raw[:40]):
                with mock.patch.object(
                    s, "Path", side_effect=AssertionError("Path before raw validation")
                ):
                    with self.assertRaisesRegex(s.Invalid, "raw lexical path"):
                        s._claim_path(raw)

    def test_snapshot_rejects_escapes_before_filesystem_source_recheck(self):
        for field, raw in (
            (3, b"/synthetic/role-root/crates/ckks-train-bench/../../../../outside.rs"),
            (5, b"/synthetic/fresh-cpu_serial/../outside-executable"),
        ):
            with self.subTest(field=field):
                r, c = self.baseline()
                auxiliary = list(r["emitted"][0])
                auxiliary[2], auxiliary[4] = "synthetic_auxiliary", ("custom-build",)
                auxiliary[field] = raw
                r["emitted"].append(tuple(auxiliary))
                data = s.receipt(r)
                s.decode("receipt", data)
                with mock.patch.object(
                    s,
                    "source_inventory",
                    side_effect=AssertionError("source recheck before path refusal"),
                ) as inventory:
                    with self.assertRaisesRegex(s.Invalid, "raw lexical path"):
                        s.verify_runtime_snapshot(
                            Path(os.fsdecode(c["cwd"])), data, "/synthetic/tools/git"
                        )
                    inventory.assert_not_called()

    def test_capsule_and_package_raw_paths_validate_before_path_construction(self):
        for raw in (
            b"/synthetic/role-root/.",
            b"/synthetic/role-root/../role-root",
            b"/synthetic/role-root\0",
            b"relative",
        ):
            with self.subTest(raw=raw):
                r, c = self.baseline()
                c["cwd"] = raw
                r["capsule"] = s.capsule(c)
                r = s.decode("receipt", s.receipt(r))
                with self.assertRaisesRegex(s.Invalid, "raw lexical path"):
                    s._validate_linkage(r, s.decode("capsule", r["capsule"]))
        for raw in (
            b"/synthetic/role-root/crates/ckks-train-bench/./Cargo.toml",
            b"/synthetic/role-root/crates/ckks-train-bench/../ckks-train-bench/Cargo.toml",
            b"/synthetic/Cargo.toml\0",
            b"relative/Cargo.toml",
        ):
            with self.subTest(raw=raw):
                r, c = self.baseline()
                c["packages"].append(("zz-synthetic-package", raw, 0, ""))
                r["capsule"] = s.capsule(c)
                r = s.decode("receipt", s.receipt(r))
                with self.assertRaisesRegex(s.Invalid, "raw lexical path"):
                    s._validate_linkage(r, s.decode("capsule", r["capsule"]))


if __name__ == "__main__":
    unittest.main()
