"""D6c2a tests: fictional byte fixtures and local bounded filesystem cases only."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
FIX = HERE.parent / "crates/ckks-train-bench/src/study_provenance/fixtures"
spec = importlib.util.spec_from_file_location(
    "support", HERE / "study_reference_build_support.py"
)
s = importlib.util.module_from_spec(spec)
if spec.loader:
    spec.loader.exec_module(s)


def literal(name):
    return bytes.fromhex((FIX / (name + ".hex")).read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fixture():
    f = json.loads((FIX / "fixture_inputs.json").read_text())
    files = []
    for name in ("source", "config", "dependency", "toolchain"):
        x = f["files"][name]
        records = [
            (
                (
                    r.get("path_utf8", "").encode()
                    if "path_utf8" in r
                    else bytes.fromhex(r["path_hex"])
                ),
                r["tag"],
                (
                    digest(bytes.fromhex(r["content_hex"]))
                    if r["tag"] == 1
                    else bytes.fromhex(r["target_hex"]) if r["tag"] == 2 else None
                ),
            )
            for r in x["records"]
        ]
        files.append(s.manifest(x["role"], records))
    env = s.environment(
        [
            (
                r["key"],
                None if r["value_hex"] is None else bytes.fromhex(r["value_hex"]),
            )
            for r in f["environment"]
        ]
    )
    c = f["capsule"]
    cap = dict(
        recipe=c["recipe"],
        algorithm=c["session_algorithm"],
        manifests=[digest(x) for x in files],
        environment=digest(env),
        lock=digest(bytes.fromhex(c["cargo_lock_hex"])),
        commit=c["source_git_sha1"],
        compiler=c["compiler_vv"],
        cargo=c["cargo_v"],
        target=c["target"],
        host=c["host"],
        profile=c["profile"],
        opt=c["opt_level"],
        debug=c["debug"],
        features=c["enabled_bench_features"],
        version=c["package_version"],
        metadata=digest(bytes.fromhex(c["metadata_stdout_hex"])),
        packages=[
            (r["id"], r["manifest_utf8"].encode(), r["tag"], r["source"])
            for r in c["packages"]
        ],
        metadata_argv=[bytes.fromhex(x) for x in c["metadata_argv_hex"]],
        build_argv=[bytes.fromhex(x) for x in c["build_argv_hex"]],
        cwd=bytes.fromhex(c["cwd_hex"]),
    )
    r = f["receipt"]
    receipt = dict(
        capsule=s.capsule(cap),
        files=files,
        environment=env,
        executable=bytes.fromhex(r["executable_path_hex"]),
        executable_hash=digest(bytes.fromhex(r["executable_bytes_hex"])),
        post=[digest(x) for x in files] + [digest(env)],
        metadata=cap["metadata"],
        emitted=[
            (
                x["package_id"],
                x["manifest_utf8"].encode(),
                x["name"],
                x["src_utf8"].encode(),
                tuple(x["kinds"]),
                (
                    None
                    if x["executable_hex"] is None
                    else bytes.fromhex(x["executable_hex"])
                ),
            )
            for x in r["emitted"]
        ],
        argv=[bytes.fromhex(x) for x in r["helper_argv_hex"]],
    )
    return f, files, env, cap, receipt


class EncodingTests(unittest.TestCase):
    def test_independent_complete_domains_and_reversed_approval(self):
        f, files, env, cap, r = fixture()
        for name, value in zip(("source", "config", "dependency", "toolchain"), files):
            self.assertEqual(literal("files_" + name), value)
        self.assertEqual(literal("environment"), env)
        self.assertEqual(literal("capsule"), s.capsule(cap))
        self.assertEqual(literal("receipt"), s.receipt(r))
        # Q1: receipt d492... comes BEFORE 4adc... in input, reverse canonical order.
        second = s.decode("receipt", literal("receipt_second_build"))
        triples = [
            (digest(literal("receipt")), digest(r["capsule"]), r["executable_hash"]),
            (
                digest(literal("receipt_second_build")),
                digest(second["capsule"]),
                second["executable_hash"],
            ),
        ]
        triples.sort(reverse=True)
        a = dict(
            id=f["approval"]["id"],
            recipe=cap["recipe"],
            algorithm=cap["algorithm"],
            designs=[
                (x["path"], digest(bytes.fromhex(x["content_hex"])))
                for x in f["approval"]["designs"]
            ],
            commit=cap["commit"],
            source=digest(files[0]),
            builds=triples,
        )
        self.assertEqual(literal("approval"), s.approval(a))

    def test_strict_decode_every_truncation_tags_counts_and_embedded_hash(self):
        for name, domain in [
            ("capsule", "capsule"),
            ("receipt", "receipt"),
            ("approval", "approval"),
            ("environment", "environment"),
            ("files_dependency", "manifest"),
        ]:
            b = literal(name)
            self.assertEqual(
                b,
                (
                    getattr(s, domain)(**s.decode(domain, b))
                    if domain == "manifest"
                    else getattr(s, domain)(s.decode(domain, b))
                ),
            )
            for cut in (0, 1, 8, len(b) - 1):
                with self.assertRaises(s.Invalid):
                    s.decode(domain, b[:cut])
            with self.assertRaises(s.Invalid):
                s.decode(domain, b + b"\0")
            with self.assertRaises(s.Invalid):
                s.decode(domain, b"\xff" * 8 + b[8:])
        r = s.decode("receipt", literal("receipt"))
        r["files"][0] = literal("tag_source_regular")
        with self.assertRaises(s.Invalid):
            s.receipt(r)

    def test_absent_empty_raw_bytes_and_canonical_sort(self):
        f, _, env, _, _ = fixture()
        pairs = [
            (
                x["key"],
                None if x["value_hex"] is None else bytes.fromhex(x["value_hex"]),
            )
            for x in f["environment"]
        ]
        pairs = [(k, b"" if k == "A" else v) for k, v in pairs]
        self.assertEqual(literal("environment_absent_to_empty"), s.environment(pairs))
        self.assertNotEqual(env, s.environment(pairs))
        with self.assertRaises(s.Invalid):
            s.environment([("A", None), ("A", b"")])
        with self.assertRaises(s.Invalid):
            s.manifest(0, [(b"x", 1, "0" * 64)] * 513)
        with self.assertRaises(s.Invalid):
            s.environment([("A", b"x" * 4097)])


class InventoryTests(unittest.TestCase):
    def test_preflight_rejects_before_hash_and_link_lexical_closure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "target").write_bytes(b"x")
            (root / "a").symlink_to("target")
            (root / "b").symlink_to("target")
            rows = s.enumerate_tree(root, allow_links=True)
            self.assertEqual(
                [os.fsencode(root / x) for x in ("a", "b", "target")],
                [r[0] for r in rows],
            )
            self.assertEqual([2, 2, 1], [r[1] for r in rows])
            with self.assertRaises(s.Invalid):
                s.enumerate_tree(root, allow_links=False)
            with self.assertRaises(s.Invalid):
                s.preflight_manifest(2, rows, max_records=2)
            (root / "outside").symlink_to("/etc/passwd")
            with self.assertRaises(s.Invalid):
                s.enumerate_tree(root, allow_links=True)

    def test_link_target_through_directory_link_retains_real_lexical_keys(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "real").mkdir()
            (root / "real/file").write_bytes(b"x")
            (root / "zdir").symlink_to("real")
            (root / "a").symlink_to("zdir/file")
            rows = s.enumerate_tree(root, allow_links=True)
            self.assertEqual(
                [os.fsencode(root / x) for x in ("a", "real/file", "zdir")],
                [x[0] for x in rows],
            )

    def test_selected_link_expands_directory_symlink_before_dotdot(self):
        for absolute in (False, True):
            with self.subTest(absolute=absolute), tempfile.TemporaryDirectory() as d:
                root = Path(d).resolve() / "inventory"
                (root / "aux/sub/deep").mkdir(parents=True)
                (root / "lib").mkdir()
                (root / "aux/target").write_bytes(b"lexical decoy")
                actual = root / "aux/sub/target"
                actual.write_bytes(b"actual file")
                (root / "aux/redirect").symlink_to("sub/deep")
                link = root / "lib/driver"
                target = (
                    os.fsencode(root) + b"/aux/redirect/../target"
                    if absolute
                    else b"../aux/redirect/../target"
                )
                link.symlink_to(os.fsdecode(target))
                self.assertEqual(actual, link.resolve(strict=True))
                self.assertEqual(
                    [
                        (os.fsencode(root / "aux/redirect"), 2, b"sub/deep"),
                        (os.fsencode(actual), 1, None),
                        (os.fsencode(link), 2, target),
                    ],
                    s.enumerate_tree(root, True, [link]),
                )
                with self.assertRaises(s.Invalid):
                    s.enumerate_tree(root, True, [link], max_records=2)
                with self.assertRaises(s.Invalid):
                    s.enumerate_tree(root, False, [link])
                with self.assertRaises(s.Invalid):
                    s.enumerate_tree(root, True, [root / ("x" * 4097)])

    def test_selected_link_dotdot_escape_rejects_actual_outside_target(self):
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d).resolve()
            root = parent / "inventory"
            (root / "aux").mkdir(parents=True)
            (root / "lib").mkdir()
            (parent / "outside/deep").mkdir(parents=True)
            actual = parent / "outside/target"
            actual.write_bytes(b"outside file")
            (root / "aux/target").write_bytes(b"in-root decoy")
            (root / "aux/redirect").symlink_to("../../outside/deep")
            link = root / "lib/driver"
            link.symlink_to("../aux/redirect/../target")
            self.assertEqual(actual, link.resolve(strict=True))
            with self.assertRaises(s.Invalid):
                s.enumerate_tree(root, True, [link])

    def test_selected_link_file_dotdot_and_cycle_do_not_collapse_lexically(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "lib").mkdir()
            (root / "aux").mkdir()
            (root / "aux/plain").write_bytes(b"regular file, not directory")
            (root / "aux/target").write_bytes(b"decoy")
            link = root / "lib/driver"
            link.symlink_to("../aux/plain/../target")
            with self.assertRaises(OSError):
                link.stat()
            with self.assertRaises(s.Invalid):
                s.enumerate_tree(root, True, [link])
            link.unlink()
            link.symlink_to("../aux/redirect/../target")
            (root / "aux/redirect").symlink_to("../lib/driver")
            with self.assertRaises(s.Invalid):
                s.enumerate_tree(root, True, [link])

    def test_link_cycle_special_and_config_appearance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "a").symlink_to("b")
            (root / "b").symlink_to("a")
            with self.assertRaises(s.Invalid):
                s.enumerate_tree(root, allow_links=True)
            (root / "a").unlink()
            (root / "b").unlink()
            os.mkfifo(root / "fifo")
            with self.assertRaises(s.Invalid):
                s.enumerate_tree(root, allow_links=True)
            os.unlink(root / "fifo")
            home = root / "cargo"
            home.mkdir()
            rows = s.config_inventory(root, home)
            self.assertTrue(all(r[1] == 0 for r in rows))
            (root / ".cargo").mkdir()
            (root / ".cargo/config").write_text("")
            with self.assertRaises(s.Invalid):
                s.config_inventory(root, home)

    def test_environment_cleared_and_recipe_closed(self):
        opts = dict(
            root="/synthetic/root",
            cargo="/synthetic/sys/bin/cargo",
            rustc="/synthetic/sys/bin/rustc",
            python="/synthetic/python",
            git="/synthetic/git",
            linker="/synthetic/cc",
            cargo_home="/synthetic/cache",
            target="/synthetic/fresh",
            home="/synthetic/home",
            tmp="/synthetic/tmp",
            request="/synthetic/request",
            jobs=4,
        )
        env = s.build_environment(opts)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("RUSTFLAGS", env)
        self.assertEqual("0", env["CARGO_INCREMENTAL"])
        self.assertEqual("true", env["CARGO_NET_OFFLINE"])
        self.assertEqual(18, len(env))
        meta, build = s.recipe_argv(opts)
        self.assertIn("--offline", meta)
        self.assertIn("--locked", build)
        self.assertEqual("study_reference", build[build.index("--bin") + 1])
        for jobs in (0, 5):
            bad = dict(opts, jobs=jobs)
            with self.assertRaises(s.Invalid):
                s.build_environment(bad)
        with self.assertRaises(s.Invalid):
            s.build_environment(dict(opts, linker="/bad\x1fpath"))

    def test_conservative_packages_smaller_emitted_subset_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            cargo = root / "cache"
            target = root / "fresh"
            target.mkdir()
            p = root / "crates/bench"
            p.mkdir(parents=True)
            (p / "Cargo.toml").write_text("")
            (p / "main.rs").write_text("")
            q = cargo / "registry/src/index/optional"
            q.mkdir(parents=True)
            (q / "Cargo.toml").write_text("")
            (q / ".cargo-checksum.json").write_text("{}")
            metadata = dict(
                version=1,
                workspace_members=["bench"],
                packages=[
                    dict(id="bench", manifest_path=str(p / "Cargo.toml"), source=None),
                    dict(
                        id="optional",
                        manifest_path=str(q / "Cargo.toml"),
                        source="registry+https://example.invalid",
                    ),
                ],
            )
            packages = s.packages(metadata, root, cargo)
            self.assertEqual(2, len(packages))
            msg = dict(
                reason="compiler-artifact",
                package_id="bench",
                manifest_path=str(p / "Cargo.toml"),
                target=dict(
                    name="study_reference", src_path=str(p / "main.rs"), kind=["bin"]
                ),
                filenames=[str(target / "binary")],
                executable=str(target / "binary"),
            )
            self.assertEqual(1, len(s.artifacts([msg], packages, target)))
            bad = copy.deepcopy(msg)
            bad["package_id"] = "unknown"
            with self.assertRaises(s.Invalid):
                s.artifacts([bad], packages, target)
            bad = copy.deepcopy(msg)
            bad["filenames"] = ["/outside"]
            with self.assertRaises(s.Invalid):
                s.artifacts([bad], packages, target)
            metadata["packages"][0]["manifest_path"] = "/outside/Cargo.toml"
            with self.assertRaises(s.Invalid):
                s.packages(metadata, root, cargo)


class RecipeFailureTests(unittest.TestCase):
    def test_caller_overrides_rejected_but_cargo_generated_loader_is_not_caller(self):
        for key in (
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "RUSTFLAGS",
            "RUSTC_BOOTSTRAP",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "CARGO_TARGET_DIR",
            "GIT_DIR",
        ):
            with self.assertRaises(s.Invalid):
                s.reject_caller_environment({key: "anything"})
        s.reject_caller_environment(
            {"PATH": "/different/compiler", "LANG": "ja_JP.UTF-8"}
        )

    def test_exact_sizing_before_content_and_post_metadata_change(self):
        _, files, env, cap, r = fixture()
        self.assertEqual(len(s.capsule(cap)), s.preflight_capsule(cap))
        emitted = s.decode("receipt", literal("receipt"))["emitted"]
        self.assertEqual(
            len(literal("receipt")),
            s.preflight_receipt(
                len(r["capsule"]),
                [len(x) for x in files],
                len(env),
                r["executable"],
                emitted,
                r["argv"],
            ),
        )
        with self.assertRaises(s.Invalid):
            s.preflight_receipt(s.CAPSULE_LIMIT, [s.LIMIT] * 4, 100, b"/x", [], [])
        with self.assertRaises(s.Invalid):
            s.unchanged_metadata(b'{"a":1}', b'{"a":2}')
        s.unchanged_metadata(b"{}", b"{}")

    def test_source_coverage_and_clean_git_failures_synthetic_command_boundary(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            for name in ("crates", "scripts", "tools"):
                (root / name).mkdir()
            for name in ("Cargo.toml", "Cargo.lock", "crates/a.rs"):
                (root / name).write_text("x")
            tracked = [b"Cargo.toml", b"Cargo.lock", b"crates/a.rs"]

            def fake(argv, cwd, env, cap=s.LIMIT):
                self.assertEqual(str(root), argv[6])
                self.assertEqual(root, Path(cwd))
                if argv[-2:] == ["rev-parse", "--show-toplevel"]:
                    return os.fsencode(root) + b"\n"
                if argv[-2:] == ["rev-parse", "HEAD"]:
                    return b"1" * 40 + b"\n"
                if "--stage" in argv:
                    return b"".join(
                        b"100644 " + b"0" * 40 + b" 0\t" + x + b"\0" for x in tracked
                    )
                return b""

            with patch.object(s, "bounded_run", fake):
                _, rows = s.source_inventory(root, "/synthetic/git", {}, designs=())
                first = s.hash_inventory(0, rows, root)
                (root / "crates/a.rs").write_text("different")
                self.assertNotEqual(first, s.hash_inventory(0, rows, root))
                for extra in ("new.rs", ".ignored"):
                    (root / "crates" / extra).write_text("x")
                    with self.assertRaises(s.Invalid):
                        s.source_inventory(root, "/synthetic/git", {}, designs=())
                    (root / "crates" / extra).unlink()
                (root / "crates/a.rs").unlink()
                with self.assertRaises(s.Invalid):
                    s.source_inventory(root, "/synthetic/git", {}, designs=())
                (root / "crates/a.rs").symlink_to("../Cargo.toml")
                with self.assertRaises(s.Invalid):
                    s.source_inventory(root, "/synthetic/git", {}, designs=())
            for reason in ("dirty tracked", "untracked", "Git unavailable"):
                with patch.object(s, "bounded_run", side_effect=s.Invalid(reason)):
                    with self.assertRaises(s.Invalid):
                        s.git_state(root, "/synthetic/git", {})

    def test_toolchain_selected_driver_closure_expands_before_dotdot(self):
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d).resolve()
            root = parent / "sys"
            for folder in (
                "bin",
                "lib/rustlib/" + s.TARGET + "/lib",
                "lib/rustlib/" + s.TARGET + "/bin",
                "aux/sub/deep",
            ):
                (root / folder).mkdir(parents=True)
            for name in ("bin/rustc", "bin/cargo", "aux/target", "aux/sub/target"):
                (root / name).write_bytes(b"synthetic structural fixture")
            redirect = root / "aux/redirect"
            redirect.symlink_to("sub/deep")
            driver = root / "lib/driver.so"
            driver.symlink_to("../aux/redirect/../target")
            tools = {"cargo": str(root / "bin/cargo"), "rustc": str(root / "bin/rustc")}
            for name in ("python", "git", "linker"):
                (parent / name).write_bytes(b"synthetic leaf")
                tools[name] = str(parent / name)
            rows = s.toolchain_inventory(root, s.TARGET, s.TARGET, tools)
            self.assertIn((os.fsencode(redirect), 2, b"sub/deep"), rows)
            self.assertIn((os.fsencode(root / "aux/sub/target"), 1, None), rows)
            self.assertNotIn(os.fsencode(root / "aux/target"), [row[0] for row in rows])
            self.assertIn((os.fsencode(driver), 2, b"../aux/redirect/../target"), rows)
            redirect.unlink()
            (parent / "outside/deep").mkdir(parents=True)
            (parent / "outside/target").write_bytes(b"outside fixture")
            redirect.symlink_to("../../outside/deep")
            with self.assertRaises(s.Invalid):
                s.toolchain_inventory(root, s.TARGET, s.TARGET, tools)

    def test_toolchain_structural_inventory_excludes_unrelated_docs_and_targets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            sysroot = root / "sys"
            sysroot.mkdir()
            for folder in (
                "bin",
                "lib/rustlib/" + s.TARGET + "/lib",
                "lib/rustlib/" + s.TARGET + "/bin/gcc-ld",
                "lib/rustlib/unused/lib",
                "share/doc",
            ):
                (sysroot / folder).mkdir(parents=True, exist_ok=True)
            for name in (
                "bin/rustc",
                "bin/cargo",
                "lib/driver.so",
                "lib/llvm.so",
                "lib/rustlib/" + s.TARGET + "/lib/std.rlib",
                "lib/rustlib/" + s.TARGET + "/bin/gcc-ld/lld",
                "lib/rustlib/unused/lib/ignored",
                "share/doc/ignored",
            ):
                (sysroot / name).write_text("x")
            tools = {
                "rustc": str(sysroot / "bin/rustc"),
                "cargo": str(sysroot / "bin/cargo"),
            }
            for key in ("python", "git", "linker"):
                (root / key).write_text("x")
                tools[key] = str(root / key)
            rows = s.toolchain_inventory(sysroot, s.TARGET, s.TARGET, tools)
            paths = [os.fsdecode(x[0]) for x in rows]
            self.assertTrue(any("llvm.so" in x for x in paths))
            self.assertFalse(any("ignored" in x for x in paths))
            (sysroot / "lib/unknown").mkdir()
            with self.assertRaises(s.Invalid):
                s.toolchain_inventory(sysroot, s.TARGET, s.TARGET, tools)
            with self.assertRaises(s.Invalid):
                s.toolchain_inventory(sysroot, "different", s.TARGET, tools)


class AdditionalBoundTests(unittest.TestCase):
    def test_fresh_target_existing_receipt_and_missing_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(s.Invalid):
                s.fresh_destination(existing)
            sentinel = root / "receipt"
            sentinel.write_bytes(b"sentinel")
            with self.assertRaises(FileExistsError):
                s.exclusive_write(sentinel, b"replace")
            self.assertEqual(b"sentinel", sentinel.read_bytes())
            cargo = root / "cache"
            pkg = cargo / "registry/src/index/package"
            pkg.mkdir(parents=True)
            (pkg / "Cargo.toml").write_text("")
            m = {
                "version": 1,
                "workspace_members": [],
                "packages": [
                    {
                        "id": "id",
                        "manifest_path": str(pkg / "Cargo.toml"),
                        "source": "registry+https://example.invalid",
                    }
                ],
            }
            with self.assertRaises(s.Invalid):
                s.packages(m, root, cargo)
            (pkg / ".cargo-checksum.json").write_text("{}")
            s.packages(m, root, cargo)
            m["packages"][0]["source"] = "git+https://example.invalid"
            with self.assertRaises(s.Invalid):
                s.packages(m, root, cargo)

    def test_preflight_rejects_oversize_fields_without_hashing(self):
        from unittest.mock import patch

        _, _, _, cap, _ = fixture()
        for field in ("version", "compiler", "cwd"):
            changed = dict(cap)
            changed[field] = (
                b"x" * 4097
                if field == "cwd"
                else "x" * (16385 if field == "compiler" else 4097)
            )
            with patch.object(
                s,
                "regular_hash",
                side_effect=AssertionError("content opened before bounds"),
            ):
                with self.assertRaises(s.Invalid):
                    s.preflight_capsule(changed)
        with self.assertRaises(s.Invalid):
            s.preflight_receipt(100, [100] * 4, 100, b"x" * 4097, [], [])

    def test_bounded_command_and_fifo_refusal(self):
        import sys

        env = {"PATH": "/nonexistent"}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaises(s.Invalid):
                s.read_regular(fifo)
            with self.assertRaises(s.Invalid):
                s.regular_hash(fifo)
            with self.assertRaises(s.Invalid):
                s.bounded_run(
                    [sys.executable, "-I", "-c", 'print("x"*10000)'], root, env, 10
                )
            with self.assertRaises(s.Invalid):
                s.bounded_run(
                    [sys.executable, "-I", "-c", "raise SystemExit(1)"], root, env, 10
                )


class CompileCaptureHarnessTests(unittest.TestCase):
    def test_actual_selected_compiler_not_path_and_cargo_loader_allowed(self):
        # Explicit SYNTHETIC command harness. Does not compile a production binary,
        # perform a production build, issue approval or construct a context.
        import subprocess
        import sys

        builds = list(
            (HERE.parent / "target/debug/build").glob(
                "ckks-train-bench-*/build-script-build"
            )
        )
        self.assertTrue(builds, "ordinary build.rs must be compiled first")
        script = max(builds, key=lambda p: p.stat().st_mtime)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            tool = root / "zz/bin"
            tool.mkdir(parents=True)
            other = root / "aa/bin"
            other.mkdir(parents=True)
            compiler = "SYNTHETIC selected compiler\nhost: " + s.TARGET + "\n"
            for p, content in [
                (tool / "rustc", compiler),
                (other / "rustc", "SYNTHETIC wrong PATH compiler\n"),
            ]:
                p.write_text(
                    "#!"
                    + str(Path(sys.executable).resolve())
                    + "\nprint("
                    + repr(content)
                    + ', end="")\n'
                )
                p.chmod(0o700)
            _, _, _, cap, _ = fixture()
            opts = dict(
                root=str(root),
                cargo=str(tool / "cargo"),
                rustc=str(tool / "rustc"),
                python=str(Path(sys.executable).resolve()),
                git=str(other / "git"),
                linker=str(other / "cc"),
                cargo_home=str(root / "cache"),
                target=str(root / "fresh"),
                home=str(root / "home"),
                tmp=str(root / "tmp"),
                request=str(root / "request"),
                jobs=4,
            )
            env = s.build_environment(opts)
            cap.update(
                recipe=s.RECIPE,
                algorithm=s.ALGORITHM,
                target=s.TARGET,
                host=s.TARGET,
                compiler=compiler,
                environment=digest(
                    s.environment([(k, os.fsencode(v)) for k, v in env.items()])
                ),
            )
            request = {
                "capsule": s.capsule(cap).hex(),
                "environment": s.environment(
                    [(k, os.fsencode(v)) for k, v in env.items()]
                ).hex(),
                "source_paths": [],
            }
            Path(opts["request"]).write_text(json.dumps(request))
            out = root / "out"
            out.mkdir()
            env.update(
                OUT_DIR=str(out),
                CARGO_MANIFEST_DIR=str(HERE.parent / "crates/ckks-train-bench"),
                TARGET=s.TARGET,
                HOST=s.TARGET,
                PROFILE="release",
                OPT_LEVEL="3",
                DEBUG="false",
                CARGO_PKG_VERSION=cap["version"],
                LD_LIBRARY_PATH="/synthetic/cargo-generated",
            )
            result = subprocess.run(
                [str(script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
            )
            self.assertEqual(
                0, result.returncode, result.stderr.decode(errors="replace")
            )
            self.assertEqual(s.capsule(cap), (out / "study_capsule.bin").read_bytes())
            # Each negative starts with a valid request, so HOST/TARGET cannot fail
            # merely because the preceding compiler mutation was left behind.
            for mutation in ("compiler", "feature", "HOST", "TARGET"):
                failout = root / mutation
                failout.mkdir()
                changed = dict(env, OUT_DIR=str(failout))
                request["capsule"] = s.capsule(cap).hex()
                if mutation == "compiler":
                    bad = dict(cap, compiler="wrong")
                    request["capsule"] = s.capsule(bad).hex()
                elif mutation == "feature":
                    changed["CARGO_FEATURE_CUDA"] = "1"
                else:
                    changed[mutation] = "synthetic-wrong-triple"
                Path(opts["request"]).write_text(json.dumps(request))
                result = subprocess.run(
                    [str(script)],
                    env=changed,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=20,
                )
                self.assertNotEqual(0, result.returncode)
                if mutation in ("HOST", "TARGET"):
                    self.assertIn(
                        "compile fact mismatch: " + mutation.lower(),
                        result.stderr.decode(errors="replace"),
                    )
                self.assertFalse((failout / "study_capsule.bin").exists())


if __name__ == "__main__":
    unittest.main()
