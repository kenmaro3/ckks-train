"""Pure broker tests plus one explicitly bounded temporary Git ignore integration test."""
import io
import struct
import unittest
from unittest.mock import patch
import study_reference_build_support as support
from study_runtime_broker import Broker, Invalid


def frame(tag, sequence, payload):
    return struct.pack("<IBQ", 9 + len(payload), tag, sequence) + payload


class Stream:
    def __init__(self, data):
        self.input = io.BytesIO(data)
        self.output = bytearray()

    def recv(self, count):
        return self.input.read(min(count, 1))

    def sendall(self, data):
        self.output.extend(data)


class RuntimeBrokerTests(unittest.TestCase):
    def test_runtime_parent_boundary_rejects_reparent_and_uid_change(self):
        with patch("os.geteuid", return_value=1000):
            broker = Broker(Stream(b""), 2, 2, parent_pid=11)
        with patch("os.getppid", return_value=11), patch("os.getuid", return_value=1000), patch("os.geteuid", return_value=1000):
            broker.check_parent()
        for parent, real, effective in [(1, 1000, 1000), (11, 0, 1000), (11, 1000, 0), (11, 0, 0)]:
            with patch("os.getppid", return_value=parent), patch("os.getuid", return_value=real), patch("os.geteuid", return_value=effective):
                with self.assertRaises(Invalid):
                    broker.check_parent()
    def test_v2_build_recipe_classification(self):
        reference = support._classify("study-reference-linux-clean-v2", "study-reference-session-v1")
        train = support._classify("study-train-cpu-serial-linux-clean-v2", "study-train-session-v1")
        self.assertEqual((reference.binary, train.binary), ("study_reference", "study_train"))
        self.assertEqual((reference.features, train.features), ((), ()))
        self.assertEqual(support._classify("study-reference-linux-clean-v1", "study-reference-session-v1").name, "study-reference-linux-clean-v1")

    def test_runtime_git_uses_broker_only(self):
        stream = Stream(frame(4, 2, struct.pack("<Q", 0) + b"output"))
        broker = Broker(stream, 2, 2)
        with patch.object(support, "_RUNTIME_BROKER", broker), patch.object(
            support.subprocess, "Popen", side_effect=AssertionError("runtime Popen")
        ):
            self.assertEqual(support.bounded_run(["/git", "rev-parse", "HEAD"], "/repo", {}), b"output")
        self.assertEqual(stream.output[4], 3)

    def test_build_git_subprocess_stays_explicit(self):
        process = unittest.mock.MagicMock()
        process.__enter__.return_value = process
        process.stdout.read.return_value = b"build"
        process.wait.return_value = 0
        with patch.object(support, "_RUNTIME_BROKER", None), patch.object(support.subprocess, "Popen", return_value=process) as popen:
            self.assertEqual(support.bounded_run(["/git"], "/repo", {}), b"build")
            popen.assert_called_once()

    def test_exact_git_argv(self):
        broker = Broker(Stream(frame(4, 2, struct.pack("<Q", 0))), 2, 2)
        for argv in [["x"] * 257, ["x" * 4097], ["a\0b"]]:
            with self.assertRaises(Invalid):
                broker.run(argv, "/repo", {}, 20)

    def test_fragmented_frames_and_sequence(self):
        broker = Broker(Stream(frame(4, 2, struct.pack("<Q", 0) + b"abc")), 2, 2)
        self.assertEqual(broker.run(["/git"], "/repo", {}, 3), b"abc")
        wrong = Broker(Stream(frame(4, 3, struct.pack("<Q", 0))), 2, 2)
        with self.assertRaises(Invalid):
            wrong.run(["/git"], "/repo", {}, 3)

    def test_eof_stops_and_keeps_hold(self):
        broker = Broker(Stream(b""), 2, 2)
        with patch("os.close", side_effect=AssertionError("closing hold")):
            with self.assertRaises(Invalid):
                broker.run(["/git"], "/repo", {}, 3)


class OperationalIgnoreTests(unittest.TestCase):
    def test_fixed_runtime_outputs_do_not_hide_untracked_source(self):
        import os
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path
        git = shutil.which("git")
        self.assertIsNotNone(git)
        rules = (Path(__file__).resolve().parent.parent / ".gitignore").read_bytes()
        with tempfile.TemporaryDirectory(prefix="hekit-operational-ignore-") as name:
            root = Path(name).resolve()
            env = {"PATH": os.defpath, "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
            def actual(argv):
                result = subprocess.run([git, "-c", "init.templateDir=", *argv], cwd=root, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                self.assertLessEqual(len(result.stdout) + len(result.stderr), 65536)
                return result.stdout
            actual(["init", "--quiet"])
            (root / ".git/info").mkdir(exist_ok=True)
            (root / ".git/info/exclude").write_bytes(rules)
            for relative in ("results/resource/unit/pending.bin", "results/native-probes/unit/pidfd-owned-unreaped/command.json"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            def untracked():
                return actual(["ls-files", "--others", "--exclude-standard", "-z"])
            self.assertEqual(untracked(), b"")
            # HEAD/diff are deliberately inert here; the tested gate consumes the
            # actual Git untracked output, never an invented clean result.
            def source_gate(argv, _root, _env):
                if argv[-2:] == ["rev-parse", "--show-toplevel"]:
                    return os.fsencode(root) + b"\n"
                if argv[-2:] == ["rev-parse", "HEAD"]:
                    return b"a" * 40 + b"\n"
                if "diff" in argv:
                    return b""
                self.assertEqual(argv[-4:], ["ls-files", "--others", "--exclude-standard", "-z"])
                return untracked()
            with patch.object(support, "bounded_run", side_effect=source_gate):
                self.assertEqual(support.git_state(root, git, env)[0], "a" * 40)
                source = root / "crates/ckks-train-bench/src/unregistered.rs"
                source.parent.mkdir(parents=True)
                source.write_bytes(b"x")
                self.assertEqual(untracked(), b"crates/ckks-train-bench/src/unregistered.rs\0")
                with self.assertRaises(support.Invalid):
                    support.git_state(root, git, env)


if __name__ == "__main__":
    unittest.main()
