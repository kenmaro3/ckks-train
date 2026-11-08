"""Local parser/file-capture checks only; never SSH or synthesize research results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import base64
import os
import subprocess
import sys

SOURCE = Path(__file__).with_name("task-42-full2-terminal-collector.remote.py")
spec = importlib.util.spec_from_file_location("full2_collection", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def command(items=None):
    args = ["./target/release/mnist"]
    for key, value in (items or list(m.EXPECTED.items())):
        args.append(key)
        if value is not None:
            args.append(value)
    return " ".join(args)


def metadata():
    return {"job": m.JOB, "workdir": m.WORKDIR, "command": command(),
            "start": "2026-01-01T00:00:00+00:00", "end": "2026-01-01T00:00:01+00:00",
            "exit_status": 0, "watchdog_killed": False}


class Decisions(unittest.TestCase):
    def entry(self, data, path):
        return data, {"remote_path": path, "bytes": len(data), "read_stamp": [1],
                      "base64": base64.b64encode(data).decode()}

    def test_refused_initial_metadata_retains_bytes(self):
        active = metadata(); del active["end"]
        for raw in (json.dumps(active).encode(), b"{bad", b"[]", b"null"):
            with self.subTest(raw=raw), patch.object(m, "read_stable", return_value=self.entry(raw, m.OUT + "/meta.json")):
                result = m.collect()
                self.assertEqual(result["kind"], "operational_collection_refusal")
                self.assertEqual(base64.b64decode(result["files"]["meta.json"]["base64"]), raw)
                self.assertIs(result["scientific_acceptance"], False)

    def test_offending_command_and_earlier_metadata_retained(self):
        raw = json.dumps(metadata()).encode()
        bad_command = b"./target/release/mnist --batch 64"
        def read(path, **kwargs):
            return self.entry(raw if path.endswith("/meta.json") else bad_command, path)
        with patch.object(m, "read_stable", side_effect=read):
            result = m.collect()
        self.assertEqual(result["kind"], "operational_collection_refusal")
        self.assertEqual(set(result["files"]), {"meta.json", "cmd.txt"})
        self.assertEqual(base64.b64decode(result["files"]["cmd.txt"]["base64"]), bad_command)

    def test_later_io_error_preserves_prior_bytes(self):
        raw = json.dumps(metadata()).encode()
        def read(path, **kwargs):
            if path.endswith("/meta.json"):
                return self.entry(raw, path)
            raise PermissionError("local test refusal")
        with patch.object(m, "read_stable", side_effect=read):
            result = m.collect()
        self.assertEqual(result["kind"], "operational_collection_refusal")
        self.assertEqual(set(result["files"]), {"meta.json"})
        self.assertEqual(result["error"]["path"], m.OUT + "/cmd.txt")

    def test_final_metadata_drift_retains_both_observations(self):
        initial = json.dumps(metadata()).encode()
        item = metadata(); item["exit_status"] = 1
        final = json.dumps(item).encode()
        count = 0
        def read(path, **kwargs):
            nonlocal count
            if path.endswith("/meta.json"):
                count += 1
                return self.entry(initial if count == 1 else final, path)
            if path.endswith("/cmd.txt"):
                return self.entry(command().encode(), path)
            if path.endswith("/workdir.txt"):
                return self.entry(m.WORKDIR.encode(), path)
            return self.entry(b"local opaque test bytes", path)
        with patch.object(m, "read_stable", side_effect=read):
            result = m.collect()
        self.assertEqual(result["kind"], "operational_collection_refusal")
        self.assertEqual(base64.b64decode(result["files"]["meta.json"]["base64"]), initial)
        self.assertEqual(base64.b64decode(result["files"]["meta.final.json"]["base64"]), final)
        self.assertIn(m.RESULT, result["files"])

    def test_fifo_refuses_without_writer(self):
        with tempfile.TemporaryDirectory(prefix="full2-collector-test-") as directory:
            fifo = Path(directory).resolve() / "fifo"; os.mkfifo(fifo)
            child = "import runpy,sys; m=runpy.run_path(sys.argv[1]); m['read_stable'](sys.argv[2])"
            completed = subprocess.run([sys.executable, "-B", "-c", child, str(SOURCE), str(fifo)],
                                       capture_output=True, timeout=2)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"not an own regular file", completed.stderr)

    def test_flag_order_is_not_identity(self):
        self.assertEqual(m.command_identity(command(list(reversed(list(m.EXPECTED.items()))))), m.EXPECTED)

    def test_command_refuses_extra_duplicate_missing_or_changed_flags(self):
        for value in (command() + " --batch 48", command() + " --extra 1",
                      command().replace("--batch 48", "--batch 64"),
                      command().replace("--faithful", ""),
                      command().replace("./target/release/mnist", "/tmp/mnist")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.command_identity(value)

    def test_terminal_identity_and_time(self):
        self.assertEqual(m.terminal_metadata(json.dumps(metadata()))["exit_status"], 0)
        for key, value in (("job", "R-41-FULL-1"), ("workdir", "/tmp"),
                           ("exit_status", None), ("exit_status", False),
                           ("watchdog_killed", 0), ("end", None),
                           ("end", "2025-01-01T00:00:00+00:00"),
                           ("end", "2026-01-01T00:00:01")):
            item = metadata(); item[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                m.terminal_metadata(json.dumps(item))

    def test_failure_metadata_is_capturable_not_success(self):
        item = metadata(); item.update(exit_status=137, watchdog_killed=True)
        parsed = m.terminal_metadata(json.dumps(item))
        self.assertEqual(parsed["exit_status"], 137)
        self.assertIs(parsed["watchdog_killed"], True)

    def test_duplicate_metadata_keys_rejected(self):
        raw = json.dumps(metadata()).replace('"job":', '"job":"other","job":', 1)
        with self.assertRaises(ValueError):
            m.terminal_metadata(raw)

    def test_actual_local_bytes_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory(prefix="full2-collector-test-") as directory:
            root = Path(directory).resolve()
            path = root / "capture"; path.write_bytes(b"local test data\x00\n")
            data, entry = m.read_stable(str(path))
            self.assertEqual(data, path.read_bytes())
            self.assertEqual(entry["bytes"], len(data))
            link = root / "link"; link.symlink_to(path)
            with self.assertRaises(OSError):
                m.read_stable(str(link))
            nested = root / "dir"; nested.mkdir()
            (nested / "capture").write_bytes(b"local")
            linked_dir = root / "linked-dir"; linked_dir.symlink_to(nested)
            with self.assertRaises(OSError):
                m.read_stable(str(linked_dir / "capture"))

    def test_per_file_bound_without_allocating_oversize_payload(self):
        with tempfile.TemporaryDirectory(prefix="full2-collector-test-") as directory:
            path = Path(directory).resolve() / "sparse"
            with path.open("wb") as f:
                f.truncate(m.PER_FILE + 1)
            with self.assertRaises(ValueError):
                m.read_stable(str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
