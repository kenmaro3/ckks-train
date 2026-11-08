"""Local parser/file-capture checks only; never SSH or synthesize research results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

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
