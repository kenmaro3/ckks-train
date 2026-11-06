"""Read-only Git-blob and temporary working-copy tests for the portable input lock."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lock_inputs", ROOT / "tools/lock_experiment_inputs.py")
LOCKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOCKER)


class ReviewedInputLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads((ROOT / "crates/ckks-train-bench/configs/experiment_input_locks.json").read_text())
        cls.git_dir = subprocess.run(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT,
            capture_output=True, check=True, text=True,
        ).stdout.strip()
        cls.blobs = {
            f["path"]: subprocess.run(
                ["git", "show", f"{LOCKER.SOURCE_COMMIT}:{f['path']}"],
                cwd=ROOT, check=True, capture_output=True,
            ).stdout
            for f in cls.lock["files"]
        }

    def test_deterministic_lock_bytes_use_exact_reviewed_git_blobs(self):
        expected = (json.dumps(LOCKER.build(ROOT), indent=2) + "\n").encode()
        self.assertEqual(expected, (ROOT / "crates/ckks-train-bench/configs/experiment_input_locks.json").read_bytes())
        self.assertEqual(expected, (json.dumps(LOCKER.build(ROOT), indent=2) + "\n").encode())
        self.assertEqual(len(self.blobs), 18)
        for entry in self.lock["files"]:
            self.assertEqual(entry["sha256"], hashlib.sha256(self.blobs[entry["path"]]).hexdigest())

    def test_changed_working_config_or_order_cannot_define_its_own_expected_hash(self):
        for changed in [LOCKER.SOURCES[0], "results/quadruped/raw/reboot_emlp23_full/train_indices_emlp3_seed2.txt"]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(prefix="hekit-input-lock-") as tmp:
                root = Path(tmp)
                # Git show only reads the real object database; no index/ref/worktree write.
                (root / ".git").write_text(f"gitdir: {self.git_dir}\n")
                for path, value in self.blobs.items():
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(value)
                target = root / changed
                target.write_bytes(target.read_bytes() + b"\nchanged working bytes\n")
                with self.assertRaisesRegex(ValueError, "working input differs from reviewed blob"):
                    LOCKER.build(root)
                self.assertFalse((root / "crates/ckks-train-bench/configs/experiment_input_locks.json").exists())
                self.assertTrue(target.read_bytes().endswith(b"changed working bytes\n"))

    def test_missing_reviewed_git_blob_is_error_without_fetch_or_fallback(self):
        with patch.object(LOCKER, "SOURCE_COMMIT", "0" * 40):
            with self.assertRaises(subprocess.CalledProcessError):
                LOCKER.build(ROOT)


if __name__ == "__main__":
    unittest.main()
