"""Read-only FULL-2 terminal capture. Never starts/stops a research process."""
import base64
import datetime
import hashlib
import json
import os
import shlex
import stat
import sys

JOB = "R-41-FULL-2"
WORKDIR = "/home/kmihara/hekit-r41full"
OUT = "/home/kmihara/hekit-runs/out/" + JOB
RESULT = "mnist196_ni_b48_full_seed0.json"
EXPECTED = {
    "--mode": "noninteractive", "--placement": "when_needed",
    "--batch": "48", "--faithful": None,
    "--dataset-npy": "results/quadruped/data/reboot_emlp1_full",
    "--indices": "results/quadruped/raw/reboot_emlp1_full/train_indices.txt",
    "--init-weights": "results/quadruped/raw/reboot_emlp1_full/init_weights_seed0",
    "--init-weights-manifest": "results/quadruped/raw/reboot_emlp1_full/config_values.json",
    "--lr-schedule": "results/quadruped/raw/reboot_emlp1_full/config_values.json",
    "--lr-key": "lr_per_step", "--epochs": "1", "--transpose": "reuse",
    "--out": "results/quadruped/" + RESULT,
}
PER_FILE = 8 * 1024 * 1024
TOTAL = 32 * 1024 * 1024


def unique_object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate JSON key: " + k)
        out[k] = v
    return out


def command_identity(command):
    if not isinstance(command, str):
        raise ValueError("command is not text")
    tokens = shlex.split(command)
    if not tokens or tokens.pop(0) != "./target/release/mnist":
        raise ValueError("unexpected executable")
    found = {}
    while tokens:
        flag = tokens.pop(0)
        if flag not in EXPECTED or flag in found:
            raise ValueError("unexpected or repeated flag: " + flag)
        value = None if EXPECTED[flag] is None else (tokens.pop(0) if tokens else None)
        if value != EXPECTED[flag]:
            raise ValueError("wrong flag value: " + flag)
        found[flag] = value
    if found != EXPECTED:
        raise ValueError("missing required flags")
    return found


def terminal_metadata(raw):
    meta = json.loads(raw, object_pairs_hook=unique_object)
    if meta.get("job") != JOB or meta.get("workdir") != WORKDIR:
        raise ValueError("job/workdir identity mismatch")
    command_identity(meta.get("command"))
    if type(meta.get("exit_status")) is not int or type(meta.get("watchdog_killed")) is not bool:
        raise ValueError("terminal exit/watchdog fields missing or mistyped")
    dates = []
    for key in ("start", "end"):
        value = meta.get(key)
        if not isinstance(value, str):
            raise ValueError("terminal timestamp missing: " + key)
        date = datetime.datetime.fromisoformat(value)
        if date.tzinfo is None:
            raise ValueError("timestamp has no timezone: " + key)
        dates.append(date)
    if dates[1] < dates[0]:
        raise ValueError("end predates start")
    return meta


def stamp(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def read_stable(path):
    """Open only the compiled absolute path, rejecting symlinks at each component."""
    parts = path.split("/")
    if parts[0] or any(p in ("", ".", "..") for p in parts[1:]):
        raise ValueError("noncanonical absolute capture path")
    held = []
    try:
        held.append(os.open("/", os.O_RDONLY | os.O_DIRECTORY))
        for component in parts[1:-1]:
            held.append(os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=held[-1]))
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=held[-1])
        held.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise ValueError("capture is not an own regular file")
        if before.st_size > PER_FILE:
            raise ValueError("capture exceeds per-file bound")
        chunks = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        if stamp(before) != stamp(after) or len(data) != before.st_size:
            raise ValueError("capture changed while read")
        # Reopen from the original absolute path, rejecting replaced directory edges.
        probe = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for i, component in enumerate(parts[1:-1], 1):
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                  dir_fd=probe)
                os.close(probe)
                probe = next_fd
                current, original = os.fstat(probe), os.fstat(held[i])
                if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
                    raise ValueError("capture ancestor changed")
            named = os.stat(parts[-1], dir_fd=probe, follow_symlinks=False)
            if stamp(named) != stamp(before):
                raise ValueError("capture name changed")
        finally:
            os.close(probe)
        return data, {
            "remote_path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "read_stamp": list(stamp(before)), "base64": base64.b64encode(data).decode("ascii"),
        }
    finally:
        for fd in reversed(held):
            os.close(fd)


def collect():
    raw, meta_entry = read_stable(OUT + "/meta.json")
    meta = terminal_metadata(raw)
    files = {"meta.json": meta_entry}
    missing = []
    total = len(raw)
    paths = [(name, OUT + "/" + name) for name in (
        "cmd.txt", "workdir.txt", "pre.txt", "post.txt", "rss.tsv", "time.txt",
        "synced-commit.txt", "projection.txt")]
    paths += [("program.log", "/home/kmihara/hekit-runs/logs/" + JOB + ".log"),
              (RESULT, WORKDIR + "/results/quadruped/" + RESULT)]
    for name, path in paths:
        try:
            data, entry = read_stable(path)
        except FileNotFoundError:
            missing.append(name)
            continue
        total += len(data)
        if total > TOTAL:
            raise ValueError("aggregate capture bound exceeded")
        if name == "cmd.txt":
            command_identity(data.decode().strip())
        if name == "workdir.txt" and data.decode().strip() != WORKDIR:
            raise ValueError("workdir.txt mismatch")
        files[name] = entry
    final_raw, final_entry = read_stable(OUT + "/meta.json")
    if final_raw != raw or final_entry["read_stamp"] != meta_entry["read_stamp"]:
        raise ValueError("terminal metadata changed during collection")
    required = {"cmd.txt", "workdir.txt", "program.log", RESULT}
    complete = not (required & set(missing))
    runner_success = meta["exit_status"] == 0 and meta["watchdog_killed"] is False
    return {
        "kind": "operational_capture_not_scientific_result",
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job": JOB, "runner_success": runner_success,
        "required_artifacts_present": complete,
        "scientific_acceptance": False,
        "missing_artifacts": missing, "files": files,
    }


if __name__ == "__main__":
    try:
        capture = collect()
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"kind": "operational_collection_refusal", "error": str(error)}))
        sys.exit(2)
    print(json.dumps(capture, separators=(",", ":")))
    sys.exit(0 if capture["runner_success"] and capture["required_artifacts_present"] else 3)
