"""Fixed R47 read-only terminal evidence capture. No scientific acceptance."""
import base64
import datetime
import hashlib
import json
import math
import os
import shlex
import stat
import sys

RUNS = "/home/kmihara/hekit-runs"
STATE = RUNS + "/post-full-v1"
TREE = "/home/kmihara/hekit-post-full-clones/hekit-post-full-r47"
COMMIT = "bd5cc493db166a282d26b05b1e069f2a0ee629ef"
PER_FILE = 8 * 1024 * 1024
TOTAL = 32 * 1024 * 1024
EXPECTED_CONFIG = {'runs': '/home/kmihara/hekit-runs', 'full_tree': '/home/kmihara/hekit-r41full', 'full_commit': 'ff2c2ebc9d7946490bcf63f8b6be9f45fe67d49f', 'not_before_epoch': 1793917065.19, 'chain_pid': 1700979, 'chain_start_ticks': '465769819', 'chain_log': '/home/kmihara/hekit-runs/logs/chain10.log', 'bundle': '/home/kmihara/hekit-runs/post-full-09202cf.bundle', 'bundle_sha256': 'bf6d2a5c5414fd7ba9f0aaacf2223addf56691483b10de7022fc734e88bdf91d', 'clone_root': '/home/kmihara/hekit-post-full-clones', 'state': '/home/kmihara/hekit-runs/post-full-v1'}
JOBS = {
    "post-full-r47-build": (False, ["cargo", "build", "-p", "hekit-cuda", "--features", "cuda", "--release", "--example", "batch_streams"]),
    "R-47-3-example": (True, ["target/release/examples/batch_streams"]),
    "R-47-3-nsys": (True, ["/usr/local/cuda-13.1/bin/nsys", "profile", "--trace=cuda", "--force-overwrite=true", "-o", "results/quadruped/raw/R-47-3/r47_3_batch", "target/release/examples/batch_streams"]),
    "R-47-3-tests": (True, ["cargo", "test", "-p", "hekit-cuda", "--features", "cuda", "--release", "--", "--ignored", "--nocapture", "--test-threads=1"]),
}


def unique_object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate JSON key: " + k)
        out[k] = v
    return out

def stamp(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def read_stable(path, byte_limit=PER_FILE):
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
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=held[-1])
        held.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise ValueError("capture is not an own regular file")
        if before.st_size > min(PER_FILE, byte_limit):
            raise ValueError("capture size %d exceeds remaining bound %d" %
                             (before.st_size, min(PER_FILE, byte_limit)))
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

def parse_object(raw):
    obj = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


def check_terminal(name, record, meta, gate_at):
    gpu, command = JOBS[name]
    if (record.get("job") != name or record.get("workdir") != TREE or
            record.get("command") != command or
            record.get("log") != RUNS + "/logs/" + name + ".log" or
            type(record.get("rc")) is not int or record["rc"] < 0 or
            not isinstance(record.get("status"), str)):
        raise ValueError("queue job identity mismatch")
    if (meta.get("job") != name or meta.get("workdir") != TREE or
            not isinstance(meta.get("command"), str) or
            shlex.split(meta["command"]) != command or
            type(meta.get("exit_status")) is not int or meta["exit_status"] != record["rc"] or
            type(meta.get("watchdog_killed")) is not bool or
            meta.get("nice") != 10 or meta.get("threads_cap") != 12 or
            meta.get("gpu_job") is not gpu):
        raise ValueError("runner terminal identity mismatch")
    synced = meta.get("hekit_synced_commit")
    if not isinstance(synced, list) or "full " + COMMIT not in synced or "dirty_files 0" not in synced:
        raise ValueError("runner source identity mismatch")
    dates = []
    for key in ("start", "end"):
        value = meta.get(key)
        if not isinstance(value, str):
            raise ValueError("missing terminal timestamp")
        date = datetime.datetime.fromisoformat(value)
        if date.tzinfo is None:
            raise ValueError("terminal timestamp lacks timezone")
        dates.append(date.timestamp())
    if not gate_at <= dates[0] <= dates[1] <= datetime.datetime.now(datetime.timezone.utc).timestamp():
        raise ValueError("terminal timestamps outside released gate/current observation")


def collect():
    files, jobs, missing = {}, {}, []
    total = 0
    current_path, stage = STATE + "/owner.json", "queue-owner"
    capture = {"kind": "operational_capture_not_scientific_result", "scientific_acceptance": False,
               "files": files, "jobs": jobs, "missing_artifacts": missing, "exit_code": 2,
               "excluded": ["optional profiler binaries", "continuously growing whole-queue log", "R48/R52 artifacts"]}

    def take(key, path, optional=False):
        nonlocal total, current_path
        current_path = path
        try:
            raw, entry = read_stable(path, byte_limit=TOTAL - total)
        except FileNotFoundError:
            if optional:
                missing.append(key)
                return None
            raise
        total += len(raw)
        files[key] = entry
        return raw

    def same_again(key, path, prior_key):
        raw = take(key, path)
        if (files[key]["sha256"], files[key]["read_stamp"]) != (files[prior_key]["sha256"], files[prior_key]["read_stamp"]):
            raise ValueError("terminal record changed between observations")
        return raw

    try:
        owner = parse_object(take("owner.json", STATE + "/owner.json"))
        if type(owner.get("pid")) is not int or owner["pid"] != 1739838 or owner.get("started") != 1793929784.594468:
            raise ValueError("registered queue owner mismatch")
        stage = "released-gate"
        gate = parse_object(take("gate-released.json", STATE + "/gate-released.json"))
        gate_at = gate.get("at")
        if (gate.get("config") != EXPECTED_CONFIG or type(gate_at) not in (int, float) or
                not math.isfinite(gate_at) or gate_at < owner["started"]):
            raise ValueError("released gate/config mismatch")
        for name in JOBS:
            stage = name
            state_path = STATE + "/" + name + ".json"
            key = name + "/"
            raw = take(key + "state.json", state_path, optional=True)
            if raw is None:
                jobs[name] = {"capture_state": "no_terminal_record_observed"}
                continue
            record = parse_object(raw)
            if name == "R-47-3-nsys" and record == {"status": "skipped: nsys unavailable"}:
                same_again(key + "state.final.json", state_path, key + "state.json")
                jobs[name] = {"capture_state": "nsys_unavailable", "queue_status": record["status"]}
                continue
            out = RUNS + "/out/" + name
            meta = parse_object(take(key + "meta.json", out + "/meta.json"))
            check_terminal(name, record, meta, gate_at)
            for filename in ("cmd.txt", "workdir.txt", "post.txt", "pre.txt", "time.txt", "rss.tsv", "time.pid", "synced-commit.txt"):
                raw = take(key + filename, out + "/" + filename, optional=filename not in ("cmd.txt", "workdir.txt", "post.txt"))
                if raw is None:
                    continue
                if filename == "cmd.txt" and shlex.split(raw.decode().strip()) != JOBS[name][1]:
                    raise ValueError("cmd.txt mismatch")
                if filename == "workdir.txt" and raw.decode().strip() != TREE:
                    raise ValueError("workdir.txt mismatch")
                if filename == "post.txt" and (not raw.splitlines() or raw.splitlines()[-1] != ("JOB_DONE rc=" + str(meta["exit_status"])).encode()):
                    raise ValueError("post.txt terminal marker mismatch")
            take(key + "program.log", RUNS + "/logs/" + name + ".log")
            same_again(key + "meta.final.json", out + "/meta.json", key + "meta.json")
            same_again(key + "state.final.json", state_path, key + "state.json")
            jobs[name] = {"capture_state": "terminal_artifacts_captured", "runner_exit": meta["exit_status"],
                          "watchdog_killed": meta["watchdog_killed"], "queue_status": record["status"]}
        stage = "final-owner-gate"
        same_again("owner.final.json", STATE + "/owner.json", "owner.json")
        same_again("gate-released.final.json", STATE + "/gate-released.json", "gate-released.json")
        capture["exit_code"] = 0
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        capture["error"] = {"stage": stage, "path": current_path, "type": type(exc).__name__, "message": str(exc)}
    capture["original_bytes_collected"] = total
    capture["observed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return capture


if __name__ == "__main__":
    captured = collect()
    print(json.dumps(captured, separators=(",", ":")))
    sys.exit(captured["exit_code"])
