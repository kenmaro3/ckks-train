"""Finite R48/R52 terminal capture preparation; operational originals, never science."""
import base64
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import stat
import sys

RUNS = "/home/kmihara/hekit-runs"
STATE = RUNS + "/post-full-v1"
TREES = {family: "/home/kmihara/hekit-post-full-clones/hekit-post-full-" + family
         for family in ("r48", "r52")}
COMMITS = {"r48": "0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3",
           "r52": "9305d2e1f4f9999296cd4b4aaf55ea631fbca78e"}
PER_FILE = 8 * 1024 * 1024
TOTAL = 32 * 1024 * 1024
EXPECTED_CONFIG = {'runs': '/home/kmihara/hekit-runs', 'full_tree': '/home/kmihara/hekit-r41full', 'full_commit': 'ff2c2ebc9d7946490bcf63f8b6be9f45fe67d49f', 'not_before_epoch': 1793917065.19, 'chain_pid': 1700979, 'chain_start_ticks': '465769819', 'chain_log': '/home/kmihara/hekit-runs/logs/chain10.log', 'bundle': '/home/kmihara/hekit-runs/post-full-09202cf.bundle', 'bundle_sha256': 'bf6d2a5c5414fd7ba9f0aaacf2223addf56691483b10de7022fc734e88bdf91d', 'clone_root': '/home/kmihara/hekit-post-full-clones', 'state': '/home/kmihara/hekit-runs/post-full-v1'}


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
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite operational JSON")
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
    finite(obj)
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


def registry(family):
    """Compiled deployed-queue invocations; no paths or argv supplied by the caller."""
    if family == "r48":
        tests = ["results/quadruped/raw/R-48-1/" + stem + "_" + backend + ".json"
                 for stem in ("n16_train_full_keys", "n16_train_level7", "small_insecure",
                              "iris_seal_2020_timing", "iris_mini_boot_ni")
                 for backend in ("cuda", "cpu")]
        def ablation(backend, opt, output, full=False):
            argv = ["target/release/ablation", "--backend", backend, "--params", "n16_train"]
            argv += [] if full else ["--top-level", "7"]
            argv += ["--optimizer", opt, "--iters", "3" if full else "5", "--warmup", "1",
                     "--data", TREES[family] + "/data/mnist", "--out", output]
            return argv, [output]
        return {
            "post-full-r48-build": (False, [(["cargo", "build", "-p", "hekit-bench", "--features", "cuda", "--release", "--bins", "--tests"], [])]),
            "R-48-1": (True, [(["cargo", "test", "-p", "hekit-bench", "--features", "cuda", "--release", "--test", "mnist_cuda", "--test", "iris_cuda", "--", "--ignored", "--nocapture", "--test-threads=1"], tests)]),
            "R-48-2": (True, [ablation("cuda", "nag", "results/quadruped/ablation_cuda.json")]),
            "R-48-2-sgd": (True, [ablation("cuda", "sgd", "results/quadruped/ablation_cuda_sgd.json")]),
            "R-48-3": (False, [ablation("cpu", "nag", "results/quadruped/ablation_cpu_l7.json"),
                               ablation("cpu", "sgd", "results/quadruped/ablation_cpu_l7_sgd.json")]),
            "R-48-4": (False, [ablation("cpu", "nag", "results/quadruped/ablation_cpu.json", full=True)]),
        }
    if family == "r52":
        jobs = {"post-full-r52-build": (False, [(["cargo", "build", "--release", "-p", "hekit-bench", "--features", "parallel", "--bins"], [])])}
        for number, blocks in ((2, "64"), (3, "128,64")):
            output = f"results/quadruped/mnist196_emlp{number}_ni_b48_subset1008_seed0.json"
            command = ["target/release/mnist", "--mode", "noninteractive", "--placement", "when_needed", "--batch", "48", "--faithful", "--blocks", blocks, "--hidden", "32", "--dataset-npy", "results/quadruped/data/reboot_emlp1_full", "--indices", f"results/reboot_emlp{number}_seed0_steps1-21_indices.txt", "--init-weights", f"results/quadruped/raw/reboot_emlp23_full/init_weights_emlp{number}_seed0", "--init-weights-manifest", "results/quadruped/raw/reboot_emlp23_full/config_values.json", "--epochs", "1", "--transpose", "reuse", "--out", output]
            jobs[f"R-52-{number - 1}"] = (False, [(command, [output])])
        return jobs
    raise ValueError("expected exactly one compiled family: r48 or r52")


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("missing terminal timestamp")
    date = datetime.datetime.fromisoformat(value)
    if date.tzinfo is None:
        raise ValueError("terminal timestamp lacks timezone")
    return date.timestamp()


def check_terminal(family, name, record, meta, gate_at, specs):
    gpu, variants = specs[name]
    command = record.get("command")
    matched = [(cmd, outputs) for cmd, outputs in variants if command == cmd]
    if (record.get("job") != name or record.get("workdir") != TREES[family] or
            len(matched) != 1 or record.get("log") != RUNS + "/logs/" + name + ".log" or
            type(record.get("rc")) is not int or not 0 <= record["rc"] <= 255 or
            type(record.get("wrapper_rc")) is not int or not -64 <= record["wrapper_rc"] <= 255 or
            type(record.get("skip")) is not bool or
            not isinstance(record.get("skip_lines"), list) or
            not all(isinstance(line, str) for line in record["skip_lines"]) or
            not isinstance(record.get("evidence_lines"), list) or
            not all(isinstance(line, str) for line in record["evidence_lines"]) or
            record.get("status") not in ("done", "done_with_skips", "failed", "failed: device OOM",
                                        "failed: RSS watchdog", "skipped", "partial", "failed: output checks/provenance")):
        raise ValueError("queue job identity/type mismatch")
    if (meta.get("job") != name or meta.get("workdir") != TREES[family] or
            not isinstance(meta.get("command"), str) or shlex.split(meta["command"]) != command or
            type(meta.get("exit_status")) is not int or meta["exit_status"] != record["rc"] or
            type(meta.get("watchdog_killed")) is not bool or
            type(meta.get("nice")) is not int or meta["nice"] != 10 or
            type(meta.get("threads_cap")) is not int or meta["threads_cap"] != 12 or
            meta.get("gpu_job") is not gpu or
            meta.get("env_thread_caps") != {key: "12" for key in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "CARGO_BUILD_JOBS")}):
        raise ValueError("runner terminal identity/type mismatch")
    pin = COMMITS[family]
    if meta.get("hekit_synced_commit") != ["commit " + pin[:7], "full " + pin, "dirty_files 0"]:
        raise ValueError("runner source identity mismatch")
    start, end = timestamp(meta.get("start")), timestamp(meta.get("end"))
    # Runner date -Is is whole seconds; same-second subsecond order is unresolved.
    wall = meta.get("wall_s_runner")
    if (not math.floor(gate_at) <= start <= end <= datetime.datetime.now(datetime.timezone.utc).timestamp() or
            type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0 or
            abs(wall - (end - start)) > 2):
        raise ValueError("terminal timestamps/wall outside released gate/current observation")
    return matched[0]


def check_log_status(family, name, record, meta, raw, has_output):
    text = raw.decode("utf-8")
    skips = [line for line in text.splitlines() if "SKIP" in line]
    if record["skip"] is not bool(skips) or record["skip_lines"] != skips:
        raise ValueError("queue skip/log disagreement")
    evidence = [line for line in text.splitlines() if re.search(r"SKIP|max.*W_gpu.*W_cpu|test result:", line)]
    if record["evidence_lines"] != evidence:
        raise ValueError("queue evidence/log disagreement")
    if not meta["watchdog_killed"] and any(line.startswith("WATCHDOG:") for line in text.splitlines()):
        raise ValueError("watchdog/log disagreement")
    rc, status = record["rc"], record["status"]
    expected = ("failed: RSS watchdog" if meta["watchdog_killed"] else
                "failed: device OOM" if rc and re.search(r"device error|out of memory", text, re.I) else
                "skipped" if rc == 3 and skips else "failed" if rc else
                "done_with_skips" if skips else "done")
    # Queue has two overrides independent of ordinary log classification.
    if status == "partial" and family == "r52" and name in ("R-52-1", "R-52-2") and not meta["watchdog_killed"]:
        return
    if (status == "failed: output checks/provenance" and rc == 0 and has_output and
            isinstance(record.get("output_error"), str) and record["output_error"]):
        return
    if status != expected:
        raise ValueError("queue status contradicts runner/log")


def collect(family):
    files, jobs, missing = {}, {}, []
    total = 0
    current_path, stage = None, "family-selection"
    capture = {"kind": "operational_capture_not_scientific_result", "scientific_acceptance": False,
               "family": family, "files": files, "jobs": jobs, "missing_artifacts": missing, "exit_code": 2,
               "excluded": ["R47 artifacts", "other family artifacts", "profiler binaries", "continuously growing whole-queue log"]}

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
        specs = registry(family)
        stage = "queue-owner"
        owner = parse_object(take("owner.json", STATE + "/owner.json"))
        if (type(owner.get("pid")) is not int or owner["pid"] != 1739838 or
                type(owner.get("started")) not in (int, float) or owner["started"] != 1793929784.594468):
            raise ValueError("registered queue owner mismatch")
        stage = "released-gate"
        gate = parse_object(take("gate-released.json", STATE + "/gate-released.json"))
        gate_at = gate.get("at")
        config = gate.get("config")
        if (config != EXPECTED_CONFIG or
                any(type(config[key]) is not type(value) for key, value in EXPECTED_CONFIG.items()) or
                type(gate_at) not in (int, float) or
                not math.isfinite(gate_at) or not owner["started"] <= gate_at <= datetime.datetime.now(datetime.timezone.utc).timestamp()):
            raise ValueError("released gate/config mismatch")
        stage = "queue-terminal"
        terminal_raw = take("terminal.json", STATE + "/terminal.json", optional=True)
        if terminal_raw is not None:
            terminal = parse_object(terminal_raw)
            ended = terminal.get("ended")
            if (terminal.get("status") not in ("completed", "completed_with_failures", "failed") or
                    type(ended) not in (int, float) or not math.isfinite(ended) or
                    not gate_at <= ended <= datetime.datetime.now(datetime.timezone.utc).timestamp() or
                    (terminal["status"] == "failed" and not isinstance(terminal.get("error"), str)) or
                    (terminal["status"] != "failed" and not isinstance(terminal.get("jobs"), list))):
                raise ValueError("malformed queue terminal summary")
        optimizer = None
        if family == "r48":
            stage = "R-48-optimizer"
            optimizer_raw = take("R-48-optimizer.json", STATE + "/R-48-optimizer.json", optional=True)
            if optimizer_raw is not None:
                optimizer = parse_object(optimizer_raw)
                if optimizer not in ({"cpu_optimizer": "nag"}, {"cpu_optimizer": "sgd"}):
                    raise ValueError("malformed optimizer record")
        for name in specs:
            stage = name
            state_path = STATE + "/" + name + ".json"
            key = name + "/"
            raw = take(key + "state.json", state_path, optional=True)
            if raw is None:
                jobs[name] = {"capture_state": "no_terminal_record_observed"}
                continue
            record = parse_object(raw)
            out = RUNS + "/out/" + name
            meta = parse_object(take(key + "meta.json", out + "/meta.json"))
            command, outputs = check_terminal(family, name, record, meta, gate_at, specs)
            if name == "R-48-3" and optimizer is not None and command[command.index("--optimizer") + 1] != optimizer["cpu_optimizer"]:
                raise ValueError("CPU command disagrees with optimizer record")
            for filename in ("cmd.txt", "workdir.txt", "post.txt"):
                raw = take(key + filename, out + "/" + filename)
                if filename == "cmd.txt" and shlex.split(raw.decode().strip()) != command:
                    raise ValueError("cmd.txt mismatch")
                if filename == "workdir.txt" and raw.decode().strip() != TREES[family]:
                    raise ValueError("workdir.txt mismatch")
                if filename == "post.txt":
                    lines = raw.decode().splitlines()
                    markers = [line for line in lines if line.startswith("JOB_DONE")]
                    if (not lines or markers != ["JOB_DONE rc=" + str(meta["exit_status"])] or
                            lines[-1] != markers[0] or not lines[0].startswith("date: ") or
                            not timestamp(meta["end"]) <= timestamp(lines[0][6:]) <= datetime.datetime.now(datetime.timezone.utc).timestamp()):
                        raise ValueError("post.txt terminal marker/date mismatch")
            raw_log = take(key + "program.log", RUNS + "/logs/" + name + ".log")
            check_log_status(family, name, record, meta, raw_log, bool(outputs) and name != "R-48-1")
            ancillary = ("pre.txt", "time.txt", "rss.tsv", "time.pid", "synced-commit.txt")
            if family == "r52":
                ancillary += ("partial.txt", "projection.txt", "partial-identities.json")
            for filename in ancillary:
                take(key + filename, out + "/" + filename, optional=True)
            for relative in outputs:
                # The output mapping is selected only by exact compiled argv equality.
                # Failed/malformed scientific JSON is preserved, not parsed or rewritten.
                take(key + "binary/" + relative.rsplit("/", 1)[1], TREES[family] + "/" + relative, optional=True)
            same_again(key + "meta.final.json", out + "/meta.json", key + "meta.json")
            same_again(key + "state.final.json", state_path, key + "state.json")
            jobs[name] = {"capture_state": "terminal_artifacts_captured", "runner_exit": meta["exit_status"],
                          "wrapper_exit": record["wrapper_rc"], "watchdog_killed": meta["watchdog_killed"],
                          "queue_status": record["status"], "scientific_acceptance": False}
        stage = "final-queue-records"
        if family == "r48" and optimizer is not None:
            same_again("R-48-optimizer.final.json", STATE + "/R-48-optimizer.json", "R-48-optimizer.json")
        if terminal_raw is not None:
            same_again("terminal.final.json", STATE + "/terminal.json", "terminal.json")
        same_again("owner.final.json", STATE + "/owner.json", "owner.json")
        same_again("gate-released.final.json", STATE + "/gate-released.json", "gate-released.json")
        capture["exit_code"] = 0
    except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
        capture["error"] = {"stage": stage, "path": current_path, "type": type(exc).__name__, "message": str(exc)}
    capture["original_bytes_collected"] = total
    capture["observed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return capture


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    captured = collect(args[0] if len(args) == 1 else None)
    print(json.dumps(captured, separators=(",", ":")))
    return captured["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
