"""One-shot read-only capture; invoke only after reviewing a terminal observation."""
from contextlib import ExitStack
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

LEDGER = Path(__file__).resolve().parent
SOURCE = LEDGER / "task-42-full2-terminal-collector.remote.py"
SOURCE_SHA = "50f2cf22ca3f540a86117c06664c6606f23fd9e2532f72cfa2eaabea168058ef"
OBSERVER_SHA = "17550e5d285a03329782d8ca9f3af7ad5b4e39468596d41781607a942572c243"
PREFIX = LEDGER / "task-42-full2-terminal-fetch-v1"
COMMAND = [
    "ssh", "-o", "BatchMode=yes", "-o", "PasswordAuthentication=no",
    "-o", "KbdInteractiveAuthentication=no", "-o", "ConnectionAttempts=1",
    "-o", "ConnectTimeout=90", "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3", "-o", "ControlMaster=no",
    "-o", "ControlPath=/tmp/ckks-train-ssh-%C", "quadruped-gpu",
    "nice -n 10 /usr/bin/python3.14 -I -B -",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_observation(prefix):
    # The root reviews these actual observation outputs; this is not historical attestation.
    prefix = Path(prefix).resolve()
    if prefix.parent != LEDGER or not prefix.name.startswith("post-full-observation-"):
        raise ValueError("expected an existing observer prefix in this ledger")
    raw = Path(str(prefix) + ".stdout").read_bytes()
    transport_raw = Path(str(prefix) + ".meta.json").read_bytes()
    transport = json.loads(transport_raw)
    observation = json.loads(raw)
    if (transport.get("returncode") != 0 or transport.get("stdout_sha256") != sha(raw)
            or transport.get("stdout_bytes") != len(raw)
            or transport.get("source_sha256") != OBSERVER_SHA):
        raise ValueError("observer transport/source/output identity is not verified")
    expected_result = "/home/kmihara/hekit-r41full/results/quadruped/mnist196_ni_b48_full_seed0.json"
    if (observation.get("kind") != "read_only_job_observation_not_scientific_result"
            or observation.get("full2_result_path") != expected_result
            or observation.get("full2_result_exists") is not True):
        raise ValueError("FULL-2 result has not been observed")
    processes = observation.get("processes", [])
    chains = [p for p in processes if p.get("pid") == 1700979]
    if len(chains) != 1 or chains[0].get("expected_start_ticks") != "465769819":
        raise ValueError("original chain observation is missing")
    chain = chains[0]
    gone = chain.get("exists") is False
    terminal = (chain.get("exists") is True and
                chain.get("start_ticks") == "465769819" and chain.get("state") in ("Z", "X"))
    replaced = (chain.get("exists") is True and isinstance(chain.get("start_ticks"), str)
                and chain["start_ticks"].isdigit() and chain["start_ticks"] != "465769819")
    if not (gone or terminal or replaced):
        raise ValueError("original FULL chain is still live or its terminal state is unknown")
    return raw, transport_raw


def main():
    args = sys.argv[1:]
    check_only = len(args) == 2 and args[0] == "--check-only"
    if len(args) != 1 and not check_only:
        raise ValueError("usage: controller.py [--check-only] OBSERVATION_PREFIX")
    source = SOURCE.read_bytes()
    if len(source) != 8408 or sha(source) != SOURCE_SHA:
        raise ValueError("reviewed collector source differs")
    observed, transport = load_observation(args[-1])
    if check_only:
        print("Observation precondition passed; no SSH or output files created.")
        return 0
    suffixes = ("launch.json", "stdout", "stderr", "meta.json", "observation.stdout", "observation.meta.json")
    paths = {s: Path(str(PREFIX) + "." + s) for s in suffixes}
    if any(p.exists() for p in paths.values()):
        raise ValueError("capture prefix already used; no automatic retry")
    metadata = {
        "kind": "operational_collection_controller_not_scientific_result",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "local_controller_pid": os.getpid(), "command": COMMAND,
        "source_sha256": sha(source), "source_bytes": len(source),
        "observer_stdout_sha256": sha(observed), "observer_meta_sha256": sha(transport),
        "automatic_retry": False, "timeout_seconds": 110,
        "timeout_termination_scope": "only the new local SSH child, never a research process",
    }
    # Reserve every output exclusively before launching. A partial reservation is never reused.
    with ExitStack() as stack:
        streams = {s: stack.enter_context(p.open("xb")) for s, p in paths.items()}
        streams["launch.json"].write(json.dumps(metadata, indent=2).encode())
        streams["observation.stdout"].write(observed)
        streams["observation.meta.json"].write(transport)
        for s in ("launch.json", "observation.stdout", "observation.meta.json"):
            streams[s].flush()
        timed_out = False
        status = 125
        try:
            child = subprocess.Popen(COMMAND, stdin=subprocess.PIPE,
                                     stdout=streams["stdout"], stderr=streams["stderr"])
            metadata["local_ssh_pid"] = child.pid
            try:
                child.communicate(source, timeout=110)
            except subprocess.TimeoutExpired:
                timed_out = True
                child.kill()
                child.communicate()
            metadata["ssh_returncode"] = child.returncode
            status = 124 if timed_out else (128 - child.returncode if child.returncode < 0 else child.returncode)
        except OSError as error:
            metadata["launch_or_local_io_error"] = str(error)
        metadata["timed_out"] = timed_out
        metadata["controller_exit_status"] = status
        metadata["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for suffix in ("stdout", "stderr"):
            streams[suffix].flush()
            data = paths[suffix].read_bytes()
            metadata[suffix + "_bytes"] = len(data)
            metadata[suffix + "_sha256"] = sha(data)
        streams["meta.json"].write(json.dumps(metadata, indent=2).encode())
    print(json.dumps(metadata, indent=2))
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError) as error:
        print("Controller refused before/around launch: " + str(error), file=sys.stderr)
        sys.exit(2)
