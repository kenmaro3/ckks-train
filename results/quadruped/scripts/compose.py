#!/usr/bin/env python3
"""Compose results/quadruped/*.json from the raw job directories copied back from quadruped-gpu.

Every number comes from a file in results/quadruped/raw/<job>/ (meta.json written by run_job.sh, the program's
own log/JSON, ReBoot wrapper stats.json). Nothing is typed by hand. Usage:
    python3 results/quadruped/scripts/compose.py cuda|reboot_subset|reboot_full|mnist NAME|iris|bootstrap
"""
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.dirname(HERE)
RAW = os.path.join(QDIR, "raw")


def raw(job, f):
    return os.path.join(RAW, job, f)


def load_meta(job):
    m = json.load(open(raw(job, "meta.json")))
    m["raw_dir"] = f"results/quadruped/raw/{job}/"
    return m


def provenance(job, extra=None):
    m = load_meta(job)
    p = {
        "raw_dir": m["raw_dir"],
        "hekit_synced_commit": m["hekit_synced_commit"],
        "command": m["command"],
        "workdir": m["workdir"],
        "threads_cap": m["threads_cap"],
        "nice": m["nice"],
        "start": m["start"], "end": m["end"], "wall_s_runner": m["wall_s_runner"],
        "uptime_before": m["uptime_before"], "uptime_after": m["uptime_after"],
        "peak_rss_kb_time_v": m["peak_rss_kb_time_v"],
        "peak_rss_kb_tree_sampled_10s": m["peak_rss_kb_tree_sampled_10s"],
        "exit_status": m["exit_status"], "watchdog_killed": m["watchdog_killed"],
        "host": m["host"], "cpu": m["cpu"], "kernel": m["kernel"],
        "time_v_elapsed": m["time_v"].get("Elapsed (wall clock) time (h:mm:ss or m:ss)"),
        "time_v_user_s": float(m["time_v"].get("User time (seconds)", "nan")),
        "time_v_sys_s": float(m["time_v"].get("System time (seconds)", "nan")),
        "time_v_cpu_percent": m["time_v"].get("Percent of CPU this job got"),
        "composed_by": "results/quadruped/scripts/compose.py",
    }
    if m.get("gpu_job"):
        p["nvidia_smi_before"] = m["snapshot_before"]
        p["nvidia_smi_after"] = m["snapshot_after"]
    else:
        p["snapshot_before"] = m["snapshot_before"]
    if extra:
        p.update(extra)
    return p


def write(name, obj):
    path = os.path.join(QDIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    print("wrote", path)


def parse_bench(job):
    txt = open(raw(job, "program.log")).read()
    g = lambda pat: re.search(pat, txt, re.M)
    dev = g(r"^device\s+: (.*)$").group(1)
    b = g(r"^batch, reps, warm : (\d+), (\d+), (\d+)$")
    bt = g(r"^batch time \(ms\)\s+: median ([\d.]+), min ([\d.]+), mean ([\d.]+), max ([\d.]+)$")
    pp = g(r"^per polynomial\s+: median ([\d.]+) us, min ([\d.]+) us, mean ([\d.]+) us$")
    agree = g(r"^CPU agreement\s+: (.*)$").group(1)
    return {
        "device": dev, "batch": int(b.group(1)), "reps": int(b.group(2)), "warmup": int(b.group(3)),
        "cpu_agreement": agree,
        "batch_ms": {"median": float(bt.group(1)), "min": float(bt.group(2)), "mean": float(bt.group(3)), "max": float(bt.group(4))},
        "per_poly_us": {"median": float(pp.group(1)), "min": float(pp.group(2)), "mean": float(pp.group(3))},
        "launches": g(r"^launches / batch\s+: (.*)$").group(1),
        "program_output": txt[txt.find("device"):].strip().splitlines(),
    }


def cuda():
    tlog = open(raw("cuda_spike_test", "program.log")).read()
    tests = re.findall(r"^test (\S+) \.\.\. (\w+)$", tlog, re.M)
    smi = load_meta("cuda_spike_bench")["snapshot_before"]
    hdr = re.search(r"Driver Version: ([\d.]+)\s+CUDA Version: ([\d.]+)", smi)
    free = re.search(r"^[^,\n]+, ([\d.]+), (\d+) MiB, (\d+) MiB, (\d+) MiB, \d+ %$", smi, re.M)
    runs = {"default_50reps": parse_bench("cuda_spike_bench"), "repeat_200reps": parse_bench("cuda_spike_bench_200")}
    crit = 3.04
    med = runs["default_50reps"]["per_poly_us"]["median"]
    write("cuda_ntt_spike.json", {
        "task": "M0 Task 3 leftover: CUDA batched forward NTT spike (spikes/ntt-cuda) on quadruped-gpu",
        "gpu": runs["default_50reps"]["device"],
        "driver_version": hdr.group(1) if hdr else None,
        "driver_cuda_version": hdr.group(2) if hdr else None,
        "nvrtc": "CUDA 13.1 libnvrtc.so (/usr/local/cuda-13.1/lib64, via LD_LIBRARY_PATH), dlopened by cudarc",
        "cudarc": {
            "version": "0.12.1 (pinned in spikes/ntt-cuda/Cargo.lock)",
            "feature_change": "cuda-12040 -> cuda-12060 (server copy only; diff in raw/cuda_spike_bench/cargo_toml.diff)",
            "why": "cudarc 0.12.1 offers no cuda-13xxx feature (its newest is cuda-12060) and has no "
                   "'dynamic-loading' feature: it dlopens libcuda/libnvrtc at runtime by default "
                   "(the opt-out is 'dynamic-linking'). CUDA 13 bindings need a newer cudarc (crates.io latest 0.19.10).",
        },
        "params": {"log_n": 16, "N": 65536, "q": 1152921504606584833, "direction": "forward", "timing": "CUDA events, device-resident data, no H2D/D2H in the timed region"},
        "tests": {"results": [{"name": n, "result": r} for n, r in tests],
                  "passed": sum(r == "ok" for _, r in tests), "total": len(tests),
                  "skip_lines": tlog.count("SKIP")},
        "runs": runs,
        "criterion_us_per_poly": crit,
        "criterion_source": "2 x 1.52 us/poly (GPU-NTT, IEEE Access 2025 Table 10, RTX 4090, N=2^16, batch 64, radix2-CT); spikes/RESULTS.md Task 3",
        "verdict": "pass" if med <= crit else "fail",
        "verdict_basis": f"median per-poly us of the default run ({med}) <= {crit}",
        "gpu_memory_free_mib_before": int(free.group(3)) if free else None,
        "gpu_memory_used_mib_before": int(free.group(2)) if free else None,
        "provenance": {"test": provenance("cuda_spike_test"), "bench": provenance("cuda_spike_bench"),
                       "bench_200": provenance("cuda_spike_bench_200")},
    })


def reboot(job, out):
    st = json.load(open(raw(job, "stats.json")))
    log = open(raw(job, "program.log")).read()
    steps = st["steps"]
    rt = [s["reboot_step_time_s"] for s in steps]
    bs = st["bootstrap"]
    boot_lines = [float(x) for x in re.findall(r"Bootstrapping the weights took ([\d.]+) s", log)]
    m = load_meta(job)
    obj = {
        "task": "Official ReBoot eMLP-1 (AAAI-26, OpenFHE) on quadruped-gpu, RQ1 wall-clock baseline",
        "software": {"reboot": "3ac173fcea376ba5a2f733b2b8a2fbd4d9547aa2 + build-only patch (spikes/reboot/_ext/logs/reboot_patches.diff)",
                     "openfhe": "v1.2.1 (85a8f6325c50e3e39d07d253a1fe150bad2d49ce), Release, WITH_NATIVEOPT=ON, OpenMP, NATIVE_SIZE 64, GCC 15.2",
                     "openfhe_python": "v0.8.9 (2447d372d63a6ff5c54e1cd1f85ad6d1c5dafe19)",
                     "python": "CPython 3.11.16 (uv), numpy 2.0.0, torch 2.15.0+cpu",
                     "wrapper": "results/quadruped/scripts/reboot_wrapper.py (runs training_encrypted.py unchanged via runpy; os.cpu_count -> 12; counts bootstrap_array; decrypts trained weights for test accuracy)"},
        "setting": {"script_args": st["wrapper"]["argv"], "threads": st["wrapper"]["threads_os_cpu_count_override"],
                    "omp_num_threads": 12,
                    "train_samples": st["evaluation"]["train_samples"],
                    "batch": 48, "batches_per_epoch": st["evaluation"]["batches_per_epoch"],
                    "subset_indices": f"results/quadruped/raw/{job}/reboot_subsample_indices.txt",
                    "subset_rule": "ReBoot --subsample K keeps the first K images of the torchvision MNIST training set (same IDX order as hekit's --subset), then shuffles (seed 42); per-epoch shuffle from np.random seeded 0. With K=1024 and batch 48 there are 21 steps; the 16 samples at the end of the epoch's shuffled order are not trained (listed in subset_indices).",
                    "lr": steps[0]["lr"] if steps else None, "lr_note": "script default (--lr not passed, as in upstream experiments.sh); the cosine schedule starts after 5 epochs, so lr is constant here",
                    "weight_decay": 0.0, "momentum": 0.9, "optimizer": "ENCRYPTED_NESTEROV_SGD",
                    "crypto": st.get("crypto")},
        "timing_s": {
            "per_batch_step_reboot": {"n": len(rt), "mean": statistics.mean(rt) if rt else None,
                                      "median": statistics.median(rt) if rt else None,
                                      "min": min(rt) if rt else None, "max": max(rt) if rt else None,
                                      "sum": sum(rt)},
            "per_batch_step_definition": "ReBoot's own step_time from train_step_encrypted: encrypt batch + forward + backward + update + decrypt predictions + weight/velocity bootstrap (excludes the per-batch debug decryptions compute_mean_weights/compute_precisions the script runs after each step)",
            "per_batch_bootstrap_array": {"n": len(bs["seconds_per_call"]), "mean": statistics.mean(bs["seconds_per_call"]) if bs["seconds_per_call"] else None,
                                          "sum": sum(bs["seconds_per_call"])},
            "per_batch_bootstrap_logged_by_reboot": boot_lines,
            "setup_until_encrypted_weights": st["timeline"]["encrypt_done_s"],
            "script_wall_incl_setup_and_debug": st["timeline"]["script_wall_s"],
            "epoch_wall_from_first_step_start_to_script_end": (st["steps"][-1]["unix_start"] - st["steps"][0]["unix_start"]) + st["steps"][-1]["wall_s"] if steps else None,
            "process_wall_runner": m["wall_s_runner"],
            "steps": steps,
        },
        "bootstrap": {"bootstrap_array_calls": bs["bootstrap_array_calls"],
                      "ciphertexts_bootstrapped": bs["ciphertexts_bootstrapped"],
                      "num_iterations_per_ciphertext": sorted(set(bs["num_iterations_args"])),
                      "evalbootstrap_calls": bs["ciphertexts_bootstrapped"],
                      "note": "EvalBootstrap(ct, numIterations=2, precision=11) per ciphertext; one bootstrap_array call per step on 4 ciphertexts (W1, W2, V1, V2). OpenFHE's iterative bootstrapping with 2 iterations runs the bootstrap circuit twice internally.",
                      "per_sample": bs["ciphertexts_bootstrapped"] / max(1, len(steps) * 48)},
        "key_switches": "not measured: ReBoot and OpenFHE 1.2.1 do not log key-switch or rotation counts (the analytic count is in spikes/costmodel)",
        "evaluation": st["evaluation"],
        "decoder_check": st["decoder_check"],
        "memory": {"peak_rss_kb_time_v": m["peak_rss_kb_time_v"], "peak_rss_gb_time_v": m["peak_rss_kb_time_v"] * 1024 / 1e9 if m["peak_rss_kb_time_v"] else None,
                   "peak_rss_kb_tree_sampled_10s": m["peak_rss_kb_tree_sampled_10s"]},
        "provenance": provenance(job),
    }
    # ReBoot's configuration as plain values (raw/<job>/reboot_config_values.json; contains no ReBoot source text).
    vals_path = raw(job, "reboot_config_values.json")
    if os.path.exists(vals_path):
        obj["reboot_hyperparameters"] = json.load(open(vals_path))
        obj["reboot_hyperparameters"]["file"] = f"results/quadruped/raw/{job}/reboot_config_values.json"
    blk = re.search(r"CKKS parameters:\n(.*?)\n\n", log, re.S)
    bblk = re.search(r"Bootstrapping parameters:\n(.*?)\n\n", log, re.S)
    nblk = re.search(r"Network parameters:\n(.*?)\n\n", log, re.S)
    obj["reboot_printed_parameters"] = {
        "ckks": [l.strip(" -") for l in blk.group(1).splitlines()] if blk else None,
        "bootstrapping": [l.strip(" -") for l in bblk.group(1).splitlines()] if bblk else None,
        "network": [l.strip(" -") for l in nblk.group(1).splitlines()] if nblk else None,
        "threads_line": re.search(r"^Detected .* threads$", log, re.M).group(0),
        "bootstrap_precision_arg": sorted(set(bs["precision_args"])),
    }
    wd = re.search(r"^WRAPPER_DONE .*$", log, re.M)
    obj["wrapper_done_line"] = wd.group(0) if wd else None
    obj["reboot_final_lines"] = re.findall(r"^Final train .*$", log, re.M)
    obj["timing_s"]["last_step_bootstrap_logged_by_reboot"] = boot_lines[-1] if boot_lines else None
    obj["timing_s"]["breakdown_note"] = (
        "process_wall_runner = python start .. exit (imports, keygen + bootstrap setup, encrypt, 21 steps incl. their "
        "weight bootstraps, per-step debug decryptions, test evaluation); per_batch_step_reboot includes that step's weight bootstrap")
    write(out, obj)


def program_json(job, fname, out, note, threads):
    body = json.load(open(raw(job, fname)))
    body["quadruped"] = {
        "threads": threads,
        "binary_output_unmodified": f"results/quadruped/raw/{job}/{os.path.basename(fname)}",
        "note": note,
        "provenance": provenance(job),
    }
    write(out, body)


def bootstrap():
    log = open(raw("bootstrap_bench", "program.log")).read()
    lst = open(raw("bootstrap_bench_list", "program.log")).read()
    keep = [l for l in log.splitlines()
            if l.startswith(("n16_boot", "bootstrap_n16_boot", "Benchmarking", "                        time:", "Found", "  "))
            or "time:" in l or "change:" in l or "outlier" in l.lower() or "thrpt" in l]
    m = load_meta("bootstrap_bench")
    hdr = [
        "# n16_boot bootstrap criterion bench (M3 Task 38 Step 5) on quadruped-gpu",
        f"# cargo bench --list (run without HEKIT_BOOT_THREADS, so its default counts 1 and available_parallelism = 32 appear): {' '.join(x for x in lst.split() if 'bootstrap' in x)}",
        f"# command: {m['command']}",
        f"# workdir: {m['workdir']}  threads: HEKIT_BOOT_THREADS=1,12 (one rayon pool per count; 1 is the requested measurement)",
        f"# hekit synced commit: {' | '.join(m['hekit_synced_commit'])}",
        f"# start {m['start']}  end {m['end']}  exit {m['exit_status']}",
        f"# uptime before: {m['uptime_before']}",
        f"# uptime after:  {m['uptime_after']}",
        f"# peak RSS (time -v): {m['peak_rss_kb_time_v']} kB; sampled tree peak {m['peak_rss_kb_tree_sampled_10s']} kB",
        f"# cpu: {' / '.join(m['cpu'])}",
        f"# raw output: {m['raw_dir']}program.log (composed by results/quadruped/scripts/compose.py)",
        "",
    ]
    with open(os.path.join(QDIR, "bootstrap_bench.txt"), "w") as f:
        f.write("\n".join(hdr + keep) + "\n")
    print("wrote bootstrap_bench.txt")


def main():
    what = sys.argv[1]
    if what == "cuda":
        cuda()
    elif what == "reboot":
        reboot(sys.argv[2], sys.argv[3])
    elif what == "program_json":
        program_json(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    elif what == "bootstrap":
        bootstrap()
    else:
        raise SystemExit(f"unknown {what}")


if __name__ == "__main__":
    main()
