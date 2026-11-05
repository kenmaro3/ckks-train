#!/usr/bin/env python3
"""Assembles results/mnist_reboot_layout.json (Task 33 Step 5) from the raw outputs of the
`mnist_reboot_layout` binary and the run logs (results/logs/, gitignored).

usage: assemble.py LAZY_JSON LAZY_LOG LAZY_ONE_JSON LAZY_ONE_LOG PS_JSON PS_LOG PS_ONE_JSON PS_ONE_LOG OUT

LAZY_JSON/PS_JSON: bin `--out` files of the full Lazy (1024 samples) and PerSample (256
samples) runs; *_LOG: their stderr + `/usr/bin/time -l` logs; *_ONE_*: the 1-batch probe runs.
"""
import json
import re
import statistics as st
import sys

lazy, lazy_log, lazy_one, lazy_one_log, ps, ps_log, ps_one, ps_one_log, out_path = sys.argv[1:10]
BATCHES_FULL = 1024 // 32

PROVENANCE = {
    "note": (
        "Not the raw output of the binary. The two modes' raw outputs (bin --out files, "
        "one per mode) were produced by target/release/mnist_reboot_layout built (cargo build "
        "--release -p hekit-bench --bin mnist_reboot_layout) in a clean git worktree of main at "
        "f8ee9e8 (2026-11-05 06:21 JST) plus the then-uncommitted crates/hekit-bench/src/bin/"
        "mnist_reboot_layout.rs, crates/hekit-nn/src/layout_reboot.rs and "
        "crates/hekit-nn/tests/layout_reboot.rs (the code of commit 10fa4e2; the later fix-round "
        "commit only adds a closed-form assertion to the bin, a smoke test and doc text). The "
        "binary was built at 06:25 JST; runs: Lazy 1024 samples (06:36 JST start), then PerSample "
        "256 samples, both under /usr/bin/time -l, run one after the other. This file was "
        "assembled from those two raw outputs plus the logs by tools/reboot_layout/assemble.py "
        "(the first version used an uncommitted throwaway script with the same logic) and "
        "re-serialized with Python json (indent 1), so number formatting differs from the "
        "binary's. Every value under modes.*.run is the binary output unchanged. "
        "The raw outputs of the two full bin runs (Lazy 1024, PerSample 256) were written to the "
        "agent's session scratch directory, not to results/logs/, and were not kept with the "
        "results at first; until the M2 fix wave the assembly could only be checked against the "
        "committed JSON (M2 final review m6). In the M2 fix wave (Task 34) both files were "
        "recovered from that scratch directory (full_lazy.json 09:27 JST, sub_persample.json "
        "10:09 JST, the paths printed in the run logs), compared equal to modes.*.run, and copied "
        "to results/logs/reboot_lazy_1024.json and results/logs/reboot_persample_256.json "
        "(gitignored); re-running this script on them plus the logs and probes reproduces this "
        "file byte for byte except the final newline."
    ),
    "hand_added_fields": {
        "modes.*.run": "the binary's raw JSON, unchanged",
        "modes.*.per_batch": "parsed by regex from the 'batch k (n samples): server S s, client C s ... load { L1' lines of the run log (server_s, client_s, load1 = 1-minute load average printed at the END of the batch, so it lags the batch); load_elevated = load1 > baseline_load1_median_first6 + 1.5",
        "modes.*.baseline_load1_median_first6": "median of load1 over the first 6 batches of the same run (itself loaded: the machine was shared)",
        "modes.*.rss_bytes_time_l": "'maximum resident set size' line of /usr/bin/time -l in the run log (the log's peak memory footprint is not copied)",
        "modes.*.one_batch_probe": "server_total from the 1-batch probe run's raw JSON (--subset 32) and its /usr/bin/time -l max RSS",
        "modes.*.samples_run, modes.*.full_epoch_run": "samples_run = training.samples of the run; full_epoch_run = (samples_run == 1024)",
        "modes.*.epoch_time_s": "measured_total_incl_client / measured_server_only copy time_s.epoch_total_incl_client / time_s.server_total of the raw output; median_server_batch_s = median of per_batch server_s; median_batch_x32_server = that x 32; median_batch_incl_client_x32 = (median server + median client) x 32; elevated_load_batches = the flagged batch indices; for PerSample (256 samples) the full-epoch figure is an EXTRAPOLATION, not measured",
    },
    "caveats": [
        "Load was shared with other jobs during both runs; the medians are not 'clean' times. Lazy batches at the start ran at about 266-268 s (about 8540 s per 32 batches). The PerSample extrapolation (median x 32) is probably inflated by load: its batch 0 (287 s at load 4.2) is already above Lazy's clean level despite only 62 more relins.",
        "r*c = slots is enforced by RebootLayout::new for any slot count (a test now covers N = 2^17).",
    ],
}


def parse(log):
    rows = []
    for line in open(log):
        m = re.search(r"batch (\d+) \((\d+) samples\): server ([\d.]+) s, client ([\d.]+) s.*load \{ ([\d.]+)", line)
        if m:
            rows.append(dict(batch=int(m[1]), server_s=float(m[3]), client_s=float(m[4]), load1=float(m[5])))
    return rows


def rss(log):
    for line in open(log):
        if "maximum resident" in line:
            return int(line.split()[0])


out = {
    "provenance": PROVENANCE,
    "task": "Task 33 Step 5: ReBoot-style per-sample layout on hekit, MNIST eMLP-1 (RQ1 baseline c)",
    "notes": [
        "Same library, n16_train, data, batch 32, lr 0.001, momentum 0.9, seed 32 and initial weights as results/mnist_emlp1.json; (r,c)=(1024,32); interactive update (client decrypts dW, NAG in plaintext, re-encrypts W); no biases (no key-switch impact); twin is bias-free.",
        "Lazy = 1 relin per batch gradient sum (generous RQ1 headline); persample = 1 relin per product (cost model default, as specified). Closed forms per batch of B samples: persample B(2 log2 r + 2 log2 c + 7) key switches; lazy B(2 log2 r + 2 log2 c + 5) + 2.",
        "Per-batch times are single-thread server time; other jobs shared the machine (flagged via load1 > baseline+1.5).",
    ],
    "modes": {},
}
for mode, raw, log, one, one_log in (
    ("lazy", lazy, lazy_log, lazy_one, lazy_one_log),
    ("persample", ps, ps_log, ps_one, ps_one_log),
):
    d = json.load(open(raw))
    n = d["training"]["samples"]
    rows = parse(log)
    base = st.median([r["load1"] for r in rows[:6]])
    for r in rows:
        r["load_elevated"] = r["load1"] > base + 1.5
    med = st.median(r["server_s"] for r in rows)
    o = json.load(open(one))
    out["modes"][mode] = {
        "run": d,
        "samples_run": n,
        "full_epoch_run": n == 1024,
        "per_batch": rows,
        "baseline_load1_median_first6": base,
        "rss_bytes_time_l": rss(log),
        "one_batch_probe": {"server_s": o["time_s"]["server_total"], "rss_bytes": rss(one_log)},
        "epoch_time_s": {
            "measured_total_incl_client": d["time_s"]["epoch_total_incl_client"],
            "measured_server_only": d["time_s"]["server_total"],
            "median_server_batch_s": med,
            "median_batch_x32_server": med * BATCHES_FULL,
            "median_batch_incl_client_x32": (med + st.median(r["client_s"] for r in rows)) * BATCHES_FULL,
            "full_1024_epoch": "measured" if n == 1024 else "EXTRAPOLATION: median server batch x 32 batches (+ median client)",
            "elevated_load_batches": [r["batch"] for r in rows if r["load_elevated"]],
        },
    }
json.dump(out, open(out_path, "w"), indent=1)
