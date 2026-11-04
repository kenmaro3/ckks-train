"""Print the RQ1 key-switch tables (Markdown) for a ReBoot model config.

Usage: python table.py [path/to/reboot_emlp1.json]
Config schema (Task 5): {"name","log_n","levels","scale_bits","layers":[{"in","out","activation"}],
"epochs","lr","momentum","batch","dataset"}.

Metric = key switches per sample per epoch (an epoch of Q samples is Q/B updates of B samples, so
per-sample-per-epoch = per-update / B). KS = rotations + relins; KS_spec = KS + bootstraps x BOOT_KS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from costmodel import bootstrap_key_switches, network_cost

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "crates/hekit-bench/configs/reboot_emlp1.json"
BATCHES = (16, 32, 64, 128)
LOG_NS = (16, 17)
PASS = 0.70


def load(path):
    cfg = json.loads(Path(path).read_text())
    layers = [(l["in"], l["out"], l.get("activation")) for l in cfg["layers"]]
    return cfg, layers


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def main(path=DEFAULT_CONFIG):
    cfg, layers = load(path)
    momentum = bool(cfg.get("momentum", 0))
    b0 = int(cfg["batch"])
    print(f"Config `{Path(path).name}`: {cfg.get('name')} on {cfg.get('dataset')}, layers "
          + " -> ".join([str(layers[0][0])] + [f"{o} ({a or 'none'})" for _, o, a in layers])
          + f", ReBoot batch {b0}, ReBoot log_n {cfg.get('log_n')}, momentum {cfg.get('momentum')}.\n")

    batches = sorted(set(BATCHES) | {b0})
    print("### Key switches per sample per epoch\n")
    print("`ratio` = diag_batched / reboot at the same N and the same batch B; `ratio@b0` compares with "
          f"ReBoot at its own batch b0 = {b0} (bootstraps amortised over b0). `cts` = ciphertexts per "
          "mini-batch for diag_batched (> 1 = d_pad*B > N/2, flagged `*`). BOOT_KS = key switches of "
          "one EvalBootstrap (2 iterations, ReBoot config).\n")
    print("| N | B | cts | diag KS | reboot KS | **ratio KS** | diag KS_spec | reboot KS_spec (B) | "
          "**ratio KS_spec** | ratio KS_spec@b0 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for log_n in LOG_NS:
        slots = 1 << (log_n - 1)
        rb0 = network_cost("reboot", layers, b0, slots, momentum=momentum)
        for B in batches:
            d = network_cost("diag_batched", layers, B, slots, momentum=momentum)
            rb = network_cost("reboot", layers, B, slots, momentum=momentum)
            rk = d["per_sample"]["ks"] / rb["per_sample"]["ks"]
            rs = d["per_sample"]["ks_spec"] / rb["per_sample"]["ks_spec"]
            rs0 = d["per_sample"]["ks_spec"] / rb0["per_sample"]["ks_spec"]
            flag = "*" if d["infeasible"] else ""
            own = " (b0)" if B == b0 else ""
            rows.append((log_n, B, rk, rs, rs0, d["infeasible"]))
            print(f"| 2^{log_n} | {B}{own} | {d['ciphertexts']}{flag} | {fmt(d['per_sample']['ks'])} | "
                  f"{fmt(rb['per_sample']['ks'])} | **{fmt(rk, 2)}** | {fmt(d['per_sample']['ks_spec'])} | "
                  f"{fmt(rb['per_sample']['ks_spec'])} | **{fmt(rs, 2)}** | {fmt(rs0, 2)} |")
    print()
    for idx, name in ((2, "KS"), (3, "KS_spec (same B)"), (4, f"KS_spec vs ReBoot@b0={b0}")):
        ok = [f"(B={r[1]}, N=2^{r[0]}){'*' if r[5] else ''}" for r in rows if r[idx] <= PASS]
        best = min(rows, key=lambda r: r[idx])
        print(f"- ratio <= {PASS} under {name}: " + (", ".join(ok) if ok else "none")
              + f". Best {fmt(best[idx], 3)} at (B={best[1]}, N=2^{best[0]}).")
    print()

    print("### Sensitivity of KS_spec ratio to BOOT_KS (same B)\n")
    print("| N | BOOT_KS | " + " | ".join(f"B={B}" for B in batches) + " |")
    print("|---|---|" + "---|" * len(batches))
    for log_n in LOG_NS:
        slots = 1 << (log_n - 1)
        base = bootstrap_key_switches(slots)
        single = bootstrap_key_switches(slots, iterations=1)["total"]
        for bk, label in ((0, "0 (= KS)"), (single, f"{single} (1 iteration)"),
                          (base["total"], f"{base['total']} (ReBoot: 2 iterations)"),
                          (2 * base["total"], f"{2 * base['total']} (2x)")):
            cells = []
            for B in batches:
                d = network_cost("diag_batched", layers, B, slots, momentum=momentum, boot_ks=bk)
                rb = network_cost("reboot", layers, B, slots, momentum=momentum, boot_ks=bk)
                cells.append(fmt(d["per_sample"]["ks_spec"] / rb["per_sample"]["ks_spec"], 2))
            print(f"| 2^{log_n} | {label} | " + " | ".join(cells) + " |")
    print()

    print(f"### Per-phase key switches per sample (N = 2^{cfg.get('log_n', 16)})\n")
    slots = 1 << (int(cfg.get("log_n", 16)) - 1)
    print("| layout | B | forward | transpose | backward | gradient | update | bootstraps/sample |")
    print("|---|---|---|---|---|---|---|---|")
    for layout, B in (("reboot", b0), ("diag_batched", 32), ("diag_batched", 64), ("diag_batched", b0)):
        r = network_cost(layout, layers, B, slots, momentum=momentum)
        cells = [fmt((r["phases"][p]["rotations"] + r["phases"][p]["relins"]) / B, 2)
                 for p in ("forward", "transpose", "backward", "gradient", "update")]
        print(f"| {layout} | {B} | " + " | ".join(cells) + f" | {fmt(r['per_sample']['bootstraps'], 3)} |")
    print()

    print(f"### Reference rows (N = 2^{cfg.get('log_n', 16)}, KS per sample)\n")
    print("| layout | B | rotations | relins | KS | bootstraps/sample |")
    print("|---|---|---|---|---|---|")
    refs = [
        ("reboot", b0, {}),
        ("reboot (lazy relin, hypothetical)", b0, {"lazy_relin": True}),
        ("diag_2020 (seal convention)", 1, {"convention": "seal"}),
        ("row_2020", 1, {}),
        ("diag_batched", 1, {}),
    ]
    for label, B, o in refs:
        layout = label.split(" ")[0]
        r = network_cost(layout, layers, B, slots, momentum=momentum, **o)
        ps = r["per_sample"]
        print(f"| {label} | {B} | {fmt(ps['rotations'])} | {fmt(ps['relins'])} | {fmt(ps['ks'])} | "
              f"{fmt(ps['bootstraps'], 3)} |")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
