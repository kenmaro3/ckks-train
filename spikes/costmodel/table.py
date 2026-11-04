"""Print the RQ1 key-switch tables (Markdown) for a ReBoot model config.

Usage: python table.py [path/to/reboot_emlp1.json]
Config schema (Task 5): {"name","log_n","levels","scale_bits","layers":[{"in","out","activation"}],
"epochs","lr","momentum","batch","dataset"}.

Metric = key switches per sample per epoch (an epoch of Q samples is Q/B updates of B samples, so
per-sample-per-epoch = per-update / B). KS = rotations + relins; KS_spec = KS + bootstraps x BOOT_KS.
Also printed: diag_batched in interactive mode (M2 Task 30/32: client re-encryption, no bootstraps;
KS only) and, at N = 2^17, ReBoot as shipped vs the generous custom-sumRows ReBoot baseline.
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
    ref_slots = 1 << (int(cfg.get("log_n", 16)) - 1)

    def ps(layout, B, slots, **o):
        return network_cost(layout, layers, B, slots, momentum=momentum, **o)["per_sample"]

    # verdict at ReBoot's own (N, B)
    d, dn, rb = ps("diag_batched", b0, ref_slots), ps("diag_batched", b0, ref_slots, compact=False), ps(
        "reboot", b0, ref_slots)
    cts = network_cost("diag_batched", layers, b0, ref_slots)["ciphertexts"]
    print(f"### Verdict at ReBoot's own setting (N = 2^{cfg.get('log_n')}, B = {b0}; diag_batched needs "
          f"{cts} ciphertext(s) per batch)\n")
    print("| metric | diag_batched | reboot | ratio | ratio without compaction | <= 0.70? |")
    print("|---|---|---|---|---|---|")
    for key, label in (("ks", "KS = rot + relin (primary)"),
                       ("ks_spec", "KS_spec = KS + boots x BOOT_KS"),
                       ("ks_literal", "literal: rot + ct-ct mults"),
                       ("ks_literal_spec", "literal_spec: rot + ct-ct + boots x BOOT_KS")):
        ratio, ratio_n = d[key] / rb[key], dn[key] / rb[key]
        print(f"| {label} | {fmt(d[key])} | {fmt(rb[key])} | **{fmt(ratio, 3)}** | {fmt(ratio_n, 3)} | "
              f"{'yes' if ratio <= PASS else 'no'} |")
    di = ps("diag_batched", b0, ref_slots, interactive=True)
    dit = ps("diag_batched", b0, ref_slots, interactive=True, client_transpose=True)
    for dd, label in ((di, "KS, interactive (server transposes; M2 Task 32 mode, no bootstraps)"),
                      (dit, "KS, interactive (client supplies W^T)")):
        ratio = dd["ks"] / rb["ks"]
        print(f"| {label} | {fmt(dd['ks'])} | {fmt(rb['ks'])} | **{fmt(ratio, 3)}** | - | "
              f"{'yes' if ratio <= PASS else 'no'} |")
    print()
    print("Interactive rows: the client decrypts and re-encrypts W and V after every update (Task 30 "
          "ClientRefresher), so there is no bootstrap, no compaction and no unpack; ReBoot's KS has no "
          "bootstrap term either. They are the reference for M2 Task 32's 10% check, not the spec metric.\n")

    tables = (
        ("Primary metric: KS = rotations + relins", "ks", "ks_spec"),
        ("Literal spec metric: rotations + ct-ct mults", "ks_literal", "ks_literal_spec"),
    )
    passes = []
    for title, k1, k2 in tables:
        print(f"### {title}, per sample per epoch\n")
        inter = k1 == "ks"
        print(f"| N | B | cts | diag {k1} | reboot {k1} | **ratio** | diag {k2} | reboot {k2} | **ratio** | "
              f"ratio {k2} without compaction |" + (" diag ks interactive | ratio ks interactive |" if inter else ""))
        print("|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if inter else ""))
        for log_n in LOG_NS:
            slots = 1 << (log_n - 1)
            for B in batches:
                net = network_cost("diag_batched", layers, B, slots, momentum=momentum)
                d, rb = net["per_sample"], ps("reboot", B, slots)
                dn = ps("diag_batched", B, slots, compact=False)
                r1, r2, r2n = d[k1] / rb[k1], d[k2] / rb[k2], dn[k2] / rb[k2]
                flag = "*" if net["infeasible"] else ""
                own = " (b0)" if B == b0 else ""
                passes.append((k1, log_n, B, r1, flag))
                passes.append((k2, log_n, B, r2, flag))
                passes.append((k2 + " (no compaction)", log_n, B, r2n, flag))
                tail = ""
                if inter:
                    di = ps("diag_batched", B, slots, interactive=True)
                    passes.append(("ks interactive", log_n, B, di["ks"] / rb["ks"], flag))
                    tail = f" {fmt(di['ks'])} | {fmt(di['ks'] / rb['ks'], 2)} |"
                print(f"| 2^{log_n} | {B}{own} | {net['ciphertexts']}{flag} | {fmt(d[k1])} | {fmt(rb[k1])} | "
                      f"**{fmt(r1, 2)}** | {fmt(d[k2])} | {fmt(rb[k2])} | **{fmt(r2, 2)}** | {fmt(r2n, 2)} |" + tail)
        print()
    print("`ratio` = diag_batched / reboot at the same N and B. `cts` = diag_batched ciphertexts per mini-batch; "
          "`*` = d_pad*B > N/2 (more than one). BOOT_KS = key switches of one EvalBootstrap in ReBoot's "
          "configuration (2 iterations): " + ", ".join(
              f"{bootstrap_key_switches(1 << (n - 1))['total']} at N = 2^{n}" for n in LOG_NS) + ".\n")
    for metric in ("ks", "ks_spec", "ks_spec (no compaction)", "ks_literal", "ks_literal_spec",
                   "ks_literal_spec (no compaction)", "ks interactive"):
        rows = [p for p in passes if p[0] == metric]
        ok = [f"(B={p[2]}, N=2^{p[1]}){p[4]}" for p in rows if p[3] <= PASS]
        best = min(rows, key=lambda p: p[3])
        print(f"- ratio <= {PASS} under {metric}: " + (", ".join(ok) if ok else "none")
              + f". Best {fmt(best[3], 3)} at (B={best[2]}, N=2^{best[1]}).")
    print()

    print("### N = 2^17: ReBoot as shipped (slot-filling r = 2048) vs generous ReBoot (custom sumRows, r = 1024)\n")
    slots17 = 1 << 16
    print("| B | cts | reboot ks shipped / generous | ratio ks | ratio ks_spec | ratio ks_literal | "
          "ratio ks_literal_spec | ratio ks interactive |")
    print("|---|---|---|---|---|---|---|---|")
    gen_passes = []
    for B in batches:
        net = network_cost("diag_batched", layers, B, slots17, momentum=momentum)
        d, di = net["per_sample"], ps("diag_batched", B, slots17, interactive=True)
        sh, gn = ps("reboot", B, slots17), ps("reboot", B, slots17, custom_sumrows=True)
        flag = "*" if net["infeasible"] else ""
        cells = []
        for key, dd in (("ks", d), ("ks_spec", d), ("ks_literal", d), ("ks_literal_spec", d), ("ks", di)):
            a, b = dd[key] / sh[key], dd[key] / gn[key]
            cells.append(f"{fmt(a, 3)} / **{fmt(b, 3)}**")
            if dd is d:
                gen_passes.append((key, B, b, flag))
        print(f"| {B}{' (b0)' if B == b0 else ''} | {net['ciphertexts']}{flag} | {fmt(sh['ks'])} / {fmt(gn['ks'])} | "
              + " | ".join(cells) + " |")
    print()
    print("Cells: ratio vs shipped / **ratio vs generous**. The generous baseline replicates ReBoot's r x c "
          "matrix twice so that sumRows needs only log2(1024) = 10 rotations; it is not what ReBoot ships.\n")
    for key in ("ks_spec", "ks_literal_spec"):
        ok = [f"(B={B}){f}" for k, B, v, f in gen_passes if k == key and v <= PASS]
        print(f"- ratio <= {PASS} under {key} vs generous ReBoot at N = 2^17: " + (", ".join(ok) if ok else "none") + ".")
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
    for label, B, o in (("reboot", b0, {}), ("diag_batched", 32, {}), ("diag_batched", 64, {}),
                        ("diag_batched", b0, {}), ("diag_batched interactive", b0, {"interactive": True})):
        r = network_cost(label.split(" ")[0], layers, B, slots, momentum=momentum, **o)
        cells = [fmt((r["phases"][p]["rotations"] + r["phases"][p]["relins"]) / B, 2)
                 for p in ("forward", "transpose", "backward", "gradient", "update")]
        print(f"| {label} | {B} | " + " | ".join(cells) + f" | {fmt(r['per_sample']['bootstraps'], 3)} |")
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
