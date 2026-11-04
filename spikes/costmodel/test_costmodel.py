"""Tests for the key-switch cost model (Task 6, RQ1).

Three kinds of checks:
  1. Reproduction of published counts: Mihara et al. 2020 (arXiv:2012.13552) Table 2, and the op
     counts Task 4 measured on SEAL for the same algorithm (spikes/RESULTS.md, "## Task 4").
  2. ReBoot (arXiv:2506.19693) per-sample counts derived from its paper (Sec. 3 eq. 4-5,
     Sec. 4.2.1) and its code (OpenFHE EvalSumRows / EvalSumCols).
  3. Formula == simulator: slotsim.py executes the layouts slot by slot on plaintext vectors with
     an op counter that refuses illegal ops (rotating / multiplying a degree-2 ciphertext). The
     result must equal dense numpy math, and the counted ops must equal the closed-form formulas.
"""

import math

import numpy as np
import pytest

import slotsim
from costmodel import (
    LAYOUTS,
    PHASES,
    bootstrap_key_switches,
    bsgs_split,
    cost,
    network_cost,
)

KEYS = {"rotations", "ct_ct_mults", "relins", "ct_pt_mults", "bootstraps_per_update"}
OPS = ("rotations", "ct_ct_mults", "relins", "ct_pt_mults")

# Paper Table 2 network: input 6, hidden 3, output 1, square activation after each dense layer.
TABLE2 = [(6, 3, "square"), (3, 1, "square")]
# Table 2 covers FF, "Transition" and BP (incl. the input-layer W1^T product); no weight gradient,
# no update.
TABLE2_SCOPE = ("forward", "transpose", "backward")


def phase(r, name, key):
    return r["phases"][name][key]


# ---------------------------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_cost_returns_required_keys(layout):
    r = cost(layout, 8, 4, 4, 1024)
    assert KEYS <= set(r)
    assert set(r["phases"]) == set(PHASES)
    assert r["key_switches"] == r["rotations"] + r["relins"]


def test_unknown_layout_raises():
    with pytest.raises(ValueError):
        cost("nope", 8, 4, 4, 1024)


# ---------------------------------------------------------------------------------------------
# 2020 paper, Table 2
# ---------------------------------------------------------------------------------------------


def test_table2_row_2020_totals():
    r = network_cost("row_2020", TABLE2, 1, 2**14, scope=TABLE2_SCOPE, first_backward=True)
    assert r["paper_mults"] == 38
    assert r["rotations"] == 63


def test_table2_row_2020_per_phase():
    r = network_cost("row_2020", TABLE2, 1, 2**14, scope=TABLE2_SCOPE, first_backward=True)
    # Table 2 rows: FF 3+1+1+1 mult / 18+3 rot; Transition 18+3 / 18+3; BP 3+1+6+1 / 3+18.
    assert (phase(r, "forward", "paper_mults"), phase(r, "forward", "rotations")) == (6, 21)
    assert (phase(r, "transpose", "paper_mults"), phase(r, "transpose", "rotations")) == (21, 21)
    assert (phase(r, "backward", "paper_mults"), phase(r, "backward", "rotations")) == (11, 21)


def test_table2_diag_2020_totals():
    r = network_cost("diag_2020", TABLE2, 1, 2**14, scope=TABLE2_SCOPE, first_backward=True)
    assert r["paper_mults"] == 17
    assert r["ct_ct_mults"] == 17
    assert r["rotations"] == 17


def test_table2_diag_2020_mults_per_phase():
    r = network_cost("diag_2020", TABLE2, 1, 2**14, scope=TABLE2_SCOPE, first_backward=True)
    assert phase(r, "forward", "ct_ct_mults") == 6 + 1 + 3 + 1
    assert phase(r, "transpose", "ct_ct_mults") == 0
    assert phase(r, "backward", "ct_ct_mults") == 1 + 1 + 3 + 1
    assert phase(r, "transpose", "rotations") == 3 + 1  # Table 1 "diagonal packing with step": M


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Table 2 lists Dense(a->b) with b rotations (FF 3+1, BP 3+6); Alg. 2 and the Sec. 2 text "
        "('N multiplications and N rotations', N = input dim) give a rotations (FF 6+3, BP 1+3). "
        "The model follows Alg. 2; totals agree (17) because FF and BP swap in/out dims."
    ),
)
def test_table2_diag_2020_rotations_per_phase():
    r = network_cost("diag_2020", TABLE2, 1, 2**14, scope=TABLE2_SCOPE, first_backward=True)
    assert phase(r, "forward", "rotations") == 3 + 1
    assert phase(r, "backward", "rotations") == 3 + 6


def test_diag_2020_seal_convention_matches_task4_iris():
    # Task 4 measured one SEAL iteration of the 2020 algorithm (alg5, stepped diagonals) on
    # Iris 4-10-3: 35 ct-ct mults, 35 relins, 31 rotations (FF 13 / TR 2 / BP 16).
    layers = [(4, 10, "square"), (10, 3, "square")]
    r = network_cost("diag_2020", layers, 1, 2**14, convention="seal", momentum=False)
    assert r["ct_ct_mults"] == 35
    assert r["relins"] == 35
    assert r["rotations"] == 31
    assert phase(r, "forward", "rotations") == 13
    assert phase(r, "transpose", "rotations") == 2
    bp = phase(r, "backward", "rotations") + phase(r, "gradient", "rotations")
    assert bp == 16
    assert phase(r, "forward", "ct_ct_mults") == 16
    assert phase(r, "backward", "ct_ct_mults") + phase(r, "gradient", "ct_ct_mults") == 19


# ---------------------------------------------------------------------------------------------
# ReBoot
# ---------------------------------------------------------------------------------------------


def test_reboot_matmul_rotations():
    S = 2**15
    # RE-Matmul = EvalSumRows over the r = S/c rows: log2(r) rotations (paper eq. 4, Sec. 4.2.1).
    r = cost("reboot", 256, 32, 1, S, packing="row", c=32, backward=False, activation=None)
    assert phase(r, "forward", "rotations") == int(math.log2(S // 32))
    assert phase(r, "forward", "ct_ct_mults") == 1
    # CE-Matmul = EvalSumCols: 2 log2(c) rotations + 1 mask (paper eq. 5, Sec. 4.2.1).
    r = cost("reboot", 32, 16, 1, S, packing="col", c=32, backward=False, activation=None)
    assert phase(r, "forward", "rotations") == 2 * 5
    assert phase(r, "forward", "ct_pt_mults") == 1


def test_reboot_emlp1_counts():
    # ReBoot eMLP-1 as Task 5 runs it: 784 -> 32 (PolyReLU) -> 10, N = 2^16, batch 48.
    layers = [(784, 32, "poly_relu"), (32, 10, None)]
    r = network_cost("reboot", layers, 48, 2**15)
    assert r["reboot_rc"] == (1024, 32)  # matches ReBoot's printed "Row size 1024, Col size 32"
    assert r["rotations"] == 30 * 48
    assert r["relins"] == 7 * 48
    assert r["ct_ct_mults"] == 7 * 48
    assert r["bootstraps_per_update"] == 4  # (W, V) x 2 layers, once per mini-batch
    assert r["per_sample"]["ks"] == pytest.approx(37.0)


def test_reboot_lazy_relin_variant_is_cheaper():
    layers = [(784, 32, "poly_relu"), (32, 10, None)]
    a = network_cost("reboot", layers, 48, 2**15)
    b = network_cost("reboot", layers, 48, 2**15, lazy_relin=True)
    assert a["relins"] - b["relins"] == 2 * (48 - 1)  # one relin per layer per batch for dW


@pytest.mark.parametrize(
    "layers,batch,slots",
    [
        ([(16, 8, "poly_relu"), (8, 4, None)], 3, 128),
        ([(6, 16, "poly_relu"), (16, 4, "square"), (4, 3, None)], 2, 256),
    ],
)
def test_reboot_sim_matches_formula_and_numpy(layers, batch, slots):
    rng = np.random.default_rng(1)
    run = slotsim.reboot_step(layers, batch, slots, rng)
    for name, ref in run["reference"].items():
        np.testing.assert_allclose(run["decoded"][name], ref, atol=1e-9, err_msg=name)
    r = network_cost("reboot", layers, batch, slots)
    for ph in PHASES:
        for op in OPS:
            assert run["counts"][ph][op] == r["phases"][ph][op], (ph, op)
    assert run["bootstraps"] == r["bootstraps_per_update"]


# ---------------------------------------------------------------------------------------------
# Bootstrapping key switches (OpenFHE 1.2.1, ReBoot's configuration)
# ---------------------------------------------------------------------------------------------


def test_bootstrap_linear_transform_rotations():
    # GetCollapsedFFTParams(2^15, budget 2): (g, b) = (64, 8) + remainder (32, 8) -> 70 + 38.
    assert bootstrap_key_switches(2**15, iterations=1)["linear_transform_rotations"] == 2 * (70 + 38)
    # 2^16 slots: two full levels (64, 8) -> 2 x 70 per transform.
    assert bootstrap_key_switches(2**16, iterations=1)["linear_transform_rotations"] == 2 * 140


def test_bootstrap_total_is_iterations_times_single():
    one = bootstrap_key_switches(2**15, iterations=1)
    two = bootstrap_key_switches(2**15, iterations=2)
    assert two["total"] == 2 * one["total"]
    assert one["total"] == (
        one["linear_transform_rotations"] + one["conjugations"] + one["evalmod_relins"]
    )


# ---------------------------------------------------------------------------------------------
# diag_batched (this work)
# ---------------------------------------------------------------------------------------------


def test_bsgs_split():
    assert bsgs_split(32) == (8, 4)  # (8-1) + (4-1) rotations + 4 relins = 14 key switches
    assert bsgs_split(1) == (1, 1)
    n1, n2 = bsgs_split(1024)
    assert n1 * n2 == 1024


def test_diag_batched_forward_terms_of_brief():
    # 1024 -> 32 at B = 32, N = 2^16: d_out tensor products, BSGS over d_out, log2(d_in/d_out) fold.
    r = cost("diag_batched", 1024, 32, 32, 2**15, backward=False, activation=None)
    n1, n2 = bsgs_split(32)
    assert phase(r, "forward", "ct_ct_mults") == 32
    assert phase(r, "forward", "relins") == n2
    assert phase(r, "forward", "rotations") == (n1 - 1) + (n2 - 1) + 5
    assert r["ciphertexts"] == 1 and not r["infeasible"]


def test_diag_batched_gradient_has_2log2B_per_diagonal_when_full():
    # When L * B == slots the batch reduction (log2 B) plus the re-broadcast (log2(S/L) = log2 B)
    # give the brief's 2 log2(B) rotations per diagonal; the rest is BSGS / lane-shift overhead.
    S, B = 2**15, 32
    r = cost("diag_batched", 1024, 32, B, S, backward=False, activation=None)
    m, (n1, n2) = 32, bsgs_split(32)
    reduce_and_broadcast = 2 * m * int(math.log2(B))
    shifts = 2 * (m - 1)  # pack into the compact ciphertext and back
    total = phase(r, "gradient", "rotations") + phase(r, "update", "rotations")
    assert total == reduce_and_broadcast + shifts + (n2 - 1)
    assert phase(r, "gradient", "relins") == m


def test_diag_batched_pads_to_powers_of_two():
    a = cost("diag_batched", 784, 30, 16, 2**15)
    b = cost("diag_batched", 1024, 32, 16, 2**15)
    for k in KEYS:
        assert a[k] == b[k]


def test_diag_batched_infeasible_needs_more_ciphertexts():
    one = cost("diag_batched", 1024, 32, 32, 2**15, backward=False)
    two = cost("diag_batched", 1024, 32, 64, 2**15, backward=False)
    assert two["ciphertexts"] == 2 and two["infeasible"]
    assert phase(two, "forward", "rotations") == 2 * phase(one, "forward", "rotations")
    # per-update work (batch reduction, pack/unpack, bootstraps) does not scale with ciphertexts
    assert two["bootstraps_per_update"] == one["bootstraps_per_update"]


def test_diag_batched_compact_weights_bootstrap_like_reboot():
    # 1024 x 32 weights fill exactly one 2^15-slot ciphertext once compacted: W + V = 2 bootstraps.
    r = cost("diag_batched", 1024, 32, 32, 2**15, backward=False)
    assert r["bootstraps_per_update"] == 2


def test_diag_batched_per_sample_ks_decreases_with_batch():
    layers = [(784, 32, "poly_relu"), (32, 10, None)]
    ks = [network_cost("diag_batched", layers, B, 2**16)["per_sample"]["ks"] for B in (16, 32, 64)]
    assert ks[0] > ks[1] > ks[2]


@pytest.mark.parametrize(
    "layers,batch,slots",
    [
        ([(16, 8, "poly_relu"), (8, 4, None)], 4, 128),  # wide, 1 ciphertext
        ([(4, 16, "poly_relu"), (16, 8, None)], 4, 128),  # tall first layer
        ([(16, 8, "poly_relu"), (8, 4, None)], 4, 32),  # 2 ciphertexts per batch (infeasible)
        ([(8, 8, "poly_relu"), (8, 16, "square"), (16, 2, None)], 2, 64),  # square / tall / wide
        ([(6, 3, "square"), (3, 1, "square")], 1, 64),  # paper Table 2 net, padded, B = 1
        ([(32, 32, "poly_relu"), (32, 4, None)], 8, 1024),  # BSGS with n2 > 1, spare periods
    ],
)
def test_diag_batched_sim_matches_formula_and_numpy(layers, batch, slots):
    rng = np.random.default_rng(7)
    run = slotsim.diag_batched_step(layers, batch, slots, rng)
    for name, ref in run["reference"].items():
        np.testing.assert_allclose(run["decoded"][name], ref, atol=1e-9, err_msg=name)
    r = network_cost("diag_batched", layers, batch, slots)
    for ph in PHASES:
        for op in OPS:
            assert run["counts"][ph][op] == r["phases"][ph][op], (ph, op)
    assert run["bootstraps"] == r["bootstraps_per_update"]
