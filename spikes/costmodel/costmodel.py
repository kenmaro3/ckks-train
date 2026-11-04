"""Key-switch cost model for CKKS MLP-training packing layouts (Task 6 spike, RQ1).

Counts homomorphic operations for ONE weight update (one mini-batch step of `batch` samples) and
derives per-sample key switches. Plaintext only: nothing here is encrypted.

Layouts (`layout` argument):

* ``"diag_batched"``  this work: batched diagonal packing, feature-major slots ``slot = j*B' + b``,
  BSGS mat-vec with lazy relinearisation, fold for wide layers, rotation-only transpose of the
  encrypted weight diagonals, compact weight/velocity ciphertexts for update + bootstrapping.
* ``"reboot"``        ReBoot (Pirillo & Colombo, arXiv:2506.19693; code
  github.com/AI-Tech-Research-Lab/ReBoot @ spikes/reboot/_ext/ReBoot, OpenFHE 1.2.1): one sample
  per ciphertext, r x c matrix packing with r*c = N/2, row/column encoding alternating per layer.
* ``"row_2020"``      Mihara et al. 2020 (arXiv:2012.13552) row packing (Alg. 1 / Alg. 3), B = 1.
* ``"diag_2020"``     Mihara et al. 2020 diagonal packing (Alg. 2 / Alg. 5), B = 1, naive per-diagonal
  rotations, one relinearisation per ct-ct product (no BSGS, no hoisting, no lazy relin).

Counted ops (per update, all samples of the mini-batch):
  rotations     Galois automorphisms = key switches (hoisted or not: a hoisted rotation still needs
                its own key-switch inner product; hoisting only shares the decomposition)
  ct_ct_mults   ciphertext x ciphertext tensor products
  relins        relinearisations (= key switches)
  ct_pt_mults   plaintext / scalar multiplications (masks, learning-rate scalars) - no key switch
  bootstraps_per_update  ciphertexts refreshed by bootstrapping per update

Key-switch metrics: ``key_switches = rotations + relins`` (brief, Step 5), and the spec metric
``ks_spec = rotations + relins + bootstraps * BOOT_KS`` where BOOT_KS is the key-switch count of one
EvalBootstrap call (see `bootstrap_key_switches`).

All dimensions are padded to powers of two for ``diag_batched`` (stated in every result as
``d_in_pad``/``d_out_pad``); the 2020 layouts use the raw dimensions as the paper does; ReBoot pads
to its own r x c rule.
"""

from __future__ import annotations

import math

LAYOUTS = ("diag_batched", "reboot", "row_2020", "diag_2020")
PHASES = ("forward", "transpose", "backward", "gradient", "update")
OPS = ("rotations", "ct_ct_mults", "relins", "ct_pt_mults")
KEYS = OPS + ("bootstraps_per_update",)

NO_ACTIVATION = ("none", "identity", "linear", "", "false", "null")


def next_pow2(x: int) -> int:
    if x < 1:
        raise ValueError(f"dimension must be >= 1, got {x}")
    return 1 << (x - 1).bit_length()


def log2i(x: int) -> int:
    """Exact log2 of a power of two."""
    if x < 1 or x & (x - 1):
        raise ValueError(f"{x} is not a power of two")
    return x.bit_length() - 1


def doublings(period: int, need: int) -> int:
    """Rotations of log-doubling replication from `period` to cover `need` slots."""
    n, cov = 0, period
    while cov < need:
        cov *= 2
        n += 1
    return n


def has_activation(act) -> bool:
    """Any named activation (square, poly_relu / PolyReLU, ...) costs 1 ct-ct mult forward and 1
    in its derivative; None / "none" / "identity" / False means no activation."""
    return not (act is None or act is False or str(act).strip().lower() in NO_ACTIVATION)


def nag_pt_mults(momentum: bool, decay: bool) -> int:
    """Scalar mults of one weight-ciphertext update (ReBoot cpp/lib/nn.cpp compute_nesterov:
    decay*W, lr*g, mom*lr*g, mom^2*lr*v, mom*v; plain SGD: decay*W, lr*g). No key switch."""
    return (4 if momentum else 1) + (1 if decay else 0)


def _zero() -> dict:
    return {op: 0 for op in OPS} | {"paper_mults": 0}


def _add(ph: dict, times: int = 1, **ops) -> None:
    for k, v in ops.items():
        ph[k] += times * v


def bsgs_split(m: int) -> tuple[int, int]:
    """BSGS split (n1 baby, n2 giant), n1*n2 = m, both powers of two, minimising the key switches
    of one encrypted-weight mat-vec: (n1-1) baby rotations + (n2-1) giant rotations + n2 relins.

    The giant-step rotation acts on the sum of n1 tensor products, which is a degree-2 ciphertext;
    it has to be relinearised first, so lazy relinearisation saves relins only inside a giant
    block (n2 relins, not 1 as the brief's formula has: 1 relin is the n2 = 1 special case).
    Ties go to the larger n1 (fewer relins, more hoistable rotations).
    """
    best = None
    n1 = m
    while n1 >= 1:
        n2 = m // n1
        ks = (n1 - 1) + (n2 - 1) + n2
        if best is None or ks < best[0]:
            best = (ks, n1, n2)
        n1 //= 2
    return best[1], best[2]


# -------------------------------------------------------------------------------------------------
# Bootstrapping (OpenFHE 1.2.1 as ReBoot configures it)
# -------------------------------------------------------------------------------------------------


def _select_layers(log_slots: int, budget: int) -> tuple[int, int, int]:
    """Port of OpenFHE ckksrns-utils.cpp SelectLayers()."""
    layers = math.ceil(log_slots / budget)
    rows = log_slots // layers
    rem = log_slots % layers
    dim = rows + (1 if rem else 0)
    if dim < budget:
        layers -= 1
        rows = log_slots // layers
        rem = log_slots - rows * layers
        dim = rows + (1 if rem else 0)
        while dim != budget:
            rows -= 1
            rem = log_slots - rows * layers
            dim = rows + (1 if rem else 0)
    return layers, rows, rem


def _lt_rotations(slots: int, budget: int) -> int:
    """Key switches of one CoeffsToSlots / SlotsToCoeffs (OpenFHE GetCollapsedFFTParams with
    dim1 = 0, EvalCoeffsToSlots): per level g-1 hoisted baby rotations (one of the g offsets is 0)
    + b-1 giant rotations; `budget - 1` full levels (if a remainder level exists) + the remainder."""
    log_slots = max(1, log2i(slots))
    layers_collapse, _rows, rem_collapse = _select_layers(log_slots, budget)

    def gb(collapse: int) -> tuple[int, int]:
        num_rot = (1 << (collapse + 1)) - 1
        g = 1 << (collapse // 2 + (2 if num_rot > 7 else 1))
        return g, (num_rot + 1) // g

    g, b = gb(layers_collapse)
    full_levels = budget - 1 if rem_collapse else budget
    total = full_levels * ((g - 1) + (b - 1))
    if rem_collapse:
        g_r, b_r = gb(rem_collapse)
        total += (g_r - 1) + (b_r - 1)
    return total


def _ps_mults(degree: int) -> int:
    """OpenFHE's own non-scalar-mult estimate for a Paterson-Stockmeyer Chebyshev series
    (ckksrns-utils.cpp: ComputeDegreesPS + 'k + 2m + 2^(m-1) - 4'); m from PopulateParameterPS for
    degree 44 (sparse-secret EvalMod polynomial, 45 coefficients) is 3, k = floor(n/(2^m-1)) + 1."""
    table = [(2, 1), (11, 2), (13, 3), (17, 2), (55, 3), (59, 4), (76, 3), (239, 4)]
    m = next(mm for hi, mm in table if degree <= hi)
    k = degree // ((1 << m) - 1) + 1
    return k + 2 * m + (1 << (m - 1)) - 4


def bootstrap_key_switches(slots: int, level_budget=(2, 2), iterations: int = 2) -> dict:
    """Key switches of one OpenFHE EvalBootstrap(ct, num_iterations) on a fully packed ciphertext
    with ReBoot's settings (experiments/config/fhe_config_mlp1.yaml: level_budget [2, 2],
    SPARSE_TERNARY, num_iterations 2 -> Meta-BTS = two full bootstraps, ckksrns-fhe.cpp:366-378).

    Per bootstrap: CoeffsToSlots + SlotsToCoeffs rotations (exact, from the code), 1 conjugation
    (full packing), and two EvalMod evaluations (real and imaginary part, ckksrns-fhe.cpp:572-588),
    each = Chebyshev series of degree 44 (estimate via OpenFHE's own PS mult formula) + R_SPARSE = 3
    double-angle squarings. The EvalMod part is an ESTIMATE (~13% of the total).
    """
    lt = _lt_rotations(slots, level_budget[0]) + _lt_rotations(slots, level_budget[1])
    conj = 1
    evalmod = 2 * (_ps_mults(44) + 3)
    one = lt + conj + evalmod
    return {
        "linear_transform_rotations": iterations * lt,
        "conjugations": iterations * conj,
        "evalmod_relins": iterations * evalmod,
        "total": iterations * one,
        "iterations": iterations,
    }


# -------------------------------------------------------------------------------------------------
# Per-layer costs
# -------------------------------------------------------------------------------------------------


def cost(layout: str, d_in: int, d_out: int, batch: int, slots: int, **opts) -> dict:
    """Ops of ONE layer (W: d_out x d_in, then activation) for one weight update of `batch`
    samples. Phases: forward (mat-vec + activation), transpose (W -> W^T, once per update),
    backward (activation derivative + W^T delta if `backward`), gradient (delta a^T summed over the
    batch), update (optimizer + re-layout; bootstraps counted separately).

    Common options: backward=True (compute W^T delta; False for the input layer),
    activation="square" (None for no activation), momentum=True (NAG with velocity, as ReBoot),
    decay=True (weight decay term).
    Layout options: reboot: packing="row"|"col", c=<col_size>, lazy_relin=False;
    diag_batched: samples_per_ct=<B'>; diag_2020: convention="paper"|"seal", first_layer=False.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    if batch < 1 or slots < 1:
        raise ValueError("batch and slots must be >= 1")
    fn = {
        "diag_batched": _diag_batched,
        "reboot": _reboot,
        "row_2020": _row_2020,
        "diag_2020": _diag_2020,
    }[layout]
    phases, extra = fn(d_in, d_out, batch, slots, **opts)
    out = {op: sum(phases[p][op] for p in PHASES) for op in OPS}
    out["paper_mults"] = sum(phases[p]["paper_mults"] for p in PHASES)
    out["key_switches"] = out["rotations"] + out["relins"]
    out["phases"] = phases
    out["batch"] = batch
    out["slots"] = slots
    out.update(extra)
    out.setdefault("ciphertexts", batch)
    out.setdefault("infeasible", False)
    return out


def _phases():
    return {p: _zero() for p in PHASES}


def _diag_batched(
    d_in,
    d_out,
    batch,
    slots,
    *,
    backward=True,
    activation="square",
    momentum=True,
    decay=True,
    samples_per_ct=None,
):
    """Batched diagonal packing (this work). Layout invariant: an activation vector of dimension d
    for B' samples is stored S-periodically, slot s holds sample (s mod B') feature
    ((s div B') mod d), so any rotation by a multiple of B' is a cyclic feature rotation.

    Weights (padded to powers of two, m = min(d_in, d_out), L = max(d_in, d_out)) are m generalised
    diagonals D_c[j] = W[j mod d_out, (j + c) mod d_in], j < L, each replicated over the B' samples
    ("expanded") and stored pre-rotated for BSGS (D~_c = rot(D_c, -k*n1*B') for c = k*n1 + j0).

    forward  (per ciphertext of B' samples): y = sum_k rot_{k n1}( sum_j0 D~_{k n1+j0} (x) rot_j0(x) )
             -> m tensor products, n1-1 baby + n2-1 giant rotations, n2 relins (bsgs_split);
             wide (d_in > d_out): + log2(d_in/d_out) fold rotations; tall (d_in < d_out): the input
             is already periodic with period d_in*B', so replication is free (no rotation).
             activation: 1 tensor + 1 relin.
    transpose (per update): diagonal i of W^T (pre-rotated for the backward BSGS) is ONE rotation of
             one expanded diagonal of W: E~_i = rot(D~_c(i), t_i - k'(i) n1 + k(c) n1), c(i) = -i
             mod m, t_i = -c (wide) or i (tall); zero-amount rotations are free (exact enumeration,
             = m - 1 for these splits). No multiplication, no relin.
    backward (per ciphertext): activation derivative (2z+1 or 2z: 1 scalar mult + 1 tensor +
             1 relin); W^T delta: same BSGS cost with fold if d_out > d_in.
    gradient: G~_c = sum_b rot_{-k n1}(delta) (x) rot_j0(a) reuses the forward baby-step rotations
             of the layer input; per ciphertext n2-1 rotations of delta + m tensor products; summed
             over ciphertexts at degree 2 (lazy), then per update: m relins, m*log2(B') rotations
             (rotate-and-add over the batch index), 1 mask per diagonal (keep b = 0 of one period),
             1 lane shift per diagonal into compact ciphertexts (m - nW rotations).
    update:  compact layout: S/L diagonals per ciphertext (nW = ceil(m*L/S) ciphertexts) for W and
             the NAG velocity V; NAG = scalar mults only; bootstrap W and V (2*nW per update).
             Unpack for the next forward: per diagonal 1 mask + 1 lane shift (m - nW rotations) +
             log2(S/L) replication rotations (over the B' samples and the S/(L*B') periods).
    Infeasible (B > S/L, i.e. d_pad*B > S): k = ceil(B/B') ciphertexts per batch; per-ciphertext
    phases scale with k, per-update phases do not.
    """
    din, dout = next_pow2(d_in), next_pow2(d_out)
    m, L = min(din, dout), max(din, dout)
    if L > slots:
        raise ValueError(f"layer {d_in}x{d_out} does not fit {slots} slots")
    bp = samples_per_ct or min(next_pow2(batch), slots // L)
    if L * bp > slots:
        raise ValueError(f"samples_per_ct={bp} too large for L={L}, slots={slots}")
    k = math.ceil(batch / bp)
    n1, n2 = bsgs_split(m)
    f_fwd = log2i(din // dout) if din > dout else 0
    f_bwd = log2i(dout // din) if dout > din else 0
    cap = slots // L
    nw = math.ceil(m / cap)
    act = has_activation(activation)

    ph = _phases()
    # forward, per ciphertext
    _add(ph["forward"], k, ct_ct_mults=m, relins=n2, rotations=(n1 - 1) + (n2 - 1) + f_fwd)
    if act:
        _add(ph["forward"], k, ct_ct_mults=1, relins=1)
    # transpose, per update
    if backward:
        _add(ph["transpose"], rotations=_transpose_rotations(din, dout, n1))
    # backward, per ciphertext
    if act:
        _add(ph["backward"], k, ct_ct_mults=1, relins=1, ct_pt_mults=1)
    if backward:
        _add(ph["backward"], k, ct_ct_mults=m, relins=n2, rotations=(n1 - 1) + (n2 - 1) + f_bwd)
    # gradient
    _add(ph["gradient"], k, rotations=n2 - 1, ct_ct_mults=m)
    _add(ph["gradient"], relins=m, rotations=m * log2i(bp) + (m - nw), ct_pt_mults=m)
    # update: optimizer on compact ciphertexts, then unpack (mask, lane shift, replicate)
    _add(ph["update"], ct_pt_mults=nw * nag_pt_mults(momentum, decay))
    _add(ph["update"], ct_pt_mults=m, rotations=(m - nw) + m * log2i(slots // L))
    for p in ph.values():
        p["paper_mults"] = p["ct_ct_mults"]
    extra = {
        "bootstraps_per_update": nw * (2 if momentum else 1),
        "ciphertexts": k,
        "samples_per_ct": bp,
        "infeasible": k > 1,
        "d_in_pad": din,
        "d_out_pad": dout,
        "bsgs": (n1, n2),
        "weight_ciphertexts": nw,
    }
    return ph, extra


def _transpose_rotations(din: int, dout: int, n1: int) -> int:
    m, L = min(din, dout), max(din, dout)
    count = 0
    for i in range(m):
        if din >= dout:
            c = (-i) % dout
            t = -c
        else:
            c = (-i) % din
            t = i
        amount = (t - (i // n1) * n1 + (c // n1) * n1) % L
        count += amount != 0
    return count


def _reboot(
    d_in,
    d_out,
    batch,
    slots,
    *,
    packing="row",
    c=None,
    backward=True,
    activation="square",
    momentum=True,
    decay=True,
    lazy_relin=False,
):
    """ReBoot, one sample per ciphertext (cpp/lib/packing.cpp loops over the batch vector).

    Matrix r x c with r * c = N/2 (paper Sec. 3; lib/cryptocontext.py:162-165 scales row_size by
    num_slots // (row_size*col_size) so the matrix fills all slots), c = col_size.
    * RE-Matmul (row-encoded W, expanded input) = 1 ct-ct mult + EvalSumRows: log2(r) = log2(S/c)
      rotations (paper eq. 4 and Sec. 4.2.1 "log2(r) additions and rotations"; OpenFHE
      base-advancedshe.cpp EvalSum2nComplexRows loops ceil(log2(m/(4*rowSize))) times).
    * CE-Matmul (column-encoded W, repeated input) = 1 ct-ct mult + EvalSumCols: log2(c) rotations,
      1 plaintext mask, log2(c) rotations (paper eq. 5 and Sec. 4.2.1 "2 log2(c)"; OpenFHE
      EvalSumCols = EvalSum2nComplex + mask + EvalSum2nComplexCols).
    * Backward of a row layer uses EvalSumCols, of a column layer EvalSumRows (packing.cpp
      row_packing_backward / col_packing_backward; paper Sec. 4.3.3 "no additional operations are
      required to reorder values" = no transpose).
    * Activation (PolyReLU z^2+z or square): 1 ct-ct mult; derivative delta*(2z+1): 1 scalar mult +
      1 ct-ct mult (nn.cpp poly_relu_backward / square_backward).
    * Weight gradient dW = sum_i X_i * delta_i (packing.cpp packing_weight_update): 1 ct-ct mult per
      sample, no rotation (paper Sec. 4.3.3: outer product = element-wise product of an expanded and
      a repeated vector). OpenFHE EvalMult relinearises every product; `lazy_relin=True` is a
      hypothetical improvement (EvalMultNoRelin + 1 relin per update) for sensitivity only.
    * Update: NAG, scalar mults only (nn.cpp compute_nesterov); then bootstrap W and V once per
      mini-batch (lib/utils/train.py:76 model.bootstrap; lib/models/models.py:131 weights +
      velocities; paper Alg. 1 lines 11-12).
    No hoisting is used (rotations inside EvalSumRows/EvalSumCols are sequentially dependent),
    no multi-sample packing, no reuse of expansions (none are needed: formats alternate).
    """
    if packing not in ("row", "col"):
        raise ValueError("packing must be 'row' or 'col'")
    if c is None:
        c = next_pow2(d_out) if packing == "row" else next_pow2(d_in)
    r = slots // c
    rows_needed = d_in if packing == "row" else d_out
    cols_needed = d_out if packing == "row" else d_in
    infeasible = r < next_pow2(rows_needed) or c < next_pow2(cols_needed) or r * c != slots
    sum_rows = log2i(r) if r >= 1 else 0
    sum_cols = 2 * log2i(c)
    b = batch
    act = has_activation(activation)

    ph = _phases()
    if packing == "row":
        _add(ph["forward"], b, ct_ct_mults=1, relins=1, rotations=sum_rows)
    else:
        _add(ph["forward"], b, ct_ct_mults=1, relins=1, rotations=sum_cols, ct_pt_mults=1)
    if act:
        _add(ph["forward"], b, ct_ct_mults=1, relins=1)
        _add(ph["backward"], b, ct_ct_mults=1, relins=1, ct_pt_mults=1)
    if backward:
        if packing == "row":
            _add(ph["backward"], b, ct_ct_mults=1, relins=1, rotations=sum_cols, ct_pt_mults=1)
        else:
            _add(ph["backward"], b, ct_ct_mults=1, relins=1, rotations=sum_rows)
    _add(ph["gradient"], b, ct_ct_mults=1)
    _add(ph["gradient"], relins=1 if lazy_relin else b)
    _add(ph["update"], ct_pt_mults=nag_pt_mults(momentum, decay))
    for p in ph.values():
        p["paper_mults"] = p["ct_ct_mults"]
    extra = {
        "bootstraps_per_update": 2 if momentum else 1,
        "ciphertexts": b,
        "samples_per_ct": 1,
        "infeasible": infeasible,
        "reboot_rc": (r, c),
    }
    return ph, extra


def _row_2020(
    d_in, d_out, batch, slots, *, backward=True, activation="square", momentum=True, decay=True
):
    """Row packing, Mihara et al. 2020. W (M x N, M = d_out, N = d_in), one ciphertext per row.

    * mat-vec (Alg. 1; Sec. 2 "consumes 2M multiplications and MN rotations"): M ct-ct (w_i x x) +
      M ct-pt (unit vector u_i), M*N rotations (Alg. 1 line 2 counts s = 0..N-1, as the paper).
    * transpose (Alg. 3; Sec. 3.1 "N x M multiplications and N x M rotations"): N*M ct-pt products
      c_i x u_j (unit vectors are plaintexts) + N*M rotations.
    * activation / derivative: 1 ct-ct each (Table 2 "square actv").
    Table 2's "mult" column = M per mat-vec (only the ct-ct half of Sec. 2's 2M) but N*M for the
    transition (which has only ct-pt products), so the key ``paper_mults`` = ct-ct + transpose ct-pt
    reproduces Table 2; ct-ct and ct-pt are also reported separately.
    * gradient (NOT in the paper; model extension): per output row k, mask delta_k (ct-pt), move it
      to slot 0 and broadcast over the row (1 + log2(N) rotations), times a (1 ct-ct).
    * update: scalar mults; M weight ciphertexts (+ M velocities with momentum) to refresh.
    Per-sample phases are multiplied by `batch` (B = 1 layout); transpose/update once per update.
    """
    M, N = d_out, d_in
    b = batch
    act = has_activation(activation)
    ph = _phases()
    _add(ph["forward"], b, ct_ct_mults=M, ct_pt_mults=M, relins=M, rotations=M * N)
    if act:
        _add(ph["forward"], b, ct_ct_mults=1, relins=1)
        _add(ph["backward"], b, ct_ct_mults=1, relins=1)
    if backward:
        _add(ph["transpose"], ct_pt_mults=N * M, rotations=N * M)
        ph["transpose"]["paper_mults"] = N * M
        _add(ph["backward"], b, ct_ct_mults=N, ct_pt_mults=N, relins=N, rotations=N * M)
    _add(ph["gradient"], b * M, ct_pt_mults=1, rotations=1 + log2i(next_pow2(N)), ct_ct_mults=1)
    _add(ph["gradient"], b * M, relins=1)
    _add(ph["update"], ct_pt_mults=M * nag_pt_mults(momentum, decay))
    for name, p in ph.items():
        if name != "transpose":
            p["paper_mults"] = p["ct_ct_mults"]
    extra = {"bootstraps_per_update": M * (2 if momentum else 1), "samples_per_ct": 1}
    return ph, extra


def _diag_2020(
    d_in,
    d_out,
    batch,
    slots,
    *,
    backward=True,
    activation="square",
    momentum=True,
    decay=True,
    convention="paper",
    first_layer=False,
):
    """Diagonal packing, Mihara et al. 2020 (B = 1, naive rotations, relin after every product).
    W (M x N, M = d_out, N = d_in) as N generalised diagonals (Alg. 2: P = sum_i w_i x rot(x, -i)).

    convention="paper" (reproduces the paper's own counting):
      mat-vec: N ct-ct, N rotations (Sec. 2 "N multiplications and N rotations"; rot(x, 0) counted)
      transpose (Alg. 5, stepped diagonals): M rotations, 0 mults (Table 1 "with step": M)
      backward W^T delta: mat-vec on the N x M transpose: M ct-ct, M rotations
      activation / derivative: 1 ct-ct each.
      NOTE Table 2 itself lists Dense(a->b) with b rotations, not a; totals coincide (see tests).
    convention="seal" (Alg. 2/5 as Task 4 implemented them on SEAL 4.1, spikes/seal-2020): only
      non-trivial rotations (N-1 per mat-vec, M-1 per transpose), plus log-doubling replication of
      the layer input from period N to N+M-1 (not for the client-encrypted first-layer input) and of
      delta from period M to M+N-1 before W^T delta.
    gradient (not in the paper; Task 4's stepped outer product G'_i = rot(delta, i) x x_rep):
      N ct-ct + N rotations (paper convention) / N-1 (seal).
    update: N weight ciphertexts (+ N velocities with momentum).
    """
    if convention not in ("paper", "seal"):
        raise ValueError("convention must be 'paper' or 'seal'")
    M, N = d_out, d_in
    b = batch
    act = has_activation(activation)
    triv = 0 if convention == "paper" else 1
    ph = _phases()
    rep_in = 0 if (convention == "paper" or first_layer) else doublings(N, N + M - 1)
    _add(ph["forward"], b, ct_ct_mults=N, relins=N, rotations=N - triv + rep_in)
    if act:
        _add(ph["forward"], b, ct_ct_mults=1, relins=1)
        _add(ph["backward"], b, ct_ct_mults=1, relins=1)
    if backward:
        _add(ph["transpose"], rotations=M - triv)
        rep_d = 0 if convention == "paper" else doublings(M, M + N - 1)
        _add(ph["backward"], b, ct_ct_mults=M, relins=M, rotations=M - triv + rep_d)
    _add(ph["gradient"], b, ct_ct_mults=N, relins=N, rotations=N - triv)
    _add(ph["update"], ct_pt_mults=N * nag_pt_mults(momentum, decay))
    for p in ph.values():
        p["paper_mults"] = p["ct_ct_mults"]
    extra = {"bootstraps_per_update": N * (2 if momentum else 1), "samples_per_ct": 1}
    return ph, extra


# -------------------------------------------------------------------------------------------------
# Network
# -------------------------------------------------------------------------------------------------


def _norm_layers(layers):
    out = []
    for l in layers:
        if isinstance(l, dict):
            out.append((int(l["in"]), int(l["out"]), l.get("activation")))
        else:
            d_in, d_out, *rest = l
            out.append((int(d_in), int(d_out), rest[0] if rest else None))
    return out


def reboot_rc(layers, slots):
    """ReBoot's (r, c): lib/cryptocontext.py get_recommended_parameters (plain MLP, no local-loss
    blocks, layers alternating row/col packing starting with row) united with the padding
    constraints of CKKSPackedLinear.add_padding (row layer: in <= r, out <= c; col layer: in <= c,
    out <= r); then r is scaled so that r * c = slots (cryptocontext.py:162-165)."""
    rows, cols = [layers[0][0]], [layers[0][1]]
    for i, (d_in, d_out, _a) in enumerate(layers):
        if i % 2 == 0:  # row-encoded: ReBoot lists out in rows; padding needs in <= r, out <= c
            rows += [d_out, d_in]
            cols += [d_out]
        else:  # column-encoded: ReBoot lists out in cols; padding needs in <= c, out <= r
            cols += [d_out, d_in]
            rows += [d_out]
    r0, c = next_pow2(max(rows)), next_pow2(max(cols))
    ratio = slots // (r0 * c)
    return r0 * ratio, c, ratio == 0


def network_cost(
    layout: str,
    layers,
    batch: int,
    slots: int,
    *,
    scope=PHASES,
    first_backward: bool = False,
    boot_ks: float | None = None,
    **opts,
) -> dict:
    """Sum of `cost` over the layers of an MLP for one weight update of `batch` samples.

    layers: [(d_in, d_out, activation), ...] or [{"in", "out", "activation"}, ...].
    The input layer does not propagate delta backward (ReBoot: propagate_backward=False; Task 4
    likewise) unless first_backward=True (Table 2 counts it). `scope` limits the phases summed.
    per_sample: ks = (rotations + relins)/batch, ks_spec = (rotations + relins +
    bootstraps*boot_ks)/batch with boot_ks = bootstrap_key_switches(slots)["total"] by default.
    """
    layers = _norm_layers(layers)
    extra_net = {}
    per_layer_opts = [dict(opts) for _ in layers]
    if layout == "reboot":
        r, c, bad = reboot_rc(layers, slots)
        extra_net["reboot_rc"] = (r, c)
        for i, o in enumerate(per_layer_opts):
            o.setdefault("packing", "row" if i % 2 == 0 else "col")
            o.setdefault("c", c)
    elif layout == "diag_batched":
        L = max(max(next_pow2(a), next_pow2(b)) for a, b, _ in layers)
        bp = opts.get("samples_per_ct") or min(next_pow2(batch), slots // L)
        for o in per_layer_opts:
            o["samples_per_ct"] = bp
    elif layout == "diag_2020":
        for i, o in enumerate(per_layer_opts):
            o.setdefault("first_layer", i == 0)

    res = [
        cost(
            layout,
            d_in,
            d_out,
            batch,
            slots,
            backward=(i > 0 or first_backward),
            activation=act,
            **per_layer_opts[i],
        )
        for i, (d_in, d_out, act) in enumerate(layers)
    ]
    phases = {p: _zero() for p in PHASES}
    for r_ in res:
        for p in scope:
            for k in OPS + ("paper_mults",):
                phases[p][k] += r_["phases"][p][k]
    out = {op: sum(phases[p][op] for p in PHASES) for op in OPS}
    out["paper_mults"] = sum(phases[p]["paper_mults"] for p in PHASES)
    out["key_switches"] = out["rotations"] + out["relins"]
    out["bootstraps_per_update"] = (
        sum(r_["bootstraps_per_update"] for r_ in res) if "update" in scope else 0
    )
    out["phases"] = phases
    out["layers"] = res
    out["batch"] = batch
    out["slots"] = slots
    out["ciphertexts"] = max(r_["ciphertexts"] for r_ in res)
    out["infeasible"] = any(r_["infeasible"] for r_ in res) or extra_net.get("_bad", False)
    if layout == "reboot":
        out["infeasible"] = out["infeasible"] or bad
    out.update(extra_net)
    if boot_ks is None:
        boot_ks = bootstrap_key_switches(slots)["total"]
    out["boot_ks"] = boot_ks
    out["per_sample"] = {
        "ks": out["key_switches"] / batch,
        "ks_spec": (out["key_switches"] + out["bootstraps_per_update"] * boot_ks) / batch,
        "rotations": out["rotations"] / batch,
        "relins": out["relins"] / batch,
        "bootstraps": out["bootstraps_per_update"] / batch,
    }
    return out
