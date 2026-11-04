"""Slot-exact plaintext simulator for the packing layouts (verification of costmodel.py).

A "ciphertext" is a float vector of `slots` entries plus a degree (1 = fresh/relinearised, 2 =
after a tensor product). The simulator counts every key-switching op and refuses what CKKS cannot
do without a key switch first (rotating or multiplying a degree-2 ciphertext), so lazy
relinearisation cannot be over-claimed. Rotation convention: rot(v, k)[s] = v[s + k] (left by k).

`diag_batched_step` and `reboot_step` run ONE training step of an MLP (forward, loss gradient
y_pred - y, backward, weight gradient summed over the batch, NAG update with weight decay as
ReBoot's compute_nesterov, bootstrap of W and V) and return the decoded intermediate values, the
numpy reference values, and the per-phase op counts.

ReBoot is GPL-3.0: `reboot_step` re-implements ReBoot's layout from its paper (Sec. 3, eq. 4-5) to
count operations; references to ReBoot source files (e.g. compute_nesterov) only say where an op
count was read. No ReBoot code is copied.
"""

from __future__ import annotations

import math

import numpy as np

from costmodel import (
    OPS,
    PHASES,
    _norm_layers,
    bsgs_split,
    has_activation,
    log2i,
    next_pow2,
    reboot_rc,
)

LR, MOM, DECAY = 0.05, 0.9, 0.01


class Ct:
    __slots__ = ("v", "deg")

    def __init__(self, v, deg=1):
        self.v = v
        self.deg = deg


class Sim:
    def __init__(self, slots):
        self.S = slots
        self.counts = {p: {op: 0 for op in OPS} for p in PHASES}
        self.phase = None
        self.bootstraps = 0

    def _c(self, op):
        self.counts[self.phase][op] += 1

    def rot(self, a, k):
        assert a.deg == 1, "rotation of a degree-2 ciphertext needs a relinearisation first"
        k %= self.S
        if k == 0:
            return a
        self._c("rotations")
        return Ct(np.roll(a.v, -k), 1)

    def tensor(self, a, b):
        assert a.deg == 1 and b.deg == 1, "tensor product of a degree-2 ciphertext"
        self._c("ct_ct_mults")
        return Ct(a.v * b.v, 2)

    def relin(self, a):
        assert a.deg == 2
        self._c("relins")
        return Ct(a.v, 1)

    def pt(self, a, p):
        self._c("ct_pt_mults")
        return Ct(a.v * p, a.deg)

    @staticmethod
    def add(a, b):
        return Ct(a.v + b.v, max(a.deg, b.deg))

    @staticmethod
    def sub(a, b):
        return Ct(a.v - b.v, max(a.deg, b.deg))

    @staticmethod
    def add_plain(a, c):
        return Ct(a.v + c, a.deg)

    def bootstrap(self, a):
        assert a.deg == 1
        self.bootstraps += 1
        return a


def _act(name, z):
    if not has_activation(name):
        return z
    return z * z + z if name == "poly_relu" else z * z


def _dact(name, z):
    return 2 * z + 1 if name == "poly_relu" else 2 * z


def _sim_act(sim, name, z):
    if not has_activation(name):
        return z
    sq = sim.relin(sim.tensor(z, z))
    return sim.add(sq, z) if name == "poly_relu" else sq


def _sim_dact(sim, name, z, delta):
    if not has_activation(name):
        return delta
    d = sim.add_plain(sim.pt(z, 2.0), 1.0 if name == "poly_relu" else 0.0)
    return sim.relin(sim.tensor(delta, d))


def _reference(layers, Ws, Vs, X, Y):
    """Dense numpy training step (ReBoot semantics: delta = y_pred - y, dW = sum over batch)."""
    acts, zs = [X], []
    for (d_in, d_out, act), W in zip(layers, Ws):
        z = acts[-1] @ W.T
        zs.append(z)
        acts.append(_act(act, z))
    out = {"forward_out": acts[-1]}
    delta = acts[-1] - Y
    for l in reversed(range(len(layers))):
        act = layers[l][2]
        dz = delta * _dact(act, zs[l]) if has_activation(act) else delta
        G = dz.T @ acts[l]
        out[f"grad_{l}"] = G
        if l > 0:
            delta = dz @ Ws[l]
            out[f"bwd_{l}"] = delta
        g = G + DECAY * Ws[l]
        upd = LR * g + MOM * LR * g + MOM * MOM * LR * Vs[l]
        out[f"V_new_{l}"] = MOM * Vs[l] + g
        out[f"W_new_{l}"] = Ws[l] - upd
    return out


def _random_problem(layers, batch, rng):
    Ws = [rng.normal(0, 0.3, (o, i)) for i, o, _ in layers]
    Vs = [rng.normal(0, 0.1, (o, i)) for i, o, _ in layers]
    X = rng.normal(0, 1, (batch, layers[0][0]))
    Y = rng.normal(0, 1, (batch, layers[-1][1]))
    return Ws, Vs, X, Y


# -------------------------------------------------------------------------------------------------
# diag_batched
# -------------------------------------------------------------------------------------------------


def diag_batched_step(layers, batch, slots, rng, compact=True, interactive=False, client_transpose=False):
    """One training step of diag_batched. interactive=True: no compaction / unpack / bootstrap; the
    server runs NAG on the expanded diagonals and the client re-encodes (lane 0 is read back).
    client_transpose=True: W^T's diagonals are client-encoded (not counted) instead of rotated."""
    layers = _norm_layers(layers)
    Ws, Vs, X, Y = _random_problem(layers, batch, rng)
    ref = _reference(layers, Ws, Vs, X, Y)
    S = slots
    sim = Sim(S)

    dims = [(next_pow2(i), next_pow2(o)) for i, o, _ in layers]
    Lmax = max(max(a, b) for a, b in dims)
    bp = min(next_pow2(batch), S // Lmax)
    k = math.ceil(batch / bp)

    def pad(M, r, c):
        out = np.zeros((r, c))
        out[: M.shape[0], : M.shape[1]] = M
        return out

    def enc_vec(M, d):  # M: (bp, d) samples x features
        s = np.arange(S)
        return Ct(M[s % bp, (s // bp) % d].astype(float))

    def dec_vec(ct, d):
        return np.array([[ct.v[j * bp + b] for j in range(d)] for b in range(bp)])

    geo = []
    for (din, dout) in dims:
        m, L = min(din, dout), max(din, dout)
        n1, n2 = bsgs_split(m)
        cap = S // L
        lanes = []
        for c in range(m):
            ell = c % cap
            lanes.append((c // cap, ell, (ell // bp) * L * bp + (ell % bp)))
        geo.append(dict(din=din, dout=dout, m=m, L=L, n1=n1, n2=n2, nw=math.ceil(m / cap), lanes=lanes))

    def diag_feat(Wp, g, c):  # D_c[j] = W[j mod dout, (j + c) mod din], j < L
        j = np.arange(g["L"])
        return Wp[j % g["dout"], (j + c) % g["din"]]

    def pre_rot_shift(g, c):  # D~_c[j] = D_c[j - k n1]
        return (c // g["n1"]) * g["n1"]

    def enc_expanded(feat, L):
        s = np.arange(S)
        return Ct(feat[(s // bp) % L].astype(float))

    def build_compact(mats, g):
        cts = [np.zeros(S) for _ in range(g["nw"])]
        for c, (n, _ell, o) in enumerate(g["lanes"]):
            f = np.roll(diag_feat(mats, g, c), pre_rot_shift(g, c))
            cts[n][np.arange(g["L"]) * bp + o] = f
        return [Ct(v) for v in cts]

    def decode_compact(cts, g):
        M = np.zeros((g["dout"], g["din"]))
        j = np.arange(g["L"])
        for c, (n, _ell, o) in enumerate(g["lanes"]):
            ft = cts[n].v[j * bp + o]
            f = np.roll(ft, -pre_rot_shift(g, c))
            M[j % g["dout"], (j + c) % g["din"]] = f
        return M

    def decode_expanded(diags, g):
        M = np.zeros((g["dout"], g["din"]))
        j = np.arange(g["L"])
        for c, ct in enumerate(diags):
            f = np.roll(ct.v[j * bp], -pre_rot_shift(g, c))
            M[j % g["dout"], (j + c) % g["din"]] = f
        return M

    Wp = [pad(W, g["dout"], g["din"]) for W, g in zip(Ws, geo)]
    Vp = [pad(V, g["dout"], g["din"]) for V, g in zip(Vs, geo)]
    # client-side initial state: expanded pre-rotated diagonals + compact W and V
    Wt = [
        [enc_expanded(np.roll(diag_feat(Wp[l], g, c), pre_rot_shift(g, c)), g["L"]) for c in range(g["m"])]
        for l, g in enumerate(geo)
    ]
    Wc = [build_compact(Wp[l], g) for l, g in enumerate(geo)]
    Vc = [build_compact(Vp[l], g) for l, g in enumerate(geo)]
    Vt = [
        [enc_expanded(np.roll(diag_feat(Vp[l], g, c), pre_rot_shift(g, c)), g["L"]) for c in range(g["m"])]
        for l, g in enumerate(geo)
    ]

    def replicate(v, L):
        for t in range(log2i(bp)):
            v = sim.add(v, sim.rot(v, -(1 << t)))
        for t in range(log2i(S // (L * bp))):
            v = sim.add(v, sim.rot(v, -L * bp * (1 << t)))
        return v

    def bsgs(diags, baby, g):
        n1, n2 = g["n1"], g["n2"]
        block0, acc = None, None
        for kk in range(n2):
            inner = None
            for j0 in range(n1):
                p = sim.tensor(diags[kk * n1 + j0], baby[j0])
                inner = p if inner is None else sim.add(inner, p)
            if kk == 0:
                block0 = inner
            else:
                r = sim.rot(sim.relin(inner), kk * n1 * bp)
                acc = r if acc is None else sim.add(acc, r)
        y = block0 if acc is None else sim.add(block0, acc)
        return sim.relin(y)

    def fold(y, out_dim, in_dim):
        for t in range(log2i(in_dim // out_dim) if in_dim > out_dim else 0):
            y = sim.add(y, sim.rot(y, out_dim * bp * (1 << t)))
        return y

    # ---- transpose (once per update, from the expanded weights) ----
    sim.phase = "transpose"
    Et = [None] * len(layers)
    for l, g in enumerate(geo):
        if l == 0:
            continue
        din, dout, n1, L = g["din"], g["dout"], g["n1"], g["L"]
        E = []
        for i in range(g["m"]):
            if din >= dout:
                c = (-i) % dout
                t = -c
            else:
                c = (-i) % din
                t = i
            amount = (t - (i // n1) * n1 + (c // n1) * n1) % L
            if client_transpose:  # client encodes the same vector at re-encryption (free)
                E.append(Ct(np.roll(Wt[l][c].v, -amount * bp)))
            else:
                E.append(sim.rot(Wt[l][c], amount * bp))
        Et[l] = E

    # ---- forward, per ciphertext ----
    Xp = np.zeros((k * bp, dims[0][0]))
    Xp[:batch, : X.shape[1]] = X
    Yp = np.zeros((k * bp, dims[-1][1]))
    Yp[:batch, : Y.shape[1]] = Y
    store = []
    sim.phase = "forward"
    for q in range(k):
        x = enc_vec(Xp[q * bp : (q + 1) * bp], dims[0][0])
        babies, zs = [], []
        for l, g in enumerate(geo):
            baby = [sim.rot(x, j * bp) for j in range(g["n1"])]
            babies.append(baby)
            y = fold(bsgs(Wt[l], baby, g), g["dout"], g["din"])
            zs.append(y)
            x = _sim_act(sim, layers[l][2], y)
        store.append(dict(babies=babies, zs=zs, out=x))
    decoded = {
        "forward_out": np.vstack([dec_vec(st["out"], dims[-1][1]) for st in store])[
            :batch, : layers[-1][1]
        ]
    }

    # ---- backward + gradient ----
    deltas = [sim.sub(st["out"], enc_vec(Yp[q * bp : (q + 1) * bp], dims[-1][1])) for q, st in enumerate(store)]
    Gsum = [None] * len(layers)
    for l in reversed(range(len(layers))):
        g = geo[l]
        n1, n2 = g["n1"], g["n2"]
        Gs = [None] * g["m"]
        new_deltas = []
        for q, st in enumerate(store):
            sim.phase = "backward"
            dz = _sim_dact(sim, layers[l][2], st["zs"][l], deltas[q])
            sim.phase = "gradient"
            drot = [dz] + [sim.rot(dz, -kk * n1 * bp) for kk in range(1, n2)]
            for c in range(g["m"]):
                p = sim.tensor(drot[c // n1], st["babies"][l][c % n1])
                Gs[c] = p if Gs[c] is None else sim.add(Gs[c], p)
            if l > 0:
                sim.phase = "backward"
                baby = [sim.rot(dz, j * bp) for j in range(n1)]
                y = fold(bsgs(Et[l], baby, g), g["din"], g["dout"])
                new_deltas.append(y)
        if l > 0:
            deltas = new_deltas
            decoded[f"bwd_{l}"] = np.vstack([dec_vec(d, g["din"]) for d in deltas])[
                :batch, : layers[l][0]
            ]
        Gsum[l] = Gs

    # ---- per-update gradient finish: relin, batch reduction, mask, pack ----
    sim.phase = "gradient"
    s = np.arange(S)
    Gc = []
    for l, g in enumerate(geo):
        L = g["L"]
        mask0 = ((s % bp == 0) & (s < L * bp)).astype(float)
        if interactive:  # reduce only: lane b = 0 holds the sum, other lanes are ignored later
            Ge = []
            for c in range(g["m"]):
                v = sim.relin(Gsum[l][c])
                for t in range(log2i(bp)):
                    v = sim.add(v, sim.rot(v, 1 << t))
                Ge.append(v)
            Gc.append(Ge)
            decoded[f"grad_{l}"] = decode_expanded(Ge, g)[: layers[l][1], : layers[l][0]]
            continue
        if not compact:  # replicate each reduced diagonal straight back to the expanded layout
            Ge = []
            for c in range(g["m"]):
                v = sim.relin(Gsum[l][c])
                for t in range(log2i(bp)):
                    v = sim.add(v, sim.rot(v, 1 << t))
                Ge.append(replicate(sim.pt(v, mask0), L))
            Gc.append(Ge)
            decoded[f"grad_{l}"] = decode_expanded(Ge, g)[: layers[l][1], : layers[l][0]]
            continue
        cts = [Ct(np.zeros(S)) for _ in range(g["nw"])]
        for c in range(g["m"]):
            v = sim.relin(Gsum[l][c])
            for t in range(log2i(bp)):
                v = sim.add(v, sim.rot(v, 1 << t))
            v = sim.pt(v, mask0)
            n, _ell, o = g["lanes"][c]
            cts[n] = sim.add(cts[n], sim.rot(v, -o))
        Gc.append(cts)
        decoded[f"grad_{l}"] = decode_compact(cts, g)[: layers[l][1], : layers[l][0]]

    # ---- update: NAG on compact ciphertexts, bootstrap, unpack ----
    sim.phase = "update"
    for l, g in enumerate(geo):
        L = g["L"]
        if interactive:  # NAG on the expanded diagonals; the client decrypts lane 0 and re-encodes
            newW, newV = [], []
            for c in range(g["m"]):
                gg = sim.add(Gc[l][c], sim.pt(Wt[l][c], DECAY))
                upd = sim.add(sim.add(sim.pt(gg, LR), sim.pt(gg, MOM * LR)), sim.pt(Vt[l][c], MOM * MOM * LR))
                newV.append(sim.add(sim.pt(Vt[l][c], MOM), gg))
                newW.append(sim.sub(Wt[l][c], upd))
            decoded[f"W_new_{l}"] = decode_expanded(newW, g)[: layers[l][1], : layers[l][0]]
            decoded[f"V_new_{l}"] = decode_expanded(newV, g)[: layers[l][1], : layers[l][0]]
            continue
        if not compact:  # NAG + bootstrap on every expanded W / V diagonal
            newW, newV = [], []
            for c in range(g["m"]):
                gg = sim.add(Gc[l][c], sim.pt(Wt[l][c], DECAY))
                upd = sim.add(sim.add(sim.pt(gg, LR), sim.pt(gg, MOM * LR)), sim.pt(Vt[l][c], MOM * MOM * LR))
                vn = sim.add(sim.pt(Vt[l][c], MOM), gg)
                newW.append(sim.bootstrap(sim.sub(Wt[l][c], upd)))
                newV.append(sim.bootstrap(vn))
            decoded[f"W_new_{l}"] = decode_expanded(newW, g)[: layers[l][1], : layers[l][0]]
            decoded[f"V_new_{l}"] = decode_expanded(newV, g)[: layers[l][1], : layers[l][0]]
            continue
        newW, newV = [], []
        for n in range(g["nw"]):
            gg = sim.add(Gc[l][n], sim.pt(Wc[l][n], DECAY))
            upd = sim.add(sim.add(sim.pt(gg, LR), sim.pt(gg, MOM * LR)), sim.pt(Vc[l][n], MOM * MOM * LR))
            vn = sim.add(sim.pt(Vc[l][n], MOM), gg)
            wn = sim.sub(Wc[l][n], upd)
            newW.append(sim.bootstrap(wn))
            newV.append(sim.bootstrap(vn))
        decoded[f"V_new_{l}"] = decode_compact(newV, g)[: layers[l][1], : layers[l][0]]
        diags = []
        for c in range(g["m"]):
            n, _ell, o = g["lanes"][c]
            lane = np.zeros(S)
            lane[np.arange(L) * bp + o] = 1.0
            diags.append(replicate(sim.rot(sim.pt(newW[n], lane), o), L))
        # expanded result must be the S-periodic layout the next forward expects
        W_new = decode_expanded(diags, g)
        for c, v in enumerate(diags):
            exp = enc_expanded(np.roll(diag_feat(W_new, g, c), pre_rot_shift(g, c)), L).v
            assert np.allclose(v.v, exp), "unpacked diagonal is not in expanded layout"
        decoded[f"W_new_{l}"] = W_new[: layers[l][1], : layers[l][0]]

    return {"decoded": decoded, "reference": ref, "counts": sim.counts, "bootstraps": sim.bootstraps}


# -------------------------------------------------------------------------------------------------
# ReBoot
# -------------------------------------------------------------------------------------------------


def reboot_step(layers, batch, slots, rng, custom_sumrows=False):
    """One training step of ReBoot's layout. custom_sumrows=True: r = rows needed (not scaled to
    fill the slots); every r x c encoding is replicated S/(r*c) times, sumRows uses log2(r)."""
    layers = _norm_layers(layers)
    Ws, Vs, X, Y = _random_problem(layers, batch, rng)
    ref = _reference(layers, Ws, Vs, X, Y)
    S = slots
    r, c, bad = reboot_rc(layers, S, fill=not custom_sumrows)
    assert not bad
    sim = Sim(S)
    s = np.arange(S)
    row_i, col_j = (s // c) % r, s % c  # replicas of the r x c matrix when r * c < S

    def enc_expanded(a):  # A[i, j] = a[i]
        v = np.zeros(S)
        ok = row_i < len(a)
        v[ok] = a[row_i[ok]]
        return Ct(v)

    def enc_repeated(a):  # A[i, j] = a[j]
        v = np.zeros(S)
        ok = col_j < len(a)
        v[ok] = a[col_j[ok]]
        return Ct(v)

    def dec_expanded(ct, d):
        return ct.v[np.arange(d) * c]

    def dec_repeated(ct, d):
        return ct.v[:d]

    def enc_w(W, row):  # W: out x in
        v = np.zeros(S)
        if row:  # rows = in, cols = out: A[i, j] = W[j, i]
            ok = (row_i < W.shape[1]) & (col_j < W.shape[0])
            v[ok] = W[col_j[ok], row_i[ok]]
        else:  # rows = out, cols = in: A[i, j] = W[i, j]
            ok = (row_i < W.shape[0]) & (col_j < W.shape[1])
            v[ok] = W[row_i[ok], col_j[ok]]
        return Ct(v)

    def dec_w(ct, W_shape, row):
        o, i = W_shape
        M = np.zeros((o, i))
        for a in range(o):
            for b in range(i):
                M[a, b] = ct.v[b * c + a] if row else ct.v[a * c + b]
        return M

    def sum_rows(v):
        for t in range(log2i(r)):
            v = sim.add(v, sim.rot(v, c * (1 << t)))
        return v

    mask = (col_j == 0).astype(float)

    def sum_cols(v):
        for t in range(log2i(c)):
            v = sim.add(v, sim.rot(v, 1 << t))
        v = sim.pt(v, mask)
        for t in range(log2i(c)):
            v = sim.add(v, sim.rot(v, -(1 << t)))
        return v

    Wct = [enc_w(W, l % 2 == 0) for l, W in enumerate(Ws)]
    Vct = [enc_w(V, l % 2 == 0) for l, V in enumerate(Vs)]
    out_row = (len(layers) - 1) % 2 == 0  # last layer row-encoded -> output repeated

    # forward / backward per sample
    G = [None] * len(layers)
    outs, bwds = [], {l: [] for l in range(1, len(layers))}
    for q in range(batch):
        sim.phase = "forward"
        x = enc_expanded(X[q])
        inputs, zs = [], []
        for l, (d_in, d_out, act) in enumerate(layers):
            inputs.append(x)
            p = sim.relin(sim.tensor(x, Wct[l]))
            z = sum_rows(p) if l % 2 == 0 else sum_cols(p)
            zs.append(z)
            x = _sim_act(sim, act, z)
        outs.append(dec_repeated(x, layers[-1][1]) if out_row else dec_expanded(x, layers[-1][1]))
        y = enc_repeated(Y[q]) if out_row else enc_expanded(Y[q])
        delta = sim.sub(x, y)
        for l in reversed(range(len(layers))):
            sim.phase = "backward"
            dz = _sim_dact(sim, layers[l][2], zs[l], delta)
            sim.phase = "gradient"
            gp = sim.relin(sim.tensor(inputs[l], dz))
            G[l] = gp if G[l] is None else sim.add(G[l], gp)
            if l > 0:
                sim.phase = "backward"
                p = sim.relin(sim.tensor(dz, Wct[l]))
                delta = sum_cols(p) if l % 2 == 0 else sum_rows(p)
                # row layer backward -> expanded; col layer backward -> repeated
                d_in = layers[l][0]
                bwds[l].append(dec_expanded(delta, d_in) if l % 2 == 0 else dec_repeated(delta, d_in))
    decoded = {"forward_out": np.vstack(outs)}
    for l, v in bwds.items():
        decoded[f"bwd_{l}"] = np.vstack(v)

    sim.phase = "update"
    for l in range(len(layers)):
        row = l % 2 == 0
        decoded[f"grad_{l}"] = dec_w(G[l], Ws[l].shape, row)
        gg = sim.add(G[l], sim.pt(Wct[l], DECAY))
        upd = sim.add(sim.add(sim.pt(gg, LR), sim.pt(gg, MOM * LR)), sim.pt(Vct[l], MOM * MOM * LR))
        vn = sim.bootstrap(sim.add(sim.pt(Vct[l], MOM), gg))
        wn = sim.bootstrap(sim.sub(Wct[l], upd))
        decoded[f"W_new_{l}"] = dec_w(wn, Ws[l].shape, row)
        decoded[f"V_new_{l}"] = dec_w(vn, Ws[l].shape, row)
    return {"decoded": decoded, "reference": ref, "counts": sim.counts, "bootstraps": sim.bootstraps}
