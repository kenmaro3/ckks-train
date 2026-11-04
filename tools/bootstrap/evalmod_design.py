#!/usr/bin/env python3
"""Numbers behind docs/bootstrap.md (Task 35): K, EvalMod depth, chain and precision model.

Usage: python3 tools/bootstrap/evalmod_design.py      (stdlib + numpy only, under a second)

Sections:
  1. Failure probability of the mod-raise overflow |I_j| >= K over all N coefficients
     (Gaussian for the dense key, exact Irwin-Hall for sparse keys).
  2. Sine error reachable with 8 EvalMod levels: Chebyshev interpolant of
     cos(2 pi (x - 1/4) / 2^r) on [-K, K] of degree 2^(8 - r) - 1, then r double angles,
     measured at the points x = I + delta (|I| <= K - 1, |delta| <= 2^-20) that occur.
  3. Smallest EvalMod depth for the dense key (K = 400, 420).
  4. Candidate modulus chains (digits, log PQ, levels after bootstrap, key size) and their
     predicted precision with the chosen EvalMod design (ruling R-M3-7).
  5. Secret / EvalMod alternatives on the chosen chain, for several q0/Delta.
  6. The chosen design in detail, the q0 = 52 fallback, and the degree-63 fallback.
"""
import math
from fractions import Fraction

import numpy as np
from numpy.polynomial import chebyshev as C

N = 2**16
SLOTS = N // 2
MAX_FACTOR = 4.2  # max |Gaussian| over 2^15 samples, in units of sigma


def sigma_i(h):
    """Std of I_j = round(b_j/q0 + sum_{s_k != 0} +-a_k/q0): h+1 iid U(-1/2, 1/2)."""
    return math.sqrt((h + 1) / 12)


def log2_fail_gauss(K, h):
    p = math.erfc((K - 0.5) / sigma_i(h) / math.sqrt(2))
    return math.log2(N * p)


def log2_fail_exact(K, h):
    """Exact Irwin-Hall tail, P(|S| >= K - 1/2) with S a sum of h+1 iid U(-1/2, 1/2)."""
    n = h + 1
    x = Fraction(n, 2) - Fraction(2 * K - 1, 2)
    if x <= 0:
        return float("-inf")
    s = sum((-1) ** k * math.comb(n, k) * (x - k) ** n for k in range(math.floor(x) + 1))
    return math.log2(N * 2 * float(s / math.factorial(n)))


def sin_error(K, r, d, delta=2.0**-20):
    """(max sin error near the integers, max double-angle amplification prod 4|cos 2^i theta|)."""
    f = lambda y: np.cos(2 * np.pi * (K * y - 0.25) / 2**r)
    coef = C.chebinterpolate(f, d)
    I = np.arange(-(K - 1), K)
    xs = (I[:, None] + np.linspace(-delta, delta, 5)[None, :]).ravel()
    c = C.chebval(xs / K, coef)
    for _ in range(r):
        c = 2 * c * c - 1
    err = float(np.max(np.abs(c - np.sin(2 * np.pi * xs))))
    th = 2 * np.pi * (I - 0.25) / 2**r
    amp = np.ones_like(th)
    for i in range(r):
        amp *= 4 * np.abs(np.cos(2**i * th))
    return max(err, 2.0**-80), float(amp.max())


def slot_noise(h):
    """Absolute slot-domain rms noise (real part) of one rescale and one key switch, key weight h.

    Rescale rounding: sqrt(N/2) * sqrt((h+1)/12) (2^13.2 for the dense key).
    Key switch (ModDown): docs/precision.md n16_train 'encrypt' row, 25.72 bits = max error
    2^16.28 over 2^15 slots, rms 2^16.28 / 4.2 = 2^14.2; scaled by sqrt((h+1)/(N/2+1)).
    """
    rs = math.sqrt(N / 2) * math.sqrt((h + 1) / 12)
    ks = 2**14.2 * math.sqrt((h + 1) / (N / 2 + 1))
    return math.hypot(rs, ks)


def digit_noise(log_qd_over_p, ndigits=2):
    """Worst-case key-error term of a key switch: sum over the digits whose modulus Q_d is close
    to P of [c]_{Q_d} e_d / P, coefficient std (Q_d/P) * 3.2 * sqrt(N/12), in slot rms.
    Counted at the working scale of the CtS output (pessimistic)."""
    return math.sqrt(N / 2) * math.sqrt(ndigits) * 2.0**log_qd_over_p * 3.2 * math.sqrt(N / 12)


ENC = 0.5 * math.log2(N / 12)  # log2 complex slot rms of rounding N plaintext coefficients


def cts_pt_rel(cts_bits, cts_levels, log_q0, h_mr, K, s_out=58):
    """log2 relative error (to the complex rms of the CtS output) from rounding the CtS diagonals.

    Each level's diagonals at scale D_l carry a slot error of rms sqrt(N/12)/D_l, i.e. a relative
    error 2^ENC / (a_l D_l) for entries of size a_l. The integer magnitudes fix
    prod(a_l D_l) = prod(q_l) * (abs_out / abs_in) / sqrt(N/2): abs_in = sqrt(N) sigma_I q0 (slots
    of the raised plaintext), abs_out = sqrt(2) (sigma_I / K) 2^s_out (packed y at scale 2^58),
    and each output sums N/2 inputs. Split evenly over the levels."""
    s = sigma_i(h_mr)
    abs_in = math.log2(math.sqrt(N) * s) + log_q0
    abs_out = math.log2(math.sqrt(2) * s / K) + s_out
    per = (cts_levels * cts_bits + abs_out - abs_in - math.log2(math.sqrt(SLOTS))) / cts_levels
    return ENC - per + 0.5 * math.log2(cts_levels)


def stc_pt_rel(stc_bits, stc_levels, log_ratio, s_in=58, s_out=42):
    """Same for SlotToCoeff: input sin(2 pi x) ~ 2 pi c / 2^log_ratio at scale 2^58 (two real
    halves packed), output m at scale 2^42 (rms 1/sqrt(3))."""
    rms_v = 2 * math.pi * 2.0**-log_ratio / math.sqrt(3 * N)
    abs_in = s_in + math.log2(math.sqrt(2) * rms_v)
    abs_out = s_out + math.log2(1 / math.sqrt(3))
    per = (stc_levels * stc_bits + abs_out - abs_in - math.log2(math.sqrt(SLOTS))) / stc_levels
    return ENC - per + 0.5 * math.log2(stc_levels)


def predict(chain, K=13, r=1, d=127, h_noise=32768, h_mr=32, log_scale=58, levels=8):
    """Predicted precision bits (max over 2^15 slots) and the bits each term alone would allow.

    chain: dict(q0, cts_bits, cts_levels, stc_bits, stc_levels, log_qd_over_p).
    Terms (EvalMod-side ones in x = t/q0 units, mapped to slots by 4.2 sqrt(N/2) q0/Delta):
      input   K * sqrt(cts_levels) * (rescale + key switch + digit term) / 2^58
      internal sqrt(levels) * (rescale + key switch) / 2^58 * amp / (2 pi)
      approx  max sine error / (2 pi)
      cts_pt  CtS diagonal rounding, relative error * sigma_I (real part, x units)
      stc_pt  StC diagonal rounding, directly in output slots (relative to rms 1/sqrt(3))
    """
    log_ratio = chain["q0"] - 42
    s = 2.0**log_scale
    noise_in = math.hypot(slot_noise(h_noise), digit_noise(chain["log_qd_over_p"]))
    err, amp = sin_error(K, r, d)
    f = MAX_FACTOR * math.sqrt(SLOTS) * 2.0**log_ratio
    e = {
        "input": f * K * math.sqrt(chain["cts_levels"]) * noise_in / s,
        "internal": f * math.sqrt(levels) * slot_noise(h_noise) / s * amp / (2 * math.pi),
        "approx": f * err / (2 * math.pi),
        "cts_pt": f * 2.0 ** cts_pt_rel(chain["cts_bits"], chain["cts_levels"], chain["q0"], h_mr, K)
        * sigma_i(h_mr),
        "stc_pt": MAX_FACTOR * 2.0 ** stc_pt_rel(chain["stc_bits"], chain["stc_levels"], log_ratio)
        / math.sqrt(3),
    }
    total = -math.log2(math.sqrt(sum(v * v for v in e.values())))
    return total, {k: -math.log2(v) for k, v in e.items()}


def sine_vs_mod_bits(log_ratio):
    """x - sin(2 pi x)/(2 pi) ~ (2 pi)^2 x^3 / 6 for the message part x = Delta c / q0.

    Inputs uniform in [-1, 1] in all 2^15 slots: slot rms 1/sqrt(3), coefficient rms
    1/sqrt(3 N); E[x^6] = 15 sigma^6 for Gaussian coefficients.
    """
    sx = 2.0**-log_ratio / math.sqrt(3 * N)
    err_x = (2 * math.pi) ** 2 / 6 * math.sqrt(15) * sx**3
    return -math.log2(MAX_FACTOR * math.sqrt(SLOTS) * 2.0**log_ratio * err_x)


def make_chain(q0, compute, stc, evalmod, cts, p, dnum):
    """q from level 0 up: q0, compute x 42, stc = (levels, bits), evalmod, cts likewise; p bits."""
    q = [q0] + [42] * compute + [stc[1]] * stc[0] + [evalmod[1]] * evalmod[0] + [cts[1]] * cts[0]
    assert len(q) % dnum == 0, "dnum must divide the prime count"
    alpha = len(q) // dnum
    max_digit = max(sum(q[i : i + alpha]) for i in range(0, len(q), alpha))
    log_p = sum(p)
    return dict(
        q=q, p=p, dnum=dnum, alpha=alpha, max_digit=max_digit, log_p=log_p,
        log_pq=sum(q) + log_p, levels_after=len(q) - 1 - (stc[0] + evalmod[0] + cts[0]),
        key_limbs=dnum * (len(q) + len(p)), q0=q0, cts_bits=cts[1], cts_levels=cts[0],
        stc_bits=stc[1], stc_levels=stc[0], log_qd_over_p=min(0.0, max_digit - log_p),
    )


CHAINS = [
    ("R-M3-5 (CtS 42, superseded)", (54, 16, (3, 42), (8, 58), (4, 42), [60] * 4, 8)),
    ("A: compute 16, StC 3x36, CtS 4x52", (54, 16, (3, 36), (8, 58), (4, 52), [60] * 4, 8)),
    ("B (chosen): compute 15, StC 4x34, CtS 4x56", (54, 15, (4, 34), (8, 58), (4, 56), [60] * 4, 8)),
    ("C: compute 14, StC 3x36, CtS 4x56, dnum 10", (54, 14, (3, 36), (8, 58), (4, 56), [46] * 4, 10)),
    ("D: compute 14, StC 3x36, CtS 4x56, dnum 6", (54, 14, (3, 36), (8, 58), (4, 56), [60] * 5, 6)),
    ("E: compute 16, StC 3x36, CtS 4x56, dnum 16", (54, 16, (3, 36), (8, 58), (4, 56), [42] * 3, 16)),
    ("F: compute 16, StC 3x36, CtS 4x56, dnum 8 (log PQ > 1750)", (54, 16, (3, 36), (8, 58), (4, 56), [60] * 4, 8)),
]
CHOSEN = make_chain(*CHAINS[2][1])


def main():
    print("## 1. log2 P[some |I_j| >= K] over N = 2^16 coefficients (per bootstrap)")
    for h, label, ks in [
        (32768, "dense (hekit default, P(+-1) = 1/4 each)", [25, 300, 350, 400, 420, 450]),
        (192, "sparse h = 192", [12, 16, 25, 28, 30, 32, 33]),
        (32, "sparse h = 32 (SSE ephemeral key)", [10, 12, 13, 14, 16]),
    ]:
        print(f"{label}: sigma_I = {sigma_i(h):.2f}")
        for K in ks:
            exact = "" if h > 1000 else f", exact {log2_fail_exact(K, h):7.2f}"
            print(f"  K = {K:3d}: Gaussian {log2_fail_gauss(K, h):7.2f}{exact}")

    print("\n## 2. log2 max sin error with 8 EvalMod levels (degree 2^(8-r) - 1); float64 floor ~2^-45")
    for K in [12, 13, 16, 25, 28, 30, 32, 34]:
        cells = []
        for r in range(4):
            e, amp = sin_error(K, r, 2 ** (8 - r) - 1)
            cells.append(f"r={r}: {math.log2(e):6.1f} (amp {amp:4.1f})")
        print(f"  K = {K:3d} | " + " | ".join(cells))

    print("\n## 3. dense key: log2 max sin error vs EvalMod levels L (r = 0, degree 2^L - 1)")
    for K in [400, 420]:
        cells = [f"L={L}: {math.log2(sin_error(K, 0, 2**L - 1)[0]):6.1f}" for L in range(9, 13)]
        print(f"  K = {K} | " + " | ".join(cells))

    print("\n## 4. candidate chains, design dense + SSE h'=32, K=13, r=1, d=127")
    for name, args in CHAINS:
        c = make_chain(*args)
        bits, t = predict(c)
        print(f"  {name}")
        print(f"     primes {len(c['q'])}, dnum {c['dnum']} (alpha {c['alpha']}), max digit {c['max_digit']}, "
              f"log P {c['log_p']} (+{c['log_p'] - c['max_digit']}), log PQ {c['log_pq']}, "
              f"levels after boot {c['levels_after']}, key limbs {c['key_limbs']}")
        print(f"     predicted {bits:5.1f}  [" + ", ".join(f"{k} {v:5.1f}" for k, v in t.items()) + "]")

    print("\n## 5. alternatives on the chosen chain (q0 = 42 + log2(q0/Delta))")
    designs = [
        ("dense + SSE h'=32, K=13, r=0, d=255", 13, 0, 255, 32),
        ("dense + SSE h'=32, K=13, r=0, d=127", 13, 0, 127, 32),
        ("dense + SSE h'=32, K=13, r=1, d=127", 13, 1, 127, 32),
        ("dense + SSE h'=32, K=13, r=2, d=63", 13, 2, 63, 32),
        ("sparse h=192, K=32, r=0, d=255", 32, 0, 255, 192),
    ]
    for log_ratio in [16, 12, 10, 8]:
        c = dict(CHOSEN, q0=42 + log_ratio)
        print(f"q0/Delta = 2^{log_ratio} (sine-vs-mod term alone: {sine_vs_mod_bits(log_ratio):.1f} bits)")
        for name, K, r, d, h_mr in designs:
            h_noise = 192 if h_mr == 192 else 32768
            bits, t = predict(c, K=K, r=r, d=d, h_noise=h_noise, h_mr=h_mr)
            print(f"  {name:38s} {bits:5.1f}  [" + ", ".join(f"{k} {v:5.1f}" for k, v in t.items()) + "]")
    sig = 420 * math.sqrt(CHOSEN["cts_levels"]) * slot_noise(32768) / 2.0**58
    for log_ratio in [16, 8]:
        b = -math.log2(MAX_FACTOR * math.sqrt(SLOTS) * 2.0**log_ratio * sig)
        print(f"dense, no SSE, K = 420, q0/Delta = 2^{log_ratio}: input-noise term alone {b:.1f} bits")

    print("\n## 6. chosen design (rulings R-M3-5, R-M3-7)")
    bits, t = predict(CHOSEN)
    print(f"  chain B, P = 4 x 60: {bits:.1f}  [" + ", ".join(f"{k} {v:.1f}" for k, v in t.items()) + "]")
    bits, _ = predict(dict(CHOSEN, log_qd_over_p=0.0))
    print(f"  same with P = 4 x 58 (digit term at worst case): {bits:.1f}")
    bits, _ = predict(dict(CHOSEN, q0=52))
    print(f"  fallback q0 = 52 bits (q0/Delta = 2^10, log PQ {CHOSEN['log_pq'] - 2}): {bits:.1f}")
    for r, d in [(0, 127), (1, 63), (2, 31)]:
        bits, _ = predict(CHOSEN, r=r, d=d)
        err, _ = sin_error(13, r, d)
        print(f"  textbook BSGS (ceil(log2(d+1)) + 1 + r <= 8): r = {r}, d = {d}: "
              f"sine error 2^{math.log2(err):.1f}, predicted {bits:.1f}")
    print(f"  uncorrected sine-vs-mod term at 2^12: {sine_vs_mod_bits(12):.1f} bits "
          f"(|m| <= 8: {sine_vs_mod_bits(12) - 9:.1f})")


if __name__ == "__main__":
    main()
