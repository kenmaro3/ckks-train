#!/usr/bin/env python3
"""Numbers behind docs/bootstrap.md (Task 35): K, EvalMod depth and the precision model.

Usage: python3 tools/bootstrap/evalmod_design.py      (numpy only, about 10 s)

Sections:
  1. Failure probability of the mod-raise overflow |I_j| >= K over all N coefficients
     (Gaussian for the dense key, exact Irwin-Hall for sparse keys).
  2. Sine error reachable with 8 EvalMod levels: Chebyshev interpolant of
     cos(2 pi (x - 1/4) / 2^r) on [-K, K] of degree 2^(8 - r) - 1, then r double angles,
     measured at the points x = I + delta (|I| <= K - 1, |delta| <= 2^-20) that occur.
  3. Smallest EvalMod depth for the dense key (K = 400, 420).
  4. Predicted bootstrap precision (bits, max over 2^15 slots) for the candidate designs,
     including the chosen design of controller ruling R-M3-5 (dense key + SSE h' = 32,
     q0 = 54 bits so q0/Delta = 2^12, P = 4 x 60 bits).
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
    n16_boot has two all-58-bit digits (4 x 58 = 232 bits): Q_d/P ~ 2^0 with P = 4 x 58 and
    2^-8 with P = 4 x 60. Counted at the working scale of the CtS output (pessimistic)."""
    return math.sqrt(N / 2) * math.sqrt(ndigits) * 2.0**log_qd_over_p * 3.2 * math.sqrt(N / 12)


def predict(K, r, d, h_noise, log_ratio, log_scale=58, levels=8, log_qd_over_p=None):
    """Predicted precision bits and the three terms (input noise, internal noise, approximation).
    With `log_qd_over_p`, the worst-case digit term is added to the CtS output noise."""
    s = 2.0**log_scale
    noise_in = slot_noise(h_noise)
    if log_qd_over_p is not None:
        noise_in = math.hypot(noise_in, digit_noise(log_qd_over_p))
    sig_in = noise_in / s  # y = x / K units, CtS output
    err, amp = sin_error(K, r, d)
    sig_int = math.sqrt(levels) * slot_noise(h_noise) / s * amp  # sin units
    to_bits = lambda sig_x: -math.log2(MAX_FACTOR * math.sqrt(SLOTS) * 2.0**log_ratio * sig_x)
    terms = (K * sig_in, sig_int / (2 * math.pi), err / (2 * math.pi))  # x units
    total = math.sqrt(sum(t * t for t in terms))
    return to_bits(total), [to_bits(t) for t in terms]


def sine_vs_mod_bits(log_ratio):
    """x - sin(2 pi x)/(2 pi) ~ (2 pi)^2 x^3 / 6 for the message part x = Delta c / q0.

    Inputs uniform in [-1, 1] in all 2^15 slots: slot rms 1/sqrt(3), coefficient rms
    1/sqrt(3 N); E[x^6] = 15 sigma^6 for Gaussian coefficients.
    """
    sx = 2.0**-log_ratio / math.sqrt(3 * N)
    err_x = (2 * math.pi) ** 2 / 6 * math.sqrt(15) * sx**3
    return -math.log2(MAX_FACTOR * math.sqrt(SLOTS) * 2.0**log_ratio * err_x)


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

    print("\n## 4. predicted precision (bits), EvalMod scale 2^58")
    designs = [
        ("dense, plan K=25 (Pfail ~ 1)", 25, 0, 255, 32768),
        ("dense + SSE h'=32, K=13, r=0, d=255", 13, 0, 255, 32768),
        ("dense + SSE h'=32, K=13, r=1, d=127", 13, 1, 127, 32768),
        ("dense + SSE h'=32, K=13, r=2, d=63", 13, 2, 63, 32768),
        ("sparse h=192, K=32, r=0, d=255", 32, 0, 255, 192),
        ("sparse h=192 + SSE h'=32, K=13, r=1, d=127", 13, 1, 127, 192),
    ]
    for log_ratio in [16, 12, 10, 8]:
        print(f"q0/Delta = 2^{log_ratio} (sine-vs-mod term alone: {sine_vs_mod_bits(log_ratio):.1f} bits)")
        for name, K, r, d, h in designs:
            bits, (t_in, t_int, t_apx) = predict(K, r, d, h, log_ratio)
            print(f"  {name:44s} {bits:5.1f}  [input {t_in:5.1f}, internal {t_int:5.1f}, approx {t_apx:5.1f}]")
    print("\n## 5. chosen design (ruling R-M3-5): dense + SSE h'=32, K=13, r=1, d=127, q0/Delta = 2^12")
    for label, lq in [("P = 4 x 60 (chosen), digit term 2^-8", -8.0), ("P = 4 x 58, digit term 2^0", 0.0)]:
        bits, terms = predict(13, 1, 127, 32768, 12, log_qd_over_p=lq)
        print(f"  {label:38s} {bits:5.1f}  [input {terms[0]:5.1f}]  digit noise 2^{math.log2(digit_noise(lq)):.1f}")
    bits, _ = predict(13, 1, 127, 32768, 10, log_qd_over_p=-8.0)
    print(f"  fallback q0 = 52 bits (q0/Delta = 2^10), P = 4 x 60: {bits:.1f}")
    print(f"  uncorrected sine-vs-mod term at 2^12: {sine_vs_mod_bits(12):.1f} bits "
          f"(|m| <= 8: {sine_vs_mod_bits(12) - 9:.1f})")
    print()
    bits, _ = predict(13, 1, 127, 32768, 16, log_scale=60)
    print(f"dense + SSE, K=13, r=1, q0/Delta = 2^16, EvalMod primes 60-bit: {bits:.1f}")
    for log_ratio in [16, 8]:
        sig = 420 * slot_noise(32768) / 2.0**58
        b = -math.log2(MAX_FACTOR * math.sqrt(SLOTS) * 2.0**log_ratio * sig)
        print(f"dense, no SSE, K = 420, q0/Delta = 2^{log_ratio}: input-noise term alone {b:.1f} bits")


if __name__ == "__main__":
    main()
