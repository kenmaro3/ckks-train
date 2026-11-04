//! Scalar CPU negacyclic NTT over `Z_q[X]/(X^N+1)` (Harvey butterflies, Shoup
//! precomputation, lazy reduction). Spike code, but the API is kept small and stable so the
//! CUDA spike can depend on this crate for bit-exact comparison.
//!
//! # Conventions (normative)
//!
//! * `N` is a power of two, `q` is prime with `q ≡ 1 (mod 2N)` and `q < 2^60`.
//! * `psi` is a primitive `2N`-th root of unity mod `q` (`psi^N = -1`), from
//!   [`primitive_2n_root`] (the smallest `g >= 2` whose candidate root has order `2N`).
//! * **Twiddle table** (`roots`, length `N`): `roots[i] = psi^bitrev(i) mod q`, where `bitrev`
//!   reverses the `log2 N` low bits of `i`. `roots[0] = 1`.
//! * **Inverse twiddle table** (`inv_roots`, length `N`): `inv_roots[i] = psi^(-bitrev(i)) mod q`.
//! * **Shoup table**: `w_shoup[i] = floor(w[i] * 2^64 / q)`.
//! * **Forward NTT** ([`ntt_forward`]): Cooley–Tukey, input in natural coefficient order,
//!   output in **bit-reversed order**: `out[bitrev(k)] = a(psi^(2k+1))`. Output fully reduced in `[0, q)`.
//!   The butterfly of group `i` at stage `m` (`m = 1, 2, 4, ..., N/2`, `t = N/(2m)`) uses
//!   `roots[m + i]` on the index pairs `(j, j + t)`, `j in [2*i*t, 2*i*t + t)`.
//! * **Inverse NTT** ([`ntt_inverse`]): Gentleman–Sande, bit-reversed input, natural-order output,
//!   includes the `N^-1` scaling, output in `[0, q)`. The butterfly of group `i` at stage
//!   `h = N/2, N/4, ..., 1` (`t = N/(2h)`) uses `inv_roots[h + i]` on `(j, j + t)`,
//!   `j in [2*i*t, 2*i*t + t)`.
//!
//! Pointwise products of two forward outputs are therefore the forward output of the
//! negacyclic product.
#![deny(missing_docs)]

/// `a * b mod q` via `u128`.
pub fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

/// `a^e mod q` by square-and-multiply.
pub fn pow_mod(mut a: u64, mut e: u64, q: u64) -> u64 {
    let mut r = 1u64;
    a %= q;
    while e > 0 {
        if e & 1 == 1 {
            r = mul_mod(r, a, q);
        }
        a = mul_mod(a, a, q);
        e >>= 1;
    }
    r
}

/// Deterministic Miller–Rabin for all `u64` (bases 2,3,5,7,11,13,17,19,23,29,31,37).
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'bases: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// Largest prime `q < 2^bits` with `q ≡ 1 (mod 2N)`. Requires `n` a power of two and `bits <= 60`.
pub fn gen_ntt_prime(n: usize, bits: u32) -> u64 {
    assert!(n.is_power_of_two() && bits <= 60);
    let step = 2 * n as u64;
    let mut q = ((1u64 << bits) - 1) / step * step + 1;
    while !is_prime(q) {
        q -= step;
    }
    q
}

/// A primitive `2N`-th root of unity `psi` mod `q` (`psi^N = q - 1`), deterministic.
pub fn primitive_2n_root(n: usize, q: u64) -> u64 {
    let e = (q - 1) / (2 * n as u64);
    (2u64..)
        .map(|g| pow_mod(g, e, q))
        .find(|&psi| pow_mod(psi, n as u64, q) == q - 1)
        .expect("no primitive root found")
}

fn bitrev(i: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        i.reverse_bits() >> (usize::BITS - bits)
    }
}

fn power_table(n: usize, q: u64, base: u64) -> Vec<u64> {
    let bits = n.trailing_zeros();
    let mut pows = Vec::with_capacity(n);
    let mut p = 1u64;
    for _ in 0..n {
        pows.push(p);
        p = mul_mod(p, base, q);
    }
    (0..n).map(|i| pows[bitrev(i, bits)]).collect()
}

/// Forward twiddle table `roots[i] = psi^bitrev(i) mod q`, length `n`.
pub fn build_roots(n: usize, q: u64, psi: u64) -> Vec<u64> {
    power_table(n, q, psi)
}

/// Inverse twiddle table `inv_roots[i] = psi^(-bitrev(i)) mod q`, length `n`.
pub fn build_inv_roots(n: usize, q: u64, psi: u64) -> Vec<u64> {
    power_table(n, q, pow_mod(psi, q - 2, q))
}

/// Shoup companions `floor(w[i] * 2^64 / q)` of a table of values `w[i] < q`.
pub fn shoup_table(w: &[u64], q: u64) -> Vec<u64> {
    w.iter()
        .map(|&x| (((x as u128) << 64) / q as u128) as u64)
        .collect()
}

/// Lazy Shoup multiplication: returns a value congruent to `y * w (mod q)` in `[0, 2q)`,
/// for any `y < 2^64`, `w < q`, `w_shoup = floor(w * 2^64 / q)`.
#[inline(always)]
fn mul_shoup_lazy(y: u64, w: u64, w_shoup: u64, q: u64) -> u64 {
    let hi = ((y as u128 * w_shoup as u128) >> 64) as u64;
    y.wrapping_mul(w).wrapping_sub(hi.wrapping_mul(q))
}

/// In-place forward negacyclic NTT (Harvey, lazy reduction). See the crate docs for the
/// exact table/ordering convention. Input coefficients must be in `[0, q)`;
/// output is in bit-reversed order, fully reduced to `[0, q)`.
///
/// `a.len()` must be a power of two equal to `roots.len()`, and `q < 2^60`.
///
/// Lazy-reduction invariant: every coefficient entering a stage is in `[0, 4q)` (`4q < 2^62`).
/// A butterfly first brings `x` into `[0, 2q)` (conditional `-2q`), computes
/// `t = mulshoup_lazy(y, w) ∈ [0, 2q)` (valid for any `y < 2^64`), and outputs
/// `x + t ∈ [0, 4q)` and `x - t + 2q ∈ (0, 4q)`. A final pass reduces `[0, 4q)` to `[0, q)`.
pub fn ntt_forward(a: &mut [u64], q: u64, roots: &[u64], roots_shoup: &[u64]) {
    let n = a.len();
    assert!(n.is_power_of_two() && roots.len() == n && roots_shoup.len() == n && q < 1 << 60);
    let two_q = 2 * q;
    let mut t = n;
    let mut m = 1;
    while m < n {
        t >>= 1;
        for i in 0..m {
            let (w, ws) = (roots[m + i], roots_shoup[m + i]);
            let (lo, hi) = a[2 * i * t..2 * (i + 1) * t].split_at_mut(t);
            for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                let xr = if *x >= two_q { *x - two_q } else { *x };
                let tt = mul_shoup_lazy(*y, w, ws, q);
                *x = xr + tt;
                *y = xr + two_q - tt;
            }
        }
        m <<= 1;
    }
    for v in a.iter_mut() {
        let mut r = *v;
        if r >= two_q {
            r -= two_q;
        }
        if r >= q {
            r -= q;
        }
        *v = r;
    }
}

/// In-place inverse negacyclic NTT (Gentleman–Sande, lazy reduction), including the `N^-1`
/// scaling. Input in bit-reversed order, coefficients in `[0, q)` (any value `< 2q` is also
/// accepted); output in natural order, fully reduced to `[0, q)`.
/// `n_inv = N^-1 mod q` and `n_inv_shoup = floor(n_inv * 2^64 / q)`.
///
/// Lazy-reduction invariant: every coefficient entering a stage is in `[0, 2q)`. A butterfly
/// outputs `x + y` reduced by a conditional `-2q` into `[0, 2q)`, and
/// `mulshoup_lazy(x - y + 2q, w) ∈ [0, 2q)` (`x - y + 2q ∈ (0, 4q)` fits in `u64`).
/// The last stage fuses the `N^-1` multiplication and a final reduction to `[0, q)`.
pub fn ntt_inverse(
    a: &mut [u64],
    q: u64,
    inv_roots: &[u64],
    inv_roots_shoup: &[u64],
    n_inv: u64,
    n_inv_shoup: u64,
) {
    let n = a.len();
    assert!(
        n >= 2
            && n.is_power_of_two()
            && inv_roots.len() == n
            && inv_roots_shoup.len() == n
            && q < 1 << 60
    );
    let two_q = 2 * q;
    let mut t = 1;
    let mut h = n >> 1;
    while h > 1 {
        for i in 0..h {
            let (w, ws) = (inv_roots[h + i], inv_roots_shoup[h + i]);
            let (lo, hi) = a[2 * i * t..2 * (i + 1) * t].split_at_mut(t);
            for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                let s = *x + *y;
                let d = *x + two_q - *y;
                *x = if s >= two_q { s - two_q } else { s };
                *y = mul_shoup_lazy(d, w, ws, q);
            }
        }
        t <<= 1;
        h >>= 1;
    }
    // Last stage (h = 1, t = n/2, single twiddle inv_roots[1]); fuse the N^-1 scaling.
    // x' = (x + y) * n_inv ; y' = (x - y) * (w * n_inv)
    let w = inv_roots[1];
    let wn = mul_mod(w, n_inv, q);
    let wn_shoup = shoup_table(&[wn], q)[0];
    let (lo, hi) = a.split_at_mut(n / 2);
    for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
        let s = *x + *y; // < 4q
        let d = *x + two_q - *y; // < 4q
        let mut r0 = mul_shoup_lazy(s, n_inv, n_inv_shoup, q);
        let mut r1 = mul_shoup_lazy(d, wn, wn_shoup, q);
        if r0 >= q {
            r0 -= q;
        }
        if r1 >= q {
            r1 -= q;
        }
        *x = r0;
        *y = r1;
    }
}
