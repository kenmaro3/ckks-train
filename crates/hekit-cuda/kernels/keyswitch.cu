// The key-switching inner product over the RNS limbs of `Q_l * P`, for NVRTC (no #include).
//
// `ks_inner_product` computes, for the lifted digits d_0 .. d_{D-1} of a level-l polynomial
// (each a `Q_l * P` polynomial, limbs [0..=l, L+1..L+k], NTT form) and the parts (b_j, a_j)
// of a key-switching key, in one launch over every limb of `Q_l * P` and both accumulators:
//
//   out_b = sum_j d_j * b_j,   out_a = sum_j d_j * a_j      (pointwise, mod each prime)
//
// which is the inner product of `hekit_ckks::keyswitch::apply_key` before its two
// `mod_down`s. The CPU sums the exact u128 products and reduces once; here every product is
// reduced (Barrett, the `pw_mul` product) and added mod q. Both give the unique residue of
// the same sum in [0, q), so the outputs are bit-identical (and fully reduced, as the inverse
// NTT of the following `mod_down` needs).
//
// Layout and launch:
//   * `digit_ptrs[j]` is the device address of digit j (D addresses; the digits are separate
//     allocations). Digit j's limb y (prime index limb_prime[y]) is at d_j + y * N.
//   * The key is one allocation of `parts` parts; part j holds b_j then a_j, each `key_limbs`
//     limbs: the key's Q limbs 0..=M (M the key's top level, M >= l) then its k P limbs.
//     Output limb y maps to key limb y for a Q limb (y < q_limbs = l + 1) and to
//     y + key_skip (key_skip = M - l) for a P limb, so the key's limbs above l are skipped in
//     place: the key's level-l limbs are not contiguous and are never gathered.
//       b_j limb t: key + ((2 j)     * key_limbs + t) * N
//       a_j limb t: key + ((2 j + 1) * key_limbs + t) * N
//   * grid = (ceil(N / blockDim.x), l + 1 + k), one thread per coefficient (`ElemLaunch` in
//     src/plan.rs); limb y = blockIdx.y with prime constants at consts + limb_prime[y] * NCONST
//     (kernels/ntt.cu). n_digits <= parts is checked by the host.
//
// The Rust module `emu_ks` (src/emu_ks.rs, tests only) replays this file thread by thread;
// tags [K*] mark corresponding lines. The modular helpers between the two `----` markers are a
// verbatim copy of kernels/pointwise.cu (tags [P*] kept; a unit test checks the copy is
// identical, so the replay of pointwise.cu covers them).

typedef unsigned long long u64;
typedef unsigned int u32;

#define NCONST 7

// ---- begin verbatim copy of kernels/pointwise.cu
// [P1] (a + b) mod q for a, b < q (`Modulus::add`).
__device__ __forceinline__ u64 add_mod(u64 a, u64 b, u64 q) {
    const u64 s = a + b;
    return (s >= q) ? s - q : s;
}

// [P2] (a - b) mod q for a, b < q (`Modulus::sub`).
__device__ __forceinline__ u64 sub_mod(u64 a, u64 b, u64 q) {
    return (a >= b) ? a - b : a + q - b;
}

// [P3a] High 128 bits of the 256-bit product (x1, x0) * (r1, r0), reduced mod 2^64
// (`mulhi_u128`): exact whenever the true value is < 2^64, which holds for the Barrett
// quotient below.
__device__ __forceinline__ u64 barrett_est(u64 x0, u64 x1, u64 r0, u64 r1) {
    const u64 t = __umul64hi(x0, r0);          // high word of x0 * r0
    const u64 m1 = x0 * r1;                    // low word of x0 * r1
    const u64 m2 = x1 * r0;                    // low word of x1 * r0
    u64 s = t + m1;
    u64 carry = (s < t) ? 1ull : 0ull;
    s += m2;
    carry += (s < m2) ? 1ull : 0ull;
    return x1 * r1 + __umul64hi(x0, r1) + __umul64hi(x1, r0) + carry;
}

// [P3] (a * b) mod q for a, b < q < 2^60: Barrett with r = floor(2^128 / q) = (r1, r0)
// (`Modulus::mul` / `reduce_wide`). x = a * b = (x1, x0) < 2^120; est = floor(x r / 2^128)
// <= x / q < 2^60, and x - est * q < 2q: one conditional subtraction.
__device__ __forceinline__ u64 mul_mod(u64 a, u64 b, u64 q, u64 r0, u64 r1) {
    const u64 x0 = a * b;
    const u64 x1 = __umul64hi(a, b);
    const u64 est = barrett_est(x0, x1, r0, r1);
    u64 r = x0 - est * q;
    if (r >= q) r -= q;
    return r;
}
// ---- end verbatim copy of kernels/pointwise.cu

// out_b = sum_j d_j * b_j and out_a = sum_j d_j * a_j over the limbs of Q_l * P (see top).
extern "C" __global__ void ks_inner_product(u64 *__restrict__ out_b, u64 *__restrict__ out_a,
                                            const u64 *__restrict__ digit_ptrs,
                                            const u64 *__restrict__ key,
                                            const u64 *__restrict__ consts,
                                            const u32 *__restrict__ limb_prime, u32 log_n,
                                            u32 n_digits, u32 q_limbs, u32 key_limbs,
                                            u32 key_skip) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;   // [K1] coefficient
    if (c >= (1u << log_n)) return;
    const u32 y = blockIdx.y;                               // [K2] output limb
    const size_t i = ((size_t)y << log_n) + c;
    const u64 *cp = consts + (size_t)limb_prime[y] * NCONST;
    const u64 q = cp[0], r0 = cp[5], r1 = cp[6];
    const size_t t = (y < q_limbs) ? (size_t)y : (size_t)y + key_skip;   // [K3] key limb
    u64 acc_b = 0, acc_a = 0;
    for (u32 j = 0; j < n_digits; ++j) {
        const u64 x = ((const u64 *)digit_ptrs[j])[i];                    // [K4]
        const u64 kb = key[(((size_t)(2 * j) * key_limbs + t) << log_n) + c];
        const u64 ka = key[(((size_t)(2 * j + 1) * key_limbs + t) << log_n) + c];
        acc_b = add_mod(acc_b, mul_mod(x, kb, q, r0, r1), q);             // [K5]
        acc_a = add_mod(acc_a, mul_mod(x, ka, q, r0, r1), q);
    }
    out_b[i] = acc_b;                                        // [K6]
    out_a[i] = acc_a;
}
