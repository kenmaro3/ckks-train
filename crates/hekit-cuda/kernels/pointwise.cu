// Element-wise kernels over the RNS limbs of device polynomials, for NVRTC (no #include):
// add, sub, neg, pointwise product, product by a per-limb Shoup constant, and the Galois
// automorphism in NTT form (index permutation) and coefficient form (signed permutation).
//
// Every kernel produces exactly the residues of the CPU code in crates/hekit-core/src/poly.rs
// (`add_assign`, `sub_assign`, `neg`, `mul_assign`, `mul_scalar_assign`, `automorphism`):
// inputs are reduced residues (< q) and every output is fully reduced to [0, q). The CPU and
// GPU agree because each operation's result is the unique representative in [0, q) (the
// Barrett product below is the same algorithm as `Modulus::mul`, with the same off-by-at-most-
// one quotient estimate). Fully reduced outputs also satisfy the inverse NTT's input bound
// (< 2q, see kernels/ntt.cu).
//
// Layout and launch: limb l (blockIdx.y) is contiguous at a + l * N; its prime index is
// p = limb_prime[l] and the prime's constants are at consts + p * NCONST (the NTT constant
// table, see kernels/ntt.cu): { q, n_inv, n_inv_shoup, w_n_inv, w_n_inv_shoup, barrett_lo,
// barrett_hi } with barrett = floor(2^128 / q). grid = (ceil(N / blockDim.x), num_limbs),
// one thread per coefficient (`ElemLaunch` in src/plan.rs). `out` may alias `a` or `b` (each
// thread reads and writes only its own index), except in the automorphisms.
//
// The Rust module `emu_arith` (src/emu_arith.rs, tests only) replays this file thread by thread;
// tags [P*] mark corresponding lines.

typedef unsigned long long u64;
typedef unsigned int u32;

#define NCONST 7

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

// [P4] (a * w) mod q, Shoup, for any u64 a, w < q, ws = floor(w * 2^64 / q)
// (`Modulus::mul_shoup`).
__device__ __forceinline__ u64 mul_shoup(u64 a, u64 w, u64 ws, u64 q) {
    const u64 hi = __umul64hi(a, ws);
    u64 r = a * w - hi * q;
    if (r >= q) r -= q;
    return r;
}

// out = a + b.
extern "C" __global__ void pw_add(u64 *out, const u64 *a, const u64 *b,
                                  const u64 *__restrict__ consts,
                                  const u32 *__restrict__ limb_prime, u32 log_n) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;   // [P5] coefficient
    if (c >= (1u << log_n)) return;
    const size_t i = ((size_t)blockIdx.y << log_n) + c;     // [P6] limb blockIdx.y
    const u64 q = consts[(size_t)limb_prime[blockIdx.y] * NCONST];
    out[i] = add_mod(a[i], b[i], q);                        // [P7]
}

// out = a - b.
extern "C" __global__ void pw_sub(u64 *out, const u64 *a, const u64 *b,
                                  const u64 *__restrict__ consts,
                                  const u32 *__restrict__ limb_prime, u32 log_n) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (1u << log_n)) return;
    const size_t i = ((size_t)blockIdx.y << log_n) + c;
    const u64 q = consts[(size_t)limb_prime[blockIdx.y] * NCONST];
    out[i] = sub_mod(a[i], b[i], q);                        // [P8]
}

// out = a * b (pointwise; the ring product when both are in NTT form).
extern "C" __global__ void pw_mul(u64 *out, const u64 *a, const u64 *b,
                                  const u64 *__restrict__ consts,
                                  const u32 *__restrict__ limb_prime, u32 log_n) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (1u << log_n)) return;
    const size_t i = ((size_t)blockIdx.y << log_n) + c;
    const u64 *cp = consts + (size_t)limb_prime[blockIdx.y] * NCONST;
    out[i] = mul_mod(a[i], b[i], cp[0], cp[5], cp[6]);     // [P9]
}

// out = -a (`Modulus::sub(0, a)`).
extern "C" __global__ void pw_neg(u64 *out, const u64 *a, const u64 *__restrict__ consts,
                                  const u32 *__restrict__ limb_prime, u32 log_n) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (1u << log_n)) return;
    const size_t i = ((size_t)blockIdx.y << log_n) + c;
    const u64 q = consts[(size_t)limb_prime[blockIdx.y] * NCONST];
    out[i] = sub_mod(0, a[i], q);                           // [P10]
}

// out = a * w_l with a per-limb Shoup constant: w[2 l] = w_l < q_l, w[2 l + 1] its companion
// (`mul_scalar_assign`: w_l = scalar mod q_l).
extern "C" __global__ void pw_mul_shoup(u64 *out, const u64 *a, const u64 *__restrict__ w,
                                        const u64 *__restrict__ consts,
                                        const u32 *__restrict__ limb_prime, u32 log_n) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (1u << log_n)) return;
    const size_t i = ((size_t)blockIdx.y << log_n) + c;
    const u64 q = consts[(size_t)limb_prime[blockIdx.y] * NCONST];
    out[i] = mul_shoup(a[i], w[2 * blockIdx.y], w[2 * blockIdx.y + 1], q);   // [P11]
}

// NTT-form automorphism: out[i] = a[perm[i]] in every limb, with SEAL's table
// (`RnsBasis::galois_permutation`): perm[i] = bitrev_{logN}(((g * bitrev_{logN+1}(N + i)) >> 1)
// & (N - 1)). g odd, g < 2N <= 2^32 (so g * (2^32 - 1) fits in 64 bits). The permutation does
// not depend on the prime. `out` must not alias `a`.
extern "C" __global__ void automorphism_ntt(u64 *__restrict__ out, const u64 *__restrict__ a,
                                            u32 log_n, u64 g) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;   // [P12]
    const u32 n = 1u << log_n;
    if (c >= n) return;
    const size_t base = (size_t)blockIdx.y << log_n;
    const u32 reversed = __brev(n + c) >> (31 - log_n);    // [P13] bitrev_{logN+1}(N + c)
    const u32 raw = (u32)(((g * (u64)reversed) >> 1) & (u64)(n - 1));
    const u32 src = __brev(raw) >> (32 - log_n);           // [P14] bitrev_{logN}(raw)
    out[base + c] = a[base + src];
}

// Coefficient-form automorphism: X^c -> X^(c g mod 2N), negated when c g mod 2N >= N. A
// scatter: every output index is written by exactly one thread (c -> c g mod N is a bijection
// for odd g). `out` must not alias `a`.
extern "C" __global__ void automorphism_coeff(u64 *__restrict__ out, const u64 *__restrict__ a,
                                              const u64 *__restrict__ consts,
                                              const u32 *__restrict__ limb_prime, u32 log_n,
                                              u64 g) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;   // [P15]
    const u32 n = 1u << log_n;
    if (c >= n) return;
    const size_t base = (size_t)blockIdx.y << log_n;
    const u64 q = consts[(size_t)limb_prime[blockIdx.y] * NCONST];
    const u64 e = ((u64)c * g) & ((2ull << log_n) - 1);    // [P16] c g mod 2N
    const u64 v = a[base + c];
    if (e < n) {
        out[base + e] = v;                                  // [P17]
    } else {
        out[base + (e - n)] = sub_mod(0, v, q);
    }
}
