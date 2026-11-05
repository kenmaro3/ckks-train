// RNS base-conversion kernels for rescaling, mod_down and mod_up, for NVRTC (no #include).
//
// One kernel, `basis_convert`, evaluates the "centered" conversion shared by the three CPU
// algorithms (crates/hekit-core/src/baseconv.rs and lift_digit_exact in
// crates/hekit-ckks/src/keyswitch.rs). For source primes q_0..q_{k-1} (product Q) and
// coefficient-form residues x_j (< q_j), every target prime r gets
//
//   s_j = (x_j + (q_j >> 1)) mod q_j                  [x + h]_Q, h = (Q - 1) / 2
//   y_j = s_j * qhat_inv_j mod q_j                    qhat_inv_j = (Q / q_j)^-1 mod q_j
//   e   = floor(sum_j double(y_j) * (1.0 / q_j))      only when k > 1 (`convert_exact`)
//   out = sum_j y_j * qhat_j mod r - e * (Q mod r) - (h mod r)    (all mod r)
//
// i.e. `BaseConverter::convert_exact([x + h]_Q) - h`. With k = 1 this is the fused one-prime
// path (qhat_inv = qhat = 1, no f64), which is bit-exact SEAL rounding. The modular steps
// give the unique residue whatever the evaluation order; the f64 estimate e is the one
// non-modular step and is evaluated with the CPU's exact operation sequence: round-to-
// nearest u64 -> f64 conversion (`as f64`), round-to-nearest product and sum in source order,
// starting from 0 (`Iterator::sum` starts from +0 or -0 depending on the rustc version; both
// give the same bits here since every term is >= +0), then floor (= truncation, f >= 0). The
// explicit _rn intrinsics are never contracted into FMAs; NVRTC would fuse a plain
// `acc + y * w` into an FMA under its default --fmad=true, which rounds differently from the
// CPU near ties (CudaDevice::new also passes --fmad=false). The near-tie test inputs
// (tests/cases/mod.rs) detect such a contraction; random inputs do not.
//
// `basis_div_finish` completes rescale / mod_down: t <- (a - t) * (Q mod r)^-1 mod r.
//
// Conversion table (`ConvTable` in src/basis.rs, u64 words), k sources, T targets:
//   [0, k)      q_j
//   [k, 2k)     qhat_inv_j          [2k, 3k) its Shoup companion
//   [3k, 4k)    bits of the f64 1.0 / q_j
//   4k + t * (6 + 2k), target t:
//     +0 r, +1 Q mod r, +2 its companion, +3 h mod r, +4 (Q mod r)^-1, +5 its companion,
//     +6 + 2j qhat_j mod r, +7 + 2j its companion.
//
// Launch: one thread per coefficient, grid = (ceil(N / blockDim.x), 1) for basis_convert
// (each thread loops over the targets, so the y_j and e are computed once) and
// (ceil(N / blockDim.x), T) for basis_div_finish (`ElemLaunch` in src/plan.rs).
// Inputs are reduced residues; outputs are fully reduced, so they can feed the inverse NTT
// (which needs < 2q) and the forward NTT directly.
//
// The Rust module `emu_arith` (src/emu_arith.rs, tests only) replays this file thread by thread;
// tags [B*] mark corresponding lines.

typedef unsigned long long u64;
typedef unsigned int u32;

// Largest number of source primes (alpha for mod_up, k for mod_down); `MAX_SRC` in
// src/basis.rs. The y_j live in a per-thread array of this size.
#define MAX_SRC 16

// [B1] (a + b) mod q, a, b < q.
__device__ __forceinline__ u64 add_mod(u64 a, u64 b, u64 q) {
    const u64 s = a + b;
    return (s >= q) ? s - q : s;
}

// [B2] (a - b) mod q, a, b < q.
__device__ __forceinline__ u64 sub_mod(u64 a, u64 b, u64 q) {
    return (a >= b) ? a - b : a + q - b;
}

// [B3] (a * w) mod q (Shoup, any u64 a).
__device__ __forceinline__ u64 mul_shoup(u64 a, u64 w, u64 ws, u64 q) {
    const u64 hi = __umul64hi(a, ws);
    u64 r = a * w - hi * q;
    if (r >= q) r -= q;
    return r;
}

// out (T limbs, target order) <- centered exact conversion of src (k limbs, source order).
extern "C" __global__ void basis_convert(u64 *__restrict__ out, const u64 *__restrict__ src,
                                         const u64 *__restrict__ tab, u32 k, u32 n_targets,
                                         u32 log_n, u32 use_f64) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;   // [B4] coefficient
    if (c >= (1u << log_n)) return;
    u64 y[MAX_SRC];
    double f = 0.0;
    for (u32 j = 0; j < k; ++j) {                           // [B5]
        const u64 q = tab[j];
        const u64 s = add_mod(src[((size_t)j << log_n) + c], q >> 1, q);
        y[j] = mul_shoup(s, tab[k + j], tab[2 * k + j], q);
        if (use_f64) {                                      // [B6]
            const double w = __longlong_as_double((long long)tab[3 * k + j]);
            f = __dadd_rn(f, __dmul_rn(__ull2double_rn(y[j]), w));
        }
    }
    // f >= 0 (a sum of non-negative terms), so truncation toward zero is floor.
    const u64 e = use_f64 ? __double2ull_rz(f) : 0ull;          // [B7]
    const u32 stride = 6 + 2 * k;
    for (u32 t = 0; t < n_targets; ++t) {                   // [B8]
        const u64 *tt = tab + 4 * k + t * stride;
        const u64 r = tt[0];
        u64 o = 0;
        for (u32 j = 0; j < k; ++j) {
            o = add_mod(o, mul_shoup(y[j], tt[6 + 2 * j], tt[7 + 2 * j], r), r);
        }
        if (use_f64) o = sub_mod(o, mul_shoup(e, tt[1], tt[2], r), r);   // [B9]
        o = sub_mod(o, tt[3], r);
        out[((size_t)t << log_n) + c] = o;                  // [B10]
    }
}

// t (T limbs) <- (a - t) * (Q mod r)^-1 for the first T limbs of a (target order).
// grid = (ceil(N / blockDim.x), T).
extern "C" __global__ void basis_div_finish(u64 *t_out, const u64 *__restrict__ a,
                                            const u64 *__restrict__ tab, u32 k, u32 log_n) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;   // [B11]
    if (c >= (1u << log_n)) return;
    const u64 *tt = tab + 4 * k + blockIdx.y * (6 + 2 * k);
    const u64 r = tt[0];
    const size_t i = ((size_t)blockIdx.y << log_n) + c;
    t_out[i] = mul_shoup(sub_mod(a[i], t_out[i], r), tt[4], tt[5], r);   // [B12]
}
