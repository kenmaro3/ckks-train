// Batched negacyclic NTT / inverse NTT over the RNS limbs of a polynomial, for NVRTC (no
// #include). Ported from the M0 spike `spikes/ntt-cuda/kernels/ntt.cu` (forward kernels), with
// one prime per limb and the Gentleman-Sande inverse added.
//
// Conventions are exactly those of `hekit_core::NttTable` (crates/hekit-core/src/ntt.rs):
//   roots[i] = psi^bitrev(i), inv_roots[i] = psi^-bitrev(i), Shoup companions
//   floor(w * 2^64 / q), q < 2^60.
//   Forward: Cooley-Tukey, natural-order input in [0, q), bit-reversed output in [0, q); at the
//   stage with distance t (m = N / (2t) groups) group i uses roots[m + i] on the pairs (j, j + t),
//   j in [2*i*t, 2*i*t + t). Values stay in [0, 4q) between stages, final reduction to [0, q).
//   Inverse: Gentleman-Sande, bit-reversed input in [0, q), natural-order output in [0, q); at the
//   stage with distance t (h = N / (2t) groups) group i uses inv_roots[h + i]; values stay in
//   [0, 2q) between stages; the last stage (h = 1, t = N/2) is fused with the N^-1 scaling:
//   x' = (x + y) * n_inv, y' = (x - y) * (inv_roots[1] * n_inv), each reduced to [0, q).
//   Input precondition of the inverse: every residue < 2q (gs_butterfly / gs_last compute
//   x + 2q - y, which wraps for y >= x + 2q). Uploads and every kernel in this directory
//   produce fully reduced residues (< q). The forward transform accepts [0, 4q).
// Every butterfly performs the same u64 operations as the CPU code, so outputs are bit-identical
// whatever the schedule (the schedule only has to respect the stage order).
//
// Layout: limb l (0 <= l < num_limbs, selected by blockIdx.y) is contiguous at a + l * N. Its
// prime index is p = limb_prime[l]; the twiddle tables of prime p are at tables + p * N, and its
// constants at consts + p * NCONST: { q, n_inv, n_inv_shoup, w_n_inv, w_n_inv_shoup,
// barrett_lo, barrett_hi } (the Barrett words are used by kernels/pointwise.cu only).
//
// Schedule (N = 2^16 u64 coefficients are 512 KiB, more than an SM's shared memory):
//   * strides t >= CHUNK (= 2048) run in global memory, one launch per stage, one thread per
//     butterfly (`ntt_fwd_global_stage` / `ntt_inv_global_stage`);
//   * strides t < CHUNK only pair indices inside one aligned CHUNK-coefficient block, so they run
//     in one launch (`ntt_fwd_shared` / `ntt_inv_shared`), one CUDA block per chunk, CHUNK / 2
//     threads, the chunk in dynamic shared memory (16 KiB).
//   Forward: global stages (t = N/2 .. CHUNK), then the shared kernel (t = CHUNK/2 .. 1, plus the
//   final [0, 4q) -> [0, q) reduction). Inverse: the shared kernel first (t = 1 .. CHUNK/2), then
//   the global stages (t = CHUNK .. N/2). When N <= CHUNK the shared kernel does every stage,
//   including (inverse) the fused last one.
// The host launch configuration is `NttPlan::new` (src/plan.rs). The Rust module `emu`
// (src/emu.rs, tests only) replays this file thread by thread; tags [A*], [F*], [I*] mark
// corresponding lines.

typedef unsigned long long u64;
typedef unsigned int u32;

#define NCONST 7

// [A1] Lazy Shoup product: value congruent to y * w (mod q) in [0, 2q), for y < 2^64, w < q,
// w_shoup = floor(w * 2^64 / q). Products are wrapping (mod 2^64).
__device__ __forceinline__ u64 mul_shoup_lazy(u64 y, u64 w, u64 w_shoup, u64 q) {
    const u64 hi = __umul64hi(y, w_shoup);
    return y * w - hi * q;
}

// [A2] Harvey CT butterfly. In: x, y in [0, 4q). Out: x' = xr + tt in [0, 4q),
// y' = xr + 2q - tt in (0, 4q), where xr in [0, 2q), tt in [0, 2q). 4q < 2^62: no overflow.
__device__ __forceinline__ void ct_butterfly(u64 &x, u64 &y, u64 w, u64 w_shoup, u64 q,
                                             u64 two_q) {
    const u64 xr = (x >= two_q) ? x - two_q : x;
    const u64 tt = mul_shoup_lazy(y, w, w_shoup, q);
    x = xr + tt;
    y = xr + two_q - tt;
}

// [A3] [0, 4q) -> [0, q).
__device__ __forceinline__ u64 reduce_4q(u64 r, u64 q, u64 two_q) {
    if (r >= two_q) r -= two_q;
    if (r >= q) r -= q;
    return r;
}

// [A4] GS butterfly. In: x, y in [0, 2q). Out: x' = (x + y) reduced to [0, 2q),
// y' = (x + 2q - y) * w lazily in [0, 2q).
__device__ __forceinline__ void gs_butterfly(u64 &x, u64 &y, u64 w, u64 w_shoup, u64 q,
                                             u64 two_q) {
    const u64 s = x + y;
    const u64 d = x + two_q - y;
    x = (s >= two_q) ? s - two_q : s;
    y = mul_shoup_lazy(d, w, w_shoup, q);
}

// [A5] Fused last GS stage (h = 1): x' = (x + y) * n_inv, y' = (x - y) * w_n_inv, both in
// [0, q). In: x, y in [0, 2q). c = consts of the prime.
__device__ __forceinline__ void gs_last(u64 &x, u64 &y, const u64 *c) {
    const u64 q = c[0];
    const u64 two_q = 2 * q;
    const u64 s = x + y;
    const u64 d = x + two_q - y;
    u64 r0 = mul_shoup_lazy(s, c[1], c[2], q);
    u64 r1 = mul_shoup_lazy(d, c[3], c[4], q);
    if (r0 >= q) r0 -= q;
    if (r1 >= q) r1 -= q;
    x = r0;
    y = r1;
}

// One global-memory forward stage with distance t = 2^log_t (m = N / (2t) groups).
// grid = (ceil((N/2) / blockDim.x), num_limbs), one thread per butterfly.
extern "C" __global__ void ntt_fwd_global_stage(u64 *a, const u64 *__restrict__ roots,
                                                const u64 *__restrict__ roots_shoup,
                                                const u64 *__restrict__ consts,
                                                const u32 *__restrict__ limb_prime, u32 log_n,
                                                u32 log_t) {
    const u32 k = blockIdx.x * blockDim.x + threadIdx.x;   // [F1] butterfly index in [0, N/2)
    if (k >= (1u << (log_n - 1))) return;                   // [F2]
    const size_t p = limb_prime[blockIdx.y];                // [F3] prime of this limb
    u64 *poly = a + ((size_t)blockIdx.y << log_n);
    const u64 *rt = roots + (p << log_n);
    const u64 *rs = roots_shoup + (p << log_n);
    const u64 q = consts[p * NCONST];
    const u32 t = 1u << log_t;
    const u32 m = 1u << (log_n - 1 - log_t);
    const u32 i = k >> log_t;                               // [F4] group, in [0, m)
    const u32 r = k & (t - 1);                              // [F5] offset inside the group
    const u32 j = (i << (log_t + 1)) + r;                   // [F6] = 2*i*t + r
    u64 x = poly[j];                                        // [F7]
    u64 y = poly[j + t];
    ct_butterfly(x, y, rt[m + i], rs[m + i], q, 2 * q);     // [F8]
    poly[j] = x;
    poly[j + t] = y;
}

// The last log_chunk forward stages (t = 2^(log_chunk-1) .. 1) on one aligned chunk of
// 2^log_chunk coefficients, then the final reduction to [0, q).
// grid = (N >> log_chunk, num_limbs), block = 2^(log_chunk - 1) threads,
// dynamic shared memory = 2^log_chunk * 8 bytes.
extern "C" __global__ void __launch_bounds__(1024)
ntt_fwd_shared(u64 *a, const u64 *__restrict__ roots, const u64 *__restrict__ roots_shoup,
               const u64 *__restrict__ consts, const u32 *__restrict__ limb_prime, u32 log_n,
               u32 log_chunk) {
    extern __shared__ u64 smem[];
    const u32 half = 1u << (log_chunk - 1);
    const u32 tid = threadIdx.x;                            // [F9] in [0, half)
    const u32 c = blockIdx.x;                               // [F10] chunk index
    const size_t p = limb_prime[blockIdx.y];
    u64 *base = a + ((size_t)blockIdx.y << log_n) + ((size_t)c << log_chunk);
    const u64 *rt = roots + (p << log_n);
    const u64 *rs = roots_shoup + (p << log_n);
    const u64 q = consts[p * NCONST];
    const u64 two_q = 2 * q;

    smem[tid] = base[tid];                                  // [F11] coalesced load
    smem[tid + half] = base[tid + half];
    __syncthreads();

    for (u32 s = 0; s < log_chunk; ++s) {                   // [F12] t = half, half/2, ..., 1
        const u32 log_t = log_chunk - 1 - s;
        const u32 t = 1u << log_t;
        const u32 m = 1u << (log_n - 1 - log_t);
        const u32 il = tid >> log_t;                        // group inside the chunk
        const u32 r = tid & (t - 1);
        const u32 jl = (il << (log_t + 1)) + r;             // local index = 2*il*t + r
        const u32 i = (c << (log_chunk - 1 - log_t)) + il;  // global group
        u64 x = smem[jl];                                   // [F13]
        u64 y = smem[jl + t];
        ct_butterfly(x, y, rt[m + i], rs[m + i], q, two_q);
        smem[jl] = x;
        smem[jl + t] = y;
        __syncthreads();
    }

    base[tid] = reduce_4q(smem[tid], q, two_q);             // [F14] coalesced store, [0, q)
    base[tid + half] = reduce_4q(smem[tid + half], q, two_q);
}

// The first log_chunk inverse stages (t = 1 .. 2^(log_chunk-1)) on one aligned chunk. When the
// chunk is the whole polynomial (log_chunk == log_n) the last of them is the fused N^-1 stage.
// Output in [0, 2q) (or [0, q) after the fused stage). Same launch shape as ntt_fwd_shared.
extern "C" __global__ void __launch_bounds__(1024)
ntt_inv_shared(u64 *a, const u64 *__restrict__ inv_roots,
               const u64 *__restrict__ inv_roots_shoup, const u64 *__restrict__ consts,
               const u32 *__restrict__ limb_prime, u32 log_n, u32 log_chunk) {
    extern __shared__ u64 smem[];
    const u32 half = 1u << (log_chunk - 1);
    const u32 tid = threadIdx.x;                            // [I1] in [0, half)
    const u32 c = blockIdx.x;                               // [I2] chunk index
    const size_t p = limb_prime[blockIdx.y];
    u64 *base = a + ((size_t)blockIdx.y << log_n) + ((size_t)c << log_chunk);
    const u64 *rt = inv_roots + (p << log_n);
    const u64 *rs = inv_roots_shoup + (p << log_n);
    const u64 *cp = consts + p * NCONST;
    const u64 q = cp[0];
    const u64 two_q = 2 * q;

    smem[tid] = base[tid];                                  // [I3] coalesced load
    smem[tid + half] = base[tid + half];
    __syncthreads();

    for (u32 log_t = 0; log_t < log_chunk; ++log_t) {       // [I4] t = 1, 2, ..., half
        const u32 t = 1u << log_t;
        const u32 h = 1u << (log_n - 1 - log_t);
        const u32 il = tid >> log_t;
        const u32 r = tid & (t - 1);
        const u32 jl = (il << (log_t + 1)) + r;
        const u32 i = (c << (log_chunk - 1 - log_t)) + il;
        u64 x = smem[jl];                                   // [I5]
        u64 y = smem[jl + t];
        if (h == 1) {
            gs_last(x, y, cp);                              // [I6] only when log_chunk == log_n
        } else {
            gs_butterfly(x, y, rt[h + i], rs[h + i], q, two_q);
        }
        smem[jl] = x;
        smem[jl + t] = y;
        __syncthreads();
    }

    base[tid] = smem[tid];                                  // [I7] coalesced store
    base[tid + half] = smem[tid + half];
}

// One global-memory inverse stage with distance t = 2^log_t (h = N / (2t) groups); the stage
// with h = 1 is the fused N^-1 stage. Same launch shape as ntt_fwd_global_stage.
extern "C" __global__ void ntt_inv_global_stage(u64 *a, const u64 *__restrict__ inv_roots,
                                                const u64 *__restrict__ inv_roots_shoup,
                                                const u64 *__restrict__ consts,
                                                const u32 *__restrict__ limb_prime, u32 log_n,
                                                u32 log_t) {
    const u32 k = blockIdx.x * blockDim.x + threadIdx.x;   // [I8]
    if (k >= (1u << (log_n - 1))) return;
    const size_t p = limb_prime[blockIdx.y];
    u64 *poly = a + ((size_t)blockIdx.y << log_n);
    const u64 *rt = inv_roots + (p << log_n);
    const u64 *rs = inv_roots_shoup + (p << log_n);
    const u64 *cp = consts + p * NCONST;
    const u64 q = cp[0];
    const u32 t = 1u << log_t;
    const u32 h = 1u << (log_n - 1 - log_t);
    const u32 i = k >> log_t;                               // [I9]
    const u32 r = k & (t - 1);
    const u32 j = (i << (log_t + 1)) + r;
    u64 x = poly[j];                                        // [I10]
    u64 y = poly[j + t];
    if (h == 1) {
        gs_last(x, y, cp);                                  // [I11]
    } else {
        gs_butterfly(x, y, rt[h + i], rs[h + i], q, 2 * q);
    }
    poly[j] = x;
    poly[j + t] = y;
}
