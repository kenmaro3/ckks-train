// Radix-2 negacyclic forward NTT over Z_q[X]/(X^N + 1), batched, for NVRTC (no #include).
//
// Conventions are exactly those of the Task 2 CPU reference (`spikes/ntt-cpu/src/lib.rs`):
//   roots[i] = psi^bitrev(i), roots_shoup[i] = floor(roots[i] * 2^64 / q), q < 2^60;
//   Cooley-Tukey, natural-order input in [0, q), bit-reversed output fully reduced to [0, q);
//   at stage m (t = N / (2m)) group i uses roots[m + i] on the pairs (j, j + t),
//   j in [2*i*t, 2*i*t + t).
// The per-butterfly arithmetic (Harvey butterfly, lazy Shoup multiplication, values kept in
// [0, 4q) between stages, final [0, 4q) -> [0, q) reduction) is the same as the CPU code, so
// the outputs are bit-identical u64s.
//
// Schedule (why not "1 block = 1 polynomial in shared memory"): N = 2^16 u64 coefficients are
// 512 KiB, more than the shared memory of an SM (228 KiB on H100, 164 KiB on A100). So:
//   * stages with t >= CHUNK (= 2048; the first log2(N / CHUNK) stages, 5 for N = 2^16) run in
//     global memory, one launch of `ntt_fwd_global_stage` per stage, one thread per butterfly,
//     many blocks per polynomial;
//   * the remaining log2(CHUNK) stages (11 for CHUNK = 2048) only pair indices inside one
//     aligned CHUNK-coefficient block, so they run in one launch of `ntt_fwd_shared`, one CUDA
//     block per chunk (CHUNK / 2 threads, one butterfly per thread per stage), with the chunk
//     held in 16 KiB of dynamic shared memory.
// Batch: polynomial p (0 <= p < batch) is contiguous at a + p * N and is selected by blockIdx.y.
// Launch configuration is computed on the host by `LaunchPlan::new` (src/lib.rs).
//
// The Rust function `emu` in src/lib.rs replays this file thread by thread; the tags [G*] / [S*]
// / [A*] mark corresponding lines in both files.

typedef unsigned long long u64;
typedef unsigned int u32;

// [A1] Lazy Shoup product: value congruent to y * w (mod q) in [0, 2q), for y < 2^64, w < q,
// w_shoup = floor(w * 2^64 / q). Products are wrapping (mod 2^64).
__device__ __forceinline__ u64 mul_shoup_lazy(u64 y, u64 w, u64 w_shoup, u64 q) {
    const u64 hi = __umul64hi(y, w_shoup);
    return y * w - hi * q;
}

// [A2] Harvey CT butterfly. In: x, y in [0, 4q). Out: x' = xr + tt in [0, 4q),
// y' = xr + 2q - tt in (0, 4q), where xr in [0, 2q), tt in [0, 2q). 4q < 2^62: no overflow.
__device__ __forceinline__ void butterfly(u64 &x, u64 &y, u64 w, u64 w_shoup, u64 q, u64 two_q) {
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

// One global-memory stage (m, t = 2^log_t, requires t >= blockDim.x so a block shares a twiddle).
// grid = (ceil((N/2) / blockDim.x), batch), one thread per butterfly.
extern "C" __global__ void ntt_fwd_global_stage(u64 *a, const u64 *__restrict__ roots,
                                                const u64 *__restrict__ roots_shoup, u64 q,
                                                u32 log_n, u32 m, u32 log_t) {
    const u32 k = blockIdx.x * blockDim.x + threadIdx.x;   // [G1] butterfly index in [0, N/2)
    if (k >= (1u << (log_n - 1))) return;                   // [G2]
    u64 *poly = a + ((size_t)blockIdx.y << log_n);          // [G3]
    const u32 t = 1u << log_t;
    const u32 i = k >> log_t;                               // [G4] group, in [0, m)
    const u32 r = k & (t - 1);                              // [G5] offset inside the group
    const u32 j = (i << (log_t + 1)) + r;                   // [G6] = 2*i*t + r
    const u64 w = roots[m + i];                             // [G7]
    const u64 w_shoup = roots_shoup[m + i];
    u64 x = poly[j];                                        // [G8]
    u64 y = poly[j + t];
    butterfly(x, y, w, w_shoup, q, 2 * q);                  // [G9]
    poly[j] = x;                                            // [G10]
    poly[j + t] = y;
}

// The last log_chunk stages (m = m0, 2*m0, ..., N/2 with m0 = N >> log_chunk) on one aligned
// chunk of 2^log_chunk coefficients, then the final reduction to [0, q).
// grid = (N >> log_chunk, batch), block = 2^(log_chunk - 1) threads,
// dynamic shared memory = 2^log_chunk * 8 bytes.
extern "C" __global__ void ntt_fwd_shared(u64 *a, const u64 *__restrict__ roots,
                                          const u64 *__restrict__ roots_shoup, u64 q, u32 log_n,
                                          u32 m0, u32 log_chunk) {
    extern __shared__ u64 smem[];
    const u32 half = 1u << (log_chunk - 1);
    const u32 tid = threadIdx.x;                            // [S1] in [0, half)
    const u32 c = blockIdx.x;                               // [S2] chunk index in [0, N >> log_chunk)
    u64 *base = a + ((size_t)blockIdx.y << log_n) + ((size_t)c << log_chunk);  // [S3]
    const u64 two_q = 2 * q;

    smem[tid] = base[tid];                                  // [S4] coalesced load
    smem[tid + half] = base[tid + half];
    __syncthreads();

    u32 m = m0;
    for (u32 s = 0; s < log_chunk; ++s) {                   // [S5] t = half, half/2, ..., 1
        const u32 log_t = log_chunk - 1 - s;
        const u32 t = 1u << log_t;
        const u32 il = tid >> log_t;                        // [S6] group inside the chunk
        const u32 r = tid & (t - 1);                        // [S7]
        const u32 jl = (il << (log_t + 1)) + r;             // [S8] local index = 2*il*t + r
        const u32 i = (c << (log_chunk - 1 - log_t)) + il;  // [S9] global group = c*CHUNK/(2t) + il
        const u64 w = roots[m + i];                         // [S10]
        const u64 w_shoup = roots_shoup[m + i];
        u64 x = smem[jl];                                   // [S11]
        u64 y = smem[jl + t];
        butterfly(x, y, w, w_shoup, q, two_q);              // [S12]
        smem[jl] = x;
        smem[jl + t] = y;
        __syncthreads();                                    // [S13]
        m <<= 1;
    }

    base[tid] = reduce_4q(smem[tid], q, two_q);             // [S14] coalesced store, [0, q)
    base[tid + half] = reduce_4q(smem[tid + half], q, two_q);
}
