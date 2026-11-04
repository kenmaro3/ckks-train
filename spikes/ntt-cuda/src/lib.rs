//! Spike 1b: batched radix-2 negacyclic forward NTT on CUDA (`kernels/ntt.cu`, compiled at
//! runtime with NVRTC through `cudarc`), bit-exact with the Task 2 CPU reference `ntt_cpu`.
//!
//! * [`NttTables`]: the same prime / root / Shoup tables as `ntt_cpu` (uploaded to the device).
//! * [`LaunchPlan`]: the launch configuration of the kernels (global-memory stages, then one
//!   shared-memory kernel for the last `log2(CHUNK)` stages; see the header of `kernels/ntt.cu`).
//! * [`emu`]: a plain-Rust replay of the kernels, thread by thread, used to test the index
//!   mapping and arithmetic on machines without a GPU.
//! * [`gpu`]: host code (device probe, NVRTC compile, launches, CUDA-event timer).
#![deny(missing_docs)]

use ntt_cpu::{build_roots, gen_ntt_prime, primitive_2n_root, shoup_table};

/// log2 of the shared-memory chunk (coefficients per CUDA block in `ntt_fwd_shared`).
pub const LOG_CHUNK: u32 = 11;
/// Coefficients per shared-memory chunk (2048 u64 = 16 KiB, below the 48 KiB default limit).
pub const CHUNK: usize = 1 << LOG_CHUNK;
/// Threads per block of the global-memory stage kernel (must be <= CHUNK so that every block
/// of a global stage lies inside one butterfly group, `t >= CHUNK`).
pub const GLOBAL_BLOCK: u32 = 256;

/// Source of the CUDA kernels.
pub const KERNEL_SRC: &str = include_str!("../kernels/ntt.cu");

/// Precomputed tables, identical to those of `ntt_cpu` (same prime generator and root).
#[derive(Clone, Debug)]
pub struct NttTables {
    /// Ring degree `N` (power of two).
    pub n: usize,
    /// `q = ntt_cpu::gen_ntt_prime(N, 60)`.
    pub q: u64,
    /// `psi = ntt_cpu::primitive_2n_root(N, q)`.
    pub psi: u64,
    /// `roots[i] = psi^bitrev(i) mod q`.
    pub roots: Vec<u64>,
    /// `floor(roots[i] * 2^64 / q)`.
    pub roots_shoup: Vec<u64>,
}

impl NttTables {
    /// Tables for degree `n` with the 60-bit prime used by the Task 2 spike.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2 && n.is_power_of_two());
        let q = gen_ntt_prime(n, 60);
        let psi = primitive_2n_root(n, q);
        let roots = build_roots(n, q, psi);
        let roots_shoup = shoup_table(&roots, q);
        Self {
            n,
            q,
            psi,
            roots,
            roots_shoup,
        }
    }
}

/// Launch of `ntt_fwd_global_stage` for one stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalStage {
    /// `gridDim` = (blocks per polynomial, batch, 1).
    pub grid: (u32, u32, u32),
    /// `blockDim` = (GLOBAL_BLOCK, 1, 1).
    pub block: (u32, u32, u32),
    /// Dynamic shared memory (0).
    pub shared_mem_bytes: u32,
    /// Stage index `m` (number of groups; twiddles `roots[m..2m]`).
    pub m: u32,
    /// `log2 t`, `t = N / (2m)` (butterfly distance).
    pub log_t: u32,
}

/// Launch of `ntt_fwd_shared` (all remaining stages + final reduction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedLaunch {
    /// `gridDim` = (chunks per polynomial, batch, 1).
    pub grid: (u32, u32, u32),
    /// `blockDim` = (chunk / 2, 1, 1).
    pub block: (u32, u32, u32),
    /// Dynamic shared memory = chunk * 8 bytes.
    pub shared_mem_bytes: u32,
    /// First stage handled in shared memory, `m0 = N / chunk`.
    pub m: u32,
    /// `log2 chunk = min(log2 N, LOG_CHUNK)`.
    pub log_chunk: u32,
}

/// Complete launch sequence for one batched forward NTT.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchPlan {
    /// `N`.
    pub n: usize,
    /// `log2 N`.
    pub log_n: u32,
    /// Number of polynomials (contiguous, polynomial `p` at offset `p * N`).
    pub batch: usize,
    /// Global-memory stages, in launch order (`m = 1, 2, ..., N / (2 chunk)`).
    pub global_stages: Vec<GlobalStage>,
    /// Final shared-memory launch.
    pub shared: SharedLaunch,
}

impl LaunchPlan {
    /// Plan for `batch` polynomials of degree `n` (`2 <= n`, power of two, `n <= 2^31`,
    /// `1 <= batch <= 65535` = max `gridDim.y`).
    pub fn new(n: usize, batch: usize) -> Self {
        assert!(n >= 2 && n.is_power_of_two() && n <= 1 << 31);
        assert!((1..=65535).contains(&batch));
        let log_n = n.trailing_zeros();
        let log_chunk = log_n.min(LOG_CHUNK);
        let batch_u32 = batch as u32;
        let butterflies = (n / 2) as u32;
        let global_stages = (0..log_n - log_chunk)
            .map(|s| GlobalStage {
                grid: (butterflies.div_ceil(GLOBAL_BLOCK), batch_u32, 1),
                block: (GLOBAL_BLOCK, 1, 1),
                shared_mem_bytes: 0,
                m: 1 << s,
                log_t: log_n - 1 - s,
            })
            .collect();
        let shared = SharedLaunch {
            grid: ((n >> log_chunk) as u32, batch_u32, 1),
            block: (1 << (log_chunk - 1), 1, 1),
            shared_mem_bytes: (8usize << log_chunk) as u32,
            m: 1 << (log_n - log_chunk),
            log_chunk,
        };
        Self {
            n,
            log_n,
            batch,
            global_stages,
            shared,
        }
    }
}

/// Plain-Rust replay of `kernels/ntt.cu`, one simulated CUDA thread at a time.
///
/// Variable names and the `[A*]` / `[G*]` / `[S*]` tags match the `.cu` file. Code between two
/// `__syncthreads()` is run for every thread of the block before the next phase starts (a valid
/// serialisation of a race-free kernel). Every phase also asserts that the simulated threads of
/// a launch (global stage) or of a block (shared stage) touch each element exactly once, which
/// would catch an index mapping that is a data race on the GPU even if it happened to give the
/// right answer sequentially.
pub mod emu {
    use crate::{GlobalStage, LaunchPlan, NttTables, SharedLaunch};

    /// CUDA `uint3` / `dim3`.
    #[derive(Clone, Copy, Debug)]
    pub struct Dim3 {
        /// x
        pub x: u32,
        /// y
        pub y: u32,
    }

    /// `__umul64hi(a, b)`: high 64 bits of the 128-bit product.
    #[inline]
    pub fn umul64hi(a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) >> 64) as u64
    }

    /// [A1] `mul_shoup_lazy`.
    #[inline]
    fn mul_shoup_lazy(y: u64, w: u64, w_shoup: u64, q: u64) -> u64 {
        let hi = umul64hi(y, w_shoup);
        y.wrapping_mul(w).wrapping_sub(hi.wrapping_mul(q))
    }

    /// [A2] `butterfly`.
    #[inline]
    fn butterfly(x: &mut u64, y: &mut u64, w: u64, w_shoup: u64, q: u64, two_q: u64) {
        let xr = if *x >= two_q { *x - two_q } else { *x };
        let tt = mul_shoup_lazy(*y, w, w_shoup, q);
        // Lazy-reduction bounds (x, y < 4q < 2^62): xr < 2q, tt < 2q.
        debug_assert!(xr < two_q && tt < two_q);
        *x = xr + tt;
        *y = xr + two_q - tt;
    }

    /// [A3] `reduce_4q`.
    #[inline]
    fn reduce_4q(mut r: u64, q: u64, two_q: u64) -> u64 {
        if r >= two_q {
            r -= two_q;
        }
        if r >= q {
            r -= q;
        }
        r
    }

    /// Counts accesses per element in one phase; asserts each element is touched exactly once.
    struct Touched(Vec<u8>);
    impl Touched {
        fn new(len: usize) -> Self {
            Self(vec![0; len])
        }
        fn hit(&mut self, idx: usize) {
            self.0[idx] += 1;
        }
        fn assert_once(&self, what: &str) {
            if let Some(p) = self.0.iter().position(|&c| c != 1) {
                panic!(
                    "{what}: element {p} touched {} times in one phase",
                    self.0[p]
                );
            }
        }
    }

    /// One thread of `ntt_fwd_global_stage`.
    #[allow(clippy::too_many_arguments)]
    fn ntt_fwd_global_stage_thread(
        block_idx: Dim3,
        block_dim: Dim3,
        thread_idx: Dim3,
        a: &mut [u64],
        roots: &[u64],
        roots_shoup: &[u64],
        q: u64,
        log_n: u32,
        m: u32,
        log_t: u32,
        touched: &mut Touched,
    ) {
        let k = block_idx.x * block_dim.x + thread_idx.x; // [G1]
        if k >= (1u32 << (log_n - 1)) {
            return; // [G2]
        }
        let poly = (block_idx.y as usize) << log_n; // [G3] offset of `poly` in `a`
        let t = 1u32 << log_t;
        let i = k >> log_t; // [G4]
        let r = k & (t - 1); // [G5]
        let j = (i << (log_t + 1)) + r; // [G6]
        let w = roots[(m + i) as usize]; // [G7]
        let w_shoup = roots_shoup[(m + i) as usize];
        let (pj, pjt) = (poly + j as usize, poly + (j + t) as usize);
        touched.hit(pj);
        touched.hit(pjt);
        let mut x = a[pj]; // [G8]
        let mut y = a[pjt];
        butterfly(&mut x, &mut y, w, w_shoup, q, 2 * q); // [G9]
        a[pj] = x; // [G10]
        a[pjt] = y;
    }

    /// One launch of `ntt_fwd_global_stage` with the configuration `g`.
    fn launch_global_stage(g: &GlobalStage, a: &mut [u64], t: &NttTables, log_n: u32) {
        let block_dim = Dim3 {
            x: g.block.0,
            y: g.block.1,
        };
        let mut touched = Touched::new(a.len());
        for by in 0..g.grid.1 {
            for bx in 0..g.grid.0 {
                for tx in 0..g.block.0 {
                    ntt_fwd_global_stage_thread(
                        Dim3 { x: bx, y: by },
                        block_dim,
                        Dim3 { x: tx, y: 0 },
                        a,
                        &t.roots,
                        &t.roots_shoup,
                        t.q,
                        log_n,
                        g.m,
                        g.log_t,
                        &mut touched,
                    );
                }
            }
        }
        touched.assert_once("ntt_fwd_global_stage");
    }

    /// One launch of `ntt_fwd_shared` with the configuration `s`; each block is simulated with
    /// its own `smem` and with one loop over threads per `__syncthreads()`-delimited phase.
    fn launch_shared(s: &SharedLaunch, a: &mut [u64], t: &NttTables, log_n: u32) {
        let (roots, roots_shoup, q) = (&t.roots, &t.roots_shoup, t.q);
        let (m0, log_chunk) = (s.m, s.log_chunk);
        for by in 0..s.grid.1 {
            for bx in 0..s.grid.0 {
                let block_idx = Dim3 { x: bx, y: by };
                let mut smem = vec![0u64; s.shared_mem_bytes as usize / 8]; // extern __shared__
                let half = 1u32 << (log_chunk - 1);
                assert_eq!(s.block.0, half, "blockDim.x must be chunk / 2");
                let c = block_idx.x; // [S2]
                let base = ((block_idx.y as usize) << log_n) + ((c as usize) << log_chunk); // [S3]
                let two_q = 2 * q;

                for tid in 0..s.block.0 {
                    // [S1]
                    smem[tid as usize] = a[base + tid as usize]; // [S4]
                    smem[(tid + half) as usize] = a[base + (tid + half) as usize];
                }
                // __syncthreads()

                let mut m = m0;
                for st in 0..log_chunk {
                    // [S5]
                    let log_t = log_chunk - 1 - st;
                    let tt = 1u32 << log_t; // `t` in the .cu file
                    let mut touched = Touched::new(smem.len());
                    for tid in 0..s.block.0 {
                        let il = tid >> log_t; // [S6]
                        let r = tid & (tt - 1); // [S7]
                        let jl = (il << (log_t + 1)) + r; // [S8]
                        let i = (c << (log_chunk - 1 - log_t)) + il; // [S9]
                        let w = roots[(m + i) as usize]; // [S10]
                        let w_shoup = roots_shoup[(m + i) as usize];
                        touched.hit(jl as usize);
                        touched.hit((jl + tt) as usize);
                        let mut x = smem[jl as usize]; // [S11]
                        let mut y = smem[(jl + tt) as usize];
                        butterfly(&mut x, &mut y, w, w_shoup, q, two_q); // [S12]
                        smem[jl as usize] = x;
                        smem[(jl + tt) as usize] = y;
                    }
                    touched.assert_once("ntt_fwd_shared stage");
                    // __syncthreads() [S13]
                    m <<= 1;
                }

                for tid in 0..s.block.0 {
                    a[base + tid as usize] = reduce_4q(smem[tid as usize], q, two_q); // [S14]
                    a[base + (tid + half) as usize] =
                        reduce_4q(smem[(tid + half) as usize], q, two_q);
                }
            }
        }
    }

    /// Replays the full launch sequence of [`LaunchPlan::new`]`(t.n, batch)` on `a`
    /// (`batch` contiguous polynomials of length `t.n`, coefficients in `[0, q)`), in place.
    pub fn forward_batch_emulated(a: &mut [u64], batch: usize, t: &NttTables) {
        assert_eq!(a.len(), t.n * batch);
        let plan = LaunchPlan::new(t.n, batch);
        for g in &plan.global_stages {
            launch_global_stage(g, a, t, plan.log_n);
        }
        launch_shared(&plan.shared, a, t, plan.log_n);
    }
}

pub mod gpu;
