//! Plain-Rust replay of `kernels/ntt.cu`, one simulated CUDA thread at a time (tests only).
//!
//! Variable names and the `[A*]` / `[F*]` / `[I*]` tags match the `.cu` file. Code between two
//! `__syncthreads()` runs for every thread of the block before the next phase starts (a valid
//! serialisation of a race-free kernel). Every phase also asserts that the simulated threads of
//! a launch (global stage) or of a block (shared stage) touch each element exactly once, which
//! catches an index mapping that would race on the GPU even if it happened to give the right
//! answer sequentially. The launch sequence is [`NttPlan`], as on the device.

use crate::plan::{GlobalStage, NttPlan, SharedLaunch};
use crate::tables::{HostNttTables, NCONST};

/// `__umul64hi(a, b)`.
fn umul64hi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// [A1]
fn mul_shoup_lazy(y: u64, w: u64, w_shoup: u64, q: u64) -> u64 {
    let hi = umul64hi(y, w_shoup);
    y.wrapping_mul(w).wrapping_sub(hi.wrapping_mul(q))
}

/// [A2]
fn ct_butterfly(x: &mut u64, y: &mut u64, w: u64, w_shoup: u64, q: u64, two_q: u64) {
    let xr = if *x >= two_q { *x - two_q } else { *x };
    let tt = mul_shoup_lazy(*y, w, w_shoup, q);
    debug_assert!(xr < two_q && tt < two_q);
    *x = xr + tt;
    *y = xr + two_q - tt;
}

/// [A3]
fn reduce_4q(mut r: u64, q: u64, two_q: u64) -> u64 {
    if r >= two_q {
        r -= two_q;
    }
    if r >= q {
        r -= q;
    }
    r
}

/// [A4]
fn gs_butterfly(x: &mut u64, y: &mut u64, w: u64, w_shoup: u64, q: u64, two_q: u64) {
    debug_assert!(*x < two_q && *y < two_q);
    let s = *x + *y;
    let d = *x + two_q - *y;
    *x = if s >= two_q { s - two_q } else { s };
    *y = mul_shoup_lazy(d, w, w_shoup, q);
}

/// [A5]
fn gs_last(x: &mut u64, y: &mut u64, c: &[u64]) {
    let q = c[0];
    let two_q = 2 * q;
    debug_assert!(*x < two_q && *y < two_q);
    let s = *x + *y;
    let d = *x + two_q - *y;
    let mut r0 = mul_shoup_lazy(s, c[1], c[2], q);
    let mut r1 = mul_shoup_lazy(d, c[3], c[4], q);
    if r0 >= q {
        r0 -= q;
    }
    if r1 >= q {
        r1 -= q;
    }
    *x = r0;
    *y = r1;
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
            panic!("{what}: element {p} touched {} times", self.0[p]);
        }
    }
}

/// Kernel arguments shared by the four kernels.
struct Args<'a> {
    roots: &'a [u64],
    roots_shoup: &'a [u64],
    consts: &'a [u64],
    limb_prime: &'a [u32],
    log_n: u32,
}

/// One launch of `ntt_fwd_global_stage` (`inverse = false`) or `ntt_inv_global_stage`.
fn launch_global(g: &GlobalStage, a: &mut [u64], k: &Args<'_>, inverse: bool) {
    let log_n = k.log_n;
    let mut touched = Touched::new(a.len());
    for by in 0..g.grid.1 {
        for bx in 0..g.grid.0 {
            for tx in 0..g.block.0 {
                let kk = bx * g.block.0 + tx; // [F1] / [I8]
                if kk >= (1u32 << (log_n - 1)) {
                    continue; // [F2]
                }
                let p = k.limb_prime[by as usize] as usize; // [F3]
                let poly = (by as usize) << log_n;
                let rt = &k.roots[p << log_n..];
                let rs = &k.roots_shoup[p << log_n..];
                let cp = &k.consts[p * NCONST..];
                let q = cp[0];
                let log_t = g.log_t;
                let t = 1u32 << log_t;
                let mh = 1u32 << (log_n - 1 - log_t); // m (forward) / h (inverse)
                let i = kk >> log_t; // [F4] / [I9]
                let r = kk & (t - 1); // [F5]
                let j = (i << (log_t + 1)) + r; // [F6]
                let (pj, pjt) = (poly + j as usize, poly + (j + t) as usize);
                touched.hit(pj);
                touched.hit(pjt);
                let (mut x, mut y) = (a[pj], a[pjt]); // [F7] / [I10]
                let w = rt[(mh + i) as usize];
                let ws = rs[(mh + i) as usize];
                if !inverse {
                    ct_butterfly(&mut x, &mut y, w, ws, q, 2 * q); // [F8]
                } else if mh == 1 {
                    gs_last(&mut x, &mut y, cp); // [I11]
                } else {
                    gs_butterfly(&mut x, &mut y, w, ws, q, 2 * q);
                }
                a[pj] = x;
                a[pjt] = y;
            }
        }
    }
    touched.assert_once(if inverse {
        "ntt_inv_global_stage"
    } else {
        "ntt_fwd_global_stage"
    });
}

/// One launch of `ntt_fwd_shared` / `ntt_inv_shared`; each block has its own `smem` and one
/// loop over its threads per `__syncthreads()`-delimited phase.
fn launch_shared(s: &SharedLaunch, a: &mut [u64], k: &Args<'_>, inverse: bool) {
    let (log_n, log_chunk) = (k.log_n, s.log_chunk);
    let half = 1u32 << (log_chunk - 1);
    assert_eq!(s.block.0, half, "blockDim.x must be chunk / 2");
    for by in 0..s.grid.1 {
        for c in 0..s.grid.0 {
            // [F10] / [I2]
            let mut smem = vec![0u64; s.shared_mem_bytes as usize / 8];
            let p = k.limb_prime[by as usize] as usize;
            let base = ((by as usize) << log_n) + ((c as usize) << log_chunk);
            let rt = &k.roots[p << log_n..];
            let rs = &k.roots_shoup[p << log_n..];
            let cp = &k.consts[p * NCONST..];
            let q = cp[0];
            let two_q = 2 * q;
            for tid in 0..half {
                // [F11] / [I3]
                smem[tid as usize] = a[base + tid as usize];
                smem[(tid + half) as usize] = a[base + (tid + half) as usize];
            }
            for s_idx in 0..log_chunk {
                // [F12]: t = half .. 1 ; [I4]: t = 1 .. half
                let log_t = if inverse {
                    s_idx
                } else {
                    log_chunk - 1 - s_idx
                };
                let t = 1u32 << log_t;
                let mh = 1u32 << (log_n - 1 - log_t);
                let mut touched = Touched::new(smem.len());
                for tid in 0..half {
                    let il = tid >> log_t;
                    let r = tid & (t - 1);
                    let jl = (il << (log_t + 1)) + r;
                    let i = (c << (log_chunk - 1 - log_t)) + il;
                    touched.hit(jl as usize);
                    touched.hit((jl + t) as usize);
                    let (mut x, mut y) = (smem[jl as usize], smem[(jl + t) as usize]);
                    if !inverse {
                        let (w, ws) = (rt[(mh + i) as usize], rs[(mh + i) as usize]);
                        ct_butterfly(&mut x, &mut y, w, ws, q, two_q); // [F13]
                    } else if mh == 1 {
                        gs_last(&mut x, &mut y, cp); // [I6]
                    } else {
                        let (w, ws) = (rt[(mh + i) as usize], rs[(mh + i) as usize]);
                        gs_butterfly(&mut x, &mut y, w, ws, q, two_q); // [I5]
                    }
                    smem[jl as usize] = x;
                    smem[(jl + t) as usize] = y;
                }
                touched.assert_once("shared stage");
                // __syncthreads()
            }
            for tid in 0..half {
                let (lo, hi) = (tid as usize, (tid + half) as usize);
                if inverse {
                    a[base + lo] = smem[lo]; // [I7]
                    a[base + hi] = smem[hi];
                } else {
                    a[base + lo] = reduce_4q(smem[lo], q, two_q); // [F14]
                    a[base + hi] = reduce_4q(smem[hi], q, two_q);
                }
            }
        }
    }
}

/// Replays the batched forward NTT of `limb_prime.len()` contiguous limbs (limb `l` uses prime
/// index `limb_prime[l]` of `t`), in place.
pub fn forward(a: &mut [u64], limb_prime: &[u32], t: &HostNttTables) {
    let plan = NttPlan::new(t.n, limb_prime.len()).unwrap();
    assert_eq!(a.len(), t.n * limb_prime.len());
    let k = Args {
        roots: &t.roots,
        roots_shoup: &t.roots_shoup,
        consts: &t.consts,
        limb_prime,
        log_n: plan.log_n,
    };
    for g in &plan.global_stages {
        launch_global(g, a, &k, false);
    }
    launch_shared(&plan.shared, a, &k, false);
}

/// Replays the batched inverse NTT (see [`forward`]), in place.
pub fn inverse(a: &mut [u64], limb_prime: &[u32], t: &HostNttTables) {
    let plan = NttPlan::new(t.n, limb_prime.len()).unwrap();
    assert_eq!(a.len(), t.n * limb_prime.len());
    let k = Args {
        roots: &t.inv_roots,
        roots_shoup: &t.inv_roots_shoup,
        consts: &t.consts,
        limb_prime,
        log_n: plan.log_n,
    };
    launch_shared(&plan.shared, a, &k, true);
    for g in plan.global_stages.iter().rev() {
        launch_global(g, a, &k, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hekit_ckks::CkksParams;
    use hekit_core::primes::ntt_primes;
    use hekit_core::{Modulus, NttTable};
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    /// Random limbs for prime indices `limb_prime`, plus the CPU forward / inverse references.
    fn check(t: &HostNttTables, tabs: &[&NttTable], limb_prime: &[u32], fill: Fill, seed: u64) {
        let n = t.n;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let input: Vec<u64> = limb_prime
            .iter()
            .flat_map(|&p| {
                let q = t.primes[p as usize];
                (0..n)
                    .map(|i| match fill {
                        Fill::Random => rng.next_u64() % q,
                        Fill::Max => q - 1,
                        Fill::Zero => 0,
                        Fill::Alternating => q - 1 - (i as u64 & 1),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut expect_fwd = input.clone();
        for (limb, &p) in expect_fwd.chunks_mut(n).zip(limb_prime) {
            tabs[p as usize].forward(limb);
        }
        let mut got = input.clone();
        forward(&mut got, limb_prime, t);
        assert_eq!(got, expect_fwd, "forward, n = {n}, limbs {limb_prime:?}");

        // Inverse of the forward output gives back the input; also inverse of the raw input
        // (any vector in [0, q)) against the CPU inverse.
        inverse(&mut got, limb_prime, t);
        assert_eq!(got, input, "inverse(forward), n = {n}");
        let mut expect_inv = input.clone();
        for (limb, &p) in expect_inv.chunks_mut(n).zip(limb_prime) {
            tabs[p as usize].inverse(limb);
        }
        let mut got = input;
        inverse(&mut got, limb_prime, t);
        assert_eq!(got, expect_inv, "inverse, n = {n}, limbs {limb_prime:?}");
    }

    #[derive(Clone, Copy)]
    enum Fill {
        Random,
        Max,
        Zero,
        Alternating,
    }

    fn standalone(n: usize, bits: u32, count: usize) -> Vec<NttTable> {
        ntt_primes(bits, n, count)
            .unwrap()
            .into_iter()
            .map(|q| NttTable::new(n, Modulus::new(q).unwrap()).unwrap())
            .collect()
    }

    #[test]
    fn small_degrees_all_paths() {
        // n <= CHUNK: shared kernel only (inverse includes the fused last stage there);
        // n = 4096, 8192: one and two global stages.
        for (k, n) in [2usize, 4, 16, 256, 2048, 4096, 8192]
            .into_iter()
            .enumerate()
        {
            let tabs = standalone(n, 60, 3);
            let refs: Vec<&NttTable> = tabs.iter().collect();
            let t = HostNttTables::new(refs.iter().copied()).unwrap();
            check(&t, &refs, &[0, 1, 2], Fill::Random, k as u64);
            // Out-of-order and repeated limb-to-prime map.
            check(&t, &refs, &[2, 0, 2, 1], Fill::Random, 100 + k as u64);
        }
    }

    #[test]
    fn extreme_inputs() {
        let tabs = standalone(1 << 13, 60, 2);
        let refs: Vec<&NttTable> = tabs.iter().collect();
        let t = HostNttTables::new(refs.iter().copied()).unwrap();
        for fill in [Fill::Max, Fill::Zero, Fill::Alternating] {
            check(&t, &refs, &[0, 1], fill, 0);
        }
    }

    #[test]
    fn test_tiny_and_seal_2020_q_and_p() {
        for params in [CkksParams::test_tiny(), CkksParams::seal_2020()] {
            let refs: Vec<&NttTable> = params
                .basis_q()
                .ntt_tables()
                .iter()
                .chain(params.basis_p().ntt_tables())
                .collect();
            let t = HostNttTables::from_params(&params).unwrap();
            let all: Vec<u32> = (0..t.num_primes() as u32).collect();
            check(&t, &refs, &all, Fill::Random, 7);
        }
    }

    /// The n16_train shape (N = 2^16: five global stages + shared), on a few Q and P primes
    /// (the full 36-prime replay is the same code path and only slower).
    #[test]
    fn n16_train_subset() {
        let params = CkksParams::n16_train();
        let refs: Vec<&NttTable> = params
            .basis_q()
            .ntt_tables()
            .iter()
            .chain(params.basis_p().ntt_tables())
            .collect();
        let t = HostNttTables::from_params(&params).unwrap();
        let last = t.num_primes() as u32 - 1;
        check(&t, &refs, &[0, 1, 31, last], Fill::Random, 16);
        check(&t, &refs, &[0, last], Fill::Max, 17);
    }
}
