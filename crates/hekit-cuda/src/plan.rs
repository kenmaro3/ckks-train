//! Launch configuration of the batched NTT kernels (`kernels/ntt.cu`) and of the element-wise
//! kernels (`kernels/pointwise.cu`, `kernels/basis.cu`). Host-only, no CUDA needed: the device
//! code launches exactly these configurations, and the CPU replay of the kernels used by the
//! tests follows them too.

use hekit_core::{HekitError, Result};

/// log2 of the shared-memory chunk (coefficients per CUDA block in the shared kernels).
pub const LOG_CHUNK: u32 = 11;
/// Coefficients per shared-memory chunk (2048 u64 = 16 KiB, below the 48 KiB default limit).
pub const CHUNK: usize = 1 << LOG_CHUNK;
/// Threads per block of the global-memory stage kernels. Must be `<= CHUNK` so that every block
/// of a global stage (`t >= CHUNK`) lies inside one butterfly group (twiddle load broadcast).
pub const GLOBAL_BLOCK: u32 = 256;
/// Largest number of limbs one launch can carry (`gridDim.y` limit).
pub const MAX_LIMBS: usize = 65535;

/// One launch of `ntt_fwd_global_stage` / `ntt_inv_global_stage` (one stage).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalStage {
    /// `gridDim` = (blocks per limb, limbs, 1).
    pub grid: (u32, u32, u32),
    /// `blockDim` = (`GLOBAL_BLOCK`, 1, 1).
    pub block: (u32, u32, u32),
    /// `log2 t`, `t` the butterfly distance (`N / (2t)` groups, twiddles at `N / (2t) + i`).
    pub log_t: u32,
}

/// One launch of `ntt_fwd_shared` / `ntt_inv_shared` (all stages with `t < chunk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedLaunch {
    /// `gridDim` = (chunks per limb, limbs, 1).
    pub grid: (u32, u32, u32),
    /// `blockDim` = (chunk / 2, 1, 1).
    pub block: (u32, u32, u32),
    /// Dynamic shared memory = chunk * 8 bytes.
    pub shared_mem_bytes: u32,
    /// `log2 chunk = min(log2 N, LOG_CHUNK)`.
    pub log_chunk: u32,
}

/// Launch sequence of one batched forward or inverse NTT over `limbs` limbs of degree `n`.
///
/// Forward: `global_stages` in order (`t = N/2, ..., CHUNK`), then `shared`.
/// Inverse: `shared` first (`t = 1, ..., chunk/2`), then `global_stages` in **reverse** order
/// (`t = CHUNK, ..., N/2`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NttPlan {
    /// `N`.
    pub n: usize,
    /// `log2 N`.
    pub log_n: u32,
    /// Number of limbs (limb `l` at offset `l * N`).
    pub limbs: usize,
    /// Global-memory stages in forward order (`log_t` descending).
    pub global_stages: Vec<GlobalStage>,
    /// The shared-memory launch.
    pub shared: SharedLaunch,
}

impl NttPlan {
    /// Plan for `limbs` limbs of degree `n`. `InvalidParams` unless `n` is a power of two with
    /// `2 <= n <= 2^31` and `1 <= limbs <= MAX_LIMBS`.
    pub fn new(n: usize, limbs: usize) -> Result<NttPlan> {
        if !(n >= 2 && n.is_power_of_two() && n <= 1 << 31) {
            return Err(HekitError::InvalidParams(format!(
                "NTT size {n} must be a power of two in [2, 2^31]"
            )));
        }
        if !(1..=MAX_LIMBS).contains(&limbs) {
            return Err(HekitError::InvalidParams(format!(
                "{limbs} limbs: one launch carries 1..={MAX_LIMBS}"
            )));
        }
        let log_n = n.trailing_zeros();
        let log_chunk = log_n.min(LOG_CHUNK);
        let limbs_u32 = limbs as u32;
        let butterflies = (n / 2) as u32;
        let global_stages = (0..log_n - log_chunk)
            .map(|s| GlobalStage {
                grid: (butterflies.div_ceil(GLOBAL_BLOCK), limbs_u32, 1),
                block: (GLOBAL_BLOCK, 1, 1),
                log_t: log_n - 1 - s,
            })
            .collect();
        let shared = SharedLaunch {
            grid: ((n >> log_chunk) as u32, limbs_u32, 1),
            block: (1 << (log_chunk - 1), 1, 1),
            shared_mem_bytes: (8usize << log_chunk) as u32,
            log_chunk,
        };
        Ok(NttPlan {
            n,
            log_n,
            limbs,
            global_stages,
            shared,
        })
    }
}

/// Threads per block of the element-wise kernels (`kernels/pointwise.cu`, `kernels/basis.cu`).
pub const ELEM_BLOCK: u32 = 256;

/// Launch shape of an element-wise kernel: one thread per coefficient, `rows` limbs in
/// `gridDim.y` (1 for `basis_convert`, which loops over its targets in each thread).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElemLaunch {
    /// `gridDim` = (`ceil(N / ELEM_BLOCK)`, rows, 1).
    pub grid: (u32, u32, u32),
    /// `blockDim` = (`ELEM_BLOCK`, 1, 1).
    pub block: (u32, u32, u32),
}

impl ElemLaunch {
    /// Shape for `rows` limbs of degree `n`. `InvalidParams` unless `n` is a power of two in
    /// `[2, 2^31]` and `1 <= rows <= MAX_LIMBS`.
    pub fn new(n: usize, rows: usize) -> Result<ElemLaunch> {
        if !(n >= 2 && n.is_power_of_two() && n <= 1 << 31) {
            return Err(HekitError::InvalidParams(format!(
                "degree {n} must be a power of two in [2, 2^31]"
            )));
        }
        if !(1..=MAX_LIMBS).contains(&rows) {
            return Err(HekitError::InvalidParams(format!(
                "{rows} limbs: one launch carries 1..={MAX_LIMBS}"
            )));
        }
        Ok(ElemLaunch {
            grid: ((n as u32).div_ceil(ELEM_BLOCK), rows as u32, 1),
            block: (ELEM_BLOCK, 1, 1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n65536_32_limbs() {
        let p = NttPlan::new(1 << 16, 32).unwrap();
        assert_eq!(p.log_n, 16);
        assert_eq!(p.global_stages.len(), 5);
        let log_ts: Vec<u32> = p.global_stages.iter().map(|g| g.log_t).collect();
        assert_eq!(log_ts, [15, 14, 13, 12, 11]);
        for g in &p.global_stages {
            assert_eq!(g.grid, (128, 32, 1));
            assert_eq!(g.block, (GLOBAL_BLOCK, 1, 1));
        }
        assert_eq!(
            p.shared,
            SharedLaunch {
                grid: (32, 32, 1),
                block: (1024, 1, 1),
                shared_mem_bytes: 16384,
                log_chunk: 11,
            }
        );
    }

    #[test]
    fn small_n_is_shared_only() {
        for log_n in 1..=LOG_CHUNK {
            let p = NttPlan::new(1 << log_n, 3).unwrap();
            assert!(p.global_stages.is_empty());
            assert_eq!(p.shared.log_chunk, log_n);
            assert_eq!(p.shared.grid, (1, 3, 1));
            assert_eq!(p.shared.block, (1 << (log_n - 1), 1, 1));
        }
    }

    #[test]
    fn elementwise_shapes() {
        let l = ElemLaunch::new(1 << 16, 36).unwrap();
        assert_eq!(l.grid, (256, 36, 1));
        assert_eq!(l.block, (ELEM_BLOCK, 1, 1));
        // N below the block size: one partial block (threads c >= N return).
        assert_eq!(ElemLaunch::new(32, 1).unwrap().grid, (1, 1, 1));
        assert!(ElemLaunch::new(1 << 10, 0).is_err());
        assert!(ElemLaunch::new(3, 1).is_err());
    }

    #[test]
    fn invalid() {
        for (n, l) in [
            (0, 1),
            (1, 1),
            (3, 1),
            (1 << 10, 0),
            (1 << 10, MAX_LIMBS + 1),
        ] {
            assert!(matches!(
                NttPlan::new(n, l),
                Err(HekitError::InvalidParams(_))
            ));
        }
    }
}
