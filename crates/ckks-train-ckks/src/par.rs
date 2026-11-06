//! Optional data parallelism (cargo feature `parallel`, rayon; Task 38).
//!
//! Without the feature every helper runs sequentially, so the default build behaves exactly
//! as before (single-threaded timings of M1 and M2). With it, the helpers run on the current
//! rayon pool: the global pool (all cores, or `RAYON_NUM_THREADS`), or the pool a caller
//! installs (`rayon::ThreadPool::install`, as the bootstrap benchmark does for 1 and N
//! threads).
//!
//! Every parallel section computes independent results and **collects them in their
//! sequential order** before any reduction, so results are bit-identical with and without
//! the feature, for any thread count. Used by: the digit decomposition and the per-limb
//! inner products of key switching, the steps of hoisted rotations, the giant-step blocks of
//! the BSGS products, and the two EvalMods of a bootstrap.

/// `(0..n).map(f).collect()`, in parallel with the feature.
pub(crate) fn map_range<U, F>(n: usize, f: F) -> Vec<U>
where
    U: Send,
    F: Fn(usize) -> U + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

/// `items.iter().map(f).collect()`, in parallel with the feature.
pub(crate) fn map_slice<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync + Send,
{
    map_range(items.len(), |i| f(&items[i]))
}

/// `(a(), b())`, the two in parallel with the feature.
pub(crate) fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[cfg(feature = "parallel")]
    {
        rayon::join(a, b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (a(), b())
    }
}
