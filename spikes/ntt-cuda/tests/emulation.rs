//! CPU emulation of the CUDA kernel schedule vs. the Task 2 reference `ntt_cpu::ntt_forward`.
//! Runs on any machine (no GPU needed): it checks the launch configuration, the
//! thread/block/stage index mapping and the per-butterfly arithmetic of `kernels/ntt.cu`.

use ntt_cpu::ntt_forward;
use ntt_cuda::emu::forward_batch_emulated;
use ntt_cuda::{CHUNK, LaunchPlan, NttTables};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn random_batch(n: usize, batch: usize, q: u64, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n * batch).map(|_| rng.gen_range(0..q)).collect()
}

fn cpu_reference(input: &[u64], t: &NttTables) -> Vec<u64> {
    let mut out = input.to_vec();
    for poly in out.chunks_mut(t.n) {
        ntt_forward(poly, t.q, &t.roots, &t.roots_shoup);
    }
    out
}

fn check(n: usize, batch: usize, seed: u64) {
    let t = NttTables::new(n);
    let input = random_batch(n, batch, t.q, seed);
    let expected = cpu_reference(&input, &t);
    let mut got = input.clone();
    forward_batch_emulated(&mut got, batch, &t);
    assert_eq!(
        got, expected,
        "emulated kernel != ntt_cpu::ntt_forward (N = {n}, batch = {batch})"
    );
}

#[test]
fn same_prime_as_task2() {
    assert_eq!(NttTables::new(1 << 16).q, 1152921504606584833);
}

#[test]
fn launch_plan_n65536_batch64() {
    let p = LaunchPlan::new(1 << 16, 64);
    assert_eq!(CHUNK, 2048);
    // 16 stages = 5 global-memory stages (t = 32768 .. 2048) + 11 shared-memory stages (t = 1024 .. 1).
    assert_eq!(p.global_stages.len(), 5);
    for (s, g) in p.global_stages.iter().enumerate() {
        assert_eq!(g.m, 1 << s);
        assert_eq!(g.log_t, 15 - s as u32);
        assert_eq!(g.grid, (128, 64, 1)); // 32768 butterflies / 256 threads, x 64 polys
        assert_eq!(g.block, (256, 1, 1));
        assert_eq!(g.shared_mem_bytes, 0);
    }
    let sh = &p.shared;
    assert_eq!(sh.m, 32); // first shared stage: m = N / CHUNK
    assert_eq!(sh.log_chunk, 11);
    assert_eq!(sh.grid, (32, 64, 1)); // 65536 / 2048 chunks, x 64 polys
    assert_eq!(sh.block, (1024, 1, 1)); // one butterfly per thread per stage
    assert_eq!(sh.shared_mem_bytes, 2048 * 8);
}

#[test]
fn launch_plan_small_n_is_shared_only() {
    let p = LaunchPlan::new(16, 3);
    assert!(p.global_stages.is_empty());
    assert_eq!(p.shared.m, 1);
    assert_eq!(p.shared.log_chunk, 4);
    assert_eq!(p.shared.grid, (1, 3, 1));
    assert_eq!(p.shared.block, (8, 1, 1));
    assert_eq!(p.shared.shared_mem_bytes, 16 * 8);
}

#[test]
fn emulation_matches_cpu_n65536_batch64() {
    check(1 << 16, 64, 1);
}

#[test]
fn emulation_matches_cpu_small_n() {
    // 2, 16, 256: shared-only (N < CHUNK); 2048: shared-only with N = CHUNK;
    // 4096, 8192: 1 and 2 global stages.
    for (k, &n) in [2usize, 16, 256, 2048, 4096, 8192].iter().enumerate() {
        check(n, 3, 100 + k as u64);
    }
}

#[test]
fn emulation_matches_cpu_extreme_inputs() {
    // All coefficients q - 1 (largest legal input) and all zero, to exercise lazy-reduction bounds.
    for n in [16usize, 1 << 16] {
        let t = NttTables::new(n);
        for fill in [t.q - 1, 0] {
            let input = vec![fill; 2 * n];
            let expected = cpu_reference(&input, &t);
            let mut got = input.clone();
            forward_batch_emulated(&mut got, 2, &t);
            assert_eq!(got, expected, "N = {n}, fill = {fill}");
        }
    }
}
