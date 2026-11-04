//! GPU kernel vs. the Task 2 CPU reference. `#[ignore]`d (needs an NVIDIA GPU + driver +
//! NVRTC); run with `cargo test --release -- --ignored`. Without a usable device each test
//! prints `SKIP: ...` and passes.

use ntt_cpu::ntt_forward;
use ntt_cuda::NttTables;
use ntt_cuda::gpu::{GpuNtt, probe_device};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// `true` (after printing why) when there is no usable CUDA device.
fn skip() -> bool {
    match probe_device() {
        Ok(_) => false,
        Err(e) => {
            eprintln!("SKIP: no usable CUDA device: {e}");
            true
        }
    }
}

fn check(n: usize, batch: usize, seed: u64) {
    let t = NttTables::new(n);
    let g = GpuNtt::new(0, &t).expect("CUDA device present but kernel setup failed");
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let input: Vec<u64> = (0..n * batch).map(|_| rng.gen_range(0..t.q)).collect();
    let mut expected = input.clone();
    for poly in expected.chunks_mut(n) {
        ntt_forward(poly, t.q, &t.roots, &t.roots_shoup);
    }
    let mut got = input;
    g.forward_batch(&mut got, batch)
        .expect("kernel launch failed");
    assert_eq!(
        got, expected,
        "GPU != ntt_cpu::ntt_forward (N = {n}, batch = {batch})"
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn gpu_matches_cpu_n65536_batch64() {
    if skip() {
        return;
    }
    check(1 << 16, 64, 1);
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn gpu_matches_cpu_small_n() {
    if skip() {
        return;
    }
    for (k, &n) in [2usize, 16, 256, 2048, 4096, 8192].iter().enumerate() {
        check(n, 3, 100 + k as u64);
    }
}
