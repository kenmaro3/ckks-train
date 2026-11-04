//! Times the batched forward NTT on the GPU with CUDA events.
//!
//! `cargo run --release --bin bench [-- <log2 N = 16> <batch = 64> <reps = 50> <warmup = 5>]`
//!
//! Data stays on the device; only the kernels (5 global-stage launches + 1 shared-memory launch
//! for N = 2^16) are inside the timed region, no host<->device copies. Each rep is timed with its
//! own event pair; the output is checked against `ntt_cpu` once before timing.

use std::process::ExitCode;

use ntt_cuda::gpu::{EventTimer, GpuNtt, probe_device};
use ntt_cuda::{LaunchPlan, NttTables};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn arg(i: usize, default: usize) -> usize {
    std::env::args()
        .nth(i)
        .map(|s| s.parse().expect("arguments must be integers"))
        .unwrap_or(default)
}

fn main() -> ExitCode {
    let (log_n, batch, reps, warmup) = (arg(1, 16), arg(2, 64), arg(3, 50), arg(4, 5));
    assert!(reps >= 20, "use at least 20 timed repetitions");
    if let Err(e) = probe_device() {
        eprintln!("no usable CUDA device: {e}");
        return ExitCode::FAILURE;
    }
    let n = 1usize << log_n;
    let t = NttTables::new(n);
    let g = GpuNtt::new(0, &t).expect("kernel setup failed");
    let plan = LaunchPlan::new(n, batch);

    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let input: Vec<u64> = (0..n * batch).map(|_| rng.gen_range(0..t.q)).collect();

    // Correctness gate.
    let mut expected = input.clone();
    for poly in expected.chunks_mut(n) {
        ntt_cpu::ntt_forward(poly, t.q, &t.roots, &t.roots_shoup);
    }
    let mut got = input.clone();
    g.forward_batch(&mut got, batch).expect("launch failed");
    let agree = got == expected;

    // Timing (in place; outputs in [0, q) are valid inputs for the next rep).
    let dev = g.device();
    let mut d_a = dev.htod_sync_copy(&input).expect("upload failed");
    for _ in 0..warmup {
        g.forward_batch_device(&mut d_a, batch)
            .expect("launch failed");
    }
    dev.synchronize().expect("sync failed");
    let timer = EventTimer::new(dev).expect("event creation failed");
    let mut ms: Vec<f64> = (0..reps)
        .map(|_| {
            timer.start().unwrap();
            g.forward_batch_device(&mut d_a, batch).unwrap();
            timer.stop_ms().unwrap() as f64
        })
        .collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = ms[reps / 2];
    let mean = ms.iter().sum::<f64>() / reps as f64;
    let per_poly_us = |batch_ms: f64| batch_ms * 1e3 / batch as f64;

    println!("device            : {}", g.describe().unwrap());
    println!("N, q              : 2^{log_n} = {n}, {}", t.q);
    println!("batch, reps, warm : {batch}, {reps}, {warmup}");
    println!(
        "launches / batch  : {} global-stage + 1 shared (chunk {} coeffs)",
        plan.global_stages.len(),
        1usize << plan.shared.log_chunk
    );
    println!(
        "CPU agreement     : {}",
        if agree { "OK (bit-exact)" } else { "MISMATCH" }
    );
    println!(
        "batch time (ms)   : median {median:.4}, min {:.4}, mean {mean:.4}, max {:.4}",
        ms[0],
        ms[reps - 1]
    );
    println!(
        "per polynomial    : median {:.3} us, min {:.3} us, mean {:.3} us",
        per_poly_us(median),
        per_poly_us(ms[0]),
        per_poly_us(mean)
    );
    if agree {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
