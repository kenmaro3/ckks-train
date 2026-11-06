//! One n16_boot bootstrap (Task 38, Step 5) on 1 thread and on N threads of a rayon pool.
//!
//! Needs the `parallel` feature and about 14 GB of memory (keys 10.3 GB, encoded diagonals
//! about 3 GB), and takes about half a minute of setup:
//!
//! `cargo bench -p ckks-train-ckks --features parallel --bench bootstrap`
//!
//! The thread counts come from `HEKIT_BOOT_THREADS` (comma separated, default
//! `1,<available_parallelism>`: on the 12-core Mac the brief's "32 threads" is run as 12).
//! Each count runs in its own `rayon::ThreadPool` (`install`), so the parallel sections of
//! key switching, hoisted rotations, BSGS and the two EvalMods use exactly that many threads.
//! The result is bit-identical for every thread count; the bench checks the output level
//! and scale once per pool.
use std::sync::Arc;
use std::time::Duration;

use ckks_train_ckks::bootstrap::{BootParams, Bootstrapper};
use ckks_train_ckks::{CkksParams, Encoder, Encryptor, Evaluator, KeyGenerator};
use ckks_train_core::Rng;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn thread_counts() -> Vec<usize> {
    let all = std::thread::available_parallelism().map_or(1, |n| n.get());
    match std::env::var("HEKIT_BOOT_THREADS") {
        Ok(s) => s
            .split(',')
            .map(|t| {
                t.trim()
                    .parse()
                    .expect("HEKIT_BOOT_THREADS: comma-separated counts")
            })
            .collect(),
        Err(_) => vec![1, all],
    }
}

fn bench(c: &mut Criterion) {
    let params = Arc::new(CkksParams::n16_boot());
    let bp = BootParams::default_n16();
    let mut kg = KeyGenerator::new(Arc::clone(&params), Rng::from_seed([38; 32])).unwrap();
    let enc = Encoder::new(Arc::clone(&params));
    let encryptor = Encryptor::new(Arc::clone(&params), kg.public_key());
    let ev = Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), None);
    let t = std::time::Instant::now();
    let bs = Bootstrapper::new(Arc::clone(&params), bp, &enc, &mut kg).unwrap();
    eprintln!(
        "n16_boot bootstrapper: {:.1} s setup, {} key switches per boot, {} Galois keys, \
         {:.2} GB of keys",
        t.elapsed().as_secs_f64(),
        bs.key_switches_per_boot(),
        bs.keys().galois().len(),
        bs.keys().bytes() as f64 / 1e9
    );
    let v: Vec<f64> = (0..params.slots())
        .map(|i| ((i % 197) as f64 / 98.0) - 1.0)
        .collect();
    let pt = enc.encode(&v, params.scale(), params.max_level()).unwrap();
    let ct = encryptor.encrypt(&pt, &mut Rng::from_seed([39; 32]));
    let ct = ev.mod_switch_to_level(&ct, 0).unwrap();

    let mut g = c.benchmark_group("bootstrap_n16_boot");
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(1));
    g.measurement_time(Duration::from_secs(60));
    for threads in thread_counts() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let out = pool.install(|| bs.bootstrap(&ev, &ct).unwrap());
        assert_eq!(
            (out.level(), out.scale()),
            (bs.output_level(), params.scale())
        );
        g.bench_function(format!("{threads}_threads"), |b| {
            b.iter(|| pool.install(|| bs.bootstrap(&ev, black_box(&ct)).unwrap()))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
