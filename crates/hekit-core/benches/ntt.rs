//! Forward / inverse negacyclic NTT, N = 2^16, 60-bit prime.
use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use hekit_core::{Modulus, NttTable, primes::ntt_primes};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

fn bench(c: &mut Criterion) {
    let n = 1usize << 16;
    let q = ntt_primes(60, n, 1).unwrap()[0];
    let table = NttTable::new(n, Modulus::new(q).unwrap()).unwrap();
    eprintln!("q = {q}, N = {n}");
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let base: Vec<u64> = (0..n).map(|_| rng.next_u64() % q).collect();
    let mut g = c.benchmark_group("ntt_n65536_q60");
    g.bench_function("forward", |b| {
        b.iter_batched_ref(
            || base.clone(),
            |a| table.forward(black_box(a)),
            BatchSize::LargeInput,
        )
    });
    g.bench_function("inverse", |b| {
        b.iter_batched_ref(
            || base.clone(),
            |a| table.inverse(black_box(a)),
            BatchSize::LargeInput,
        )
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
