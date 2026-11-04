//! Forward / inverse negacyclic NTT, N = 2^16, 60-bit prime.
use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use hekit_core::{Modulus, NttTable, primes::ntt_primes};

fn bench(c: &mut Criterion) {
    let n = 1usize << 16;
    let q = ntt_primes(60, n, 1).unwrap()[0];
    let table = NttTable::new(n, Modulus::new(q).unwrap()).unwrap();
    eprintln!("q = {q}, N = {n}");
    let mut s = 1u64;
    let base: Vec<u64> = (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 4) % q
        })
        .collect();
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
