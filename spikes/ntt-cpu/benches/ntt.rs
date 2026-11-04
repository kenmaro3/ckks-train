use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ntt_cpu::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn bench(c: &mut Criterion) {
    let n = 1usize << 16;
    let q = gen_ntt_prime(n, 60);
    let psi = primitive_2n_root(n, q);
    let roots = build_roots(n, q, psi);
    let roots_shoup = shoup_table(&roots, q);
    let inv_roots = build_inv_roots(n, q, psi);
    let inv_roots_shoup = shoup_table(&inv_roots, q);
    let n_inv = pow_mod(n as u64, q - 2, q);
    let n_inv_shoup = shoup_table(&[n_inv], q)[0];
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let base: Vec<u64> = (0..n).map(|_| rng.gen_range(0..q)).collect();
    eprintln!("q = {q}, N = {n}");

    // Input is cloned in setup (not timed); values stay in [0, q) for each run.
    let mut g = c.benchmark_group("ntt_n65536_q60");
    g.bench_function("forward", |b| {
        b.iter_batched_ref(
            || base.clone(),
            |a| ntt_forward(black_box(a), q, &roots, &roots_shoup),
            criterion::BatchSize::LargeInput,
        )
    });
    g.bench_function("inverse", |b| {
        b.iter_batched_ref(
            || base.clone(),
            |a| {
                ntt_inverse(
                    black_box(a),
                    q,
                    &inv_roots,
                    &inv_roots_shoup,
                    n_inv,
                    n_inv_shoup,
                )
            },
            criterion::BatchSize::LargeInput,
        )
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
