//! 32 rotations of one ciphertext (seal_2020, top level): `rotate` x 32 vs one
//! `hoisted_rotate` with the same 32 steps (Task 18).
use std::sync::Arc;

use ckks_train_ckks::{CkksParams, Encoder, Encryptor, Evaluator, KeyGenerator};
use ckks_train_core::Rng;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench(c: &mut Criterion) {
    let params = Arc::new(CkksParams::seal_2020());
    let steps: Vec<i32> = (1..=32).collect();
    let mut kg = KeyGenerator::new(Arc::clone(&params), Rng::from_seed([1; 32])).unwrap();
    let encryptor = Encryptor::new(Arc::clone(&params), kg.public_key());
    let eval = Evaluator::new(Arc::clone(&params), None, Some(kg.galois_keys(&steps)));
    let enc = Encoder::new(Arc::clone(&params));
    let v: Vec<f64> = (0..params.slots())
        .map(|i| (i % 97) as f64 / 97.0)
        .collect();
    let top = params.max_level();
    let pt = enc.encode(&v, params.scale(), top).unwrap();
    let ct = encryptor.encrypt(&pt, &mut Rng::from_seed([2; 32]));
    eprintln!(
        "seal_2020: N = 2^{}, level {top}, dnum {}, alpha {}, {} steps",
        params.log_n(),
        params.dnum(),
        params.alpha(),
        steps.len()
    );
    let mut g = c.benchmark_group("rotate32_seal2020_top");
    g.sample_size(10);
    g.bench_function("rotate_x32", |b| {
        b.iter(|| {
            steps
                .iter()
                .map(|&s| eval.rotate(black_box(&ct), s).unwrap())
                .collect::<Vec<_>>()
        })
    });
    g.bench_function("hoisted_rotate_32", |b| {
        b.iter(|| eval.hoisted_rotate(black_box(&ct), &steps).unwrap())
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
