//! CKKS operation micro-benchmarks for the M1 gate (Task 24): the same operations as the
//! SEAL 4.1.2 micro-bench `tools/seal-kat/bench`, on `seal_2020` and `n16_train`.
//!
//! Every operation runs on fresh ciphertexts at the top level with the preset scale:
//! `encrypt` (public key), `add`, `mul_relin_rescale` (`mul` = tensor + relinearize, then
//! `rescale`), `rotate1`, `rotate_x16` (16 separate `rotate`, steps 1..=16),
//! `hoisted_rotate_16` (one `hoisted_rotate` with steps 1..=16), and the `d = 64`
//! matrix-vector products `matvec_plain_d64` / `matvec_enc_d64` (BSGS 8 x 8: 14 rotations,
//! 7 of them hoisted; 64 plaintext / ciphertext products; 0 / 8 relinearizations; the
//! result is returned before rescaling).
//!
//! hekit has no threading (no rayon), so every number is single-threaded, like SEAL.
//!
//! `HEKIT_BENCH_PRESET=seal_2020` or `=n16_train` restricts the run to one preset (key
//! generation for n16_train takes a while and about 10 GB of memory: 21 Galois keys of
//! 302 MB each, 64 encrypted diagonals).
//!
//! Run: `cargo bench -p hekit-bench --bench ckks_ops`; the medians are read from
//! `target/criterion/<preset>/<op>/new/estimates.json`.
use std::sync::Arc;
use std::time::Duration;

use criterion::{Criterion, SamplingMode, black_box, criterion_group, criterion_main};
use hekit_ckks::{
    Ciphertext, CkksParams, Diagonals, Encoder, Encryptor, Evaluator, KeyGenerator, Plaintext,
    bsgs_rotation_steps, diagonals_from_matrix,
};
use hekit_core::Rng;
use ndarray::Array2;

/// Matrix dimension of the matvec benchmarks.
const D: usize = 64;
/// Number of rotations in `rotate_x16` / `hoisted_rotate_16`.
const ROTS: i32 = 16;

/// Uniform values in `[-1, 1)`.
fn rand_vals(r: &mut Rng, len: usize) -> Vec<f64> {
    (0..len)
        .map(|_| 2.0 * (r.next_u64() >> 11) as f64 * 2f64.powi(-53) - 1.0)
        .collect()
}

fn wanted(name: &str) -> bool {
    match std::env::var("HEKIT_BENCH_PRESET") {
        Ok(v) if !v.is_empty() && v != "all" => v == name,
        _ => true,
    }
}

fn bench_preset(c: &mut Criterion, name: &str, params: CkksParams) {
    if !wanted(name) {
        return;
    }
    let params = Arc::new(params);
    let (slots, top, scale) = (params.slots(), params.max_level(), params.scale());
    let mut steps: Vec<i32> = (1..=ROTS).collect();
    for s in bsgs_rotation_steps(D, 1) {
        if !steps.contains(&s) {
            steps.push(s);
        }
    }
    let t0 = std::time::Instant::now();
    let mut kg = KeyGenerator::new(Arc::clone(&params), Rng::from_seed([24; 32])).unwrap();
    let pk = kg.public_key();
    let rk = kg.relin_key();
    let gks = kg.galois_keys(&steps);
    let enc = Encoder::new(Arc::clone(&params));
    let encryptor = Encryptor::new(Arc::clone(&params), pk);
    let eval = Evaluator::new(Arc::clone(&params), Some(rk), Some(gks));
    let mut r = Rng::from_seed([25; 32]);
    let x = rand_vals(&mut r, slots);
    let y = rand_vals(&mut r, slots);
    let pt_x = enc.encode(&x, scale, top).unwrap();
    let cx = encryptor.encrypt(&pt_x, &mut r);
    let cy = encryptor.encrypt(&enc.encode(&y, scale, top).unwrap(), &mut r);
    // matvec: random 64 x 64 matrix, x replicated with period 64.
    let w = Array2::from_shape_vec((D, D), rand_vals(&mut r, D * D)).unwrap();
    let diags = diagonals_from_matrix(&w, slots, 1);
    let v = rand_vals(&mut r, D);
    let vr: Vec<f64> = (0..slots).map(|j| v[j % D]).collect();
    let cv = encryptor.encrypt(&enc.encode(&vr, scale, top).unwrap(), &mut r);
    let pdiag = Diagonals::<Plaintext>::encode(&enc, &diags, 1, scale, top).unwrap();
    let cdiag =
        Diagonals::<Ciphertext>::encrypt(&enc, &encryptor, &diags, 1, scale, top, &mut r).unwrap();
    eprintln!(
        "{name}: N = 2^{}, level {top}, dnum {}, alpha {}, {} special primes, {} Galois keys; \
         setup {:.1} s",
        params.log_n(),
        params.dnum(),
        params.alpha(),
        params.p().len(),
        steps.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut g = c.benchmark_group(name);
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(1));
    // Flat sampling: for the slow operations criterion then runs ~1 iteration per sample
    // instead of 1 + 2 + ... + 10.
    g.sampling_mode(SamplingMode::Flat);
    g.measurement_time(Duration::from_secs(3));
    let mut er = Rng::from_seed([26; 32]);
    g.bench_function("encrypt", |b| {
        b.iter(|| encryptor.encrypt(black_box(&pt_x), &mut er))
    });
    g.bench_function("add", |b| {
        b.iter(|| eval.add(black_box(&cx), black_box(&cy)).unwrap())
    });
    g.bench_function("mul_relin_rescale", |b| {
        b.iter(|| {
            eval.rescale(&eval.mul(black_box(&cx), black_box(&cy)).unwrap())
                .unwrap()
        })
    });
    g.bench_function("rotate1", |b| {
        b.iter(|| eval.rotate(black_box(&cx), 1).unwrap())
    });
    let rot_steps: Vec<i32> = (1..=ROTS).collect();
    g.bench_function("rotate_x16", |b| {
        b.iter(|| {
            rot_steps
                .iter()
                .map(|&s| eval.rotate(black_box(&cx), s).unwrap())
                .collect::<Vec<_>>()
        })
    });
    g.bench_function("hoisted_rotate_16", |b| {
        b.iter(|| eval.hoisted_rotate(black_box(&cx), &rot_steps).unwrap())
    });
    g.bench_function("matvec_plain_d64", |b| {
        b.iter(|| {
            eval.matvec_plain(black_box(&pdiag), black_box(&cv))
                .unwrap()
        })
    });
    g.bench_function("matvec_enc_d64", |b| {
        b.iter(|| eval.matvec_enc(black_box(&cdiag), black_box(&cv)).unwrap())
    });
    g.finish();
}

fn bench(c: &mut Criterion) {
    bench_preset(c, "seal_2020", CkksParams::seal_2020());
    bench_preset(c, "n16_train", CkksParams::n16_train());
}

criterion_group!(benches, bench);
criterion_main!(benches);
