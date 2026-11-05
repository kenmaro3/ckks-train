//! R-47-3: `matvec_enc` (d = 64, n16_train, keys capped at level 5, inputs at level 5) four
//! times, one after the other on one stream, against `CudaEvaluator::matvec_enc_batch` of the
//! same four products on devices with 2 and 4 streams (Task 47 review I2), under two memory
//! pool release thresholds: the default (`CudaDevice::DEFAULT_POOL_RELEASE_THRESHOLD`,
//! 256 MiB) and `u64::MAX` (the pool keeps everything; re-review N1). Prints, per variant,
//! the median / min / standard deviation of `HEKIT_BATCH_REPS` (default 30) timed
//! repetitions after 3 warm-up runs, each ending with `CudaDevice::sync`, and checks that every
//! batch result equals the sequential one bit for bit. `HEKIT_POOL_THRESHOLDS=default` or
//! `=max` runs one threshold only. Every device trims its pool (`trim_pool(0)`) before it is
//! dropped.
//!
//! Run on a CUDA machine:
//! `cargo run -p hekit-cuda --features cuda --release --example batch_streams`
//! (about 1.0 GiB of GPU memory at the peak, 5 GB of host memory for the CPU keys).
use std::sync::Arc;
use std::time::Instant;

use hekit_ckks::{
    Ciphertext, CkksParams, Diagonals, Encoder, Encryptor, EvalConfig, GaloisKey, HeOps,
    KeyGenerator, bsgs_rotation_steps,
};
use hekit_core::Rng;
use hekit_cuda::{CudaCiphertext, CudaDevice, CudaEvaluator};

const D: usize = 64;
const LEVEL: usize = 5;
const JOBS: usize = 4;
const MIB: f64 = (1u64 << 20) as f64;

fn rand_vals(r: &mut Rng, len: usize) -> Vec<f64> {
    (0..len)
        .map(|_| 2.0 * (r.next_u64() >> 11) as f64 * 2f64.powi(-53) - 1.0)
        .collect()
}

fn stats(mut t: Vec<f64>) -> (f64, f64, f64) {
    t.sort_by(f64::total_cmp);
    let mean = t.iter().sum::<f64>() / t.len() as f64;
    let var = t.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / t.len() as f64;
    (t[t.len() / 2], t[0], var.sqrt())
}

fn time(reps: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
    for _ in 0..3 {
        f();
    }
    let t: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    stats(t)
}

fn main() {
    let reps: usize = std::env::var("HEKIT_BATCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let params = Arc::new(CkksParams::n16_train());
    let slots = params.slots();
    let steps = bsgs_rotation_steps(D, 1);
    let t0 = Instant::now();
    let mut kg = KeyGenerator::new(Arc::clone(&params), Rng::from_seed([73; 32])).unwrap();
    let rk = kg.relin_key();
    let gks = kg.galois_keys(&steps);
    let enc = Encoder::new(Arc::clone(&params));
    let encryptor = Encryptor::new(Arc::clone(&params), kg.public_key());
    let mut r = Rng::from_seed([74; 32]);
    let x = encryptor.encrypt(
        &enc.encode(&rand_vals(&mut r, slots), params.scale(), LEVEL)
            .unwrap(),
        &mut r,
    );
    let diags: Vec<Vec<f64>> = (0..D).map(|_| rand_vals(&mut r, slots)).collect();
    let diag = Diagonals::<Ciphertext>::encrypt(
        &enc,
        &encryptor,
        &diags,
        1,
        params.scale(),
        LEVEL,
        &mut r,
    )
    .unwrap();
    let keys: Vec<&GaloisKey> = steps.iter().map(|&s| gks.get(s).unwrap()).collect();
    println!(
        "batch_streams: n16_train, d = {D}, level {LEVEL}, {JOBS} products, {reps} reps; CPU \
         setup {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    let thresholds: Vec<(&str, u64)> = match std::env::var("HEKIT_POOL_THRESHOLDS").as_deref() {
        Ok("default") => vec![("default", CudaDevice::DEFAULT_POOL_RELEASE_THRESHOLD)],
        Ok("max") => vec![("max", u64::MAX)],
        _ => vec![
            ("default", CudaDevice::DEFAULT_POOL_RELEASE_THRESHOLD),
            ("max", u64::MAX),
        ],
    };
    let mut reference: Option<Vec<Ciphertext>> = None;
    for &(tname, threshold) in &thresholds {
        println!("pool release threshold: {tname} ({threshold} bytes)");
        for streams in [1usize, 2, 4] {
            let dev = CudaDevice::new(0, &params, streams).expect("CUDA device");
            dev.set_pool_release_threshold(threshold)
                .expect("pool threshold");
            let ev = CudaEvaluator::with_keys(
                &dev,
                Arc::clone(&params),
                Some(&rk),
                &keys,
                EvalConfig::default(),
                LEVEL,
            )
            .expect("keys fit");
            let ux = ev.upload_ct(&x).unwrap();
            let ud = diag.try_map(|c| ev.upload_ct(c)).unwrap();
            let xs: Vec<CudaCiphertext<'_>> = vec![ux.clone(); JOBS];
            let ds = vec![ud.clone(); JOBS];
            if streams == 1 {
                let (med, min, sd) = time(reps, || {
                    for _ in 0..JOBS {
                        let _ = ev.matvec_enc(&ud, &ux).unwrap();
                    }
                    dev.sync().unwrap();
                });
                println!(
                    "[{tname}] sequential x{JOBS}, 1 stream:   median {med:8.3} ms  min {min:8.3}  sd {sd:6.3}  \
                 ({:.3} ms per product)",
                    med / JOBS as f64
                );
                let one = ev.matvec_enc(&ud, &ux).unwrap().download().unwrap();
                reference = Some(vec![one; JOBS]);
            } else {
                let (med, min, sd) = time(reps, || {
                    let _ = ev.matvec_enc_batch(&ds, &xs).unwrap();
                    dev.sync().unwrap();
                });
                println!(
                    "[{tname}] batch x{JOBS}, {streams} streams:    median {med:8.3} ms  min {min:8.3}  sd {sd:6.3}"
                );
                let got = ev.matvec_enc_batch(&ds, &xs).unwrap();
                for (g, w) in got.iter().zip(reference.as_ref().unwrap()) {
                    let g = g.download().unwrap();
                    assert!(
                        g.polys()
                            .iter()
                            .zip(w.polys())
                            .all(|(a, b)| a.limbs() == b.limbs()),
                        "batch result differs from the sequential one"
                    );
                }
            }
            println!(
                "  device: {}, available after: {:.0} MiB",
                dev.describe(),
                dev.available_memory().unwrap() as f64 / MIB
            );
            drop((xs, ds, ux, ud, ev));
            dev.trim_pool(0).expect("trim");
        }
    }
    // Leave the process-wide default for anything after this example.
    if let Ok(dev) = CudaDevice::new(0, &params, 1) {
        let _ = dev.set_pool_release_threshold(CudaDevice::DEFAULT_POOL_RELEASE_THRESHOLD);
        let _ = dev.trim_pool(0);
    }
}
