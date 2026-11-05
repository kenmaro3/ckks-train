//! GPU vs CPU micro-benchmarks on `n16_train` (Task 47, Step 5; feature `cuda`).
//!
//! The operations: `add`, `relinearize` (one key switch; the closest HEonGPU comparison),
//! `mul_relin_rescale` (`mul` = tensor + relinearize, then `rescale`), `rotate1`,
//! `hoisted_rotate_16` (one hoisted rotation, steps 1..=16) and `matvec_enc_d64` (BSGS 8 x 8
//! with encrypted diagonals: 64 tensors, 8 relins, 14 rotations, unrescaled), on fresh
//! ciphertexts. Each is measured
//!
//! * on the GPU (`gpu/<op>`, `CudaEvaluator`, default flags): each iteration queues the
//!   operation and waits for the device (`CudaDevice::sync`), so the time is the operation's
//!   latency, not its queueing time;
//! * on the CPU (`cpu_<t>t/<op>`, `Evaluator` with the `parallel` feature) in a rayon pool of
//!   `t` threads for every `t` of `HEKIT_BENCH_THREADS` (comma separated; default
//!   `1,<available cores, at most 12>`).
//!
//! **GPU memory.** Every GPU operation first trims the device's memory pool
//! (`CudaDevice::trim_pool(0)`, the previous block's memory goes back to the driver) and
//! checks that its keys, inputs and working set fit in the available device memory
//! (`CudaDevice::available_memory`), and otherwise prints `<op>: skipped: needs X MiB (Y MiB
//! available)`. The pool's release threshold is the device default (256 MiB; below the keys
//! and inputs in use at n16_train, so it acts almost like threshold 0 here). A full n16_train
//! key is 288 MiB, so `hoisted_rotate_16` (16 keys) and
//! `matvec_enc_d64` (15 keys and 64 top-level encrypted diagonals) do not fit next to the
//! server's other tenant. They are also measured **at a lower level with keys capped at that
//! level** (`..._L9`, `..._L5`: inputs at level 9 or 5, keys uploaded for levels `<= 9` / `<= 5`,
//! 42 / 20 MiB each): that is a faithful measurement of the operation **at that level** (a
//! capped key computes exactly what the full key computes there), and the CPU is measured at
//! the same level under the same label. `matvec_enc_batch4_L5` runs 4 such products (shared
//! operands) on 4 streams (`CudaEvaluator::matvec_enc_batch`); `matvec_enc_x4_L5` runs the
//! same 4 products one after the other on one stream.
//!
//! Host memory: about 10 GB (21 full Galois keys of 302 MB on the CPU, 64 encrypted diagonals
//! at the top level). Run on quadruped-gpu (commands in the Task 47 report):
//! `HEKIT_BENCH_THREADS=1,12 cargo bench -p hekit-bench --features cuda --bench cuda_ops`;
//! the medians are read from `target/criterion/cuda_ops_n16_train/<id>/new/estimates.json`.
use std::sync::Arc;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, Criterion, SamplingMode, black_box, criterion_group, criterion_main,
};
use hekit_ckks::{
    Ciphertext, CkksParams, Diagonals, Encoder, Encryptor, EvalConfig, Evaluator, GaloisKey, HeOps,
    KeyGenerator, bsgs_rotation_steps,
};
use hekit_core::Rng;
use hekit_cuda::matvec::matvec_enc_workspace_limbs;
use hekit_cuda::testing::{ksk_bytes, table_bytes};
use hekit_cuda::{CudaDevice, CudaEvaluator};

const D: usize = 64;
const ROTS: i32 = 16;
const MIB: f64 = (1u64 << 20) as f64;

fn rand_vals(r: &mut Rng, len: usize) -> Vec<f64> {
    (0..len)
        .map(|_| 2.0 * (r.next_u64() >> 11) as f64 * 2f64.powi(-53) - 1.0)
        .collect()
}

/// `HEKIT_CUDA_BENCH_SMOKE=1`: a small parameter set (N = 2^12) to check the plumbing (e.g.
/// on a machine without a GPU, with `-- --test`); results go to `cuda_ops_smoke`.
fn smoke() -> bool {
    std::env::var("HEKIT_CUDA_BENCH_SMOKE").is_ok_and(|v| v == "1")
}

fn group_name() -> &'static str {
    if smoke() {
        "cuda_ops_smoke"
    } else {
        "cuda_ops_n16_train"
    }
}

fn thread_counts() -> Vec<usize> {
    let all = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(12);
    match std::env::var("HEKIT_BENCH_THREADS") {
        Ok(v) if !v.is_empty() => v
            .split(',')
            .map(|t| {
                t.trim()
                    .parse()
                    .expect("HEKIT_BENCH_THREADS: comma-separated counts")
            })
            .collect(),
        _ if all > 1 => vec![1, all],
        _ => vec![1],
    }
}

/// Host-side inputs shared by the CPU and the GPU runs.
struct Inputs {
    params: Arc<CkksParams>,
    cpu: Evaluator,
    a: Ciphertext,
    b: Ciphertext,
    t: Ciphertext,
    v: Ciphertext,
    diag: Diagonals<Ciphertext>,
    rot_steps: Vec<i32>,
    mv_steps: Vec<i32>,
}

impl Inputs {
    fn new() -> Inputs {
        let t0 = std::time::Instant::now();
        let params = Arc::new(if smoke() {
            // N = 2^12, 12 primes of 42 bits in 3 digits of 4, 4 special primes (insecure):
            // the n16_train structure at a size for a plumbing check.
            let mut q = vec![58];
            q.extend([42; 11]);
            CkksParams::new(12, &q, &[58; 4], 42, 3).unwrap()
        } else {
            CkksParams::n16_train()
        });
        let (slots, top, scale) = (params.slots(), params.max_level(), params.scale());
        let rot_steps: Vec<i32> = (1..=ROTS).collect();
        let mv_steps = bsgs_rotation_steps(D, 1);
        let mut steps = rot_steps.clone();
        steps.extend(mv_steps.iter().filter(|s| !rot_steps.contains(s)));
        let mut kg = KeyGenerator::new(Arc::clone(&params), Rng::from_seed([47; 32])).unwrap();
        let pk = kg.public_key();
        let rk = kg.relin_key();
        let gks = kg.galois_keys(&steps);
        let enc = Encoder::new(Arc::clone(&params));
        let encryptor = Encryptor::new(Arc::clone(&params), pk);
        let cpu = Evaluator::new(Arc::clone(&params), Some(rk), Some(gks));
        let mut r = Rng::from_seed([48; 32]);
        let ct = |r: &mut Rng| {
            let pt = enc.encode(&rand_vals(r, slots), scale, top).unwrap();
            encryptor.encrypt(&pt, r)
        };
        let (a, b, v) = (ct(&mut r), ct(&mut r), ct(&mut r));
        let diags: Vec<Vec<f64>> = (0..D).map(|_| rand_vals(&mut r, slots)).collect();
        let diag =
            Diagonals::<Ciphertext>::encrypt(&enc, &encryptor, &diags, 1, scale, top, &mut r)
                .unwrap();
        let t = cpu.mul_no_relin(&a, &b).unwrap();
        eprintln!(
            "{}: N = 2^{}, level {top}, dnum {}, alpha {}, {} special primes, {} Galois \
             keys; setup {:.1} s",
            group_name(),
            params.log_n(),
            params.dnum(),
            params.alpha(),
            params.p().len(),
            steps.len(),
            t0.elapsed().as_secs_f64()
        );
        Inputs {
            params,
            cpu,
            a,
            b,
            t,
            v,
            diag,
            rot_steps,
            mv_steps,
        }
    }

    fn at(&self, ct: &Ciphertext, level: usize) -> Ciphertext {
        self.cpu.mod_switch_to_level(ct, level).unwrap()
    }

    fn diag_at(&self, level: usize) -> Diagonals<Ciphertext> {
        self.diag.mod_switch_to_level(&self.cpu, level).unwrap()
    }
}

fn group(c: &mut Criterion) -> BenchmarkGroup<'_, WallTime> {
    let mut g = c.benchmark_group(group_name());
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(1));
    g.sampling_mode(SamplingMode::Flat);
    g.measurement_time(Duration::from_secs(3));
    g
}

// ---------------------------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------------------------

fn bench_cpu(c: &mut Criterion, x: &Inputs) {
    let ev = &x.cpu;
    let (a9, v9, v5) = (x.at(&x.a, 9), x.at(&x.v, 9), x.at(&x.v, 5));
    let (d9, d5) = (x.diag_at(9), x.diag_at(5));
    for threads in thread_counts() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let p = |name: &str| format!("cpu_{threads}t/{name}");
        let mut g = group(c);
        g.bench_function(p("add"), |b| {
            b.iter(|| pool.install(|| ev.add(black_box(&x.a), black_box(&x.b)).unwrap()))
        });
        g.bench_function(p("relinearize"), |b| {
            b.iter(|| pool.install(|| ev.relinearize(black_box(&x.t)).unwrap()))
        });
        g.bench_function(p("mul_relin_rescale"), |b| {
            b.iter(|| pool.install(|| ev.rescale(&ev.mul(&x.a, &x.b).unwrap()).unwrap()))
        });
        g.bench_function(p("rotate1"), |b| {
            b.iter(|| pool.install(|| ev.rotate(black_box(&x.a), 1).unwrap()))
        });
        g.bench_function(p("hoisted_rotate_16"), |b| {
            b.iter(|| pool.install(|| ev.hoisted_rotate(black_box(&x.a), &x.rot_steps).unwrap()))
        });
        g.bench_function(p("hoisted_rotate_16_L9"), |b| {
            b.iter(|| pool.install(|| ev.hoisted_rotate(black_box(&a9), &x.rot_steps).unwrap()))
        });
        g.bench_function(p("matvec_enc_d64"), |b| {
            b.iter(|| pool.install(|| ev.matvec_enc(black_box(&x.diag), &x.v).unwrap()))
        });
        g.bench_function(p("matvec_enc_d64_L9"), |b| {
            b.iter(|| pool.install(|| ev.matvec_enc(black_box(&d9), &v9).unwrap()))
        });
        g.bench_function(p("matvec_enc_d64_L5"), |b| {
            b.iter(|| pool.install(|| ev.matvec_enc(black_box(&d5), &v5).unwrap()))
        });
        g.finish();
    }
}

// ---------------------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------------------

/// `true` if `need` bytes are free on the device, else prints the skip line.
fn fits(dev: &CudaDevice, label: &str, need: usize) -> bool {
    // The previous block's keys and temporaries are dropped by now: give the memory the pool
    // kept back to the driver (shared GPU, re-review N1), then count what is available
    // (driver free memory plus the pool's unused reserve).
    dev.trim_pool(0).unwrap();
    let free = dev.available_memory().unwrap_or(0);
    if free >= need {
        eprintln!(
            "gpu/{label}: needs {:.0} MiB, {:.0} MiB available",
            need as f64 / MIB,
            free as f64 / MIB
        );
        true
    } else {
        println!(
            "gpu/{label}: skipped: needs {:.0} MiB ({:.0} MiB available)",
            need as f64 / MIB,
            free as f64 / MIB
        );
        false
    }
}

/// What runs besides the inputs: one key switch, or `n` matvec working sets (d = 64).
#[derive(Clone, Copy)]
enum Work {
    None,
    KeySwitch,
    Matvec(usize),
}

/// Device bytes of `keys` keys capped at `key_level`, `polys` polynomials at `level`, the
/// working set `work` at `level` (a key switch: two digit sets and two accumulators over
/// `Q_l P` and 4 polynomials over `Q_l`; a matvec: `matvec_enc_workspace_limbs`), plus
/// 40 MiB of slack.
fn need(
    p: &CkksParams,
    keys: usize,
    key_level: usize,
    level: usize,
    polys: usize,
    work: Work,
) -> usize {
    let limb = 8 * p.n();
    let (q, qp) = (level + 1, level + 1 + p.p().len());
    let digits = hekit_ckks::keyswitch::digit_ranges(p, level).len();
    let ws = match work {
        Work::None => 0,
        Work::KeySwitch => (2 * digits + 2) * qp + 4 * q,
        Work::Matvec(n) => n * matvec_enc_workspace_limbs(p, level, D),
    };
    keys * ksk_bytes(p, key_level).unwrap() + (polys * q + ws) * limb + (40 << 20)
}

fn gpu_eval<'d>(
    dev: &'d CudaDevice,
    x: &Inputs,
    relin: bool,
    steps: &[i32],
    key_level: usize,
) -> CudaEvaluator<'d> {
    let gks = x.cpu.galois_keys().unwrap();
    let keys: Vec<&GaloisKey> = steps.iter().map(|&s| gks.get(s).unwrap()).collect();
    CudaEvaluator::with_keys(
        dev,
        Arc::clone(&x.params),
        if relin { x.cpu.relin_key() } else { None },
        &keys,
        EvalConfig::default(),
        key_level,
    )
    .unwrap()
}

fn bench_gpu(c: &mut Criterion, x: &Inputs) {
    let p = &x.params;
    let top = p.max_level();
    let dev = match CudaDevice::new(0, p, 4) {
        Ok(d) => d,
        Err(e) => {
            println!("gpu: skipped (no usable CUDA device: {e})");
            return;
        }
    };
    let (free, total) = dev.mem_info().unwrap();
    eprintln!(
        "gpu: {}, {:.0} MiB free of {:.0} MiB, tables {:.0} MiB",
        dev.describe(),
        free as f64 / MIB,
        total as f64 / MIB,
        table_bytes(p) as f64 / MIB
    );
    let sync = || dev.sync().unwrap();
    let mut g = group(c);

    // add (no keys).
    if fits(&dev, "add", need(p, 0, 0, top, 8, Work::None)) {
        let ev = gpu_eval(&dev, x, false, &[], top);
        let (a, b) = (ev.upload_ct(&x.a).unwrap(), ev.upload_ct(&x.b).unwrap());
        g.bench_function("gpu/add", |bn| {
            bn.iter(|| {
                let r = ev.add(black_box(&a), black_box(&b)).unwrap();
                sync();
                r
            })
        });
    }
    // relinearize, mul_relin_rescale (full relin key).
    if fits(
        &dev,
        "mul_relin_rescale",
        need(p, 1, top, top, 12, Work::KeySwitch),
    ) {
        let ev = gpu_eval(&dev, x, true, &[], top);
        let (a, b) = (ev.upload_ct(&x.a).unwrap(), ev.upload_ct(&x.b).unwrap());
        let t = ev.upload_ct(&x.t).unwrap();
        g.bench_function("gpu/relinearize", |bn| {
            bn.iter(|| {
                let r = ev.relinearize(black_box(&t)).unwrap();
                sync();
                r
            })
        });
        g.bench_function("gpu/mul_relin_rescale", |bn| {
            bn.iter(|| {
                let r = ev.rescale(&ev.mul(&a, &b).unwrap()).unwrap();
                sync();
                r
            })
        });
    }
    // rotate1 (one full key).
    if fits(&dev, "rotate1", need(p, 1, top, top, 8, Work::KeySwitch)) {
        let ev = gpu_eval(&dev, x, false, &[1], top);
        let a = ev.upload_ct(&x.a).unwrap();
        g.bench_function("gpu/rotate1", |bn| {
            bn.iter(|| {
                let r = ev.rotate(black_box(&a), 1).unwrap();
                sync();
                r
            })
        });
    }
    // hoisted_rotate_16: full keys, then keys capped at level 9.
    for (label, level) in [("hoisted_rotate_16", top), ("hoisted_rotate_16_L9", 9)] {
        // Inputs, 16 results and two digit sets.
        if fits(
            &dev,
            label,
            need(p, 16, level, level, 2 + 32, Work::KeySwitch),
        ) {
            let ev = gpu_eval(&dev, x, false, &x.rot_steps, level);
            let a = ev.upload_ct(&x.at(&x.a, level)).unwrap();
            g.bench_function(format!("gpu/{label}"), |bn| {
                bn.iter(|| {
                    let r = ev.hoisted_rotate(black_box(&a), &x.rot_steps).unwrap();
                    sync();
                    r
                })
            });
        }
    }
    // matvec_enc_d64: full keys, then capped at 9 and 5; at 5 also 4 products on 4 streams.
    for (label, level) in [
        ("matvec_enc_d64", top),
        ("matvec_enc_d64_L9", 9),
        ("matvec_enc_d64_L5", 5),
    ] {
        let keys = x.mv_steps.len() + 1;
        if !fits(
            &dev,
            label,
            need(p, keys, level, level, 2 * D + 4, Work::Matvec(1)),
        ) {
            continue;
        }
        let ev = gpu_eval(&dev, x, true, &x.mv_steps, level);
        let v = ev.upload_ct(&x.at(&x.v, level)).unwrap();
        let diag = x.diag_at(level).try_map(|c| ev.upload_ct(c)).unwrap();
        g.bench_function(format!("gpu/{label}"), |bn| {
            bn.iter(|| {
                let r = ev.matvec_enc(black_box(&diag), &v).unwrap();
                sync();
                r
            })
        });
        if level == 5 {
            let batch = "matvec_enc_batch4_L5";
            if fits(&dev, batch, need(p, 0, 0, level, 8, Work::Matvec(4))) {
                let ds = vec![diag.clone(); 4];
                let vs = vec![v.clone(); 4];
                g.bench_function(format!("gpu/{batch}"), |bn| {
                    bn.iter(|| ev.matvec_enc_batch(black_box(&ds), &vs).unwrap())
                });
                g.bench_function("gpu/matvec_enc_x4_L5", |bn| {
                    bn.iter(|| {
                        let r: Vec<_> = (0..4)
                            .map(|_| ev.matvec_enc(black_box(&diag), &v).unwrap())
                            .collect();
                        sync();
                        r
                    })
                });
            }
        }
    }
    g.finish();
    // Leave nothing cached for the other processes on the GPU.
    dev.trim_pool(0).unwrap();
}

fn bench(c: &mut Criterion) {
    let x = Inputs::new();
    bench_gpu(c, &x);
    bench_cpu(c, &x);
}

criterion_group!(benches, bench);
criterion_main!(benches);
