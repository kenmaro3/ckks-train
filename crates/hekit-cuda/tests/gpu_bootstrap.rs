#![cfg(feature = "cuda")]
use hekit_ckks::bootstrap::{BootParams, Bootstrapper};
use hekit_ckks::encrypt::{Decryptor, Encryptor};
use hekit_ckks::{BootstrapOps, CkksParams, Encoder, EvalConfig, Evaluator, HeOps, KeyGenerator};
use hekit_core::sampler::Rng;
use hekit_cuda::{CudaBootstrapper, CudaDevice, CudaEvaluator};
use std::sync::Arc;

#[test]
#[ignore = "requires CUDA GPU, >=20 GiB free GPU and >=20 GiB free host RAM; unmeasured until explicitly run"]
fn n16_boot_full_slot_precision_and_cpu_parity() {
    run(16, 20usize << 30);
}

#[test]
#[ignore = "requires CUDA GPU; code-only until explicitly run on hardware"]
fn miniature_bootstrap_precision_and_cpu_parity() {
    run(10, 512usize << 20);
}

fn run(log_n: u32, required: usize) {
    let p = if log_n == 16 {
        CkksParams::n16_boot()
    } else {
        let mut bits = vec![54];
        bits.extend([42; 16]);
        bits.extend([34; 4]);
        bits.extend([58; 7]);
        bits.extend([56; 4]);
        CkksParams::new(log_n, &bits, &[60; 4], 42, 8).unwrap()
    };
    let p = Arc::new(p);
    let dev = CudaDevice::new(0, &p, 1).expect("explicit GPU run requires a CUDA device");
    let free = dev.available_memory().unwrap();
    if free < required {
        eprintln!(
            "SKIP UNMEASURED: GPU bootstrap needs conservative {required} free bytes, available {free}; no keys generated or precision/timing claim"
        );
        return;
    }
    let mut bp = BootParams::default_n16();
    bp.log_slots = log_n - 1;
    let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([49; 32])).unwrap();
    let enc = Encoder::new(p.clone());
    let encryptor = Encryptor::new(p.clone(), kg.public_key());
    let dec = Decryptor::new(p.clone(), kg.secret_key());
    let rk = kg.relin_key();
    let bs = Bootstrapper::new(p.clone(), bp, &enc, &mut kg).unwrap();
    let cpu = Evaluator::new(p.clone(), Some(rk.clone()), None);
    let gpu = CudaEvaluator::with_keys(
        &dev,
        p.clone(),
        Some(&rk),
        &[],
        EvalConfig::NAIVE,
        p.max_level(),
    )
    .unwrap();
    let cuda_bs = CudaBootstrapper::new(&gpu, &bs).unwrap();
    let mut rng = Rng::from_seed([51; 32]);
    let values: Vec<f64> = (0..p.slots())
        .map(|_| 2.0 * ((rng.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0)
        .collect();
    let ct = encryptor.encrypt(&enc.encode(&values, p.scale(), 0).unwrap(), &mut rng);
    let expected = bs.bootstrap(&cpu, &ct).unwrap();
    let input = gpu.upload_ct(&ct).unwrap();
    let start = std::time::Instant::now();
    // The existing generic noninteractive refresher accepts this device bootstrapper
    // with real device/evaluator lifetimes; no host fallback adapter is involved.
    use hekit_nn::refresh::{BootstrapRefresher, Refresher};
    let mut refresher = BootstrapRefresher::new(&cuda_bs, &gpu);
    assert_eq!(refresher.output_level(), 16);
    let result = refresher.refresh(&input).unwrap();
    dev.sync().unwrap();
    let elapsed = start.elapsed();
    let actual = gpu.download_ct(&result).unwrap();
    assert_eq!(actual.level(), 16);
    assert_eq!(actual.scale(), p.scale());
    for (a, b) in actual.polys().iter().zip(expected.polys()) {
        assert_eq!(a.limbs(), b.limbs());
    }
    assert_eq!(gpu.counter(), cpu.counter());
    assert_eq!(gpu.counter(), bs.cost_per_boot());
    let decoded = enc.decode(&dec.decrypt(&actual));
    let error = values
        .iter()
        .zip(decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(error < 2f64.powi(-16), "{} bits", -error.log2());
    eprintln!(
        "MEASURED: N={} slots={} bits={} bootstrap_ms={} device={}; target220ms met={}",
        p.n(),
        p.slots(),
        -error.log2(),
        elapsed.as_secs_f64() * 1000.0,
        dev.describe(),
        elapsed.as_secs_f64() <= 0.220
    );
    assert_eq!(cuda_bs.counter(), hekit_ckks::OpCounter::default());
    let before = gpu.counter();
    let wrong = input.with_scale(p.scale() * 2.0).unwrap();
    assert!(BootstrapOps::bootstrap(&cuda_bs, &gpu, &wrong).is_err());
    assert_eq!(gpu.counter(), before);
    let high = encryptor.encrypt(&enc.encode(&values, p.scale(), 1).unwrap(), &mut rng);
    let high = gpu.upload_ct(&high).unwrap();
    assert!(matches!(
        BootstrapOps::bootstrap(&cuda_bs, &gpu, &high),
        Err(hekit_core::HekitError::InvalidParams(_))
    ));
    let missing = CudaEvaluator::with_keys(
        &dev,
        p.clone(),
        None,
        &[],
        EvalConfig::OPTIMIZED,
        p.max_level(),
    )
    .unwrap();
    assert!(BootstrapOps::bootstrap(&cuda_bs, &missing, &input).is_err());
    let foreign = CudaEvaluator::with_keys(
        &dev,
        Arc::new((*p).clone()),
        None,
        &[],
        EvalConfig::OPTIMIZED,
        p.max_level(),
    )
    .unwrap();
    assert!(BootstrapOps::bootstrap(&cuda_bs, &foreign, &input).is_err());
    assert!(CudaBootstrapper::new(&foreign, &bs).is_err());
    assert_eq!(gpu.counter(), before);
    assert_eq!(missing.counter(), hekit_ckks::OpCounter::default());
    assert_eq!(foreign.counter(), hekit_ckks::OpCounter::default());
}
