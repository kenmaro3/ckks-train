use ckks_train_ckks::bootstrap::{BootParams, evalmod::eval_mod};
use ckks_train_ckks::encrypt::Encryptor;
use ckks_train_ckks::{CkksParams, Encoder, Evaluator, KeyGenerator};
use ckks_train_core::sampler::Rng;
use std::sync::Arc;

#[test]
fn shared_evalmod_matches_cpu_bits_and_counts() {
    let p = Arc::new(
        CkksParams::new(
            10,
            &[54, 58, 58, 58, 58, 58, 58, 58],
            &[60, 60, 60, 60],
            42,
            2,
        )
        .unwrap(),
    );
    let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([49; 32])).unwrap();
    let enc = Encoder::new(p.clone());
    let encryptor = Encryptor::new(p.clone(), kg.public_key());
    let ev = Evaluator::new(p.clone(), Some(kg.relin_key()), None);
    let pt = enc
        .encode(&[0.001, -0.002, 0.0, 0.5], 2f64.powi(58), 7)
        .unwrap();
    let ct = encryptor.encrypt(&pt, &mut Rng::from_seed([50; 32]));
    let expected = ev.eval_mod(&ct, &BootParams::default_n16()).unwrap();
    let counts = ev.counter();
    ev.reset_counter();
    let actual = eval_mod(&ev, &ct, &BootParams::default_n16()).unwrap();
    assert_eq!(actual.scale(), expected.scale());
    assert_eq!(actual.level(), expected.level());
    for (a, b) in actual.polys().iter().zip(expected.polys()) {
        assert_eq!(a.limbs(), b.limbs());
    }
    assert_eq!(ev.counter(), counts);
}
