//! Precision measurement harness: how many bits of accuracy each basic operation keeps.
//!
//! [`run_precision_suite`] runs encode/decode, encrypt/decrypt, add, multiply + rescale,
//! rotate and the BSGS matrix-vector products (plaintext and encrypted diagonals) on random
//! inputs in `[-1, 1)` with fixed seeds (so the numbers are reproducible), and reports
//! [`precision_bits`] of each result against the exact (`f64`) answer over **all** `N/2`
//! slots. The inputs sit at the top level with the preset's scale; `PrecisionReport::level`
//! is the level of the ciphertext that was decrypted (one below the top after a rescale).
//!
//! Only the keys the suite needs are generated (relinearization key and the Galois keys of
//! [`bsgs_rotation_steps`] for `d = 16`, which include step 1 for the rotation test), because
//! key generation dominates the cost for `n16_train`.

use std::sync::Arc;

use ndarray::Array2;

use hekit_core::sampler::Rng;

use crate::ciphertext::Ciphertext;
use crate::encoder::Encoder;
use crate::encrypt::{Decryptor, Encryptor};
use crate::evaluator::Evaluator;
use crate::keys::KeyGenerator;
use crate::linalg::{Diagonals, bsgs_rotation_steps, diagonals_from_matrix};
use crate::params::CkksParams;
use crate::plaintext::Plaintext;

/// Matrix dimension of the matvec measurements.
const MATVEC_D: usize = 16;

/// Precision in bits of `actual` against `expected`: `-log2(max_i |expected[i] - actual[i]|)`.
///
/// An exact match (maximum error `0`) gives `f64::INFINITY`. A NaN in either slice gives NaN
/// (which fails any `bits >= threshold` check).
///
/// # Panics
/// If the slices have different lengths or are empty.
pub fn precision_bits(expected: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(expected.len(), actual.len(), "length mismatch");
    assert!(!expected.is_empty(), "empty input");
    let err = expected
        .iter()
        .zip(actual)
        .map(|(e, a)| (e - a).abs())
        .fold(0.0, |m: f64, x| {
            if m.is_nan() || x.is_nan() {
                f64::NAN
            } else {
                m.max(x)
            }
        });
    if err == 0.0 {
        f64::INFINITY
    } else {
        -err.log2()
    }
}

/// The precision of one operation, see [`run_precision_suite`].
#[derive(Clone, Debug, PartialEq)]
pub struct PrecisionReport {
    /// Operation name: `encode/decode`, `encrypt`, `add`, `mul+rescale`, `rotate`, `matvec`
    /// (plaintext diagonals) or `matvec_enc` (encrypted diagonals).
    pub op: String,
    /// Level of the decrypted ciphertext (of the plaintext for `encode/decode`).
    pub level: usize,
    /// [`precision_bits`] over all slots.
    pub bits: f64,
}

/// Uniform values in `[-1, 1)`.
fn rand_vals(r: &mut Rng, len: usize) -> Vec<f64> {
    (0..len)
        .map(|_| 2.0 * (r.next_u64() >> 11) as f64 * 2f64.powi(-53) - 1.0)
        .collect()
}

/// Measures every basic operation on `params`; returns one report per operation, in the order
/// `encode/decode`, `encrypt`, `add`, `mul+rescale`, `rotate`, `matvec`, `matvec_enc`.
///
/// # Panics
/// If a homomorphic operation fails (the presets never do), or if `params` has fewer than
/// 2 levels or fewer than `MATVEC_D` (16) slots.
pub fn run_precision_suite(params: CkksParams) -> Vec<PrecisionReport> {
    let params = Arc::new(params);
    let (slots, top, scale) = (params.slots(), params.max_level(), params.scale());
    assert!(top >= 2 && slots >= MATVEC_D, "parameters too small");
    let mut kg =
        KeyGenerator::new(Arc::clone(&params), Rng::from_seed([20; 32])).expect("key generator");
    let pk = kg.public_key();
    let rk = kg.relin_key();
    let gks = kg.galois_keys(&bsgs_rotation_steps(MATVEC_D));
    let enc = Encoder::new(Arc::clone(&params));
    let encryptor = Encryptor::new(Arc::clone(&params), pk);
    let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
    let eval = Evaluator::new(Arc::clone(&params), Some(rk), Some(gks));
    let mut r = Rng::from_seed([21; 32]);

    let mut reports = Vec::new();
    let mut report = |op: &str, level: usize, expected: &[f64], actual: &[f64]| {
        reports.push(PrecisionReport {
            op: op.to_string(),
            level,
            bits: precision_bits(expected, actual),
        });
    };
    let encrypt = |v: &[f64], r: &mut Rng| -> Ciphertext {
        encryptor.encrypt(&enc.encode(v, scale, top).expect("encode"), r)
    };
    let decode = |ct: &Ciphertext| -> Vec<f64> { enc.decode(&dec.decrypt(ct)) };

    let x = rand_vals(&mut r, slots);
    let y = rand_vals(&mut r, slots);

    // encode/decode
    let pt: Plaintext = enc.encode(&x, scale, top).expect("encode");
    report("encode/decode", pt.level(), &x, &enc.decode(&pt));

    // encrypt
    let (cx, cy) = (encrypt(&x, &mut r), encrypt(&y, &mut r));
    report("encrypt", cx.level(), &x, &decode(&cx));

    // add
    let sum = eval.add(&cx, &cy).expect("add");
    let want: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    report("add", sum.level(), &want, &decode(&sum));

    // mul + rescale
    let prod = eval
        .rescale(&eval.mul(&cx, &cy).expect("mul"))
        .expect("rescale");
    let want: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a * b).collect();
    report("mul+rescale", prod.level(), &want, &decode(&prod));

    // rotate by 1: slot i of the result is slot i + 1.
    let rot = eval.rotate(&cx, 1).expect("rotate");
    let want: Vec<f64> = (0..slots).map(|i| x[(i + 1) % slots]).collect();
    report("rotate", rot.level(), &want, &decode(&rot));

    // matvec: W x for a random d x d matrix and x replicated.
    let d = MATVEC_D;
    let w = Array2::from_shape_vec((d, d), rand_vals(&mut r, d * d)).expect("shape");
    let v = rand_vals(&mut r, d);
    let vr: Vec<f64> = (0..slots).map(|j| v[j % d]).collect();
    let want: Vec<f64> = (0..slots)
        .map(|j| (0..d).map(|k| w[[j % d, k]] * v[k]).sum())
        .collect();
    let diags = diagonals_from_matrix(&w, slots);
    let cv = encrypt(&vr, &mut r);
    let pdiag = Diagonals::<Plaintext>::encode(&enc, &diags, scale, top).expect("diagonals");
    let out = eval
        .rescale(&eval.matvec_plain(&pdiag, &cv).expect("matvec"))
        .expect("rescale");
    report("matvec", out.level(), &want, &decode(&out));
    let cdiag = Diagonals::<Ciphertext>::encrypt(&enc, &encryptor, &diags, scale, top, &mut r)
        .expect("encrypted diagonals");
    let out = eval
        .rescale(&eval.matvec_enc(&cdiag, &cv).expect("matvec_enc"))
        .expect("rescale");
    report("matvec_enc", out.level(), &want, &decode(&out));

    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_anywhere_gives_nan() {
        let nan = f64::NAN;
        for (e, a) in [
            ([nan, 0.0], [0.0, 0.0]),
            ([0.0, nan], [0.0, 0.0]),
            ([0.0, 0.0], [nan, 0.0]),
            ([0.0, 0.0], [0.0, nan]),
            ([nan, 0.0], [0.0, 1e-9]),
            ([0.0, 0.0], [nan, 1e-9]),
        ] {
            assert!(precision_bits(&e, &a).is_nan(), "{e:?} vs {a:?}");
        }
    }
}
