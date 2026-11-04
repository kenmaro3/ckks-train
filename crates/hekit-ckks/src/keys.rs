//! Secret, public and key-switching keys, and the key generator.
//!
//! # Conventions
//!
//! * Secret key: ternary `s` (exactly `h` nonzero coefficients when the parameters set a
//!   Hamming weight `h`, otherwise each coefficient `-1, 0, 1` with probability 1/4, 1/2,
//!   1/4). Errors: discrete Gaussian, `sigma = 3.2` ([`ERROR_STD_DEV`]), tail cut `6 sigma`.
//! * Every key polynomial is stored in **NTT form** (SEAL 4.1 convention).
//! * Galois elements use generator **3** (SEAL 4.1 `GaloisTool::generator_ = 3`, the same
//!   generator as [`Encoder::rot_group`](crate::Encoder::rot_group)): a rotation by `k`
//!   slots (to the left; negative `k` to the right) uses `3^(k mod N/2) mod 2N`
//!   ([`galois_elt`]), so `galois_elt(-k)` is the inverse of `galois_elt(k)` mod `2N`.
//!   Conjugation uses `2N - 1` ([`conjugation_elt`]). (The plan text says `5^k`; SEAL
//!   interop is the binding requirement and SEAL uses 3, so 3 it is.)
//!
//! # Representation over `Q * P`
//!
//! An `RnsPoly` holds a prefix of one basis, while keys live modulo `Q * P` (`Q` = all data
//! primes `q_0..=q_L`, `P` = all special primes). Following [`hekit_core::baseconv`], an
//! element of `R_{QP}` is a [`QpPoly`]: a **pair of NTT-form polynomials**, the `Q` part over
//! `params.basis_q()` at level `L` and the `P` part over `params.basis_p()` (all `k` limbs).
//! Small polynomials (secret, errors) are sampled over `basis_q` and their centered
//! coefficients copied into `basis_p` limbs, so both parts represent the same integer
//! polynomial; uniform polynomials are sampled independently per part (uniform residues per
//! prime are uniform modulo `QP` by CRT).
//!
//! * [`SecretKey`]: `s` over `Q * P`. Decryption at level `l` uses the `Q` part truncated to
//!   level `l` ([`SecretKey::at_level`]).
//! * [`PublicKey`]: `(b, a) = (-a s + e, a)` over `Q * P`. Encryption at level `l` works
//!   over `Q_l * P` and divides by `P` (SEAL's `encrypt_zero_asymmetric` followed by
//!   `divide_and_round_q_last`), so the fresh noise is dominated by the rounding error
//!   instead of `e u + e_0 + e_1 s`.
//! * [`KskKey`]: hybrid key-switching key from a source key `s'` to `s`, `dnum` parts. With
//!   digit `j` = primes `D_j = j*alpha .. (j+1)*alpha` (`alpha = params.alpha()`), part `j`
//!   is `(b_j, a_j)` with
//!
//!   ```text
//!   b_j = -a_j s + e_j + P * g_j * s'   (mod Q P),   a_j uniform mod Q P,
//!   g_j = 1 mod q_i for i in D_j,  g_j = 0 mod q_i for the other data primes,
//!   (so P * g_j * s' is 0 mod every special prime)
//!   ```
//!
//!   i.e. `g_j = (Q/Q_{D_j}) * [(Q/Q_{D_j})^{-1}]_{Q_{D_j}}`, the CRT gadget. In RNS this is
//!   SEAL's construction: limb `i` of `b_j`'s `Q` part gets `+ [P]_{q_i} * s'_i` for
//!   `i in D_j`, nothing else changes. Because `g_j` is defined residue-wise, the same key
//!   serves every level `l` (truncate the `Q` parts) and a digit clipped by the level.
//!   [`RelinKey`] uses `s' = s^2`, [`GaloisKey`] uses `s' = sigma_g(s)` (`s(X^g)`).
//!
//! # How Task 17 uses this
//!
//! Key switching of `d` (NTT form, level `l`, decrypting under `s'`) to `s`:
//!
//! 1. for each part `j` whose digit starts at `j*alpha <= l` (clip the digit to
//!    `j*alpha .. min((j+1)*alpha, l+1)`, skip the rest):
//!    `(dq, dp) = d.mod_up_digit(digit, params.basis_p())`;
//! 2. accumulate `dq * b_j.q_at_level(l)`, `dp * b_j.p()` and `dq * a_j.q_at_level(l)`,
//!    `dp * a_j.p()` (pointwise NTT products; `P` parts are used whole);
//! 3. `acc_bq.mod_down(&acc_bp)`, `acc_aq.mod_down(&acc_ap)` give `(kb, ka)` at level `l`
//!    with `kb + ka s ~ d s'` (noise `~ sum_j d_j e_j / P` + rounding).
//!
//! Relinearization of `(c0, c1, c2)`: `(c0 + kb, c1 + ka)` with `d = c2` and the relin key.
//! Rotation by `k`: `d = sigma_g(c1)`, `g = galois_elt(N, k)`, result
//! `(sigma_g(c0) + kb, ka)` with the Galois key for `g` (NTT-form `automorphism`). The unit
//! test `key_switch_with_layout` in this module performs exactly these steps.
//!
//! Noise note: [`RnsPoly::mod_up_digit`] lifts the digit's representative in `[0, Q_j)`, so
//! `d_j` has mean `Q_j / 2`. When `Q_j ~ P` (seal_2020: digit 0 is the 60-bit `q_0`, `P` one
//! 60-bit prime) the mean adds `(q_0 / 2P) * S(X) * e_0` (`S = 1 + X + ... + X^{N-1}`, `e_0`
//! the error of key part 0): `|S(zeta)| ~ 2N/pi` at slot 0, so the key-switch error is
//! `~ 2^-18 * |N(0,1)| * Delta`, concentrated at slot 0 (a fixed draw per key). A centered
//! lift (`(-Q_j/2, Q_j/2]`, Task 17) gives `~ 2^-22.6`.

use std::collections::HashMap;
use std::sync::Arc;

use hekit_core::error::{HekitError, Result};
use hekit_core::poly::RnsPoly;
use hekit_core::rns::RnsBasis;
use hekit_core::sampler::Rng;

use crate::encoder::GALOIS_GENERATOR;
use crate::params::CkksParams;

/// Standard deviation of the discrete Gaussian error distribution.
pub const ERROR_STD_DEV: f64 = 3.2;

/// The Galois element of a rotation by `step` slots: `3^(step mod N/2) mod 2N` (`step` is
/// reduced into `[0, N/2)`, so negative steps give the inverse element). Step 0 gives 1.
pub fn galois_elt(n: usize, step: i32) -> u64 {
    let m = 2 * n as u64;
    let slots = (n / 2).max(1) as i64;
    let mut k = (step as i64).rem_euclid(slots) as u64;
    let (mut base, mut acc) = (GALOIS_GENERATOR % m, 1 % m);
    while k > 0 {
        if k & 1 == 1 {
            acc = acc * base % m;
        }
        base = base * base % m;
        k >>= 1;
    }
    acc
}

/// The Galois element of complex conjugation, `2N - 1`.
pub fn conjugation_elt(n: usize) -> u64 {
    2 * n as u64 - 1
}

/// An element of `R_{QP}` as a pair (`Q` part over `basis_q`, `P` part over `basis_p`), both
/// in the same form (keys: NTT form). See the module docs.
#[derive(Debug, Clone)]
pub struct QpPoly {
    q: RnsPoly,
    p: RnsPoly,
}

impl QpPoly {
    /// Pairs a `Q` part and a `P` part (same ring degree and form, `debug_assert`ed).
    pub fn new(q: RnsPoly, p: RnsPoly) -> QpPoly {
        debug_assert_eq!(q.basis().n(), p.basis().n(), "parts of different N");
        debug_assert_eq!(q.is_ntt_form(), p.is_ntt_form(), "parts in different forms");
        QpPoly { q, p }
    }

    /// The `Q` part (all data primes for keys).
    pub fn q(&self) -> &RnsPoly {
        &self.q
    }

    /// The `P` part (all special primes).
    pub fn p(&self) -> &RnsPoly {
        &self.p
    }

    /// A copy of the `Q` part truncated to `level` (limbs `q_0..=q_level`).
    ///
    /// # Panics
    /// If `level` exceeds the level of the `Q` part.
    pub fn q_at_level(&self, level: usize) -> RnsPoly {
        truncated(&self.q, level)
    }

    /// Consumes the pair and returns `(q_part, p_part)`.
    pub fn into_parts(self) -> (RnsPoly, RnsPoly) {
        (self.q, self.p)
    }
}

/// A copy of `poly` truncated to `level` (plain limb truncation).
///
/// # Panics
/// If `level > poly.level()`.
pub(crate) fn truncated(poly: &RnsPoly, level: usize) -> RnsPoly {
    assert!(
        level <= poly.level(),
        "cannot truncate a level-{} polynomial to level {level}",
        poly.level()
    );
    if level == poly.level() {
        return poly.clone();
    }
    RnsPoly::from_limbs(
        poly.basis(),
        poly.limbs()[..=level].to_vec(),
        poly.is_ntt_form(),
    )
    .expect("a prefix of valid limbs is valid")
}

/// The secret key `s` over `Q * P`, NTT form (see the module docs).
#[derive(Clone)]
pub struct SecretKey(QpPoly);

/// Redacts the secret coefficients.
impl std::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretKey(<redacted>)")
    }
}

impl SecretKey {
    /// Wraps `s` given over `Q * P` in NTT form (e.g. a key loaded from another library,
    /// [`seal_compat`](crate::seal_compat)); the caller checks the bases and the form.
    pub(crate) fn from_qp(qp: QpPoly) -> SecretKey {
        SecretKey(qp)
    }

    /// `s` as an element of `R_{QP}` (NTT form).
    pub fn qp(&self) -> &QpPoly {
        &self.0
    }

    /// `s` over `q_0..=q_level` (NTT form), as used for decryption at `level`.
    ///
    /// # Panics
    /// If `level > L`.
    pub fn at_level(&self, level: usize) -> RnsPoly {
        self.0.q_at_level(level)
    }
}

/// The public key `(b, a) = (-a s + e, a)` over `Q * P`, NTT form.
#[derive(Debug, Clone)]
pub struct PublicKey {
    b: QpPoly,
    a: QpPoly,
}

impl PublicKey {
    /// Wraps `(b, a)` over `Q * P` in NTT form (e.g. a key loaded from another library,
    /// [`seal_compat`](crate::seal_compat)); the caller checks the bases and the form.
    pub(crate) fn from_parts(b: QpPoly, a: QpPoly) -> PublicKey {
        PublicKey { b, a }
    }

    /// `b = -a s + e`.
    pub fn b(&self) -> &QpPoly {
        &self.b
    }

    /// `a`, uniform modulo `QP`.
    pub fn a(&self) -> &QpPoly {
        &self.a
    }
}

/// A hybrid key-switching key: `dnum` parts `(b_j, a_j)` over `Q * P` (module docs).
#[derive(Debug, Clone)]
pub struct KskKey {
    parts: Vec<(QpPoly, QpPoly)>,
}

impl KskKey {
    /// The parts `(b_j, a_j)`, `j = 0..dnum`, NTT form.
    pub fn parts(&self) -> &[(QpPoly, QpPoly)] {
        &self.parts
    }

    /// Number of parts (`params.dnum()`).
    pub fn dnum(&self) -> usize {
        self.parts.len()
    }
}

/// Relinearization key: a [`KskKey`] from `s^2` to `s`.
#[derive(Debug, Clone)]
pub struct RelinKey(KskKey);

impl RelinKey {
    /// The underlying key-switching key.
    pub fn key(&self) -> &KskKey {
        &self.0
    }
}

/// Galois key: a [`KskKey`] from `sigma_elt(s) = s(X^elt)` to `s`.
#[derive(Debug, Clone)]
pub struct GaloisKey {
    elt: u64,
    key: KskKey,
}

impl GaloisKey {
    /// The Galois element `g` (odd, `< 2N`).
    pub fn elt(&self) -> u64 {
        self.elt
    }

    /// The underlying key-switching key.
    pub fn key(&self) -> &KskKey {
        &self.key
    }
}

/// A set of Galois keys indexed by Galois element.
#[derive(Debug, Clone)]
pub struct GaloisKeys {
    n: usize,
    keys: HashMap<u64, GaloisKey>,
}

impl GaloisKeys {
    /// The key for a rotation by `step` slots (looked up by [`galois_elt`], so `k` and
    /// `k + N/2` share a key), if generated.
    pub fn get(&self, step: i32) -> Option<&GaloisKey> {
        self.get_elt(galois_elt(self.n, step))
    }

    /// The key for Galois element `elt`, if generated.
    pub fn get_elt(&self, elt: u64) -> Option<&GaloisKey> {
        self.keys.get(&elt)
    }

    /// Adds `key` (e.g. [`KeyGenerator::conjugation_key`]), replacing and returning a key
    /// with the same element.
    ///
    /// # Panics
    /// If `key` is for a different ring degree.
    pub fn insert(&mut self, key: GaloisKey) -> Option<GaloisKey> {
        let n = key.key.parts.first().map(|(b, _)| b.q().basis().n());
        assert_eq!(n, Some(self.n), "Galois key for a different ring degree");
        self.keys.insert(key.elt, key)
    }

    /// Number of distinct Galois keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// `true` when no key was generated.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

/// Generates the secret key on construction, then public and key-switching keys on demand.
pub struct KeyGenerator {
    params: Arc<CkksParams>,
    sk: SecretKey,
    rng: Rng,
}

/// Redacts the secret key and the RNG state.
impl std::fmt::Debug for KeyGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyGenerator")
            .field("params", &self.params)
            .field("sk", &self.sk)
            .field("rng", &"<redacted>")
            .finish()
    }
}

impl KeyGenerator {
    /// Samples a fresh secret key.
    ///
    /// Errors with `InvalidParams` when the parameters set a Hamming weight `h` with `h = 0`
    /// or `h > N`.
    pub fn new(params: Arc<CkksParams>, rng: Rng) -> Result<KeyGenerator> {
        let n = params.n();
        if let Some(h) = params.hamming_weight() {
            if h == 0 || h > n {
                return Err(HekitError::InvalidParams(format!(
                    "secret-key Hamming weight {h} must be in 1..=N = {n}"
                )));
            }
        }
        let mut rng = rng;
        let top = params.max_level();
        let s = rng.ternary_poly(params.basis_q(), top, params.hamming_weight());
        let sk = SecretKey(small_qp(params.basis_p(), s));
        Ok(KeyGenerator { params, sk, rng })
    }

    /// The parameters.
    pub fn params(&self) -> &Arc<CkksParams> {
        &self.params
    }

    /// A copy of the secret key.
    pub fn secret_key(&self) -> SecretKey {
        self.sk.clone()
    }

    /// A fresh public key.
    pub fn public_key(&mut self) -> PublicKey {
        let (b, a) = self.rlwe_sample();
        PublicKey { b, a }
    }

    /// A fresh relinearization key (`s^2 -> s`).
    pub fn relin_key(&mut self) -> RelinKey {
        let mut s2 = self.sk.0.q.clone();
        s2.mul_assign(&self.sk.0.q);
        RelinKey(self.ksk(&s2))
    }

    /// A fresh Galois key for the element `elt` (`sigma_elt(s) -> s`).
    ///
    /// # Panics
    /// If `elt` is even or `>= 2N`.
    pub fn galois_key(&mut self, elt: u64) -> GaloisKey {
        let two_n = 2 * self.params.n() as u64;
        assert!(
            elt % 2 == 1 && elt < two_n,
            "Galois element {elt} must be odd and < 2N = {two_n}"
        );
        let s_prime = self.sk.0.q.automorphism(elt);
        GaloisKey {
            elt,
            key: self.ksk(&s_prime),
        }
    }

    /// Galois keys for rotations by each of `steps` (one key per distinct element).
    pub fn galois_keys(&mut self, steps: &[i32]) -> GaloisKeys {
        let n = self.params.n();
        let mut keys = HashMap::new();
        for &step in steps {
            let elt = galois_elt(n, step);
            keys.entry(elt).or_insert_with(|| self.galois_key(elt));
        }
        GaloisKeys { n, keys }
    }

    /// The Galois key for complex conjugation (element `2N - 1`).
    pub fn conjugation_key(&mut self) -> GaloisKey {
        self.galois_key(conjugation_elt(self.params.n()))
    }

    /// An RLWE sample `(b, a) = (-a s + e, a)` over `Q * P` (NTT form).
    fn rlwe_sample(&mut self) -> (QpPoly, QpPoly) {
        let (bq, bp) = (self.params.basis_q(), self.params.basis_p());
        let mut aq = self.rng.uniform_poly(bq, bq.len() - 1);
        let mut ap = self.rng.uniform_poly(bp, bp.len() - 1);
        aq.to_ntt();
        ap.to_ntt();
        let e = self.rng.gaussian_poly(bq, bq.len() - 1, ERROR_STD_DEV);
        let e = small_qp(bp, e);
        let b = [(&aq, &self.sk.0.q, &e.q), (&ap, &self.sk.0.p, &e.p)].map(|(a, s, e)| {
            let mut b = a.clone();
            b.mul_assign(s);
            b.neg();
            b.add_assign(e);
            b
        });
        let [b_q, b_p] = b;
        (QpPoly::new(b_q, b_p), QpPoly::new(aq, ap))
    }

    /// The hybrid key-switching key from `s_prime` (NTT form, `Q` part at level `L`) to `s`:
    /// part `j` is an RLWE sample with `[P]_{q_i} * s'_i` added to limb `i` of `b`'s `Q`
    /// part for every prime `i` of digit `j` (module docs).
    fn ksk(&mut self, s_prime: &RnsPoly) -> KskKey {
        let basis_q = Arc::clone(self.params.basis_q());
        assert!(
            s_prime.is_ntt_form() && s_prime.level() + 1 == basis_q.len(),
            "source key must be in NTT form over all data primes"
        );
        let (n, alpha) = (basis_q.n(), self.params.alpha());
        let specials = self.params.p().to_vec();
        let mut parts = Vec::with_capacity(self.params.dnum());
        for j in 0..self.params.dnum() {
            let (b, a) = self.rlwe_sample();
            let digit = j * alpha..(j + 1) * alpha;
            let limbs = basis_q
                .moduli()
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    if digit.contains(&i) {
                        let q = m.value();
                        let p_mod = specials.iter().fold(1 % q, |acc, &p| m.mul(acc, p % q));
                        let s = m.shoup(p_mod);
                        s_prime.limbs()[i]
                            .iter()
                            .map(|&x| m.mul_shoup(x, &s))
                            .collect()
                    } else {
                        vec![0u64; n]
                    }
                })
                .collect();
            let gadget = RnsPoly::from_limbs(&basis_q, limbs, true).expect("reduced limbs");
            let (mut b_q, b_p) = b.into_parts();
            b_q.add_assign(&gadget);
            parts.push((QpPoly::new(b_q, b_p), a));
        }
        KskKey { parts }
    }
}

/// Lifts a small coefficient-form polynomial over `basis_q` (any level) to `Q_l * P` and
/// converts both parts to NTT form.
pub(crate) fn small_qp(basis_p: &Arc<RnsBasis>, mut q: RnsPoly) -> QpPoly {
    let mut p = lift_small(&q, basis_p);
    q.to_ntt();
    p.to_ntt();
    QpPoly::new(q, p)
}

/// Copies the centered coefficients of a small coefficient-form polynomial (read from limb
/// 0; `|x| < q_0 / 2`) into every limb of `target`. Coefficient form.
pub(crate) fn lift_small(src: &RnsPoly, target: &Arc<RnsBasis>) -> RnsPoly {
    debug_assert!(!src.is_ntt_form(), "lift_small needs coefficient form");
    debug_assert_eq!(src.basis().n(), target.n(), "different ring degrees");
    let q0 = src.basis().moduli()[0].value();
    let limbs = target
        .moduli()
        .iter()
        .map(|m| {
            let t = m.value();
            src.limbs()[0]
                .iter()
                .map(|&x| {
                    if x > q0 / 2 {
                        let r = (q0 - x) % t;
                        if r == 0 { 0 } else { t - r }
                    } else {
                        x % t
                    }
                })
                .collect()
        })
        .collect();
    RnsPoly::from_limbs(target, limbs, false).expect("reduced limbs")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::{Ciphertext, Encoder};

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    /// N = 2^12, 6 data primes in 3 digits of 2, 2 special primes: exercises alpha > 1,
    /// k > 1 and digit clipping.
    fn small_params() -> Arc<CkksParams> {
        Arc::new(CkksParams::new(12, &[60, 40, 40, 40, 40, 40], &[60, 60], 40, 3).unwrap())
    }

    fn centered(x: u64, q: u64) -> i64 {
        if x > q / 2 {
            -((q - x) as i64)
        } else {
            x as i64
        }
    }

    /// Centered coefficients of every limb of an NTT-form polynomial.
    fn centered_limbs(p: &RnsPoly) -> Vec<Vec<i64>> {
        let mut p = p.clone();
        p.to_coeff();
        p.limbs()
            .iter()
            .zip(p.basis().moduli())
            .map(|(l, m)| l.iter().map(|&x| centered(x, m.value())).collect())
            .collect()
    }

    fn secret_coeffs(sk: &SecretKey) -> Vec<i64> {
        centered_limbs(sk.qp().q()).swap_remove(0)
    }

    fn rand_vals(seed: u8, len: usize) -> Vec<f64> {
        let mut r = rng(seed);
        (0..len)
            .map(|_| (r.next_u64() >> 11) as f64 * 2f64.powi(-52) - 1.0)
            .collect()
    }

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn hamming_weight_above_n_is_invalid() {
        let n = 1usize << 15;
        let params = Arc::new(CkksParams::seal_2020().with_hamming_weight(n + 1));
        assert!(matches!(
            KeyGenerator::new(params, rng(1)),
            Err(HekitError::InvalidParams(_))
        ));
        let params = Arc::new(CkksParams::seal_2020().with_hamming_weight(0));
        assert!(matches!(
            KeyGenerator::new(params, rng(1)),
            Err(HekitError::InvalidParams(_))
        ));
        // h = N is allowed (every coefficient +-1).
        let params = Arc::new(CkksParams::new(4, &[30, 20], &[30], 20, 1).unwrap());
        let params = Arc::new((*params).clone().with_hamming_weight(16));
        let kg = KeyGenerator::new(params, rng(1)).unwrap();
        assert!(secret_coeffs(&kg.secret_key()).iter().all(|c| c.abs() == 1));
    }

    #[test]
    fn debug_redacts_secrets() {
        let params = Arc::new(CkksParams::new(4, &[30, 20], &[30], 20, 1).unwrap());
        let kg = KeyGenerator::new(Arc::clone(&params), rng(1)).unwrap();
        let sk = kg.secret_key();
        let dec = Decryptor::new(Arc::clone(&params), sk.clone());
        let enc = Encryptor::with_secret_key(params, sk.clone());
        for s in [
            format!("{sk:?}"),
            format!("{kg:?}"),
            format!("{dec:?}"),
            format!("{enc:?}"),
        ] {
            assert!(s.contains("<redacted>"), "{s}");
            assert!(!s.contains("limbs"), "{s}");
        }
    }

    #[test]
    fn secret_key_shape() {
        let params = Arc::new((*small_params()).clone().with_hamming_weight(64));
        let kg = KeyGenerator::new(Arc::clone(&params), rng(2)).unwrap();
        let sk = kg.secret_key();
        assert!(sk.qp().q().is_ntt_form() && sk.qp().p().is_ntt_form());
        assert_eq!(sk.qp().q().level(), params.max_level());
        assert_eq!(sk.qp().p().level(), params.p().len() - 1);
        assert!(Arc::ptr_eq(sk.qp().q().basis(), params.basis_q()));
        assert!(Arc::ptr_eq(sk.qp().p().basis(), params.basis_p()));
        // Every limb of both parts is the same ternary polynomial of weight 64.
        let s = secret_coeffs(&sk);
        assert_eq!(s.iter().filter(|&&c| c != 0).count(), 64);
        assert!(s.iter().all(|c| c.abs() <= 1));
        for limb in centered_limbs(sk.qp().q())
            .into_iter()
            .chain(centered_limbs(sk.qp().p()))
        {
            assert_eq!(limb, s);
        }
        assert_eq!(sk.at_level(2).level(), 2);

        // Dense secret: about half the coefficients nonzero.
        let kg = KeyGenerator::new(small_params(), rng(3)).unwrap();
        let nz = secret_coeffs(&kg.secret_key())
            .iter()
            .filter(|&&c| c != 0)
            .count();
        assert!((1800..2300).contains(&nz), "{nz}");
    }

    #[test]
    fn galois_elements_use_generator_3() {
        let params = small_params();
        let enc = Encoder::new(Arc::clone(&params));
        let (n, slots) = (params.n(), params.slots());
        let m = 2 * n as u64;
        assert_eq!(galois_elt(n, 0), 1);
        assert_eq!(galois_elt(n, 1), 3);
        assert_eq!(galois_elt(n, 2), 9);
        assert_eq!(conjugation_elt(n), m - 1);
        for k in [1i32, 2, 5, 100, slots as i32 - 1] {
            assert_eq!(galois_elt(n, k), enc.rot_group()[k as usize] as u64);
            assert_eq!(galois_elt(n, k) * galois_elt(n, -k) % m, 1, "k = {k}");
            assert_eq!(galois_elt(n, k), galois_elt(n, k + slots as i32));
        }
        assert_eq!(galois_elt(n, slots as i32), 1);
        assert_eq!(galois_elt(n, -(slots as i32)), 1);
    }

    #[test]
    fn public_key_is_rlwe_sample() {
        let params = small_params();
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(4)).unwrap();
        let sk = kg.secret_key();
        let pk = kg.public_key();
        for (b, a, s) in [
            (pk.b().q(), pk.a().q(), sk.qp().q()),
            (pk.b().p(), pk.a().p(), sk.qp().p()),
        ] {
            let mut e = a.clone();
            e.mul_assign(s);
            e.add_assign(b);
            let limbs = centered_limbs(&e);
            let bound = (6.0 * ERROR_STD_DEV) as i64;
            assert!(limbs.iter().flatten().all(|c| c.abs() <= bound));
            assert!(limbs.iter().all(|l| *l == limbs[0]));
        }
    }

    /// Checks `b_j + a_j s - P g_j s' = e_j` small over all of `Q * P`.
    fn check_ksk(params: &CkksParams, sk: &SecretKey, s_prime: &RnsPoly, key: &KskKey) {
        assert_eq!(key.dnum(), params.dnum());
        let alpha = params.alpha();
        let bound = (6.0 * ERROR_STD_DEV) as i64;
        let p_prod = |q: u64| {
            params
                .p()
                .iter()
                .fold(1u128, |acc, &p| acc * p as u128 % q as u128) as u64
        };
        let sp = centered_limbs(s_prime);
        for (j, (b, a)) in key.parts().iter().enumerate() {
            assert!(b.q().is_ntt_form() && a.p().is_ntt_form());
            assert_eq!(b.q().level(), params.max_level());
            for (bb, aa, s, is_q) in [
                (b.q(), a.q(), sk.qp().q(), true),
                (b.p(), a.p(), sk.qp().p(), false),
            ] {
                let mut t = aa.clone();
                t.mul_assign(s);
                t.add_assign(bb);
                let t = centered_limbs(&t);
                let mut e0: Option<Vec<i64>> = None;
                for (i, (limb, m)) in t.iter().zip(bb.basis().moduli()).enumerate() {
                    let q = m.value();
                    let in_digit = is_q && (j * alpha..(j + 1) * alpha).contains(&i);
                    let e: Vec<i64> = if in_digit {
                        let pm = p_prod(q);
                        limb.iter()
                            .zip(&sp[i])
                            .map(|(&x, &y)| {
                                let g = (y.rem_euclid(q as i64) as u128 * pm as u128 % q as u128)
                                    as u64;
                                centered((x.rem_euclid(q as i64) as u64 + q - g) % q, q)
                            })
                            .collect()
                    } else {
                        limb.clone()
                    };
                    assert!(
                        e.iter().all(|c| c.abs() <= bound),
                        "part {j} limb {i}: error not small"
                    );
                    match &e0 {
                        None => e0 = Some(e),
                        Some(e0) => assert_eq!(*e0, e, "part {j} limb {i}: inconsistent"),
                    }
                }
            }
        }
    }

    #[test]
    fn relin_key_parts_encrypt_p_g_s2() {
        for params in [small_params(), Arc::new(CkksParams::seal_2020())] {
            let mut kg = KeyGenerator::new(Arc::clone(&params), rng(5)).unwrap();
            let sk = kg.secret_key();
            let rk = kg.relin_key();
            let mut s2 = sk.qp().q().clone();
            s2.mul_assign(sk.qp().q());
            check_ksk(&params, &sk, &s2, rk.key());
        }
    }

    #[test]
    fn galois_key_parts_encrypt_p_g_sigma_s() {
        let params = small_params();
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(6)).unwrap();
        let sk = kg.secret_key();
        let gks = kg.galois_keys(&[1, -3, 1 + params.slots() as i32]);
        assert_eq!(gks.len(), 2);
        assert!(!gks.is_empty());
        assert!(gks.get(2).is_none());
        let conj = kg.conjugation_key();
        assert_eq!(conj.elt(), 2 * params.n() as u64 - 1);
        for gk in [gks.get(1).unwrap(), gks.get(-3).unwrap(), &conj] {
            let sig = sk.qp().q().automorphism(gk.elt());
            check_ksk(&params, &sk, &sig, gk.key());
        }
        assert_eq!(gks.get(1).unwrap().elt(), 3);
        assert!(gks.get_elt(galois_elt(params.n(), -3)).is_some());
    }

    /// Hybrid key switch exactly as described in "How Task 17 uses this".
    fn key_switch(params: &CkksParams, d: &RnsPoly, key: &KskKey) -> (RnsPoly, RnsPoly) {
        let l = d.level();
        let alpha = params.alpha();
        let mut acc: Option<[RnsPoly; 4]> = None;
        for (j, (b, a)) in key.parts().iter().enumerate() {
            if j * alpha > l {
                break;
            }
            let (dq, dp) =
                d.mod_up_digit(j * alpha..((j + 1) * alpha).min(l + 1), params.basis_p());
            let mut t = [
                b.q_at_level(l),
                b.p().clone(),
                a.q_at_level(l),
                a.p().clone(),
            ];
            t[0].mul_assign(&dq);
            t[1].mul_assign(&dp);
            t[2].mul_assign(&dq);
            t[3].mul_assign(&dp);
            match &mut acc {
                None => acc = Some(t),
                Some(acc) => acc.iter_mut().zip(&t).for_each(|(x, y)| x.add_assign(y)),
            }
        }
        let [mut bq, bp, mut aq, ap] = acc.unwrap();
        bq.mod_down(&bp);
        aq.mod_down(&ap);
        (bq, aq)
    }

    #[test]
    fn key_switch_with_layout() {
        for params in [small_params(), Arc::new(CkksParams::seal_2020())] {
            let enc = Encoder::new(Arc::clone(&params));
            let mut kg = KeyGenerator::new(Arc::clone(&params), rng(7)).unwrap();
            let pk = kg.public_key();
            let rk = kg.relin_key();
            let gks = kg.galois_keys(&[3]);
            let encryptor = Encryptor::new(Arc::clone(&params), pk);
            let decryptor = Decryptor::new(Arc::clone(&params), kg.secret_key());
            let mut r = rng(8);
            let slots = params.slots();
            let v1 = rand_vals(9, slots);
            let v2 = rand_vals(10, slots);
            let delta = params.scale();
            for level in [params.max_level(), 2, 0] {
                let c1 = encryptor.encrypt(&enc.encode(&v1, delta, level).unwrap(), &mut r);
                let c2 = encryptor.encrypt(&enc.encode(&v2, delta, level).unwrap(), &mut r);

                // Relinearize the tensor product (skip at level 0: Delta^2 > q_0).
                if level > 0 {
                    let (a, b) = (c1.polys(), c2.polys());
                    let mut d0 = a[0].clone();
                    d0.mul_assign(&b[0]);
                    let mut d1 = a[0].clone();
                    d1.mul_assign(&b[1]);
                    let mut t = a[1].clone();
                    t.mul_assign(&b[0]);
                    d1.add_assign(&t);
                    let mut d2 = a[1].clone();
                    d2.mul_assign(&b[1]);
                    let (kb, ka) = key_switch(&params, &d2, rk.key());
                    d0.add_assign(&kb);
                    d1.add_assign(&ka);
                    let ct = Ciphertext::new(vec![d0, d1], delta * delta).unwrap();
                    let got = enc.decode(&decryptor.decrypt(&ct));
                    let want: Vec<f64> = v1.iter().zip(&v2).map(|(x, y)| x * y).collect();
                    let e = max_err(&got, &want);
                    eprintln!(
                        "N=2^{} level {level}: relin error 2^{:.2}",
                        params.log_n(),
                        e.log2()
                    );
                    assert!(e < 2f64.powi(-20), "relin error {e:e}");
                }

                // Rotate left by 3.
                let g = galois_elt(params.n(), 3);
                let (kb, ka) = key_switch(
                    &params,
                    &c1.polys()[1].automorphism(g),
                    gks.get(3).unwrap().key(),
                );
                let mut r0 = c1.polys()[0].automorphism(g);
                r0.add_assign(&kb);
                let ct = Ciphertext::new(vec![r0, ka], delta).unwrap();
                let got = enc.decode(&decryptor.decrypt(&ct));
                let want: Vec<f64> = (0..slots).map(|i| v1[(i + 3) % slots]).collect();
                let e = max_err(&got, &want);
                eprintln!(
                    "N=2^{} level {level}: rotation error 2^{:.2}",
                    params.log_n(),
                    e.log2()
                );
                // seal_2020: digit 0 is q_0 (60 bits) ~ P and `mod_up_digit` lifts the
                // non-centered representative in [0, q_0), adding (q_0/2P) S(X) e_0 with
                // S = sum X^i, e_0 the key's part-0 error. At slot 0, |S(zeta)| ~ 2N/pi and
                // Re e_0(zeta) ~ N(0, sigma^2 N/2), so the slot error is
                // 0.5 * (2N/pi) * sigma sqrt(N/2) / Delta * |N(0,1)| ~ 2^-17.97 * |N(0,1)|
                // (a fixed draw per key seed). 2^-15 is ~7.9 standard deviations, so the
                // bound still catches any layout error (which gives O(1) errors) without
                // being seed-lucky. Task 17's centered lift brings this to ~2^-22.6. The
                // layout is pinned tightly by the N = 2^12 case (alpha = 2, k = 2, digit
                // P-ratio 2^-20).
                let bound = if params.log_n() == 15 { -15 } else { -24 };
                assert!(e < 2f64.powi(bound), "rotation error {e:e}");
            }
        }
    }
}
