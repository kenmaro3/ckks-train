//! Test cases of the arithmetic and base-conversion kernels (Task 45), shared by the GPU tests
//! (`tests/gpu_arith.rs`, run on a CUDA machine) and by the CPU replay of the kernels
//! (`src/emu_arith.rs`, run everywhere): both implement [`Exec`] and call the same `check_*`
//! functions, so every GPU check also runs in the replay, with the same inputs.
//!
//! Every check compares with the CPU implementation bit for bit, on [`INPUTS`] random inputs
//! (plus a few extra extreme or coefficient-form inputs where noted).
#![allow(dead_code)]

pub mod eval;
pub mod ks;

use std::sync::Arc;

use hekit_ckks::keys::{conjugation_elt, galois_elt};
use hekit_ckks::keyswitch::{decompose, digit_ranges};
use hekit_ckks::{CkksParams, QpPoly};
use hekit_core::{RnsBasis, RnsPoly};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

/// Random inputs per operation and parameter set.
pub const INPUTS: usize = 8;

/// One implementation of the operations under test, on host polynomials (the GPU version
/// uploads, runs the kernels and downloads; the replay runs the kernel replays).
pub trait Exec {
    /// `a + b` (same primes and form).
    fn add(&self, a: &RnsPoly, b: &RnsPoly) -> RnsPoly;
    /// `a - b`.
    fn sub(&self, a: &RnsPoly, b: &RnsPoly) -> RnsPoly;
    /// Pointwise `a * b` (NTT form).
    fn mul_pointwise(&self, a: &RnsPoly, b: &RnsPoly) -> RnsPoly;
    /// `-a`.
    fn neg(&self, a: &RnsPoly) -> RnsPoly;
    /// `a * scalar`.
    fn mul_scalar(&self, a: &RnsPoly, scalar: u64) -> RnsPoly;
    /// `a + b` on `Q_l * P` pairs held as one device polynomial (non-contiguous primes).
    fn add_qp(&self, a: &QpPoly, b: &QpPoly) -> QpPoly;
    /// Forward NTT of a coefficient-form `Q_l * P` pair, inverse of an NTT-form one, as one
    /// device polynomial (one launch sequence over non-contiguous primes).
    fn ntt_qp(&self, x: &QpPoly) -> QpPoly;
    /// The Galois automorphism.
    fn automorphism(&self, a: &RnsPoly, galois_elt: u64) -> RnsPoly;
    /// Rescale by the last prime.
    fn rescale(&self, a: &RnsPoly) -> RnsPoly;
    /// Division of a `Q_l * P` pair by `P` with rounding.
    fn mod_down(&self, x: &QpPoly) -> RnsPoly;
    /// The exact centered lift of digit `digit` of `x` to `Q_l * P`.
    fn mod_up(&self, x: &RnsPoly, digit: usize) -> QpPoly;
}

/// The levels of the [`INPUTS`] inputs: top, near top, and spread down to 0.
pub fn levels(params: &CkksParams) -> [usize; INPUTS] {
    let l = params.max_level();
    [l, l - 1, 3 * l / 4, l / 2, l / 3, 2, 1, 0]
}

/// Uniform residues at `level` over `basis`.
pub fn random_poly(basis: &Arc<RnsBasis>, level: usize, ntt_form: bool, seed: u64) -> RnsPoly {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let limbs = basis.moduli()[..=level]
        .iter()
        .map(|m| (0..basis.n()).map(|_| rng.next_u64() % m.value()).collect())
        .collect();
    RnsPoly::from_limbs(basis, limbs, ntt_form).unwrap()
}

/// Every residue `q - 1` (the largest operands).
pub fn max_poly(basis: &Arc<RnsBasis>, level: usize, ntt_form: bool) -> RnsPoly {
    let limbs = basis.moduli()[..=level]
        .iter()
        .map(|m| vec![m.value() - 1; basis.n()])
        .collect();
    RnsPoly::from_limbs(basis, limbs, ntt_form).unwrap()
}

/// A random `Q_l * P` pair (all `P` primes).
pub fn random_qp(params: &CkksParams, level: usize, ntt_form: bool, seed: u64) -> QpPoly {
    let p_level = params.p().len() - 1;
    QpPoly::new(
        random_poly(params.basis_q(), level, ntt_form, seed),
        random_poly(params.basis_p(), p_level, ntt_form, seed ^ 0x5eed),
    )
}

fn assert_same(got: &RnsPoly, want: &RnsPoly, what: &str) {
    assert_eq!(got.level(), want.level(), "{what}: level");
    assert_eq!(got.is_ntt_form(), want.is_ntt_form(), "{what}: form");
    assert!(
        got.basis().moduli() == want.basis().moduli(),
        "{what}: different primes"
    );
    if let Some(i) = (0..got.limbs().len()).find(|&i| got.limbs()[i] != want.limbs()[i]) {
        let c = (0..got.basis().n())
            .find(|&c| got.limbs()[i][c] != want.limbs()[i][c])
            .unwrap();
        panic!(
            "{what}: limb {i} coefficient {c}: got {}, CPU {}",
            got.limbs()[i][c],
            want.limbs()[i][c]
        );
    }
}

fn assert_same_qp(got: &QpPoly, want: &QpPoly, what: &str) {
    assert_same(got.q(), want.q(), &format!("{what} (Q part)"));
    assert_same(got.p(), want.p(), &format!("{what} (P part)"));
}

fn tag(params: &CkksParams) -> String {
    format!("N=2^{} L={}", params.log_n(), params.max_level())
}

/// add / sub / neg / pointwise product / scalar product against `add_assign`, `sub_assign`,
/// `neg`, `mul_assign`, `mul_scalar_assign`. Input `i`: level `levels[i]`; inputs 3 and 7 are
/// over the `P` primes; input 0 is all `q - 1`; add/sub/neg/scalar run in both forms, the
/// product in NTT form; scalars are full-range `u64` (reduced per prime).
pub fn check_pointwise<E: Exec>(e: &E, params: &CkksParams) {
    let mut rng = ChaCha20Rng::seed_from_u64(4500);
    for (i, &level) in levels(params).iter().enumerate() {
        let seed = 4500 + 10 * i as u64;
        let (basis, level) = if i % 4 == 3 {
            (params.basis_p(), level.min(params.p().len() - 1))
        } else {
            (params.basis_q(), level)
        };
        let what = |op: &str| format!("{op} {} input {i} level {level}", tag(params));
        for ntt in [false, true] {
            let (a, b) = if i == 0 {
                (max_poly(basis, level, ntt), max_poly(basis, level, ntt))
            } else {
                (
                    random_poly(basis, level, ntt, seed + ntt as u64),
                    random_poly(basis, level, ntt, seed + 5),
                )
            };
            let mut want = a.clone();
            want.add_assign(&b);
            assert_same(&e.add(&a, &b), &want, &what("add"));
            let mut want = a.clone();
            want.sub_assign(&b);
            assert_same(&e.sub(&a, &b), &want, &what("sub"));
            let mut want = b.clone();
            want.sub_assign(&a);
            assert_same(&e.sub(&b, &a), &want, &what("sub (swapped)"));
            let mut want = a.clone();
            want.neg();
            assert_same(&e.neg(&a), &want, &what("neg"));
            let scalar = if i == 0 { u64::MAX } else { rng.next_u64() };
            let mut want = a.clone();
            want.mul_scalar_assign(scalar);
            assert_same(&e.mul_scalar(&a, scalar), &want, &what("mul_scalar"));
            if ntt {
                let mut want = a.clone();
                want.mul_assign(&b);
                assert_same(&e.mul_pointwise(&a, &b), &want, &what("mul_pointwise"));
            }
        }
    }
}

/// `Q_l * P` pairs as one device polynomial (limbs `0..=l` then the `P` primes, so for
/// `l < L` the prime list is not contiguous): add, forward NTT and inverse NTT against the
/// CPU on each part.
pub fn check_qp_layout<E: Exec>(e: &E, params: &CkksParams) {
    for (i, &level) in levels(params).iter().enumerate() {
        let what = format!("{} input {i} level {level}", tag(params));
        let x = random_qp(params, level, false, 4600 + i as u64);
        let y = random_qp(params, level, false, 4650 + i as u64);
        let (mut q, mut p) = (x.q().clone(), x.p().clone());
        q.add_assign(y.q());
        p.add_assign(y.p());
        assert_same_qp(
            &e.add_qp(&x, &y),
            &QpPoly::new(q, p),
            &format!("add_qp {what}"),
        );
        let f = e.ntt_qp(&x);
        let (mut q, mut p) = (x.q().clone(), x.p().clone());
        q.to_ntt();
        p.to_ntt();
        assert_same_qp(&f, &QpPoly::new(q, p), &format!("ntt over Q_l * P {what}"));
        assert_same_qp(&e.ntt_qp(&f), &x, &format!("intt over Q_l * P {what}"));
    }
}

/// `automorphism` against `RnsPoly::automorphism`: even inputs in NTT form, odd inputs in
/// coefficient form; rotations by several steps, conjugation and the identity; input 7 is over
/// the `P` primes.
pub fn check_automorphism<E: Exec>(e: &E, params: &CkksParams) {
    let n = params.n();
    let elts = [
        galois_elt(n, 1),
        galois_elt(n, -1),
        galois_elt(n, 5),
        conjugation_elt(n),
        galois_elt(n, 1000),
        galois_elt(n, -77),
        1,
        galois_elt(n, (n / 4) as i32 - 3),
    ];
    for (i, (&level, &g)) in levels(params).iter().zip(&elts).enumerate() {
        let (basis, level) = if i == 7 {
            (params.basis_p(), params.p().len() - 1)
        } else {
            (params.basis_q(), level)
        };
        let a = random_poly(basis, level, i % 2 == 0, 4700 + i as u64);
        let what = format!("automorphism g={g} {} input {i} level {level}", tag(params));
        assert_same(&e.automorphism(&a, g), &a.automorphism(g), &what);
    }
}

/// `rescale` against `RnsPoly::rescale_by_last`: the [`INPUTS`] random inputs in NTT form
/// (level >= 1), two in coefficient form and one all-`(q - 1)` input.
pub fn check_rescale<E: Exec>(e: &E, params: &CkksParams) {
    let l = params.max_level();
    let mut inputs: Vec<RnsPoly> = levels(params)
        .iter()
        .enumerate()
        .map(|(i, &lv)| random_poly(params.basis_q(), lv.max(1), true, 4800 + i as u64))
        .collect();
    inputs.push(random_poly(params.basis_q(), l, false, 4890));
    inputs.push(random_poly(params.basis_q(), 3, false, 4891));
    inputs.push(max_poly(params.basis_q(), l, true));
    for (i, a) in inputs.iter().enumerate() {
        let mut want = a.clone();
        want.rescale_by_last();
        let what = format!("rescale {} input {i} level {}", tag(params), a.level());
        assert_same(&e.rescale(a), &want, &what);
    }
}

/// `mod_down` against `RnsPoly::mod_down` (`k = 1`: exact SEAL rounding; `k > 1`: the f64
/// overflow estimate): [`INPUTS`] random `Q_l * P` pairs in NTT form, two in coefficient form
/// and one all-`(q - 1)` pair.
pub fn check_mod_down<E: Exec>(e: &E, params: &CkksParams) {
    let l = params.max_level();
    let mut inputs: Vec<QpPoly> = levels(params)
        .iter()
        .enumerate()
        .map(|(i, &lv)| random_qp(params, lv, true, 4900 + i as u64))
        .collect();
    inputs.push(random_qp(params, l, false, 4990));
    inputs.push(random_qp(params, 2, false, 4991));
    let p_level = params.p().len() - 1;
    inputs.push(QpPoly::new(
        max_poly(params.basis_q(), l, true),
        max_poly(params.basis_p(), p_level, true),
    ));
    for (i, x) in inputs.iter().enumerate() {
        let mut want = x.q().clone();
        want.mod_down(x.p());
        let what = format!("mod_down {} input {i} level {}", tag(params), x.q().level());
        assert_same(&e.mod_down(x), &want, &what);
    }
}

/// `mod_up` against `hekit_ckks::keyswitch::decompose` (the exact centered lift used by key
/// switching): [`INPUTS`] random NTT-form polynomials, every digit of each.
pub fn check_mod_up<E: Exec>(e: &E, params: &CkksParams) {
    for (i, &level) in levels(params).iter().enumerate() {
        let x = random_poly(params.basis_q(), level, true, 5000 + i as u64);
        for (j, want) in decompose(params, &x).iter().enumerate() {
            let what = format!("mod_up {} input {i} level {level} digit {j}", tag(params));
            assert_same_qp(&e.mod_up(&x, j), want, &what);
        }
    }
}

/// Residue modulo `r` of a coefficient whose centered-conversion input `[X + h]_D` (`D` a
/// product of primes including `r`, `h = (D - 1) / 2`, so `h = (r - 1) / 2 mod r`) is `m`
/// (even `c`) or `D - m` (odd `c`), `m = c / 2 + 1`: i.e. `X = s - h mod D`.
///
/// Near-tie inputs: `[X + h]_D / D` is within `N / D` of 0 or 1, where the f64 overflow
/// estimate `floor(sum_j y_j / q_j)` of `BaseConverter::convert_exact` depends on the exact
/// rounding sequence (an FMA-contracted sum differs from the CPU's in 1-18 % of such
/// coefficients; review I1). Uniform random inputs land there with probability ~2^-50.
fn near_tie_residue(r: u64, c: usize) -> u64 {
    let (r, m) = (r as u128, (c / 2 + 1) as u128);
    let s = if c % 2 == 0 { m % r } else { (r - m % r) % r };
    ((s + r - (r >> 1)) % r) as u64
}

/// Limbs of `basis` (first `level + 1`) filled with [`near_tie_residue`] (coefficient form).
fn near_tie_limbs(basis: &Arc<RnsBasis>, level: usize) -> Vec<Vec<u64>> {
    basis.moduli()[..=level]
        .iter()
        .map(|m| {
            (0..basis.n())
                .map(|c| near_tie_residue(m.value(), c))
                .collect()
        })
        .collect()
}

/// A `Q_l * P` pair whose `P` part puts `[X + h]_P` next to 0 or `P` in every coefficient
/// (random `Q` part), in the requested form.
pub fn near_tie_qp(params: &CkksParams, level: usize, ntt_form: bool, seed: u64) -> QpPoly {
    let mut q = random_poly(params.basis_q(), level, false, seed);
    let p_level = params.p().len() - 1;
    let mut p = RnsPoly::from_limbs(
        params.basis_p(),
        near_tie_limbs(params.basis_p(), p_level),
        false,
    )
    .unwrap();
    if ntt_form {
        q.to_ntt();
        p.to_ntt();
    }
    QpPoly::new(q, p)
}

/// A level-`level` NTT-form polynomial whose every digit `D` (of [`digit_ranges`]) puts
/// `[x + h]_D` next to 0 or `Q_D` in every coefficient.
pub fn near_tie_digits(params: &CkksParams, level: usize) -> RnsPoly {
    let basis = params.basis_q();
    let mut limbs = vec![vec![]; level + 1];
    for d in digit_ranges(params, level) {
        for i in d {
            let r = basis.moduli()[i].value();
            limbs[i] = (0..basis.n()).map(|c| near_tie_residue(r, c)).collect();
        }
    }
    let mut x = RnsPoly::from_limbs(basis, limbs, false).unwrap();
    x.to_ntt();
    x
}

/// `mod_down` on near-tie inputs ([`near_tie_qp`]): levels `L`, `L / 2` and 0 in NTT form and
/// level `L` in coefficient form.
pub fn check_mod_down_near_ties<E: Exec>(e: &E, params: &CkksParams) {
    let l = params.max_level();
    for (i, (level, ntt)) in [(l, true), (l / 2, true), (0, true), (l, false)]
        .into_iter()
        .enumerate()
    {
        let x = near_tie_qp(params, level, ntt, 5100 + i as u64);
        let mut want = x.q().clone();
        want.mod_down(x.p());
        let what = format!("mod_down near-tie {} input {i} level {level}", tag(params));
        assert_same(&e.mod_down(&x), &want, &what);
    }
}

/// `mod_up` on near-tie inputs ([`near_tie_digits`]): levels `L`, `L - 1` and `L / 2` (full
/// and level-clipped digits), every digit.
pub fn check_mod_up_near_ties<E: Exec>(e: &E, params: &CkksParams) {
    let l = params.max_level();
    for level in [l, l - 1, l / 2] {
        let x = near_tie_digits(params, level);
        for (j, want) in decompose(params, &x).iter().enumerate() {
            let what = format!("mod_up near-tie {} level {level} digit {j}", tag(params));
            assert_same_qp(&e.mod_up(&x, j), want, &what);
        }
    }
}
