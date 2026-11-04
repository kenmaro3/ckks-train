//! Task 33: the ReBoot layout (Pirillo & Colombo, arXiv:2506.19693, §3 eq. 4–5 and §4.1–4.3)
//! as an in-library baseline (`hekit_nn::layout_reboot`), implemented from the paper only.
//!
//! Every homomorphic result is decrypted and compared **slot for slot** (the whole `r x c`
//! matrix, padding included) with the packed plaintext result within 2^-15, and every
//! counter is compared with the paper's operation count, written out here independently of
//! the library:
//!
//! - `row_sum` (eq. 4): `log2 r` rotations, nothing else.
//! - `expand` (eq. 5): `2 log2 c` rotations, 1 plaintext mask (ct-pt) and its rescale.
//! - `vecmat` = RE-Matmul (§4.2.1, depth 1): 1 ct-ct mult (1 relin), 1 rescale, `log2 r`
//!   rotations.
//! - `vecmat_col` = CE-Matmul (§4.2.1, depth 2): 1 ct-ct mult (1 relin), 1 mask, 2 rescales,
//!   `2 log2 c` rotations.
//! - `outer` (§4.3.3 weight gradient): 1 ct-ct mult (1 relin), 1 rescale, no rotation.
//!
//! The default suite runs at N = 2^12 with `(r, c) = (16, 128)`; the brief's full size
//! `(r, c) = (16, 2048)` on `n16_train` (N = 2^16, 302 MB per key) is `#[ignore]`d (run with
//! `cargo test --release -p hekit-nn --test layout_reboot -- --ignored --test-threads=1`).
//! One shared `Arc<CkksParams>` per context, Galois keys for exactly the steps a test needs,
//! fixed ChaCha20 seeds.

use std::sync::Arc;

use hekit_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter,
};
use hekit_core::error::HekitError;
use hekit_core::sampler::Rng;
use hekit_nn::activation::{Activation, Square};
use hekit_nn::layout_reboot::{GradRelin, RebootLayout};
use ndarray::{Array1, Array2};

/// The brief's gate: every slot within 2^-15.
const TOL: f64 = 1.0 / 32768.0;

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

/// Uniform in `[-1, 1)`.
fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

fn rand_vec(r: &mut Rng, n: usize, amp: f64) -> Vec<f64> {
    (0..n).map(|_| amp * uni(r)).collect()
}

fn rand_mat(r: &mut Rng, rows: usize, cols: usize, amp: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| amp * uni(r))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn log2(n: usize) -> u64 {
    assert!(n.is_power_of_two());
    n.trailing_zeros() as u64
}

/// `x W` for a row vector `x` (`d`) and `W` (`d x k`).
fn xw(x: &[f64], w: &Array2<f64>) -> Vec<f64> {
    Array1::from(x.to_vec()).dot(w).to_vec()
}

/// `delta^T a` (`delta.len() x a.len()`), entry `(i, j) = delta[i] a[j]`.
fn delta_t_a(delta: &[f64], a: &[f64]) -> Array2<f64> {
    Array2::from_shape_fn((delta.len(), a.len()), |(i, j)| delta[i] * a[j])
}

struct Ctx {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
    lay: RebootLayout,
    /// Level at which the test encrypts its fresh inputs.
    level: usize,
}

impl Ctx {
    fn encrypt(&self, v: &[f64], r: &mut Rng) -> Ciphertext {
        let pt = self
            .encoder
            .encode(v, self.params.scale(), self.level)
            .unwrap();
        self.enc.encrypt(&pt, r)
    }

    fn decrypt(&self, ct: &Ciphertext) -> Vec<f64> {
        self.encoder.decode(&self.dec.decrypt(ct))
    }
}

/// One shared `Arc<CkksParams>`, a relin key and Galois keys for exactly `steps`.
fn ctx_with(params: CkksParams, r: usize, c: usize, steps: &[i32], level: usize, seed: u8) -> Ctx {
    let params = Arc::new(params);
    let lay = RebootLayout::new(r, c, params.slots()).unwrap();
    let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
    let pk = kg.public_key();
    let rk = kg.relin_key();
    let gks = kg.galois_keys(steps);
    assert_eq!(gks.len(), steps.len(), "one key per step");
    Ctx {
        encoder: Encoder::new(Arc::clone(&params)),
        enc: Encryptor::new(Arc::clone(&params), pk),
        dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
        ev: Evaluator::new(Arc::clone(&params), Some(rk), Some(gks)),
        params,
        lay,
        level,
    }
}

/// Small parameters: N = 2^12 (2048 slots), Delta = 2^40, 7 levels (enough for the chained
/// two-layer step), `(r, c) = (16, 128)`, Galois keys for all of the layout's steps.
fn small(seed: u8) -> Ctx {
    let mut q = vec![60];
    q.extend([40; 7]);
    // dnum = 8: one 40-bit prime per key-switching digit, below the 60-bit special prime.
    let params = CkksParams::new(12, &q, &[60], 40, 8).unwrap();
    let lay = RebootLayout::new(16, 128, params.slots()).unwrap();
    let top = params.max_level();
    ctx_with(params, 16, 128, &lay.galois_steps(), top, seed)
}

// ------------------------------------------------------------------------------------------
// Layout shape, steps and packing (plaintext)
// ------------------------------------------------------------------------------------------

#[test]
fn new_validates_shape() {
    assert!(RebootLayout::new(16, 128, 2048).is_ok());
    assert!(RebootLayout::new(1, 2048, 2048).is_ok());
    assert!(RebootLayout::new(2048, 1, 2048).is_ok());
    for (r, c) in [(16, 64), (32, 128), (12, 128), (16, 0), (0, 2048), (3, 683)] {
        assert!(
            matches!(
                RebootLayout::new(r, c, 2048),
                Err(HekitError::InvalidParams(_))
            ),
            "({r}, {c}) must be rejected"
        );
    }
    let l = RebootLayout::new(16, 128, 2048).unwrap();
    assert_eq!((l.r(), l.c(), l.slots()), (16, 128, 2048));
}

#[test]
fn rotation_steps_follow_eq4_and_eq5() {
    let l = RebootLayout::new(16, 128, 2048).unwrap();
    // eq. 4: Rot(s, c 2^(log2 r - 1 - t)), t = 0..log2 r - 1.
    assert_eq!(l.row_sum_steps(), vec![1024, 512, 256, 128]);
    // eq. 5: Rot(u, 2^(log2 c - 1 - t)) then Rot(s, -2^t).
    assert_eq!(
        l.expand_steps(),
        vec![64, 32, 16, 8, 4, 2, 1, -1, -2, -4, -8, -16, -32, -64]
    );
    let mut all = l.row_sum_steps();
    all.extend(l.expand_steps());
    all.sort_unstable();
    assert_eq!(l.galois_steps(), all);
}

#[test]
fn packing_matches_section_4_1() {
    let l = RebootLayout::new(4, 8, 32).unwrap();
    let a = [1.0, 2.0, 3.0];
    // Expanded: every column holds a copy of a, zero-padded to r (slot i c + j = a[i]).
    let e = l.pack_expanded(&a).unwrap();
    for i in 0..4 {
        for j in 0..8 {
            assert_eq!(e[i * 8 + j], if i < 3 { a[i] } else { 0.0 });
        }
    }
    // Repeated: every row holds a copy of a, zero-padded to c (slot i c + j = a[j]).
    let p = l.pack_repeated(&a).unwrap();
    for i in 0..4 {
        for j in 0..8 {
            assert_eq!(p[i * 8 + j], if j < 3 { a[j] } else { 0.0 });
        }
    }
    // Row-encoded W (d x k = 3 x 5, d <= r, k <= c): W itself, zero-padded
    // (slot i c + j = W[i][j]).
    let w = Array2::from_shape_fn((3, 5), |(i, j)| (10 * i + j) as f64);
    let wr = l.pack_row(&w).unwrap();
    // Column-encoded V (d x k = 6 x 3, k <= r, d <= c): V^T, zero-padded
    // (slot i c + j = V[j][i]).
    let v = Array2::from_shape_fn((6, 3), |(i, j)| (10 * i + j) as f64 + 0.5);
    let wc = l.pack_col(&v).unwrap();
    for i in 0..4 {
        for j in 0..8 {
            let row = if i < 3 && j < 5 { w[[i, j]] } else { 0.0 };
            let col = if i < 3 && j < 6 { v[[j, i]] } else { 0.0 };
            assert_eq!(wr[i * 8 + j], row);
            assert_eq!(wc[i * 8 + j], col);
        }
    }
    assert_eq!(l.unpack_row(&wr, 3, 5), w);
    assert_eq!(l.unpack_col(&wc, 6, 3), v);
    assert_eq!(l.unpack_expanded(&e, 3), a.to_vec());
    assert_eq!(l.unpack_repeated(&p, 3), a.to_vec());
    // Too long / too large.
    assert!(l.pack_expanded(&[0.0; 5]).is_err());
    assert!(l.pack_repeated(&[0.0; 9]).is_err());
    assert!(l.pack_row(&Array2::zeros((5, 8))).is_err());
    assert!(l.pack_col(&Array2::zeros((8, 5))).is_err());
    // The eq. 5 mask: 1 at the first slot of every row.
    let m = l.first_column_mask();
    assert_eq!(m.len(), 32);
    for (k, &v) in m.iter().enumerate() {
        assert_eq!(v, if k % 8 == 0 { 1.0 } else { 0.0 });
    }
}

// ------------------------------------------------------------------------------------------
// The operations, checked at a given context (shared by the small and the full-size tests)
// ------------------------------------------------------------------------------------------

/// eq. 4: column sums replicated along the rows; `log2 r` rotations.
fn check_row_sum(c: &Ctx, seed: u8) {
    let (r, cc) = (c.lay.r(), c.lay.c());
    let mut g = rng(seed);
    let m = rand_vec(&mut g, r * cc, 1.0);
    let ct = c.encrypt(&m, &mut g);
    c.ev.reset_counter();
    let s = c.lay.row_sum(&c.ev, &ct).unwrap();
    let k = c.ev.counter();
    let want: Vec<f64> = (0..r * cc)
        .map(|p| (0..r).map(|i| m[i * cc + p % cc]).sum())
        .collect();
    let err = max_abs_diff(&c.decrypt(&s), &want);
    eprintln!("row_sum (r, c) = ({r}, {cc}): max err 2^{:.1}", err.log2());
    assert!(err < TOL, "row_sum err {err:e}");
    assert_eq!(
        k,
        OpCounter {
            rotations: log2(r),
            ..Default::default()
        }
    );
    assert_eq!((s.level(), s.scale()), (ct.level(), ct.scale()));
}

/// eq. 5: row sums replicated along each row; `2 log2 c` rotations + 1 mask (1 level), the
/// result has exactly the scale Delta.
fn check_expand(c: &Ctx, seed: u8) {
    let (r, cc) = (c.lay.r(), c.lay.c());
    let mut g = rng(seed);
    let m = rand_vec(&mut g, r * cc, 1.0 / (cc as f64).sqrt());
    let ct = c.encrypt(&m, &mut g);
    c.ev.reset_counter();
    let s = c.lay.expand(&c.ev, &c.encoder, &ct).unwrap();
    let k = c.ev.counter();
    let want: Vec<f64> = (0..r * cc)
        .map(|p| (0..cc).map(|j| m[(p / cc) * cc + j]).sum())
        .collect();
    let err = max_abs_diff(&c.decrypt(&s), &want);
    eprintln!("expand (r, c) = ({r}, {cc}): max err 2^{:.1}", err.log2());
    assert!(err < TOL, "expand err {err:e}");
    assert_eq!(
        k,
        OpCounter {
            rotations: 2 * log2(cc),
            ct_pt_mults: 1,
            rescales: 1,
            ..Default::default()
        }
    );
    assert_eq!(s.level(), ct.level() - 1);
    assert_eq!(
        s.scale(),
        c.params.scale(),
        "exact-scale mask: result at Delta"
    );
}

/// RE-Matmul: `x W` (x expanded, W row-encoded, d = r rows, k = c columns), repeated output.
fn check_vecmat(c: &Ctx, seed: u8) {
    let (r, cc) = (c.lay.r(), c.lay.c());
    let mut g = rng(seed);
    let x = rand_vec(&mut g, r, 1.0);
    let w = rand_mat(&mut g, r, cc, 1.0 / (r as f64).sqrt());
    let xe = c.encrypt(&c.lay.pack_expanded(&x).unwrap(), &mut g);
    let wr = c.encrypt(&c.lay.pack_row(&w).unwrap(), &mut g);
    c.ev.reset_counter();
    let z = c.lay.vecmat(&c.ev, &xe, &wr).unwrap();
    let k = c.ev.counter();
    let want = c.lay.pack_repeated(&xw(&x, &w)).unwrap();
    let err = max_abs_diff(&c.decrypt(&z), &want);
    eprintln!(
        "vecmat (RE) (r, c) = ({r}, {cc}): max err 2^{:.1}",
        err.log2()
    );
    assert!(err < TOL, "vecmat err {err:e}");
    assert_eq!(
        k,
        OpCounter {
            rotations: log2(r),
            relins: 1,
            ct_ct_tensor: 1,
            rescales: 1,
            ..Default::default()
        }
    );
    assert_eq!(k.key_switches(), log2(r) + 1);
    assert_eq!(z.level(), xe.level() - 1);
}

/// CE-Matmul: `a W` (a repeated, d = c, W column-encoded, k = r outputs), expanded output.
fn check_vecmat_col(c: &Ctx, seed: u8) {
    let (r, cc) = (c.lay.r(), c.lay.c());
    let mut g = rng(seed);
    let a = rand_vec(&mut g, cc, 1.0);
    let w = rand_mat(&mut g, cc, r, 1.0 / (cc as f64).sqrt());
    let ar = c.encrypt(&c.lay.pack_repeated(&a).unwrap(), &mut g);
    let wc = c.encrypt(&c.lay.pack_col(&w).unwrap(), &mut g);
    c.ev.reset_counter();
    let z = c.lay.vecmat_col(&c.ev, &c.encoder, &ar, &wc).unwrap();
    let k = c.ev.counter();
    let want = c.lay.pack_expanded(&xw(&a, &w)).unwrap();
    let err = max_abs_diff(&c.decrypt(&z), &want);
    eprintln!(
        "vecmat_col (CE) (r, c) = ({r}, {cc}): max err 2^{:.1}",
        err.log2()
    );
    assert!(err < TOL, "vecmat_col err {err:e}");
    assert_eq!(
        k,
        OpCounter {
            rotations: 2 * log2(cc),
            relins: 1,
            ct_ct_tensor: 1,
            ct_pt_mults: 1,
            rescales: 2,
            ..Default::default()
        }
    );
    assert_eq!(k.key_switches(), 2 * log2(cc) + 1);
    assert_eq!(z.level(), ar.level() - 2);
    assert_eq!(z.scale(), c.params.scale());
}

/// Weight gradient: `delta^T a` from delta expanded (`r`) and a repeated (`c`).
fn check_outer(c: &Ctx, seed: u8) {
    let (r, cc) = (c.lay.r(), c.lay.c());
    let mut g = rng(seed);
    let delta = rand_vec(&mut g, r, 1.0);
    let a = rand_vec(&mut g, cc, 1.0);
    let de = c.encrypt(&c.lay.pack_expanded(&delta).unwrap(), &mut g);
    let ar = c.encrypt(&c.lay.pack_repeated(&a).unwrap(), &mut g);
    c.ev.reset_counter();
    let dw = c.lay.outer(&c.ev, &de, &ar).unwrap();
    let k = c.ev.counter();
    let want = delta_t_a(&delta, &a);
    // Slot (i, j) = delta[i] a[j] is the row encoding of delta^T a (r x c), i.e. the column
    // encoding of its transpose: the gradient of a column-encoded layer.
    let got = c.decrypt(&dw);
    let err = max_abs_diff(&got, &c.lay.pack_row(&want).unwrap());
    eprintln!("outer (r, c) = ({r}, {cc}): max err 2^{:.1}", err.log2());
    assert!(err < TOL, "outer err {err:e}");
    // Read as a column-encoded `c x r` layer: W = (delta^T a)^T = a^T delta.
    let as_col = c.lay.unpack_col(&got, cc, r);
    let tr = want.t();
    assert!(
        as_col
            .iter()
            .zip(tr.iter())
            .all(|(x, y)| (x - y).abs() < TOL)
    );
    assert_eq!(
        k,
        OpCounter {
            relins: 1,
            ct_ct_tensor: 1,
            rescales: 1,
            ..Default::default()
        }
    );
    assert_eq!(dw.level(), de.level() - 1);
}

// ------------------------------------------------------------------------------------------
// Default suite: N = 2^12, (r, c) = (16, 128)
// ------------------------------------------------------------------------------------------

#[test]
fn row_sum_small() {
    check_row_sum(&small(1), 11);
}

#[test]
fn expand_small() {
    check_expand(&small(2), 12);
}

#[test]
fn vecmat_small() {
    check_vecmat(&small(3), 13);
}

#[test]
fn vecmat_col_small() {
    check_vecmat_col(&small(4), 14);
}

#[test]
fn outer_small() {
    check_outer(&small(5), 15);
}

/// Batch gradient = sum over the samples' outer products (one sample per ciphertext). Per
/// sample: 1 tensor and 1 relin (`GradRelin::PerSample`, the cost model's `_reboot` default)
/// or 1 tensor and one relin for the whole batch (`GradRelin::Lazy`, `lazy_relin=True`); one
/// rescale for the sum in both.
#[test]
fn outer_sum_is_the_batch_gradient() {
    let c = small(6);
    let (r, cc, b) = (c.lay.r(), c.lay.c(), 3);
    let mut g = rng(16);
    let mut want = Array2::<f64>::zeros((r, cc));
    let mut pairs = Vec::new();
    for _ in 0..b {
        let delta = rand_vec(&mut g, r, 1.0);
        let a = rand_vec(&mut g, cc, 1.0);
        want += &delta_t_a(&delta, &a);
        pairs.push((
            c.encrypt(&c.lay.pack_expanded(&delta).unwrap(), &mut g),
            c.encrypt(&c.lay.pack_repeated(&a).unwrap(), &mut g),
        ));
    }
    let refs: Vec<(&Ciphertext, &Ciphertext)> = pairs.iter().map(|(x, y)| (x, y)).collect();
    for (mode, relins) in [(GradRelin::PerSample, b as u64), (GradRelin::Lazy, 1)] {
        c.ev.reset_counter();
        let dw = c.lay.outer_sum(&c.ev, &refs, mode).unwrap();
        let k = c.ev.counter();
        let err = max_abs_diff(&c.decrypt(&dw), &c.lay.pack_row(&want).unwrap());
        assert!(err < TOL, "{mode:?} err {err:e}");
        assert_eq!(
            k,
            OpCounter {
                relins,
                ct_ct_tensor: b as u64,
                rescales: 1,
                ..Default::default()
            },
            "{mode:?}"
        );
        assert_eq!(dw.level(), c.level - 1);
    }
    assert!(c.lay.outer_sum(&c.ev, &[], GradRelin::Lazy).is_err());
}

/// The paper's alternation (§4.2.1, §4.3): an RE layer (`W1` row-encoded, 16 -> 64, Square)
/// followed by a CE layer (`W2` column-encoded, 64 -> 10), one training step on 2 samples,
/// one sample per ciphertext, no repacking anywhere:
///
/// forward  `z1 = RE(x_exp, W1_row)` (repeated), `a1 = z1^2`, `y = CE(a1, W2_col)` (expanded),
/// loss     `g = y - t` (expanded),
/// backward `dW2_col = g (.) a1` (outer), `da1 = RE(g, W2_col)` (repeated: the column
///          encoding is the transpose, §4.3.3), `dz1 = 2 z1 (.) da1`, `dW1_row = x (.) dz1`.
///
/// Values against plaintext; per-sample counter against the M0 cost model `_reboot` for this
/// network (layer 1 row-packed with Square and no input backward, layer 2 column-packed with
/// backward): rotations `2 log2 r + 2 log2 c`, relins = tensors = 7, ct-pt 2 (CE mask and the
/// Square derivative's constant).
#[test]
fn two_layer_step_alternates_encodings() {
    let c = small(7);
    let (r, cc) = (c.lay.r(), c.lay.c());
    let (d0, d1, d2, b) = (16, 64, 10, 2usize);
    let mut g = rng(17);
    let w1 = rand_mat(&mut g, d0, d1, 0.2);
    let w2 = rand_mat(&mut g, d1, d2, 0.05);
    let w1e = c.encrypt(&c.lay.pack_row(&w1).unwrap(), &mut g);
    let w2e = c.encrypt(&c.lay.pack_col(&w2).unwrap(), &mut g);
    let mut dw1_want = Array2::<f64>::zeros((d0, d1));
    let mut dw2_want = Array2::<f64>::zeros((d1, d2));
    let mut g1 = Vec::new();
    let mut g2 = Vec::new();
    c.ev.reset_counter();
    for _ in 0..b {
        let x = rand_vec(&mut g, d0, 1.0);
        let t = rand_vec(&mut g, d2, 1.0);
        // Plaintext reference.
        let z1 = xw(&x, &w1);
        let a1: Vec<f64> = z1.iter().map(|v| v * v).collect();
        let y = xw(&a1, &w2);
        let gl: Vec<f64> = y.iter().zip(&t).map(|(p, q)| p - q).collect();
        let w2g = w2.dot(&Array1::from(gl.clone())).to_vec();
        let dz1: Vec<f64> = z1.iter().zip(&w2g).map(|(z, u)| 2.0 * z * u).collect();
        dw1_want += &Array2::from_shape_fn((d0, d1), |(i, j)| x[i] * dz1[j]);
        dw2_want += &Array2::from_shape_fn((d1, d2), |(i, j)| a1[i] * gl[j]);
        // Encrypted.
        let xe = c.encrypt(&c.lay.pack_expanded(&x).unwrap(), &mut g);
        let z1e = c.lay.vecmat(&c.ev, &xe, &w1e).unwrap();
        let a1e = Square.forward(&c.ev, &c.encoder, &z1e).unwrap();
        let ye = c.lay.vecmat_col(&c.ev, &c.encoder, &a1e, &w2e).unwrap();
        let neg_t = c
            .lay
            .pack_expanded(&t.iter().map(|v| -v).collect::<Vec<_>>());
        let pt = c
            .encoder
            .encode(&neg_t.unwrap(), ye.scale(), ye.level())
            .unwrap();
        let ge = c.ev.add_plain(&ye, &pt).unwrap();
        let da1e = c.lay.vecmat(&c.ev, &ge, &w2e).unwrap();
        let dz1e = Square.backward(&c.ev, &c.encoder, &z1e, &da1e).unwrap();
        assert!(max_abs_diff(&c.lay.unpack_repeated(&c.decrypt(&dz1e), d1), &dz1) < TOL);
        g2.push((ge, a1e));
        g1.push((xe, dz1e));
    }
    let p1: Vec<(&Ciphertext, &Ciphertext)> = g1.iter().map(|(x, y)| (x, y)).collect();
    let p2: Vec<(&Ciphertext, &Ciphertext)> = g2.iter().map(|(x, y)| (x, y)).collect();
    let dw1 = c.lay.outer_sum(&c.ev, &p1, GradRelin::PerSample).unwrap();
    let dw2 = c.lay.outer_sum(&c.ev, &p2, GradRelin::PerSample).unwrap();
    let k = c.ev.counter();
    let e1 = c.lay.unpack_row(&c.decrypt(&dw1), d0, d1);
    let e2 = c.lay.unpack_col(&c.decrypt(&dw2), d1, d2);
    let err1 = max_abs_diff(e1.as_slice().unwrap(), dw1_want.as_slice().unwrap());
    let err2 = max_abs_diff(e2.as_slice().unwrap(), dw2_want.as_slice().unwrap());
    eprintln!(
        "two-layer step: dW1 max err 2^{:.1}, dW2 max err 2^{:.1}, levels {} / {}",
        err1.log2(),
        err2.log2(),
        dw1.level(),
        dw2.level()
    );
    assert!(err1 < TOL && err2 < TOL, "dW1 {err1:e} dW2 {err2:e}");
    // Per sample (paper §4.2.1 + cost model `_reboot`): forward RE log2 r rotations + 1 relin,
    // Square 1 relin, CE 2 log2 c rotations + 1 relin + 1 mask; backward RE on W2_col log2 r
    // rotations + 1 relin, Square' 1 relin + 1 ct-pt; gradients 2 relins. No bootstrap
    // (interactive mode).
    let b = b as u64;
    assert_eq!(k.rotations, b * (2 * log2(r) + 2 * log2(cc)));
    assert_eq!(k.relins, b * 7);
    assert_eq!(k.ct_ct_tensor, b * 7);
    assert_eq!(k.ct_pt_mults, b * 2);
    assert_eq!(k.bootstraps, 0);
    assert_eq!(k.key_switches(), b * (2 * log2(r) + 2 * log2(cc) + 7));
    // Levels: dW1 = 7 levels below the inputs (RE 1, Square 1, CE 2, RE 1, Square' 1, outer 1).
    assert_eq!(dw1.level(), c.level - 7);
    assert_eq!(dw2.level(), c.level - 5);
}

/// Failures are checked before any work and count nothing.
#[test]
fn failures_count_nothing() {
    let c = small(8);
    let mut g = rng(18);
    let v = vec![0.5; c.params.slots()];
    let x = c.encrypt(&v, &mut g);
    // Level 0: no rescale possible.
    let x0 = c.ev.mod_switch_to_level(&x, 0).unwrap();
    let x1 = c.ev.mod_switch_to_level(&x, 1).unwrap();
    c.ev.reset_counter();
    assert!(c.lay.vecmat(&c.ev, &x0, &x0).is_err());
    assert!(c.lay.expand(&c.ev, &c.encoder, &x0).is_err());
    assert!(c.lay.vecmat_col(&c.ev, &c.encoder, &x1, &x1).is_err());
    assert!(c.lay.outer(&c.ev, &x0, &x0).is_err());
    assert_eq!(c.ev.counter(), OpCounter::default());
    // Missing Galois keys: an evaluator with only the row-sum keys cannot expand.
    let row_only = ctx_with(
        CkksParams::new(12, &[60, 40, 40], &[60], 40, 3).unwrap(),
        16,
        128,
        &RebootLayout::new(16, 128, 2048).unwrap().row_sum_steps(),
        2,
        9,
    );
    let y = row_only.encrypt(&v, &mut g);
    row_only.ev.reset_counter();
    assert!(matches!(
        row_only.lay.expand(&row_only.ev, &row_only.encoder, &y),
        Err(HekitError::InvalidParams(_))
    ));
    assert_eq!(row_only.ev.counter(), OpCounter::default());
    assert!(row_only.lay.row_sum(&row_only.ev, &y).is_ok());
    // A layout for another slot count.
    let other = RebootLayout::new(16, 64, 1024).unwrap();
    assert!(matches!(
        other.row_sum(&c.ev, &x),
        Err(HekitError::InvalidParams(_))
    ));
}

// ------------------------------------------------------------------------------------------
// Full size (brief): (r, c) = (16, 2048) on n16_train, N = 2^16 — ignored in the default suite
// ------------------------------------------------------------------------------------------

/// Level of the fresh inputs in the full-size tests (low, so the operations are fast; the
/// keys are full size regardless).
const FULL_LEVEL: usize = 3;

/// `row_sum`, RE-Matmul and `outer` at (16, 2048): relin key + the 4 row-sum Galois keys
/// (5 x 302 MB).
#[test]
#[ignore = "n16_train, 5 keys of 302 MB; run with --ignored --test-threads=1"]
fn full_size_row_ops() {
    let params = CkksParams::n16_train();
    let lay = RebootLayout::new(16, 2048, params.slots()).unwrap();
    let t = std::time::Instant::now();
    let c = ctx_with(params, 16, 2048, &lay.row_sum_steps(), FULL_LEVEL, 21);
    eprintln!(
        "full_size_row_ops: keygen {:.1} s",
        t.elapsed().as_secs_f64()
    );
    let t = std::time::Instant::now();
    check_row_sum(&c, 31);
    check_vecmat(&c, 32);
    check_outer(&c, 33);
    eprintln!("full_size_row_ops: ops {:.1} s", t.elapsed().as_secs_f64());
}

/// `expand` and CE-Matmul at (16, 2048): relin key + the 22 expand Galois keys
/// (23 x 302 MB).
#[test]
#[ignore = "n16_train, 23 keys of 302 MB; run with --ignored --test-threads=1"]
fn full_size_expand_ops() {
    let params = CkksParams::n16_train();
    let lay = RebootLayout::new(16, 2048, params.slots()).unwrap();
    let t = std::time::Instant::now();
    let c = ctx_with(params, 16, 2048, &lay.expand_steps(), FULL_LEVEL, 22);
    eprintln!(
        "full_size_expand_ops: keygen {:.1} s",
        t.elapsed().as_secs_f64()
    );
    let t = std::time::Instant::now();
    check_expand(&c, 34);
    check_vecmat_col(&c, 35);
    eprintln!(
        "full_size_expand_ops: ops {:.1} s",
        t.elapsed().as_secs_f64()
    );
}
