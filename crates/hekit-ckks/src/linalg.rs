//! Matrix-vector products by the diagonal method with baby-step giant-step (BSGS) rotations.
//!
//! # Layout (Halevi-Shoup diagonals)
//!
//! A square `d x d` matrix `W` (`d` a power of two dividing the slot count `N/2`) is stored as
//! its `d` generalized diagonals: diagonal `i` holds `W[j][(j + i) mod d]` in slot `j`,
//! repeated `slots / d` times ([`diagonals_from_matrix`]). The vector `x` must be encrypted
//! *replicated*: slot `j` holds `x[j mod d]` for every `j < slots`. Then
//!
//! `(W x)[j] = sum_{i < d} diag_i[j] * rot(x, i)[j]`
//!
//! and the result holds `(W x)[j mod d]` in slot `j` (replicated again, so it can be fed to the
//! next product). In general, for any `d` slot vectors `diag_i` and any `x`, slot `j` of the
//! result is `sum_{i < d} diag_i[j] * x[(j + i) mod slots]`.
//!
//! # BSGS split
//!
//! [`bsgs_split`]: `baby = 2^ceil(log2(d) / 2)`, `giant = d / baby` (so `baby >= giant`;
//! d = 16 -> 4 x 4, d = 32 -> 8 x 4, d = 64 -> 8 x 8). With `g = baby` and `i = g k1 + k2`
//! (`k1 < giant`, `k2 < baby`),
//!
//! `W x = sum_{k1} rot( sum_{k2} rot(diag_{g k1 + k2}, -g k1) * rot(x, k2), g k1 )`.
//!
//! [`Diagonals`] stores the pre-rotated diagonals `rot(diag_{g k1 + k2}, -g k1)` at
//! `entries[k1 * baby + k2]` (pre-rotation is done in the clear, before encoding or
//! encryption). The `baby - 1` nonzero baby rotations of `x` share one decomposition
//! ([`Evaluator::hoisted_rotate`]); then each nonzero giant step costs one rotation: in all
//! `baby + giant - 2` key-switching rotations, needing Galois keys for exactly the steps
//! [`bsgs_rotation_steps`] (`1..baby` and `g * k1` for `k1 in 1..giant`).
//!
//! # Levels and scales
//!
//! All entries are encoded (or encrypted) at one `scale` and `level`; that level must equal
//! `x.level()` (else `LevelMismatch` from the multiplication). Every term of the sum then has
//! scale `s_x * scale` exactly, so the additions never see a scale mismatch. The products
//! return the result **before rescaling**, like [`Evaluator::mul`]: 2 components, level
//! `x.level()`, scale `x.scale() * scale`; the caller rescales once (typically with
//! `scale = Delta`, giving back `~Delta` one level down).

use ndarray::Array2;

use hekit_core::error::{HekitError, Result};
use hekit_core::sampler::Rng;

use crate::ciphertext::Ciphertext;
use crate::encoder::Encoder;
use crate::encrypt::Encryptor;
use crate::evaluator::Evaluator;
use crate::plaintext::Plaintext;

/// The pre-rotated diagonals of a `d x d` matrix in BSGS order (see the
/// [module docs](self)): `entries[k1 * baby + k2]` is diagonal `g k1 + k2` (`g = baby`)
/// rotated by `-g k1`. `T` is [`Plaintext`] ([`Evaluator::matvec_plain`]) or [`Ciphertext`]
/// ([`Evaluator::matvec_enc`]).
#[derive(Debug, Clone)]
pub struct Diagonals<T> {
    d: usize,
    baby: usize,
    giant: usize,
    entries: Vec<T>,
}

impl<T> Diagonals<T> {
    /// The matrix dimension `d`.
    pub fn d(&self) -> usize {
        self.d
    }

    /// The number of baby steps (`= g`, the giant stride); see [`bsgs_split`].
    pub fn baby(&self) -> usize {
        self.baby
    }

    /// The number of giant steps; see [`bsgs_split`].
    pub fn giant(&self) -> usize {
        self.giant
    }

    /// The `d` pre-rotated diagonals, `entries[k1 * baby + k2]`.
    pub fn entries(&self) -> &[T] {
        &self.entries
    }

    /// The rotation steps whose Galois keys the products need: [`bsgs_rotation_steps`]`(d)`.
    pub fn rotation_steps(&self) -> Vec<i32> {
        bsgs_rotation_steps(self.d)
    }
}

impl Diagonals<Plaintext> {
    /// Pre-rotates and encodes `diagonals` (`d` vectors of `N/2` values each, e.g. from
    /// [`diagonals_from_matrix`]) at `scale` and `level`.
    ///
    /// Errors (`InvalidParams`): `d` zero, not a power of two or not dividing `N/2`; a
    /// diagonal whose length is not `N/2`; and the [`Encoder::encode`] errors.
    pub fn encode(
        encoder: &Encoder,
        diagonals: &[Vec<f64>],
        scale: f64,
        level: usize,
    ) -> Result<Diagonals<Plaintext>> {
        let (baby, giant, pre) = prerotated(diagonals, encoder.params().slots())?;
        let entries = pre
            .iter()
            .map(|v| encoder.encode(v, scale, level))
            .collect::<Result<Vec<_>>>()?;
        Ok(Diagonals {
            d: diagonals.len(),
            baby,
            giant,
            entries,
        })
    }
}

impl Diagonals<Ciphertext> {
    /// Pre-rotates, encodes (at `scale` and `level`) and encrypts `diagonals`; errors as
    /// [`Diagonals::<Plaintext>::encode`]. `encoder` and `encryptor` must be for the same
    /// parameters as the evaluator that will use the result.
    pub fn encrypt(
        encoder: &Encoder,
        encryptor: &Encryptor,
        diagonals: &[Vec<f64>],
        scale: f64,
        level: usize,
        rng: &mut Rng,
    ) -> Result<Diagonals<Ciphertext>> {
        let pts = Diagonals::<Plaintext>::encode(encoder, diagonals, scale, level)?;
        Ok(Diagonals {
            d: pts.d,
            baby: pts.baby,
            giant: pts.giant,
            entries: pts
                .entries
                .iter()
                .map(|p| encryptor.encrypt(p, rng))
                .collect(),
        })
    }
}

/// The BSGS split `(baby, giant)` of `d`: `baby = 2^ceil(log2(d) / 2)`, `giant = d / baby`.
///
/// # Panics
/// If `d` is not a power of two.
pub fn bsgs_split(d: usize) -> (usize, usize) {
    assert!(d.is_power_of_two(), "d = {d} must be a power of two");
    let log_d = d.trailing_zeros();
    let baby = 1usize << log_d.div_ceil(2);
    (baby, d / baby)
}

/// The rotation steps used by the BSGS products for dimension `d`: the baby steps
/// `1, ..., baby - 1`, then the giant steps `baby * k1` for `k1 = 1, ..., giant - 1`
/// (`baby + giant - 2` steps, all distinct, all `< d`).
///
/// # Panics
/// If `d` is not a power of two.
pub fn bsgs_rotation_steps(d: usize) -> Vec<i32> {
    let (baby, giant) = bsgs_split(d);
    (1..baby)
        .chain((1..giant).map(|k1| baby * k1))
        .map(|s| s as i32)
        .collect()
}

/// The `d` generalized diagonals of the square matrix `w` laid out for `slots` slots:
/// slot `j` of diagonal `i` is `w[j mod d][(j + i) mod d]` (diagonal `i`, slot `j < d`, is
/// `w[j][(j + i) mod d]`, repeated `slots / d` times).
///
/// # Panics
/// If `w` is not square, `d` is not a power of two, or `d` does not divide `slots`.
pub fn diagonals_from_matrix(w: &Array2<f64>, slots: usize) -> Vec<Vec<f64>> {
    let (rows, cols) = w.dim();
    assert_eq!(rows, cols, "matrix must be square (got {rows} x {cols})");
    let d = rows;
    assert!(d.is_power_of_two(), "d = {d} must be a power of two");
    assert!(slots % d == 0, "d = {d} must divide the slot count {slots}");
    (0..d)
        .map(|i| (0..slots).map(|j| w[[j % d, (j % d + i) % d]]).collect())
        .collect()
}

/// Checks `diagonals` and returns `(baby, giant, pre)` with `pre[k1 * baby + k2]` =
/// `rot(diagonals[g k1 + k2], -g k1)` (`g = baby`, rotation over all `slots`).
fn prerotated(diagonals: &[Vec<f64>], slots: usize) -> Result<(usize, usize, Vec<Vec<f64>>)> {
    let d = diagonals.len();
    if !d.is_power_of_two() || slots % d != 0 {
        return Err(HekitError::InvalidParams(format!(
            "number of diagonals {d} must be a power of two dividing the slot count {slots}"
        )));
    }
    if let Some(v) = diagonals.iter().find(|v| v.len() != slots) {
        return Err(HekitError::InvalidParams(format!(
            "a diagonal has {} values, expected the slot count {slots}",
            v.len()
        )));
    }
    let (baby, giant) = bsgs_split(d);
    let mut pre = Vec::with_capacity(d);
    for k1 in 0..giant {
        let shift = baby * k1;
        for k2 in 0..baby {
            let v = &diagonals[shift + k2];
            // rot(v, -shift)[j] = v[(j - shift) mod slots].
            pre.push((0..slots).map(|j| v[(j + slots - shift) % slots]).collect());
        }
    }
    Ok((baby, giant, pre))
}

impl Evaluator {
    /// `W x` for plaintext diagonals of `W` (BSGS, see the [module docs](crate::linalg)).
    ///
    /// Costs `d` [`Evaluator::mul_plain`], `baby + giant - 2` rotations (`baby - 1` of them
    /// hoisted) and no relinearization. Returns the unrescaled product: level `x.level()`,
    /// scale `x.scale() * entry scale`.
    ///
    /// Errors: `LevelMismatch` if the entries are not at `x.level()`; `InvalidParams` if `x`
    /// is not a 2-component ciphertext over this evaluator's basis or a Galois key of
    /// [`Diagonals::rotation_steps`] is missing.
    pub fn matvec_plain(&self, diag: &Diagonals<Plaintext>, x: &Ciphertext) -> Result<Ciphertext> {
        self.bsgs(diag, x, |w, xr| self.mul_plain(xr, w), Ok)
    }

    /// `W x` for encrypted diagonals of `W` (BSGS, see the [module docs](crate::linalg)).
    ///
    /// Per giant step, the `baby` tensor products [`Evaluator::mul_no_relin`] are summed at
    /// degree 2 and relinearized once before the giant rotation: `d` tensor products, `giant`
    /// relinearizations (one per giant step, including step 0) and `baby + giant - 2`
    /// rotations (`baby - 1` of them hoisted). Returns the unrescaled product: 2 components,
    /// level `x.level()`, scale `x.scale() * entry scale`.
    ///
    /// Errors: `LevelMismatch` if the entries are not at `x.level()`; `NoLevelsLeft` at
    /// level 0; `InvalidParams` if `x` or an entry is not a 2-component ciphertext over this
    /// evaluator's basis, or the relinearization key or a Galois key of
    /// [`Diagonals::rotation_steps`] is missing.
    pub fn matvec_enc(&self, diag: &Diagonals<Ciphertext>, x: &Ciphertext) -> Result<Ciphertext> {
        self.bsgs(
            diag,
            x,
            |w, xr| self.mul_no_relin(w, xr),
            |ct| self.relinearize(&ct),
        )
    }

    /// `sum_{k1} rot(finish(sum_{k2} mul(entries[k1 baby + k2], rot(x, k2))), g k1)`.
    fn bsgs<T>(
        &self,
        diag: &Diagonals<T>,
        x: &Ciphertext,
        mul: impl Fn(&T, &Ciphertext) -> Result<Ciphertext>,
        finish: impl Fn(Ciphertext) -> Result<Ciphertext>,
    ) -> Result<Ciphertext> {
        let (baby, giant) = (diag.baby, diag.giant);
        let baby_steps: Vec<i32> = (0..baby as i32).collect();
        // Step 0 is a copy (no key switch, not counted).
        let xs = self.hoisted_rotate(x, &baby_steps)?;
        let mut acc: Option<Ciphertext> = None;
        for k1 in 0..giant {
            let row = &diag.entries[k1 * baby..(k1 + 1) * baby];
            let mut inner: Option<Ciphertext> = None;
            for (w, xr) in row.iter().zip(&xs) {
                let t = mul(w, xr)?;
                inner = Some(match inner {
                    None => t,
                    Some(s) => self.add(&s, &t)?,
                });
            }
            let mut block = finish(inner.expect("baby >= 1"))?;
            if k1 > 0 {
                block = self.rotate(&block, (baby * k1) as i32)?;
            }
            acc = Some(match acc {
                None => block,
                Some(s) => self.add(&s, &block)?,
            });
        }
        Ok(acc.expect("giant >= 1"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::OpCounter;
    use crate::encrypt::Decryptor;
    use crate::keys::KeyGenerator;
    use crate::params::CkksParams;
    use std::sync::Arc;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    /// Uniform values in `[-1, 1)`.
    fn rand_vals(r: &mut Rng, len: usize) -> Vec<f64> {
        (0..len)
            .map(|_| 2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0)
            .collect()
    }

    fn rand_matrix(r: &mut Rng, d: usize) -> Array2<f64> {
        Array2::from_shape_vec((d, d), rand_vals(r, d * d)).unwrap()
    }

    /// `v` rotated left by `s` (slot `j` gets `v[(j + s) mod len]`).
    fn rot(v: &[f64], s: i64) -> Vec<f64> {
        let n = v.len() as i64;
        (0..n).map(|j| v[(j + s).rem_euclid(n) as usize]).collect()
    }

    /// `b - a` field by field.
    fn delta(a: &OpCounter, b: &OpCounter) -> OpCounter {
        OpCounter {
            rotations: b.rotations - a.rotations,
            relins: b.relins - a.relins,
            ct_ct_tensor: b.ct_ct_tensor - a.ct_ct_tensor,
            ct_pt_mults: b.ct_pt_mults - a.ct_pt_mults,
            rescales: b.rescales - a.rescales,
            bootstraps: b.bootstraps - a.bootstraps,
        }
    }

    #[test]
    fn split_rule() {
        let cases = [
            (1, (1, 1)),
            (2, (2, 1)),
            (4, (2, 2)),
            (8, (4, 2)),
            (16, (4, 4)),
            (32, (8, 4)),
            (64, (8, 8)),
            (128, (16, 8)),
        ];
        for (d, split) in cases {
            assert_eq!(bsgs_split(d), split, "d = {d}");
            let steps = bsgs_rotation_steps(d);
            assert_eq!(steps.len(), split.0 + split.1 - 2, "d = {d}");
        }
        assert_eq!(bsgs_rotation_steps(16), vec![1, 2, 3, 4, 8, 12]);
        assert_eq!(
            bsgs_rotation_steps(32),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 16, 24]
        );
    }

    #[test]
    fn diagonal_layout() {
        let w = Array2::from_shape_fn((4, 4), |(r, c)| (10 * r + c) as f64);
        let diags = diagonals_from_matrix(&w, 8);
        assert_eq!(diags.len(), 4);
        // Diagonal 1: w[0][1], w[1][2], w[2][3], w[3][0], repeated twice.
        assert_eq!(diags[1], vec![1.0, 12.0, 23.0, 30.0, 1.0, 12.0, 23.0, 30.0]);
        assert_eq!(diags[0], vec![0.0, 11.0, 22.0, 33.0, 0.0, 11.0, 22.0, 33.0]);
    }

    #[test]
    #[should_panic(expected = "square")]
    fn non_square_matrix_panics() {
        diagonals_from_matrix(&Array2::zeros((4, 8)), 16);
    }

    /// The BSGS identity in the clear: pre-rotated diagonals give `W x` on replicated `x`.
    #[test]
    fn bsgs_identity_in_the_clear() {
        let mut r = rng(1);
        for (d, slots) in [(4, 16), (16, 64), (32, 64), (64, 64)] {
            let w = rand_matrix(&mut r, d);
            let x = rand_vals(&mut r, d);
            let xr: Vec<f64> = (0..slots).map(|j| x[j % d]).collect();
            let (baby, giant, pre) = prerotated(&diagonals_from_matrix(&w, slots), slots).unwrap();
            let mut y = vec![0.0; slots];
            for k1 in 0..giant {
                let mut inner = vec![0.0; slots];
                for k2 in 0..baby {
                    let xk = rot(&xr, k2 as i64);
                    for j in 0..slots {
                        inner[j] += pre[k1 * baby + k2][j] * xk[j];
                    }
                }
                for (yj, v) in y.iter_mut().zip(rot(&inner, (baby * k1) as i64)) {
                    *yj += v;
                }
            }
            for j in 0..slots {
                let want: f64 = (0..d).map(|c| w[[j % d, c]] * x[c]).sum();
                assert!((y[j] - want).abs() < 1e-12, "d = {d} slot {j}");
            }
        }
    }

    struct Ctx {
        params: Arc<CkksParams>,
        enc: Encoder,
        encryptor: Encryptor,
        dec: Decryptor,
        eval: Evaluator,
    }

    /// A context with Galois keys for exactly [`bsgs_rotation_steps`]`(d)`.
    fn ctx(params: CkksParams, d: usize, seed: u8) -> Ctx {
        let params = Arc::new(params);
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let pk = kg.public_key();
        let rk = kg.relin_key();
        let steps = bsgs_rotation_steps(d);
        let gks = kg.galois_keys(&steps);
        assert_eq!(gks.len(), steps.len());
        Ctx {
            enc: Encoder::new(Arc::clone(&params)),
            encryptor: Encryptor::new(Arc::clone(&params), pk),
            dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
            eval: Evaluator::new(Arc::clone(&params), Some(rk), Some(gks)),
            params,
        }
    }

    /// Max error of slot `j` against `want[j mod d]`, over all slots.
    fn max_err(got: &[f64], want: &[f64]) -> f64 {
        got.iter()
            .enumerate()
            .map(|(j, g)| (g - want[j % want.len()]).abs())
            .fold(0.0, f64::max)
    }

    /// seal_2020, random `W` (d x d) and `x` in [-1, 1): both products decode to `W x`
    /// (after one rescale) within 2^-15 in every slot, with the documented op counts.
    fn check_matvec(d: usize, seed: u8) {
        let c = ctx(CkksParams::seal_2020(), d, seed);
        let (slots, top, delta_s) = (c.params.slots(), c.params.max_level(), c.params.scale());
        let (baby, giant) = bsgs_split(d);
        let mut r = rng(seed.wrapping_add(100));
        let w = rand_matrix(&mut r, d);
        let x = rand_vals(&mut r, d);
        let want: Vec<f64> = (0..d)
            .map(|i| (0..d).map(|j| w[[i, j]] * x[j]).sum())
            .collect();
        let diags = diagonals_from_matrix(&w, slots);
        let xr: Vec<f64> = (0..slots).map(|j| x[j % d]).collect();
        let ct_x = c
            .encryptor
            .encrypt(&c.enc.encode(&xr, delta_s, top).unwrap(), &mut r);
        let e = &c.eval;

        let pdiag = Diagonals::<Plaintext>::encode(&c.enc, &diags, delta_s, top).unwrap();
        assert_eq!((pdiag.d(), pdiag.baby(), pdiag.giant()), (d, baby, giant));
        assert_eq!(pdiag.entries().len(), d);
        let before = e.counter();
        let y = e.matvec_plain(&pdiag, &ct_x).unwrap();
        let ops = delta(&before, &e.counter());
        assert_eq!(ops.relins, 0, "plain relins");
        assert_eq!(ops.rotations, (baby + giant - 2) as u64, "plain rotations");
        assert_eq!((ops.ct_pt_mults, ops.ct_ct_tensor), (d as u64, 0));
        assert_eq!((y.len(), y.level(), y.scale()), (2, top, delta_s * delta_s));
        let err = max_err(
            &c.enc.decode(&c.dec.decrypt(&e.rescale(&y).unwrap())),
            &want,
        );
        eprintln!("d = {d}: matvec_plain error 2^{:.2}, {ops:?}", err.log2());
        assert!(err < 2f64.powi(-15), "plain d = {d}: {err:e}");

        let cdiag =
            Diagonals::<Ciphertext>::encrypt(&c.enc, &c.encryptor, &diags, delta_s, top, &mut r)
                .unwrap();
        let before = e.counter();
        let y = e.matvec_enc(&cdiag, &ct_x).unwrap();
        let ops = delta(&before, &e.counter());
        assert_eq!(ops.relins, giant as u64, "enc relins");
        assert_eq!(ops.rotations, (baby + giant - 2) as u64, "enc rotations");
        assert_eq!((ops.ct_ct_tensor, ops.ct_pt_mults), (d as u64, 0));
        assert_eq!((y.len(), y.level(), y.scale()), (2, top, delta_s * delta_s));
        let err = max_err(
            &c.enc.decode(&c.dec.decrypt(&e.rescale(&y).unwrap())),
            &want,
        );
        eprintln!("d = {d}: matvec_enc error 2^{:.2}, {ops:?}", err.log2());
        assert!(err < 2f64.powi(-15), "enc d = {d}: {err:e}");
    }

    #[test]
    fn matvec_d16() {
        check_matvec(16, 19);
    }

    /// d = 32: baby 8, giant 4 (not a square split).
    #[test]
    fn matvec_d32() {
        check_matvec(32, 20);
    }

    #[test]
    fn matvec_d64() {
        check_matvec(64, 21);
    }

    #[test]
    fn matvec_errors() {
        let params = CkksParams::new(12, &[60, 40, 40, 40], &[60], 40, 4).unwrap();
        let c = ctx(params, 4, 22);
        let (slots, top, s) = (c.params.slots(), c.params.max_level(), c.params.scale());
        let mut r = rng(23);
        let three = vec![vec![0.0; slots]; 3];
        assert!(matches!(
            Diagonals::<Plaintext>::encode(&c.enc, &three, s, top),
            Err(HekitError::InvalidParams(_))
        ));
        let short = vec![vec![0.0; slots - 1]; 4];
        assert!(matches!(
            Diagonals::<Plaintext>::encode(&c.enc, &short, s, top),
            Err(HekitError::InvalidParams(_))
        ));
        let diags = diagonals_from_matrix(&rand_matrix(&mut r, 4), slots);
        let x = c.encryptor.encrypt(
            &c.enc.encode(&rand_vals(&mut r, slots), s, top).unwrap(),
            &mut r,
        );
        let low = Diagonals::<Plaintext>::encode(&c.enc, &diags, s, top - 1).unwrap();
        assert_eq!(
            c.eval.matvec_plain(&low, &x).unwrap_err(),
            HekitError::LevelMismatch
        );
        let low =
            Diagonals::<Ciphertext>::encrypt(&c.enc, &c.encryptor, &diags, s, top - 1, &mut r)
                .unwrap();
        assert_eq!(
            c.eval.matvec_enc(&low, &x).unwrap_err(),
            HekitError::LevelMismatch
        );
        // d = 8 needs keys for steps 1, 2, 3 (baby 4 > 2) and 4: step 3 is missing here.
        let d8 = diagonals_from_matrix(&rand_matrix(&mut r, 8), slots);
        let p8 = Diagonals::<Plaintext>::encode(&c.enc, &d8, s, top).unwrap();
        assert_eq!(
            c.eval.matvec_plain(&p8, &x).unwrap_err(),
            HekitError::InvalidParams("missing galois key".into())
        );
    }
}
