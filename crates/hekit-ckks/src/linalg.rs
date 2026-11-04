//! Matrix-vector products by the diagonal method with baby-step giant-step (BSGS) rotations.
//!
//! # Layout (Halevi-Shoup diagonals, strided)
//!
//! A square `d x d` matrix `W` (`d` a power of two) acts on `stride` independent vectors
//! ("lanes") `x_0, ..., x_{stride-1}` at once. With `D = d * stride` (a divisor of the slot
//! count `N/2`), slot `t` with `t mod D = j * stride + b` (`j < d`, `b < stride`) holds
//! `x_b[j]`: feature-major packing, `slot(j, b) = j * stride + b`, with the block of `D` slots
//! repeated `slots / D` times. `stride = 1` is the classic replicated layout (slot `j` holds
//! `x[j mod d]`); `stride = batch` with `D = slots` is M2's layout (Task 25).
//!
//! Diagonal `i` holds `W[j][(j + i) mod d]` in slot `j * stride + b` for every lane `b` (and
//! every replica), see [`diagonals_from_matrix`]. Since a rotation by `i * stride` moves
//! feature `j + i` of every lane onto feature `j`,
//!
//! `(W x_b)[j] = sum_{i < d} diag_i[j stride + b] * rot(x, i stride)[j stride + b]`
//!
//! and the result is in the same layout as `x` (slot `j * stride + b` holds `(W x_b)[j]`), so
//! it can be fed to the next product. In general, for any `d` slot vectors `diag_i` and any
//! `x`, slot `t` of the result is `sum_{i < d} diag_i[t] * x[(t + i * stride) mod slots]`.
//!
//! # BSGS split
//!
//! [`bsgs_split`]: `baby = 2^ceil(log2(d) / 2)`, `giant = d / baby` (so `baby >= giant`;
//! d = 16 -> 4 x 4, d = 32 -> 8 x 4, d = 64 -> 8 x 8). With `g = baby`, `s = stride` and
//! `i = g k1 + k2` (`k1 < giant`, `k2 < baby`),
//!
//! `W x = sum_{k1} rot( sum_{k2} rot(diag_{g k1 + k2}, -g k1 s) * rot(x, k2 s), g k1 s )`.
//!
//! [`Diagonals`] stores the pre-rotated diagonals `rot(diag_{g k1 + k2}, -g k1 s)` at
//! `entries[k1 * baby + k2]` (pre-rotation is done in the clear, before encoding or
//! encryption; [`Diagonals::from_prerotated`] accepts entries prepared by the caller). The
//! `baby - 1` nonzero baby rotations of `x` share one decomposition
//! ([`Evaluator::hoisted_rotate`]); then each nonzero giant step costs one rotation: in all
//! `baby + giant - 2` key-switching rotations, needing Galois keys for exactly the steps
//! [`bsgs_rotation_steps`]`(d, stride)` (`k2 * s` for `k2 in 1..baby` and `g * k1 * s` for
//! `k1 in 1..giant`).
//!
//! # Levels and scales
//!
//! All entries are encoded (or encrypted) at one `scale` and `level`; that level must equal
//! `x.level()` (else `LevelMismatch`). Every term of the sum then has scale `s_x * scale`
//! exactly, so the additions never see a scale mismatch. The products return the result
//! **before rescaling**, like [`Evaluator::mul`]: 2 components, level `x.level()`, scale
//! `x.scale() * scale`; the caller rescales once (typically with `scale = Delta`, giving back
//! `~Delta` one level down).
//!
//! # Checks before work
//!
//! [`Evaluator::matvec_plain`] and [`Evaluator::matvec_enc`] validate everything before the
//! first homomorphic operation: the shape of `x` and of every entry, equal entry levels (and
//! `== x.level()`), equal entry scales, `d * stride | slots`, every Galois key of
//! [`Diagonals::rotation_steps`] and (for `matvec_enc`) the relinearization key. So a failing
//! product returns its error without having counted anything ([`crate::counter`]).

use ndarray::Array2;

use hekit_core::error::{HekitError, Result};
use hekit_core::sampler::Rng;

use crate::ciphertext::Ciphertext;
use crate::encoder::Encoder;
use crate::encrypt::Encryptor;
use crate::evaluator::Evaluator;
use crate::plaintext::Plaintext;

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::plaintext::Plaintext {}
    impl Sealed for crate::ciphertext::Ciphertext {}
}

/// An entry type of [`Diagonals`] usable by the products: [`Plaintext`] or [`Ciphertext`]
/// (sealed). Gives the level, scale and slot count that [`Diagonals::from_prerotated`] and
/// the products validate.
pub trait DiagonalEntry: sealed::Sealed {
    /// The level of the entry.
    fn level(&self) -> usize;
    /// The scale of the entry.
    fn scale(&self) -> f64;
    /// The slot count `N/2` of the ring the entry lives in.
    fn slots(&self) -> usize;
}

impl DiagonalEntry for Plaintext {
    fn level(&self) -> usize {
        Plaintext::level(self)
    }
    fn scale(&self) -> f64 {
        Plaintext::scale(self)
    }
    fn slots(&self) -> usize {
        self.poly().basis().n() / 2
    }
}

impl DiagonalEntry for Ciphertext {
    fn level(&self) -> usize {
        Ciphertext::level(self)
    }
    fn scale(&self) -> f64 {
        Ciphertext::scale(self)
    }
    fn slots(&self) -> usize {
        self.polys()[0].basis().n() / 2
    }
}

/// The pre-rotated diagonals of a `d x d` matrix in BSGS order for a layout of `stride`
/// lanes (see the [module docs](self)): `entries[k1 * baby + k2]` is diagonal `g k1 + k2`
/// (`g = baby`) rotated by `-g k1 stride`.
///
/// `T` is [`Plaintext`] ([`Evaluator::matvec_plain`]) or [`Ciphertext`]
/// ([`Evaluator::matvec_enc`]); [`Diagonals::try_map`] can also produce other types (e.g.
/// decrypted `Vec<f64>` values) with the same shape. The split is always
/// [`bsgs_split`]`(d)`.
#[derive(Debug, Clone)]
pub struct Diagonals<T> {
    d: usize,
    baby: usize,
    giant: usize,
    stride: usize,
    entries: Vec<T>,
}

impl<T> Diagonals<T> {
    /// The matrix dimension `d`.
    pub fn d(&self) -> usize {
        self.d
    }

    /// The number of baby steps (`= g`, the giant stride in diagonals); see [`bsgs_split`].
    pub fn baby(&self) -> usize {
        self.baby
    }

    /// The number of giant steps; see [`bsgs_split`].
    pub fn giant(&self) -> usize {
        self.giant
    }

    /// The slot distance between consecutive features of one lane (the number of lanes
    /// packed per block of `d * stride` slots); 1 for the classic replicated layout.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The `d` pre-rotated diagonals, `entries[k1 * baby + k2]`.
    pub fn entries(&self) -> &[T] {
        &self.entries
    }

    /// Mutable access to the entries (e.g. for an in-place weight update). The count is
    /// fixed; changing levels or scales so that they no longer agree is reported by the
    /// products (`LevelMismatch` / `ScaleMismatch`) before any work.
    pub fn entries_mut(&mut self) -> &mut [T] {
        &mut self.entries
    }

    /// Consumes the diagonals and returns the entries (rebuild with
    /// [`Diagonals::from_prerotated`]).
    pub fn into_entries(self) -> Vec<T> {
        self.entries
    }

    /// Applies `f` to every entry, keeping `d`, the split and the stride; the first error is
    /// returned. The result is not re-validated (for [`Plaintext`] / [`Ciphertext`] results
    /// the products check levels and scales at use; to validate eagerly, pass
    /// [`Diagonals::into_entries`] to [`Diagonals::from_prerotated`]).
    pub fn try_map<U>(&self, f: impl FnMut(&T) -> Result<U>) -> Result<Diagonals<U>> {
        Ok(Diagonals {
            d: self.d,
            baby: self.baby,
            giant: self.giant,
            stride: self.stride,
            entries: self.entries.iter().map(f).collect::<Result<Vec<_>>>()?,
        })
    }

    /// The rotation steps whose Galois keys the products need:
    /// [`bsgs_rotation_steps`]`(d, stride)`.
    pub fn rotation_steps(&self) -> Vec<i32> {
        bsgs_rotation_steps(self.d, self.stride)
    }
}

impl<T: DiagonalEntry> Diagonals<T> {
    /// Wraps `entries` that are already pre-rotated for BSGS (`entries[k1 * baby + k2]` =
    /// `rot(diag_{g k1 + k2}, -g k1 stride)`, `(baby, giant) = bsgs_split(d)`), e.g. built by
    /// the caller in its own layout code or returned by [`Diagonals::into_entries`].
    ///
    /// Errors: `InvalidParams` if `d` is not a power of two, `entries.len() != d`,
    /// `stride == 0` or `d * stride` does not divide the slot count; `LevelMismatch` if the
    /// entries are not all at one level; `ScaleMismatch` if their scales are not all
    /// exactly equal.
    pub fn from_prerotated(d: usize, stride: usize, entries: Vec<T>) -> Result<Diagonals<T>> {
        if !d.is_power_of_two() || entries.len() != d {
            return Err(HekitError::InvalidParams(format!(
                "{} entries for d = {d} (d must be a power of two, one entry per diagonal)",
                entries.len()
            )));
        }
        check_stride(d, stride, entries[0].slots())?;
        check_entries(&entries)?;
        let (baby, giant) = bsgs_split(d);
        Ok(Diagonals {
            d,
            baby,
            giant,
            stride,
            entries,
        })
    }
}

impl Diagonals<Plaintext> {
    /// Pre-rotates and encodes `diagonals` (`d` vectors of `N/2` values each, e.g. from
    /// [`diagonals_from_matrix`] with the same `stride`) at `scale` and `level`.
    ///
    /// To move plaintext diagonals to another level, encode them again at that level (there
    /// is no plaintext `mod_switch`).
    ///
    /// Errors (`InvalidParams`): `d` zero or not a power of two; `stride == 0` or
    /// `d * stride` not dividing `N/2`; a diagonal whose length is not `N/2`; and the
    /// [`Encoder::encode`] errors.
    pub fn encode(
        encoder: &Encoder,
        diagonals: &[Vec<f64>],
        stride: usize,
        scale: f64,
        level: usize,
    ) -> Result<Diagonals<Plaintext>> {
        let (baby, giant, pre) = prerotated(diagonals, encoder.params().slots(), stride)?;
        let entries = pre
            .iter()
            .map(|v| encoder.encode(v, scale, level))
            .collect::<Result<Vec<_>>>()?;
        Ok(Diagonals {
            d: diagonals.len(),
            baby,
            giant,
            stride,
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
        stride: usize,
        scale: f64,
        level: usize,
        rng: &mut Rng,
    ) -> Result<Diagonals<Ciphertext>> {
        let pts = Diagonals::<Plaintext>::encode(encoder, diagonals, stride, scale, level)?;
        pts.try_map(|p| Ok(encryptor.encrypt(p, rng)))
    }

    /// Every entry switched down to `level` ([`Evaluator::mod_switch_to_level`]: primes
    /// dropped, scale unchanged), e.g. to multiply an `x` that has been rescaled.
    ///
    /// Errors as [`Evaluator::mod_switch_to_level`] (`InvalidParams` for `level` above the
    /// entries' level or entries not over `eval`'s basis).
    pub fn mod_switch_to_level(
        &self,
        eval: &Evaluator,
        level: usize,
    ) -> Result<Diagonals<Ciphertext>> {
        self.try_map(|ct| eval.mod_switch_to_level(ct, level))
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

/// The rotation steps used by the BSGS products for dimension `d` and lane stride
/// `stride`: the baby steps `k2 * stride` for `k2 = 1, ..., baby - 1`, then the giant steps
/// `baby * k1 * stride` for `k1 = 1, ..., giant - 1` (`baby + giant - 2` steps, all
/// distinct, all `< d * stride`). `stride = 1` gives `1..baby` and `baby * k1`.
///
/// # Panics
/// If `d` is not a power of two, `stride == 0`, or `d * stride` exceeds `i32::MAX`.
pub fn bsgs_rotation_steps(d: usize, stride: usize) -> Vec<i32> {
    let (baby, giant) = bsgs_split(d);
    assert!(stride > 0, "stride must be positive");
    assert!(
        d.checked_mul(stride)
            .is_some_and(|n| n <= i32::MAX as usize),
        "d * stride = {d} * {stride} overflows a rotation step"
    );
    (1..baby)
        .map(|k2| k2 * stride)
        .chain((1..giant).map(|k1| baby * k1 * stride))
        .map(|s| s as i32)
        .collect()
}

/// The `d` generalized diagonals of the square matrix `w` laid out for `slots` slots with
/// `stride` lanes: slot `t` of diagonal `i`, with `t mod (d * stride) = j * stride + b`, is
/// `w[j][(j + i) mod d]` (the same for every lane `b` and every replica of the
/// `d * stride`-slot block). `stride = 1`: slot `j` is `w[j mod d][(j mod d + i) mod d]`.
///
/// # Panics
/// If `w` is not square, `d` is not a power of two, `stride == 0`, or `d * stride` does not
/// divide `slots`.
pub fn diagonals_from_matrix(w: &Array2<f64>, slots: usize, stride: usize) -> Vec<Vec<f64>> {
    let (rows, cols) = w.dim();
    assert_eq!(rows, cols, "matrix must be square (got {rows} x {cols})");
    let d = rows;
    assert!(d.is_power_of_two(), "d = {d} must be a power of two");
    assert!(stride > 0, "stride must be positive");
    let block = d * stride;
    assert!(
        slots % block == 0,
        "d * stride = {d} * {stride} must divide the slot count {slots}"
    );
    (0..d)
        .map(|i| {
            (0..slots)
                .map(|t| {
                    let j = (t % block) / stride;
                    w[[j, (j + i) % d]]
                })
                .collect()
        })
        .collect()
}

/// `InvalidParams` unless `stride >= 1` and `d * stride` divides `slots`.
fn check_stride(d: usize, stride: usize, slots: usize) -> Result<()> {
    match d.checked_mul(stride) {
        Some(block) if stride > 0 && slots % block == 0 => Ok(()),
        _ => Err(HekitError::InvalidParams(format!(
            "d * stride = {d} * {stride} must divide the slot count {slots} (stride >= 1)"
        ))),
    }
}

/// `LevelMismatch` / `ScaleMismatch` unless all entries share one level and one scale
/// (exactly).
fn check_entries<T: DiagonalEntry>(entries: &[T]) -> Result<()> {
    let Some(first) = entries.first() else {
        return Ok(());
    };
    if entries.iter().any(|e| e.level() != first.level()) {
        return Err(HekitError::LevelMismatch);
    }
    if entries.iter().any(|e| e.scale() != first.scale()) {
        return Err(HekitError::ScaleMismatch);
    }
    Ok(())
}

/// Checks `diagonals` and returns `(baby, giant, pre)` with `pre[k1 * baby + k2]` =
/// `rot(diagonals[g k1 + k2], -g k1 stride)` (`g = baby`, rotation over all `slots`).
fn prerotated(
    diagonals: &[Vec<f64>],
    slots: usize,
    stride: usize,
) -> Result<(usize, usize, Vec<Vec<f64>>)> {
    let d = diagonals.len();
    if !d.is_power_of_two() {
        return Err(HekitError::InvalidParams(format!(
            "number of diagonals {d} must be a power of two"
        )));
    }
    check_stride(d, stride, slots)?;
    if let Some(v) = diagonals.iter().find(|v| v.len() != slots) {
        return Err(HekitError::InvalidParams(format!(
            "a diagonal has {} values, expected the slot count {slots}",
            v.len()
        )));
    }
    let (baby, giant) = bsgs_split(d);
    let mut pre = Vec::with_capacity(d);
    for k1 in 0..giant {
        let shift = (baby * k1 * stride) % slots;
        for k2 in 0..baby {
            let v = &diagonals[baby * k1 + k2];
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
    /// Errors, all reported before any work: `InvalidParams` if `x` is not a 2-component
    /// ciphertext over this evaluator's basis, an entry is not over it, `d * stride` does
    /// not divide the slot count, or a Galois key of [`Diagonals::rotation_steps`] is
    /// missing (`"missing galois key"`); `LevelMismatch` if the entries are not all at
    /// `x.level()`; `ScaleMismatch` if their scales differ.
    pub fn matvec_plain(&self, diag: &Diagonals<Plaintext>, x: &Ciphertext) -> Result<Ciphertext> {
        self.check_galois_input(x)?;
        for p in diag.entries() {
            self.check_basis(std::slice::from_ref(p.poly()))?;
        }
        self.check_bsgs(diag, x)?;
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
    /// Errors, all reported before any work: those of [`Evaluator::matvec_plain`] (with
    /// entries that must also be 2-component ciphertexts), `NoLevelsLeft` at level 0, and
    /// `InvalidParams` for a missing (or foreign) relinearization key.
    pub fn matvec_enc(&self, diag: &Diagonals<Ciphertext>, x: &Ciphertext) -> Result<Ciphertext> {
        self.check_galois_input(x)?;
        for c in diag.entries() {
            self.check_galois_input(c)?;
        }
        self.check_bsgs(diag, x)?;
        if x.level() == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        self.relin_key_checked()?;
        self.bsgs(
            diag,
            x,
            |w, xr| self.mul_no_relin(w, xr),
            |ct| self.relinearize(&ct),
        )
    }

    /// The checks shared by both products: entry levels (`== x.level()`) and scales, the
    /// stride, and every Galois key of [`Diagonals::rotation_steps`].
    fn check_bsgs<T: DiagonalEntry>(&self, diag: &Diagonals<T>, x: &Ciphertext) -> Result<()> {
        check_entries(diag.entries())?;
        if diag.entries().iter().any(|e| e.level() != x.level()) {
            return Err(HekitError::LevelMismatch);
        }
        check_stride(diag.d(), diag.stride(), self.params().slots())?;
        self.check_rotation_keys(&diag.rotation_steps())
    }

    /// `sum_{k1} rot(finish(sum_{k2} mul(entries[k1 baby + k2], rot(x, k2 s))), g k1 s)`.
    /// The callers have validated everything, so no step fails part-way.
    fn bsgs<T>(
        &self,
        diag: &Diagonals<T>,
        x: &Ciphertext,
        mul: impl Fn(&T, &Ciphertext) -> Result<Ciphertext>,
        finish: impl Fn(Ciphertext) -> Result<Ciphertext>,
    ) -> Result<Ciphertext> {
        let (baby, giant, stride) = (diag.baby, diag.giant, diag.stride);
        let baby_steps: Vec<i32> = (0..baby).map(|k2| (k2 * stride) as i32).collect();
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
                block = self.rotate(&block, (baby * k1 * stride) as i32)?;
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

    /// `lanes` vectors of length `d` packed with `stride = lanes`: slot `j * stride + b` (and
    /// its replicas every `d * stride` slots) holds `xs[b][j]`.
    fn pack(xs: &[Vec<f64>], slots: usize) -> Vec<f64> {
        let (stride, d) = (xs.len(), xs[0].len());
        (0..slots)
            .map(|t| {
                let p = t % (d * stride);
                xs[p % stride][p / stride]
            })
            .collect()
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
            let steps = bsgs_rotation_steps(d, 1);
            assert_eq!(steps.len(), split.0 + split.1 - 2, "d = {d}");
        }
        assert_eq!(bsgs_rotation_steps(16, 1), vec![1, 2, 3, 4, 8, 12]);
        assert_eq!(
            bsgs_rotation_steps(32, 1),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 16, 24]
        );
        // Strided: baby k2 * stride, giant g * k1 * stride.
        assert_eq!(bsgs_rotation_steps(16, 8), vec![8, 16, 24, 32, 64, 96]);
        assert_eq!(bsgs_rotation_steps(1, 4), Vec::<i32>::new());
    }

    #[test]
    #[should_panic(expected = "stride")]
    fn zero_stride_panics() {
        bsgs_rotation_steps(16, 0);
    }

    #[test]
    fn diagonal_layout() {
        let w = Array2::from_shape_fn((4, 4), |(r, c)| (10 * r + c) as f64);
        let diags = diagonals_from_matrix(&w, 8, 1);
        assert_eq!(diags.len(), 4);
        // Diagonal 1: w[0][1], w[1][2], w[2][3], w[3][0], repeated twice.
        assert_eq!(diags[1], vec![1.0, 12.0, 23.0, 30.0, 1.0, 12.0, 23.0, 30.0]);
        assert_eq!(diags[0], vec![0.0, 11.0, 22.0, 33.0, 0.0, 11.0, 22.0, 33.0]);
        // Stride 2: slot j * 2 + b holds w[j][(j + i) mod 4] for both lanes b, and the block
        // of d * stride = 8 slots repeats.
        let diags = diagonals_from_matrix(&w, 16, 2);
        let d1 = [1.0, 1.0, 12.0, 12.0, 23.0, 23.0, 30.0, 30.0];
        assert_eq!(diags[1], [d1, d1].concat());
    }

    #[test]
    #[should_panic(expected = "square")]
    fn non_square_matrix_panics() {
        diagonals_from_matrix(&Array2::zeros((4, 8)), 16, 1);
    }

    #[test]
    #[should_panic(expected = "must divide the slot count")]
    fn stride_must_fit_slots() {
        diagonals_from_matrix(&Array2::zeros((4, 4)), 16, 8);
    }

    /// The BSGS identity in the clear: pre-rotated (strided) diagonals give `W x_b` in every
    /// lane `b` of the packed `x`.
    #[test]
    fn bsgs_identity_in_the_clear() {
        let mut r = rng(1);
        for (d, slots, stride) in [
            (4, 16, 1),
            (16, 64, 1),
            (32, 64, 1),
            (64, 64, 1),
            (4, 16, 4),
            (16, 256, 8),
            (16, 128, 8),
            (8, 128, 2),
        ] {
            let w = rand_matrix(&mut r, d);
            let xs: Vec<Vec<f64>> = (0..stride).map(|_| rand_vals(&mut r, d)).collect();
            let xr = pack(&xs, slots);
            let diags = diagonals_from_matrix(&w, slots, stride);
            let (baby, giant, pre) = prerotated(&diags, slots, stride).unwrap();
            let mut y = vec![0.0; slots];
            for k1 in 0..giant {
                let mut inner = vec![0.0; slots];
                for k2 in 0..baby {
                    let xk = rot(&xr, (k2 * stride) as i64);
                    for j in 0..slots {
                        inner[j] += pre[k1 * baby + k2][j] * xk[j];
                    }
                }
                for (yj, v) in y.iter_mut().zip(rot(&inner, (baby * k1 * stride) as i64)) {
                    *yj += v;
                }
            }
            for (t, yt) in y.iter().enumerate() {
                let p = t % (d * stride);
                let (j, b) = (p / stride, p % stride);
                let want: f64 = (0..d).map(|c| w[[j, c]] * xs[b][c]).sum();
                assert!((yt - want).abs() < 1e-12, "d {d} stride {stride} slot {t}");
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

    /// A context with Galois keys for exactly `steps` (and a relinearization key if `relin`).
    fn ctx_with(params: CkksParams, steps: &[i32], relin: bool, seed: u8) -> Ctx {
        let params = Arc::new(params);
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let pk = kg.public_key();
        let rk = kg.relin_key();
        let gks = kg.galois_keys(steps);
        Ctx {
            enc: Encoder::new(Arc::clone(&params)),
            encryptor: Encryptor::new(Arc::clone(&params), pk),
            dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
            eval: Evaluator::new(Arc::clone(&params), relin.then_some(rk), Some(gks)),
            params,
        }
    }

    /// A context with Galois keys for exactly [`bsgs_rotation_steps`]`(d, stride)`.
    fn ctx(params: CkksParams, d: usize, stride: usize, seed: u8) -> Ctx {
        let steps = bsgs_rotation_steps(d, stride);
        let c = ctx_with(params, &steps, true, seed);
        assert_eq!(c.eval.galois_keys().unwrap().len(), steps.len());
        c
    }

    fn small() -> CkksParams {
        CkksParams::new(12, &[60, 40, 40, 40], &[60], 40, 4).unwrap()
    }

    /// Max error of slot `t` against `want[b][j]` (`t mod (d stride) = j stride + b`).
    fn max_err_lanes(got: &[f64], want: &[Vec<f64>]) -> f64 {
        let (stride, d) = (want.len(), want[0].len());
        got.iter()
            .enumerate()
            .map(|(t, g)| {
                let p = t % (d * stride);
                (g - want[p % stride][p / stride]).abs()
            })
            .fold(0.0, f64::max)
    }

    /// seal_2020, random `W` (d x d) and `stride` lanes `x_b` in [-1, 1): both products
    /// decode to `W x_b` in every lane (after one rescale) within 2^-15 in every slot, with
    /// the documented op counts.
    fn check_matvec(d: usize, stride: usize, seed: u8) {
        let c = ctx(CkksParams::seal_2020(), d, stride, seed);
        let (slots, top, delta_s) = (c.params.slots(), c.params.max_level(), c.params.scale());
        let (baby, giant) = bsgs_split(d);
        let mut r = rng(seed.wrapping_add(100));
        let w = rand_matrix(&mut r, d);
        let xs: Vec<Vec<f64>> = (0..stride).map(|_| rand_vals(&mut r, d)).collect();
        let want: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| {
                (0..d)
                    .map(|i| (0..d).map(|j| w[[i, j]] * x[j]).sum())
                    .collect()
            })
            .collect();
        let diags = diagonals_from_matrix(&w, slots, stride);
        let ct_x = c.encryptor.encrypt(
            &c.enc.encode(&pack(&xs, slots), delta_s, top).unwrap(),
            &mut r,
        );
        let e = &c.eval;

        let pdiag = Diagonals::<Plaintext>::encode(&c.enc, &diags, stride, delta_s, top).unwrap();
        assert_eq!(
            (pdiag.d(), pdiag.baby(), pdiag.giant(), pdiag.stride()),
            (d, baby, giant, stride)
        );
        assert_eq!(pdiag.entries().len(), d);
        assert_eq!(pdiag.rotation_steps(), bsgs_rotation_steps(d, stride));
        let before = e.counter();
        let y = e.matvec_plain(&pdiag, &ct_x).unwrap();
        let ops = delta(&before, &e.counter());
        assert_eq!(ops.relins, 0, "plain relins");
        assert_eq!(ops.rotations, (baby + giant - 2) as u64, "plain rotations");
        assert_eq!((ops.ct_pt_mults, ops.ct_ct_tensor), (d as u64, 0));
        assert_eq!((y.len(), y.level(), y.scale()), (2, top, delta_s * delta_s));
        let got = c.enc.decode(&c.dec.decrypt(&e.rescale(&y).unwrap()));
        let err = max_err_lanes(&got, &want);
        eprintln!(
            "d = {d}, stride {stride}: matvec_plain error 2^{:.2}, {ops:?}",
            err.log2()
        );
        assert!(err < 2f64.powi(-15), "plain d = {d}: {err:e}");

        let cdiag = Diagonals::<Ciphertext>::encrypt(
            &c.enc,
            &c.encryptor,
            &diags,
            stride,
            delta_s,
            top,
            &mut r,
        )
        .unwrap();
        let before = e.counter();
        let y = e.matvec_enc(&cdiag, &ct_x).unwrap();
        let ops = delta(&before, &e.counter());
        assert_eq!(ops.relins, giant as u64, "enc relins");
        assert_eq!(ops.rotations, (baby + giant - 2) as u64, "enc rotations");
        assert_eq!((ops.ct_ct_tensor, ops.ct_pt_mults), (d as u64, 0));
        assert_eq!((y.len(), y.level(), y.scale()), (2, top, delta_s * delta_s));
        let got = c.enc.decode(&c.dec.decrypt(&e.rescale(&y).unwrap()));
        let err = max_err_lanes(&got, &want);
        eprintln!(
            "d = {d}, stride {stride}: matvec_enc error 2^{:.2}, {ops:?}",
            err.log2()
        );
        assert!(err < 2f64.powi(-15), "enc d = {d}: {err:e}");
    }

    #[test]
    fn matvec_d16() {
        check_matvec(16, 1, 19);
    }

    /// d = 32: baby 8, giant 4 (not a square split).
    #[test]
    fn matvec_d32() {
        check_matvec(32, 1, 20);
    }

    #[test]
    fn matvec_d64() {
        check_matvec(64, 1, 21);
    }

    /// M2's feature-major layout (`slot(j, b) = j * batch + b`): d = 16, 8 batch lanes, each
    /// lane against the plaintext `W x_b`.
    #[test]
    fn matvec_d16_stride8() {
        check_matvec(16, 8, 24);
    }

    #[test]
    fn matvec_errors() {
        let c = ctx(small(), 4, 1, 22);
        let (slots, top, s) = (c.params.slots(), c.params.max_level(), c.params.scale());
        let mut r = rng(23);
        let three = vec![vec![0.0; slots]; 3];
        assert!(matches!(
            Diagonals::<Plaintext>::encode(&c.enc, &three, 1, s, top),
            Err(HekitError::InvalidParams(_))
        ));
        let short = vec![vec![0.0; slots - 1]; 4];
        assert!(matches!(
            Diagonals::<Plaintext>::encode(&c.enc, &short, 1, s, top),
            Err(HekitError::InvalidParams(_))
        ));
        let four = vec![vec![0.0; slots]; 4];
        for stride in [0, 3, slots] {
            assert!(
                matches!(
                    Diagonals::<Plaintext>::encode(&c.enc, &four, stride, s, top),
                    Err(HekitError::InvalidParams(_))
                ),
                "stride {stride}"
            );
        }
        let diags = diagonals_from_matrix(&rand_matrix(&mut r, 4), slots, 1);
        let x = c.encryptor.encrypt(
            &c.enc.encode(&rand_vals(&mut r, slots), s, top).unwrap(),
            &mut r,
        );
        let low = Diagonals::<Plaintext>::encode(&c.enc, &diags, 1, s, top - 1).unwrap();
        assert_eq!(
            c.eval.matvec_plain(&low, &x).unwrap_err(),
            HekitError::LevelMismatch
        );
        let low =
            Diagonals::<Ciphertext>::encrypt(&c.enc, &c.encryptor, &diags, 1, s, top - 1, &mut r)
                .unwrap();
        assert_eq!(
            c.eval.matvec_enc(&low, &x).unwrap_err(),
            HekitError::LevelMismatch
        );
        // d = 8 needs keys for steps 1, 2, 3 (baby 4 > 2) and 4: step 3 is missing here.
        let d8 = diagonals_from_matrix(&rand_matrix(&mut r, 8), slots, 1);
        let p8 = Diagonals::<Plaintext>::encode(&c.enc, &d8, 1, s, top).unwrap();
        assert_eq!(
            c.eval.matvec_plain(&p8, &x).unwrap_err(),
            HekitError::InvalidParams("missing galois key".into())
        );
    }

    /// m1: every key is checked before any work, so a failing product counts nothing.
    #[test]
    fn missing_keys_fail_before_any_work() {
        let d = 16;
        let steps = bsgs_rotation_steps(d, 1);
        // All baby keys (1, 2, 3) and giant step 4, but not giant steps 8 and 12.
        let c = ctx_with(small(), &steps[..4], true, 25);
        let (slots, top, s) = (c.params.slots(), c.params.max_level(), c.params.scale());
        let mut r = rng(26);
        let diags = diagonals_from_matrix(&rand_matrix(&mut r, d), slots, 1);
        let x = c.encryptor.encrypt(
            &c.enc.encode(&rand_vals(&mut r, slots), s, top).unwrap(),
            &mut r,
        );
        let p = Diagonals::<Plaintext>::encode(&c.enc, &diags, 1, s, top).unwrap();
        let ce = Diagonals::<Ciphertext>::encrypt(&c.enc, &c.encryptor, &diags, 1, s, top, &mut r)
            .unwrap();
        let before = c.eval.counter();
        assert_eq!(
            c.eval.matvec_plain(&p, &x).unwrap_err(),
            HekitError::InvalidParams("missing galois key".into())
        );
        assert_eq!(
            c.eval.matvec_enc(&ce, &x).unwrap_err(),
            HekitError::InvalidParams("missing galois key".into())
        );
        assert_eq!(c.eval.counter(), before, "a failed matvec counts nothing");

        // A missing relinearization key fails matvec_enc up front too.
        let c = ctx_with(small(), &steps, false, 27);
        let x = c.encryptor.encrypt(
            &c.enc.encode(&rand_vals(&mut r, slots), s, top).unwrap(),
            &mut r,
        );
        let ce = Diagonals::<Ciphertext>::encrypt(&c.enc, &c.encryptor, &diags, 1, s, top, &mut r)
            .unwrap();
        assert!(matches!(
            c.eval.matvec_enc(&ce, &x),
            Err(HekitError::InvalidParams(_))
        ));
        assert_eq!(c.eval.counter(), OpCounter::default());
    }

    /// I1: `from_prerotated` validates count, stride, levels and scales; `into_entries` /
    /// `from_prerotated` round-trips; `entries_mut` changes are re-validated at use.
    #[test]
    fn from_prerotated_validates_and_round_trips() {
        let d = 4;
        let c = ctx(small(), d, 2, 28);
        let (slots, top, s) = (c.params.slots(), c.params.max_level(), c.params.scale());
        let mut r = rng(29);
        let w = rand_matrix(&mut r, d);
        let diags = diagonals_from_matrix(&w, slots, 2);
        let p = Diagonals::<Plaintext>::encode(&c.enc, &diags, 2, s, top).unwrap();
        let xs: Vec<Vec<f64>> = (0..2).map(|_| rand_vals(&mut r, d)).collect();
        let x = c
            .encryptor
            .encrypt(&c.enc.encode(&pack(&xs, slots), s, top).unwrap(), &mut r);
        let want = c
            .enc
            .decode(&c.dec.decrypt(&c.eval.matvec_plain(&p, &x).unwrap()));

        let entries = p.clone().into_entries();
        assert_eq!(entries.len(), d);
        // Wrong count, bad d, bad strides.
        assert!(matches!(
            Diagonals::from_prerotated(d, 2, entries[..3].to_vec()),
            Err(HekitError::InvalidParams(_))
        ));
        assert!(matches!(
            Diagonals::from_prerotated(3, 2, entries[..3].to_vec()),
            Err(HekitError::InvalidParams(_))
        ));
        for stride in [0, 3, slots / 2] {
            assert!(
                matches!(
                    Diagonals::from_prerotated(d, stride, entries.clone()),
                    Err(HekitError::InvalidParams(_))
                ),
                "stride {stride}"
            );
        }
        // Mixed levels and scales.
        let mut mixed = entries.clone();
        mixed[3] = c.enc.encode(&diags[3], s, top - 1).unwrap();
        assert_eq!(
            Diagonals::from_prerotated(d, 2, mixed).unwrap_err(),
            HekitError::LevelMismatch
        );
        let mut mixed = entries.clone();
        mixed[1] = c.enc.encode(&diags[1], s * 2.0, top).unwrap();
        assert_eq!(
            Diagonals::from_prerotated(d, 2, mixed).unwrap_err(),
            HekitError::ScaleMismatch
        );
        // Round trip: same product.
        let back = Diagonals::from_prerotated(d, 2, entries).unwrap();
        assert_eq!(
            (back.d(), back.baby(), back.giant(), back.stride()),
            (4, 2, 2, 2)
        );
        let got = c
            .enc
            .decode(&c.dec.decrypt(&c.eval.matvec_plain(&back, &x).unwrap()));
        assert_eq!(got, want);

        // entries_mut can break the invariants; the product re-checks them.
        let mut broken = back.clone();
        broken.entries_mut()[2] = c.enc.encode(&diags[2], s * 2.0, top).unwrap();
        assert_eq!(
            c.eval.matvec_plain(&broken, &x).unwrap_err(),
            HekitError::ScaleMismatch
        );
        let mut broken = back;
        broken.entries_mut()[0] = c.enc.encode(&diags[0], s, top - 1).unwrap();
        assert_eq!(
            c.eval.matvec_plain(&broken, &x).unwrap_err(),
            HekitError::LevelMismatch
        );
    }

    /// I1: `try_map` keeps the shape (e.g. decrypting the diagonals gives the pre-rotated
    /// values) and propagates errors; `mod_switch_to_level` gives diagonals that multiply a
    /// lower-level `x`.
    #[test]
    fn try_map_and_mod_switch() {
        let d = 8;
        let c = ctx(small(), d, 1, 30);
        let (slots, top, s) = (c.params.slots(), c.params.max_level(), c.params.scale());
        let mut r = rng(31);
        let w = rand_matrix(&mut r, d);
        let diags = diagonals_from_matrix(&w, slots, 1);
        let ce = Diagonals::<Ciphertext>::encrypt(&c.enc, &c.encryptor, &diags, 1, s, top, &mut r)
            .unwrap();
        let (_, _, pre) = prerotated(&diags, slots, 1).unwrap();
        let dec: Diagonals<Vec<f64>> = ce
            .try_map(|ct| Ok(c.enc.decode(&c.dec.decrypt(ct))))
            .unwrap();
        assert_eq!(
            (dec.d(), dec.baby(), dec.giant(), dec.stride()),
            (8, 4, 2, 1)
        );
        for (got, want) in dec.entries().iter().zip(&pre) {
            let err = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(err < 2f64.powi(-20), "decrypted diagonal error {err:e}");
        }
        let failed: Result<Diagonals<()>> = ce.try_map(|_| Err(HekitError::NoLevelsLeft));
        assert_eq!(failed.unwrap_err(), HekitError::NoLevelsLeft);

        // Down to level 1, then the product with a level-1 x.
        let low = ce.mod_switch_to_level(&c.eval, 1).unwrap();
        assert!(low.entries().iter().all(|e| e.level() == 1));
        assert!(matches!(
            ce.mod_switch_to_level(&c.eval, top + 1),
            Err(HekitError::InvalidParams(_))
        ));
        let x = rand_vals(&mut r, d);
        let xr: Vec<f64> = (0..slots).map(|j| x[j % d]).collect();
        let ct_x = c
            .encryptor
            .encrypt(&c.enc.encode(&xr, s, 1).unwrap(), &mut r);
        let y = c.eval.matvec_enc(&low, &ct_x).unwrap();
        assert_eq!(y.level(), 1);
        let got = c.enc.decode(&c.dec.decrypt(&c.eval.rescale(&y).unwrap()));
        let want: Vec<f64> = (0..d)
            .map(|i| (0..d).map(|j| w[[i, j]] * x[j]).sum())
            .collect();
        let err = got
            .iter()
            .enumerate()
            .map(|(j, g)| (g - want[j % d]).abs())
            .fold(0.0, f64::max);
        assert!(err < 2f64.powi(-15), "level-1 matvec_enc error {err:e}");
    }
}
