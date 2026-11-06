//! The ReBoot packing (Pirillo & Colombo, "ReBoot", arXiv:2506.19693, AAAI-26) on hekit:
//! the in-library baseline (c) of the plan, so that layouts are compared on the same library.
//!
//! **Provenance.** ReBoot's code is GPL-3.0. This module is written from the paper only
//! (§3 eq. 4–5, §4.1 packing, §4.2.1 RE/CE-Matmul, §4.3 forward / backward, Alg. 1); no
//! ReBoot source was read, copied or paraphrased. Where the paper leaves a choice, the cheaper
//! one is taken (M0 ruling: model ReBoot generously) and noted below.
//!
//! # Packing (§3, §4.1)
//!
//! One ciphertext holds one `r x c` matrix, `r c = N/2`, flattened **row-major**: slot
//! `i c + j` holds entry `(i, j)` (§3 writes the flattened matrix `[W_11, ..., W_1c, W_21,
//! ...]`). Both `r` and `c` are powers of two. One sample per ciphertext.
//!
//! * **expanded** vector `a` (`len <= r`, [`RebootLayout::pack_expanded`]): every column is a
//!   copy of `a`, zero-padded to `r`: slot `i c + j = a[i]`.
//! * **repeated** vector `a` (`len <= c`, [`RebootLayout::pack_repeated`]): every row is a copy
//!   of `a`, zero-padded to `c`: slot `i c + j = a[j]`.
//! * **row-encoded** `W` (`d x k`, `d <= r`, `k <= c`, [`RebootLayout::pack_row`]): `W` itself,
//!   zero-padded: slot `i c + j = W[i][j]`.
//! * **column-encoded** `W` (`d x k`, `k <= r`, `d <= c`, [`RebootLayout::pack_col`]): `W^T`,
//!   zero-padded: slot `i c + j = W[j][i]`.
//!
//! `W` is `d_in x d_out` and a layer computes the row vector `z = a W` (§4.1).
//!
//! # Operations and their cost (rotation = left, as the paper's `Rot(a, k) = [a_{k+1}, ...]`)
//!
//! | op | paper | rotations | relins | ct-pt | levels |
//! |---|---|---|---|---|---|
//! | [`RebootLayout::row_sum`] | eq. 4 (sum across rows) | `log2 r` | 0 | 0 | 0 |
//! | [`RebootLayout::expand`] | eq. 5 (sum across columns) | `2 log2 c` | 0 | 1 (mask) | 1 |
//! | [`RebootLayout::vecmat`] | RE-Matmul §4.2.1 | `log2 r` | 1 | 0 | 1 |
//! | [`RebootLayout::vecmat_col`] | CE-Matmul §4.2.1 | `2 log2 c` | 1 | 1 (mask) | 2 |
//! | [`RebootLayout::outer`] | weight gradient §4.3.3 | 0 | 1 | 0 | 1 |
//! | [`RebootLayout::outer_sum`] | batch gradient (sum) | 0 | `B` or 1 | 0 | 1 |
//!
//! The levels are the paper's depths (§4.2.1: RE-Matmul depth 1, CE-Matmul depth 2 "the
//! element-wise multiplication and the masking step"). Every op rescales internally, so each
//! returned ciphertext is ready for the next one. The eq. 5 mask cannot be folded into
//! anything: it acts on the rotated sum of an encrypted product (the weights are encrypted),
//! so it costs its level; a constant can be folded into it for free
//! ([`RebootLayout::expand_to_scale`]'s `mask_value`).
//!
//! * `row_sum` (eq. 4): `s <- s + Rot(s, c 2^(log2 r - 1 - t))`, `t = 0 .. log2 r - 1`; every
//!   slot `(i, j)` ends with the column sum `sum_i' s(i', j)` (the rotations by multiples of
//!   `c` wrap cyclically over the `r` rows because `r c` is the slot count).
//! * `expand` (eq. 5): (1) `u <- u + Rot(u, 2^(log2 c - 1 - t))`: slot `(i, 0)` holds the row
//!   sum, the other slots garbage (they mix the next row); (2) mask `m` = 1 at slot `(i, 0)`;
//!   (3) `s <- s + Rot(s, -2^t)`, `t = 0 .. log2 c - 1`: the row sum fills its row. The result
//!   is in the **expanded** format when the input was `a_rep (.) W_col`.
//! * RE-Matmul `z_rep = row_sum(a_exp (.) W_row)`: slot `(i, j) = sum_i' a[i'] W[i'][j] =
//!   (a W)[j]`, the **repeated** format (§4.2.1).
//! * CE-Matmul `z_exp = expand(a_rep (.) W_col)`: slot `(i, j) = sum_j' a[j'] W[j'][i] =
//!   (a W)[i]`, the **expanded** format.
//! * So an RE layer's output feeds a CE layer and vice versa: layers **alternate** row / column
//!   encoding (§4.2.1: odd layers RE, even layers CE) and nothing is ever repacked.
//! * Backward (§4.3.3): the column encoding of `W` is the row encoding of `W^T`, so the input
//!   gradient of a CE layer is RE-Matmul of its (expanded) output gradient with the same
//!   `W_col` (`vecmat(g_exp, W_col)` = `(W g)` repeated), and that of an RE layer is CE-Matmul
//!   of its (repeated) output gradient with the same `W_row` (`vecmat_col(dz_rep, W_row)` =
//!   `(W dz)` expanded). No transpose, no extra op.
//! * Weight gradients (§4.3.3): `dW_row = a_exp (.) dz_rep` (slot `(i, j) = a[i] dz[j]`) and
//!   `dW_col = a_rep (.) g_exp` (slot `(i, j) = g[i] a[j]`): one element-wise product
//!   ([`RebootLayout::outer`], which takes the expanded operand first); the result is directly
//!   in the layer's own encoding.
//! * Mini-batch: the paper (Alg. 1) shows one sample; one sample per ciphertext and the batch
//!   gradient as the **sum** of the per-sample products is this baseline's reading
//!   ([`RebootLayout::outer_sum`]; scale by `lr / B` in the update).
//!
//! # Generous choices (where the paper leaves one)
//!
//! * The rotate-and-add sequences of eq. 4 / eq. 5 are the cheapest correct sequences in
//!   rotation count (`log2` many). They are sequentially dependent (each rotates the previous
//!   partial sum), so they cannot be hoisted; hoisting `r - 1` rotations of one ciphertext
//!   would cost more key switches, not fewer.
//! * Every product is rescaled **before** its rotations (rotations at the lower level are
//!   cheaper; the counts are the same).
//! * The batch gradient sums the unrescaled products and rescales once. The paper does not
//!   mention relinearization at all, so the policy is a parameter ([`GradRelin`]):
//!   [`GradRelin::Lazy`] (one relin per batch sum, the cost model's hypothetical
//!   `lazy_relin=True`) is the generous baseline and the RQ1 headline; [`GradRelin::PerSample`]
//!   (one relin per product, the M0 cost model's `_reboot` default) is the baseline "as
//!   specified" by the cost model. Report both.
//!
//!   Closed forms for the eMLP-1 step (RE layer, activation, CE layer, backward, two
//!   gradients) per sample: `2 log2 r + 2 log2 c` rotations and 7 relins ([`GradRelin::PerSample`],
//!   `2 log2 r + 2 log2 c + 7` key switches). With [`GradRelin::Lazy`]: 5 relins per sample plus
//!   one per `outer_sum`, i.e. per batch of `B` the key switches are
//!   `B (2 log2 r + 2 log2 c + 5) + 2` (`35.0625` per sample at `(1024, 32)`, `B = 32`).
//!
//! # Not modelled
//!
//! * Alg. 1 line 10's delayed gradient (the paper applies a gradient computed one step
//!   earlier) is not modelled: the update here uses the current batch's gradient. It changes
//!   accuracy only, never an operation count.
//! * Padding and refresh: slots outside the `d x k` block (or the vector's length) are zero
//!   in a fresh packing, but after homomorphic work they only hold noise-sized values, and the
//!   eq. 5 intermediate slots are garbage by construction. A client that refreshes a value
//!   (decrypt, re-encrypt) must therefore **re-pack** it with the `pack_*` / `unpack_*`
//!   functions from the true `d x k` entries, never re-encrypt the raw slot vector: otherwise
//!   the padding drifts. With zero padding, `vecmat` / `outer` keep padded slots at zero.
//!
//! # Exact-scale path (M2 rulings)
//!
//! The eq. 5 mask is encoded at [`Evaluator::target_plain_scale`] of the rotated sum and
//! multiplied with [`Evaluator::mul_plain_to_scale`], so the expanded result has **exactly**
//! the target scale (`Delta = params.scale()` for [`RebootLayout::expand`] and
//! [`RebootLayout::vecmat_col`]; always reachable, it is a power of two). Additions inside
//! eq. 4 / eq. 5 add a ciphertext to its own rotation (identical scale); [`RebootLayout::outer_sum`]
//! requires identical product scales and levels across the batch (identical histories), so no
//! addition here relabels a scale.
//!
//! # Levels and errors
//!
//! Operands of a product are aligned by switching the higher one down (free). Every op checks
//! before any work (so a failure counts nothing): the layout's slot count equals the
//! evaluator's, every needed Galois key is present (`InvalidParams("missing galois key")`),
//! and enough levels are left (`NoLevelsLeft`). Galois keys: [`RebootLayout::row_sum_steps`]
//! for `row_sum` / `vecmat`, [`RebootLayout::expand_steps`] for `expand` / `vecmat_col`
//! ([`RebootLayout::galois_steps`] = both).

#[cfg(doc)]
use ckks_train_ckks::Evaluator;
use ckks_train_ckks::{DiagonalEntry, Encoder, HeOps};
use ckks_train_core::error::{HekitError, Result};
use ndarray::Array2;

/// Relinearization policy of [`RebootLayout::outer_sum`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradRelin {
    /// One relinearization per sample's product (`B` relins): the M0 cost model's `_reboot`
    /// default, the baseline "as specified".
    PerSample,
    /// Sum the degree-2 products and relinearize once (1 relin): the cost model's
    /// hypothetical `lazy_relin=True`: the generous baseline, the RQ1 headline. Honoured only
    /// when the evaluator's [`ckks_train_ckks::EvalConfig::lazy_relin`] flag is on (default); with
    /// it off (the RQ3 ablation) this acts as [`GradRelin::PerSample`].
    Lazy,
}

/// The ReBoot `r x c` single-ciphertext layout (`r c = slots`, both powers of two); see the
/// [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebootLayout {
    r: usize,
    c: usize,
}

fn invalid<T>(msg: String) -> Result<T> {
    Err(HekitError::InvalidParams(msg))
}

fn log2(n: usize) -> u32 {
    n.trailing_zeros()
}

impl RebootLayout {
    /// The `r x c` layout for `slots` slots. Errors (`InvalidParams`): `r` or `c` not a power
    /// of two, or `r c != slots`.
    pub fn new(r: usize, c: usize, slots: usize) -> Result<RebootLayout> {
        if !(r.is_power_of_two() && c.is_power_of_two()) {
            return invalid(format!("r = {r} and c = {c} must be powers of two"));
        }
        if r.checked_mul(c) != Some(slots) {
            return invalid(format!("r c = {r} x {c} must equal the slot count {slots}"));
        }
        Ok(RebootLayout { r, c })
    }

    /// Rows `r`.
    pub fn r(&self) -> usize {
        self.r
    }

    /// Columns `c`.
    pub fn c(&self) -> usize {
        self.c
    }

    /// Slots `r c`.
    pub fn slots(&self) -> usize {
        self.r * self.c
    }

    /// The rotation steps of eq. 4 in order: `c 2^(log2 r - 1 - t)`, `t = 0 .. log2 r - 1`.
    pub fn row_sum_steps(&self) -> Vec<i32> {
        (0..log2(self.r))
            .rev()
            .map(|e| (self.c << e) as i32)
            .collect()
    }

    /// The rotation steps of eq. 5 in order: `2^(log2 c - 1 - t)` (row sums), then `-2^t`
    /// (replication), `t = 0 .. log2 c - 1`.
    pub fn expand_steps(&self) -> Vec<i32> {
        let k = log2(self.c);
        let down = (0..k).rev().map(|e| 1i32 << e);
        let up = (0..k).map(|e| -(1i32 << e));
        down.chain(up).collect()
    }

    /// All Galois key steps of the layout ([`RebootLayout::row_sum_steps`] and
    /// [`RebootLayout::expand_steps`]), sorted, without duplicates.
    pub fn galois_steps(&self) -> Vec<i32> {
        let mut s = self.row_sum_steps();
        s.extend(self.expand_steps());
        s.sort_unstable();
        s.dedup();
        s
    }

    /// The **expanded** format of `a` (`len <= r`): slot `i c + j = a[i]`, zero-padded.
    pub fn pack_expanded(&self, a: &[f64]) -> Result<Vec<f64>> {
        if a.len() > self.r {
            return invalid(format!(
                "expanded vector of length {} > r = {}",
                a.len(),
                self.r
            ));
        }
        let mut v = vec![0.0; self.slots()];
        for (i, &x) in a.iter().enumerate() {
            v[i * self.c..(i + 1) * self.c].fill(x);
        }
        Ok(v)
    }

    /// The **repeated** format of `a` (`len <= c`): slot `i c + j = a[j]`, zero-padded.
    pub fn pack_repeated(&self, a: &[f64]) -> Result<Vec<f64>> {
        if a.len() > self.c {
            return invalid(format!(
                "repeated vector of length {} > c = {}",
                a.len(),
                self.c
            ));
        }
        let mut v = vec![0.0; self.slots()];
        for row in v.chunks_mut(self.c) {
            row[..a.len()].copy_from_slice(a);
        }
        Ok(v)
    }

    /// The **row encoding** of `W` (`d x k`, `d <= r`, `k <= c`): slot `i c + j = W[i][j]`.
    pub fn pack_row(&self, w: &Array2<f64>) -> Result<Vec<f64>> {
        let (d, k) = w.dim();
        if d > self.r || k > self.c {
            return invalid(format!(
                "row-encoded {d} x {k} exceeds r x c = {} x {}",
                self.r, self.c
            ));
        }
        let mut v = vec![0.0; self.slots()];
        for ((i, j), &x) in w.indexed_iter() {
            v[i * self.c + j] = x;
        }
        Ok(v)
    }

    /// The **column encoding** of `W` (`d x k`, `k <= r`, `d <= c`), i.e. `W^T` zero-padded:
    /// slot `i c + j = W[j][i]`.
    pub fn pack_col(&self, w: &Array2<f64>) -> Result<Vec<f64>> {
        self.pack_row(&w.t().to_owned())
    }

    /// The first `len` entries of an expanded vector (column 0).
    pub fn unpack_expanded(&self, slots: &[f64], len: usize) -> Vec<f64> {
        (0..len).map(|i| slots[i * self.c]).collect()
    }

    /// The first `len` entries of a repeated vector (row 0).
    pub fn unpack_repeated(&self, slots: &[f64], len: usize) -> Vec<f64> {
        slots[..len].to_vec()
    }

    /// The `d x k` matrix `W` of a row encoding.
    pub fn unpack_row(&self, slots: &[f64], d: usize, k: usize) -> Array2<f64> {
        Array2::from_shape_fn((d, k), |(i, j)| slots[i * self.c + j])
    }

    /// The `d x k` matrix `W` of a column encoding (slot `i c + j = W[j][i]`).
    pub fn unpack_col(&self, slots: &[f64], d: usize, k: usize) -> Array2<f64> {
        Array2::from_shape_fn((d, k), |(i, j)| slots[j * self.c + i])
    }

    /// The eq. 5 mask: 1 at the first slot of every row (`i c`), 0 elsewhere.
    pub fn first_column_mask(&self) -> Vec<f64> {
        (0..self.slots())
            .map(|k| if k % self.c == 0 { 1.0 } else { 0.0 })
            .collect()
    }

    /// Eq. 4, summation across rows: every slot `(i, j)` gets the column sum
    /// `sum_i' m(i', j)`. `log2 r` rotations, no level.
    ///
    /// Errors: `InvalidParams` for a slot-count mismatch or a missing Galois key (checked
    /// first); evaluator errors.
    pub fn row_sum<E: HeOps>(&self, ev: &E, m: &E::Ct) -> Result<E::Ct> {
        self.check(ev, &self.row_sum_steps(), m.level(), 0)?;
        self.row_sum_unchecked(ev, m.clone())
    }

    /// Eq. 5, summation across columns: every slot `(i, j)` gets the row sum
    /// `sum_j' m(i, j')`, at exactly the scale `Delta = params.scale()`, one level below `m`.
    /// `2 log2 c` rotations, 1 ct-pt (mask), 1 rescale.
    ///
    /// The `Encoder` encodes the mask (at `m`'s level and the exact-scale plaintext scale).
    /// Errors as [`RebootLayout::row_sum`], plus `NoLevelsLeft` at level 0.
    pub fn expand<E: HeOps>(&self, ev: &E, enc: &Encoder, m: &E::Ct) -> Result<E::Ct> {
        self.expand_to_scale(ev, enc, m, 1.0, ev.params().scale())
    }

    /// [`RebootLayout::expand`] with the constant `mask_value` folded into the mask (the
    /// result is `mask_value` times the row sums, at no extra cost) and an explicit target
    /// scale (exactly the result's scale; see [`Evaluator::exact_product_scale`] for which
    /// targets are reachable, e.g. any power of two).
    pub fn expand_to_scale<E: HeOps>(
        &self,
        ev: &E,
        enc: &Encoder,
        m: &E::Ct,
        mask_value: f64,
        target_scale: f64,
    ) -> Result<E::Ct> {
        self.check(ev, &self.expand_steps(), m.level(), 1)?;
        if !mask_value.is_finite() {
            return invalid(format!("mask value {mask_value} must be finite"));
        }
        let ps = ev.target_plain_scale(m, target_scale)?;
        self.expand_unchecked(ev, enc, m.clone(), mask_value, target_scale, ps)
    }

    /// RE-Matmul (§4.2.1): `z_rep = row_sum(x_exp (.) w_row)`, the **repeated** `x W` for `x`
    /// expanded and `W` row-encoded (or, in the backward pass, `W g` for `g` expanded and
    /// `W` column-encoded). 1 ct-ct mult (1 relin), 1 rescale, `log2 r` rotations: depth 1.
    /// The result is at `min(level) - 1` with scale `s_x s_w / q_l`.
    ///
    /// Errors: as [`RebootLayout::row_sum`], plus `NoLevelsLeft` at level 0; evaluator errors
    /// (missing relin key).
    pub fn vecmat<E: HeOps>(&self, ev: &E, x_expanded: &E::Ct, w: &E::Ct) -> Result<E::Ct> {
        let level = x_expanded.level().min(w.level());
        self.check(ev, &self.row_sum_steps(), level, 1)?;
        let p = self.product(ev, x_expanded, w)?;
        self.row_sum_unchecked(ev, p)
    }

    /// CE-Matmul (§4.2.1): `z_exp = expand(x_rep (.) w_col)`, the **expanded** `x W` for `x`
    /// repeated and `W` column-encoded (or, backward, `W dz` for `dz` repeated and `W`
    /// row-encoded). 1 ct-ct mult (1 relin), 1 mask, 2 rescales, `2 log2 c` rotations: depth 2.
    /// The result is at `min(level) - 2` with scale exactly `Delta`.
    ///
    /// Errors as [`RebootLayout::vecmat`] (`NoLevelsLeft` below level 2).
    pub fn vecmat_col<E: HeOps>(
        &self,
        ev: &E,
        enc: &Encoder,
        x_repeated: &E::Ct,
        w: &E::Ct,
    ) -> Result<E::Ct> {
        let level = x_repeated.level().min(w.level());
        self.check(ev, &self.expand_steps(), level, 2)?;
        let target = ev.params().scale();
        let p = self.product(ev, x_repeated, w)?;
        let ps = ev.target_plain_scale(&p, target)?;
        self.expand_unchecked(ev, enc, p, 1.0, target, ps)
    }

    /// The weight gradient of one sample (§4.3.3): `x_exp (.) y_rep`, slot `(i, j) = x[i]
    /// y[j]`, rescaled. With `x = a` (layer input) and `y = dz` this is `dW_row`; with `x = g`
    /// (output gradient) and `y = a` it is `dW_col` (= `g^T a` row-major). 1 ct-ct mult
    /// (1 relin), 1 rescale, no rotation; result at `min(level) - 1`.
    ///
    /// Errors: `InvalidParams` for a slot-count mismatch, `NoLevelsLeft` at level 0 (checked
    /// first); evaluator errors.
    pub fn outer<E: HeOps>(&self, ev: &E, x_expanded: &E::Ct, y_repeated: &E::Ct) -> Result<E::Ct> {
        let level = x_expanded.level().min(y_repeated.level());
        self.check(ev, &[], level, 1)?;
        self.product(ev, x_expanded, y_repeated)
    }

    /// The mini-batch weight gradient `sum_b x_b (.) y_b` over one ciphertext pair per sample:
    /// `B` ct-ct tensors, `B` relins ([`GradRelin::PerSample`]) or 1 ([`GradRelin::Lazy`]),
    /// one rescale of the sum. All pairs must align to the same level with the same product
    /// scale (identical histories; no relabelling).
    ///
    /// Errors (checked first): `InvalidParams` for an empty batch or a slot-count mismatch,
    /// `LevelMismatch` / `ScaleMismatch` across samples, `NoLevelsLeft` at level 0; evaluator
    /// errors.
    pub fn outer_sum<E: HeOps>(
        &self,
        ev: &E,
        pairs: &[(&E::Ct, &E::Ct)],
        relin: GradRelin,
    ) -> Result<E::Ct> {
        let Some(&(x0, y0)) = pairs.first() else {
            return invalid("outer_sum of an empty batch".into());
        };
        let level = x0.level().min(y0.level());
        let scale = x0.scale() * y0.scale();
        for &(x, y) in pairs {
            if x.level().min(y.level()) != level {
                return Err(HekitError::LevelMismatch);
            }
            if x.scale() * y.scale() != scale {
                return Err(HekitError::ScaleMismatch);
            }
        }
        self.check(ev, &[], level, 1)?;
        // `Lazy` is honoured only with the evaluator's `lazy_relin` flag (RQ3 ablation).
        let lazy = relin == GradRelin::Lazy && ev.config().lazy_relin;
        let mut acc: Option<E::Ct> = None;
        for &(x, y) in pairs {
            let (x, y) = (
                ev.mod_switch_to_level(x, level)?,
                ev.mod_switch_to_level(y, level)?,
            );
            let p = if lazy {
                ev.mul_no_relin(&x, &y)?
            } else {
                ev.mul(&x, &y)?
            };
            acc = Some(match acc {
                None => p,
                Some(a) => ev.add(&a, &p)?,
            });
        }
        let mut s = acc.expect("non-empty batch");
        if lazy {
            s = ev.relinearize(&s)?;
        }
        ev.rescale(&s)
    }

    /// Checks the slot count, the Galois keys for `steps` and that `level >= need`.
    fn check<E: HeOps>(&self, ev: &E, steps: &[i32], level: usize, need: usize) -> Result<()> {
        let slots = ev.params().slots();
        if slots != self.slots() {
            return invalid(format!(
                "layout {} x {} is for {} slots, the evaluator has {slots}",
                self.r,
                self.c,
                self.slots()
            ));
        }
        if steps.iter().any(|&s| !ev.has_galois_key(s)) {
            return invalid("missing galois key".into());
        }
        if level < need {
            return Err(HekitError::NoLevelsLeft);
        }
        Ok(())
    }

    /// `rescale(a * b)` after switching the higher operand down.
    fn product<E: HeOps>(&self, ev: &E, a: &E::Ct, b: &E::Ct) -> Result<E::Ct> {
        let level = a.level().min(b.level());
        let a = ev.mod_switch_to_level(a, level)?;
        let b = ev.mod_switch_to_level(b, level)?;
        ev.rescale(&ev.mul(&a, &b)?)
    }

    /// Eq. 4 without the checks.
    fn row_sum_unchecked<E: HeOps>(&self, ev: &E, mut s: E::Ct) -> Result<E::Ct> {
        for step in self.row_sum_steps() {
            s = ev.add(&s, &ev.rotate(&s, step)?)?;
        }
        Ok(s)
    }

    /// Eq. 5 without the checks; `plain_scale` is `target_plain_scale(u, target_scale)`.
    fn expand_unchecked<E: HeOps>(
        &self,
        ev: &E,
        enc: &Encoder,
        mut u: E::Ct,
        mask_value: f64,
        target_scale: f64,
        plain_scale: f64,
    ) -> Result<E::Ct> {
        let k = log2(self.c);
        for e in (0..k).rev() {
            u = ev.add(&u, &ev.rotate(&u, 1 << e)?)?;
        }
        let mask: Vec<f64> = self
            .first_column_mask()
            .into_iter()
            .map(|m| m * mask_value)
            .collect();
        let pt = ev.upload_pt_owned(enc.encode(&mask, plain_scale, u.level())?)?;
        let mut s = ev.rescale(&ev.mul_plain_to_scale(&u, &pt, target_scale)?)?;
        for e in 0..k {
            s = ev.add(&s, &ev.rotate(&s, -(1 << e))?)?;
        }
        Ok(s)
    }
}
