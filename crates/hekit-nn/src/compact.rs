//! Compact re-packing of the encrypted weight diagonals for bootstrapping (M3, Task 40;
//! `docs/layout.md` §6).
//!
//! A layer's `m` diagonals ([`EncWeights`]) are each replicated over the `S / L` slot
//! positions of the layout (`S = slots`, `L = max(n_out, n_in)`). [`compact_pack`] keeps one
//! copy of each and puts `cap = S / L` diagonals into one ciphertext, so a layer is
//! bootstrapped in [`boot_count`] `= ceil(m / cap)` ciphertexts instead of `m`.
//! [`compact_unpack`] restores the replicated, BSGS pre-rotated diagonals for the next
//! forward. This is premise 1 of the M0 cost model (`spikes/costmodel/costmodel.py`,
//! `_diag_batched`, `compact=True`), and the rotation counts are the model's:
//!
//! | op | rotations | ct-pt | relins | levels |
//! |---|---|---|---|---|
//! | [`compact_pack`] | `m - n_ct` | `m` | 0 | 1 |
//! | [`compact_unpack`] | `(m - n_ct) + m log2(S / L)` | `m` | 0 | 1 |
//! | [`compact_shift`] (pack without its mask, Task 41) | `m - n_ct` | 0 | 0 | 0 |
//! | [`compact_extract`] (one position, e.g. a bias, Task 41) | `[o(t) != 0] + log2(S / L)` | 1 | 0 | 1 |
//!
//! The non-interactive trainer ([`crate::trainer_ni`], `docs/layout.md` §6.4 and §7) keeps the
//! weights, the NAG velocity and the biases in compact form between steps: the masked,
//! reduce-only gradient goes in by [`compact_shift`] (its batch-sum mask is the pack mask),
//! and the weights and biases come out by [`compact_unpack`] and [`compact_extract`].
//!
//! # Layout of a compact ciphertext
//!
//! Diagonal entry `c` goes to ciphertext `c div cap` at position `t = c mod cap`, with slot
//! offset [`CompactPlan::offset`]`(t) = (t mod B) + (t >> lane_shift) L B`: the low bits of `t`
//! choose a lane, the high bits a copy of the `L`-period. Slot `o(t) + j B` holds entry `c`'s
//! value at feature `j < L`; every other slot is 0. The offset is linear in the bits of `t`,
//! so both directions are binary trees of rotations by the [`CompactPlan::steps`] (one per
//! merge or split). They use `log2(min(m, cap))` shift keys, not `cap - 1`
//! ([`compact_rotation_steps`]).
//!
//! # Re-laying replicated layouts (ruling R-M3-3)
//!
//! Pack reads only lane 0 of the first period of every diagonal (slots `j B`, `j < L`), and
//! unpack writes every lane and every block copy from that one slot. A pack → unpack round
//! trip therefore returns block-periodic diagonals (all copies identical up to the
//! ciphertext noise), whatever drift the input copies had. So the compact path refreshes a
//! replicated layout by bootstrapping, which [`crate::refresh::BootstrapRefresher`] alone
//! cannot do. It also means pack accepts an input that is valid only in lane 0 (the
//! reduce-only gradient of `docs/layout.md` §3).
//!
//! # Scales and levels
//!
//! Both masks are encoded at [`Evaluator::target_plain_scale`](hekit_ckks::Evaluator::target_plain_scale) and applied with
//! [`Evaluator::mul_plain_to_scale`](hekit_ckks::Evaluator::mul_plain_to_scale), so every output has **exactly** its input's scale:
//! [`compact_pack`] returns `level(W) - 1`, and [`compact_unpack`] returns
//! `level(packed) - 1`. On n16_boot, weights that are packed, bootstrapped (to level 16) and
//! unpacked are on level 15 at `Delta`.
//!
//! # Deviations from the brief's interface
//!
//! - `diags_per_ct = slots / L` with `L = max(n_out, n_in)`, the diagonal length. The brief's
//!   `slots / d_in` is the wide-layer case (`L = n_in`). For a tall layer (`n_out > n_in`) the
//!   diagonals have length `n_out`, and `slots / d_in` copies would overlap.
//! - [`boot_count`] counts the `min(pow2(d_out), pow2(d_in))` padded diagonals. That is the
//!   brief's `ceil(d_out / diags_per_ct)` for a wide layer with a power-of-two `d_out`. A
//!   padded `d_out` still has `pow2(d_out)` nonzero diagonals (`docs/layout.md` §2).
//! - The pack and unpack shifts are binary trees (same rotation count as one lane shift per
//!   diagonal, far fewer Galois keys). The unpack tree rotates distinct ciphertexts, so
//!   [`Evaluator::hoisted_rotate`](hekit_ckks::Evaluator::hoisted_rotate) has nothing to share and is not used.

use std::sync::Arc;

use hekit_ckks::linalg::Diagonals;
use hekit_ckks::{DiagonalEntry, Encoder, HeCiphertext, HeOps};
use hekit_core::error::{HekitError, Result};

use crate::dense::EncWeights;
use crate::layout::{LayerShape, Layout};

/// The compact re-packing of one `d_out x d_in` layer in a [`Layout`]
/// (`docs/layout.md` §6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactPlan {
    /// Output dimension (rows of `W`).
    pub d_out: usize,
    /// Input dimension (columns of `W`).
    pub d_in: usize,
    /// Lanes of the layout (`B`).
    pub batch: usize,
    /// Diagonals per compact ciphertext, `cap = slots / L` (`L = max(n_out, n_in)`).
    pub diags_per_ct: usize,
    /// `log2(batch)`: position `t` takes lane `t mod B` and period copy `t >> lane_shift`.
    pub lane_shift: usize,
}

impl CompactPlan {
    /// The plan of a `d_out x d_in` layer in `layout`.
    ///
    /// # Panics
    /// If the layer does not fit `layout` ([`LayerShape::new`]).
    pub fn for_layer(layout: &Layout, d_out: usize, d_in: usize) -> CompactPlan {
        let sh = LayerShape::new(d_out, d_in, layout);
        CompactPlan {
            d_out,
            d_in,
            batch: layout.batch,
            diags_per_ct: layout.slots / sh.l,
            lane_shift: layout.log_batch(),
        }
    }

    /// The diagonal length `L = max(pow2(d_out), pow2(d_in))` in features.
    pub fn diag_len(&self) -> usize {
        self.d_out
            .next_power_of_two()
            .max(self.d_in.next_power_of_two())
    }

    /// The number of diagonals `m = min(pow2(d_out), pow2(d_in))`.
    pub fn diagonals(&self) -> usize {
        self.d_out
            .next_power_of_two()
            .min(self.d_in.next_power_of_two())
    }

    /// The slot count `S = diags_per_ct * L`.
    pub fn slots(&self) -> usize {
        self.diags_per_ct * self.diag_len()
    }

    /// The number of compact ciphertexts, `ceil(m / diags_per_ct)`.
    pub fn ciphertexts(&self) -> usize {
        self.diagonals().div_ceil(self.diags_per_ct)
    }

    /// The diagonals in compact ciphertext `i`: `min(cap, m - i cap)` (0 past the last).
    pub fn diagonals_in(&self, i: usize) -> usize {
        self.diagonals()
            .saturating_sub(i * self.diags_per_ct)
            .min(self.diags_per_ct)
    }

    /// The slot offset of position `t < diags_per_ct`: `(t mod B) + (t >> lane_shift) L B`.
    pub fn offset(&self, t: usize) -> usize {
        (t % self.batch) + (t >> self.lane_shift) * self.diag_len() * self.batch
    }

    /// The tree steps `step_k`, `k < log2(diags_per_ct)`: `2^k` for the lane bits
    /// (`k < lane_shift`), then `L B 2^(k - lane_shift)` for the period bits. `offset(t)` is
    /// the sum of `step_k` over the set bits `k` of `t`.
    pub fn steps(&self) -> Vec<usize> {
        let bits = self.diags_per_ct.trailing_zeros() as usize;
        (0..bits).map(|k| self.offset(1 << k)).collect()
    }

    /// The number of tree levels a ciphertext of `n` diagonals uses: `ceil(log2 n)`.
    pub(crate) fn tree_bits(n: usize) -> usize {
        n.next_power_of_two().trailing_zeros() as usize
    }

    /// The rotations of [`compact_pack`]: `m - n_ct`.
    pub fn pack_rotations(&self) -> usize {
        self.diagonals() - self.ciphertexts()
    }

    /// The rotations of [`compact_unpack`]: `(m - n_ct) + m log2(diags_per_ct)`.
    pub fn unpack_rotations(&self) -> usize {
        self.pack_rotations() + self.diagonals() * self.diags_per_ct.trailing_zeros() as usize
    }

    /// `InvalidParams` unless the plan's slot count is `slots` (the parameters').
    fn check_layout_slots(&self, slots: usize) -> Result<()> {
        if self.slots() != slots {
            return Err(HekitError::InvalidParams(format!(
                "compact plan for {} slots, parameters with {slots} slots",
                self.slots()
            )));
        }
        Ok(())
    }
}

/// The number of bootstraps (compact ciphertexts) of a layer with `d_out` rows and the plan's
/// `d_in` columns: `ceil(min(pow2(d_out), pow2(d_in)) / diags_per_ct)`. For the plan's own
/// layer this is [`CompactPlan::ciphertexts`]. For a wide layer with a power-of-two `d_out` it
/// is the brief's `ceil(d_out / diags_per_ct)`.
pub fn boot_count(plan: &CompactPlan, d_out: usize) -> usize {
    let m = d_out.next_power_of_two().min(plan.d_in.next_power_of_two());
    m.div_ceil(plan.diags_per_ct)
}

/// The Galois steps of [`compact_pack`] and [`compact_unpack`], sorted, one per distinct
/// rotation (steps equal modulo the slot count share a key, e.g. `±S/2`):
/// `-step_k` for `k < log2(diags_per_ct)` (the pack tree and the replication) and `+step_k`
/// for `k < ceil(log2 min(m, diags_per_ct))` (the unpack tree). That is
/// `log2(S/L) + log2(min(m, S/L))` keys at most.
pub fn compact_rotation_steps(plan: &CompactPlan) -> Vec<i32> {
    let steps = plan.steps();
    let s = plan.slots() as i64;
    let split_bits = CompactPlan::tree_bits(plan.diagonals().min(plan.diags_per_ct));
    let mut out: Vec<i32> = steps.iter().map(|&k| -(k as i32)).collect();
    out.extend(steps[..split_bits].iter().map(|&k| k as i32));
    // One representative per element: keep the first of each class modulo the slot count.
    let mut seen = std::collections::HashSet::new();
    out.retain(|&k| seen.insert((k as i64).rem_euclid(s)));
    out.sort_unstable();
    out
}

/// `InvalidParams` unless every step has a Galois key in `ev`.
fn check_keys<E: HeOps>(ev: &E, steps: &[i32]) -> Result<()> {
    let have = |k: i32| ev.has_galois_key(k);
    match steps.iter().find(|&&k| !have(k)) {
        Some(k) => Err(HekitError::InvalidParams(format!(
            "compact: missing galois key for step {k}"
        ))),
        None => Ok(()),
    }
}

/// The values of `mask0`: 1 on lane 0 of the first period (slots `j B`, `j < L`), 0 elsewhere.
fn mask0_values(plan: &CompactPlan) -> Vec<f64> {
    let span = plan.diag_len() * plan.batch;
    (0..plan.slots())
        .map(|s| {
            if s < span && s % plan.batch == 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// `mask0` (lane 0 of the first period: slots `j B`, `j < L`) encoded so that
/// `rescale(mul_plain_to_scale(ct, mask, ct.scale()))` is exactly `ct.scale()` on
/// `ct.level() - 1`.
fn mask0<E: HeOps>(ev: &E, plan: &CompactPlan, ct: &E::Ct) -> Result<E::Pt> {
    let scale = ev.target_plain_scale(ct, ct.scale())?;
    ev.upload_pt_owned(Encoder::new(Arc::clone(ev.params())).encode(
        &mask0_values(plan),
        scale,
        ct.level(),
    )?)
}

/// `rescale(ct * mask)` at exactly `ct.scale()`.
fn apply_mask<E: HeOps>(ev: &E, ct: &E::Ct, mask: &E::Pt) -> Result<E::Ct> {
    ev.rescale(&ev.mul_plain_to_scale(ct, mask, ct.scale())?)
}

/// `InvalidParams` unless `ct` is a 2-component ciphertext over the evaluator's basis.
fn check_input<E: HeOps>(ev: &E, ct: &E::Ct) -> Result<()> {
    if !ev.owns(ct) || ct.size() != 2 {
        return Err(HekitError::InvalidParams(
            "compact: inputs must be 2-component ciphertexts over the evaluator's basis".into(),
        ));
    }
    Ok(())
}

/// Packs the layer's `m` diagonals into [`CompactPlan::ciphertexts`] compact ciphertexts
/// (`docs/layout.md` §6.2): per diagonal, `rescale(mul_plain_to_scale(P_c, mask0, scale))`
/// keeps lane 0 of the first period. Then [`compact_shift`] (per ciphertext, a tree over the
/// bits of the position `t` adds the groups as `left + rot(right, -step_k)`). Ciphertext `i`
/// holds entry `i cap + t` on slots `offset(t) + j B`, `j < L`.
///
/// Costs `m - n_ct` rotations, `m` ct-pt products, `m` rescales and no relin. The outputs are
/// on `w.level() - 1` with exactly `w.scale()`. Only slots `j B` (`j < L`) of each input
/// diagonal are read.
///
/// When the input is already zero outside those slots (a gradient masked by the batch-sum
/// mask, `docs/layout.md` §6.4), call [`compact_shift`] directly: the same rotations, no mask
/// and no level.
///
/// Errors, before any homomorphic work: `InvalidParams` if `w` is not the plan's layer
/// (`d_out`, `d_in`, lanes), the plan's slot count is not the parameters', an entry is not a
/// 2-component ciphertext over the evaluator's basis, or a key of [`compact_rotation_steps`]
/// is missing. Then those of [`Evaluator::target_plain_scale`](hekit_ckks::Evaluator::target_plain_scale) (`NoLevelsLeft` at level 0,
/// `ScaleMismatch` for a scale no rescale can keep).
pub fn compact_pack<E: HeOps>(
    ev: &E,
    w: &EncWeights<E::Ct>,
    plan: &CompactPlan,
) -> Result<Vec<E::Ct>> {
    let diags = w.diagonals();
    if w.d_out() != plan.d_out
        || w.d_in() != plan.d_in
        || diags.stride() != plan.batch
        || diags.d() != plan.diagonals()
    {
        return Err(HekitError::InvalidParams(format!(
            "compact_pack: weights {} x {} (stride {}) for a plan of {} x {} (batch {})",
            w.d_out(),
            w.d_in(),
            diags.stride(),
            plan.d_out,
            plan.d_in,
            plan.batch
        )));
    }
    plan.check_layout_slots(ev.params().slots())?;
    diags
        .entries()
        .iter()
        .try_for_each(|c| check_input(ev, c))?;
    check_keys(ev, &compact_rotation_steps(plan))?;
    let mask = mask0(ev, plan, &diags.entries()[0])?;
    let masked = diags
        .entries()
        .iter()
        .map(|p| apply_mask(ev, p, &mask))
        .collect::<Result<Vec<_>>>()?;
    Ok(shift_tree(ev, &masked, plan))
}

/// The mask-free half of [`compact_pack`] (`docs/layout.md` §6.4): `entries` are the layer's
/// `m` diagonals (BSGS order) that are **already zero outside** lane 0 of the first period
/// (slots `j B`, `j < L`; e.g. the reduce-only gradient after its batch-sum mask), and the
/// result is the [`CompactPlan::ciphertexts`] compact ciphertexts: per ciphertext, the tree
/// `left + rot(right, -step_k)` over the bits of the position. Ciphertext `i` holds entry
/// `i cap + t` on slots `offset(t) + j B`.
///
/// Costs exactly `m - n_ct` rotations ([`CompactPlan::pack_rotations`]) and nothing else: no
/// product, no level, no relin. The outputs keep the level and the scale of the inputs. This
/// is the cost model's "1 lane shift per diagonal into compact ciphertexts (m - nW
/// rotations)", whose mask is the batch-sum mask (`spikes/costmodel/costmodel.py`,
/// `_diag_batched`, `compact=True`).
///
/// The precondition is not checked (it is a property of the slots): a nonzero slot outside
/// `{j B : j < L}` lands on another position's slots and corrupts it.
///
/// Errors, before any homomorphic work: `InvalidParams` if there are not
/// [`CompactPlan::diagonals`] entries, the plan's slot count is not the parameters', an entry
/// is not a 2-component ciphertext over the evaluator's basis, or a key `-step_k` of the tree
/// is missing; `LevelMismatch` / `ScaleMismatch` unless all entries share one level and
/// exactly one scale.
pub fn compact_shift<E: HeOps>(
    ev: &E,
    entries: &[E::Ct],
    plan: &CompactPlan,
) -> Result<Vec<E::Ct>> {
    if entries.len() != plan.diagonals() {
        return Err(HekitError::InvalidParams(format!(
            "compact_shift: {} entries for a plan of {} diagonals",
            entries.len(),
            plan.diagonals()
        )));
    }
    plan.check_layout_slots(ev.params().slots())?;
    entries.iter().try_for_each(|c| check_input(ev, c))?;
    check_uniform_slice(entries)?;
    check_keys(ev, &shift_steps(plan))?;
    Ok(shift_tree(ev, entries, plan))
}

/// The Galois steps of [`compact_shift`]: `-step_k` for `k < ceil(log2 min(m, cap))`.
pub fn shift_steps(plan: &CompactPlan) -> Vec<i32> {
    let bits = CompactPlan::tree_bits(plan.diagonals().min(plan.diags_per_ct));
    plan.steps()[..bits].iter().map(|&k| -(k as i32)).collect()
}

/// `LevelMismatch` / `ScaleMismatch` unless all of `cts` share one level and one scale.
fn check_uniform_slice<C: DiagonalEntry>(cts: &[C]) -> Result<()> {
    for c in cts {
        if c.level() != cts[0].level() {
            return Err(HekitError::LevelMismatch);
        }
        if c.scale() != cts[0].scale() {
            return Err(HekitError::ScaleMismatch);
        }
    }
    Ok(())
}

/// The shift tree of [`compact_shift`] on checked inputs. Rotations by `-step_k` and
/// additions of equal level and scale cannot fail on checked inputs, so this panics instead
/// of returning an error there.
fn shift_tree<E: HeOps>(ev: &E, entries: &[E::Ct], plan: &CompactPlan) -> Vec<E::Ct> {
    let steps = plan.steps();
    let cap = plan.diags_per_ct;
    (0..plan.ciphertexts())
        .map(|i| {
            let n = plan.diagonals_in(i);
            let mut groups: Vec<E::Ct> = entries[i * cap..i * cap + n].to_vec();
            // Bit k: groups[2u] (bit 0) absorbs groups[2u + 1] (bit 1) shifted right by step_k.
            for &step in steps.iter().take(CompactPlan::tree_bits(n)) {
                let mut next = Vec::with_capacity(groups.len().div_ceil(2));
                let mut it = groups.into_iter();
                while let Some(left) = it.next() {
                    next.push(match it.next() {
                        Some(right) => {
                            let r = ev
                                .rotate(&right, -(step as i32))
                                .expect("checked: key present, 2 components");
                            ev.add(&left, &r).expect("checked: equal level and scale")
                        }
                        None => left,
                    });
                }
                groups = next;
            }
            groups.pop().expect("n >= 1 diagonals per ciphertext")
        })
        .collect()
}

/// The slots `offset(t) + j B` (`j < L`) of position `t`, as a plaintext layout helper: the
/// compact form of plaintext vectors. `entries[c]` is a vector of all slots whose lane-0,
/// first-period slots `j B` (`j < L`) hold the values of entry `c` (e.g.
/// [`crate::layout::pack_weight_diagonals`]); the result is the
/// [`CompactPlan::ciphertexts`] compact slot vectors (`docs/layout.md` §6.1), what
/// [`compact_pack`] computes homomorphically. For the client (encrypting an initial state)
/// and for tests.
///
/// # Panics
/// If `entries` does not have [`CompactPlan::diagonals`] vectors of [`CompactPlan::slots`]
/// values.
pub fn compact_pack_plain(entries: &[Vec<f64>], plan: &CompactPlan) -> Vec<Vec<f64>> {
    assert_eq!(entries.len(), plan.diagonals(), "one vector per diagonal");
    let (l, b, cap) = (plan.diag_len(), plan.batch, plan.diags_per_ct);
    (0..plan.ciphertexts())
        .map(|i| {
            let mut out = vec![0.0; plan.slots()];
            for t in 0..plan.diagonals_in(i) {
                place_plain(&mut out, &entries[i * cap + t], plan, t);
            }
            debug_assert!(l * b <= plan.slots());
            out
        })
        .collect()
}

/// Writes lane 0 of the first period of `v` (slots `j B`, `j < L`) to position `t` of the
/// compact slot vector `out` (slots `offset(t) + j B`).
///
/// # Panics
/// If `v` or `out` does not have [`CompactPlan::slots`] values or `t >= diags_per_ct`.
pub fn place_plain(out: &mut [f64], v: &[f64], plan: &CompactPlan, t: usize) {
    let (l, b) = (plan.diag_len(), plan.batch);
    assert!(t < plan.diags_per_ct, "position {t} out of range");
    assert_eq!(v.len(), plan.slots(), "input slot count");
    assert_eq!(out.len(), plan.slots(), "output slot count");
    let o = plan.offset(t);
    for j in 0..l {
        out[o + j * b] = v[j * b];
    }
}

/// The Galois steps of [`compact_extract`] at position `t`: `offset(t)` (if nonzero) and the replication steps `-step_k`, `k < log2(diags_per_ct)`.
pub fn extract_steps(plan: &CompactPlan, t: usize) -> Vec<i32> {
    let mut out: Vec<i32> = plan.steps().iter().map(|&k| -(k as i32)).collect();
    let o = plan.offset(t);
    if o != 0 {
        out.push(o as i32);
    }
    out.sort_unstable();
    out
}

/// One position of a compact ciphertext back in replicated form (`docs/layout.md` §7.3, the
/// bias path): `y = rescale(mul_plain_to_scale(rot(ct, offset(t)), mask0, target))`, then the
/// replication `y = y + rot(y, -step_k)` for every `k < log2(cap)`. Slot `j B + b + p L B`
/// of the result holds position `t`'s value at feature `j < L` for every lane `b` and copy `p`:
/// an `L`-periodic vector, i.e. the canonical form of any vector of period `n | L` stored
/// there (a bias of `n_out <= L` values).
///
/// Costs `[offset(t) != 0] + log2(cap)` rotations, 1 ct-pt, 1 rescale, no relin. The result
/// is on `ct.level() - 1` with **exactly** `target_scale` (the mask plaintext carries the
/// scale change, e.g. from the compact `Delta` to a bias's product scale).
///
/// Errors, before any homomorphic work: `InvalidParams` if `t >= diags_per_ct`, the plan's
/// slot count is not the parameters', `ct` is not a 2-component ciphertext over the
/// evaluator's basis, or a key of [`extract_steps`] is missing. Then those of
/// [`Evaluator::target_plain_scale`](hekit_ckks::Evaluator::target_plain_scale) (`NoLevelsLeft` at level 0, `ScaleMismatch` when
/// `target_scale` is not reachable from this level, `InvalidParams` for a plaintext scale
/// below `2^20`).
pub fn compact_extract<E: HeOps>(
    ev: &E,
    ct: &E::Ct,
    plan: &CompactPlan,
    t: usize,
    target_scale: f64,
) -> Result<E::Ct> {
    if t >= plan.diags_per_ct {
        return Err(HekitError::InvalidParams(format!(
            "compact_extract: position {t} of {}",
            plan.diags_per_ct
        )));
    }
    plan.check_layout_slots(ev.params().slots())?;
    check_input(ev, ct)?;
    check_keys(ev, &extract_steps(plan, t))?;
    let mask = mask0_values(plan);
    let sp = ev.target_plain_scale(ct, target_scale)?;
    let pt =
        ev.upload_pt_owned(Encoder::new(Arc::clone(ev.params())).encode(&mask, sp, ct.level())?)?;
    let leaf = ev.rotate(ct, plan.offset(t) as i32)?;
    let mut y = ev.rescale(&ev.mul_plain_to_scale(&leaf, &pt, target_scale)?)?;
    for &step in &plan.steps() {
        let r = ev.rotate(&y, -(step as i32))?;
        y = ev.add(&y, &r)?;
    }
    Ok(y)
}

/// Unpacks compact ciphertexts ([`compact_pack`]'s output, e.g. after bootstrapping) into
/// the layer's replicated, BSGS pre-rotated diagonals (`docs/layout.md` §6.3): per
/// ciphertext a tree from the top bit down splits each group into itself and
/// `rot(·, +step_k)`, so leaf `t` holds entry `i cap + t` on slots `j B`. Then per diagonal:
/// `rescale(mul_plain_to_scale(leaf, mask0, scale))`, and the replication
/// `y = y + rot(y, -step_k)` for every `k < log2(cap)` writes every lane and every block copy.
///
/// Costs `(m - n_ct) + m log2(S / L)` rotations, `m` ct-pt products, `m` rescales and no
/// relin. The weights are on `packed.level() - 1` with exactly `packed.scale()`, and
/// block-periodic by construction.
///
/// Errors, before any homomorphic work: `InvalidParams` if `packed` does not have
/// [`CompactPlan::ciphertexts`] entries, an entry is not a 2-component ciphertext over the
/// evaluator's basis, the plan's slot count is not the parameters', or a key of
/// [`compact_rotation_steps`] is missing; `LevelMismatch` / `ScaleMismatch` unless all entries
/// share one level and exactly one scale. Then those of [`Evaluator::target_plain_scale`](hekit_ckks::Evaluator::target_plain_scale).
pub fn compact_unpack<E: HeOps>(
    ev: &E,
    packed: &[E::Ct],
    plan: &CompactPlan,
) -> Result<EncWeights<E::Ct>> {
    if packed.len() != plan.ciphertexts() {
        return Err(HekitError::InvalidParams(format!(
            "compact_unpack: {} ciphertexts for a plan of {}",
            packed.len(),
            plan.ciphertexts()
        )));
    }
    plan.check_layout_slots(ev.params().slots())?;
    packed.iter().try_for_each(|c| check_input(ev, c))?;
    for c in packed {
        if c.level() != packed[0].level() {
            return Err(HekitError::LevelMismatch);
        }
        if c.scale() != packed[0].scale() {
            return Err(HekitError::ScaleMismatch);
        }
    }
    check_keys(ev, &compact_rotation_steps(plan))?;
    let mask = mask0(ev, plan, &packed[0])?;
    let steps = plan.steps();
    let mut entries = Vec::with_capacity(plan.diagonals());
    for (i, ct) in packed.iter().enumerate() {
        let n = plan.diagonals_in(i);
        // leaves[t] = rot(ct, offset(t)); split from the top bit down.
        let mut leaves = vec![ct.clone()];
        for k in (0..CompactPlan::tree_bits(n)).rev() {
            let width = 1usize << (k + 1);
            let mut next = Vec::with_capacity(2 * leaves.len());
            for (g, leaf) in leaves.into_iter().enumerate() {
                // Group g covers positions g * width .. (g + 1) * width; its right half starts
                // at g * width + width / 2.
                let right = (g * width + width / 2 < n)
                    .then(|| ev.rotate(&leaf, steps[k] as i32))
                    .transpose()?;
                next.push(leaf);
                next.extend(right);
            }
            leaves = next;
        }
        debug_assert_eq!(leaves.len(), n);
        for leaf in &leaves {
            let mut y = apply_mask(ev, leaf, &mask)?;
            for &step in &steps {
                let r = ev.rotate(&y, -(step as i32))?;
                y = ev.add(&y, &r)?;
            }
            entries.push(y);
        }
    }
    // The smallest layout holding the layer (`from_diagonals` only checks the fit and the
    // lanes); the diagonals are the same in every layout with these slots and lanes.
    let layout = Layout::new(plan.slots(), plan.diag_len(), plan.batch)?;
    EncWeights::from_diagonals(
        Diagonals::from_prerotated(plan.diagonals(), plan.batch, entries)?,
        plan.d_out,
        plan.d_in,
        &layout,
    )
}
