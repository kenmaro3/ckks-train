//! CoeffToSlot and SlotToCoeff: the homomorphic special DFT of bootstrapping, with its
//! `log_slots` radix-2 stages collapsed into a few levels (Task 36). The derivation, the scale
//! history and the key count are in `docs/bootstrap.md` §9.
//!
//! # The special DFT in hekit's slot order
//!
//! Let `n = 2^log_slots = N/2` (full packing). A plaintext with real coefficients `m_0..m_{N-1}`
//! decodes to the slots `z_j = m(zeta^(3^j))` ([`crate::encoder`], generator 3). Put
//! `u_k = m_k + i m_{k+n}` (`k < n`). The standard radix-2 special FFT needs a generator `g`
//! with `g = 1 mod 4`, so that `zeta_j^n = i` for every slot; SEAL's generator 3 alternates
//! between `i` and `-i`. Since `3^j = (-3)^j` for even `j` and `-(-3)^j` for odd `j`, and `-3`
//! generates the same subgroup as 5,
//!
//! `z = C_o(U'' u)`, with `U''` the special FFT for the generator `-3` and `C_o` the
//! conjugation of the odd slots.
//!
//! [`dft_stages`] factors `U''` (in the bit-reversed layout: slot `j` holds `u_{bitrev(j)}`)
//! into `log_slots` butterfly stages `S_len`, `len = 2, 4, ..., n` (pairs at distance
//! `len / 2`, twiddles `exp(2 pi i ((-3)^j mod 4 len) / (4 len))`), merged into `levels`
//! groups. `C_o` is not complex-linear, but every stage except the distance-1 ones maps even
//! slots to even slots, and for such a stage `C_o S C_o = rho(S)`, the matrix with its odd rows
//! conjugated. So `C_o` costs nothing except at the one group that holds the distance-1 stage:
//!
//! * **SlotToCoeff** (`z` from the real slot vectors `a = Re u`, `b = Im u`): `z =
//!   rho(H_L) ... rho(H_2) [rho(H_1) a + rho(i H_1) b]`. The first stage takes two inputs, the
//!   `re` and `im` ciphertexts, and needs no conjugation.
//! * **CoeffToSlot**: `u = H_1^-1 C_o rho(R) z` with `R` the product of the other groups'
//!   inverses. The last stage is real-linear, `w = A h + B conj(h)` with
//!   `A = (D_e H D_e + D_o conj(H) D_o) / 2`, `B = (D_e H D_o + D_o conj(H) D_e) / 2`
//!   (`H = H_1^-1`, `D_e` / `D_o` the even / odd slot masks), which gives
//!   `w = C_o(u) / 2`. Then `re = w + conj(w)` and `im = X^(N/2) (conj(w) - w)`: multiplying
//!   by the monomial `X^(N/2)` multiplies slot `j` by `i^(3^j) = (-1)^j i`, an exact operation
//!   that undoes the odd-slot conjugation. CoeffToSlot uses two conjugations in all.
//!
//! The outputs are in the **bit-reversed order** of every bootstrapping DFT: slot `j` of `re`
//! holds `m_{bitrev(j)}` and slot `j` of `im` holds `m_{n + bitrev(j)}`; SlotToCoeff takes
//! that order back. EvalMod is slot-wise, so the order costs nothing.
//!
//! # Level collapsing, factor and scales
//!
//! `levels` groups split the `log_slots` stages evenly; with a remainder, the groups farthest
//! from the distance-1 stage get the extra stage ([`stage_radices`]): the distance-1 group
//! is the two-input or real-linear one, where every diagonal is multiplied twice. A
//! complex `factor` (1/K, 1/(2 pi), the scale step) is folded into the stage matrices:
//! CoeffToSlot then gives `re + i im = factor * u`, SlotToCoeff gives `factor * z`. Its
//! modulus is spread evenly over the stages.
//!
//! Each stage is a BSGS product of plaintext diagonals ([`Evaluator::matvec_plain_sum`]) at
//! one level, followed by one rescale; [`encode_stages`] picks every diagonal scale so that
//! the ciphertext lands **exactly** on a planned scale after each rescale (powers of two in
//! between, `output_scale` after the last stage; the exact-scale path of
//! [`crate::evaluator`]).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ndarray::Array2;
use num_complex::Complex64;

use hekit_core::error::{HekitError, Result};
use hekit_core::poly::RnsPoly;

use crate::ciphertext::Ciphertext;
use crate::encoder::Encoder;
use crate::evaluator::{Evaluator, MIN_PLAIN_SCALE};
use crate::linalg::{Diagonals, bsgs_rotation_steps_with_offset, bsgs_split};
use crate::plaintext::Plaintext;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// An `n x n` complex matrix stored by its nonzero generalized diagonals: diagonal `o`
/// (`0 <= o < n`) holds `M[t][(t + o) mod n]` at position `t`. The DFT stages have at most
/// `2^(r+1) - 1` of them for `r` merged radix-2 stages, so the `n = 2^15` stages of
/// `n16_boot` fit in a few MB.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagMatrix {
    n: usize,
    diags: BTreeMap<usize, Vec<Complex64>>,
}

/// The BSGS layout of one stage: diagonals `offset..offset + d` at slot distance `stride`
/// (see [`crate::linalg`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape {
    d: usize,
    stride: usize,
    offset: i64,
}

impl DiagMatrix {
    /// The `n x n` zero matrix.
    pub fn zero(n: usize) -> DiagMatrix {
        DiagMatrix {
            n,
            diags: BTreeMap::new(),
        }
    }

    /// The `n x n` identity.
    pub fn identity(n: usize) -> DiagMatrix {
        let mut m = DiagMatrix::zero(n);
        m.diags.insert(0, vec![ONE; n]);
        m
    }

    /// The nonzero diagonals of a square dense matrix.
    ///
    /// # Panics
    /// If `m` is not square.
    pub fn from_dense(m: &Array2<Complex64>) -> DiagMatrix {
        let (rows, cols) = m.dim();
        assert_eq!(rows, cols, "matrix must be square (got {rows} x {cols})");
        let mut out = DiagMatrix::zero(rows);
        for ((r, c), &v) in m.indexed_iter() {
            if v != ZERO {
                out.add_entry(r, c, v);
            }
        }
        out
    }

    /// The dense matrix (for tests and small `n`).
    pub fn to_dense(&self) -> Array2<Complex64> {
        let mut m = Array2::from_elem((self.n, self.n), ZERO);
        for (&o, v) in &self.diags {
            for (t, &x) in v.iter().enumerate() {
                m[[t, (t + o) % self.n]] = x;
            }
        }
        m
    }

    /// The dimension `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The offsets of the stored diagonals, ascending.
    pub fn offsets(&self) -> Vec<usize> {
        self.diags.keys().copied().collect()
    }

    /// Diagonal `offset` (`M[t][(t + offset) mod n]` at `t`), if stored.
    pub fn diagonal(&self, offset: usize) -> Option<&[Complex64]> {
        self.diags.get(&offset).map(Vec::as_slice)
    }

    /// `self * rhs` (apply `rhs` first).
    ///
    /// # Panics
    /// If the dimensions differ.
    pub fn mul(&self, rhs: &DiagMatrix) -> DiagMatrix {
        assert_eq!(self.n, rhs.n, "dimension mismatch");
        let n = self.n;
        let mut out = DiagMatrix::zero(n);
        // (A B)[t][t + o1 + o2] += A[t][t + o1] * B[t + o1][t + o1 + o2].
        for (&o1, a) in &self.diags {
            for (&o2, b) in &rhs.diags {
                let c = out
                    .diags
                    .entry((o1 + o2) % n)
                    .or_insert_with(|| vec![ZERO; n]);
                for t in 0..n {
                    c[t] += a[t] * b[(t + o1) % n];
                }
            }
        }
        out
    }

    /// `M x`.
    ///
    /// # Panics
    /// If `x.len() != n`.
    pub fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), self.n, "vector length");
        let n = self.n;
        let mut y = vec![ZERO; n];
        for (&o, v) in &self.diags {
            for t in 0..n {
                y[t] += v[t] * x[(t + o) % n];
            }
        }
        y
    }

    /// `c M`.
    pub fn scaled(&self, c: Complex64) -> DiagMatrix {
        self.map_entries(|_, _, v| v * c)
    }

    /// `M[row][col] += v`.
    fn add_entry(&mut self, row: usize, col: usize, v: Complex64) {
        let n = self.n;
        self.diags
            .entry((col + n - row) % n)
            .or_insert_with(|| vec![ZERO; n])[row] += v;
    }

    /// The matrix with entry `(row, col)` replaced by `f(row, col, value)`.
    fn map_entries(&self, f: impl Fn(usize, usize, Complex64) -> Complex64) -> DiagMatrix {
        let n = self.n;
        let diags = self
            .diags
            .iter()
            .map(|(&o, v)| {
                let w = v
                    .iter()
                    .enumerate()
                    .map(|(t, &x)| f(t, (t + o) % n, x))
                    .collect();
                (o, w)
            })
            .collect();
        DiagMatrix { n, diags }
    }

    /// `rho(M) = D_e M + D_o conj(M)`: the odd rows conjugated (module docs).
    fn rho(&self) -> DiagMatrix {
        self.map_entries(|r, _, v| if r % 2 == 1 { v.conj() } else { v })
    }

    /// `(D_e M D_e + D_o conj(M) D_o, D_e M D_o + D_o conj(M) D_e)`: the parts of
    /// `C_o M C_o` acting on `x` and on `conj(x)` (module docs).
    fn conj_sandwich(&self) -> (DiagMatrix, DiagMatrix) {
        let part = |same: bool| {
            self.map_entries(|r, c, v| {
                if (r % 2 == c % 2) != same {
                    ZERO
                } else if r % 2 == 1 {
                    v.conj()
                } else {
                    v
                }
            })
        };
        (part(true), part(false))
    }

    /// `true` when every stored diagonal has an even offset (even slots map to even slots).
    fn is_parity_preserving(&self) -> bool {
        self.diags.keys().all(|o| o % 2 == 0)
    }

    /// The BSGS layout of the stored diagonals: the stride is the largest power of two
    /// dividing every offset, the offsets `k * stride` are read as `k` in `(-m/2, m/2]`
    /// (`m = n / stride`), and the window `offset..offset + d` is the smallest power of two
    /// `d` that covers them with `offset` a multiple of the baby step (so the giant step
    /// that reaches diagonal 0 costs no rotation). If no window below `m` fits (the offsets
    /// wrap around), `d = m` and `offset = 0`.
    fn shape(&self) -> Shape {
        let nonzero: Vec<usize> = self.diags.keys().copied().filter(|&o| o != 0).collect();
        let Some(tz) = nonzero.iter().map(|o| o.trailing_zeros()).min() else {
            return Shape {
                d: 1,
                stride: 1,
                offset: 0,
            };
        };
        let stride = 1usize << tz;
        let m = (self.n / stride) as i64;
        let ks: Vec<i64> = self
            .diags
            .keys()
            .map(|&o| {
                let k = (o / stride) as i64;
                if k > m / 2 { k - m } else { k }
            })
            .collect();
        let (kmin, kmax) = (*ks.iter().min().unwrap(), *ks.iter().max().unwrap());
        let mut d = ((kmax - kmin + 1) as usize).next_power_of_two();
        while (d as i64) < m {
            let g = bsgs_split(d).0 as i64;
            let offset = (kmin.div_euclid(g) * g).min(0);
            if offset > -(d as i64) && offset + d as i64 > kmax {
                return Shape { d, stride, offset };
            }
            d *= 2;
        }
        Shape {
            d: m as usize,
            stride,
            offset: 0,
        }
    }

    /// The diagonals `offset .. offset + d` of `shape` as slot vectors (zero where not
    /// stored). `InvalidParams` if a stored diagonal is outside the window.
    fn window(&self, shape: Shape) -> Result<Vec<Vec<Complex64>>> {
        let n = self.n as i64;
        let idx: Vec<usize> = (0..shape.d as i64)
            .map(|i| ((shape.offset + i) * shape.stride as i64).rem_euclid(n) as usize)
            .collect();
        if let Some(o) = self.diags.keys().find(|o| !idx.contains(o)) {
            return Err(HekitError::InvalidParams(format!(
                "diagonal {o} outside the stage layout {shape:?}"
            )));
        }
        Ok(idx
            .iter()
            .map(|o| {
                self.diags
                    .get(o)
                    .cloned()
                    .unwrap_or_else(|| vec![ZERO; self.n])
            })
            .collect())
    }
}

/// `e^(2 pi i k / m)`.
fn root(k: usize, m: usize) -> Complex64 {
    let (s, c) = (2.0 * std::f64::consts::PI * k as f64 / m as f64).sin_cos();
    Complex64::new(c, s)
}

/// `b^e mod m`.
fn pow_mod(b: usize, mut e: usize, m: usize) -> usize {
    let (mut base, mut acc) = (b % m, 1 % m);
    while e > 0 {
        if e & 1 == 1 {
            acc = acc * base % m;
        }
        base = base * base % m;
        e >>= 1;
    }
    acc
}

/// The radix-2 butterfly stage `S_len` of the special FFT on `n` points (generator `-3`):
/// for each block of `len` slots and `j < len / 2`, `(x, y) -> (x + w_j y, x - w_j y)` on the
/// pair `(j, j + len/2)` of the block, `w_j = exp(2 pi i ((-3)^j mod 4 len) / (4 len))`; or
/// its inverse `(a, b) -> ((a + b) / 2, (a - b) / (2 w_j))`.
fn butterfly(n: usize, len: usize, inverse: bool) -> DiagMatrix {
    let (h, m4) = (len / 2, 4 * len);
    let mut s = DiagMatrix::zero(n);
    for i in (0..n).step_by(len) {
        for j in 0..h {
            let w = root(pow_mod(m4 - 3, j, m4), m4);
            let (top, bot) = (i + j, i + j + h);
            if inverse {
                let half = Complex64::new(0.5, 0.0);
                let wi = half / w;
                s.add_entry(top, top, half);
                s.add_entry(top, bot, half);
                s.add_entry(bot, top, wi);
                s.add_entry(bot, bot, -wi);
            } else {
                s.add_entry(top, top, ONE);
                s.add_entry(top, bot, w);
                s.add_entry(bot, top, ONE);
                s.add_entry(bot, bot, -w);
            }
        }
    }
    s
}

/// The radix exponents of the `levels` groups, from the group holding the distance-1
/// butterflies (applied first in SlotToCoeff, last in CoeffToSlot) up to the group holding
/// distance `n/2`: `log_slots / levels` each, and the `log_slots mod levels` groups farthest
/// from distance 1 get one more. `n16_boot` (15 over 4): `[3, 4, 4, 4]`.
///
/// # Panics
/// If `levels` is not in `1..=log_slots`.
pub fn stage_radices(log_slots: u32, levels: usize) -> Vec<u32> {
    assert!(
        (1..=log_slots as usize).contains(&levels),
        "levels {levels} outside 1..={log_slots}"
    );
    let (base, extra) = (log_slots / levels as u32, log_slots as usize % levels);
    (0..levels)
        .map(|k| base + u32::from(k >= levels - extra))
        .collect()
}

/// The `2^log_slots`-point special DFT `U''` (generator `-3`, bit-reversed layout; see the
/// [module docs](self)) collapsed into `levels` stages, in the order they are applied.
///
/// `inverse = false` (SlotToCoeff): stage `k` is the product of the butterflies of group
/// `k` of [`stage_radices`] (distance-1 group first), and the stages multiply to
/// `U'' P` with `P` the bit-reversal permutation. `inverse = true` (CoeffToSlot): the inverse
/// groups in the reverse order (distance-`n/2` group first), multiplying to `P U''^-1`.
///
/// Errors (`InvalidParams`): `log_slots` outside `1..=16` or `levels` outside
/// `1..=log_slots`.
pub fn dft_stages(log_slots: u32, levels: usize, inverse: bool) -> Result<Vec<DiagMatrix>> {
    if !(1..=16).contains(&log_slots) || !(1..=log_slots as usize).contains(&levels) {
        return Err(HekitError::InvalidParams(format!(
            "dft_stages: log_slots {log_slots} outside 1..=16 or levels {levels} outside \
             1..=log_slots"
        )));
    }
    let n = 1usize << log_slots;
    let mut groups = Vec::with_capacity(levels);
    let mut k = 1u32;
    for r in stage_radices(log_slots, levels) {
        let mut g = DiagMatrix::identity(n);
        for s in k..k + r {
            let b = butterfly(n, 1 << s, inverse);
            // Forward: S_hi ... S_lo (S_lo applied first); inverse: S_lo^-1 ... S_hi^-1.
            g = if inverse { g.mul(&b) } else { b.mul(&g) };
        }
        groups.push(g);
        k += r;
    }
    if inverse {
        groups.reverse();
    }
    Ok(groups)
}

/// The rotation steps whose Galois keys the encoded `stages` need (sorted, distinct, reduced
/// into `1..n`), computed from the diagonal layout alone, so it works for `n16_boot` without
/// encoding anything. Pass the CoeffToSlot and SlotToCoeff stages together to get the keys of
/// a bootstrap; [`Evaluator::coeff_to_slot`] also needs the conjugation key
/// ([`crate::KeyGenerator::conjugation_key`]), which is not a rotation.
pub fn galois_steps_for(stages: &[DiagMatrix]) -> Vec<i32> {
    let mut steps = BTreeSet::new();
    for s in stages {
        let sh = s.shape();
        let n = s.n() as i64;
        for st in bsgs_rotation_steps_with_offset(sh.d, sh.stride, sh.offset) {
            let r = (st as i64).rem_euclid(n);
            if r != 0 {
                steps.insert(r as i32);
            }
        }
    }
    steps.into_iter().collect()
}

/// The direction of a stage list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Coefficients to slots ([`dft_stages`] with `inverse = true`).
    CoeffToSlot,
    /// Slots to coefficients ([`dft_stages`] with `inverse = false`).
    SlotToCoeff,
}

/// What [`encode_stages`] builds: the direction, the level of the first stage, the scale
/// path and the folded factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageSpec {
    /// Which transform the stages are (must match the `inverse` of [`dft_stages`]).
    pub direction: Direction,
    /// Level of the first stage's input; stage `k` runs at `start_level - k`.
    pub start_level: usize,
    /// Exact scale of the input ciphertext(s).
    pub input_scale: f64,
    /// Exact scale after the last stage's rescale (at `start_level - stages`).
    pub output_scale: f64,
    /// Complex factor folded into the transform (`1` for the plain DFT); see the
    /// [module docs](self).
    pub factor: Complex64,
}

/// One stage of CoeffToSlot or SlotToCoeff: plaintext diagonals at one level and the exact
/// ciphertext scale after its rescale.
#[derive(Debug, Clone)]
pub struct LinearTransform {
    /// The diagonals applied to the stage input.
    pub diags: Diagonals<Plaintext>,
    /// The diagonals of a two-input stage, with the same layout: applied to the conjugate of
    /// the stage input in CoeffToSlot's last stage, to the `im` ciphertext in SlotToCoeff's
    /// first stage (then `diags` takes `re`); `None` for a complex-linear stage.
    pub second: Option<Diagonals<Plaintext>>,
    /// The level of the stage input (and of the diagonals).
    pub level: usize,
    /// The exact ciphertext scale after the stage's rescale (to `level - 1`).
    pub scale: f64,
}

impl LinearTransform {
    /// The rotation steps of `diags` and `second` (as [`Diagonals::rotation_steps`]).
    pub fn rotation_steps(&self) -> Vec<i32> {
        let mut s = self.diags.rotation_steps();
        if let Some(b) = &self.second {
            s.extend(b.rotation_steps());
        }
        s
    }
}

/// The stages adapted to hekit's slot order with `factor` folded in (module docs): for each
/// stage the matrix applied to the input and the matrix of the second input, if any.
fn adapt(
    stages: &[DiagMatrix],
    direction: Direction,
    factor: Complex64,
) -> Result<Vec<(DiagMatrix, Option<DiagMatrix>)>> {
    let l = stages.len();
    if l == 0 {
        return Err(HekitError::InvalidParams("no stages".into()));
    }
    if !(factor.re.is_finite() && factor.im.is_finite() && factor.norm() > 0.0) {
        return Err(HekitError::InvalidParams(format!(
            "factor {factor} must be finite and nonzero"
        )));
    }
    let mag = Complex64::new(factor.norm().powf(1.0 / l as f64), 0.0);
    let phase = factor / factor.norm();
    // The stages that rho() must move C_o through: all but the distance-1 group.
    let others = match direction {
        Direction::CoeffToSlot => &stages[..l - 1],
        Direction::SlotToCoeff => &stages[1..],
    };
    if others.iter().any(|s| !s.is_parity_preserving()) {
        return Err(HekitError::InvalidParams(format!(
            "{direction:?}: only the distance-1 group may mix even and odd slots (stages in \
             the wrong order or of the other direction?)"
        )));
    }
    let plain = |s: &DiagMatrix| (s.rho().scaled(mag), None);
    Ok(match direction {
        Direction::CoeffToSlot => {
            let h = stages[l - 1].scaled(phase * mag * 0.5);
            let (a, b) = h.conj_sandwich();
            others
                .iter()
                .map(plain)
                .chain(std::iter::once((a, Some(b))))
                .collect()
        }
        Direction::SlotToCoeff => {
            let c = phase * mag;
            let h = &stages[0];
            let first = (
                h.rho().scaled(c),
                Some(h.scaled(Complex64::i()).rho().scaled(c)),
            );
            std::iter::once(first)
                .chain(others.iter().map(plain))
                .collect()
        }
    })
}

/// The scale targets after each of `l` stages: powers of two on the geometric path from
/// `input` to `output`, and exactly `output` after the last.
fn scale_path(input: f64, output: f64, l: usize) -> Vec<f64> {
    let (a, b) = (input.log2(), output.log2());
    (1..=l)
        .map(|k| {
            if k == l {
                output
            } else {
                (a + (b - a) * k as f64 / l as f64).round().exp2()
            }
        })
        .collect()
}

/// Adapts `stages` to hekit's slot order, folds `spec.factor` in, and encodes stage `k` at
/// level `spec.start_level - k` (module docs). The diagonal scale of stage `k` is
/// `L_k / s_k`, where `s_k` is the planned input scale and `L_k` the label whose rescale
/// gives exactly the next planned scale ([`Evaluator::exact_product_scale`]); the planned
/// scales are powers of two between `input_scale` and `output_scale`, and `output_scale`
/// after the last stage (`docs/bootstrap.md` §9 has the `n16_boot` history).
///
/// Errors (`InvalidParams`): the stages are not `encoder`'s full slot count in size, or not
/// of `spec.direction`'s shape (only the distance-1 group may mix even and odd slots); `factor`
/// not finite and nonzero; `start_level` above the top level or below the number of stages;
/// a scale not positive and finite; a diagonal scale below `2^20`; and the encoding errors.
/// `ScaleMismatch` if `output_scale` cannot be reached exactly from the last level.
pub fn encode_stages(
    stages: &[DiagMatrix],
    encoder: &Encoder,
    spec: &StageSpec,
) -> Result<Vec<LinearTransform>> {
    let params = encoder.params();
    let bad = |m: String| Err(HekitError::InvalidParams(m));
    if let Some(s) = stages.iter().find(|s| s.n() != params.slots()) {
        return bad(format!(
            "stage of size {} for {} slots (only full packing is supported)",
            s.n(),
            params.slots()
        ));
    }
    let l = stages.len();
    if spec.start_level > params.max_level() || spec.start_level < l {
        return bad(format!(
            "start level {} cannot hold {l} stages (max level {})",
            spec.start_level,
            params.max_level()
        ));
    }
    for s in [spec.input_scale, spec.output_scale] {
        if !(s.is_finite() && s > 0.0) {
            return bad(format!("scale {s} must be positive and finite"));
        }
    }
    let adapted = adapt(stages, spec.direction, spec.factor)?;
    let plan = diagonal_scales(&Evaluator::new(Arc::clone(params), None, None), spec, l)?;
    let mut out = Vec::with_capacity(l);
    for (k, ((main, second), (sp, target))) in adapted.into_iter().zip(plan).enumerate() {
        let level = spec.start_level - k;
        let shape = stages[k].shape();
        let encode = |m: &DiagMatrix| {
            Diagonals::<Plaintext>::encode_complex(
                encoder,
                &m.window(shape)?,
                shape.stride,
                shape.offset,
                sp,
                level,
            )
        };
        out.push(LinearTransform {
            diags: encode(&main)?,
            second: second.as_ref().map(encode).transpose()?,
            level,
            scale: target,
        });
    }
    Ok(out)
}

/// The `(diagonal scale, scale after the rescale)` of each of `l` stages of `spec`
/// ([`encode_stages`]): stage `k` at level `start_level - k` takes the planned scale `s_k`
/// to the target `t_k` of [`scale_path`] exactly, with the diagonal scale `L_k / s_k`
/// (`L_k` from [`Evaluator::exact_product_scale`]). `InvalidParams` for a diagonal scale
/// below `2^20`; `ScaleMismatch` for an unreachable target.
fn diagonal_scales(eval: &Evaluator, spec: &StageSpec, l: usize) -> Result<Vec<(f64, f64)>> {
    let mut s_in = spec.input_scale;
    let mut out = Vec::with_capacity(l);
    for (k, target) in scale_path(spec.input_scale, spec.output_scale, l)
        .into_iter()
        .enumerate()
    {
        let level = spec.start_level - k;
        let sp = eval.exact_product_scale(level, target)? / s_in;
        if sp < MIN_PLAIN_SCALE {
            return Err(HekitError::InvalidParams(format!(
                "stage {k}: diagonal scale {sp:e} below 2^20 (scale {s_in:e} -> {target:e} at \
                 level {level})"
            )));
        }
        out.push((sp, target));
        s_in = target;
    }
    Ok(out)
}

impl Evaluator {
    /// CoeffToSlot: from `ct` (at `ctos[0].level`, with the planned input scale) to the
    /// pair `(re, im)` whose slots hold the real coefficient halves of `ct`'s plaintext in
    /// bit-reversed order, times the folded factor (`re + i im = factor * u`, see the
    /// [module docs](self)): slot `j` of `re` is `m_{bitrev(j)} / scale`, of `im`
    /// `m_{n + bitrev(j)} / scale`. Both are at level `ctos[0].level - ctos.len()` with
    /// scale `ctos.last().scale`.
    ///
    /// Costs one BSGS product and one rescale per stage, the conjugation of the last stage's
    /// input and the conjugation of the result (two conjugation key switches).
    ///
    /// Errors, all before any work: `InvalidParams` if `ctos` is empty, `ct` is not a
    /// 2-component ciphertext over this evaluator's basis, or a rotation key
    /// ([`LinearTransform::rotation_steps`]) or the conjugation key is missing;
    /// `LevelMismatch` unless the stage levels are `ct.level(), ct.level() - 1, ...`;
    /// `ScaleMismatch` unless `ct.scale()` is the scale the stages were encoded for.
    pub fn coeff_to_slot(
        &self,
        ct: &Ciphertext,
        ctos: &[LinearTransform],
    ) -> Result<(Ciphertext, Ciphertext)> {
        self.check_galois_input(ct)?;
        self.check_stages(ctos, ct.level(), ct.scale())?;
        self.check_conjugation_key()?;
        let mut x = ct.clone();
        for lt in ctos {
            let xc = lt.second.as_ref().map(|_| self.conjugate(&x)).transpose()?;
            x = self.apply_stage(lt, &x, xc.as_ref())?;
        }
        let xc = self.conjugate(&x)?;
        let re = self.add(&x, &xc)?;
        let im = self.mul_by_x_half(&self.sub(&xc, &x)?)?;
        Ok((re, im))
    }

    /// SlotToCoeff: from the real slot vectors `re`, `im` (bit-reversed order, as
    /// [`Evaluator::coeff_to_slot`] returns them) to the ciphertext whose plaintext has the
    /// coefficients `re` (first half) and `im` (second half), with its slots times the folded
    /// factor. The result is at level `stoc[0].level - stoc.len()` with scale
    /// `stoc.last().scale`.
    ///
    /// Costs one BSGS product and one rescale per stage; the first stage multiplies the
    /// diagonals of both inputs and shares the giant rotations. No conjugation.
    ///
    /// Errors, all before any work: `InvalidParams` if `stoc` is empty, its first stage has
    /// no `second` diagonals or a later one has, an input is not a 2-component ciphertext over
    /// this evaluator's basis, or a rotation key is missing; `LevelMismatch` unless
    /// `re` and `im` are both at `stoc[0].level` and the stage levels descend by one;
    /// `ScaleMismatch` unless `re` and `im` have exactly the scale the stages were encoded
    /// for.
    pub fn slot_to_coeff(
        &self,
        re: &Ciphertext,
        im: &Ciphertext,
        stoc: &[LinearTransform],
    ) -> Result<Ciphertext> {
        self.check_galois_input(re)?;
        self.check_galois_input(im)?;
        if stoc.first().is_none_or(|lt| lt.second.is_none())
            || stoc[1..].iter().any(|lt| lt.second.is_some())
        {
            return Err(HekitError::InvalidParams(
                "slot_to_coeff: only the first stage (and it must) takes two inputs".into(),
            ));
        }
        if re.level() != im.level() {
            return Err(HekitError::LevelMismatch);
        }
        if re.scale() != im.scale() {
            return Err(HekitError::ScaleMismatch);
        }
        self.check_stages(stoc, re.level(), re.scale())?;
        let mut x = self.apply_stage(&stoc[0], re, Some(im))?;
        for lt in &stoc[1..] {
            x = self.apply_stage(lt, &x, None)?;
        }
        Ok(x)
    }

    /// The checks of a stage list before any work: levels descending from `level`, every
    /// diagonal scale exactly the one that takes the planned input scale (`scale`, then each
    /// stage's `scale`) to the stage's `scale`, matching layouts, and every rotation key.
    fn check_stages(&self, stages: &[LinearTransform], level: usize, scale: f64) -> Result<()> {
        if stages.is_empty() {
            return Err(HekitError::InvalidParams("no stages".into()));
        }
        let mut s_in = scale;
        for (k, lt) in stages.iter().enumerate() {
            if lt.level + k != level {
                return Err(HekitError::LevelMismatch);
            }
            let label = self.exact_product_scale(lt.level, lt.scale)?;
            let sp = lt.diags.entries()[0].scale();
            if sp != label / s_in {
                return Err(HekitError::ScaleMismatch);
            }
            if let Some(b) = &lt.second {
                if (b.d(), b.stride(), b.offset())
                    != (lt.diags.d(), lt.diags.stride(), lt.diags.offset())
                {
                    return Err(HekitError::InvalidParams(
                        "second diagonals of a different layout".into(),
                    ));
                }
                if b.entries()[0].scale() != sp {
                    return Err(HekitError::ScaleMismatch);
                }
            }
            self.check_rotation_keys(&lt.rotation_steps())?;
            s_in = lt.scale;
        }
        Ok(())
    }

    /// One stage: the BSGS sum of `diags * x` (and `second * x2`), labelled with the exact
    /// pre-rescale scale of [`Evaluator::exact_product_scale`] (it differs from the f64
    /// product of the scales by at most one rounding) and rescaled to exactly `lt.scale`.
    fn apply_stage(
        &self,
        lt: &LinearTransform,
        x: &Ciphertext,
        x2: Option<&Ciphertext>,
    ) -> Result<Ciphertext> {
        let mut terms = vec![(&lt.diags, x)];
        if let (Some(b), Some(x2)) = (&lt.second, x2) {
            terms.push((b, x2));
        }
        let y = self.matvec_plain_sum(&terms)?;
        let label = self.exact_product_scale(lt.level, lt.scale)?;
        if ((y.scale() - label) / label).abs() > 1e-15 {
            return Err(HekitError::ScaleMismatch);
        }
        self.rescale(&Ciphertext::new(y.into_polys(), label)?)
    }

    /// `ct * X^(N/2)`: an exact signed permutation of the coefficients that multiplies slot
    /// `j` by `i^(3^j)`, that is `i` for even `j` and `-i` for odd `j`. No key, no level, no
    /// noise (not counted).
    fn mul_by_x_half(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.check_basis(ct.polys())?;
        let n = self.params().n();
        let limbs = (0..=ct.level())
            .map(|_| {
                let mut v = vec![0u64; n];
                v[n / 2] = 1;
                v
            })
            .collect();
        let mut mono = RnsPoly::from_limbs(self.params().basis_q(), limbs, false)?;
        mono.to_ntt();
        let mut c = ct.polys().to_vec();
        c.iter_mut().for_each(|p| p.mul_assign(&mono));
        Ciphertext::new(c, ct.scale())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::BootParams;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::keys::KeyGenerator;
    use crate::params::CkksParams;
    use hekit_core::sampler::Rng;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    /// Uniform values in `[-1, 1)`.
    fn rand_vals(r: &mut Rng, len: usize) -> Vec<f64> {
        (0..len)
            .map(|_| 2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0)
            .collect()
    }

    fn c(re: f64) -> Complex64 {
        Complex64::new(re, 0.0)
    }

    fn bitrev(i: usize, bits: u32) -> usize {
        if bits == 0 {
            0
        } else {
            i.reverse_bits() >> (usize::BITS - bits)
        }
    }

    fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    /// `z_j = sum_k m_k zeta^(3^j k)`, `zeta = exp(2 pi i / 2N)`: hekit's slots of the real
    /// polynomial `m` (see [`crate::encoder`]).
    fn slots_of(m: &[f64]) -> Vec<Complex64> {
        let big_n = m.len();
        let two_n = 2 * big_n;
        (0..big_n / 2)
            .map(|j| {
                let g = pow_mod(3, j, two_n);
                m.iter()
                    .enumerate()
                    .map(|(k, &mk)| root(g * k % two_n, two_n) * mk)
                    .sum()
            })
            .collect()
    }

    /// `u` in the bit-reversed layout: slot `j` is `m_{bitrev(j)} + i m_{n + bitrev(j)}`.
    fn packed_coeffs(m: &[f64]) -> Vec<Complex64> {
        let n = m.len() / 2;
        let bits = n.trailing_zeros();
        (0..n)
            .map(|j| {
                let k = bitrev(j, bits);
                Complex64::new(m[k], m[n + k])
            })
            .collect()
    }

    fn odd_conj(z: &[Complex64]) -> Vec<Complex64> {
        z.iter()
            .enumerate()
            .map(|(j, &v)| if j % 2 == 1 { v.conj() } else { v })
            .collect()
    }

    /// CoeffToSlot in the clear on the adapted stages, with the evaluator's semantics:
    /// two-input stages take the conjugate, then `re = w + conj(w)`,
    /// `im = (-1)^j i (conj(w) - w)`. Returns `re + i im`.
    fn clear_ctos(st: &[(DiagMatrix, Option<DiagMatrix>)], z: &[Complex64]) -> Vec<Complex64> {
        let mut x = z.to_vec();
        for (a, b) in st {
            let mut y = a.apply(&x);
            if let Some(b) = b {
                let xc: Vec<Complex64> = x.iter().map(|v| v.conj()).collect();
                y.iter_mut().zip(b.apply(&xc)).for_each(|(p, q)| *p += q);
            }
            x = y;
        }
        x.iter()
            .enumerate()
            .map(|(j, &w)| {
                let re = w + w.conj();
                let eps = if j % 2 == 0 { 1.0 } else { -1.0 };
                let im = Complex64::new(0.0, eps) * (w.conj() - w);
                re + Complex64::i() * im
            })
            .collect()
    }

    /// SlotToCoeff in the clear: the first stage takes `(a, b)`.
    fn clear_stoc(st: &[(DiagMatrix, Option<DiagMatrix>)], a: &[f64], b: &[f64]) -> Vec<Complex64> {
        let av: Vec<Complex64> = a.iter().map(|&v| c(v)).collect();
        let bv: Vec<Complex64> = b.iter().map(|&v| c(v)).collect();
        let mut x = st[0].0.apply(&av);
        x.iter_mut()
            .zip(st[0].1.as_ref().unwrap().apply(&bv))
            .for_each(|(p, q)| *p += q);
        for (m, _) in &st[1..] {
            x = m.apply(&x);
        }
        x
    }

    #[test]
    fn radices_split_evenly_with_the_extra_stages_away_from_distance_1() {
        assert_eq!(stage_radices(15, 4), vec![3, 4, 4, 4]);
        assert_eq!(stage_radices(4, 4), vec![1, 1, 1, 1]);
        assert_eq!(stage_radices(4, 3), vec![1, 1, 2]);
        assert_eq!(stage_radices(4, 2), vec![2, 2]);
        assert_eq!(stage_radices(4, 1), vec![4]);
        assert_eq!(stage_radices(15, 1), vec![15]);
        assert!(dft_stages(4, 0, false).is_err());
        assert!(dft_stages(4, 5, true).is_err());
        assert!(dft_stages(0, 1, true).is_err());
    }

    /// The stages multiply to `U'' P` (forward) and `P U''^-1` (inverse) for every level
    /// count, and `C_o(U'' u)` is hekit's slot vector: checked against the definition and
    /// against the encoder.
    #[test]
    fn dft_stages_compose_to_the_special_dft() {
        let mut r = rng(1);
        let m = rand_vals(&mut r, 32);
        let (z, u) = (slots_of(&m), packed_coeffs(&m));
        // The convention matches the encoder: decoding the coefficient plaintext gives z.
        let params = Arc::new(CkksParams::test_tiny());
        let enc = Encoder::new(Arc::clone(&params));
        let s = 2f64.powi(40);
        let got = enc.decode_complex(&coeff_plaintext(&params, &m, s, 0));
        assert!(max_err(&got, &z) < 1e-9, "{}", max_err(&got, &z));
        for levels in 1..=4 {
            let fwd = dft_stages(4, levels, false).unwrap();
            let inv = dft_stages(4, levels, true).unwrap();
            assert_eq!((fwd.len(), inv.len()), (levels, levels));
            let mut x = u.clone();
            for st in &fwd {
                x = st.apply(&x);
            }
            assert!(max_err(&odd_conj(&x), &z) < 1e-12, "levels {levels}");
            let mut y = odd_conj(&z);
            for st in &inv {
                y = st.apply(&y);
            }
            assert!(max_err(&y, &u) < 1e-12, "levels {levels}");
            // Dense round trip and the radix structure: stage k of the forward list has
            // offsets that are multiples of 2^(radix sum below k).
            let prod = fwd
                .iter()
                .chain(&inv)
                .fold(DiagMatrix::identity(16), |acc, s| s.mul(&acc));
            let eye = DiagMatrix::identity(16).to_dense();
            let diff = (&prod.to_dense() - &eye)
                .iter()
                .map(|v| v.norm())
                .fold(0.0, f64::max);
            assert!(diff < 1e-12, "levels {levels}: {diff:e}");
            let mut low = 0;
            for (st, rad) in fwd.iter().zip(stage_radices(4, levels)) {
                assert!(st.offsets().iter().all(|o| o % (1 << low) == 0));
                assert_eq!(DiagMatrix::from_dense(&st.to_dense()), *st);
                low += rad;
            }
        }
    }

    /// The BSGS layouts and the key count of n16_boot (log_slots 15, 4 + 4 stages), from the
    /// diagonal layout only: no encoding, no key generation.
    #[test]
    fn n16_boot_rotation_steps() {
        let bp = BootParams::default_n16();
        let cts = dft_stages(bp.log_slots, bp.ctos_levels, true).unwrap();
        let stc = dft_stages(bp.log_slots, bp.stoc_levels, false).unwrap();
        let shapes: Vec<Shape> = stc.iter().map(DiagMatrix::shape).collect();
        let sh = |d, stride, offset| Shape { d, stride, offset };
        assert_eq!(
            shapes,
            vec![
                sh(16, 1, -8),
                sh(32, 8, -16),
                sh(32, 128, -16),
                sh(16, 2048, 0)
            ]
        );
        let cts_shapes: Vec<Shape> = cts.iter().rev().map(DiagMatrix::shape).collect();
        assert_eq!(cts_shapes, shapes);
        let all: Vec<DiagMatrix> = cts.iter().chain(&stc).cloned().collect();
        let steps = galois_steps_for(&all);
        assert_eq!(galois_steps_for(&cts), steps);
        assert_eq!(galois_steps_for(&stc), steps);
        let (cts_ops, stc_ops) = (
            planned_ops(&cts, Direction::CoeffToSlot),
            planned_ops(&stc, Direction::SlotToCoeff),
        );
        eprintln!("n16_boot key switches / plaintext products: CtS {cts_ops:?}, StC {stc_ops:?}");
        assert_eq!((cts_ops, stc_ops), ((34, 93), (35, 108)));
        let bound = 2 * bp.log_slots as usize + bp.ctos_levels + bp.stoc_levels;
        assert!(steps.len() <= bound, "{} > {bound}", steps.len());
        assert_eq!(steps.len(), 32);
        // One key: dnum parts x 2 polynomials x (32 + 4) limbs x N words.
        let p = Arc::new(CkksParams::n16_boot());
        // The scale history of docs/bootstrap.md §9: CtS q_0 -> 2^55 .. 2^58 from level 31,
        // StC 2^58 Delta / q_0 (~2^46) -> 2^45 .. 2^42 = Delta from level 20.
        let eval = Evaluator::new(Arc::clone(&p), None, None);
        let q0 = p.q()[0] as f64;
        let after = crate::bootstrap::levels_after_boot(&p, &bp);
        let specs = [
            (
                p.max_level(),
                q0,
                p.scale_at(p.max_level() - bp.ctos_levels),
                Direction::CoeffToSlot,
            ),
            (
                after + bp.stoc_levels,
                2f64.powi(58) * p.scale() / q0,
                p.scale(),
                Direction::SlotToCoeff,
            ),
        ];
        for (start_level, input_scale, output_scale, direction) in specs {
            let spec = StageSpec {
                direction,
                start_level,
                input_scale,
                output_scale,
                factor: ONE,
            };
            let plan = diagonal_scales(&eval, &spec, 4).unwrap();
            assert_eq!(plan[3].1, output_scale);
            let fmt: Vec<String> = plan
                .iter()
                .map(|(sp, t)| format!("D 2^{:.3} -> 2^{:.3}", sp.log2(), t.log2()))
                .collect();
            eprintln!(
                "n16_boot {direction:?} from level {start_level}, input 2^{:.3}: {}",
                input_scale.log2(),
                fmt.join(", ")
            );
            assert!(plan.iter().all(|&(sp, _)| sp >= 2f64.powi(32)));
        }
        let key_bytes = p.dnum() * 2 * (p.q().len() + p.p().len()) * p.n() * 8;
        let total = (steps.len() + 1) as f64 * key_bytes as f64;
        eprintln!(
            "n16_boot: {} rotation keys + 1 conjugation key, {:.3} GB each, {:.2} GB in all; \
             steps {steps:?}",
            steps.len(),
            key_bytes as f64 / 1e9,
            total / 1e9
        );
    }

    /// The adapted stages give hekit's CoeffToSlot / SlotToCoeff in the clear, for every
    /// level count, and a folded factor multiplies the result.
    #[test]
    fn adapted_stages_in_the_clear() {
        let mut r = rng(2);
        let m = rand_vals(&mut r, 32);
        let (z, u) = (slots_of(&m), packed_coeffs(&m));
        let (a, b): (Vec<f64>, Vec<f64>) = u.iter().map(|v| (v.re, v.im)).unzip();
        let f = Complex64::new(0.37, -1.21);
        for levels in 1..=4 {
            let inv = dft_stages(4, levels, true).unwrap();
            let fwd = dft_stages(4, levels, false).unwrap();
            let cts = adapt(&inv, Direction::CoeffToSlot, ONE).unwrap();
            assert!(
                max_err(&clear_ctos(&cts, &z), &u) < 1e-12,
                "CtS levels {levels}"
            );
            let cts_f = adapt(&inv, Direction::CoeffToSlot, f).unwrap();
            let want: Vec<Complex64> = u.iter().map(|v| f * v).collect();
            assert!(
                max_err(&clear_ctos(&cts_f, &z), &want) < 1e-12,
                "CtS factor"
            );
            let stc = adapt(&fwd, Direction::SlotToCoeff, ONE).unwrap();
            assert!(
                max_err(&clear_stoc(&stc, &a, &b), &z) < 1e-12,
                "StC levels {levels}"
            );
            let stc_f = adapt(&fwd, Direction::SlotToCoeff, f).unwrap();
            let want: Vec<Complex64> = z.iter().map(|v| f * v).collect();
            assert!(
                max_err(&clear_stoc(&stc_f, &a, &b), &want) < 1e-12,
                "StC factor"
            );
            // Every adapted diagonal fits the layout of its stage.
            for (st, (x, y)) in inv.iter().zip(&cts).chain(fwd.iter().zip(&stc)) {
                x.window(st.shape()).unwrap();
                if let Some(y) = y {
                    y.window(st.shape()).unwrap();
                }
            }
            if levels > 1 {
                // The wrong direction is caught by the parity check.
                assert!(adapt(&fwd, Direction::CoeffToSlot, ONE).is_err());
                assert!(adapt(&inv, Direction::SlotToCoeff, ONE).is_err());
            }
        }
        assert!(
            adapt(
                &dft_stages(4, 2, true).unwrap(),
                Direction::CoeffToSlot,
                ZERO
            )
            .is_err()
        );
    }

    /// The key switches and plaintext products the evaluator performs for `stages` (adapted
    /// for `direction`), from the layouts and the zero diagonals alone, as
    /// `Evaluator::matvec_plain_sum` skips them; CoeffToSlot adds its two conjugations.
    fn planned_ops(stages: &[DiagMatrix], direction: Direction) -> (u64, u64) {
        let adapted = adapt(stages, direction, ONE).unwrap();
        let (mut rot, mut pt) = (0u64, 0u64);
        for (st, (a, b)) in stages.iter().zip(&adapted) {
            let sh = st.shape();
            let (baby, giant) = bsgs_split(sh.d);
            let zero: Vec<Vec<bool>> = std::iter::once(a)
                .chain(b.as_ref())
                .map(|m| {
                    m.window(sh)
                        .unwrap()
                        .iter()
                        .map(|v| v.iter().all(|x| *x == ZERO))
                        .collect()
                })
                .collect();
            for z in &zero {
                pt += z.iter().filter(|&&x| !x).count() as u64;
                rot += (1..baby)
                    .filter(|&k2| (0..giant).any(|k1| !z[k1 * baby + k2]))
                    .count() as u64;
            }
            rot += (0..giant)
                .filter(|&k1| {
                    sh.offset + (baby * k1) as i64 != 0
                        && zero.iter().any(|z| (0..baby).any(|k2| !z[k1 * baby + k2]))
                })
                .count() as u64;
            if direction == Direction::CoeffToSlot && b.is_some() {
                rot += 1;
            }
        }
        if direction == Direction::CoeffToSlot {
            rot += 1;
        }
        (rot, pt)
    }

    /// The tiny bootstrap parameters of the tests: test_tiny's bands (CoeffToSlot 4 at
    /// levels 8..=11, EvalMod 2, SlotToCoeff 4 at levels 2..=5, one compute level), an
    /// ephemeral weight 4 (of N = 32) and K = 6 = ceil(7 sigma_I + 1/2), degree 3.
    fn tiny_boot(params: &CkksParams) -> BootParams {
        let bp = BootParams {
            log_slots: 4,
            ctos_levels: 4,
            evalmod_levels: 2,
            stoc_levels: 4,
            k_intervals: 6,
            cheb_degree: 3,
            double_angle: 0,
            sparse_hw: None,
            ephemeral_hw: Some(4),
        };
        bp.validate(params).unwrap();
        bp
    }

    #[test]
    fn tiny_boot_params_validate() {
        let p = CkksParams::test_tiny();
        let bp = tiny_boot(&p);
        assert_eq!(crate::bootstrap::levels_after_boot(&p, &bp), 1);
        assert_eq!(bp.min_k_intervals(&p), 6);
        assert!(BootParams::default_n16().validate(&p).is_err());
    }

    /// A plaintext with the coefficients `round(c_k * scale)` at `level`.
    fn coeff_plaintext(params: &CkksParams, c: &[f64], scale: f64, level: usize) -> Plaintext {
        let basis = params.basis_q();
        let limbs = basis.moduli()[..=level]
            .iter()
            .map(|m| {
                let q = m.value();
                c.iter()
                    .map(|&x| {
                        let v = (x * scale).round();
                        let a = (v.abs() as u64) % q;
                        if v < 0.0 && a != 0 { q - a } else { a }
                    })
                    .collect()
            })
            .collect();
        let mut poly = RnsPoly::from_limbs(basis, limbs, false).unwrap();
        poly.to_ntt();
        Plaintext::new(poly, scale).unwrap()
    }

    /// The coefficients `/ scale` of a plaintext at level <= 1 (centered, exact CRT in i128).
    fn coeffs_of(params: &CkksParams, pt: &Plaintext) -> Vec<f64> {
        let mut p = pt.poly().clone();
        p.to_coeff();
        let q: Vec<i128> = params.q()[..=p.level()]
            .iter()
            .map(|&v| v as i128)
            .collect();
        assert!(q.len() <= 2);
        (0..params.n())
            .map(|i| {
                let a = p.limbs()[0][i] as i128;
                let v = if q.len() == 1 {
                    a
                } else {
                    // x = a + q0 * ((b - a) q0^-1 mod q1).
                    let b = p.limbs()[1][i] as i128;
                    let inv = modinv(q[0] % q[1], q[1]);
                    let t = ((b - a).rem_euclid(q[1]) * inv).rem_euclid(q[1]);
                    a + q[0] * t
                };
                let big: i128 = q.iter().product();
                let v = if v > big / 2 { v - big } else { v };
                v as f64 / pt.scale()
            })
            .collect()
    }

    fn modinv(a: i128, m: i128) -> i128 {
        let (mut t, mut nt, mut r, mut nr) = (0i128, 1i128, m, a.rem_euclid(m));
        while nr != 0 {
            let q = r / nr;
            (t, nt) = (nt, t - q * nt);
            (r, nr) = (nr, r - q * nr);
        }
        t.rem_euclid(m)
    }

    struct Ctx {
        params: Arc<CkksParams>,
        enc: Encoder,
        encryptor: Encryptor,
        dec: Decryptor,
        eval: Evaluator,
    }

    /// test_tiny with Galois keys for `steps` and (if `conj`) the conjugation key.
    fn ctx(steps: &[i32], conj: bool, seed: u8) -> Ctx {
        let params = Arc::new(CkksParams::test_tiny());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let pk = kg.public_key();
        let mut gks = kg.galois_keys(steps);
        if conj {
            gks.insert(kg.conjugation_key());
        }
        Ctx {
            enc: Encoder::new(Arc::clone(&params)),
            encryptor: Encryptor::new(Arc::clone(&params), pk),
            dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
            eval: Evaluator::new(Arc::clone(&params), None, Some(gks)),
            params,
        }
    }

    /// Keys for every level count of log_slots 4.
    fn all_steps() -> Vec<i32> {
        let mut s = BTreeSet::new();
        for levels in 1..=4 {
            for inv in [false, true] {
                s.extend(galois_steps_for(&dft_stages(4, levels, inv).unwrap()));
            }
        }
        s.into_iter().collect()
    }

    /// The CoeffToSlot transforms of the tiny bands: `levels` stages from the top level
    /// (11), input scale q_0 (the mod-raise relabelling of Task 38), output exactly
    /// `scale_at` of the level below the stages (2^58 after 4 stages).
    fn tiny_ctos(c: &Ctx, levels: usize, factor: Complex64) -> Vec<LinearTransform> {
        let top = c.params.max_level();
        let spec = StageSpec {
            direction: Direction::CoeffToSlot,
            start_level: top,
            input_scale: c.params.q()[0] as f64,
            output_scale: 2f64.powi(58),
            factor,
        };
        encode_stages(&dft_stages(4, levels, true).unwrap(), &c.enc, &spec).unwrap()
    }

    /// The SlotToCoeff transforms: `levels` stages from level 5 (the top of the tiny StC
    /// band) down to exactly the preset scale 2^42.
    fn tiny_stoc(
        c: &Ctx,
        levels: usize,
        input_scale: f64,
        factor: Complex64,
    ) -> Vec<LinearTransform> {
        let spec = StageSpec {
            direction: Direction::SlotToCoeff,
            start_level: 5,
            input_scale,
            output_scale: c.params.scale(),
            factor,
        };
        encode_stages(&dft_stages(4, levels, false).unwrap(), &c.enc, &spec).unwrap()
    }

    /// Brief, Step 1: random coefficients -> encrypt -> coeff_to_slot -> decrypt gives the
    /// coefficients (bit-reversed order) within 2^-20, for every level count; `im` and `re`
    /// have no imaginary part. Also the bootstrap case: coefficients `I + delta` with
    /// `|I| <= K - 1` and the factor 1/K folded in.
    #[test]
    fn coeff_to_slot_recovers_the_coefficients() {
        let c = ctx(&all_steps(), true, 3);
        let mut r = rng(4);
        let (n, top) = (c.params.slots(), c.params.max_level());
        let q0 = c.params.q()[0] as f64;
        let k = 6.0;
        for levels in [4, 3, 2, 1] {
            for (case, factor) in [("unit", 1.0), ("1/K", 1.0 / k)] {
                let mut m = rand_vals(&mut r, 2 * n);
                if case == "1/K" {
                    // Integer overflow parts |I| <= K - 1, as after the mod-raise.
                    for (i, v) in m.iter_mut().enumerate() {
                        *v += (i % 11) as f64 - 5.0;
                    }
                }
                let ct = c
                    .encryptor
                    .encrypt(&coeff_plaintext(&c.params, &m, q0, top), &mut r);
                let ctos = tiny_ctos(&c, levels, c64(factor));
                let (re, im) = c.eval.coeff_to_slot(&ct, &ctos).unwrap();
                assert_eq!((re.level(), im.level()), (top - levels, top - levels));
                assert_eq!((re.scale(), im.scale()), (2f64.powi(58), 2f64.powi(58)));
                let bits = n.trailing_zeros();
                let gr = c.enc.decode_complex(&c.dec.decrypt(&re));
                let gi = c.enc.decode_complex(&c.dec.decrypt(&im));
                let want_re: Vec<Complex64> =
                    (0..n).map(|j| c64(factor * m[bitrev(j, bits)])).collect();
                let want_im: Vec<Complex64> = (0..n)
                    .map(|j| c64(factor * m[n + bitrev(j, bits)]))
                    .collect();
                let err = max_err(&gr, &want_re).max(max_err(&gi, &want_im));
                eprintln!(
                    "coeff_to_slot {levels} stages, {case}: max error 2^{:.1}",
                    err.log2()
                );
                assert!(err < 2f64.powi(-20), "{levels} stages {case}: {err:e}");
            }
        }
    }

    fn c64(v: f64) -> Complex64 {
        Complex64::new(v, 0.0)
    }

    /// Brief, Step 1: slot_to_coeff(coeff_to_slot(ct)) is the identity within 2^-18 (on the
    /// coefficients and on the slots), ending exactly on the preset scale; with 1/K folded
    /// into CoeffToSlot and K into SlotToCoeff as well.
    #[test]
    fn slot_to_coeff_inverts_coeff_to_slot() {
        let c = ctx(&all_steps(), true, 5);
        let mut r = rng(6);
        let (n, top) = (c.params.slots(), c.params.max_level());
        let q0 = c.params.q()[0] as f64;
        for (cl, sl, k) in [(4, 4, 1.0), (2, 4, 6.0), (4, 3, 6.0), (1, 2, 1.0)] {
            let m = rand_vals(&mut r, 2 * n);
            let pt = coeff_plaintext(&c.params, &m, q0, top);
            let ct = c.encryptor.encrypt(&pt, &mut r);
            let (re, im) = c
                .eval
                .coeff_to_slot(&ct, &tiny_ctos(&c, cl, c64(1.0 / k)))
                .unwrap();
            // Drop to the StC band (EvalMod would run in between).
            let re = c.eval.mod_switch_to_level(&re, 5).unwrap();
            let im = c.eval.mod_switch_to_level(&im, 5).unwrap();
            let stoc = tiny_stoc(&c, sl, re.scale(), c64(k));
            let out = c.eval.slot_to_coeff(&re, &im, &stoc).unwrap();
            assert_eq!((out.level(), out.scale()), (5 - sl, c.params.scale()));
            let dec = c.dec.decrypt(&out);
            let err_slots = max_err(&c.enc.decode_complex(&dec), &slots_of(&m));
            let mut err = err_slots;
            if out.level() <= 1 {
                let got = coeffs_of(&c.params, &dec);
                let e = got
                    .iter()
                    .zip(&m)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                err = err.max(e);
            }
            eprintln!(
                "CtS {cl} stages -> StC {sl} stages (K = {k}): max error 2^{:.1} (slots 2^{:.1})",
                err.log2(),
                err_slots.log2()
            );
            assert!(err < 2f64.powi(-18), "{cl}/{sl}: {err:e}");
        }
    }

    /// A complex factor folded into the stages multiplies the encrypted result: CoeffToSlot
    /// gives `re + i im = c u`, SlotToCoeff gives `c z`.
    #[test]
    fn folded_factor_multiplies_the_encrypted_result() {
        let c = ctx(&all_steps(), true, 7);
        let mut r = rng(8);
        let (n, top) = (c.params.slots(), c.params.max_level());
        let q0 = c.params.q()[0] as f64;
        let f = Complex64::new(0.6, -0.45);
        let m = rand_vals(&mut r, 2 * n);
        let u = packed_coeffs(&m);
        let ct = c
            .encryptor
            .encrypt(&coeff_plaintext(&c.params, &m, q0, top), &mut r);
        let run = |factor| {
            let (re, im) = c
                .eval
                .coeff_to_slot(&ct, &tiny_ctos(&c, 2, factor))
                .unwrap();
            let gr = c.enc.decode(&c.dec.decrypt(&re));
            let gi = c.enc.decode(&c.dec.decrypt(&im));
            gr.iter()
                .zip(&gi)
                .map(|(&a, &b)| Complex64::new(a, b))
                .collect::<Vec<_>>()
        };
        let plain = run(ONE);
        let folded = run(f);
        let want: Vec<Complex64> = plain.iter().map(|v| f * v).collect();
        assert!(max_err(&plain, &u) < 2f64.powi(-20));
        let err = max_err(&folded, &want);
        eprintln!("folded CtS factor: max error 2^{:.1}", err.log2());
        assert!(err < 2f64.powi(-20), "{err:e}");

        let (a, b): (Vec<f64>, Vec<f64>) = u.iter().map(|v| (v.re, v.im)).unzip();
        let s = 2f64.powi(46);
        let re = c
            .encryptor
            .encrypt(&c.enc.encode(&a, s, 5).unwrap(), &mut r);
        let im = c
            .encryptor
            .encrypt(&c.enc.encode(&b, s, 5).unwrap(), &mut r);
        let z = slots_of(&m);
        for factor in [ONE, f] {
            let out = c
                .eval
                .slot_to_coeff(&re, &im, &tiny_stoc(&c, 4, s, factor))
                .unwrap();
            let got = c.enc.decode_complex(&c.dec.decrypt(&out));
            let want: Vec<Complex64> = z.iter().map(|v| factor * v).collect();
            let err = max_err(&got, &want);
            eprintln!("StC factor {factor}: max error 2^{:.1}", err.log2());
            assert!(err < 2f64.powi(-18), "{err:e}");
        }
    }

    /// The encoded stages need exactly the keys of `galois_steps_for`, and each two-input
    /// stage shares its giant rotations: counted rotations per transform.
    #[test]
    fn encoded_steps_and_op_counts() {
        let c = ctx(&all_steps(), true, 9);
        let mut r = rng(10);
        let top = c.params.max_level();
        let ctos = tiny_ctos(&c, 2, ONE);
        let stoc = tiny_stoc(&c, 2, 2f64.powi(58), ONE);
        let n = c.params.slots() as i32;
        let mut encoded: Vec<i32> = ctos
            .iter()
            .chain(&stoc)
            .flat_map(LinearTransform::rotation_steps)
            .map(|s| s.rem_euclid(n))
            .filter(|&s| s != 0)
            .collect();
        encoded.sort_unstable();
        encoded.dedup();
        let mut stages = dft_stages(4, 2, true).unwrap();
        stages.extend(dft_stages(4, 2, false).unwrap());
        assert_eq!(encoded, galois_steps_for(&stages));
        assert!(ctos[0].second.is_none() && ctos[1].second.is_some());
        assert!(stoc[0].second.is_some() && stoc[1].second.is_none());

        let m = rand_vals(&mut r, 32);
        let ct = c.encryptor.encrypt(
            &coeff_plaintext(&c.params, &m, c.params.q()[0] as f64, top),
            &mut r,
        );
        let before = c.eval.counter();
        let (re, im) = c.eval.coeff_to_slot(&ct, &ctos).unwrap();
        let cts_ops = c.eval.counter();
        let re = c.eval.mod_switch_to_level(&re, 5).unwrap();
        let im = c.eval.mod_switch_to_level(&im, 5).unwrap();
        c.eval.slot_to_coeff(&re, &im, &stoc).unwrap();
        let stc_ops = c.eval.counter();
        eprintln!(
            "2 + 2 stages, log_slots 4: CtS rotations {} (incl. 2 conjugations), ct-pt {}; \
             StC rotations {}, ct-pt {}",
            cts_ops.rotations - before.rotations,
            cts_ops.ct_pt_mults - before.ct_pt_mults,
            stc_ops.rotations - cts_ops.rotations,
            stc_ops.ct_pt_mults - cts_ops.ct_pt_mults
        );
        assert_eq!(cts_ops.rescales - before.rescales, 2);
        assert_eq!(stc_ops.rescales - cts_ops.rescales, 2);
        // The planner of the n16_boot counts agrees with the counter.
        let cts_plan = planned_ops(&dft_stages(4, 2, true).unwrap(), Direction::CoeffToSlot);
        let stc_plan = planned_ops(&dft_stages(4, 2, false).unwrap(), Direction::SlotToCoeff);
        assert_eq!(
            cts_plan,
            (
                cts_ops.rotations - before.rotations,
                cts_ops.ct_pt_mults - before.ct_pt_mults
            )
        );
        assert_eq!(
            stc_plan,
            (
                stc_ops.rotations - cts_ops.rotations,
                stc_ops.ct_pt_mults - cts_ops.ct_pt_mults
            )
        );
    }

    /// Every error is reported before any work (the counter does not move).
    #[test]
    fn errors_before_any_work() {
        let c = ctx(&all_steps(), false, 11);
        let mut r = rng(12);
        let top = c.params.max_level();
        let q0 = c.params.q()[0] as f64;
        let ctos = tiny_ctos(&c, 2, ONE);
        let m = rand_vals(&mut r, 32);
        let ct = c
            .encryptor
            .encrypt(&coeff_plaintext(&c.params, &m, q0, top), &mut r);
        let before = c.eval.counter();
        // No conjugation key.
        assert_eq!(
            c.eval.coeff_to_slot(&ct, &ctos).unwrap_err(),
            HekitError::InvalidParams("missing galois key".into())
        );
        let full = ctx(&all_steps(), true, 11);
        // Wrong input scale, wrong level, no stages.
        let ct2 = c
            .encryptor
            .encrypt(&coeff_plaintext(&c.params, &m, 2f64.powi(42), top), &mut r);
        let ct2 = Ciphertext::new(ct2.into_polys(), 2f64.powi(42)).unwrap();
        let ctos_f = tiny_ctos(&full, 2, ONE);
        assert_eq!(
            full.eval
                .coeff_to_slot(&remap(&full, &ct2), &ctos_f)
                .unwrap_err(),
            HekitError::ScaleMismatch
        );
        let low = full
            .eval
            .mod_switch_to_level(&remap(&full, &ct), top - 1)
            .unwrap();
        assert_eq!(
            full.eval.coeff_to_slot(&low, &ctos_f).unwrap_err(),
            HekitError::LevelMismatch
        );
        assert!(matches!(
            full.eval.coeff_to_slot(&remap(&full, &ct), &[]),
            Err(HekitError::InvalidParams(_))
        ));
        // Missing rotation keys.
        let few = ctx(&[1], true, 13);
        let ctos_few = tiny_ctos(&few, 2, ONE);
        assert_eq!(
            few.eval
                .coeff_to_slot(&remap(&few, &ct), &ctos_few)
                .unwrap_err(),
            HekitError::InvalidParams("missing galois key".into())
        );
        // slot_to_coeff: CtS stages, mismatched inputs.
        let stoc = tiny_stoc(&full, 2, 2f64.powi(46), ONE);
        let s = 2f64.powi(46);
        let a = full
            .encryptor
            .encrypt(&full.enc.encode(&[0.5], s, 5).unwrap(), &mut r);
        let b = full
            .encryptor
            .encrypt(&full.enc.encode(&[0.5], s * 2.0, 5).unwrap(), &mut r);
        assert!(matches!(
            full.eval.slot_to_coeff(&a, &a, &tiny_ctos(&full, 2, ONE)),
            Err(HekitError::InvalidParams(_))
        ));
        assert_eq!(
            full.eval.slot_to_coeff(&a, &b, &stoc).unwrap_err(),
            HekitError::ScaleMismatch
        );
        let a4 = full.eval.mod_switch_to_level(&a, 4).unwrap();
        assert_eq!(
            full.eval.slot_to_coeff(&a, &a4, &stoc).unwrap_err(),
            HekitError::LevelMismatch
        );
        assert_eq!(
            full.eval.slot_to_coeff(&a4, &a4, &stoc).unwrap_err(),
            HekitError::LevelMismatch
        );
        assert_eq!(c.eval.counter(), before);
        assert_eq!(full.eval.counter(), Default::default());
        assert_eq!(few.eval.counter(), Default::default());

        // encode_stages errors.
        let spec = StageSpec {
            direction: Direction::CoeffToSlot,
            start_level: 1,
            input_scale: q0,
            output_scale: 2f64.powi(58),
            factor: ONE,
        };
        let st = dft_stages(4, 2, true).unwrap();
        assert!(matches!(
            encode_stages(&st, &full.enc, &spec),
            Err(HekitError::InvalidParams(_))
        ));
        let spec = StageSpec {
            start_level: top + 1,
            ..spec
        };
        assert!(encode_stages(&st, &full.enc, &spec).is_err());
        let spec = StageSpec {
            start_level: top,
            ..spec
        };
        assert!(matches!(
            encode_stages(&dft_stages(3, 2, true).unwrap(), &full.enc, &spec),
            Err(HekitError::InvalidParams(_))
        ));
        assert!(matches!(
            encode_stages(&dft_stages(4, 2, false).unwrap(), &full.enc, &spec),
            Err(HekitError::InvalidParams(_))
        ));
        // A path whose diagonal scale would drop below 2^20 (2^54 -> 2^7 -> 2^-40).
        let tiny = StageSpec {
            output_scale: 2f64.powi(-40),
            ..spec
        };
        assert!(matches!(
            encode_stages(&st, &full.enc, &tiny),
            Err(HekitError::InvalidParams(_))
        ));
        encode_stages(&st, &full.enc, &spec).unwrap();
    }

    /// `ct` re-encrypted is not possible across contexts; the tests share the parameters
    /// `Arc` only within one context, so rebuild `ct` over `c`'s basis by its limbs.
    fn remap(c: &Ctx, ct: &Ciphertext) -> Ciphertext {
        let polys = ct
            .polys()
            .iter()
            .map(|p| {
                RnsPoly::from_limbs(c.params.basis_q(), p.limbs().to_vec(), p.is_ntt_form())
                    .unwrap()
            })
            .collect();
        Ciphertext::new(polys, ct.scale()).unwrap()
    }

    /// `X^(N/2)` multiplies slot j by i for even j and -i for odd j.
    #[test]
    fn monomial_half_multiplies_by_signed_i() {
        let c = ctx(&[], false, 14);
        let mut r = rng(15);
        let v = rand_vals(&mut r, 16);
        let ct = c
            .encryptor
            .encrypt(&c.enc.encode(&v, 2f64.powi(40), 3).unwrap(), &mut r);
        let got = c
            .enc
            .decode_complex(&c.dec.decrypt(&c.eval.mul_by_x_half(&ct).unwrap()));
        let want: Vec<Complex64> = v
            .iter()
            .enumerate()
            .map(|(j, &x)| Complex64::new(0.0, if j % 2 == 0 { x } else { -x }))
            .collect();
        assert!(max_err(&got, &want) < 2f64.powi(-25));
    }
}
