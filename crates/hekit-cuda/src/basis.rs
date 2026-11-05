//! Host-side preparation of the base-conversion kernels (`kernels/basis.cu`) and the launch
//! sequences of rescaling, `mod_down` and `mod_up`. Host-only, no CUDA needed.
//!
//! * [`PrimeTable`]: the device's prime indices (Q primes `0..=L`, then P primes
//!   `L+1..`). A device polynomial carries one prime index per limb (any list of distinct
//!   indices), so a `Q_l * P` polynomial is a single polynomial with limbs
//!   `[0, ..., l, L+1, ..., L+k]`.
//! * [`ConvTable`]: the constants of one conversion (source primes to target primes), built
//!   from [`hekit_core::BaseConverter`]'s precomputed values.
//! * [`ModUpPlan`]: which limbs a digit lift reads, converts and copies.
//! * The algorithms themselves (`divide_round_tail`, `mod_up_digit`) are written once
//!   against a small kernel-level interface that the CUDA device and the CPU replay of the
//!   kernels (tests) both implement, so the tests replay the exact launch sequence the device
//!   runs.

use std::ops::Range;
use std::sync::Arc;

use hekit_ckks::keyswitch::digit_ranges;
use hekit_ckks::{CkksParams, QpPoly};
use hekit_core::{BaseConverter, HekitError, Modulus, Result, RnsBasis, RnsPoly};

/// Largest number of source primes of one conversion (`MAX_SRC` in `kernels/basis.cu`):
/// `alpha` for `mod_up`, `k` for `mod_down`, 1 for rescaling.
pub const MAX_SRC: usize = 16;

/// Words of the per-target block of a [`ConvTable`] with `k` sources.
pub fn target_stride(k: usize) -> usize {
    6 + 2 * k
}

fn invalid(msg: impl Into<String>) -> HekitError {
    HekitError::InvalidParams(msg.into())
}

/// The primes of a CKKS parameter set in device-table order: `Q` primes (indices `0..=L`)
/// then `P` primes (indices `L+1..L+1+k`), as in [`crate::tables::HostNttTables::from_params`].
#[derive(Clone, Debug)]
pub struct PrimeTable {
    n: usize,
    nq: usize,
    moduli: Vec<Modulus>,
    basis_q: Arc<RnsBasis>,
    basis_p: Arc<RnsBasis>,
}

impl PrimeTable {
    /// The table of `params`.
    pub fn from_params(params: &CkksParams) -> PrimeTable {
        let moduli = params
            .basis_q()
            .moduli()
            .iter()
            .chain(params.basis_p().moduli())
            .copied()
            .collect();
        PrimeTable {
            n: params.n(),
            nq: params.q().len(),
            moduli,
            basis_q: Arc::clone(params.basis_q()),
            basis_p: Arc::clone(params.basis_p()),
        }
    }

    /// Ring degree `N`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of `Q` primes (`L + 1`); the `P` primes start at this index.
    pub fn nq(&self) -> usize {
        self.nq
    }

    /// Number of `P` primes.
    pub fn np(&self) -> usize {
        self.moduli.len() - self.nq
    }

    /// The moduli in table order.
    pub fn moduli(&self) -> &[Modulus] {
        &self.moduli
    }

    /// Prime indices of a host polynomial's limbs: `0..=l` if its primes are the first `l + 1`
    /// `Q` primes, `L+1..L+1+j` if they are the first `j` `P` primes. Compared by value (and
    /// degree), so any basis with these primes is accepted. `InvalidParams` otherwise.
    pub fn primes_of(&self, poly: &RnsPoly) -> Result<Vec<u32>> {
        let limbs = poly.level() + 1;
        let basis = poly.basis();
        if basis.n() == self.n {
            let primes = &basis.moduli()[..limbs];
            if self.moduli[..self.nq].get(..limbs) == Some(primes) {
                return Ok((0..limbs as u32).collect());
            }
            if self.moduli[self.nq..].get(..limbs) == Some(primes) {
                let nq = self.nq as u32;
                return Ok((nq..nq + limbs as u32).collect());
            }
        }
        Err(invalid(
            "polynomial's primes are not a prefix of this device's Q or P basis",
        ))
    }

    /// `[0, ..., level, L+1, ..., L+k]`: the limbs of an element of `R_{Q_l P}` (all `P`
    /// primes).
    pub fn qp_primes(&self, level: usize) -> Vec<u32> {
        (0..=level as u32)
            .chain(self.nq as u32..self.moduli.len() as u32)
            .collect()
    }

    /// `Some(l)` when `primes` is `[0, ..., l]` (a `Q` prefix).
    pub fn q_prefix_level(&self, primes: &[u32]) -> Option<usize> {
        (!primes.is_empty()
            && primes.len() <= self.nq
            && primes.iter().enumerate().all(|(i, &p)| p as usize == i))
        .then(|| primes.len() - 1)
    }

    /// `Some(j)` when `primes` is the first `j >= 1` `P` primes.
    pub fn p_prefix_len(&self, primes: &[u32]) -> Option<usize> {
        (!primes.is_empty()
            && primes.len() <= self.np()
            && primes
                .iter()
                .enumerate()
                .all(|(i, &p)| p as usize == self.nq + i))
        .then_some(primes.len())
    }

    /// `Some((a, b))` when `primes` is a `Q` prefix of `a >= 1` primes followed by a `P`
    /// prefix of `b >= 1` primes.
    pub fn qp_split(&self, primes: &[u32]) -> Option<(usize, usize)> {
        let a = primes
            .iter()
            .take_while(|&&p| (p as usize) < self.nq)
            .count();
        let q = self.q_prefix_level(&primes[..a])?;
        let b = self.p_prefix_len(&primes[a..])?;
        Some((q + 1, b))
    }

    /// The host basis of a polynomial with limbs `primes`: the parameter set's `Q` basis for
    /// a `Q` prefix, its `P` basis for a `P` prefix; `InvalidParams` for any other prime list
    /// (e.g. `Q_l * P`: see [`PrimeTable::qp_q_limbs`]).
    pub fn host_basis(&self, primes: &[u32]) -> Result<&Arc<RnsBasis>> {
        if self.q_prefix_level(primes).is_some() {
            Ok(&self.basis_q)
        } else if self.p_prefix_len(primes).is_some() {
            Ok(&self.basis_p)
        } else {
            Err(invalid(
                "limbs are not a prefix of the Q or P primes (a Q_l * P polynomial is \
                 downloaded with download_qp and divided with mod_down, not rescaled)",
            ))
        }
    }

    /// Number of `Q` limbs of a `Q_l * P` polynomial with limbs `primes` = `[0..=l]` followed
    /// by **every** `P` prime ([`PrimeTable::qp_primes`]); `InvalidParams` otherwise.
    pub fn qp_q_limbs(&self, primes: &[u32]) -> Result<usize> {
        match self.qp_split(primes) {
            Some((a, b)) if b == self.np() => Ok(a),
            _ => Err(invalid(
                "limbs are not 0..=l followed by every P prime (a Q_l * P polynomial)",
            )),
        }
    }

    /// Rebuilds a host polynomial from flat limbs (`primes.len() * N` residues) with limbs
    /// `primes`, over [`PrimeTable::host_basis`]. `InvalidParams` for another prime list,
    /// a length mismatch or unreduced residues.
    pub fn host_poly(&self, primes: &[u32], flat: &[u64], ntt_form: bool) -> Result<RnsPoly> {
        let basis = self.host_basis(primes)?;
        if flat.len() != primes.len() * self.n {
            return Err(invalid("flat limbs do not match the prime list"));
        }
        let limbs = flat.chunks_exact(self.n).map(<[u64]>::to_vec).collect();
        RnsPoly::from_limbs(basis, limbs, ntt_form)
    }

    /// Rebuilds a [`QpPoly`] (the `Q_l` part and the whole `P` part) from flat limbs with
    /// limbs `primes` ([`PrimeTable::qp_q_limbs`]).
    pub fn host_qp(&self, primes: &[u32], flat: &[u64], ntt_form: bool) -> Result<QpPoly> {
        let a = self.qp_q_limbs(primes)?;
        if flat.len() != primes.len() * self.n {
            return Err(invalid("flat limbs do not match the prime list"));
        }
        let (q, p) = flat.split_at(a * self.n);
        Ok(QpPoly::new(
            self.host_poly(&primes[..a], q, ntt_form)?,
            self.host_poly(&primes[a..], p, ntt_form)?,
        ))
    }

    /// Prime indices and flat limbs of a [`QpPoly`] (`Q` part over a `Q` prefix, `P` part over
    /// all `P` primes, same form).
    pub fn qp_of(&self, x: &QpPoly) -> Result<(Vec<u32>, Vec<u64>)> {
        let (q, p) = (self.primes_of(x.q())?, self.primes_of(x.p())?);
        if self.q_prefix_level(&q).is_none()
            || p.len() != self.np()
            || self.p_prefix_len(&p).is_none()
            || x.q().is_ntt_form() != x.p().is_ntt_form()
        {
            return Err(invalid(
                "a Q_l * P pair needs a Q part over Q primes, the whole P part, one form",
            ));
        }
        let flat = x.q().limbs().iter().chain(x.p().limbs()).flatten().copied();
        Ok((q.into_iter().chain(p).collect(), flat.collect()))
    }

    /// Per-limb Shoup constants of `scalar` for limbs `primes`: `[w_0, w_0', w_1, w_1', ...]`
    /// with `w_l = scalar mod q_l` (`pw_mul_shoup`, as `RnsPoly::mul_scalar_assign`).
    pub fn scalar_shoup(&self, primes: &[u32], scalar: u64) -> Vec<u64> {
        primes
            .iter()
            .flat_map(|&p| {
                let s = self.moduli[p as usize].shoup(scalar);
                [s.value(), s.companion()]
            })
            .collect()
    }

    /// The conversion table from the primes `src` to the primes `targets` (indices of this
    /// table). `InvalidParams` when either list is empty or out of range, an index repeats
    /// (within or across the lists), or there are more than [`MAX_SRC`] sources.
    pub fn conv_table(&self, src: &[u32], targets: &[u32]) -> Result<ConvTable> {
        ConvTable::new(&self.moduli, src, targets)
    }
}

/// Constants of one centered exact conversion (`basis_convert`, `basis_div_finish` in
/// `kernels/basis.cu`; layout documented there). Prime indices refer to a [`PrimeTable`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvTable {
    /// Source prime indices (the order of the source limbs).
    pub src: Vec<u32>,
    /// Target prime indices (the order of the output limbs).
    pub targets: Vec<u32>,
    /// Whether the f64 overflow estimate is applied (`src.len() > 1`, as in
    /// [`BaseConverter::convert_exact`]).
    pub use_f64: bool,
    /// The flat table.
    pub data: Vec<u64>,
}

impl ConvTable {
    /// Builds the table for `moduli[src] -> moduli[targets]` (see [`PrimeTable::conv_table`]).
    pub fn new(moduli: &[Modulus], src: &[u32], targets: &[u32]) -> Result<ConvTable> {
        if src.is_empty() || targets.is_empty() {
            return Err(invalid("a conversion needs source and target primes"));
        }
        if src.len() > MAX_SRC {
            return Err(invalid(format!(
                "{} source primes: the kernels handle at most {MAX_SRC}",
                src.len()
            )));
        }
        let mut seen = vec![false; moduli.len()];
        for &i in src.iter().chain(targets) {
            let slot = seen
                .get_mut(i as usize)
                .ok_or_else(|| invalid(format!("prime index {i} out of range")))?;
            if *slot {
                return Err(invalid(format!("prime index {i} repeats")));
            }
            *slot = true;
        }
        let from: Vec<&Modulus> = src.iter().map(|&i| &moduli[i as usize]).collect();
        let to: Vec<&Modulus> = targets.iter().map(|&i| &moduli[i as usize]).collect();
        let from_v: Vec<u64> = from.iter().map(|m| m.value()).collect();
        let to_v: Vec<u64> = to.iter().map(|m| m.value()).collect();
        let bc = BaseConverter::new(&from_v, &to_v);
        let k = src.len();
        let mut data = Vec::with_capacity(4 * k + targets.len() * target_stride(k));
        data.extend(&from_v);
        let qhat_inv: Vec<_> = from
            .iter()
            .zip(bc.q_hat_inv())
            .map(|(m, &w)| m.shoup(w))
            .collect();
        data.extend(qhat_inv.iter().map(|s| s.value()));
        data.extend(qhat_inv.iter().map(|s| s.companion()));
        // The same f64 as `convert_exact`: 1.0 / q_j.
        data.extend(from_v.iter().map(|&q| (1.0 / q as f64).to_bits()));
        for (r, row) in to.iter().zip(bc.q_hat_mod_to()) {
            // Q mod r as in `convert_exact`; h = (Q - 1) / 2 mod r as `half_mod`.
            let q_mod = from_v
                .iter()
                .fold(1 % r.value(), |acc, &q| r.mul(acc, q % r.value()));
            let h = r.mul(r.sub(q_mod, 1 % r.value()), r.value().div_ceil(2));
            let q_mod_s = r.shoup(q_mod);
            let inv = r.shoup(
                r.inv(q_mod)
                    .ok_or_else(|| invalid("source and target primes are not coprime"))?,
            );
            data.extend([
                r.value(),
                q_mod_s.value(),
                q_mod_s.companion(),
                h,
                inv.value(),
                inv.companion(),
            ]);
            for &w in row {
                let s = r.shoup(w);
                data.extend([s.value(), s.companion()]);
            }
        }
        Ok(ConvTable {
            src: src.to_vec(),
            targets: targets.to_vec(),
            use_f64: k > 1,
            data,
        })
    }

    /// Number of source primes `k`.
    pub fn k(&self) -> usize {
        self.src.len()
    }
}

/// The steps of lifting digit `digit_index` of a level-`l` polynomial (`Q` primes `0..=l`,
/// NTT form) to `Q_l * P` (`mod_up`, the exact centered lift of
/// `hekit_ckks::keyswitch::decompose`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModUpPlan {
    /// The level `l`.
    pub level: usize,
    /// The digit's limbs (prime indices `j*alpha .. min((j+1)*alpha, l+1)`).
    pub digit: Range<usize>,
    /// The conversion targets: the other `Q_l` primes (increasing), then every `P` prime.
    pub targets: Vec<u32>,
    /// The output's primes, [`PrimeTable::qp_primes`]`(l)`.
    pub out_primes: Vec<u32>,
}

impl ModUpPlan {
    /// The plan for an input with limbs `primes`. `InvalidParams` unless `primes` is a `Q`
    /// prefix `[0..=l]` and `digit_index < digit_ranges(params, l).len()`.
    pub fn new(
        params: &CkksParams,
        table: &PrimeTable,
        primes: &[u32],
        digit_index: usize,
    ) -> Result<ModUpPlan> {
        let level = table
            .q_prefix_level(primes)
            .ok_or_else(|| invalid("mod_up needs a polynomial over the Q primes 0..=l"))?;
        let ranges = digit_ranges(params, level);
        let digit = ranges.get(digit_index).cloned().ok_or_else(|| {
            invalid(format!(
                "digit {digit_index} out of range: a level-{level} polynomial has {} digits",
                ranges.len()
            ))
        })?;
        let targets = (0..=level as u32)
            .filter(|&i| !digit.contains(&(i as usize)))
            .chain(table.nq() as u32..table.moduli().len() as u32)
            .collect();
        Ok(ModUpPlan {
            level,
            digit,
            targets,
            out_primes: table.qp_primes(level),
        })
    }

    /// The digit's prime indices (the conversion sources).
    pub fn src(&self) -> Vec<u32> {
        self.digit.clone().map(|i| i as u32).collect()
    }
}

/// `InvalidParams` unless `galois_elt` is odd and `< 2N` (as `RnsPoly::automorphism`
/// requires).
pub fn check_galois_elt(n: usize, galois_elt: u64) -> Result<()> {
    if galois_elt % 2 == 1 && galois_elt < 2 * n as u64 {
        Ok(())
    } else {
        Err(invalid(format!(
            "Galois element {galois_elt} must be odd and < 2N = {}",
            2 * n
        )))
    }
}

/// Number of limbs kept by rescaling a polynomial with limbs `primes` (all but the last).
/// `InvalidParams` with fewer than two limbs, or unless `primes` is a `Q` or `P` prefix (as
/// for an `RnsPoly`, so the result is exactly `RnsPoly::rescale_by_last`; dropping the last
/// `P` prime of a `Q_l * P` polynomial has no CPU counterpart).
pub fn rescale_keep(table: &PrimeTable, primes: &[u32]) -> Result<usize> {
    table.host_basis(primes)?;
    if primes.len() < 2 {
        return Err(invalid(
            "cannot rescale a polynomial with one limb (level 0)",
        ));
    }
    Ok(primes.len() - 1)
}

/// Number of `Q` limbs kept by `mod_down` of a polynomial with limbs `primes`, which must be
/// a `Q` prefix followed by **every** `P` prime ([`PrimeTable::qp_q_limbs`], the layout of
/// `upload_qp` / `download_qp` and of key switching).
pub fn mod_down_keep(table: &PrimeTable, primes: &[u32]) -> Result<usize> {
    table.qp_q_limbs(primes)
}

/// Kernel-level operations the multi-step algorithms below are written against: implemented
/// by the CUDA device (`src/arith.rs`) and by the CPU replay of the kernels (tests).
///
/// Buffers hold whole limbs of `N` residues; limb positions and counts are in limbs.
#[cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
pub(crate) trait Backend {
    /// A buffer of limbs.
    type Buf;
    /// A conversion table ready for [`Backend::convert`].
    type Table;
    /// A buffer of `limbs` limbs, contents unspecified (every algorithm writes before it
    /// reads).
    fn alloc(&self, limbs: usize) -> Result<Self::Buf>;
    /// Copies `count` limbs from `src[src_at..]` to `dst[dst_at..]` (`count >= 1`).
    fn copy_limbs(
        &self,
        src: &Self::Buf,
        src_at: usize,
        dst: &mut Self::Buf,
        dst_at: usize,
        count: usize,
    ) -> Result<()>;
    /// Batched forward (or inverse) NTT of all limbs of `buf`, limb `i` with prime
    /// `primes[i]`.
    fn ntt(&self, buf: &mut Self::Buf, primes: &[u32], inverse: bool) -> Result<()>;
    /// The conversion table for `src -> targets`.
    fn conv_table(&self, src: &[u32], targets: &[u32]) -> Result<Self::Table>;
    /// `basis_convert`: `out` (one limb per target) from `src` (one limb per source,
    /// coefficient form).
    fn convert(&self, out: &mut Self::Buf, src: &Self::Buf, table: &Self::Table) -> Result<()>;
    /// `basis_div_finish`: `t <- (a - t) * (Q mod r)^-1` on the first `targets` limbs.
    fn div_finish(&self, t: &mut Self::Buf, a: &Self::Buf, table: &Self::Table) -> Result<()>;
}

/// Divides the polynomial `x` (limbs `primes`, form `ntt_form`) by the product of its last
/// `primes.len() - keep` primes with rounding and drops them: `RnsPoly::rescale_by_last`
/// (one dropped prime) and `RnsPoly::mod_down` (the `P` primes). Returns the `keep` limbs.
///
/// Launches: copy the tail limbs; inverse NTT (NTT form); `basis_convert`; forward NTT (NTT
/// form); `basis_div_finish`. Every intermediate is fully reduced.
#[cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
pub(crate) fn divide_round_tail<B: Backend>(
    b: &B,
    x: &B::Buf,
    primes: &[u32],
    keep: usize,
    ntt_form: bool,
) -> Result<B::Buf> {
    debug_assert!(keep >= 1 && keep < primes.len());
    let (head, tail) = primes.split_at(keep);
    let table = b.conv_table(tail, head)?;
    let mut src = b.alloc(tail.len())?;
    b.copy_limbs(x, keep, &mut src, 0, tail.len())?;
    if ntt_form {
        b.ntt(&mut src, tail, true)?;
    }
    let mut t = b.alloc(keep)?;
    b.convert(&mut t, &src, &table)?;
    if ntt_form {
        b.ntt(&mut t, head, false)?;
    }
    b.div_finish(&mut t, x, &table)?;
    Ok(t)
}

/// The exact centered lift of one digit (`plan`) of `x` (NTT form) to `Q_l * P`, in NTT form
/// with limbs `plan.out_primes`: the digit limbs are copied, every other limb is converted.
///
/// Launches: copy the digit limbs; inverse NTT; `basis_convert` to the other `Q_l` primes and
/// the `P` primes; forward NTT of the converted limbs (one launch sequence, non-contiguous
/// primes); copy into the output around the digit.
#[cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
pub(crate) fn mod_up_digit<B: Backend>(b: &B, x: &B::Buf, plan: &ModUpPlan) -> Result<B::Buf> {
    let src_primes = plan.src();
    let d = plan.digit.clone();
    let table = b.conv_table(&src_primes, &plan.targets)?;
    let mut src = b.alloc(d.len())?;
    b.copy_limbs(x, d.start, &mut src, 0, d.len())?;
    b.ntt(&mut src, &src_primes, true)?;
    let mut conv = b.alloc(plan.targets.len())?;
    b.convert(&mut conv, &src, &table)?;
    b.ntt(&mut conv, &plan.targets, false)?;
    // Output limbs: conv[..d.start], x[d], conv[d.start..] (the other Q limbs after the digit,
    // then P, are contiguous in both).
    let mut out = b.alloc(plan.out_primes.len())?;
    if d.start > 0 {
        b.copy_limbs(&conv, 0, &mut out, 0, d.start)?;
    }
    b.copy_limbs(x, d.start, &mut out, d.start, d.len())?;
    b.copy_limbs(
        &conv,
        d.start,
        &mut out,
        d.end,
        plan.targets.len() - d.start,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_table_shapes() {
        let params = CkksParams::seal_2020();
        let t = PrimeTable::from_params(&params);
        assert_eq!((t.nq(), t.np(), t.n()), (10, 1, 1 << 15));
        assert_eq!(t.qp_primes(2), vec![0, 1, 2, 10]);
        assert_eq!(t.q_prefix_level(&[0, 1, 2]), Some(2));
        assert_eq!(t.q_prefix_level(&[1, 2]), None);
        assert_eq!(t.q_prefix_level(&[]), None);
        assert_eq!(t.p_prefix_len(&[10]), Some(1));
        assert_eq!(t.p_prefix_len(&[0]), None);
        assert_eq!(t.qp_split(&[0, 1, 2, 10]), Some((3, 1)));
        assert_eq!(t.qp_split(&[0, 2, 10]), None);
        assert_eq!(t.qp_split(&[0, 1]), None);
        assert_eq!(t.qp_split(&[10]), None);
        assert_eq!(mod_down_keep(&t, &[0, 1, 10]).unwrap(), 2);
        assert!(mod_down_keep(&t, &[0, 1]).is_err());
        assert_eq!(rescale_keep(&t, &[0, 1, 2]).unwrap(), 2);
        assert!(rescale_keep(&t, &[0]).is_err());
        assert!(rescale_keep(&t, &[0, 1, 10]).is_err()); // Q_l * P: no CPU counterpart
        assert!(rescale_keep(&t, &[1, 2]).is_err());
        let t16 = PrimeTable::from_params(&CkksParams::n16_train());
        assert_eq!(rescale_keep(&t16, &[32, 33]).unwrap(), 1); // a P prefix
        assert_eq!(mod_down_keep(&t16, &[0, 1, 32, 33, 34, 35]).unwrap(), 2);
        assert!(mod_down_keep(&t16, &[0, 1, 32, 33]).is_err()); // P prefix only

        let q = RnsPoly::zero(params.basis_q(), 3);
        assert_eq!(t.primes_of(&q).unwrap(), vec![0, 1, 2, 3]);
        let p = RnsPoly::zero(params.basis_p(), 0);
        assert_eq!(t.primes_of(&p).unwrap(), vec![10]);
        let other = CkksParams::test_tiny();
        assert!(t.primes_of(&RnsPoly::zero(other.basis_q(), 0)).is_err());

        // Host rebuilds: Q / P prefixes as RnsPoly, Q_l * P as QpPoly, nothing else.
        let n = t.n();
        let flat = vec![1u64; 4 * n];
        let back = t.host_poly(&[0, 1, 2, 3], &flat, true).unwrap();
        assert!(std::sync::Arc::ptr_eq(back.basis(), params.basis_q()));
        assert!(back.is_ntt_form() && back.level() == 3);
        assert!(t.host_poly(&[10], &flat[..n], false).is_ok());
        assert!(t.host_poly(&[0, 1, 2, 10], &flat, false).is_err());
        assert!(t.host_poly(&[1, 2, 3, 4], &flat, false).is_err());
        assert!(t.host_poly(&[0, 1], &flat, false).is_err());
        let qp = t.host_qp(&[0, 1, 2, 10], &flat, false).unwrap();
        assert_eq!((qp.q().level(), qp.p().level()), (2, 0));
        assert_eq!(t.qp_of(&qp).unwrap(), (vec![0, 1, 2, 10], flat.clone()));
        assert!(t.host_qp(&[0, 1, 2, 3], &flat, false).is_err());
        assert_eq!(t.scalar_shoup(&[0], 5)[0], 5);

        assert!(check_galois_elt(8, 15).is_ok());
        assert!(check_galois_elt(8, 16).is_err());
        assert!(check_galois_elt(8, 17).is_err());
    }

    #[test]
    fn conv_table_layout_and_rejections() {
        let params = CkksParams::n16_train();
        let t = PrimeTable::from_params(&params);
        let ms = t.moduli();
        let tab = t.conv_table(&[32, 33], &[0, 5]).unwrap();
        assert!(tab.use_f64);
        assert_eq!(tab.data.len(), 4 * 2 + 2 * target_stride(2));
        let (q0, q1) = (ms[32].value(), ms[33].value());
        assert_eq!(&tab.data[..2], &[q0, q1]);
        assert_eq!(f64::from_bits(tab.data[6]), 1.0 / q0 as f64);
        for (ti, &r) in [0usize, 5].iter().enumerate() {
            let m = &ms[r];
            let blk = &tab.data[8 + ti * target_stride(2)..];
            let qmod = (q0 as u128 * q1 as u128 % m.value() as u128) as u64;
            assert_eq!(blk[0], m.value());
            assert_eq!(blk[1], qmod);
            assert_eq!(m.mul(blk[3], 2), m.sub(qmod, 1)); // h = (Q - 1) / 2
            assert_eq!(m.mul(blk[4], qmod), 1);
            assert_eq!(blk[5], m.shoup(blk[4]).companion());
            assert_eq!(blk[6], q1 % m.value()); // Q / q_0 = q_1
            assert_eq!(blk[8], q0 % m.value());
        }
        assert!(!t.conv_table(&[31], &[0]).unwrap().use_f64);
        for (src, dst) in [
            (&[][..], &[0u32][..]),
            (&[1], &[]),
            (&[1], &[1]),
            (&[1, 1], &[0]),
            (&[99], &[0]),
        ] {
            assert!(matches!(
                t.conv_table(src, dst),
                Err(HekitError::InvalidParams(_))
            ));
        }
        let many: Vec<u32> = (0..17).collect();
        assert!(t.conv_table(&many, &[20]).is_err());
    }

    #[test]
    fn mod_up_plans() {
        let params = CkksParams::n16_train(); // alpha 4, 4 P primes
        let t = PrimeTable::from_params(&params);
        let primes: Vec<u32> = (0..=9).collect();
        let p = ModUpPlan::new(&params, &t, &primes, 2).unwrap();
        assert_eq!(p.digit, 8..10);
        assert_eq!(p.src(), vec![8, 9]);
        let want: Vec<u32> = (0..8).chain(32..36).collect();
        assert_eq!(p.targets, want);
        assert_eq!(p.out_primes, t.qp_primes(9));
        assert!(ModUpPlan::new(&params, &t, &primes, 3).is_err());
        assert!(ModUpPlan::new(&params, &t, &[1, 2], 0).is_err());
    }
}
