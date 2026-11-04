//! Bootstrap parameters: level budget of CoeffToSlot / EvalMod / SlotToCoeff and the EvalMod
//! approximation constants. The design and the calculations behind the constants are in
//! `docs/bootstrap.md`.

use hekit_core::error::{HekitError, Result};

use crate::params::{
    CkksParams, N16_BOOT_CTOS_LEVELS, N16_BOOT_EVALMOD_LEVELS, N16_BOOT_STOC_LEVELS,
};

/// EvalMod interval `K` of [`BootParams::default_n16`]: the polynomial is accurate on
/// `x = t / q_0` in `[-K, K]` (covers `|I| <= K - 1`). 13 gives a failure probability of
/// 2^-39.7 per bootstrap **only if the mod-raise runs under a sparse h' = 32 key**
/// (sparse-secret encapsulation, docs/bootstrap.md §3, §7); a dense key needs K ~ 420.
pub const N16_K_INTERVALS: usize = 13;
/// Chebyshev degree of [`BootParams::default_n16`]: 127, i.e. ceil(log2(128)) = 7 levels;
/// sine error 2^-43.8 at K = 13 after one double angle (docs/bootstrap.md §4).
pub const N16_CHEB_DEGREE: usize = 127;
/// Double-angle steps of [`BootParams::default_n16`]: 1 level, 7 + 1 = 8 = EvalMod band.
pub const N16_DOUBLE_ANGLE: usize = 1;

/// Bootstrapping parameters. The bands are consumed from the top of the modulus chain:
/// CoeffToSlot uses the top `ctos_levels` primes, EvalMod the next `evalmod_levels`, SlotToCoeff
/// the next `stoc_levels`; the ciphertext ends at [`levels_after_boot`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootParams {
    /// log2 of the number of slots bootstrapped (`log_n - 1` for full slots).
    pub log_slots: u32,
    /// CoeffToSlot levels (top of the chain).
    pub ctos_levels: usize,
    /// EvalMod levels (below CoeffToSlot).
    pub evalmod_levels: usize,
    /// SlotToCoeff levels (below EvalMod).
    pub stoc_levels: usize,
    /// EvalMod interval half-width `K` in units of `q_0`.
    pub k_intervals: usize,
    /// Degree of the Chebyshev interpolant of `cos(2 pi (x - 1/4) / 2^r)` on `[-K, K]`.
    pub cheb_degree: usize,
    /// Number `r` of double-angle steps `c <- 2c^2 - 1` after the Chebyshev evaluation.
    pub double_angle: usize,
    /// Hamming weight of the secret the mod-raise sees; `None` = dense ternary (hekit default).
    pub sparse_hw: Option<usize>,
}

impl BootParams {
    /// Full-slot parameters for [`CkksParams::n16_boot`]: CoeffToSlot 4, EvalMod 8, SlotToCoeff 3,
    /// K = 13, degree 127, r = 1, dense secret (`sparse_hw = None`).
    ///
    /// The EvalMod constants assume sparse-secret encapsulation to h' = 32 before the mod-raise,
    /// which the plan does not have yet: with the dense key itself no 8-level EvalMod exists,
    /// and at q_0/Delta = 2^16 the predicted precision is 14.4 bits (< 16). See
    /// docs/bootstrap.md §5-§7 before using these values.
    pub fn default_n16() -> BootParams {
        BootParams {
            log_slots: 15,
            ctos_levels: N16_BOOT_CTOS_LEVELS,
            evalmod_levels: N16_BOOT_EVALMOD_LEVELS,
            stoc_levels: N16_BOOT_STOC_LEVELS,
            k_intervals: N16_K_INTERVALS,
            cheb_degree: N16_CHEB_DEGREE,
            double_angle: N16_DOUBLE_ANGLE,
            sparse_hw: None,
        }
    }

    /// Levels EvalMod consumes: `ceil(log2(cheb_degree + 1)) + double_angle` (baby-step
    /// giant-step Chebyshev evaluation, then one level per double angle).
    pub fn evalmod_depth(&self) -> usize {
        let cheb = (self.cheb_degree + 1).next_power_of_two().trailing_zeros() as usize;
        cheb + self.double_angle
    }

    /// Checks the parameters against `params`. Errors (`InvalidParams`): `log_slots` outside
    /// `1..=log_n - 1`; a band of zero levels; the three bands not leaving at least one compute
    /// level above the base; `k_intervals` or `cheb_degree` zero; [`Self::evalmod_depth`] above
    /// `evalmod_levels`; `sparse_hw > N`.
    pub fn validate(&self, params: &CkksParams) -> Result<()> {
        let bad = |m: String| Err(HekitError::InvalidParams(m));
        if !(1..params.log_n()).contains(&self.log_slots) {
            return bad(format!(
                "log_slots {} outside 1..={}",
                self.log_slots,
                params.log_n() - 1
            ));
        }
        if self.ctos_levels == 0 || self.evalmod_levels == 0 || self.stoc_levels == 0 {
            return bad("every bootstrap band needs at least one level".into());
        }
        if levels_after_boot(params, self) == 0 {
            return bad(format!(
                "bands {} + {} + {} leave no compute level below max_level {}",
                self.ctos_levels,
                self.evalmod_levels,
                self.stoc_levels,
                params.max_level()
            ));
        }
        if self.k_intervals == 0 || self.cheb_degree == 0 {
            return bad("k_intervals and cheb_degree must be positive".into());
        }
        if self.evalmod_depth() > self.evalmod_levels {
            return bad(format!(
                "EvalMod depth {} (degree {}, {} double angles) exceeds {} levels",
                self.evalmod_depth(),
                self.cheb_degree,
                self.double_angle,
                self.evalmod_levels
            ));
        }
        if let Some(h) = self.sparse_hw {
            if h > params.n() {
                return bad(format!("sparse_hw {h} exceeds N = {}", params.n()));
            }
        }
        Ok(())
    }
}

/// Level of a ciphertext right after bootstrapping:
/// `max_level - (ctos_levels + evalmod_levels + stoc_levels)`, saturating at 0 (a chain too short
/// for the bands gives 0; [`BootParams::validate`] rejects it). 16 for `n16_boot` with
/// [`BootParams::default_n16`].
pub fn levels_after_boot(params: &CkksParams, bp: &BootParams) -> usize {
    params
        .max_level()
        .saturating_sub(bp.ctos_levels + bp.evalmod_levels + bp.stoc_levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_n16_values() {
        let bp = BootParams::default_n16();
        assert_eq!(bp.log_slots, 15);
        assert_eq!(
            (bp.ctos_levels, bp.evalmod_levels, bp.stoc_levels),
            (4, 8, 3)
        );
        assert_eq!(bp.k_intervals, N16_K_INTERVALS);
        assert_eq!(bp.cheb_degree, N16_CHEB_DEGREE);
        assert_eq!(bp.double_angle, N16_DOUBLE_ANGLE);
        assert_eq!(bp.sparse_hw, None);
    }

    #[test]
    fn n16_boot_leaves_16_levels() {
        assert_eq!(
            levels_after_boot(&CkksParams::n16_boot(), &BootParams::default_n16()),
            16
        );
    }

    /// `levels_after_boot` follows the parameters, it is not a constant.
    #[test]
    fn levels_after_boot_is_derived() {
        let p = CkksParams::n16_boot();
        let bp = BootParams {
            ctos_levels: 3,
            ..BootParams::default_n16()
        };
        assert_eq!(levels_after_boot(&p, &bp), 17);
        assert_eq!(
            levels_after_boot(&CkksParams::n16_train(), &BootParams::default_n16()),
            16
        );
        let tiny = CkksParams::new(5, &[30, 20, 20, 20], &[30], 20, 1).unwrap();
        let bp = BootParams {
            log_slots: 4,
            ctos_levels: 2,
            evalmod_levels: 2,
            stoc_levels: 2,
            ..BootParams::default_n16()
        };
        assert_eq!(levels_after_boot(&tiny, &bp), 0);
    }

    /// The EvalMod band of `n16_boot` is exactly its eight 58-bit primes, CoeffToSlot the top
    /// four and SlotToCoeff the three just below EvalMod.
    #[test]
    fn default_n16_matches_n16_boot_bands() {
        let p = CkksParams::n16_boot();
        let bp = BootParams::default_n16();
        let after = levels_after_boot(&p, &bp);
        let q = p.q_bits();
        let stoc = after + 1..=after + bp.stoc_levels;
        let evalmod = *stoc.end() + 1..=*stoc.end() + bp.evalmod_levels;
        let ctos = *evalmod.end() + 1..=*evalmod.end() + bp.ctos_levels;
        assert_eq!(*ctos.end(), p.max_level());
        assert!(stoc.clone().all(|l| q[l] == 42), "{stoc:?}");
        assert!(evalmod.clone().all(|l| q[l] == 58), "{evalmod:?}");
        assert!(ctos.clone().all(|l| q[l] == 42), "{ctos:?}");
        assert_eq!(p.scale_at(*evalmod.start()), 2f64.powi(58));
        assert_eq!(p.scale_at(after), p.scale());
    }

    /// docs/bootstrap.md §4: ceil(log2(degree + 1)) Chebyshev levels plus one per double angle.
    #[test]
    fn default_n16_evalmod_depth_fills_band() {
        let bp = BootParams::default_n16();
        assert_eq!(bp.evalmod_depth(), 8);
        assert_eq!(bp.evalmod_depth(), bp.evalmod_levels);
        let deg = |d| BootParams {
            cheb_degree: d,
            double_angle: 0,
            ..BootParams::default_n16()
        };
        assert_eq!(deg(1).evalmod_depth(), 1);
        assert_eq!(deg(2).evalmod_depth(), 2);
        assert_eq!(deg(3).evalmod_depth(), 2);
        assert_eq!(deg(255).evalmod_depth(), 8);
        assert_eq!(deg(256).evalmod_depth(), 9);
    }

    #[test]
    fn validate_accepts_default() {
        BootParams::default_n16()
            .validate(&CkksParams::n16_boot())
            .unwrap();
    }

    #[test]
    fn validate_rejects_bad_params() {
        let p = CkksParams::n16_boot();
        let d = BootParams::default_n16();
        let bad = [
            BootParams {
                log_slots: 0,
                ..d.clone()
            },
            BootParams {
                log_slots: 16,
                ..d.clone()
            },
            BootParams {
                ctos_levels: 0,
                ..d.clone()
            },
            BootParams {
                evalmod_levels: 0,
                ..d.clone()
            },
            BootParams {
                stoc_levels: 0,
                ..d.clone()
            },
            BootParams {
                ctos_levels: 20,
                ..d.clone()
            },
            BootParams {
                k_intervals: 0,
                ..d.clone()
            },
            BootParams {
                cheb_degree: 0,
                ..d.clone()
            },
            BootParams {
                cheb_degree: 128,
                ..d.clone()
            },
            BootParams {
                double_angle: 2,
                ..d.clone()
            },
            BootParams {
                sparse_hw: Some((1 << 16) + 1),
                ..d.clone()
            },
        ];
        for bp in bad {
            assert!(
                matches!(bp.validate(&p), Err(HekitError::InvalidParams(_))),
                "{bp:?}"
            );
        }
        let ok = BootParams {
            ctos_levels: 19,
            sparse_hw: Some(192),
            ..d
        };
        ok.validate(&p).unwrap();
        assert_eq!(levels_after_boot(&p, &ok), 1);
    }
}
