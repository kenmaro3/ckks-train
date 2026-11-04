//! Bootstrap parameters: level budget of CoeffToSlot / EvalMod / SlotToCoeff and the EvalMod
//! approximation constants. The design and the calculations behind the constants are in
//! `docs/bootstrap.md`.

use hekit_core::error::{HekitError, Result};

use crate::params::{
    CkksParams, N16_BOOT_CTOS_LEVELS, N16_BOOT_EVALMOD_LEVELS, N16_BOOT_STOC_LEVELS,
};

/// Ephemeral Hamming weight h' of [`BootParams::default_n16`]: sparse-secret encapsulation
/// (Bossuat, Troncoso-Pastoriza, Hubaux, ACNS 2022) switches to an h' = 32 key for the mod-raise
/// (controller ruling R-M3-5, docs/bootstrap.md §7).
pub const N16_EPHEMERAL_HW: usize = 32;
/// EvalMod interval `K` of [`BootParams::default_n16`]: the polynomial is accurate on
/// `x = t / q_0` in `[-K, K]` (covers `|I| <= K - 1`). **Assumes SSE with h' = 32**: then 13
/// gives a failure probability of 2^-39.7 per bootstrap (docs/bootstrap.md §3); a mod-raise under
/// the dense key itself would need K ~ 420 and 12 EvalMod levels.
pub const N16_K_INTERVALS: usize = 13;
/// Chebyshev degree of [`BootParams::default_n16`] (assumes SSE with h' = 32, K = 13): 127, i.e.
/// ceil(log2(128)) = 7 levels **with the level-optimal, scale-invariant Chebyshev evaluation of
/// Bossuat et al. (Eurocrypt 2021)**, a requirement on Task 37 (textbook Paterson-Stockmeyer
/// needs 8, which only the fallback chain with 8 EvalMod levels has); sine error 2^-45.4 at
/// K = 13 (docs/bootstrap.md §4).
pub const N16_CHEB_DEGREE: usize = 127;
/// Double-angle steps of [`BootParams::default_n16`]: none (ruling R-M3-8). Degree 127 alone
/// reaches the float64 floor at K = 13, and 7 + 0 = 7 = EvalMod band.
pub const N16_DOUBLE_ANGLE: usize = 0;

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
    /// Hamming weight of the main secret key; `None` = dense ternary (hekit default, -1/0/1 with
    /// probability 1/4, 1/2, 1/4, so h ~ N/2). Must equal `CkksParams::hamming_weight()`, which
    /// is what the key generator uses ([`BootParams::validate`] checks it).
    pub sparse_hw: Option<usize>,
    /// Hamming weight h' of the ephemeral key of sparse-secret encapsulation: the ciphertext is
    /// switched to it at level 0, mod-raised under it, and switched back to the main key at the
    /// top level. `None` = mod-raise under the main key.
    pub ephemeral_hw: Option<usize>,
}

impl BootParams {
    /// Full-slot parameters for [`CkksParams::n16_boot`] (rulings R-M3-5, R-M3-7, R-M3-8):
    /// CoeffToSlot 4, EvalMod 7, SlotToCoeff 4, dense main key (`sparse_hw = None`),
    /// sparse-secret encapsulation to `ephemeral_hw = Some(32)`, and K = 13, degree 127,
    /// r = 0. The EvalMod constants assume that SSE: they are sized for an h' = 32 mod-raise
    /// key (docs/bootstrap.md §3, §4).
    /// Predicted precision 17.4 bits at q_0/Delta = 2^12 (docs/bootstrap.md §6). The 7-level
    /// EvalMod needs the level-optimal Chebyshev evaluation (see [`BootParams::evalmod_depth`]).
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
            ephemeral_hw: Some(N16_EPHEMERAL_HW),
        }
    }

    /// Hamming weight of the key the mod-raise runs under: `ephemeral_hw`, else the real main
    /// key's weight `params.hamming_weight()`, else `N / 2` (the expected weight of the dense
    /// ternary key).
    pub fn mod_raise_hw(&self, params: &CkksParams) -> usize {
        self.ephemeral_hw.unwrap_or_else(|| main_key_hw(params))
    }

    /// Coarse lower bound on `k_intervals`: `7 sigma_I + 1/2` with
    /// `sigma_I = sqrt((h + 1) / 12)` for the mod-raise key weight `h` (docs/bootstrap.md §3).
    /// A 7-sigma Gaussian tail is about 2^-22.5 per bootstrap at N = 2^16, so this is necessary,
    /// not sufficient; the document sizes K for 2^-30 (13 for h' = 32, 32 for h = 192, ~420 for
    /// the dense key).
    pub fn min_k_intervals(&self, params: &CkksParams) -> usize {
        let h = self.mod_raise_hw(params) as f64;
        (7.0 * ((h + 1.0) / 12.0).sqrt() + 0.5).ceil() as usize
    }

    /// Levels EvalMod consumes: `ceil(log2(cheb_degree + 1))` for the Chebyshev evaluation plus
    /// one level per double angle, `ceil(log2(cheb_degree + 1)) + double_angle`.
    ///
    /// The Chebyshev part assumes the **level-optimal, scale-invariant baby-step giant-step
    /// evaluation of Bossuat et al. (Eurocrypt 2021, Alg. 2)**: power-of-two giant steps, and
    /// baby-step scalar products that are not rescaled on their own but whose scales are chosen so
    /// one rescale after the giant-step product lands on the target scale. Task 37 implements it
    /// ([`Evaluator::eval_chebyshev`](crate::Evaluator::eval_chebyshev), whose depth is
    /// [`chebyshev_depth`](crate::bootstrap::chebyshev_depth)). Textbook Paterson-Stockmeyer (or rescaling each `c_i T_i`) costs one level more,
    /// 8 for the default; then the fallback is the e8dbe3e chain with 8 EvalMod levels and the same
    /// r = 0, degree 127 (docs/bootstrap.md §5).
    pub fn evalmod_depth(&self) -> usize {
        let cheb = (self.cheb_degree + 1).next_power_of_two().trailing_zeros() as usize;
        cheb + self.double_angle
    }

    /// Checks the parameters against `params`. Errors (`InvalidParams`): `log_slots` outside
    /// `1..=log_n - 1`; a band of zero levels; the three bands not leaving at least one compute
    /// level above the base; `k_intervals` or `cheb_degree` zero; [`Self::evalmod_depth`] (the
    /// level-optimal count, 7 for the default) above `evalmod_levels`; `sparse_hw` different
    /// from `params.hamming_weight()` (the weight the key generator really uses);
    /// `ephemeral_hw` outside `1..=N` or not below the main key's weight
    /// (`params.hamming_weight()`, or `N / 2` for the dense key); `k_intervals` below
    /// [`Self::min_k_intervals`] (for example the SSE constants without SSE).
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
                "EvalMod depth {} (degree {}, {} double angles, level-optimal evaluation) \
                 exceeds {} levels",
                self.evalmod_depth(),
                self.cheb_degree,
                self.double_angle,
                self.evalmod_levels
            ));
        }
        let n = params.n();
        if self.sparse_hw != params.hamming_weight() {
            return bad(format!(
                "sparse_hw {:?} differs from the parameters' secret Hamming weight {:?}",
                self.sparse_hw,
                params.hamming_weight()
            ));
        }
        if let Some(e) = self.ephemeral_hw {
            if !(1..=n).contains(&e) {
                return bad(format!("ephemeral_hw {e} outside 1..={n}"));
            }
            let main = main_key_hw(params);
            if e >= main {
                return bad(format!(
                    "ephemeral_hw {e} must be below the main key weight {main}"
                ));
            }
        }
        let min_k = self.min_k_intervals(params);
        if self.k_intervals < min_k {
            return bad(format!(
                "k_intervals {} below {min_k} = 7 sigma_I + 1/2 for a mod-raise key of weight {}",
                self.k_intervals,
                self.mod_raise_hw(params)
            ));
        }
        Ok(())
    }
}

/// Weight of the main secret key the key generator uses: `params.hamming_weight()`, or `N / 2`
/// (expected weight) for the dense ternary default.
fn main_key_hw(params: &CkksParams) -> usize {
    params.hamming_weight().unwrap_or(params.n() / 2)
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
            (4, 7, 4)
        );
        assert_eq!(
            (bp.k_intervals, bp.cheb_degree, bp.double_angle),
            (13, 127, 0)
        );
        assert_eq!(bp.sparse_hw, None);
        assert_eq!(bp.ephemeral_hw, Some(32));
    }

    #[test]
    fn min_k_follows_mod_raise_key() {
        let p = CkksParams::n16_boot();
        let d = BootParams::default_n16();
        assert_eq!(d.mod_raise_hw(&p), 32);
        assert_eq!(d.min_k_intervals(&p), 13);
        let p192 = CkksParams::n16_boot().with_hamming_weight(192);
        let h192 = BootParams {
            sparse_hw: Some(192),
            ephemeral_hw: None,
            ..d.clone()
        };
        assert_eq!(h192.min_k_intervals(&p192), 29);
        // The weight comes from the real key, not from `sparse_hw`.
        assert_eq!(h192.min_k_intervals(&p), 367);
        let dense = BootParams {
            ephemeral_hw: None,
            ..d
        };
        assert_eq!(dense.mod_raise_hw(&p), 1 << 15);
        assert_eq!(dense.min_k_intervals(&p), 367);
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

    /// The EvalMod band of `n16_boot` is exactly its seven 58-bit primes, CoeffToSlot the top
    /// four (56-bit) and SlotToCoeff the four (34-bit) just below EvalMod.
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
        assert!(stoc.clone().all(|l| q[l] == 34), "{stoc:?}");
        assert!(evalmod.clone().all(|l| q[l] == 58), "{evalmod:?}");
        assert!(ctos.clone().all(|l| q[l] == 56), "{ctos:?}");
        assert!(q[1..=after].iter().all(|&b| b == 42), "compute band");
        assert_eq!(p.scale_at(*evalmod.start()), 2f64.powi(58));
        assert_eq!(p.scale_at(after), p.scale());
    }

    /// docs/bootstrap.md §4: ceil(log2(degree + 1)) Chebyshev levels plus one per double angle;
    /// degree 127 with r = 0 fills the 7-level band (level-optimal evaluation, ruling R-M3-8).
    #[test]
    fn default_n16_evalmod_depth_fills_band() {
        let bp = BootParams::default_n16();
        assert_eq!(bp.evalmod_depth(), 7);
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
                double_angle: 1,
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
            BootParams {
                sparse_hw: Some(0),
                ..d.clone()
            },
            BootParams {
                ephemeral_hw: Some(0),
                ..d.clone()
            },
            BootParams {
                ephemeral_hw: Some((1 << 16) + 1),
                ..d.clone()
            },
            BootParams {
                ephemeral_hw: Some(1 << 15),
                ..d.clone()
            },
            // `sparse_hw` must match the parameters (dense here).
            BootParams {
                sparse_hw: Some(16),
                ..d.clone()
            },
            BootParams {
                ephemeral_hw: None,
                ..d.clone()
            },
            BootParams {
                k_intervals: 12,
                ..d.clone()
            },
            BootParams {
                sparse_hw: Some(192),
                ephemeral_hw: None,
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
            ..d.clone()
        };
        ok.validate(&p).unwrap();
        assert_eq!(levels_after_boot(&p, &ok), 1);
        // The h = 192 alternative of docs/bootstrap.md §4 (no SSE): K = 32, degree 255, r = 0,
        // 8 EvalMod levels, so it needs the e8dbe3e chain B ([54, 42 x 15, 34 x 4, 58 x 8,
        // 56 x 4], built here), and it is valid only on parameters whose key really has
        // weight 192. (`validate` checks depths and weights, not the prime size of each band.)
        let mut chain_b = vec![54];
        chain_b.extend([42; 15]);
        chain_b.extend([34; 4]);
        chain_b.extend([58; 8]);
        chain_b.extend([56; 4]);
        let p_b = CkksParams::new(16, &chain_b, &[60; 4], 42, 8).unwrap();
        let p192 = p_b.clone().with_hamming_weight(192);
        let h192 = BootParams {
            sparse_hw: Some(192),
            ephemeral_hw: None,
            evalmod_levels: 8,
            k_intervals: 32,
            cheb_degree: 255,
            double_angle: 0,
            ..d.clone()
        };
        assert!(matches!(
            BootParams {
                evalmod_levels: 7,
                ..h192.clone()
            }
            .validate(&p192),
            Err(HekitError::InvalidParams(_))
        ));
        h192.validate(&p192).unwrap();
        assert_eq!(levels_after_boot(&p192, &h192), 15);
        let q = p192.q_bits();
        assert!(q[20..=27].iter().all(|&b| b == 58), "chain B EvalMod band");
        for wrong_key in [&p_b, &p] {
            assert!(matches!(
                h192.validate(wrong_key),
                Err(HekitError::InvalidParams(_))
            ));
        }
        // Ephemeral key not sparser than a weight-16 main key.
        let p16 = CkksParams::n16_boot().with_hamming_weight(16);
        let e32 = BootParams {
            sparse_hw: Some(16),
            ..d.clone()
        };
        assert!(matches!(
            e32.validate(&p16),
            Err(HekitError::InvalidParams(_))
        ));
        BootParams {
            sparse_hw: Some(16),
            ephemeral_hw: Some(8),
            k_intervals: 8,
            ..d
        }
        .validate(&p16)
        .unwrap();
    }
}
