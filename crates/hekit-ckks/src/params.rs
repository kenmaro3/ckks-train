//! CKKS parameter sets and presets.

use hekit_core::error::{HekitError, Result};
use hekit_core::primes::ntt_primes;
use hekit_core::rns::RnsBasis;
use std::sync::Arc;

/// Smallest supported `log2(N)`.
pub const MIN_LOG_N: u32 = 1;
/// Largest supported `log2(N)`.
pub const MAX_LOG_N: u32 = 17;

/// `n16_train` data-prime count L (confirmed by Task 22, >= 141 bits, see docs/security.md;
/// rule for adjustments: dnum | q.len(), k = ceil(max_digit_bits / 58)).
pub const N16_TRAIN_L: usize = 31;
/// `n16_train` special-prime count k (confirmed by Task 22, >= 141 bits, see docs/security.md;
/// rule for adjustments: dnum | q.len(), k = ceil(max_digit_bits / 58)).
pub const N16_TRAIN_K: usize = 4;
/// `n16_train` digit count (confirmed by Task 22, >= 141 bits, see docs/security.md;
/// rule for adjustments: dnum | q.len(), k = ceil(max_digit_bits / 58)).
pub const N16_TRAIN_DNUM: usize = 8;

/// `n16_boot` compute levels above the base (42-bit primes, levels 1..=16), see
/// docs/bootstrap.md §1 (chain of rulings R-M3-7, R-M3-8).
pub const N16_BOOT_COMPUTE_LEVELS: usize = 16;
/// `n16_boot` SlotToCoeff levels (34-bit primes, levels 17..=20).
pub const N16_BOOT_STOC_LEVELS: usize = 4;
/// `n16_boot` EvalMod levels (58-bit primes, levels 21..=27): degree-127 Chebyshev, no double
/// angle, which needs the level-optimal evaluation (ruling R-M3-8). Fallback if Task 37 cannot
/// fit 7 levels: the e8dbe3e chain (EvalMod 8, compute 15, log PQ 1748), docs/bootstrap.md §5.
pub const N16_BOOT_EVALMOD_LEVELS: usize = 7;
/// `n16_boot` CoeffToSlot levels (56-bit primes, levels 28..=31): CoeffToSlot works on values of
/// size about K q_0, so the rounding of its plaintext diagonals needs large primes
/// (docs/bootstrap.md §6).
pub const N16_BOOT_CTOS_LEVELS: usize = 4;
/// `n16_boot` special-prime count (60-bit each: log P = 240 exceeds the largest digit, 4 x 58 =
/// 232, by 8 bits, so the key-error term of the EvalMod digits is negligible; ruling R-M3-5).
/// Named like [`N16_TRAIN_K`]; not to be confused with the EvalMod interval `K`
/// (`BootParams::k_intervals`).
pub const N16_BOOT_K: usize = 4;
/// `n16_boot` digit count (32 q primes, 4 per digit).
pub const N16_BOOT_DNUM: usize = 8;

/// Bit sizes `(q_bits, p_bits)` of `n16_boot`: q from level 0 up is
/// `[54, 42 x 16 (compute), 34 x 4 (SlotToCoeff), 58 x 7 (EvalMod), 56 x 4 (CoeffToSlot)]`,
/// p is `[60 x 4]`; log PQ = 1732. The 54-bit base gives q_0 / Delta = 2^12
/// (docs/bootstrap.md §1, §6).
pub fn n16_boot_bits() -> (Vec<u32>, Vec<u32>) {
    let mut q = vec![54];
    q.extend(vec![42; N16_BOOT_COMPUTE_LEVELS]);
    q.extend(vec![34; N16_BOOT_STOC_LEVELS]);
    q.extend(vec![58; N16_BOOT_EVALMOD_LEVELS]);
    q.extend(vec![56; N16_BOOT_CTOS_LEVELS]);
    (q, vec![60; N16_BOOT_K])
}

/// CKKS parameters: ring degree, RNS moduli Q (data) and P (special), scale, key-switch digits.
#[derive(Clone, Debug)]
pub struct CkksParams {
    log_n: u32,
    q: Vec<u64>,
    p: Vec<u64>,
    q_bits: Vec<u32>,
    p_bits: Vec<u32>,
    scale: f64,
    dnum: usize,
    hamming_weight: Option<usize>,
    basis_q: Arc<RnsBasis>,
    basis_p: Arc<RnsBasis>,
}

impl CkksParams {
    /// Builds parameters. Primes are NTT-friendly (`q = 1 mod 2N`) and generated jointly,
    /// exactly as SEAL 4.1's `CoeffModulus::Create(N, q_bits ++ p_bits)`: for each distinct
    /// bit size `b` occurring `c` times, the `c` largest primes of that size are generated in
    /// descending order and handed out **from the smallest** (SEAL's `pop_back`) in order of
    /// appearance, so the first `b`-bit entry gets the `c`-th largest prime and the last one
    /// the largest. (seal_2020: `q_0` is the second-largest 60-bit prime, the special prime
    /// the largest; the 40-bit primes ascend.)
    ///
    /// Errors (`InvalidParams`): `log_n` outside `[MIN_LOG_N, MAX_LOG_N]`, empty `q_bits` or
    /// `p_bits`, any bit size outside `2..=60`, `dnum` zero or not dividing `q_bits.len()`,
    /// or `scale_bits` not in `1..=60`.
    pub fn new(
        log_n: u32,
        q_bits: &[u32],
        p_bits: &[u32],
        scale_bits: u32,
        dnum: usize,
    ) -> Result<CkksParams> {
        let bad = |m: String| Err(HekitError::InvalidParams(m));
        if !(MIN_LOG_N..=MAX_LOG_N).contains(&log_n) {
            return bad(format!("log_n {log_n} outside [{MIN_LOG_N}, {MAX_LOG_N}]"));
        }
        if q_bits.is_empty() || p_bits.is_empty() {
            return bad("q_bits and p_bits must be non-empty".into());
        }
        if let Some(b) = q_bits
            .iter()
            .chain(p_bits)
            .find(|&&b| !(2..=60).contains(&b))
        {
            return bad(format!("prime bit size {b} outside 2..=60"));
        }
        if !(1..=60).contains(&scale_bits) {
            return bad(format!("scale_bits {scale_bits} outside 1..=60"));
        }
        if dnum == 0 || q_bits.len() % dnum != 0 {
            return bad(format!(
                "dnum {dnum} must divide the number of q primes {}",
                q_bits.len()
            ));
        }
        let n = 1usize << log_n;
        let all: Vec<u32> = q_bits.iter().chain(p_bits).copied().collect();
        let mut distinct = all.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let pools: Vec<(u32, Vec<u64>)> = distinct
            .iter()
            .map(|&b| {
                let count = all.iter().filter(|&&x| x == b).count();
                ntt_primes(b, n, count).map(|v| (b, v))
            })
            .collect::<Result<_>>()?;
        let mut next = vec![0usize; pools.len()];
        let primes: Vec<u64> = all
            .iter()
            .map(|&b| {
                let i = pools.iter().position(|(pb, _)| *pb == b).unwrap();
                // SEAL: `prime_table[size].back()` then `pop_back()`.
                let pool = &pools[i].1;
                let v = pool[pool.len() - 1 - next[i]];
                next[i] += 1;
                v
            })
            .collect();
        let (q, p) = primes.split_at(q_bits.len());
        let (q, p) = (q.to_vec(), p.to_vec());
        let basis_q = RnsBasis::new(n, &q)?;
        let basis_p = RnsBasis::new(n, &p)?;
        Ok(CkksParams {
            log_n,
            q,
            p,
            q_bits: q_bits.to_vec(),
            p_bits: p_bits.to_vec(),
            scale: (scale_bits as f64).exp2(),
            dnum,
            hamming_weight: None,
            basis_q,
            basis_p,
        })
    }

    /// SEAL-2020 preset: N = 2^15, moduli {60, 40x9, 60}, scale 2^40, dnum 10.
    pub fn seal_2020() -> CkksParams {
        let mut q_bits = vec![60];
        q_bits.extend([40; 9]);
        Self::new(15, &q_bits, &[60], 40, 10).expect("seal_2020 preset is valid")
    }

    /// Training preset: N = 2^16, q = {58, 42 x L}, p = {58 x k}, scale 2^42.
    /// Constants confirmed by Task 22 (see [`N16_TRAIN_L`], `docs/security.md`).
    pub fn n16_train() -> CkksParams {
        let mut q_bits = vec![58];
        q_bits.extend(vec![42; N16_TRAIN_L]);
        let p_bits = vec![58; N16_TRAIN_K];
        Self::new(16, &q_bits, &p_bits, 42, N16_TRAIN_DNUM).expect("n16_train preset is valid")
    }

    /// Bootstrapping preset: N = 2^16, q and p from [`n16_boot_bits`] (32 q primes, max level
    /// 31, log PQ = 1732), scale 2^42, dnum 8, dense ternary secret. Per-level scales come from
    /// [`CkksParams::scale_at`]. Level design and precision analysis: `docs/bootstrap.md`; the
    /// 128-bit security estimate is Task 39's.
    pub fn n16_boot() -> CkksParams {
        let (q_bits, p_bits) = n16_boot_bits();
        Self::new(16, &q_bits, &p_bits, 42, N16_BOOT_DNUM).expect("n16_boot preset is valid")
    }

    /// **Insecure toy parameters, for tests only** (N = 2^5 gives no security at all): the
    /// `n16_boot` band layout in miniature for the bootstrapping tests, with `log_slots` 4
    /// (16 full slots). q from level 0 up is `[54, 42 (compute), 34 x 4 (SlotToCoeff),
    /// 58 x 2 (EvalMod), 56 x 4 (CoeffToSlot)]` (max level 11), p = `[60]`, scale 2^42,
    /// dnum 12 (one prime per digit, every digit below p), dense ternary secret.
    /// `BootParams::default_n16` does not fit it: tests build their own `BootParams` (four
    /// CoeffToSlot, two EvalMod and four SlotToCoeff levels, an ephemeral weight and `K` for
    /// N = 2^5) and pass it through `BootParams::validate`.
    pub fn test_tiny() -> CkksParams {
        let mut q_bits = vec![54, 42];
        q_bits.extend([34; 4]);
        q_bits.extend([58; 2]);
        q_bits.extend([56; 4]);
        Self::new(5, &q_bits, &[60], 42, 12).expect("test_tiny preset is valid")
    }

    /// Returns a copy with a ternary secret-key Hamming weight (validated `<= N` by KeyGenerator).
    pub fn with_hamming_weight(mut self, h: usize) -> CkksParams {
        self.hamming_weight = Some(h);
        self
    }

    /// Ring degree N.
    pub fn n(&self) -> usize {
        1 << self.log_n
    }
    /// log2(N).
    pub fn log_n(&self) -> u32 {
        self.log_n
    }
    /// Number of plaintext slots, N/2.
    pub fn slots(&self) -> usize {
        self.n() / 2
    }
    /// Highest ciphertext level, `q.len() - 1`.
    pub fn max_level(&self) -> usize {
        self.q.len() - 1
    }
    /// Total modulus bits, log2(P * Q).
    pub fn log_pq(&self) -> f64 {
        self.q
            .iter()
            .chain(&self.p)
            .map(|&m| (m as f64).log2())
            .sum()
    }
    /// Data primes q_0..q_L.
    pub fn q(&self) -> &[u64] {
        &self.q
    }
    /// Special primes.
    pub fn p(&self) -> &[u64] {
        &self.p
    }
    /// Nominal bit sizes of the q primes.
    pub fn q_bits(&self) -> &[u32] {
        &self.q_bits
    }
    /// Nominal bit sizes of the p primes.
    pub fn p_bits(&self) -> &[u32] {
        &self.p_bits
    }
    /// Default scale Delta.
    pub fn scale(&self) -> f64 {
        self.scale
    }
    /// Nominal default scale at `level`: `2^(bits of q_level)` for `level >= 1`, and the preset
    /// scale [`CkksParams::scale`] at level 0. It is the scale a ciphertext-ciphertext product
    /// keeps when it is rescaled by `q_level`, so it is the working scale of the compute band
    /// (2^42) and of the `n16_boot` EvalMod band (2^58). It is **not** an invariant of the
    /// bootstrap bands: in CoeffToSlot and SlotToCoeff the ciphertext is multiplied only by
    /// plaintext diagonals and carries its own scale (about 2^54 to 2^58 in CoeffToSlot against
    /// a nominal 2^56, about 2^46 to 2^42 in SlotToCoeff against a nominal 2^34; docs/bootstrap.md
    /// §2), and exact scales differ from these powers of two by the prime / 2^bits ratios.
    /// For single-size presets it equals `scale()` at every level.
    ///
    /// # Panics
    /// If `level > max_level()`.
    pub fn scale_at(&self, level: usize) -> f64 {
        assert!(
            level <= self.max_level(),
            "level {level} above max_level {}",
            self.max_level()
        );
        if level == 0 {
            self.scale
        } else {
            (self.q_bits[level] as f64).exp2()
        }
    }
    /// Number of key-switching digits.
    pub fn dnum(&self) -> usize {
        self.dnum
    }
    /// Primes per key-switching digit, `q.len() / dnum`.
    pub fn alpha(&self) -> usize {
        self.q.len() / self.dnum
    }
    /// Secret-key Hamming weight, if set.
    pub fn hamming_weight(&self) -> Option<usize> {
        self.hamming_weight
    }
    /// RNS basis of all q primes.
    pub fn basis_q(&self) -> &Arc<RnsBasis> {
        &self.basis_q
    }
    /// RNS basis of the special primes.
    pub fn basis_p(&self) -> &Arc<RnsBasis> {
        &self.basis_p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_2020_shape() {
        let p = CkksParams::seal_2020();
        assert!((p.log_pq() - 480.0).abs() <= 1.0, "{}", p.log_pq());
        assert_eq!(p.max_level(), 9);
        assert_eq!(p.n(), 1 << 15);
        assert_eq!(p.slots(), 1 << 14);
        assert_eq!((p.dnum(), p.alpha(), p.p().len()), (10, 1, 1));
        assert_eq!(p.scale(), 2f64.powi(40));
        assert_eq!(p.hamming_weight(), None);
        assert_eq!(p.basis_q().len(), 10);
        assert_eq!(p.basis_p().len(), 1);
    }

    /// The exact list printed by SEAL 4.1.2 for `CoeffModulus::Create(32768, {60, 40 x 9, 60})`
    /// (`tools/seal-kat/interop`, `values.json` field `coeff_modulus`).
    #[test]
    fn seal_2020_matches_seal_primes() {
        let p = CkksParams::seal_2020();
        let seal: [u64; 11] = [
            1152921504598720513,
            1099501731841,
            1099502518273,
            1099502714881,
            1099503370241,
            1099503894529,
            1099504549889,
            1099506515969,
            1099507695617,
            1099510054913,
            1152921504606584833,
        ];
        assert_eq!(p.q(), &seal[..10]);
        assert_eq!(p.p(), &seal[10..]);
    }

    /// SEAL hands out each bit size's primes from the smallest of the generated (largest)
    /// primes upwards: `{58, 58, 58}` gets the third, second and first largest 58-bit primes.
    #[test]
    fn prime_assignment_follows_seal() {
        let p = CkksParams::new(12, &[58, 40, 58], &[58], 40, 3).unwrap();
        let p58 = ntt_primes(58, 1 << 12, 3).unwrap();
        let p40 = ntt_primes(40, 1 << 12, 1).unwrap();
        assert_eq!(p.q(), &[p58[2], p40[0], p58[1]]);
        assert_eq!(p.p(), &[p58[0]]);
    }

    #[test]
    fn all_primes_distinct_and_ntt_friendly() {
        for p in [
            CkksParams::seal_2020(),
            CkksParams::n16_train(),
            CkksParams::n16_boot(),
        ] {
            let all: Vec<u64> = p.q().iter().chain(p.p()).copied().collect();
            for (i, a) in all.iter().enumerate() {
                assert!(!all[..i].contains(a));
                assert_eq!(a % (2 * p.n() as u64), 1);
            }
            for (&m, &b) in p
                .q()
                .iter()
                .zip(p.q_bits())
                .chain(p.p().iter().zip(p.p_bits()))
            {
                assert_eq!(64 - m.leading_zeros(), b);
            }
        }
    }

    #[test]
    fn n16_train_shape() {
        let p = CkksParams::n16_train();
        assert_eq!(p.n(), 1 << 16);
        assert_eq!(p.q().len(), 32);
        assert_eq!((p.dnum(), p.alpha(), p.p().len()), (8, 4, 4));
        assert!(p.log_pq() <= 1750.0, "{}", p.log_pq());
    }

    #[test]
    fn digit_bits_within_log_p() {
        for p in [
            CkksParams::seal_2020(),
            CkksParams::n16_train(),
            CkksParams::n16_boot(),
        ] {
            // Nominal bit sizes: exact log2 sums of equal-size primes differ by ~1e-11.
            let log_p: u32 = p.p_bits().iter().sum();
            for d in p.q_bits().chunks(p.alpha()) {
                assert!(
                    d.iter().sum::<u32>() <= log_p,
                    "digit {d:?} > log P {log_p}"
                );
            }
        }
    }

    #[test]
    fn n16_boot_shape() {
        let p = CkksParams::n16_boot();
        assert_eq!(p.n(), 1 << 16);
        assert_eq!(p.max_level(), 31);
        assert!((p.log_pq() - 1732.0).abs() <= 2.0, "{}", p.log_pq());
        assert!(p.log_pq() <= 1750.0, "{}", p.log_pq());
        assert_eq!((p.dnum(), p.alpha(), p.p().len()), (8, 4, 4));
        assert_eq!(p.scale(), 2f64.powi(42));
        assert_eq!(p.hamming_weight(), None);
    }

    /// Level i is a ciphertext over primes 0..=i; the bands from the bottom are base, compute,
    /// SlotToCoeff, EvalMod, CoeffToSlot (docs/bootstrap.md §1).
    #[test]
    fn n16_boot_bands() {
        let p = CkksParams::n16_boot();
        let q = p.q_bits();
        assert_eq!(q[0], 54);
        assert!(q[1..=16].iter().all(|&b| b == 42), "compute band");
        assert!(q[17..=20].iter().all(|&b| b == 34), "SlotToCoeff band");
        assert!(q[21..=27].iter().all(|&b| b == 58), "EvalMod band");
        assert!(q[28..=31].iter().all(|&b| b == 56), "CoeffToSlot band");
        assert_eq!(p.p_bits(), &[60; 4]);
    }

    #[test]
    fn n16_boot_scale_at() {
        let p = CkksParams::n16_boot();
        assert_eq!(p.scale_at(0), 2f64.powi(42));
        assert_eq!(p.scale_at(24), 2f64.powi(58));
        assert_eq!(p.scale_at(31), 2f64.powi(56));
        for l in 1..=16 {
            assert_eq!(p.scale_at(l), 2f64.powi(42), "level {l}");
        }
        for l in 17..=20 {
            assert_eq!(p.scale_at(l), 2f64.powi(34), "level {l}");
        }
        for l in 21..=27 {
            assert_eq!(p.scale_at(l), 2f64.powi(58), "level {l}");
        }
        for l in 28..=31 {
            assert_eq!(p.scale_at(l), 2f64.powi(56), "level {l}");
        }
    }

    /// For the single-size presets `scale_at` is the default scale at every level.
    #[test]
    fn scale_at_is_default_scale_for_uniform_presets() {
        for p in [CkksParams::seal_2020(), CkksParams::n16_train()] {
            for l in 0..=p.max_level() {
                assert_eq!(p.scale_at(l), p.scale(), "level {l}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "level 32 above max_level 31")]
    fn scale_at_above_max_level_panics() {
        CkksParams::n16_boot().scale_at(32);
    }

    #[test]
    fn n16_boot_rejects_dnum_7() {
        let (q_bits, p_bits) = n16_boot_bits();
        assert_eq!(q_bits.len(), 32);
        assert!(CkksParams::new(16, &q_bits, &p_bits, 42, 7).is_err());
    }

    /// Exact (not nominal) digit sizes: every digit, including the all-58-bit EvalMod digit
    /// (q_24..q_27), is at least 8 bits below log P = 4 x 60.
    #[test]
    fn n16_boot_digits_below_p_exactly() {
        let p = CkksParams::n16_boot();
        let log_p: f64 = p.p().iter().map(|&m| (m as f64).log2()).sum();
        for d in p.q().chunks(p.alpha()) {
            let bits: f64 = d.iter().map(|&m| (m as f64).log2()).sum();
            assert!(bits <= log_p - 7.99, "digit {bits} vs log P {log_p}");
        }
    }

    #[test]
    fn invalid_params_rejected() {
        assert!(CkksParams::new(15, &[60, 40], &[60], 40, 3).is_err());
        assert!(CkksParams::new(15, &[60, 40], &[60], 40, 0).is_err());
        assert!(CkksParams::new(0, &[60, 40], &[60], 40, 1).is_err());
        assert!(CkksParams::new(18, &[60, 40], &[60], 40, 1).is_err());
        assert!(CkksParams::new(15, &[], &[60], 40, 1).is_err());
        assert!(CkksParams::new(15, &[61, 40], &[60], 40, 1).is_err());
        assert!(CkksParams::new(15, &[60, 40], &[61], 40, 1).is_err());
        assert!(CkksParams::new(15, &[60, 40], &[60], 61, 1).is_err());
        assert!(CkksParams::new(15, &[60, 40], &[60], 40, 2).is_ok());
    }

    #[test]
    fn test_tiny_shape() {
        let p = CkksParams::test_tiny();
        assert_eq!((p.n(), p.slots(), p.max_level()), (32, 16, 11));
        assert_eq!(
            p.q_bits(),
            &[54, 42, 34, 34, 34, 34, 58, 58, 56, 56, 56, 56]
        );
        assert_eq!((p.dnum(), p.alpha(), p.p_bits()), (12, 1, &[60][..]));
        assert_eq!(p.scale(), 2f64.powi(42));
        assert_eq!(p.scale_at(1), p.scale());
        assert_eq!(p.scale_at(7), 2f64.powi(58));
    }

    #[test]
    fn hamming_weight_builder() {
        let p = CkksParams::new(4, &[30, 20], &[30], 20, 1)
            .unwrap()
            .with_hamming_weight(8);
        assert_eq!(p.hamming_weight(), Some(8));
    }
}
