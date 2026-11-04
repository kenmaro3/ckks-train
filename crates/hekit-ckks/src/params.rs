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
        for p in [CkksParams::seal_2020(), CkksParams::n16_train()] {
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
        for p in [CkksParams::seal_2020(), CkksParams::n16_train()] {
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
    fn hamming_weight_builder() {
        let p = CkksParams::new(4, &[30, 20], &[30], 20, 1)
            .unwrap()
            .with_hamming_weight(8);
        assert_eq!(p.hamming_weight(), Some(8));
    }
}
