//! Deterministic primality testing and NTT-friendly prime search.

use crate::error::{HekitError, Result};

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    (a as u128 * b as u128 % m as u128) as u64
}

fn pow_mod(mut a: u64, mut e: u64, m: u64) -> u64 {
    let mut acc = 1 % m;
    a %= m;
    while e > 0 {
        if e & 1 == 1 {
            acc = mul_mod(acc, a, m);
        }
        a = mul_mod(a, a, m);
        e >>= 1;
    }
    acc
}

/// Deterministic Miller-Rabin for all `u64`, with bases 2, 3, 5, ..., 37
/// (the first twelve primes suffice for every `n < 3.3 * 10^24`).
pub fn is_prime_u64(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for p in BASES {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'bases: for a in BASES {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// Primes `q` with `2^(bits-1) < q < 2^bits` and `q = 1 (mod 2n)`, in descending
/// order without duplicates; returns the `count` primes after skipping the
/// `skip` largest ones.
///
/// `skip` splits the descending stream (e.g. to take disjoint batches of primes of one
/// size). It does **not** by itself reproduce SEAL's `CoeffModulus::Create`: SEAL
/// generates the `c` needed primes of one bit size in descending order and hands them out
/// from the smallest (`pop_back`) in order of appearance, i.e. the whole set of `c` primes
/// reversed; `hekit_ckks::CkksParams::new` does that with [`ntt_primes`].
///
/// Fails with `InvalidParams` if `bits` is not in `2..=60`, `n` is not a power of
/// two, or fewer than `skip + count` such primes exist.
pub fn ntt_primes_skip(bits: u32, n: usize, skip: usize, count: usize) -> Result<Vec<u64>> {
    if !(2..=60).contains(&bits) {
        return Err(HekitError::InvalidParams(format!(
            "prime bit size {bits} must be in 2..=60"
        )));
    }
    if !n.is_power_of_two() {
        return Err(HekitError::InvalidParams(format!(
            "n = {n} must be a power of two"
        )));
    }
    let step = (n as u128) * 2;
    let (lo, hi) = (1u128 << (bits - 1), 1u128 << bits);
    let not_enough = || {
        HekitError::InvalidParams(format!(
            "not enough {bits}-bit primes = 1 mod {step} (skip {skip}, count {count})"
        ))
    };
    if step >= hi {
        return if count == 0 {
            Ok(Vec::new())
        } else {
            Err(not_enough())
        };
    }
    // Largest value < 2^bits that is 1 mod 2n.
    let mut q = (hi - 2) / step * step + 1;
    let mut seen = 0usize;
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        if q <= lo {
            return Err(not_enough());
        }
        if is_prime_u64(q as u64) {
            if seen >= skip {
                out.push(q as u64);
            }
            seen += 1;
        }
        if q < step {
            return Err(not_enough());
        }
        q -= step;
    }
    Ok(out)
}

/// The `count` largest primes `q` with `2^(bits-1) < q < 2^bits` and `q = 1 (mod 2n)`,
/// in descending order. See [`ntt_primes_skip`] for the error conditions.
pub fn ntt_primes(bits: u32, n: usize, count: usize) -> Result<Vec<u64>> {
    ntt_primes_skip(bits, n, 0, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HekitError;

    #[test]
    fn known_primes_and_composites() {
        assert!(is_prime_u64(1_152_921_504_606_846_883)); // 2^60 - 93
        assert!(is_prime_u64(2) && is_prime_u64(3) && is_prime_u64(u64::MAX - 58));
        assert!(!is_prime_u64(0) && !is_prime_u64(1) && !is_prime_u64(u64::MAX));
        // Carmichael numbers and strong pseudoprimes to the first bases.
        for c in [561u64, 1105, 1729, 3_215_031_751, 3_825_123_056_546_413_051] {
            assert!(!is_prime_u64(c), "{c}");
        }
        // 2^61 - 1 (Mersenne), and a product of two large primes.
        assert!(is_prime_u64((1 << 61) - 1));
        assert!(!is_prime_u64(4_294_967_291 * 4_294_967_279));
    }

    #[test]
    fn matches_trial_division_below_20000() {
        for n in 0u64..20_000 {
            let naive = n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0);
            assert_eq!(is_prime_u64(n), naive, "{n}");
        }
    }

    #[test]
    fn seal_style_primes() {
        let ps = ntt_primes(40, 1 << 15, 9).unwrap();
        assert_eq!(ps.len(), 9);
        for (i, &q) in ps.iter().enumerate() {
            assert!(is_prime_u64(q));
            assert_eq!((q - 1) % (2 << 15), 0);
            assert!(q > 1 << 39 && q < 1 << 40);
            if i > 0 {
                assert!(q < ps[i - 1]);
            }
        }
    }

    #[test]
    fn sixty_bit_and_skip() {
        let ps = ntt_primes(60, 1 << 15, 3).unwrap();
        assert!(ps.iter().all(|&q| q > 1 << 59 && q < 1 << 60));
        let tail = ntt_primes_skip(60, 1 << 15, 1, 2).unwrap();
        assert_eq!(tail, ps[1..]);
        assert_eq!(ntt_primes(60, 1 << 15, 0).unwrap(), Vec::<u64>::new());
    }

    #[test]
    fn errors() {
        assert!(matches!(
            ntt_primes(61, 1 << 15, 1),
            Err(HekitError::InvalidParams(_))
        ));
        assert!(matches!(
            ntt_primes(40, 0, 1),
            Err(HekitError::InvalidParams(_))
        ));
        assert!(matches!(
            ntt_primes(40, 3, 1),
            Err(HekitError::InvalidParams(_))
        ));
        // bits = 17, 2n = 2^16: the only candidate in (2^16, 2^17) is 65537, which is prime.
        assert_eq!(ntt_primes(17, 1 << 15, 1).unwrap(), vec![65537]);
        assert!(matches!(
            ntt_primes(17, 1 << 15, 5),
            Err(HekitError::InvalidParams(_))
        ));
    }
}
