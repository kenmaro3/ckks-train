//! Word-sized (< 2^60) modular arithmetic with Barrett and Shoup multiplication.

use crate::error::{HekitError, Result};

/// A modulus `2 <= q < 2^60` with precomputed Barrett constant `floor(2^128 / q)`.
///
/// Unless stated otherwise, operands of the arithmetic methods must already be
/// reduced (`< q`); this is checked with `debug_assert!`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Modulus {
    q: u64,
    /// `floor(2^128 / q)` as `(low word, high word)`.
    barrett: (u64, u64),
}

/// A fixed multiplicand `w < q` with its Shoup companion `floor(w * 2^64 / q)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shoup {
    w: u64,
    w_shoup: u64,
}

impl Shoup {
    /// The multiplicand `w`.
    pub fn value(&self) -> u64 {
        self.w
    }

    /// The companion `floor(w * 2^64 / q)`.
    pub fn companion(&self) -> u64 {
        self.w_shoup
    }
}

/// High 128 bits of the 256-bit product `x * y`.
fn mulhi_u128(x: u128, y: u128) -> u128 {
    let (x0, x1) = (x as u64 as u128, x >> 64);
    let (y0, y1) = (y as u64 as u128, y >> 64);
    let ll = x0 * y0;
    let lh = x0 * y1;
    let hl = x1 * y0;
    let hh = x1 * y1;
    // Middle column: carry out of the low 64 bits plus the low halves of lh, hl.
    let mid = (ll >> 64) + (lh as u64 as u128) + (hl as u64 as u128);
    hh + (lh >> 64) + (hl >> 64) + (mid >> 64)
}

impl Modulus {
    /// Creates a modulus. Fails with `InvalidParams` unless `2 <= q < 2^60`.
    pub fn new(q: u64) -> Result<Modulus> {
        if !(2..(1u64 << 60)).contains(&q) {
            return Err(HekitError::InvalidParams(format!(
                "modulus {q} must satisfy 2 <= q < 2^60"
            )));
        }
        let qq = q as u128;
        // floor(2^128 / q) = floor((2^128 - 1) / q), plus one when q | 2^128.
        let mut ratio = u128::MAX / qq;
        if u128::MAX % qq + 1 == qq {
            ratio += 1;
        }
        Ok(Modulus {
            q,
            barrett: (ratio as u64, (ratio >> 64) as u64),
        })
    }

    /// The modulus value `q`.
    pub fn value(&self) -> u64 {
        self.q
    }

    /// Reduces an arbitrary `u128` below `q^2` (in particular `a*b` with `a, b < q`).
    ///
    /// The Barrett quotient estimate is off by at most 2, so two conditional
    /// subtractions give the exact remainder.
    fn reduce_wide(&self, x: u128) -> u64 {
        let ratio = (self.barrett.0 as u128) | ((self.barrett.1 as u128) << 64);
        let est = mulhi_u128(x, ratio);
        // x < q^2 < 2^120, so est*q <= x and the difference fits in u64 range (< 3q).
        let mut r = (x - est * self.q as u128) as u64;
        if r >= self.q {
            r -= self.q;
        }
        if r >= self.q {
            r -= self.q;
        }
        r
    }

    /// `(a + b) mod q` for `a, b < q`.
    pub fn add(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q && b < self.q);
        let s = a + b;
        if s >= self.q { s - self.q } else { s }
    }

    /// `(a - b) mod q` for `a, b < q`.
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q && b < self.q);
        if a >= b { a - b } else { a + self.q - b }
    }

    /// `(a * b) mod q` via Barrett reduction. Exact for all `a, b < q`
    /// (precondition, `debug_assert`ed).
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q && b < self.q);
        self.reduce_wide(a as u128 * b as u128)
    }

    /// `a^e mod q` by square-and-multiply. `a` is reduced first; `a^0 = 1 mod q`.
    pub fn pow(&self, a: u64, mut e: u64) -> u64 {
        let mut base = a % self.q;
        let mut acc = 1 % self.q;
        while e > 0 {
            if e & 1 == 1 {
                acc = self.mul(acc, base);
            }
            base = self.mul(base, base);
            e >>= 1;
        }
        acc
    }

    /// Modular inverse of `a` (reduced first) by the extended Euclidean algorithm;
    /// `None` when `gcd(a, q) != 1` (also for `a = 0`). Works for composite `q`.
    pub fn inv(&self, a: u64) -> Option<u64> {
        let (mut r0, mut r1) = (self.q as i128, (a % self.q) as i128);
        let (mut t0, mut t1) = (0i128, 1i128);
        while r1 != 0 {
            let quo = r0 / r1;
            (r0, r1) = (r1, r0 - quo * r1);
            (t0, t1) = (t1, t0 - quo * t1);
        }
        if r0 != 1 {
            return None;
        }
        Some(t0.rem_euclid(self.q as i128) as u64)
    }

    /// Precomputes the Shoup companion of `w` (`w` is reduced first).
    pub fn shoup(&self, w: u64) -> Shoup {
        let w = w % self.q;
        Shoup {
            w,
            w_shoup: (((w as u128) << 64) / self.q as u128) as u64,
        }
    }

    /// Lazy Shoup product: returns a value in `[0, 2q)` congruent to `a * s.w`.
    /// Valid for any `a: u64` (not only `a < q`); intended for lazy NTT butterflies.
    pub fn mul_shoup_lazy(&self, a: u64, s: &Shoup) -> u64 {
        let qhat = ((a as u128 * s.w_shoup as u128) >> 64) as u64;
        a.wrapping_mul(s.w).wrapping_sub(qhat.wrapping_mul(self.q))
    }

    /// Shoup product `(a * s.w) mod q`, fully reduced to `[0, q)`. Valid for any `a: u64`.
    pub fn mul_shoup(&self, a: u64, s: &Shoup) -> u64 {
        let r = self.mul_shoup_lazy(a, s);
        if r >= self.q { r - self.q } else { r }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barrett_constant_is_floor_2_128_over_q() {
        for q in [
            2u64,
            3,
            4,
            1 << 59,
            (1 << 60) - 1,
            1_152_921_504_606_846_883,
        ] {
            let m = Modulus::new(q).unwrap();
            let r = (m.barrett.0 as u128) | ((m.barrett.1 as u128) << 64);
            let lo = r.wrapping_mul(q as u128);
            let rem = (u128::MAX % q as u128 + 1) % q as u128; // 2^128 mod q
            // 2^128 = r*q + rem  =>  r*q = 2^128 - rem (wraps to 0 - rem)
            assert_eq!(lo, 0u128.wrapping_sub(rem), "q={q}");
        }
    }

    #[test]
    fn worst_case_mul() {
        for q in [
            3u64,
            (1 << 60) - 1,
            (1 << 59) + 1,
            1_152_921_504_606_846_883,
        ] {
            let m = Modulus::new(q).unwrap();
            let a = q - 1;
            assert_eq!(m.mul(a, a), 1 % q);
            assert_eq!(m.mul(a, 0), 0);
            let s = m.shoup(a);
            assert_eq!(m.mul_shoup(a, &s), 1 % q);
            assert_eq!(
                m.mul_shoup(u64::MAX, &s),
                (u64::MAX as u128 * a as u128 % q as u128) as u64
            );
        }
    }

    #[test]
    fn inv_small() {
        let m = Modulus::new(7).unwrap();
        assert_eq!(m.inv(3), Some(5));
        assert_eq!(m.inv(0), None);
        assert_eq!(Modulus::new(8).unwrap().inv(4), None);
        assert_eq!(Modulus::new(2).unwrap().inv(1), Some(1));
    }
}
