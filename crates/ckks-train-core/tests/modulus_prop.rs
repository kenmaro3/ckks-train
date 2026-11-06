//! Property tests for `Modulus` (Barrett and Shoup multiplication).
use ckks_train_core::{HekitError, Modulus};
use proptest::prelude::*;

/// Random odd modulus near 2^59 (and anywhere in [3, 2^60)).
fn odd_q() -> impl Strategy<Value = u64> {
    prop_oneof![
        (((1u64 << 58) + 1)..((1u64 << 59) + (1 << 58))).prop_map(|x| x | 1),
        (3u64..(1 << 60)).prop_map(|x| x | 1),
    ]
}

proptest! {
    #[test]
    fn mul_matches_u128(q in odd_q(), a in any::<u64>(), b in any::<u64>()) {
        let m = Modulus::new(q).unwrap();
        let (a, b) = (a % q, b % q);
        prop_assert_eq!(m.mul(a, b), (a as u128 * b as u128 % q as u128) as u64);
    }

    #[test]
    fn mul_matches_u128_extremes(q in odd_q(), da in 0u64..3, db in 0u64..3) {
        let m = Modulus::new(q).unwrap();
        let (a, b) = (q - 1 - da.min(q - 1), q - 1 - db.min(q - 1));
        prop_assert_eq!(m.mul(a, b), (a as u128 * b as u128 % q as u128) as u64);
    }

    #[test]
    fn add_sub_match(q in odd_q(), a in any::<u64>(), b in any::<u64>()) {
        let m = Modulus::new(q).unwrap();
        let (a, b) = (a % q, b % q);
        prop_assert_eq!(m.add(a, b), ((a as u128 + b as u128) % q as u128) as u64);
        prop_assert_eq!(m.sub(a, b), ((a as u128 + q as u128 - b as u128) % q as u128) as u64);
    }

    #[test]
    fn shoup_matches_u128(q in odd_q(), a in any::<u64>(), w in any::<u64>()) {
        let m = Modulus::new(q).unwrap();
        let (a, w) = (a % q, w % q);
        let s = m.shoup(w);
        let want = (a as u128 * w as u128 % q as u128) as u64;
        prop_assert_eq!(m.mul_shoup(a, &s), want);
        let lazy = m.mul_shoup_lazy(a, &s);
        prop_assert!(lazy < 2 * q && lazy % q == want);
    }

    #[test]
    fn inverse_roundtrip(q in odd_q(), a in any::<u64>()) {
        let m = Modulus::new(q).unwrap();
        let a = a % q;
        match m.inv(a) {
            Some(x) => { prop_assert!(x < q); prop_assert_eq!(m.mul(a, x), 1); }
            None => {
                // no inverse => gcd(a, q) != 1
                let (mut x, mut y) = (a, q);
                while y != 0 { (x, y) = (y, x % y); }
                prop_assert!(x != 1);
            }
        }
    }

    #[test]
    fn pow_matches_naive(q in odd_q(), a in any::<u64>(), e in 0u64..64) {
        let m = Modulus::new(q).unwrap();
        let a = a % q;
        let mut want = 1 % q;
        for _ in 0..e { want = (want as u128 * a as u128 % q as u128) as u64; }
        prop_assert_eq!(m.pow(a, e), want);
    }
}

#[test]
fn rejects_out_of_range() {
    assert!(matches!(
        Modulus::new(1 << 61),
        Err(HekitError::InvalidParams(_))
    ));
    assert!(matches!(
        Modulus::new(1 << 60),
        Err(HekitError::InvalidParams(_))
    ));
    assert!(matches!(
        Modulus::new(u64::MAX),
        Err(HekitError::InvalidParams(_))
    ));
    assert!(matches!(Modulus::new(0), Err(HekitError::InvalidParams(_))));
    assert!(matches!(Modulus::new(1), Err(HekitError::InvalidParams(_))));
    assert!(Modulus::new(2).is_ok());
    assert!(Modulus::new((1 << 60) - 1).is_ok());
}
