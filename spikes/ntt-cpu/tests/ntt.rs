use ntt_cpu::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

struct Tables {
    q: u64,
    roots: Vec<u64>,
    roots_shoup: Vec<u64>,
    inv_roots: Vec<u64>,
    inv_roots_shoup: Vec<u64>,
    n_inv: u64,
    n_inv_shoup: u64,
}

fn tables(n: usize) -> Tables {
    let q = gen_ntt_prime(n, 60);
    let psi = primitive_2n_root(n, q);
    let roots = build_roots(n, q, psi);
    let inv_roots = build_inv_roots(n, q, psi);
    let n_inv = pow_mod(n as u64, q - 2, q);
    Tables {
        q,
        roots_shoup: shoup_table(&roots, q),
        inv_roots_shoup: shoup_table(&inv_roots, q),
        roots,
        inv_roots,
        n_inv,
        n_inv_shoup: shoup_table(&[n_inv], q)[0],
    }
}

fn rand_poly(rng: &mut ChaCha20Rng, n: usize, q: u64) -> Vec<u64> {
    (0..n).map(|_| rng.gen_range(0..q)).collect()
}

fn naive_negacyclic(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    let mut c = vec![0u64; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            let p = mul_mod(ai, bj, q);
            let k = i + j;
            if k < n {
                c[k] = (c[k] + p) % q;
            } else {
                c[k - n] = (c[k - n] + q - p) % q;
            }
        }
    }
    c
}

#[test]
fn prime_properties() {
    for logn in [10u32, 16] {
        let n = 1usize << logn;
        let q = gen_ntt_prime(n, 60);
        assert!(q < 1 << 60 && q > 1 << 59);
        assert_eq!(q % (2 * n as u64), 1);
        let psi = primitive_2n_root(n, q);
        assert_eq!(pow_mod(psi, n as u64, q), q - 1);
    }
}

#[test]
fn product_matches_naive_n1024() {
    let n = 1 << 10;
    let t = tables(n);
    let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEE);
    for _ in 0..3 {
        let a = rand_poly(&mut rng, n, t.q);
        let b = rand_poly(&mut rng, n, t.q);
        let expect = naive_negacyclic(&a, &b, t.q);
        let (mut fa, mut fb) = (a.clone(), b.clone());
        ntt_forward(&mut fa, t.q, &t.roots, &t.roots_shoup);
        ntt_forward(&mut fb, t.q, &t.roots, &t.roots_shoup);
        let mut fc: Vec<u64> = fa
            .iter()
            .zip(&fb)
            .map(|(&x, &y)| mul_mod(x, y, t.q))
            .collect();
        ntt_inverse(
            &mut fc,
            t.q,
            &t.inv_roots,
            &t.inv_roots_shoup,
            t.n_inv,
            t.n_inv_shoup,
        );
        assert_eq!(fc, expect);
    }
}

#[test]
fn roundtrip_n65536() {
    let n = 1 << 16;
    let t = tables(n);
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let a = rand_poly(&mut rng, n, t.q);
    let mut x = a.clone();
    ntt_forward(&mut x, t.q, &t.roots, &t.roots_shoup);
    assert!(
        x.iter().all(|&v| v < t.q),
        "forward output must be fully reduced"
    );
    ntt_inverse(
        &mut x,
        t.q,
        &t.inv_roots,
        &t.inv_roots_shoup,
        t.n_inv,
        t.n_inv_shoup,
    );
    assert_eq!(x, a);
}
