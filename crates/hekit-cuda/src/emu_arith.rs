//! Plain-Rust replay of `kernels/pointwise.cu` and `kernels/basis.cu`, one simulated CUDA
//! thread at a time (tests only), and the replay implementation of the multi-step algorithms
//! of [`crate::basis`] (the same launch sequence as the device, with the NTT replay of
//! [`crate::emu`]).
//!
//! Function names, variable names and the `[P*]` / `[B*]` tags match the `.cu` files. The
//! launch shapes are [`ElemLaunch`], as on the device. The scatter kernel
//! (`automorphism_coeff`) asserts that every output element is written exactly once.
//!
//! The tests run the shared cases of `tests/cases/mod.rs` (the same checks, inputs and CPU
//! references as the GPU tests in `tests/gpu_arith.rs`) through [`EmuExec`].

use hekit_ckks::{CkksParams, QpPoly};
use hekit_core::RnsPoly;

use crate::basis::{
    Backend, ConvTable, ModUpPlan, PrimeTable, check_galois_elt, divide_round_tail, mod_down_keep,
    mod_up_digit, rescale_keep, target_stride,
};
use crate::plan::ElemLaunch;
use crate::tables::{HostNttTables, NCONST};

/// `__umul64hi(a, b)`.
fn umul64hi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// `__brev(x)`.
fn brev(x: u32) -> u32 {
    x.reverse_bits()
}

/// [P1] / [B1]
pub(crate) fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    debug_assert!(a < q && b < q);
    let s = a + b;
    if s >= q { s - q } else { s }
}

/// [P2] / [B2]
fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    debug_assert!(a < q && b < q);
    if a >= b { a - b } else { a + q - b }
}

/// [P3a]
fn barrett_est(x0: u64, x1: u64, r0: u64, r1: u64) -> u64 {
    let t = umul64hi(x0, r0);
    let m1 = x0.wrapping_mul(r1);
    let m2 = x1.wrapping_mul(r0);
    let mut s = t.wrapping_add(m1);
    let mut carry = u64::from(s < t);
    s = s.wrapping_add(m2);
    carry += u64::from(s < m2);
    x1.wrapping_mul(r1)
        .wrapping_add(umul64hi(x0, r1))
        .wrapping_add(umul64hi(x1, r0))
        .wrapping_add(carry)
}

/// [P3]
pub(crate) fn mul_mod(a: u64, b: u64, q: u64, r0: u64, r1: u64) -> u64 {
    debug_assert!(a < q && b < q);
    let x0 = a.wrapping_mul(b);
    let x1 = umul64hi(a, b);
    let est = barrett_est(x0, x1, r0, r1);
    let mut r = x0.wrapping_sub(est.wrapping_mul(q));
    debug_assert!(r < 2 * q, "Barrett estimate off by more than 1");
    if r >= q {
        r -= q;
    }
    r
}

/// [P4] / [B3]
fn mul_shoup(a: u64, w: u64, ws: u64, q: u64) -> u64 {
    let hi = umul64hi(a, ws);
    let mut r = a.wrapping_mul(w).wrapping_sub(hi.wrapping_mul(q));
    if r >= q {
        r -= q;
    }
    r
}

/// Runs `body(blockIdx.y, c)` for every thread of an element-wise launch with `c < N`
/// ([P5] / [P12] / [P15] / [B4] / [B11]).
pub(crate) fn each_thread(l: &ElemLaunch, log_n: u32, mut body: impl FnMut(u32, u32)) {
    for by in 0..l.grid.1 {
        for bx in 0..l.grid.0 {
            for tx in 0..l.block.0 {
                let c = bx * l.block.0 + tx;
                if c >= (1u32 << log_n) {
                    continue;
                }
                body(by, c);
            }
        }
    }
}

/// `i` of [P6] and the prime's constants.
pub(crate) fn limb<'a>(
    by: u32,
    c: u32,
    log_n: u32,
    consts: &'a [u64],
    limb_prime: &[u32],
) -> (usize, &'a [u64]) {
    let i = ((by as usize) << log_n) + c as usize;
    let p = limb_prime[by as usize] as usize;
    (i, &consts[p * NCONST..(p + 1) * NCONST])
}

/// Element-wise binary kernels (`pw_add`, `pw_sub`, `pw_mul`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Binary {
    Add,
    Sub,
    Mul,
}

/// Arguments common to the element-wise kernels.
pub(crate) struct Pw<'a> {
    launch: ElemLaunch,
    consts: &'a [u64],
    limb_prime: &'a [u32],
    log_n: u32,
}

pub(crate) fn pw_binary(k: &Pw<'_>, op: Binary, out: &mut [u64], a: &[u64], b: &[u64]) {
    each_thread(&k.launch, k.log_n, |by, c| {
        let (i, cp) = limb(by, c, k.log_n, k.consts, k.limb_prime);
        out[i] = match op {
            Binary::Add => add_mod(a[i], b[i], cp[0]), // [P7]
            Binary::Sub => sub_mod(a[i], b[i], cp[0]), // [P8]
            Binary::Mul => mul_mod(a[i], b[i], cp[0], cp[5], cp[6]), // [P9]
        };
    });
}

/// `pw_neg`
fn pw_neg(k: &Pw<'_>, out: &mut [u64], a: &[u64]) {
    each_thread(&k.launch, k.log_n, |by, c| {
        let (i, cp) = limb(by, c, k.log_n, k.consts, k.limb_prime);
        out[i] = sub_mod(0, a[i], cp[0]); // [P10]
    });
}

/// `pw_mul_shoup`
fn pw_mul_shoup(k: &Pw<'_>, out: &mut [u64], a: &[u64], w: &[u64]) {
    each_thread(&k.launch, k.log_n, |by, c| {
        let (i, cp) = limb(by, c, k.log_n, k.consts, k.limb_prime);
        let l = by as usize;
        out[i] = mul_shoup(a[i], w[2 * l], w[2 * l + 1], cp[0]); // [P11]
    });
}

/// `automorphism_ntt`
pub(crate) fn automorphism_ntt(l: &ElemLaunch, out: &mut [u64], a: &[u64], log_n: u32, g: u64) {
    let n = 1u32 << log_n;
    each_thread(l, log_n, |by, c| {
        let base = (by as usize) << log_n;
        let reversed = brev(n + c) >> (31 - log_n); // [P13]
        let raw = ((g.wrapping_mul(reversed as u64) >> 1) & (n as u64 - 1)) as u32;
        let src = brev(raw) >> (32 - log_n); // [P14]
        out[base + c as usize] = a[base + src as usize];
    });
}

/// `automorphism_coeff` (asserts every output element is written exactly once).
pub(crate) fn automorphism_coeff(k: &Pw<'_>, out: &mut [u64], a: &[u64], g: u64) {
    let log_n = k.log_n;
    let n = 1u64 << log_n;
    let mut touched = vec![0u8; out.len()];
    each_thread(&k.launch, log_n, |by, c| {
        let (_, cp) = limb(by, c, log_n, k.consts, k.limb_prime);
        let base = (by as usize) << log_n;
        let e = (c as u64).wrapping_mul(g) & ((2u64 << log_n) - 1); // [P16]
        let v = a[base + c as usize];
        let (idx, val) = if e < n {
            (base + e as usize, v) // [P17]
        } else {
            (base + (e - n) as usize, sub_mod(0, v, cp[0]))
        };
        touched[idx] += 1;
        out[idx] = val;
    });
    if let Some(p) = touched.iter().position(|&t| t != 1) {
        panic!(
            "automorphism_coeff: element {p} written {} times",
            touched[p]
        );
    }
}

/// `basis_convert` (the kernel's argument list).
#[allow(clippy::too_many_arguments)]
fn basis_convert(
    l: &ElemLaunch,
    out: &mut [u64],
    src: &[u64],
    tab: &[u64],
    k: u32,
    n_targets: u32,
    log_n: u32,
    use_f64: bool,
    fma: bool,
) {
    assert!(k as usize <= crate::basis::MAX_SRC);
    each_thread(l, log_n, |_, c| {
        let k = k as usize;
        let mut y = [0u64; crate::basis::MAX_SRC];
        let mut f = 0.0f64;
        for j in 0..k {
            // [B5]
            let q = tab[j];
            let s = add_mod(src[(j << log_n) + c as usize], q >> 1, q);
            y[j] = mul_shoup(s, tab[k + j], tab[2 * k + j], q);
            if use_f64 {
                // [B6]: __ull2double_rn, __dmul_rn, __dadd_rn (Rust's `as f64`, `*`, `+`)
                let w = f64::from_bits(tab[3 * k + j]);
                if fma {
                    // Test-only mutant: the FMA a contracting compiler would emit.
                    f = (y[j] as f64).mul_add(w, f);
                } else {
                    f += y[j] as f64 * w;
                }
            }
        }
        // [B7]: __double2ull_rz (truncation); the CPU's `f.floor() as u64` is the same for f >= 0.
        debug_assert!(f >= 0.0);
        let e = if use_f64 { f as u64 } else { 0 };
        let stride = target_stride(k);
        for t in 0..n_targets as usize {
            // [B8]
            let tt = &tab[4 * k + t * stride..4 * k + (t + 1) * stride];
            let r = tt[0];
            let mut o = 0;
            for j in 0..k {
                o = add_mod(o, mul_shoup(y[j], tt[6 + 2 * j], tt[7 + 2 * j], r), r);
            }
            if use_f64 {
                o = sub_mod(o, mul_shoup(e, tt[1], tt[2], r), r); // [B9]
            }
            o = sub_mod(o, tt[3], r);
            out[(t << log_n) + c as usize] = o; // [B10]
        }
    });
}

/// `basis_div_finish`
fn basis_div_finish(l: &ElemLaunch, t_out: &mut [u64], a: &[u64], tab: &[u64], k: u32, log_n: u32) {
    let k = k as usize;
    each_thread(l, log_n, |by, c| {
        let base = 4 * k + by as usize * target_stride(k);
        let tt = &tab[base..base + target_stride(k)];
        let r = tt[0];
        let i = ((by as usize) << log_n) + c as usize;
        t_out[i] = mul_shoup(sub_mod(a[i], t_out[i], r), tt[4], tt[5], r); // [B12]
    });
}

/// The replay backend: buffers are `Vec<u64>`, kernels are the replays above and the NTT
/// replay of [`crate::emu`].
pub struct EmuBackend {
    pub(crate) primes: PrimeTable,
    pub(crate) tables: HostNttTables,
    pub(crate) log_n: u32,
    /// Replace the f64 product-and-sum of `basis_convert` by an FMA (a deliberately wrong
    /// mutant, to show that the near-tie inputs detect contraction). Always `false` except
    /// in `fma_contraction_is_caught_by_near_ties`.
    fma: bool,
}

impl EmuBackend {
    /// The backend for `params` (tables in device order).
    pub fn new(params: &CkksParams) -> EmuBackend {
        EmuBackend {
            primes: PrimeTable::from_params(params),
            tables: HostNttTables::from_params(params).unwrap(),
            log_n: params.n().trailing_zeros(),
            fma: false,
        }
    }

    pub(crate) fn n(&self) -> usize {
        1 << self.log_n
    }

    pub(crate) fn pw<'a>(&'a self, limb_prime: &'a [u32]) -> Pw<'a> {
        Pw {
            launch: ElemLaunch::new(self.n(), limb_prime.len()).unwrap(),
            consts: &self.tables.consts,
            limb_prime,
            log_n: self.log_n,
        }
    }
}

impl Backend for EmuBackend {
    type Buf = Vec<u64>;
    type Table = ConvTable;

    fn alloc(&self, limbs: usize) -> hekit_core::Result<Vec<u64>> {
        // Poison: a read before a write shows up as a mismatch or a range assertion.
        Ok(vec![u64::MAX; limbs * self.n()])
    }

    fn copy_limbs(
        &self,
        src: &Vec<u64>,
        src_at: usize,
        dst: &mut Vec<u64>,
        dst_at: usize,
        count: usize,
    ) -> hekit_core::Result<()> {
        let n = self.n();
        assert!(count >= 1);
        dst[dst_at * n..(dst_at + count) * n]
            .copy_from_slice(&src[src_at * n..(src_at + count) * n]);
        Ok(())
    }

    fn ntt(&self, buf: &mut Vec<u64>, primes: &[u32], inverse: bool) -> hekit_core::Result<()> {
        if inverse {
            crate::emu::inverse(buf, primes, &self.tables);
        } else {
            crate::emu::forward(buf, primes, &self.tables);
        }
        Ok(())
    }

    fn conv_table(&self, src: &[u32], targets: &[u32]) -> hekit_core::Result<ConvTable> {
        self.primes.conv_table(src, targets)
    }

    fn convert(&self, out: &mut Vec<u64>, src: &Vec<u64>, t: &ConvTable) -> hekit_core::Result<()> {
        let l = ElemLaunch::new(self.n(), 1)?;
        let (k, nt) = (t.k() as u32, t.targets.len() as u32);
        assert_eq!(src.len(), t.k() * self.n());
        assert_eq!(out.len(), t.targets.len() * self.n());
        basis_convert(
            &l, out, src, &t.data, k, nt, self.log_n, t.use_f64, self.fma,
        );
        Ok(())
    }

    fn div_finish(
        &self,
        t_out: &mut Vec<u64>,
        a: &Vec<u64>,
        t: &ConvTable,
    ) -> hekit_core::Result<()> {
        let l = ElemLaunch::new(self.n(), t.targets.len())?;
        basis_div_finish(&l, t_out, a, &t.data, t.k() as u32, self.log_n);
        Ok(())
    }
}

/// [`cases::Exec`] on the replay: host polynomial -> flat limbs and prime indices (as
/// `CudaDevice::upload`), replayed kernels in the device's launch sequence, back to host
/// polynomials (as `download` / `download_qp`).
pub struct EmuExec {
    params: CkksParams,
    b: EmuBackend,
}

impl EmuExec {
    /// Replay of a device for `params`.
    pub fn new(params: &CkksParams) -> EmuExec {
        EmuExec {
            params: params.clone(),
            b: EmuBackend::new(params),
        }
    }

    /// The FMA-contracted mutant (see `EmuBackend::fma`).
    fn with_fma(params: &CkksParams) -> EmuExec {
        let mut e = EmuExec::new(params);
        e.b.fma = true;
        e
    }

    fn up(&self, p: &RnsPoly) -> (Vec<u32>, Vec<u64>) {
        let primes = self.b.primes.primes_of(p).unwrap();
        (primes, p.limbs().concat())
    }

    fn down(&self, primes: &[u32], flat: &[u64], ntt: bool) -> RnsPoly {
        self.b.primes.host_poly(primes, flat, ntt).unwrap()
    }

    fn binary(&self, op: Binary, a: &RnsPoly, b: &RnsPoly) -> RnsPoly {
        let ((pa, fa), (pb, fb)) = (self.up(a), self.up(b));
        assert_eq!(pa, pb);
        let mut out = self.b.alloc(pa.len()).unwrap();
        pw_binary(&self.b.pw(&pa), op, &mut out, &fa, &fb);
        self.down(&pa, &out, a.is_ntt_form())
    }
}

use crate::cases::Exec;

impl Exec for EmuExec {
    fn add(&self, a: &RnsPoly, b: &RnsPoly) -> RnsPoly {
        self.binary(Binary::Add, a, b)
    }

    fn sub(&self, a: &RnsPoly, b: &RnsPoly) -> RnsPoly {
        self.binary(Binary::Sub, a, b)
    }

    fn mul_pointwise(&self, a: &RnsPoly, b: &RnsPoly) -> RnsPoly {
        self.binary(Binary::Mul, a, b)
    }

    fn neg(&self, a: &RnsPoly) -> RnsPoly {
        let (p, f) = self.up(a);
        let mut out = self.b.alloc(p.len()).unwrap();
        pw_neg(&self.b.pw(&p), &mut out, &f);
        self.down(&p, &out, a.is_ntt_form())
    }

    fn mul_scalar(&self, a: &RnsPoly, scalar: u64) -> RnsPoly {
        let (p, f) = self.up(a);
        let w = self.b.primes.scalar_shoup(&p, scalar);
        let mut out = self.b.alloc(p.len()).unwrap();
        pw_mul_shoup(&self.b.pw(&p), &mut out, &f, &w);
        self.down(&p, &out, a.is_ntt_form())
    }

    fn add_qp(&self, a: &QpPoly, b: &QpPoly) -> QpPoly {
        let t = &self.b.primes;
        let ((pa, fa), (pb, fb)) = (t.qp_of(a).unwrap(), t.qp_of(b).unwrap());
        assert_eq!(pa, pb);
        let mut out = self.b.alloc(pa.len()).unwrap();
        pw_binary(&self.b.pw(&pa), Binary::Add, &mut out, &fa, &fb);
        t.host_qp(&pa, &out, a.q().is_ntt_form()).unwrap()
    }

    fn ntt_qp(&self, x: &QpPoly) -> QpPoly {
        let t = &self.b.primes;
        let (p, mut f) = t.qp_of(x).unwrap();
        let inverse = x.q().is_ntt_form();
        self.b.ntt(&mut f, &p, inverse).unwrap();
        t.host_qp(&p, &f, !inverse).unwrap()
    }

    fn automorphism(&self, a: &RnsPoly, g: u64) -> RnsPoly {
        check_galois_elt(self.b.n(), g).unwrap();
        let (p, f) = self.up(a);
        let mut out = self.b.alloc(p.len()).unwrap();
        if a.is_ntt_form() {
            let l = ElemLaunch::new(self.b.n(), p.len()).unwrap();
            automorphism_ntt(&l, &mut out, &f, self.b.log_n, g);
        } else {
            automorphism_coeff(&self.b.pw(&p), &mut out, &f, g);
        }
        self.down(&p, &out, a.is_ntt_form())
    }

    fn rescale(&self, a: &RnsPoly) -> RnsPoly {
        let (p, f) = self.up(a);
        let keep = rescale_keep(&self.b.primes, &p).unwrap();
        let out = divide_round_tail(&self.b, &f, &p, keep, a.is_ntt_form()).unwrap();
        self.down(&p[..keep], &out, a.is_ntt_form())
    }

    fn mod_down(&self, x: &QpPoly) -> RnsPoly {
        let (p, f) = self.b.primes.qp_of(x).unwrap();
        let keep = mod_down_keep(&self.b.primes, &p).unwrap();
        let ntt = x.q().is_ntt_form();
        let out = divide_round_tail(&self.b, &f, &p, keep, ntt).unwrap();
        self.down(&p[..keep], &out, ntt)
    }

    fn mod_up(&self, x: &RnsPoly, digit: usize) -> QpPoly {
        assert!(x.is_ntt_form());
        let (p, f) = self.up(x);
        let plan = ModUpPlan::new(&self.params, &self.b.primes, &p, digit).unwrap();
        let out = mod_up_digit(&self.b, &f, &plan).unwrap();
        self.b.primes.host_qp(&plan.out_primes, &out, true).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases;
    use hekit_core::Modulus;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};

    /// The Barrett product equals `Modulus::mul` on extreme and random operands, for prime
    /// sizes from 2 bits to 60 bits (the `debug_assert` checks the off-by-one bound).
    #[test]
    fn barrett_product_matches_modulus_mul() {
        let mut rng = ChaCha20Rng::seed_from_u64(45);
        for q in [
            3u64,
            65537,
            (1 << 40) - 87,
            1_152_921_504_606_846_883,
            (1 << 60) - 93,
            (1 << 59) + 1,
        ] {
            let m = Modulus::new(q).unwrap();
            let r = crate::tables::barrett_ratio(q);
            let (r0, r1) = (r as u64, (r >> 64) as u64);
            let mut pairs = vec![(q - 1, q - 1), (0, q - 1), (1, q - 1), (q / 2, q - 2)];
            pairs.extend((0..10_000).map(|_| (rng.next_u64() % q, rng.next_u64() % q)));
            for (a, b) in pairs {
                assert_eq!(mul_mod(a, b, q, r0, r1), m.mul(a, b), "q={q} a={a} b={b}");
            }
        }
    }

    /// `barrett_est` is the exact high half of the 256-bit product whenever that fits in 64
    /// bits (`hekit_core`'s `mulhi_u128`), including carries out of both middle words.
    #[test]
    fn barrett_estimate_is_exact() {
        // Exact floor(x * r / 2^128) from 32-bit limbs.
        fn mulhi_ref(x: u128, r: u128) -> u128 {
            let xs = [
                x as u32,
                (x >> 32) as u32,
                (x >> 64) as u32,
                (x >> 96) as u32,
            ];
            let rs = [
                r as u32,
                (r >> 32) as u32,
                (r >> 64) as u32,
                (r >> 96) as u32,
            ];
            let mut acc = [0u64; 9];
            for (i, &a) in xs.iter().enumerate() {
                for (j, &b) in rs.iter().enumerate() {
                    let p = a as u64 * b as u64;
                    acc[i + j] += p & 0xffff_ffff;
                    acc[i + j + 1] += p >> 32;
                }
            }
            for i in 0..8 {
                acc[i + 1] += acc[i] >> 32;
                acc[i] &= 0xffff_ffff;
            }
            (acc[4] as u128)
                | ((acc[5] as u128) << 32)
                | ((acc[6] as u128) << 64)
                | ((acc[7] as u128) << 96)
        }
        let mut rng = ChaCha20Rng::seed_from_u64(451);
        let mut carries = 0;
        for i in 0..200_000 {
            let q = (rng.next_u64() >> 4) | 3; // up to 2^60
            let r = crate::tables::barrett_ratio(q);
            let x = if i % 2 == 0 {
                // a * b with a, b < q (the kernel's range).
                (rng.next_u64() % q) as u128 * (rng.next_u64() % q) as u128
            } else {
                // Words chosen to make the middle-column carries likely.
                (rng.next_u64() as u128 | 0xffff_ffff_0000_0000)
                    | (((rng.next_u64() >> 8) as u128) << 64)
            };
            let want = mulhi_ref(x, r);
            if want >> 64 != 0 {
                continue;
            }
            let (x0, x1, r0, r1) = (x as u64, (x >> 64) as u64, r as u64, (r >> 64) as u64);
            let t = umul64hi(x0, r0);
            let s = t.wrapping_add(x0.wrapping_mul(r1));
            if s < t || s.wrapping_add(x1.wrapping_mul(r0)) < s {
                carries += 1;
            }
            assert_eq!(barrett_est(x0, x1, r0, r1) as u128, want, "x={x} q={q}");
        }
        assert!(carries > 1000, "only {carries} carries exercised");
    }

    /// Review M4: the replays (`src/emu.rs`, this file) are hand transcriptions of the `.cu`
    /// files, so a kernel edit alone would leave them green while the GPU diverges. The
    /// sources are pinned by hash: after editing a kernel, update its replay, have both
    /// re-reviewed, then update the hash here.
    #[test]
    fn kernel_sources_match_the_replayed_versions() {
        fn fnv1a(s: &str) -> u64 {
            s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
                (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
            })
        }
        for (file, src, pinned) in [
            (
                "ntt.cu (replay: src/emu.rs)",
                crate::NTT_KERNEL_SRC,
                0xcc94cf8a6cfb5408,
            ),
            (
                "pointwise.cu (replay: src/emu_arith.rs)",
                crate::POINTWISE_KERNEL_SRC,
                0xe2dae47bea8894a2,
            ),
            (
                "basis.cu (replay: src/emu_arith.rs)",
                crate::BASIS_KERNEL_SRC,
                0x4d52e9211d6eb679,
            ),
        ] {
            let h = fnv1a(src);
            assert_eq!(
                h, pinned,
                "kernels/{file} changed (FNV-1a {h:#018x}): update and re-review its replay, \
                 then pin the new hash"
            );
        }
    }

    #[test]
    fn replay_pointwise_seal_2020() {
        cases::check_pointwise(
            &EmuExec::new(&CkksParams::seal_2020()),
            &CkksParams::seal_2020(),
        );
    }

    #[test]
    fn replay_pointwise_n16_train() {
        let p = CkksParams::n16_train();
        cases::check_pointwise(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_qp_layout_seal_2020() {
        let p = CkksParams::seal_2020();
        cases::check_qp_layout(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_qp_layout_n16_train() {
        let p = CkksParams::n16_train();
        cases::check_qp_layout(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_automorphism_seal_2020() {
        let p = CkksParams::seal_2020();
        cases::check_automorphism(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_automorphism_n16_train() {
        let p = CkksParams::n16_train();
        cases::check_automorphism(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_rescale_seal_2020() {
        let p = CkksParams::seal_2020();
        cases::check_rescale(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_rescale_n16_train() {
        let p = CkksParams::n16_train();
        cases::check_rescale(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_mod_down_seal_2020() {
        let p = CkksParams::seal_2020();
        cases::check_mod_down(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_mod_down_n16_train() {
        let p = CkksParams::n16_train();
        cases::check_mod_down(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_mod_up_seal_2020() {
        let p = CkksParams::seal_2020();
        cases::check_mod_up(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_mod_down_near_ties_seal_2020() {
        let p = CkksParams::seal_2020();
        cases::check_mod_down_near_ties(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_mod_down_near_ties_n16_train() {
        let p = CkksParams::n16_train();
        cases::check_mod_down_near_ties(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_mod_up_near_ties_seal_2020() {
        let p = CkksParams::seal_2020();
        cases::check_mod_up_near_ties(&EmuExec::new(&p), &p);
    }

    #[test]
    fn replay_mod_up_near_ties_n16_train() {
        let p = CkksParams::n16_train();
        cases::check_mod_up_near_ties(&EmuExec::new(&p), &p);
    }

    /// Review I1: an FMA-contracted f64 estimate (what NVRTC's default `--fmad=true` could emit
    /// for a plain `f += y * w`) passes the random-input checks but fails the near-tie
    /// checks, for `mod_down` (k = 4) and `mod_up` (alpha = 4) on n16_train. So the GPU
    /// near-tie tests observe contraction rather than rely on the `_rn` intrinsics' docs.
    #[test]
    fn fma_contraction_is_caught_by_near_ties() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let p = CkksParams::n16_train();
        let fma = EmuExec::with_fma(&p);
        cases::check_mod_down(&fma, &p); // random inputs: blind to contraction
        let fails = |f: &dyn Fn()| catch_unwind(AssertUnwindSafe(f)).is_err();
        assert!(
            fails(&|| cases::check_mod_down_near_ties(&fma, &p)),
            "mod_down near ties did not detect FMA contraction"
        );
        assert!(
            fails(&|| cases::check_mod_up_near_ties(&fma, &p)),
            "mod_up near ties did not detect FMA contraction"
        );
    }

    #[test]
    fn replay_mod_up_n16_train() {
        let p = CkksParams::n16_train();
        cases::check_mod_up(&EmuExec::new(&p), &p);
    }
}
