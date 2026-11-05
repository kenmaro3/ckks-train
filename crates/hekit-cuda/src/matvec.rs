//! Backend-neutral BSGS matrix-vector products (no CUDA needed): the orchestration of
//! `hekit_ckks::Evaluator::matvec_plain` / `matvec_enc` written once against
//! [`hekit_ckks::HeOps`], with every [`EvalConfig`] flag honoured exactly as the CPU does
//! (Task 43 review M3: one shared reference for the flag semantics and counts).
//!
//! [`bsgs`] runs the same operations, in the same order, as the CPU evaluator's private
//! `bsgs` for one term with no skipped entry: the baby rotations of `x` through
//! [`HeOps::hoisted_rotate`] (which honours `hoist`), the products, one `finish` and one
//! giant rotation per block with `bsgs` on, or one `finish` and one rotation per product of a
//! nonzero giant shift with `bsgs` off, and the block sums in their sequential order. Every
//! primitive of a bit-exact backend gives the CPU's residues, and modular sums do not depend
//! on their order, so the product is **bit-identical** to the CPU's, with identical counts.
//! The tests run [`bsgs`] on the CPU evaluator itself and compare it with
//! `Evaluator::matvec_plain` / `matvec_enc`, bit for bit and count for count, for every flag
//! combination; the CUDA evaluator (`CudaEvaluator`, `cuda` feature) calls the same function.
//!
//! [`check_shape`] holds the shape checks of the CPU products (entry levels and scales, the
//! level of `x`, the stride); the backend adds its own operand and key checks in the CPU's
//! order. [`matvec_enc_workspace_limbs`] estimates the device working set of one product.

use hekit_ckks::linalg::DiagonalEntry;
use hekit_ckks::{CkksParams, Diagonals, EvalConfig, HeOps};
use hekit_core::{HekitError, Result};

/// Runs the composite operation `f` on a fork of `ev` (same keys and flags, own counter)
/// and adds the fork's counts to `ev` **only if `f` succeeds**: a composite operation that
/// fails part-way (e.g. a device allocation failure in its 40th product) counts nothing, as
/// the counter contract requires (Task 47 review I1). The device backend runs `mul`,
/// `hoisted_rotate` without hoisting and the matrix-vector products through it.
pub fn counted<E: HeOps, R>(ev: &E, f: impl FnOnce(&E) -> Result<R>) -> Result<R> {
    let local = ev.fork();
    let r = f(&local)?;
    ev.add_counts(&local.counter());
    Ok(r)
}

/// `relinearize(mul_no_relin(a, b))`: `Evaluator::mul` after its checks.
pub fn mul_relin<E: HeOps>(ev: &E, a: &E::Ct, b: &E::Ct) -> Result<E::Ct> {
    ev.relinearize(&ev.mul_no_relin(a, b)?)
}

/// One `rotate` per step: `Evaluator::hoisted_rotate` with `hoist` off, after its key checks.
pub fn rotate_each<E: HeOps>(ev: &E, ct: &E::Ct, steps: &[i32]) -> Result<Vec<E::Ct>> {
    steps.iter().map(|&s| ev.rotate(ct, s)).collect()
}

/// `W x` by BSGS for the pre-rotated diagonals `diag` (any offset, no skipped entry):
/// `sum_{k1} rot(finish(sum_{k2} mul(entries[k1 baby + k2], rot(x, k2 s))), (offset + baby
/// k1) s)` with `share_giant` (the `bsgs` flag), else every product of a nonzero giant shift
/// finished and rotated on its own (the CPU's naive form; see `hekit_ckks::ops`, "What the
/// RQ3 table must state").
///
/// The caller has checked everything ([`check_shape`], operands, keys), so no step fails
/// part-way except on a backend error. Counts are those of the primitives `ev` runs.
pub fn bsgs<E: HeOps, T>(
    ev: &E,
    diag: &Diagonals<T>,
    x: &E::Ct,
    share_giant: bool,
    mul: impl Fn(&T, &E::Ct) -> Result<E::Ct>,
    finish: impl Fn(E::Ct) -> Result<E::Ct>,
) -> Result<E::Ct> {
    let (baby, giant, stride, offset) = (diag.baby(), diag.giant(), diag.stride(), diag.offset());
    let entries = diag.entries();
    // Step 0 is a copy (no key switch, not counted), as on the CPU.
    let steps: Vec<i32> = (0..baby).map(|k2| (k2 * stride) as i32).collect();
    let xs = ev.hoisted_rotate(x, &steps)?;
    let mut acc: Option<E::Ct> = None;
    for k1 in 0..giant {
        let shift = (offset + (baby * k1) as i64) * stride as i64;
        let row = &entries[k1 * baby..(k1 + 1) * baby];
        let block = if !share_giant && shift != 0 {
            // Naive: one finish and one giant rotation per product.
            let mut sum: Option<E::Ct> = None;
            for (w, xk) in row.iter().zip(&xs) {
                let t = ev.rotate(&finish(mul(w, xk)?)?, shift as i32)?;
                sum = Some(match sum {
                    None => t,
                    Some(s) => ev.add(&s, &t)?,
                });
            }
            sum
        } else {
            let mut inner: Option<E::Ct> = None;
            for (w, xk) in row.iter().zip(&xs) {
                let t = mul(w, xk)?;
                inner = Some(match inner {
                    None => t,
                    Some(s) => ev.add(&s, &t)?,
                });
            }
            match inner {
                None => None,
                Some(inner) => {
                    let b = finish(inner)?;
                    Some(if shift != 0 {
                        ev.rotate(&b, shift as i32)?
                    } else {
                        b
                    })
                }
            }
        };
        if let Some(b) = block {
            acc = Some(match acc {
                None => b,
                Some(s) => ev.add(&s, &b)?,
            });
        }
    }
    acc.ok_or_else(|| HekitError::InvalidParams("BSGS product with no diagonal".into()))
}

/// `Evaluator::matvec_plain`'s product (after its checks): [`bsgs`] with `mul_plain(x_r, w)`
/// and the `bsgs` flag of `ev`.
pub fn matvec_plain_bsgs<E: HeOps>(ev: &E, diag: &Diagonals<E::Pt>, x: &E::Ct) -> Result<E::Ct> {
    bsgs(
        ev,
        diag,
        x,
        ev.config().bsgs,
        |w, xr| ev.mul_plain(xr, w),
        Ok,
    )
}

/// `Evaluator::matvec_enc`'s product (after its checks): [`bsgs`] with the `bsgs` flag of
/// `ev`, and with `lazy_relin` the degree-2 products `mul_no_relin(w, x_r)` summed and
/// relinearized once per `finish`, without it one `mul(w, x_r)` per product.
pub fn matvec_enc_bsgs<E: HeOps>(ev: &E, diag: &Diagonals<E::Ct>, x: &E::Ct) -> Result<E::Ct> {
    let share = ev.config().bsgs;
    if ev.config().lazy_relin {
        bsgs(
            ev,
            diag,
            x,
            share,
            |w, xr| ev.mul_no_relin(w, xr),
            |ct| ev.relinearize(&ct),
        )
    } else {
        bsgs(ev, diag, x, share, |w, xr| ev.mul(w, xr), Ok)
    }
}

/// The shape checks of `Evaluator::matvec_plain` / `matvec_enc`, in the CPU's order: the
/// entries share one level (`LevelMismatch`) and one scale (`ScaleMismatch`), that level is
/// `x_level` (`LevelMismatch`), and `d * stride` divides `slots` (`InvalidParams`).
pub fn check_shape<T: DiagonalEntry>(
    diag: &Diagonals<T>,
    x_level: usize,
    slots: usize,
) -> Result<()> {
    let entries = diag.entries();
    if let Some(first) = entries.first() {
        if entries.iter().any(|e| e.level() != first.level()) {
            return Err(HekitError::LevelMismatch);
        }
        if entries.iter().any(|e| e.scale() != first.scale()) {
            return Err(HekitError::ScaleMismatch);
        }
    }
    if entries.iter().any(|e| e.level() != x_level) {
        return Err(HekitError::LevelMismatch);
    }
    match diag.d().checked_mul(diag.stride()) {
        Some(block) if diag.stride() > 0 && slots % block == 0 => Ok(()),
        _ => Err(HekitError::InvalidParams(format!(
            "d * stride = {} * {} must divide the slot count {slots} (stride >= 1)",
            diag.d(),
            diag.stride()
        ))),
    }
}

/// An upper estimate of the device working set (in limbs of `N` words) of one
/// [`matvec_enc_bsgs`] at `level` for dimension `d`, beyond its inputs and keys (used to
/// pre-allocate the streams of a batch and by the tests' memory declarations):
///
/// * the `baby - 1` rotated copies of `x` (2 polynomials over `Q_l` each);
/// * a block: the running sum and a new degree-2 product (3 + 3), the tensor temporaries (2),
///   the block result and the accumulator (2 + 2): 12 polynomials over `Q_l`;
/// * one key switch: the lifted digits and one set of automorphism-rotated digits
///   (`2 dnum_l` over `Q_l P`), the two inner-product accumulators (2 over `Q_l P`) and the
///   `mod_down` / finishing temporaries (4 over `Q_l`).
///
/// The `bsgs` / `lazy_relin` / `hoist` flags change the order of the same kinds of
/// temporaries, not their peak beyond this bound.
pub fn matvec_enc_workspace_limbs(params: &CkksParams, level: usize, d: usize) -> usize {
    let (baby, _) = hekit_ckks::bsgs_split(d);
    let q = level + 1;
    let qp = q + params.p().len();
    let digits = hekit_ckks::keyswitch::digit_ranges(params, level).len();
    2 * (baby - 1) * q + 12 * q + (2 * digits + 2) * qp + 4 * q
}

/// The counts one `matvec_enc` of dimension `d` (offset 0) adds under `config`:
/// `(rotations, relins, ct_ct_tensor)` from [`EvalConfig::matvec_rotations`] /
/// [`EvalConfig::matvec_enc_relins`] (the conformance reference of Task 43 M3).
pub fn expected_matvec_enc_counts(config: &EvalConfig, d: usize) -> (u64, u64, u64) {
    (
        config.matvec_rotations(d),
        config.matvec_enc_relins(d),
        d as u64,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hekit_ckks::{
        Ciphertext, CkksParams, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter, Plaintext,
        bsgs_rotation_steps_with_offset,
    };
    use hekit_core::sampler::Rng;

    use super::*;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    fn rand_vals(r: &mut Rng, len: usize) -> Vec<f64> {
        (0..len)
            .map(|_| (r.next_u64() >> 11) as f64 * 2f64.powi(-52) - 1.0)
            .collect()
    }

    fn bits(c: &Ciphertext) -> Vec<Vec<Vec<u64>>> {
        c.polys().iter().map(|p| p.limbs().to_vec()).collect()
    }

    struct Fx {
        params: Arc<CkksParams>,
        enc: Encoder,
        encryptor: Encryptor,
        kg: KeyGenerator,
        rk: hekit_ckks::RelinKey,
    }

    fn fixture(params: CkksParams, seed: u8) -> Fx {
        let params = Arc::new(params);
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let pk = kg.public_key();
        let rk = kg.relin_key();
        Fx {
            enc: Encoder::new(Arc::clone(&params)),
            encryptor: Encryptor::new(Arc::clone(&params), pk),
            params,
            kg,
            rk,
        }
    }

    fn small() -> CkksParams {
        CkksParams::new(12, &[60, 40, 40, 40], &[60], 40, 4).unwrap()
    }

    /// The shared orchestration on the CPU evaluator against the CPU's own
    /// `matvec_plain` / `matvec_enc`: bit-identical results, scales and levels, and identical
    /// counts, for every flag combination, several `d`, strides and offsets; the matvec_enc
    /// counts also match the closed forms of `EvalConfig` (offset 0).
    #[test]
    fn shared_bsgs_matches_cpu_for_every_flag_combination() {
        let mut fx = fixture(small(), 41);
        let (slots, top, delta) = (fx.params.slots(), fx.params.max_level(), fx.params.scale());
        // (d, stride, offset)
        let shapes: [(usize, usize, i64); 9] = [
            (1, 1, 0),
            (2, 1, 0),
            (4, 2, 0),
            (8, 1, -3),
            (16, 1, 0),
            (16, 1, -15),
            (32, 4, -17),
            (64, 1, 0),
            (64, 2, -5),
        ];
        let mut steps: Vec<i32> = Vec::new();
        for &(d, s, o) in &shapes {
            for st in bsgs_rotation_steps_with_offset(d, s, o) {
                if !steps.contains(&st) {
                    steps.push(st);
                }
            }
        }
        let gks = fx.kg.galois_keys(&steps);
        let mut r = rng(42);
        for config in EvalConfig::all() {
            let ev = Evaluator::with_config(
                Arc::clone(&fx.params),
                Some(fx.rk.clone()),
                Some(gks.clone()),
                config,
            );
            for &(d, stride, offset) in &shapes {
                let x = fx.encryptor.encrypt(
                    &fx.enc
                        .encode(&rand_vals(&mut r, slots), delta, top)
                        .unwrap(),
                    &mut r,
                );
                let pts: Vec<Plaintext> = (0..d)
                    .map(|_| {
                        fx.enc
                            .encode(&rand_vals(&mut r, slots), delta, top)
                            .unwrap()
                    })
                    .collect();
                let cts: Vec<Ciphertext> = pts
                    .iter()
                    .map(|p| fx.encryptor.encrypt(p, &mut r))
                    .collect();
                let pd = Diagonals::from_prerotated_with_offset(d, stride, offset, pts).unwrap();
                let cd = Diagonals::from_prerotated_with_offset(d, stride, offset, cts).unwrap();
                check_shape(&pd, top, slots).unwrap();
                check_shape(&cd, top, slots).unwrap();
                let what = format!("{config:?} d {d} stride {stride} offset {offset}");

                ev.reset_counter();
                let want = ev.matvec_plain(&pd, &x).unwrap();
                let want_c = ev.counter();
                ev.reset_counter();
                let got = matvec_plain_bsgs(&ev, &pd, &x).unwrap();
                assert_eq!(ev.counter(), want_c, "plain counts {what}");
                assert_eq!(bits(&got), bits(&want), "plain values {what}");
                assert_eq!(
                    (got.level(), got.scale().to_bits()),
                    (want.level(), want.scale().to_bits())
                );

                ev.reset_counter();
                let want = ev.matvec_enc(&cd, &x).unwrap();
                let want_c = ev.counter();
                ev.reset_counter();
                let got = matvec_enc_bsgs(&ev, &cd, &x).unwrap();
                let c = ev.counter();
                assert_eq!(c, want_c, "enc counts {what}");
                assert_eq!(bits(&got), bits(&want), "enc values {what}");
                assert_eq!(
                    (got.len(), got.level(), got.scale().to_bits()),
                    (want.len(), want.level(), want.scale().to_bits())
                );
                if offset == 0 {
                    let (rot, rel, ten) = expected_matvec_enc_counts(&config, d);
                    assert_eq!(
                        (c.rotations, c.relins, c.ct_ct_tensor, c.ct_pt_mults),
                        (rot, rel, ten, 0),
                        "closed form {what}"
                    );
                }
            }
        }
    }

    /// The Task 43 conformance numbers at d = 16 (`baby = giant = 4`) for all 8 flag
    /// combinations: all off 15 rotations / 16 relins, default 6 / 4.
    #[test]
    fn closed_forms_d16() {
        let all = EvalConfig::all();
        assert_eq!(expected_matvec_enc_counts(&all[0], 16), (6, 4, 16));
        assert_eq!(expected_matvec_enc_counts(&all[7], 16), (15, 16, 16));
        let rel: Vec<u64> = all.iter().map(|c| c.matvec_enc_relins(16)).collect();
        assert_eq!(rel, [4, 16, 13, 16, 4, 16, 13, 16]);
    }

    #[test]
    fn shape_checks_follow_the_cpu() {
        let mut fx = fixture(small(), 43);
        let (slots, top, delta) = (fx.params.slots(), fx.params.max_level(), fx.params.scale());
        let _ = &mut fx.kg;
        let enc = |v: f64, s: f64, l: usize| fx.enc.encode(&vec![v; slots], s, l).unwrap();
        let ok =
            Diagonals::from_prerotated(2, 1, vec![enc(0.5, delta, top), enc(0.25, delta, top)])
                .unwrap();
        assert!(check_shape(&ok, top, slots).is_ok());
        assert!(matches!(
            check_shape(&ok, top - 1, slots),
            Err(HekitError::LevelMismatch)
        ));
        // Entries changed in place after construction.
        let mut lv = ok.clone();
        lv.entries_mut()[1] = enc(0.25, delta, top - 1);
        assert!(matches!(
            check_shape(&lv, top, slots),
            Err(HekitError::LevelMismatch)
        ));
        let mut sc = ok.clone();
        sc.entries_mut()[1] = enc(0.25, delta * 2.0, top);
        assert!(matches!(
            check_shape(&sc, top, slots),
            Err(HekitError::ScaleMismatch)
        ));
        assert!(matches!(
            check_shape(&ok, top, 3),
            Err(HekitError::InvalidParams(_))
        ));
        let _ = OpCounter::default();
    }

    /// The shared evaluator cases (`tests/cases/eval.rs`, run on the GPU with
    /// `CudaEvaluator`) on a second CPU evaluator: the harness itself, its expected counts
    /// for the default flags, and the d = 16 conformance numbers for all 8 flag combinations.
    #[test]
    fn evaluator_cases_on_the_cpu() {
        use crate::cases::eval::{EvalFixture, check_matvec_conformance, check_sequence};
        let fx = EvalFixture::new(small(), 4, 47);
        let (a, b) = (fx.cpu(EvalConfig::default()), fx.cpu(EvalConfig::default()));
        check_sequence(&a, &b, &fx, fx.params.max_level(), 3);
        let fx = EvalFixture::new(small(), 16, 48);
        for config in EvalConfig::all() {
            let (a, b) = (fx.cpu(config), fx.cpu(config));
            check_matvec_conformance(&a, &b, &fx, fx.params.max_level(), 5);
        }
    }

    /// Task 47 review I1: a composite operation run through [`counted`] that fails at any of
    /// its primitives (an injected `Device` error, as a device allocation failure part-way)
    /// leaves the counter unchanged; without a failure its counts are the CPU's. Without
    /// `counted` the same failure leaves partial counts (the test is sensitive).
    #[test]
    fn composites_count_nothing_on_failure() {
        use crate::cases::eval::EvalFixture;
        use crate::cases::failing::Failing;
        let fx = EvalFixture::new(small(), 8, 49);
        let top = fx.params.max_level();
        let x = fx.encrypt(top, 1);
        let y = fx.encrypt(top, 2);
        let (pd, cd) = fx.diagonals(top, 3);
        let steps = [0, 1, 2, 1, fx.steps[fx.steps.len() - 1]];
        for config in [EvalConfig::OPTIMIZED, EvalConfig::NAIVE] {
            let cpu = fx.cpu(config);
            let w = Failing::new(fx.cpu(config));
            type Op<'a> = Box<dyn Fn(&Failing) -> Result<Vec<Ciphertext>> + 'a>;
            let ops: Vec<(&str, Op<'_>, OpCounter)> = vec![
                (
                    "matvec_enc",
                    Box::new(|e| matvec_enc_bsgs(e, &cd, &x).map(|c| vec![c])),
                    {
                        cpu.reset_counter();
                        cpu.matvec_enc(&cd, &x).unwrap();
                        cpu.counter()
                    },
                ),
                (
                    "matvec_plain",
                    Box::new(|e| matvec_plain_bsgs(e, &pd, &x).map(|c| vec![c])),
                    {
                        cpu.reset_counter();
                        cpu.matvec_plain(&pd, &x).unwrap();
                        cpu.counter()
                    },
                ),
                (
                    "mul",
                    Box::new(|e| mul_relin(e, &x, &y).map(|c| vec![c])),
                    {
                        cpu.reset_counter();
                        cpu.mul(&x, &y).unwrap();
                        cpu.counter()
                    },
                ),
                ("rotate_each", Box::new(|e| rotate_each(e, &x, &steps)), {
                    cpu.reset_counter();
                    cpu.hoisted_rotate(&x, &steps).unwrap();
                    cpu.counter()
                }),
            ];
            for (name, op, want) in &ops {
                let what = format!("{name} {config:?}");
                w.reset_counter();
                w.fail_at(None);
                counted(&w, |e| op(e)).unwrap();
                assert_eq!(w.counter(), *want, "{what}: counts");
                let n = w.calls();
                assert!(n >= 2, "{what}: {n} primitives");
                for k in 0..n {
                    w.reset_counter();
                    w.fail_at(Some(k));
                    let r = counted(&w, |e| op(e));
                    assert!(
                        matches!(r, Err(HekitError::Device(_))),
                        "{what}: fail at {k}"
                    );
                    assert_eq!(w.counter(), OpCounter::default(), "{what}: fail at {k}");
                }
                // Sensitivity: the same failure without `counted` leaves partial counts.
                w.reset_counter();
                w.fail_at(Some(n - 1));
                assert!(op(&w).is_err());
                assert_ne!(w.counter(), OpCounter::default(), "{what}: partial counts");
            }
        }
    }

    /// Task 47 review M4 (Task 43 N2): `HostBootstrapper` merges nothing when the upload of
    /// the bootstrapped ciphertext fails (the host evaluator did the work, the backend's
    /// counter is unchanged).
    #[test]
    fn host_bootstrapper_merges_nothing_on_failed_upload() {
        use crate::cases::failing::Failing;
        use hekit_ckks::bootstrap::{BootParams, Bootstrapper};
        use hekit_ckks::{BootstrapOps, HostBootstrapper};
        let (q, p) = hekit_ckks::params::n16_boot_bits();
        let params = Arc::new(CkksParams::new(10, &q, &p, 42, 8).unwrap());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(51)).unwrap();
        let enc = Encoder::new(Arc::clone(&params));
        let bp = BootParams {
            log_slots: 9,
            ..BootParams::default_n16()
        };
        let bs = Bootstrapper::new(Arc::clone(&params), bp, &enc, &mut kg).unwrap();
        let ev = Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), None);
        let encryptor = Encryptor::new(Arc::clone(&params), kg.public_key());
        let mut r = rng(52);
        let pt = enc
            .encode(&rand_vals(&mut r, params.slots()), params.scale(), 0)
            .unwrap();
        let ct = encryptor.encrypt(&pt, &mut r);
        let w = Failing::new(ev.fork());
        let host = ev.fork();
        let hb = HostBootstrapper::new(&bs, &host);
        w.fail_uploads(true);
        let r = hb.bootstrap(&w, &ct);
        assert!(matches!(r, Err(HekitError::Device(_))), "{:?}", r.err());
        assert_eq!(w.counter(), OpCounter::default());
        assert_eq!(host.counter().bootstraps, 1, "the host did bootstrap");
        // With uploads working, exactly the host's delta of one bootstrap is merged.
        w.fail_uploads(false);
        let before = host.counter();
        hb.bootstrap(&w, &ct).unwrap();
        assert_eq!(w.counter(), &host.counter() - &before);
        assert_eq!(w.counter().bootstraps, 1);
    }

    #[test]
    fn workspace_estimates() {
        let p = CkksParams::n16_train(); // alpha 4, k 4
        // Level 31: 8 digits of 36 limbs; baby 8 for d = 64.
        assert_eq!(
            matvec_enc_workspace_limbs(&p, 31, 64),
            2 * 7 * 32 + 12 * 32 + 18 * 36 + 4 * 32
        );
        // Level 9: 3 digits over 14 limbs.
        assert_eq!(
            matvec_enc_workspace_limbs(&p, 9, 4),
            2 * 10 + 120 + 8 * 14 + 40
        );
    }
}
