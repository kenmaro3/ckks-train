//! Task 40: compact re-packing of the weight diagonals (`ckks_train_nn::compact`,
//! `docs/layout.md` §6).
//!
//! - The plan, offsets, tree steps, key set and counts equal the M0 cost model
//!   (`spikes/costmodel/costmodel.py`, `_diag_batched`, `compact=True`) for the same
//!   `(d_in, d_out, batch, slots)`. The model's values are hard-coded below from a run of the
//!   spike (it is not edited or imported).
//! - `compact_unpack(compact_pack(W))` on a small preset (N = 2^10, insecure, tests only)
//!   for square, wide, tall and padded layers, with full, several and partial compact
//!   ciphertexts and with lane and period-copy offsets. The diagonals come back within
//!   2^-18, the levels drop by exactly 1 each, the scales are kept exactly, and the counter
//!   equals the model's counts with 0 relins. The inputs have drifted block copies and lanes
//!   (2^-10), and the output is block-periodic again (ruling R-M3-3).
//! - The brief's gate on n16_boot, `(d_out, d_in) = (32, 1024)`, batch 32 (ignored: heavy).
//!
//! The evaluators have **no relinearization key**: a relin would be an error.

use std::sync::Arc;
use std::time::Instant;

use ckks_train_ckks::bootstrap::{BootParams, levels_after_boot};
use ckks_train_ckks::linalg::Diagonals;
use ckks_train_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter,
};
use ckks_train_core::error::HekitError;
use ckks_train_core::sampler::Rng;
use ckks_train_nn::compact::{
    CompactPlan, boot_count, compact_extract, compact_pack, compact_pack_plain,
    compact_rotation_steps, compact_shift, compact_unpack, extract_steps, shift_steps,
};
use ckks_train_nn::dense::EncWeights;
use ckks_train_nn::layout::{LayerShape, Layout, pack_weight_diagonals};
use ndarray::Array2;

/// The brief's gate on the round trip.
const TOL: f64 = 1.0 / (1u64 << 18) as f64;

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

/// Uniform in `[-1, 1)`.
fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

fn vec_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// `max_s |v[s] - v[s mod period]|`: 0 for a `period`-periodic vector.
fn spread(v: &[f64], period: usize) -> f64 {
    v.iter()
        .enumerate()
        .map(|(s, x)| (x - v[s % period]).abs())
        .fold(0.0, f64::max)
}

/// `max_s |v[s] - v[s - s mod B]|`: 0 when all lanes of every feature agree.
fn lane_spread(v: &[f64], b: usize) -> f64 {
    v.iter()
        .enumerate()
        .map(|(s, x)| (x - v[s - s % b]).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------------------------
// The plan against the cost model (no encryption).
// ---------------------------------------------------------------------------------------------

/// `costmodel._diag_batched(d_in, d_out, batch, slots, momentum=False, decay=False)`:
/// `weight_ciphertexts` (= `bootstraps_per_update` without momentum), the lane shifts
/// (`gradient` rotations minus those of `interactive=True`) and the `update` rotations (the
/// unpack), from a run of the spike.
/// `(d_in, d_out, batch, slots)` of a cost-model call.
type Setting = (usize, usize, usize, usize);

const MODEL: [(Setting, usize, usize, usize); 7] = [
    // (d_in, d_out, batch, slots), n_w, lane shifts, unpack rotations
    ((1024, 32, 32, 1 << 15), 1, 194 - 163, 191),
    ((784, 32, 32, 1 << 15), 1, 194 - 163, 191),
    ((32, 10, 32, 1 << 15), 1, 96 - 81, 175),
    ((16, 16, 8, 1 << 14), 1, 64 - 49, 175),
    ((64, 16, 8, 1 << 14), 1, 64 - 49, 143),
    ((16, 64, 8, 1 << 14), 1, 64 - 49, 143),
    ((256, 256, 32, 1 << 14), 4, 1539 - 1287, 1788),
];

#[test]
fn plan_matches_the_cost_model() {
    for &((d_in, d_out, batch, slots), nw, shifts, unpack) in &MODEL {
        let d = d_in.max(d_out).next_power_of_two();
        // The model's batch per ciphertext B' = min(pow2(batch), slots / L).
        let lanes = batch.next_power_of_two().min(slots / d);
        let layout = Layout::new(slots, d, lanes).unwrap();
        let plan = CompactPlan::for_layer(&layout, d_out, d_in);
        let what = format!("{d_out} x {d_in}, B {lanes}, S {slots}");
        assert_eq!(plan.ciphertexts(), nw, "{what}: n_ct");
        assert_eq!(boot_count(&plan, d_out), nw, "{what}: boot_count");
        assert_eq!(plan.pack_rotations(), shifts, "{what}: pack rotations");
        assert_eq!(plan.unpack_rotations(), unpack, "{what}: unpack rotations");
    }
}

/// The brief's plan values, the offsets (disjoint, covering) and the key set.
#[test]
fn plan_offsets_steps_and_keys() {
    let layout = Layout::new(1 << 15, 1024, 32).unwrap();
    let p = CompactPlan::for_layer(&layout, 32, 1024);
    assert_eq!(
        p,
        CompactPlan {
            d_out: 32,
            d_in: 1024,
            batch: 32,
            diags_per_ct: 32,
            lane_shift: 5
        }
    );
    assert_eq!(
        (p.diag_len(), p.diagonals(), p.slots()),
        (1024, 32, 1 << 15)
    );
    assert_eq!(boot_count(&p, 32), 1, "32 diagonals in one ciphertext");
    assert_eq!(boot_count(&p, 64), 2);
    assert_eq!(p.steps(), vec![1, 2, 4, 8, 16]);
    assert_eq!(
        compact_rotation_steps(&p),
        vec![-16, -8, -4, -2, -1, 1, 2, 4, 8, 16],
        "10 keys, all batch reduce/replicate steps"
    );
    // The docs/layout.md §6.4 table: eMLP-1 W1 (32 x 784) and W2 (10 x 32) in d = 1024.
    let w1 = CompactPlan::for_layer(&layout, 32, 784);
    assert_eq!(
        (w1.ciphertexts(), compact_rotation_steps(&w1).len()),
        (1, 10)
    );
    let w2 = CompactPlan::for_layer(&layout, 10, 32);
    assert_eq!(w2.diags_per_ct, 1024);
    assert_eq!((w2.pack_rotations(), w2.unpack_rotations()), (15, 175));
    assert_eq!(compact_rotation_steps(&w2).len(), 14);

    // Lane and period offsets: N = 2^10 layout d = 32, B = 4 (4 block copies).
    let layout = Layout::new(512, 32, 4).unwrap();
    for (d_out, d_in, cap, keys) in [(32, 32, 16, 7), (8, 32, 16, 7), (16, 16, 32, 9)] {
        let p = CompactPlan::for_layer(&layout, d_out, d_in);
        assert_eq!(p.diags_per_ct, cap);
        let (l, b) = (p.diag_len(), p.batch);
        let mut seen = vec![false; p.slots()];
        for t in 0..cap {
            for j in 0..l {
                let s = p.offset(t) + j * b;
                assert!(!seen[s], "positions overlap");
                seen[s] = true;
            }
        }
        assert!(
            seen.iter().all(|&x| x),
            "a full compact ciphertext covers every slot"
        );
        for (k, &st) in p.steps().iter().enumerate() {
            assert_eq!(st, p.offset(1 << k));
        }
        let steps = compact_rotation_steps(&p);
        assert_eq!(steps.len(), keys, "{d_out} x {d_in}: {steps:?}");
        let logcap = cap.trailing_zeros() as usize;
        let split = p.diagonals().min(cap).trailing_zeros() as usize;
        // log2(S/L) + log2(min(m, S/L)), less one when +-S/2 coincide.
        let half = p.steps().last() == Some(&(p.slots() / 2)) && split == logcap;
        assert_eq!(keys, logcap + split - usize::from(half));
    }
}

// ---------------------------------------------------------------------------------------------
// Encrypted round trips.
// ---------------------------------------------------------------------------------------------

struct Ctx {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
}

/// One shared `Arc`, Galois keys for exactly `steps`, no relinearization key.
fn ctx(params: CkksParams, steps: &[i32], seed: u8) -> Ctx {
    let params = Arc::new(params);
    let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
    let pk = kg.public_key();
    let gks = kg.galois_keys(steps);
    assert_eq!(gks.len(), steps.len(), "one key per step");
    Ctx {
        encoder: Encoder::new(Arc::clone(&params)),
        enc: Encryptor::new(Arc::clone(&params), pk),
        dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
        ev: Evaluator::new(Arc::clone(&params), None, Some(gks)),
        params,
    }
}

/// N = 2^10 (512 slots), q = {60, 40 x 3}, p = {60}, scale 2^40. Insecure, tests only.
fn small() -> CkksParams {
    CkksParams::new(10, &[60, 40, 40, 40], &[60], 40, 4).unwrap()
}

/// What one round trip measured.
#[derive(Debug)]
struct Trip {
    pack: OpCounter,
    unpack: OpCounter,
    /// `max |packed - expected compact layout|`.
    packed_err: f64,
    /// `max |unpacked diagonals - clean replicated diagonals|`.
    err: f64,
    /// Block spread of the unpacked diagonals (0 for block-periodic).
    block_spread: f64,
    /// Lane spread of the unpacked diagonals.
    lane_spread: f64,
    /// Block spread of the drifted input.
    input_spread: f64,
    seconds: (f64, f64),
}

/// `W` (`d_out x d_in`, entries in `[-1, 1)`) packed for `layout`, encrypted at `level` with
/// `Delta`, every slot except the ones pack reads (lane 0 of the first period) shifted by its
/// own `drift * uni`: drifted lanes and block copies. Runs pack, then unpack, and checks the
/// level, scale and count contract. Returns the measurements.
fn round_trip(
    c: &Ctx,
    layout: &Layout,
    d_out: usize,
    d_in: usize,
    level: usize,
    drift: f64,
    r: &mut Rng,
) -> Trip {
    let sh = LayerShape::new(d_out, d_in, layout);
    let w = Array2::from_shape_fn((d_out, d_in), |_| uni(r));
    let clean = pack_weight_diagonals(&w, layout, (sh.baby, sh.giant));
    let (l, b) = (sh.l, layout.batch);
    let read = |s: usize| s < l * b && s % b == 0;
    let drifted: Vec<Vec<f64>> = clean
        .iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .map(|(s, x)| if read(s) { *x } else { x + drift * uni(r) })
                .collect()
        })
        .collect();
    let delta = c.params.scale();
    let entries: Vec<Ciphertext> = drifted
        .iter()
        .map(|v| {
            c.enc
                .encrypt(&c.encoder.encode(v, delta, level).unwrap(), r)
        })
        .collect();
    let weights = EncWeights::from_diagonals(
        Diagonals::from_prerotated(sh.m, b, entries).unwrap(),
        d_out,
        d_in,
        layout,
    )
    .unwrap();
    let plan = CompactPlan::for_layer(layout, d_out, d_in);

    let c0 = c.ev.counter();
    let t0 = Instant::now();
    let packed = compact_pack(&c.ev, &weights, &plan).unwrap();
    let t_pack = t0.elapsed().as_secs_f64();
    let c1 = c.ev.counter();
    assert_eq!(packed.len(), plan.ciphertexts());
    assert_eq!(packed.len(), boot_count(&plan, d_out));
    for ct in &packed {
        assert_eq!(ct.level(), level - 1, "pack consumes exactly 1 level");
        assert_eq!(ct.scale(), delta, "pack keeps the scale exactly");
    }
    // The compact layout: slot offset(t) + j B holds P_{i cap + t}[j], 0 elsewhere.
    let cap = plan.diags_per_ct;
    let mut packed_err: f64 = 0.0;
    for (i, ct) in packed.iter().enumerate() {
        let mut want = vec![0.0; c.params.slots()];
        for t in 0..plan.diagonals_in(i) {
            for j in 0..l {
                want[plan.offset(t) + j * b] = clean[i * cap + t][j * b];
            }
        }
        packed_err = packed_err.max(vec_diff(&c.encoder.decode(&c.dec.decrypt(ct)), &want));
    }

    let t0 = Instant::now();
    let back = compact_unpack(&c.ev, &packed, &plan).unwrap();
    let t_unpack = t0.elapsed().as_secs_f64();
    let c2 = c.ev.counter();
    assert_eq!((back.d_out(), back.d_in()), (d_out, d_in));
    assert_eq!(
        back.level(),
        level - 2,
        "pack + unpack consume exactly 2 levels"
    );
    assert_eq!(back.scale(), delta, "unpack keeps the scale exactly");

    let got: Vec<Vec<f64>> = back
        .diagonals()
        .entries()
        .iter()
        .map(|ct| c.encoder.decode(&c.dec.decrypt(ct)))
        .collect();
    let err = got
        .iter()
        .zip(&clean)
        .map(|(g, w)| vec_diff(g, w))
        .fold(0.0, f64::max);
    let bk = layout.block();
    let block_spread = got.iter().map(|g| spread(g, bk)).fold(0.0, f64::max);
    let lane = got.iter().map(|g| lane_spread(g, b)).fold(0.0, f64::max);
    let input_spread = drifted.iter().map(|g| spread(g, bk)).fold(0.0, f64::max);
    // The weights themselves (lane 0 of every diagonal).
    let wd = back.decrypt(&c.dec, &c.encoder, layout);
    let we = wd
        .iter()
        .zip(&w)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(we < TOL, "weights {we:e}");

    let trip = Trip {
        pack: &c1 - &c0,
        unpack: &c2 - &c1,
        packed_err,
        err,
        block_spread,
        lane_spread: lane,
        input_spread,
        seconds: (t_pack, t_unpack),
    };
    let m = plan.diagonals() as u64;
    let what = format!("{d_out} x {d_in} (B {b})");
    // The counter equals the model's counts; no relin, no tensor.
    let want_pack = OpCounter {
        rotations: plan.pack_rotations() as u64,
        ct_pt_mults: m,
        rescales: m,
        ..OpCounter::default()
    };
    let want_unpack = OpCounter {
        rotations: plan.unpack_rotations() as u64,
        ct_pt_mults: m,
        rescales: m,
        ..OpCounter::default()
    };
    assert_eq!(trip.pack, want_pack, "{what}: pack counter");
    assert_eq!(trip.unpack, want_unpack, "{what}: unpack counter");
    assert_eq!(trip.pack.relins + trip.unpack.relins, 0);
    eprintln!(
        "{what}: n_ct {}, rot {} + {}, packed 2^{:.1}, round trip 2^{:.1}, block spread 2^{:.1} \
         (input 2^{:.1}), lane spread 2^{:.1}, {:.2} + {:.2} s",
        packed.len(),
        trip.pack.rotations,
        trip.unpack.rotations,
        trip.packed_err.log2(),
        trip.err.log2(),
        trip.block_spread.log2(),
        trip.input_spread.log2(),
        trip.lane_spread.log2(),
        trip.seconds.0,
        trip.seconds.1,
    );
    assert!(
        trip.packed_err < TOL,
        "{what}: packed {:e}",
        trip.packed_err
    );
    assert!(trip.err < TOL, "{what}: round trip {:e}", trip.err);
    assert!(
        trip.block_spread < TOL,
        "{what}: blocks {:e}",
        trip.block_spread
    );
    assert!(
        trip.lane_spread < TOL,
        "{what}: lanes {:e}",
        trip.lane_spread
    );
    trip
}

/// Square, wide, tall and padded layers on N = 2^10, layout `d = 32`, `B = 4` (4 block copies):
/// two full compact ciphertexts (32 x 32), one partial (8 x 32, 32 x 8), period-copy offsets
/// everywhere. Inputs drifted by 2^-10 in every slot pack does not read.
#[test]
fn compact_round_trip_small() {
    let layout = Layout::new(512, 32, 4).unwrap();
    let shapes = [(32, 32), (8, 32), (32, 8), (10, 20), (16, 16)];
    let mut steps: Vec<i32> = shapes
        .iter()
        .flat_map(|&(o, i)| compact_rotation_steps(&CompactPlan::for_layer(&layout, o, i)))
        .collect();
    steps.sort_unstable();
    steps.dedup();
    let c = ctx(small(), &steps, 1);
    let mut r = rng(2);
    let drift = 2f64.powi(-10);
    for (d_out, d_in) in shapes {
        let t = round_trip(&c, &layout, d_out, d_in, 3, drift, &mut r);
        assert!(
            t.input_spread > drift / 2.0,
            "the input copies have drifted"
        );
    }
    // B = 1 (a single lane, period copies only) and a single-block layout (lanes only).
    for (layout, d_out, d_in) in [
        (Layout::new(512, 32, 1).unwrap(), 16, 32),
        (Layout::new(512, 32, 16).unwrap(), 32, 32),
    ] {
        let plan = CompactPlan::for_layer(&layout, d_out, d_in);
        let c = ctx(small(), &compact_rotation_steps(&plan), 3);
        round_trip(&c, &layout, d_out, d_in, 2, drift, &mut r);
    }
}

/// Task 41: `compact_shift` on inputs that are already zero outside lane 0 of the first
/// period is `compact_pack` without the mask: the same compact layout
/// (`compact_pack_plain`), `m - n_ct` rotations and nothing else (no ct-pt, no level). And
/// `compact_extract` returns one position in replicated form at an exact target scale with
/// `[o(t) != 0] + log2(cap)` rotations, 1 ct-pt and 1 level.
#[test]
fn compact_shift_and_extract() {
    let layout = Layout::new(512, 32, 4).unwrap();
    let mut r = rng(6);
    for (d_out, d_in) in [(32, 32), (8, 32), (10, 20)] {
        let plan = CompactPlan::for_layer(&layout, d_out, d_in);
        let mut steps = compact_rotation_steps(&plan);
        steps.extend(shift_steps(&plan));
        for t in 0..plan.diags_per_ct {
            steps.extend(extract_steps(&plan, t));
        }
        let mut seen = std::collections::HashSet::new();
        steps.retain(|&k| k.rem_euclid(512) != 0 && seen.insert(k.rem_euclid(512)));
        let c = ctx(small(), &steps, 7);
        let sh = LayerShape::new(d_out, d_in, &layout);
        let (l, b) = (sh.l, layout.batch);
        let w = Array2::from_shape_fn((d_out, d_in), |_| uni(&mut r));
        let clean = pack_weight_diagonals(&w, &layout, (sh.baby, sh.giant));
        // Zero outside the slots the pack reads.
        let masked: Vec<Vec<f64>> = clean
            .iter()
            .map(|v| {
                v.iter()
                    .enumerate()
                    .map(|(s, x)| if s < l * b && s % b == 0 { *x } else { 0.0 })
                    .collect()
            })
            .collect();
        let delta = c.params.scale();
        let cts: Vec<Ciphertext> = masked
            .iter()
            .map(|v| {
                c.enc
                    .encrypt(&c.encoder.encode(v, delta, 2).unwrap(), &mut r)
            })
            .collect();
        let before = c.ev.counter();
        let packed = compact_shift(&c.ev, &cts, &plan).unwrap();
        let got = &c.ev.counter() - &before;
        assert_eq!(
            got,
            OpCounter {
                rotations: plan.pack_rotations() as u64,
                ..OpCounter::default()
            },
            "{d_out} x {d_in}: only the lane shifts"
        );
        let want = compact_pack_plain(&masked, &plan);
        for (ct, v) in packed.iter().zip(&want) {
            assert_eq!(
                (ct.level(), ct.scale()),
                (2, delta),
                "no level, exact scale"
            );
            let e = vec_diff(&c.encoder.decode(&c.dec.decrypt(ct)), v);
            assert!(e <= TOL, "{d_out} x {d_in}: 2^{}", e.log2());
        }
        // Every position back in replicated form at another exact scale.
        let target = delta * delta / c.params.q()[2] as f64 * 2.0;
        for (i, ct) in packed.iter().enumerate() {
            for t in 0..plan.diagonals_in(i) {
                let before = c.ev.counter();
                let y = compact_extract(&c.ev, ct, &plan, t, target).unwrap();
                let got = &c.ev.counter() - &before;
                let rot =
                    usize::from(plan.offset(t) != 0) + plan.diags_per_ct.trailing_zeros() as usize;
                assert_eq!(
                    (got.rotations, got.ct_pt_mults, got.relins),
                    (rot as u64, 1, 0)
                );
                assert_eq!((y.level(), y.scale()), (1, target));
                let e = vec_diff(
                    &c.encoder.decode(&c.dec.decrypt(&y)),
                    &clean[i * plan.diags_per_ct + t],
                );
                assert!(e <= TOL, "extract {t}: 2^{}", e.log2());
            }
        }
        // Wrong entry count: refused before any work.
        let before = c.ev.counter();
        assert!(compact_shift(&c.ev, &cts[1..], &plan).is_err());
        assert_eq!(c.ev.counter(), before);
    }
}

/// Input errors are reported before any work (the counter does not move).
#[test]
fn compact_errors_count_nothing() {
    let layout = Layout::new(512, 32, 4).unwrap();
    let plan = CompactPlan::for_layer(&layout, 32, 32);
    let steps = compact_rotation_steps(&plan);
    let c = ctx(small(), &steps[1..], 4);
    let mut r = rng(5);
    let w = Array2::from_shape_fn((32, 32), |_| uni(&mut r));
    let delta = c.params.scale();
    let sh = LayerShape::new(32, 32, &layout);
    let entries: Vec<Ciphertext> = pack_weight_diagonals(&w, &layout, (sh.baby, sh.giant))
        .iter()
        .map(|v| {
            c.enc
                .encrypt(&c.encoder.encode(v, delta, 3).unwrap(), &mut r)
        })
        .collect();
    let weights = EncWeights::from_diagonals(
        Diagonals::from_prerotated(32, 4, entries.clone()).unwrap(),
        32,
        32,
        &layout,
    )
    .unwrap();
    let inv = |e: HekitError| matches!(e, HekitError::InvalidParams(_));
    let before = c.ev.counter();
    // A missing key.
    assert!(inv(compact_pack(&c.ev, &weights, &plan).unwrap_err()));
    assert!(inv(compact_unpack(&c.ev, &entries[..2], &plan).unwrap_err()));
    // A plan of another layer.
    let other = CompactPlan::for_layer(&layout, 16, 32);
    assert!(inv(compact_pack(&c.ev, &weights, &other).unwrap_err()));
    // The wrong number of compact ciphertexts.
    assert!(inv(compact_unpack(&c.ev, &entries[..3], &plan).unwrap_err()));
    // Mixed levels.
    let low = c.ev.mod_switch_to_level(&entries[1], 2).unwrap();
    assert_eq!(
        compact_unpack(&c.ev, &[entries[0].clone(), low], &plan).unwrap_err(),
        HekitError::LevelMismatch
    );
    // A plan for other parameters.
    let wrong = CompactPlan::for_layer(&Layout::new(1024, 32, 4).unwrap(), 32, 32);
    assert!(inv(
        compact_unpack(&c.ev, &entries[..1], &wrong).unwrap_err()
    ));
    assert_eq!(c.ev.counter(), before);
}

/// The brief's gate on n16_boot: `(d_out, d_in) = (32, 1024)`, batch 32 (layout `d = 1024`,
/// one block). The diagonals start on the bootstrap output level (16), where the trainer's
/// refreshed weights arrive. Round trip within 2^-18, `boot_count == 1` (the cost model's
/// `weight_ciphertexts`), pack 31 and unpack 191 rotations (the model's lane shifts and
/// unpack), 0 relins, 2 levels. Only the 10 Galois keys of [`compact_rotation_steps`]
/// (~0.3 GB each).
#[test]
#[ignore = "heavy: n16_boot (N = 2^16), 10 Galois keys, ~5 GB peak, ~50 s; run with --ignored"]
fn compact_round_trip_n16_boot() {
    let t0 = Instant::now();
    let params = CkksParams::n16_boot();
    let level = levels_after_boot(&params, &BootParams::default_n16());
    assert_eq!(level, 16);
    let layout = Layout::new(params.slots(), 1024, 32).unwrap();
    let plan = CompactPlan::for_layer(&layout, 32, 1024);
    assert_eq!(
        boot_count(&plan, 32),
        MODEL[0].1,
        "the cost model's bootstraps"
    );
    let steps = compact_rotation_steps(&plan);
    assert_eq!(steps.len(), 10);
    let c = ctx(params, &steps, 6);
    eprintln!(
        "n16_boot keygen ({} Galois keys): {:.1} s",
        steps.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut r = rng(7);
    let t = round_trip(&c, &layout, 32, 1024, level, 0.0, &mut r);
    assert_eq!(
        t.pack.rotations as usize, MODEL[0].2,
        "the model's lane shifts"
    );
    assert_eq!(
        t.unpack.rotations as usize, MODEL[0].3,
        "the model's unpack"
    );
    eprintln!("{t:?}");
}
