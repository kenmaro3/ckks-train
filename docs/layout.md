# Slot layout for encrypted MLP training (M2, Task 25)

This document fixes the slot layout, the formulas of every layout operation, their costs, and the
level budget of one training iteration. `crates/hekit-nn/src/layout.rs` (packing, transpose plan)
and `crates/hekit-nn/src/sim.rs` (plaintext slot simulator) implement exactly these formulas. The
encrypted code of Tasks 26–30 must run the same op sequence as the `sim_*` functions, so its
`OpCounter` can be asserted equal to the simulator's `SimCounter`.

Conventions:
- `rot(v, k)[s] = v[(s + k) mod slots]` (left rotation by `k`). This matches `Evaluator::rotate`.
- `pow2(x)` is the next power of two `>= x`.
- `B = layout.batch` is the lane count, `g = baby` of `bsgs_split(m)`, and `log` means `log2`.

## 1. Layout

```
Layout { slots, d, batch }     d, batch powers of two, d * batch divides slots
slot(j, b) = j * batch + b     feature j < d, lane (sample) b < batch
```

The layout is feature-major. It is M1's strided matvec layout (`docs/m1-report.md` §7) with
`stride = batch`. The block of `D = d * batch` slots repeats `slots / D` times.

**Deviation from the brief (`d * batch == slots`).** The layout only requires that `D` divides
`slots`. `d * batch == slots` is the special case with a single block. Task 26 and Task 30 run
`d = 16` (or 8) with `batch = 8` on seal_2020 (`slots = 2^14`). Strict equality would force
`batch = 2^14 / d` lanes. The batch reduction would then cost `log(2^14 / d)` rotations instead of
`log 8`, and Task 28's count `2 * log2(batch)` would not hold. Replicating the block is free (the
client packs it). Every op below maps a `D`-periodic vector to a `D`-periodic vector, so the
replicas never interact. M1's `diagonals_from_matrix(w, slots, stride)` and `matvec_enc` already
support replicas.

### Canonical (replicated) form of a vector

A vector of logical dimension `d_v` has padded dimension `n = pow2(d_v) <= d`. It is stored
`n`-periodically in the feature index:

```
slot j*B + b  holds  x_b[j mod n]     (0 if (j mod n) >= d_v or lane b is unused)
```

Example: `n = 4` in a `d = 8` layout with `B = 2`. The digit is the feature, the letter is the lane.

```
slot:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
value: 0a 0b 1a 1b 2a 2b 3a 3b 0a 0b 1a 1b 2a 2b 3a 3b
```

Every vector in the network is kept in this form. The client packs inputs, labels and biases this
way (`pack_activations`, `pack_bias`). The layer outputs come out of the matvec, the fold and the
slot-wise ops already in this form. So the network needs **no replicate op** (see §3).

## 2. Weights: generalised diagonals

Take `W` of size `d_out x d_in`, zero-padded to `n_out x n_in` with `n_out = pow2(d_out)` and
`n_in = pow2(d_in)`. Let `L = max(n_out, n_in)` and `m = min(n_out, n_in)`. The `m` generalised
diagonals, as functions of the feature index `j` (`L`-periodic), are

```
D_c[j] = W[j mod n_out][(j + c) mod n_in],    c < m
```

Each diagonal is replicated over all `B` lanes (`slot j*B + b` holds `D_c[j]` for every `b`). In
BSGS order with `(baby, giant) = bsgs_split(m)`, `g = baby`, and `c = g*k1 + k2`, the stored entry
is pre-rotated:

```
P_c = rot(D_c, -g*k1*B)             entries[k1*baby + k2] = P_{g k1 + k2}
```

This is `pack_weight_diagonals(w, layout, bsgs_split(m))`. For square `W` (`n_out = n_in = m`) it
is element for element equal to `diagonals_from_matrix(w, slots, batch)` followed by M1's BSGS
pre-rotation. The equality is tested. So Task 26 wraps the entries with
`Diagonals::from_prerotated(m, batch, ·)` and calls `matvec_enc` directly. The Galois keys are
`bsgs_rotation_steps(m, batch)`.

### 2.1 Matvec

```
y = matvec(P, x) = sum_{k1<giant} rot( sum_{k2<baby} P_{g k1+k2} * rot(x, k2*B), g*k1*B )
y[j] = sum_{c<m} D_c[j] * x[(j + c)]          (feature units)
```

This is M1's `matvec_enc` with `d = m` and `stride = B`. It costs `m` tensor products, `giant`
relins, and `baby + giant - 2` rotations (`baby - 1` of them hoisted). The output is not rescaled.

### 2.2 Rectangular W: fold (wide) / free replicate (tall)

- **Wide, `n_in > n_out` (fold).** `L = n_in` and `m = n_out`. `y` is `n_in`-periodic. For an
  output row `r`, the features `j = r + q*n_out` (`q < n_in/n_out`) together cover every column
  `(j + c) mod n_in` exactly once. So `z_r = sum_q y[r + q*n_out]`:

  ```
  fold: for t < log(n_in/n_out):  y = y + rot(y, n_out * 2^t * B)
  ```

  This takes `log(n_in/n_out)` rotations and additions and **no mask**. The result is
  `n_out`-periodic, which is the canonical form of the output.
- **Tall, `n_in < n_out` (replicate).** `L = n_out` and `m = n_in`. The canonical `x` is already
  `n_in`-periodic, so `x[(j+c)] = x[(j+c) mod n_in]` and `y[j] = z_{j mod n_out}` directly. This
  needs **zero** extra rotations.

Costs for one forward matvec. "pad" means padding W to a square `L x L` and using M1 unchanged.
"fold/rep" means this design. "bwd" is the `W^T` matvec of the backward pass, whose fold is
`log(n_out/n_in)`. The transpose column counts rotations. All options cost 1 level (the matvec
rescale), and fold/rep adds only additions.

| W (d_out x d_in) | pad: rot / tensor / relin / transpose | fold/rep: fwd rot / tensor / relin | fold/rep: bwd rot | fold/rep: transpose |
|---|---|---|---|---|
| 16 x 16 | 6 / 16 / 4 / 15 | 6 / 16 / 4 | 6 | 15 |
| 16 x 64 | 14 / 64 / 8 / 63 | 8 / 16 / 4 | 6 | 15 |
| 64 x 16 | 14 / 64 / 8 / 63 | 6 / 16 / 4 | 8 | 15 |
| 8 x 4 (Iris W1) | 4 / 8 / 2 / 7 | 2 / 4 / 2 | 3 | 3 |
| 2 x 8 (Iris W2) | 4 / 8 / 2 / 7 | 3 / 2 / 1 | 1 | 1 |
| 32 x 784 (eMLP-1 W1) | 62 / 1024 / 32 / 1023 | 15 / 32 / 4 | 10 | 31 |
| 10 x 32 (eMLP-1 W2) | 10 / 32 / 4 / 31 | 7 / 16 / 4 | 6 | 15 |

**Choice:** fold for `n_in > n_out`, free replicate for `n_in < n_out`, and M1 unchanged for square.
Padding to a square is never cheaper. This is the spike's choice (`spikes/costmodel/costmodel.py`,
`_diag_batched`).

### 2.3 Transpose: rotations only (2020 paper Algorithm 5, generalised)

`W^T` (`n_in x n_out`) has the same `L`, `m` and BSGS split. Its diagonals are
`E_i[j] = W[(j+i) mod n_out][j mod n_in]`. Write `E_i[j] = D_c[j + t]`. Matching the row and the
column indices gives `t ≡ i (mod n_out)` and `t + c ≡ 0 (mod n_in)`. This has exactly one solution
`c < m` in each case:

```
wide (n_in >= n_out):  c = (-i) mod n_out,  t = -c      (mod L)
tall (n_in <  n_out):  c = (-i) mod n_in,   t =  i      (mod L)
E_i = rot(D_c, t*B)
```

With both sides pre-rotated (`Q_i = rot(E_i, -g*floor(i/g)*B)`):

```
Q_i = rot(P_c, a_i * B),   a_i = (t - g*floor(i/g) + g*floor(c/g)) mod L
```

`transpose_sources` gives `c(i)`, `rotations_for_transpose` gives `a_i * B` (in slots, one per
output entry, 0 = copy), and `transpose_galois_steps` gives the distinct nonzero steps (keys for
Task 27). The transpose uses **one rotation of one diagonal per output diagonal. It has no additions,
no masks and no levels.** It works because every diagonal is stored `L`-periodic (replicated) in
every lane, so a rotation by a multiple of `B` is a cyclic feature rotation. For these BSGS splits
exactly `m - 1` amounts are nonzero. `a_0 = 0`. In the square and tall cases `t ≡ i`, so
`a_i = (i mod g) + g floor(c/g)`, which is at most `m - 1 < L`. It is nonzero when
`i mod g != 0`. For `i = g k1` (`k1 >= 1`), `c = m - g k1 >= g`, so `floor(c/g) >= 1`. The wide
case (`t = -c`) is enumerated by the tests. `transpose_is_rotation_only_and_exact` checks this for 7 shapes and 2 batches. Every combination closes with rotations only, so the cost
model needs no change for the transpose.

## 3. Gradients: outer product and batch sum

The weight gradient `dW = delta^T A` (`d_out x d_in`; `delta` is the `n_out`-periodic layer
gradient `dz` and `A` is the `n_in`-periodic layer input) has generalised diagonals
`G_c[j] = sum_b delta_b[j mod n_out] A_b[(j+c) mod n_in]`. Pre-rotated:

```
rot(G_c, -g k1 B) = batchsum( rot(delta, -g k1 B) * rot(A, k2 B) )       c = g k1 + k2
```

This holds because lane sums commute with rotations by multiples of `B`. `sim_outer_batchsum` runs:

1. `baby - 1` hoisted rotations `rot(A, k2 B)` and `giant - 1` rotations `rot(delta, -g k1 B)`.
2. `m` tensor products, then `m` relins (rotation needs degree 1), then a rescale (1 level).
3. reduce: `for t < log B: v = v + rot(v, 2^t)`. Lane 0 of each feature now holds the lane sum
   (`log B` rotations).
4. `v = v * (c * mask0)` with `mask0[s] = [s mod B == 0]` and `c` the folded constant (1 ct-pt
   multiply, then a rescale: 1 level).
5. replicate: `for t < log B: v = v + rot(v, -2^t)` (`log B` rotations).

Rotations: `(baby - 1) + (giant - 1) + 2 m log B`. Tensors: `m`. Relins: `m`. ct-pt: `m`.
Levels: 2. The bias gradient `sim_batchsum(dz)` is steps 3–5: `2 log B` rotations, 1 ct-pt and
1 level.

**The mask is unavoidable for a replicated result.** Rotations, additions and scalar (constant)
multiplications only produce circulant maps, since they commute with `rot(·, 1)`. "Sum the `B`
lanes of each feature and write the sum to every lane" does not commute with `rot(·, 1)` in the
feature-major layout, so any rotation-and-add circuit needs one non-constant plaintext. The
design therefore folds the mask into the multiplication that is needed anyway, the constant
`c = lr / n_valid` (SGD) or `(1 + mu) lr / n_valid` (NAG, §4). The mask then costs **no extra
level**.

Mask-free alternative (cost model `interactive=True`): stop after step 3. The server applies `c`
with `mul_scalar` (1 level, so the level count is the same) and updates `W` with a gradient that
is valid only in lane 0. The client refresh then has to read lane 0 and re-replicate the weights.
That saves `m log B` rotations per layer and update, but the refresh is no longer a pure
decrypt/re-encrypt. M3's bootstrap refresh could not do it, and Task 30's `Refresher::refresh(ct)`
does not have that signature. `sim_batch_reduce` alone is this variant. **Default: steps 1–5
(replicated `dW`, pure refresh).** This is a choice for Task 32 to revisit if RQ1 wants the cheaper
count.

### 3.1 Are the masks integral? (research-lead ruling, item 1)

A slot vector `z` (real, `N/2` slots) encodes to a polynomial whose constant coefficient is
`(2/N) * sum_s z_s`, the mean of the slots. For a 0/1 mask with `k` ones, that is `2k/N`, which is
an integer only for `k = 0` or `k = N/2`. So **every non-constant 0/1 mask is a non-integral
polynomial**. Encoding it at scale 1 rounds the mask away (it decodes to about 0). Only constant
masks survive. The test `masks_at_scale_one_are_destroyed` shows this with the real encoder on
seal_2020. The lane-0 mask, with `B = 8`, decodes to an error of about 1. The all-ones mask is
exact. The zero-level "integer mask" route is therefore unavailable.

Masks in this design:

| mask | where | status |
|---|---|---|
| transpose | — | none (rotations only, §2.3) |
| fold | — | none (§2.2) |
| replicate (tall layer) | — | none (canonical form) |
| batch-sum lane-0 `mask0` | steps 4 of §3 | unavoidable; folded with the constant `c` into the one ct-pt multiply that `c` needs anyway: 0 extra levels |
| valid lanes (`n_valid < B`) | `z * (2 * valid)` for Square' | folded with the derivative constant 2. It runs on `z`, off the critical path: 0 extra levels (§4) |

Masks are encoded at exact scale. The plaintext scale is `s_target * q_l / s_ct`, then the product
is rescaled, as for `mul_scalar`. The last row is how non-power-of-two batches work. A padded
lane has `x = 0`, but the bias makes its activations nonzero. Then `delta = a - y` is nonzero
there too. Multiplying the derivative by `valid` zeroes the padded lanes of `dz`. `W^T dz` does not
mix lanes, so every gradient and every lower-layer `dz` is clean. Padded **features** stay zero
without a mask, provided `act(0) = 0` (Square and z² + z): the padded rows and columns of `W` and
`b` receive zero gradients by induction.

## 4. Level budget of one training iteration

Each op is listed with the levels it consumes. Everything starts fresh at `max_level` (weights,
biases and `x` encrypted by the client; `y` encrypted at the exact scale and level of `a_L`).
Equal-level and equal-scale operands come from `mod_switch_to_level` (free) and exact-scale
encoding.

| op | levels |
|---|---|
| forward per layer: `matvec_enc` + fold + bias add, then rescale | 1 |
| activation Square / z² + z (the `+z` term aligned by a parallel `mul_scalar(z, 1)`) | 1 |
| loss gradient `delta = a_L - y` | 0 |
| derivative factor `s = z * (2*valid)` (+ `valid` for z² + z). `z` is at a higher level, off the critical path | (1, parallel) |
| `dz = delta * s` | 1 |
| transpose `W^T` (rotations only) | 0 |
| backward input `W^T dz` (matvec + fold + rescale) | 1 |
| weight gradient: outer product (tensor + relin + rescale) | 1 |
| batch sum with `mask0 * c` (`c = lr / n_valid`, constant folded) | 1 |
| update `W - c dW` (scales equal by construction) | 0 |

**Task 30 network, 4 → 8 → 2, Square, batch 8, seal_2020 (`max_level` 9).** This is the
critical path. The test `train_step_level_table` checks every row with `sim_train_step`.

| step | value | level after |
|---|---|---|
| 0 | x, W1, b1, W2, b2 fresh | 9 |
| 1 | z1 = W1 x + b1 | 8 |
| 2 | a1 = z1² | 7 |
| 3 | z2 = W2 a1 + b2 | 6 |
| 4 | a2 = z2² | 5 |
| 5 | δ2 = a2 − y | 5 |
| 6 | s2 = z2 · 2valid (parallel) | 5 |
| 7 | dz2 = δ2 · s2 | 4 |
| 8 | δ1 = W2ᵀ dz2 (W2ᵀ at 9, mod-switched) | 3 |
| 9 | outer(dz2, a1) | 3 |
| 10 | ĝ2 = batchsum(·) · c·mask0; W2' = W2 − ĝ2 | 2 |
| 11 | b2' = b2 − batchsum(dz2) · c·mask0 | 3 |
| 12 | s1 = z1 · 2valid (parallel) | 7 |
| 13 | dz1 = δ1 · s1 | 2 |
| 14 | outer(dz1, x) | 1 |
| 15 | ĝ1, W1' = W1 − ĝ1 | **0** |
| 16 | b1' | 1 |

**Verdict for seal_2020: exactly 9 levels, so no client refresh is needed inside the iteration.**
Task 30 refreshes all weights after the update, as specified (W1' is at level 0). The margin is
**zero**. One extra multiplication on the layer-1 gradient path pushes it to 10 levels. That
happens with any of the following:
- a separate `lr` multiply (Task 28's literal API: `weight_grad` returns `dW / batch`, then
  `Sgd::step` calls `mul_scalar(lr)`);
- a separate `1/batch`;
- a separate mask level;
- NAG done as a multiply on the gradient.

So the constants must be folded (ruling item 2). The gradient constant `c` goes into the `mask0`
plaintext of step 4. NAG (`g = G/n + decay W`, `V' = mu V + g`, `W' = W - lr(1+mu) g - mu² lr V`)
is rewritten with the scaled velocity `U = lr(1+mu)V`:
- `h = lr(1+mu)/n * G`, with the mask folded, plus `lr(1+mu) decay * W` (a scalar on the fresh `W`);
- `U' = mu U + h`;
- `W' = W - h - (mu²/(1+mu)) U`.

The only multiplications by `mu` act on fresh `U` and `W`, off the critical path, so NAG also fits
in 9. With that folding, **`seal_2020_l10` is not needed** for the Iris network, including
Task 31's batch 20 (32 lanes, `valid` mask, same depth). If Task 28 keeps an unfused API, the
depth is 10: use `seal_2020_l10` for Task 31's batch-20 run only, or refresh `dz1` once before the
layer-1 gradient.

General depth for an `L`-layer MLP with one-level activations: `2L` (forward) `+ 1` (the top `dz`)
`+ 2(L-1)` (`W^T` and `dz` per lower layer) `+ 2` (gradient and masked constant) `= 4L + 1`.

**eMLP-1, 784 → 32 → 10, z² + z, n16_train (`max_level` 31).** `d = 1024` and `B <= 32` lanes. The
structure is the same as above with `L = 2`: depth 9 fused, 10 or 11 unfused. Both are far below
31, so there is **no mid-iteration refresh**. That leaves room for about 5 more layers at 4 levels
each.

## 5. Differences from the M0 spike (`spikes/costmodel/slotsim.py`) and the cost-model change Task 32 needs

The layout, the diagonals `D_c`, the pre-rotation, the fold, the free replicate, the transpose
formula `a_i` and the transpose count `m - 1` are the spike's (`diag_batched`). They are ported to
Rust as is. Differences:

1. **Batch sum and replication.** The spike's `interactive=True` (the variant Task 32 compares
   against) reduces only. Its gradient costs `m log B'` rotations, with no mask and no replication,
   and the client re-encodes from lane 0. This design replicates on the server: `+ m log B`
   rotations and `+ m` ct-pt per layer per update, at no extra level. Task 32's cost model must
   add `m * log2(B')` rotations to the gradient phase in interactive mode (or Task 28 must adopt
   the lane-0 variant of §3 together with a re-laying refresh).
2. **Baby rotations of the layer input in the gradient.** The spike reuses the forward pass's baby
   rotations `rot(A, k2 B)`. M1's `matvec_enc` does not return them, so the gradient recomputes
   them: `+ (baby - 1)` hoisted rotations per layer per update. Task 32: add `n1 - 1` per layer to
   the gradient phase, or expose the babies from the matvec.
3. **Constants.** The spike counts `nag_pt_mults` ct-pt mults per diagonal in the update. Here the
   gradient constant is folded into the mask (`m` ct-pt in the gradient phase, already counted
   there). NAG adds scalar mults on fresh `W` and `U` only. ct-pt counts are not key switches, so
   the key-switch comparison is unaffected.
4. **Lazy relin over ciphertexts.** The spike sums gradient products over several ciphertexts
   (`k > 1`) before one relin. Here `B` lanes fit in one ciphertext (`k = 1`), so this is the same.

Neither change touches the forward, transpose or backward counts. With 1 and 2 applied, the cost
model's interactive gradient phase is `(n2-1) + (n1-1) + 2 m log2(B')` rotations and `m` relins,
which equals `sim_outer_batchsum`. The spike itself is not edited (project rule).
