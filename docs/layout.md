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
`i mod g != 0`. For `i = g k1` (`k1 >= 1`), `c = m - g k1 >= g`, so `floor(c/g) >= 1`. In the
wide case `t = -c`. Write `i = g k1 + k2`. For `k2 = 0`, `a = -i`. For `k2 > 0`,
`c = m - i` gives `a = k2 - g(k1 + 1)`, which lies in `(-L, 0)`. So `a = 0 (mod L)` only when
`i = 0`. `transpose_is_rotation_only_and_exact` checks the count for 7 shapes and 2 batches.
Every combination closes with rotations only, so the cost model needs no change to its transpose
rotation count.

#### Galois-key cost of the transpose

The amounts `a_i * B` are mostly steps that no other op uses, so each one needs its own Galois
key. One key costs ≈ 302 MB on n16_train (`docs/m1-report.md` §7: dnum 8 × 2 × 36 limbs ×
2^16 × 8 B) and ≈ 58 MB on seal_2020 (10 × 2 × 11 × 2^15 × 8 B). These are the distinct rotation
keys of one full training step: forward and backward BSGS, folds, transpose, gradient giants
`-g k1 B`, and batch reduce/replicate `±2^t`.

| network | lanes B | keys, full step | keys used only by the transpose | rotation-key memory |
|---|---|---|---|---|
| Iris (Task 30), 4 → 8 → 2, seal_2020 | 8 | 11 | 1 | ≈ 0.6 GB |
| Iris (Task 31), 4 → 8 → 3, seal_2020 | 32 | 17 | 3 | ≈ 1.0 GB |
| eMLP-1, 784 → 32 → 10, n16_train (2^15 slots) | 32 | 45 | 14 | ≈ 13.6 GB (4.2 GB for the transpose-only keys) |

For eMLP-1 the W2 transpose (`10 x 32`, padded `16 x 32`, wide) uses steps `{17..31} * B`. 14 of
them are new keys. On the 36 GB machine this matters.

**Zero-new-key alternative.** Rotate twice with keys the step already has.
- Square or tall: `a_i B = (i mod g) B + g floor(c/g) B`, one baby key and one giant key of the
  matvec.
- Wide: `a_i B = k2 B + (-g (k1 + 1)) B`, a baby key and a negative giant key of the gradient. For
  `k2 = 0`, `a_i B = -g k1 B` is one gradient key. Combinations with fold keys cover the rest.

This takes 2 rotations instead of 1 for most entries, and 0 new keys. A breadth-first search over
the existing keys gives these transpose totals:

| network | new keys | rotations per transpose |
|---|---|---|
| Iris 4→8→2 | 1 → 0 | 1 → 2 |
| Iris 4→8→3 | 3 → 0 | 3 → 5 |
| eMLP-1 W2 | 14 → 0 (−4.2 GB) | 15 → 27 |

The transpose runs once per update, so the extra rotations are small next to the per-update
total. Default in `rotations_for_transpose`: one rotation per entry (fewest key switches,
matching the cost model). Tasks 27, 30 and 32 should pick the two-rotation form when key memory
binds (eMLP-1 on n16_train). The key-switch delta is `+12` per update for eMLP-1.

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

**The mask costs exactly one level.**
- The constant `c` does not need it. `c = lr / n_valid` (SGD) or `(1 + mu) lr / n_valid` (NAG)
  folds for free into the exact-scale plaintext of the top derivative factor
  `s_L = z_L * (2 c valid)`. Better, it can go into that plaintext's target scale, so the message
  magnitude, and therefore the precision, is unchanged. That plaintext is needed anyway and sits
  off the critical path.
- Every backward op is linear in `dz`, so all `dz_l`, all `dW_l` and all bias gradients carry `c`.
  Each layer's `s_l` plaintext is a free scale knob for matching `scale(W_l)` exactly at the update.
  The bias gradient can also take its own `mul_scalar` off the critical path.
- `sim_train_step_sgd` puts `c` into `mask0` instead. Both placements give the same values and the
  same count of 1 ct-pt per diagonal.
- So step 4 exists only for the lane-0 mask, which costs **+1 level**.

**It is unavoidable without extra products.** Rotations, additions and scalar (constant)
multiplications only produce circulant maps, since they commute with `rot(·, 1)`. "Sum the `B`
lanes of each feature and write the sum to every lane" does not commute with `rot(·, 1)` in the
feature-major layout, so any circuit of rotations, additions and constants needs a non-constant
plaintext. One way around it spends products instead of the level: mask the lane-rotated copies
of `A` on the input side, off the critical path, and sum the lane-shifted products. That costs
about `B` times more tensors and relins per diagonal. It is not used.

**Reduce-only variant (cost model `interactive=True`).** Stop after step 3, with `c` folded into
`s_L` as above. The server updates `W` with a gradient that is valid only in lane 0, and the
other lanes carry partial sums. The client refresh then has to read lane 0 and re-replicate the
weights.
- It saves `m log B` rotations, `m` ct-pt and **one level** per layer and update.
- But the refresh is no longer a pure decrypt/re-encrypt. M3's bootstrap refresh could not do it,
  and Task 30's `Refresher::refresh(ct)` does not have that signature.
- `sim_batch_reduce` alone is this variant.

| gradient variant | rotations per layer per update | levels on the gradient path | training-step depth |
|---|---|---|---|
| replicated (default, steps 1–5) | `(baby-1) + (giant-1) + 2 m log B` | 2 | `4L + 1` |
| reduce-only (lane 0, re-laying client refresh) | `(baby-1) + (giant-1) + m log B` | 1 | `4L` |

**Default: steps 1–5 (replicated `dW`, pure refresh).** Task 30 and Task 32 should revisit this
if the level margin or RQ1's rotation count matters more than a pure refresh.

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
| batch-sum lane-0 `mask0` | step 4 of §3 | unavoidable for a replicated `dW` without about `B` times more products: **+1 level** (the constant `c` folds for free into `s_L`). The reduce-only variant has no mask |
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
| batch sum with `mask0` (replicated variant only; `c` folded into `s_L` or here) | 1 (0 reduce-only) |
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

**Verdict for seal_2020.**
- **Replicated design (default):** exactly 9 levels (`4L + 1`), so no client refresh is needed
  inside the iteration. Task 30 refreshes all weights after the update, as specified (W1' is at
  level 0). The margin is **zero**.
- **Reduce-only variant:** 8 levels (`4L`), so W1' ends at level 1 with one level of margin. The
  cost is a client refresh that re-lays lane 0.

In the replicated design, one extra multiplication on the layer-1 gradient path pushes it to 10
levels. That happens with any of the following:
- a separate `lr` multiply (Task 28's literal API: `weight_grad` returns `dW / batch`, then
  `Sgd::step` calls `mul_scalar(lr)`);
- a separate `1/batch`;
- NAG done as a multiply on the gradient.

So the constants must be folded (ruling item 2). The gradient constant `c` goes into `s_L` (or
into the target scale of its plaintext), or into the `mask0` plaintext of step 4, as the simulator
does. NAG (`g = G/n + decay W`, `V' = mu V + g`, `W' = W - lr(1+mu) g - mu² lr V`)
is rewritten with the scaled velocity `U = lr(1+mu)V`:
- `h = lr(1+mu)/n * G`, with the constant folded into `s_L` or `mask0`, plus `lr(1+mu) decay * W` (a scalar on the fresh `W`);
- `U' = mu U + h`;
- `W' = W - h - (mu²/(1+mu)) U`.

**Weight decay is deferred.** The implemented optimizers (`hekit_nn::optim`, Task 28) have no
`decay` term: `Sgd` and `Nag` use `g = G/n` only (as if `decay = 0`). The `lr(1+mu) decay * W`
term above is a `mul_scalar` on the fresh `W` (one level off the critical path, no new keys), so
it fits the same budget when added. Neither the 2020 paper (Iris, SGD) nor Task 31 needs it;
ReBoot's weight decay (stated only in its RUN.md, `docs/m0-report.md`) is not reproduced until it
is added. The biases take plain SGD steps under NAG (no bias momentum).

The only multiplications by `mu` act on fresh `U` and `W`, off the critical path, so NAG also fits
in 9. With that folding, **`seal_2020_l10` is not needed** for the Iris network, including
Task 31's batch 20 (32 lanes, `valid` mask, same depth). If Task 28 keeps an unfused API, the
depth is 10: use `seal_2020_l10` for Task 31's batch-20 run only, or refresh `dz1` once before the
layer-1 gradient.

General depth for an `L`-layer MLP with one-level activations: `2L` (forward) `+ 1` (the top `dz`)
`+ 2(L-1)` (`W^T` and `dz` per lower layer) `+ 2` (outer product and lane-0 mask) `= 4L + 1`.
The reduce-only variant drops the mask level and needs `4L`.

**eMLP-1, 784 → 32 → 10, z² + z, n16_train (`max_level` 31).** `d = 1024` and `B <= 32` lanes. The
structure is the same as above with `L = 2`: depth 9 when constants are folded (8 reduce-only),
10 or 11 unfused. (With the linear output layer of `reboot_emlp1.json`, as implemented in Task 32
with `activation::Identity`, the measured trace uses 7 levels: the output activation and its `dz`
cost none; `docs/m2-report.md` §4.) All are far below 31, so there is **no mid-iteration refresh**. That leaves room for about 5 more layers at 4 levels
each.

## 5. Differences from the M0 spike (`spikes/costmodel/slotsim.py`) and the cost-model change Task 32 needs

The layout, the diagonals `D_c`, the pre-rotation, the fold, the free replicate, the transpose
formula `a_i` and the transpose count `m - 1` are the spike's (`diag_batched`). They are ported to
Rust as is. Differences:

1. **Batch sum and replication.** The spike's `interactive=True` (the variant Task 32 compares
   against) reduces only. Its gradient costs `m log B'` rotations, with no mask and no replication,
   and the client re-encodes from lane 0. This design replicates on the server: `+ m log B`
   rotations, `+ m` ct-pt and **+1 level** (the lane-0 mask) per layer per update. The step depth
   is `4L + 1` instead of `4L`, which is 9 vs 8 on seal_2020. Task 32's cost model must add
   `m * log2(B')` rotations to the gradient phase in interactive mode and note the extra level.
   Alternatively, Task 28 can adopt the reduce-only variant of §3 together with a re-laying
   refresh.
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

5. **Galois keys.** The cost model counts key switches, not keys. The rotation-only transpose
   needs up to `m - 1` extra keys per transposed layer (14 keys ≈ 4.2 GB for eMLP-1 on n16_train,
   §2.3). The zero-new-key form costs `+12` transpose rotations per update for eMLP-1. Task 32
   should report which form it measured.

None of these changes touches the forward, transpose or backward counts. With 1 and 2 applied, the cost
model's interactive gradient phase is `(n2-1) + (n1-1) + 2 m log2(B')` rotations and `m` relins,
which equals `sim_outer_batchsum`. The spike itself is not edited (project rule).

## 6. Compact re-packing for bootstrapping (M3, Task 40)

Bootstrapping refreshes a whole ciphertext, so the cost of a non-interactive update is the
number of ciphertexts that hold the weights. The `m` diagonals of §2 are each replicated over
`S/L` positions (`S = slots`, `L = max(n_out, n_in)`): `B` lanes times the `S / (L B)` copies of
the `L`-periodic feature index. One copy of each is enough. The compact form puts
`cap = S / L` diagonals into one ciphertext, so a layer needs `ceil(m / cap)` bootstraps
instead of `m`. This is premise 1 of the M0 cost model (`spikes/costmodel/costmodel.py`,
`_diag_batched`, `compact=True`). `crates/hekit-nn/src/compact.rs` implements this section.

### 6.1 Plan and offsets

```
CompactPlan { d_out, d_in, batch = B, diags_per_ct = cap = S / L, lane_shift = log B }
n_ct = ceil(m / cap)                   ciphertexts (= bootstraps) per layer
diagonal c (entry index, BSGS order) -> ciphertext i = c div cap, position t = c mod cap
o(t) = (t mod B) + (t >> lane_shift) * L * B          the slot offset of position t
```

The low `log B` bits of `t` choose a lane, the higher bits a copy of the `L`-period. Compact
ciphertext `i` holds, for `j < L` and `t < n_i` (`n_i = min(cap, m - i cap)` diagonals):

```
slot o(t) + j B   holds   P_{i cap + t}[j]        (0 in every other slot)
```

The positions `{o(t) + j B}` of different `t` are disjoint and cover all `S` slots when
`n_i = cap`. The entries `P_c` are kept pre-rotated (§2), and the re-packing treats each one as an
opaque `L`-periodic feature vector, so the pre-rotation passes through unchanged.

`o(t)` is linear in the bits of `t`: `o(t) = sum_k bit_k(t) * step_k` with

```
step_k = 2^k                         for k < log B           (lane bits)
step_k = L B 2^(k - log B)           for log B <= k < log cap (copy bits)
```

So the shifts can be done as a binary tree over the bits of `t`, with one rotation per merge.

### 6.2 Pack (consumes 1 level, 0 relins)

Let `mask0[s] = [s < L B and s mod B == 0]`: lane 0 of the first `L`-period (`L` ones). For each
compact ciphertext `i`:

1. `x_t = rescale(mul_plain_to_scale(P_{i cap + t}, mask0, scale(P)))` for `t < n_i`
   (1 ct-pt and 1 rescale per diagonal). This is exactly `scale(P)` on level `l - 1`.
2. A tree over the bits of `t`, from bit 0 up: merge the groups `{bit_k(t) = 0}` and
   `{bit_k(t) = 1}` (equal higher bits) as `left + rot(right, -step_k)`. A group without a
   right half passes up unchanged.

The result is `sum_t rot(x_t, -o(t))`, the compact ciphertext of §6.1. Every merge joins two
nonempty groups, so `n_i - 1` rotations per ciphertext and **`m - n_ct` rotations** per layer.
That is the cost model's "1 lane shift per diagonal into compact ciphertexts (m - nW
rotations)". The tree uses only the `log2(min(m, cap))` steps `-step_k`, not `cap - 1` distinct
shifts.

The mask reads only lane 0 of the first period (slots `j B`, `j < L`). So pack also accepts an
input that is valid only there, e.g. the reduce-only gradient of §3 (the cost model packs the
reduced gradient, sharing this mask with the batch-sum mask). With a replicated input the block
copies and lanes it drops can carry any drift: unpack rebuilds them from one copy (§6.4).

### 6.3 Unpack (consumes 1 level, 0 relins)

For each compact ciphertext `i` (on level `l'`, e.g. the bootstrap output):

1. A tree over the bits of `t`, from the top bit down. The group with offset base `a` holds
   `rot(packed_i, a)`. It splits into the half with `bit_k = 0` (the same ciphertext) and the half
   with `bit_k = 1`, which is `rot(·, +step_k)` and exists only if some `t < n_i` lies in it.
   Leaf `t` is `rot(packed_i, o(t))`, which has `P_{i cap + t}[j]` on slot `j B`. That is
   `n_i - 1` rotations.
2. `y_t = rescale(mul_plain_to_scale(leaf_t, mask0, scale(packed)))` (1 ct-pt, 1 level). Only
   slots `j B`, `j < L`, survive.
3. Replicate: `for k < log cap: y = y + rot(y, -step_k)`. The lane bits fill the `B` lanes and the
   copy bits fill the `S / (L B)` copies. After these `log2(S / L)` rotations slot
   `j B + b + p L B` holds `P[j]` for every lane `b` and copy `p`: the replicated, pre-rotated
   diagonal of §2.

Rotations: **`(m - n_ct) + m log2(S / L)`**, ct-pt `m`, relins 0, levels 1. This is the cost
model's unpack, "per diagonal 1 mask + 1 lane shift (m - nW rotations) + log2(S/L) replication
rotations". The replication writes every lane and every block copy from one slot. So the output
is block-periodic by construction, which is what `BootstrapRefresher` cannot provide for a
replicated layout (ruling R-M3-3; its `bind_layout` still refuses such layouts).

### 6.4 Exact scale, levels and keys

- Both masks are encoded at `target_plain_scale(ct, scale(ct))` on the ciphertext's level
  (§3.1: a 0/1 mask cannot be encoded at scale 1), so each rescale returns **exactly** the input
  scale. Rotations and additions keep scales. Pack output: level `l - 1` at `scale(P)`. Unpack
  output: level `l' - 1` at `scale(packed)`.
- A round trip pack, then bootstrap (to `levels_after_boot`, 16 on n16_boot, at `Delta`), then
  unpack costs `+2` levels per update. Weights refreshed this way arrive on level `16 - 1 = 15`
  of n16_boot: `Refresher::output_level` for the bootstrap minus the unpack level.
- **The fused path (Task 41, ruling R-M3-11): `compact_shift`.** `compact_pack(EncWeights)` is
  the mask of §6.2 followed by the shift tree. Composed literally with the replicated gradient of
  §3 (`tensor → mask0 → replicate → W − dW → pack mask → boot → unpack`) it costs, against the
  cost model, **one more level per update**, `m log2 B` replication rotations, `m` more ct-pt
  and, for NAG, a pack and an unpack of the velocity too (≈ +650 KS per eMLP-1 update). The
  model's path never builds an expanded gradient: the reduce-only gradient (§3, steps 1–3) is
  masked once by the **first-period** lane-0 mask `mask0_L[s] = [s < L B and s mod B = 0]`, with
  the constant `c = grad_scale / n` folded in (`Dense::weight_grad_reduced`, 1 level), and that
  mask is also the pack mask. `compact_shift(ev, entries, plan)` is the shift tree alone: its
  precondition is that every entry is zero outside `{j B : j < L}`, and it costs exactly
  `m - n_ct` rotations, no ct-pt, **no level**, no relin (keys: `-step_k`,
  `k < ceil(log2 min(m, cap))`). The weights and the velocity then stay in compact form
  between steps (the optimizer runs on the compact ciphertexts, §7). Pack (with its mask) is
  only for a state that starts expanded. `compact_extract(ev, ct, plan, t, target)` is the
  unpack of one position (§7.3): `rot(ct, o(t))`, `mask0_L` at an exact target scale, and the
  `log2(cap)` replications (`[o(t) != 0] + log2(cap)` rotations, 1 ct-pt, 1 level).
- Galois steps (`compact_rotation_steps`), one key per distinct element:
  `{-step_k : k < log cap}` (pack tree and replication) and `{+step_k : k < log2 min(m, cap)}`
  (unpack tree). That is `log2(S/L) + log2(min(m, S/L))` keys. In a single-block layout
  (`d B = S`) with `L = d`, `cap = B` and all steps are lane steps `±2^k`: exactly the batch
  reduce/replicate keys the training step already has (§3), so **no new keys**.

| layer (preset) | S | L, m | B | cap | n_ct = boots | pack rot | unpack rot | keys |
|---|---|---|---|---|---|---|---|---|
| 32 x 1024 (n16_boot) | 2^15 | 1024, 32 | 32 | 32 | 1 | 31 | 31 + 32·5 = 191 | 10 |
| eMLP-1 W1 32 x 784 (n16_boot) | 2^15 | 1024, 32 | 32 | 32 | 1 | 31 | 191 | 10 |
| eMLP-1 W2 10 x 32 (n16_boot, d = 1024) | 2^15 | 32, 16 | 32 | 1024 | 1 | 15 | 15 + 16·10 = 175 | 14 |

The rotation counts and `n_ct` are the cost model's `gradient` lane shifts (`194 - 163 = 31`
against `interactive=True`), its `update` rotations (191 and 175) and its `weight_ciphertexts`,
for the same `(d_in, d_out, batch, slots)`. The test `compact_round_trip_n16_boot` checks the
first row on the real n16_boot preset.

## 7. Non-interactive training (M3, Task 41)

`crates/hekit-nn/src/trainer_ni.rs` (`NonInteractiveTrainer`) trains with no client in the
loop: every level is restored by bootstrapping (`docs/bootstrap.md` §11). The trainer holds an
`Evaluator` (relinearization and Galois keys) and a `Bootstrapper` (its evaluation keys), and no
`Decryptor`: the type has no field that can hold a secret key, and its tests build the
`Decryptor` only after training.

### 7.1 One update (ruling R-M3-11, the cost model's compact path)

`k` micro-batch ciphertexts of `B` lanes (ruling R-M3-4: `B = 64` is `k = 2` ciphertexts of 32
lanes, `B = 48` is not a power of two and is not run):

| step | per | ops | levels |
|---|---|---|---|
| forward, loss gradient, backward (as §4; `W^T` once per update) | micro-batch | as interactive | as §4 |
| weight gradient `Dense::weight_grad_reduced`: babies `(b-1)` + giants `(g-1)` (hoisted), `m` tensors **summed over the `k` micro-batches before one relin per diagonal**, reduce `m log2 B`, `mask0_L` with `c` folded | update | `k(b+g-2) + m log2 B` rot, `m` relin, `m` ct-pt | 2 |
| `compact_shift` into the compact layout (§6.4) | update | `m - n_ct` rot | 0 |
| bias gradient `Dense::bias_grad_reduced`: the `k` `dz` summed, reduce, `mask0_{L_host}` with `lr / n`; shift to its position | update | `log2 B + [o(t) != 0]` rot, 1 ct-pt | 1 |
| optimizer on the compact state (NAG on `W`, `U`; SGD on the biases, §7.3) | update | `2 n_ct` scalar products | 0 on the critical path |
| bootstrap every compact `W` and `U` (§7.2 says when) | update | `2 n_ct` per layer (+1 per bias-only container) | → `levels_after_boot` |
| `compact_unpack` of `W` only | update | `(m - n_ct) + m log2(S/L)` rot, `m` ct-pt | 1 |
| `compact_extract` of every bias, at the exact product scale of its next forward | update | `[o(t) != 0] + log2(S/L_host)` rot, 1 ct-pt | 1 |

The NAG update on a compact container `i` of layer `l` is, at level
`lv = min(W_i, h_i, h_b of the biases it hosts, U_i - 1)`:
`W_i' = W_i - h_i - sum h_b - (mu^2 / (1 + mu)) U_i` and `U_i' = mu U_i + h_i` (`U = 0` before
the first step: `U' = h`, `W' = W - h - sum h_b`). The velocity's bias positions stay zero, so
the biases take plain SGD steps (the M2 deviation).

### 7.2 Placements (`BootPlacement`)

- **`AfterUpdate`** (the cost model's): every compact `W` and `U` is bootstrapped after every
  update. The next step starts from `levels_after_boot - 1` (15 on n16_boot) after the unpack.
- **`AfterActivation`**: in every forward, each hidden activation output `a_l` (`l < L`, each
  micro-batch ciphertext) is multiplied by 1 onto exactly `Delta` (1 ct-pt, 1 level, so the next
  layer's bias is at `Delta^2`), dropped to level 0 and bootstrapped; the next layer and its
  weight gradient use the bootstrapped `a_l`. The weights are bootstrapped **only when the next
  step would not fit** their remaining levels (the structural level arithmetic of §7.4, run on
  the state the update leaves). Cost: `(L - 1) k` activation bootstraps per update.
  **Precondition: `|a_l| <= 1` in every slot** (the bootstrap's EvalMod range). Larger
  activations are not clipped: the bootstrap runs and its error grows (quadratically in `|m|`
  above 1; Task 38 measured 17.38 bits absolute at `|m| = 4` against 17.46 at `|m| <= 1`,
  `docs/bootstrap.md` §11.4). The range is the caller's: the `mnist` and `iris` benches record
  the largest bootstrapped activation the plaintext twin sees, decrypt the bootstrapped
  activations of every update on the client side (`last_activation_bootstraps`) and report
  their error against the twin, and flag any input above 1.
- **`WhenNeeded`** (not in the brief): the same lazy weight schedule without the activation
  bootstraps.

**The lazy decision and partial batches.** The weights skip a bootstrap only if the *next*
step fits. A partial batch costs one more level at a linear output layer (the valid-lane mask
on `dz`), so the decision needs the next step's fullness: `train_epoch_valid` declares it for
every update but the epoch's last (`set_next_batch_full`), and an undeclared next step is
planned as partial (conservative: an update that could have skipped bootstraps instead).
`boot_schedule` follows the same rule (one function, `must_boot`).

For eMLP-1 on n16_boot: a step takes 7 levels from the unpacked weights (5 with activation
bootstraps), and 15 are left after a bootstrap and an unpack. So `WhenNeeded` bootstraps the
weights every second update (`15 → 8`, unpack `→ 7`, `→ 0`), and so does `AfterActivation`
(`15 → 10`, `9 → 4`; a third step would need `3 → -2`). **`AfterActivation` does not bootstrap the
weights less often than `WhenNeeded` for eMLP-1** and pays one activation bootstrap per
micro-batch on top. It halves the weight bootstraps only against `AfterUpdate`. Its real gain is
depth: the forward of the upper layers restarts at every activation, so a deep network that does
not fit after a bootstrap may fit with it (the depth-overflow configuration of §7.4: 25 levels
with `AfterUpdate`, 13 with `AfterActivation`). The backward still descends through every layer
(2 levels per layer).

### 7.3 The compact state and the biases (`CompactLayout`, `CompactState`)

- Per layer its `CompactPlan` and `n_ct` compact ciphertexts, all at exactly `Delta`. The client
  encrypts the initial state (`NonInteractiveTrainer::encrypt_network`): the compact containers
  at `levels_after_boot` (as a bootstrap leaves them), zero velocity, and the expanded `Mlp` at
  `levels_after_boot - 1` with every bias at the exact product scale of its first forward (Task
  40 carry-in: the chain starts at the refresh level, `Mlp::encrypt_at`, `Mlp::batch_spec_at`).
- **Biases ride in the weight containers.** A bias of padded length `n` takes one free position
  `t` of a container whose period `L >= n` (canonical form: slot `o(t) + j B` holds
  `b[j mod n]`, `j < L`). Rule: containers with the largest `L` first (fewest extraction
  rotations), a free power-of-two position first (its offset is a tree step, whose keys exist),
  else the lowest free one. eMLP-1: W1's container is full (32 of 32 positions); both biases go
  to W2's container (`L = 32`, 16 of 1024 positions used) at `t = 16` (`o = 16`) and `t = 32`
  (`o = 1024`): **no extra bootstrap and no new key**. A bias that finds no room gets a
  bias-only container (`L = layout.d`, one more bootstrap per update, SGD only).
- The bias gradient is masked by `mask0_{L_host}` (first period of the host's `L`) at exactly
  `Delta`, then shifted by `-o(t)`. The bias for the next forward is `compact_extract` at the
  exact product scale `s_x Delta` of that forward's input (`Delta^2` for the first layer and
  after an activation bootstrap; the activation chain's scale otherwise). When no rescale from
  the container's level reaches that scale (`ScaleMismatch`, about 11% of arbitrary targets),
  the extraction is retried one level lower, but never below the level its forward runs at, so
  the levels of §7.4 do not change.
- Bootstrapping refreshes every slot independently, so a replicated layout (`d B < S`) keeps
  drifting copies under `BootstrapRefresher` (ruling R-M3-3). The compact path re-lays them by
  construction: extraction and unpack write every lane and block copy from one slot. Tested on a
  4-copy layout (block spread 2^-31.5 after 3 updates). The activation bootstraps of
  `AfterActivation` act on transient values recomputed every step, so their per-copy noise does
  not accumulate.

### 7.4 Levels (`levels_per_iter`)

`levels_per_iter(mlp, placement)` is the exact level count of one step from the unpacked
weights (all on one level `w`) to the lowest level of the step: the updated compact state. It is
the same op-by-op level arithmetic the trainer runs (`z = in - 1`, `a = z - depth`, `s = z -
derivative_depth`, `dz = min(s, delta) - 1` or `delta` for a linear layer on a full batch,
`delta_{l-1} = dz - 1`, `h_l = min(dz_l, a_{l-1}) - 2`, `h_b = dz - 1`, an activation bootstrap
restarts at `levels_after_boot`), and the tests compare every value's level with the trainer's
trace. `NonInteractiveTrainer::new` returns `InvalidParams` when `levels_per_iter >
levels_after_boot - 1` (Review Focus 3).

eMLP-1, `AfterUpdate`, from 15: `z1 14, a1 13, z2 12, a2 = delta2 = dz2 12, delta1 11, s1 13,
dz1 10, h1 8, hb1 9, h2 10, hb2 11`; W1's container lands on 8, W2's (which hosts both biases) on
9. **7 levels**, against `4L + 1 = 9` for two quadratic layers (the linear output layer saves the
activation and the `dz` product). With the unpack, 8 levels per update: the M0 premise "+2
levels per update" (mask and unpack) over the 6 of the mask-free interactive step.

| network | `AfterUpdate` / `WhenNeeded` | `AfterActivation` |
|---|---|---|
| eMLP-1 (z² + z, linear) | 7 | 5 |
| 16 → 16 → 4, z² + z twice | 9 | 7 |
| degree-4 polynomial × 4 layers (the depth-overflow configuration) | **25 > 15: refused** | 13 |

### 7.5 Counts against the cost model

`predict_update_for` is the closed form of one update, op for op the trainer's (the tests assert
the measured counter equal to it, bootstraps included, on every update). Against
`network_cost("diag_batched", …, compact=True)` (ported in `crates/hekit-bench/src/bin/mnist.rs`,
`cost_model_ni`, and checked against the spike's numbers), with Task 32's terms where they apply:

| eMLP-1, n16_boot, `AfterUpdate`, Reuse transpose | B = 32 (k = 1) | B = 64 (k = 2) | B = 48 (formula only, k = 2) |
|---|---|---|---|
| model: rot + relin / bootstraps per update | 703 + 58 / 4 | 739 + 68 / 4 | 739 + 68 / 4 |
| + recomputed babies `k (n1 - 1)` per layer (Task 32 §5.2) | +14 | +28 | +28 |
| + BSGS split, gradient `(b-1)+(g-1) - (n1-1)-(n2-1)` (matvec KS equal) | −2 | −4 | −4 |
| + Reuse transpose (27 against `m - 1 = 15`) | +12 | +12 | +12 |
| + bias gradients (`log2 B + 1` per layer) and extraction (`1 + log2(S / 32)` per bias), not in the model | +34 | +34 | +34 |
| = closed form, KS without bootstraps | 819 | 877 | 877 |
| KS per sample with 4 × 111 per update | **39.47** (model 37.66, ×1.048) | **20.64** (model 19.55, ×1.056) | 27.52 (model 26.06) |

The interactive corrections of Task 32 that do not apply here: the replication `m log2 B` (the
gradient is reduce-only) and the bias's replicate half (`log2 B` instead of `2 log2 B`). The lazy
placements change only the bootstrap term: `WhenNeeded` 2 bootstraps per update on average
(32.53 KS/sample at B = 32, 17.17 at B = 64), `AfterActivation` 2 + `k` (36.00 at B = 32, 20.64 at
B = 64).

### 7.6 Keys (eMLP-1, n16_boot)

`ni_rotation_steps`: 35 training Galois keys with the Reuse transpose (forward, reduce-only
gradient, backward, transpose, the compact pack/unpack trees and the bias offsets), plus the
relinearization key: ≈ 10.9 GB at 0.302 GB each, next to the bootstrapper's 10.27 GB of keys and
≈ 3 GB of encoded CtS/StC diagonals.
