# Block profile composition (D3b core)

This slice composes the reviewed D2c2 interactive weighted step and D3a AA boundary
contract. It does not introduce a new loss, optimizer, packing layout, or NI step.
Local blocks retain their own heads and targets; inference uses only each block's
first activation and the separate output stack.

## Reason and API

For microbatches i with n_i valid lanes, each stage uses N = sum_i n_i and gradient
`g_s = (1/N) sum_i sum_valid gradient(loss_s)`. Each stage performs one optimizer
update and one state/weight refresh pass for that logical batch. Frozen biases are
configured on every child trainer before block assembly, including the output stack.
For scheduled NAG, `V' = mu V + g` and `W' = W - lr (g + mu V')`; the existing
fixed reference-rate representation and dual-gradient calculation remain in Trainer.
Configure NAG reference maxima before boxing the interactive optimizers.

`BlockTrainer::train_step_valid(net, &[(x, stage_targets, n_valid)])` accepts a
borrowed slice in microbatch order. Nonempty input, a checked total sample count,
`0 < n_valid <= layout.batch`, stage/trainer counts, and every part's target count
are checked before evaluator work. Each stage calls
`Trainer::train_step_valid_with_hidden` once over all parts. Its returned
`hidden[part][0]` uses pre-update weights. At each boundary and for each part it is
converted to Delta and refreshed exactly once, retaining the part's count and target
references. The output stack is never handed off. `train_batch_valid` is the singleton
wrapper, preserving existing callers and arithmetic order. Additional boundary work
is `(stages - 1) * parts` ct-pt products, rescales and client refreshes, with no key
switches at the boundary.

`Trainer::validate_lr` validates every layer read-only; a crate-private infallible
commit helper applies a validated rate. `BlockTrainer::set_lr` first validates every
stage, then commits every stage. Unsupported custom optimizers keep their existing
reject-by-default contract. NI rate and reference setters are similarly split into
read-only validation and crate-private infallible commit, retaining their existing
error messages and accepted values. Block NI setters validate all children first.
Rejected configuration changes perform no evaluator work and change no rates,
references, weights, state, counters, traces, or refresh counts.

This atomicity guarantee applies to configuration setters only. A training error in
a later stage can leave earlier stages updated; individual optimizer and refresh
errors retain Trainer's existing partial-update limitations. There is no whole-model
training transaction. NI math and boundary behavior are unchanged: AA reuses the
already canonical first activation bootstrap without another Delta conversion or
bootstrap, while WN/AU retain their explicit boundary bootstrap.

## Ownership and evidence gates

Owned production edits are limited to trainer rate helpers, NI rate/reference
helpers, and block interactive composition/atomic setters. Tests use original tiny
synthetic CPU graphs and insecure test parameters, not a full encrypted eMLP
campaign. The independent plaintext oracle evaluates each stage's inputs from old
weights and differentiates only that stage's local loss. Four scheduled updates,
unequal 8+4 parts with dirty inactive lanes, every logical weight/bias, boundary
values, full operation counters, singleton compatibility, and rejected updates are
checked. The existing AA tests remain unchanged.

Packed NN work, benchmark CPU/CUDA bridges, CLI wiring, scientific/full-block
runs, GPU timing, and Task 54 completion remain separate gates. No performance,
production security, full-profile coverage, or noninteractive speed claim follows
from these synthetic tests.
