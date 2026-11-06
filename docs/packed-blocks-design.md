# Packed local blocks and forward-only inference — Task54 D4c3 (revision 2 / FIX1)

2026-11-07. **Root-ratified design only; independent SPEC and QUALITY design reviews passed.**
This document grants no implementation or execution permission. Design preparation used only file inspection and ordinary scalar calculations; this package makes no production-source changes and has run no key generation, HE, dataset campaign, GPU or remote work.
The existing c2 single-stage result is not block evidence. The full 21-profile ×
10-seed × CPU/GPU × I/NI/ablation matrix and user-held real-dataset block gate remain.

Original revision1's five artifacts and full new-file patch are preserved byte-for-
byte in `task-54-d4c3-design-original-freeze-archive/`. FIX1 addresses independent
review F1/F2/M1 only, under root rulings recorded in progress.md before this edit.
Training arrays/rates/seeds/presets/E/epsilon and training bounds are unchanged.

## 1. Ownership and necessary small shared extensions

Proposed future c3 ownership: new `crates/ckks-train-nn/src/packed_training/block.rs`,
narrow forward-only additions in `stage.rs`, exports in `mod.rs`, new
`tests/packed_blocks.rs` and private `tests/packed_blocks_helpers/`. The D6c2a bench
agent remains sole production implementer until root assigns otherwise. No legacy
trainer, optimizer, loss, preset, profile binder, bench or CUDA source ownership.

Root permits these two exact future ownership extensions **only after design
ratification and explicit source assignment**; no source edits are currently authorized:

1. `refresh.rs`: change only `PackedRefreshRequest::validate_response` from private
   to `pub(super)`, retaining its body. The I hidden-boundary path needs its existing
   private request-token equality check plus output validation. Reading a response's
   public ciphertext and rewrapping it would allow cross-request replay. Alternative:
   add a module-private `finish_hidden_response` in the same file which calls that
   method and returns the vector. No public token or packet constructor expansion.
2. `dag.rs`: expose a narrow module-private `ForwardPlanner<'p>` adapter over existing
   `Planner<'p>`, with `new(&'p CkksParams, &'p Encoder)->Self`,
   `set_sample(Option<usize>)`, `record(&str,History)->History`,
   `affine(&str,History,History,&LayerGeometry,bool)->Result<History>`,
   `bias(&str,History,Option<History>)->Result<History>`,
   `activation(&str,History,VectorSpec,bool)->Result<History>`, and
   `finish(self)->(Vec<PackedTraceEntry>,OpCounter)`. Each delegates unchanged.
   Keep `product`, `mask`, `derivative` internal and all
   recurrence bodies unchanged. An internal forward helper in stage.rs calls this
   adapter, while its runtime counterpart calls existing stage `Runner` methods.
   Alternative: put the small forward-only traversal in dag.rs behind one
   `pub(super) plan_forward_paths` function. Either choice needs shared ownership;
   copying the full c2 planner or using a training DAG as inference is rejected.

No trainer.rs extension is needed: existing module-private `bootstrap_hidden`
already performs canonical lowering, output/contract/count validation and encrypted
event capture. Existing `refresh::counter_delta`, `dag::checked_counts` and existing
layer generation accessors are reusable within packed_training.

## 2. Immutable graph payload and transaction API

Names below are proposed concrete signatures, not present APIs. C is HeCiphertext +
Clone; E is HeOps<Ct=C>. All public structs have private fields, immutable accessors,
and checked constructors; no public mutable payload, token, plan or stage vector.

```rust,ignore
PackedBlockGraph::new(ev, geometry: PackedGeometryPlan,
                     stages: Vec<PackedStage<C>>) -> Result<Self>;
graph.stages() -> &[PackedStage<C>];
graph.geometry() -> &PackedGeometryPlan;

PackedGraphRecord::new(id: usize, input: PackedVector<C>,
                      targets: Vec<PackedVector<C>>, valid: bool) -> Self;
PackedGraphBatch::new(ev, &graph, records: Vec<PackedGraphRecord<C>>,
                     partitions: Vec<PackedPartition>) -> Result<Self>;
PackedPartition::new(start: usize, end: usize, declared_valid: usize) -> Self;
batch.records() -> &[PackedGraphRecord<C>];
batch.valid_ids() -> &[usize];
batch.valid_count() -> usize;

PackedGraphNext::Known(PackedGraphBatchMeta, PackedUpdateMeta)
PackedGraphNext::Unknown
PackedGraphNext::Terminal

PackedBlockNiTrainer::new(&E, &Encoder, &B, PackedNiPlacement) -> Result<Self>;
server.plan_update(&graph, &batch, rate, relin, next) -> Result<PackedGraphNiPlan>;
server.prepare_planned(&graph, &batch, &plan) -> PackedGraphNiResult<C>;
server.prepare_update(&graph, &batch, rate, relin, next) -> PackedGraphNiResult<C>;
// Result = Result<(PackedGraphCandidate<C>, PackedGraphNiReceipt<C>),
//                 PackedGraphNiFailure<C>>

PackedBlockInteractiveTrainer::new(&E, &Encoder) -> Result<Self>;
server.prepare_update(&graph, &batch, rate, relin, &mut client)
    -> PackedGraphInteractiveResult<C>;

candidate.graph() -> &PackedBlockGraph<C>;
candidate.into_graph() -> PackedBlockGraph<C>; // single infallible owned transfer
```

Graph construction checks every ordered stage against geometry's exact layer pair,
ordinals 2s/2s+1, common lattice/context, target model width/real classes/config,
W/U/bias representation and domain, and equal completed indices across all stages.
It checks backbone output spec equals the next backbone input spec. It does not
accept a local head output as an adjacency edge. No re-packing conversion is implied.
The graph owns a private process-local Arc identity, renewed in the candidate.
Existing layer generation tokens remain the only stage-update freshness mechanism.

Record identity is the original physical position: constructor requires
`record.id == position`, preserving c2's usize IDs and valid holes. Future external
dataset IDs can be separately recorded by the binder; they cannot silently replace
the c2 position identity. Batch partitions must be ordered contiguous nonempty
physical ranges covering 0..records.len exactly once, with each declared valid count
equal to its actual flags. Their sum must be positive and equal to the unique valid
ID count. Empty-valid partitions are allowed; empty physical ranges are rejected.
Every record has exactly graph.len targets. Validate *all* target and initial-input
contexts, two components, finite exact Delta, levels, vector specs and widths,
including invalid payloads, before HE. Invalid payloads then have zero arithmetic,
zero denominator, zero snapshot and zero refresh contribution.

Checked batch owns a private Arc identity and immutable encrypted payloads. A graph
plan retains that token and graph token plus c2 source generation-bound stage plans;
`prepare_planned` requires the same batch payload identity, not merely equal levels.
It also rechecks current generations/config/capability and exact metadata. This
strengthens c2 metadata binding at the composition boundary without changing c2.

Child record adapters retain the original physical length/order. For invalid
downstream positions only, clone the first valid handoff into an inert input
placeholder, keep that position's encrypted target and `valid=false`. This is no
HE work and does not claim a computed hidden for an invalid sample. The first valid
handoff exists because total n>0. Never compact/reindex valid records. An alternative
indexed-record stage API would require broader shared ownership and is unnecessary.

### Complete preflight and honest future metadata

Before the first current HE operation, validate the whole graph/batch, all target
representations, partitions, rates, key union and every current stage DAG. Extract
each actual old stage's metadata with existing generation tokens. For current stage0
use actual input histories; for current stages s>0 use the fixed boundary contract's
CtMeta for every valid input and actual target histories. I derives that history from
the client's checked output contract; NI WN/AU use bootstrap output, AA uses the
same post-hidden contract. This is justified by mandatory boundary restoration and
checked at runtime. It does not guess the prior stage's hidden.pre history.

`PackedGraphBatchMeta::new(params, geometry, index, ordered_valid_ids,
initial_input_histories, per_stage_target_histories, partitions)` validates complete
next schedule metadata, including stage target specs/levels, next valid IDs and
denominator. Known next input histories for s>0 are **derived from mandatory next
boundary restoration**, not copied from the current input or guessed state history.
For each stage create `PackedNextStep::Known` with those derived histories and its
own next target histories/rate/index. Call `PackedNiPlan::new` on the current stage
metadata and derived current batch. It returns full committed histories and c2's
carried/refreshed next-DAG decision. This preflights all current stages before work.
Unknown and Terminal map to existing conservative per-stage state refresh and retain
their distinct future-unadmitted/terminal status. No graph-wide Ready alias.

Hypothetical full-four-update planning chains `plan.committed()` metadata (no usable
generation token). Actual execution reconstructs and binds fresh current plans from
real old stages; compare to frozen hypothetical histories/counts, never execute the
hypothetical plan. Error other than NoLevelsLeft is not repair permission. c2's
known-next state-frequency decision remains authoritative; no hardcoded WN schedule.

I preflight uses `plan_stage_dag` from actual old stage metadata with derived
contract inputs, no hidden bootstrap seam. Pre-check complete updated role contracts
and client output before any current work. `PackedInteractiveStage::prepare_step`
can repeat its cheap runtime planning; it must agree with the graph preflight.

### Ownership, execution and failure

The NI graph session borrows external E/Encoder/B, stores their immutable contract
and a reusable `PackedNiTrainer<'a,E,B>` constructed from the same external borrows.
It is not self-referential. One child handles successive borrowed old stages;
results are owned and carry no borrow of a temporary adapter. The I session owns
the analogous child and accepts `&mut client` only on its I method. No enum erases
these two APIs into an NI callback path. NI contains no secrets, plaintext arrays,
decryptor/materializer/client/RNG or loss callback.

For s in graph order: build encrypted records from the old input/handoff; execute
the preflighted c2 child against `&old.stages[s]`; take the owned stage candidate;
retain its owned receipt; create boundary inputs from that receipt's old hidden.
All stage candidates stay in a local Vec, not in old graph. On any stage, boundary
or final verification failure, discard candidate ownership and return failure.
Only after all succeed create the private complete graph candidate. Caller performs
`graph = candidate.into_graph()`. No layerwise or stagewise assignment/checkpoint.

Mode-specific boundary behavior for each valid record, s < last:

| Mode | Exact payload source and work |
|---|---|
| I | receipt.step().forward().hidden(), CanonicalHidden request, client.refresh, existing request-token/output validation, wrap checked response. One client hidden event. |
| WN/AU | receipt.step().forward().hidden_pre(), existing bootstrap_hidden; canonical lowering only, one bootstrap, no second mask. |
| AA | receipt.step().forward().hidden_post() is required; pass that exact selected ciphertext payload via read-only handle/clone, no projection, normalization or bootstrap. |

The final stack has no downstream boundary, so I/WN/AU never add a final boundary
event. AA still has its final stack's existing within-stage hidden seam before its
classifier; that is part of c2, not an output-stack handoff. Child derivative1 uses
old z1; child affine2 and outer1 consume exactly its selected AA post object.

Cloning encrypted handles for downstream records is allowed; recomputing forward,
bootstrapping again, or selecting a numerically close ciphertext is not exact reuse.
Receipts carry `HandoffOrigin::AaPost {stage, valid_position, child_event_index}`;
test-only recording evaluator tags ciphertext payload identities through clone and
records actual affine2/outer1 multiplication operands and downstream input binding.
No production arbitrary ID setter or cryptographic authenticity claim is added.

Graph receipts retain ordered child receipts, valid IDs/partition map, one tagged
handoff per boundary/valid record, encrypted source/output snapshots, I timings,
predicted/observed traces and a single inclusive graph counter interval. NI handoff
events reuse existing PackedBootstrapEvent with global ordinal/record roles; I
handoffs retain request role and payload, response, timing and optional nested delta.
AA reuse has a child event reference and a zero-work handoff interval, not a second
bootstrap event. Every counter field is checked with checked integer summation.

Failure records contain phase {Preflight, Stage, Handoff, Verification}, stage index,
optional record/refresh role, completed ordered child receipts and handoffs, current
typed child failure (if any), inclusive observed delta or None on reset, error and
attempted/completed/validated event status. Child failure's partial events remain
accessible. It cannot expose a commit method. All old W/U/bias ciphertexts, indices
and generation tokens remain unchanged; HE counters and external side effects do
not roll back. A failed bootstrap can lack output/counts: preserve its attempted
role and observed enclosing interval, never report it as validated.

## 3. Forward-only API, implementation seam and counts

```rust,ignore
enum PackedForwardPart { Backbone, OutputStack }
PackedForwardRecord::new(id, input: PackedVector<C>, valid: bool) -> Self;
stage.plan_forward(ev, enc, records, part, optional_hidden_boot_contract)
    -> Result<PackedForwardPlan>;
stage.prepare_forward(ev, enc, records, &plan, optional_planned_bootstrap)
    -> Result<(PackedForwardOutput<C>, PackedForwardReceipt<C>), PackedForwardFailure<C>>;
ni_graph.plan_forward(&graph, &input_batch, inference_plan) -> Result<PackedGraphForwardPlan>;
ni_graph.prepare_forward(&graph, &input_batch, &plan) -> PackedGraphForwardResult<C>;
interactive_graph.prepare_forward(&graph, &input_batch, inference_plan, &mut client)
    -> PackedGraphForwardResult<C>;
```

Forward input batch has only IDs/validity/input/partitions; targets/rates/update index
are not arguments. `PackedInferencePlan` explicitly selects boundary policy plus
final-hidden policy: NI BoundaryBootstrap or AllHiddenBootstrap, I ClientBoundary,
or no-refresh paths only if their complete depth trace admits them. The bounded
fixture uses boundary policy for WN/AU and I, AllHiddenBootstrap for AA. It is recorded
independently of the training plan and has its own predicted/actual counters.

Backbone traverses only W0, optional b0, hidden activation and optional fixed hidden
refresh. OutputStack additionally traverses W1, optional b1 and configured output
activation. It never calls plan_step, plan_stage_dag, layer.request, loss, derivative,
outer sum, optimizer.prepare or state reconstruction. Backbone planning/runtime
must not read/evaluate local-head ciphertexts or velocities. Bind only used layer
generations in the forward plan; do not call stage.metadata(), which inspects U and
the unused head. The graph owns/binds its graph identity and each selected plan.
No state candidate or generation mutation results from forward-only execution.

A small private forward traversal in stage.rs shares Runner::{affine,bias,activation}
with the training runner and existing fixed bootstrap_hidden seam. Its parameter-only
counterpart uses the narrowly shared dag Planner methods above, real q/F64 order and
actual Encoder feasibility. It produces observed trace comparisons just like c2.
No duplicated full training engine and no dummy targets/zero learning rate trick.

For eMLP nonboot inference per valid record on lattice128×4: a backbone costs
2 tensors/relins, 3 ct-pt, 5 rescales, and 7 rotations (RE) or 4 (CE). Final head
adds 1 tensor/relin, 1 ct-pt, 2 rescales and its 4/7 rotations. For G stages,
`tensor=relin=n*(2G+1)`, `ct_pt=n*(3G+1)`, `rescale=n*(5G+2)`;
rotations are n*sum(selected orientations). All SSE/bootstrap fields are zero
before separately counted refresh. At n3: G2 has rotations54, tensors/relins15,
ct-pt21, rescales36; G3 has rotations66, tensors/relins21, ct-pt30, rescales51.
These are independent mathematical literals, pending exact metadata validation.
Actual seven fields are rotations, relins, ct_ct_tensor, ct_pt_mults, rescales,
sse_switches, bootstraps. Derived key-switch totals are not an eighth independent
counter. Never add child/event observations to already inclusive graph measurements.

## 4. Fixed prospective original tiny fixture

Reuse only the reviewed full-chain construction: N1024, unchanged n16_boot_bits()
Q/P, scale42, dnum8, BootParams default_n16 except log_slots9, output16, 512 complex
slots. Initial W and every fresh input/target at level16 exact Delta; U absent.
No keys selected/tested here, no chain/profile change, no repeat of c2's calibration84
or primary212. Existing calibration is finite context evidence, not this fixture's
precision. Every actual block event still must pass its own prospective gates.

G2 uses `PackedGeometryPlan::new(3, [StageShape(3,3,2);2],512)`;
G3 uses the analogous three stages. All widths3 are non-power-of-two; nonsymmetric
matrices exercise RE/CE then CE/RE then RE/CE. Lattice rows128/columns4; real classes2,
all three classifier channels trained, target channel2 zero. Every stage starts
from these original logical output-by-input matrices:

```text
W0 = [[5/16,1/16, 1/16], [1/16,5/16,1/16], [1/16,1/8,1/4]]
W1 = [[5/16,1/8,-1/16], [1/16,5/16,1/16], [-1/16,1/16,1/4]]
X0=[3/4,3/8,1/2]; X1=[3/8,3/4,1/2]; X2=[1/2,3/8,3/4]
X3=[-1/2,1/2,-1/2] // invalid sentinel
Y0=[3/4,0,0]; Y1=[0,3/4,0]; Y2=[3/8,3/8,0]
Y3=[7/8,-7/8,7/8] // invalid sentinel; per-stage encrypted representation still checked
```

Same logical Y at each stage, independently encrypted in that stage's required
Expanded/Repeated/Expanded target representation. No biases. mu0.9, fixed r0.08,
rates `[0.04,0.01,0.02,0.08]`; valid IDs `[0,1,2]`, `[0]`, `[0,2]`, `[0,1,2]`.
Partitions are physical ranges [0,2),[2,4), valid counts 2+1,1+0,1+1,2+1.
Each stage uses the total n `[3,1,2,3]`, lr/n once. Pure metadata control uses48
distinct IDs with32+16 valid partitions and n48, without encrypting48 samples.

Independent oracle uses scalar matrix loops, h=z*z+z, local delta=out-target,
old-head transpose times delta, derivative=2*z_old+1, mean of valid outer products.
It evolves unscaled V'=0.9V+g and W'=W-lr*(g+0.9V'), then U=0.08*1.9*V'.
All forward inputs are captured from old backbone values; no backward chain crosses
stages. `task-54-d4c3-scalar.py` and its JSON output fix every per-update input,
hidden, output, gradient mean, W/U/V and postcommit inference result for G2/G3.

Literal independent packer uses s=4*i+j. RE matrices store W[j][i] if i,j<3;
CE stores W[i][j]. Expanded vector is v[i] when i<3 across4 columns; Repeated
is v[j] when j<3 across128 rows. All other slots zero. Do not call production
pack/mask helpers in the numeric oracle. Expected training key union is
[-2,-1,1,2,4,8,16,32,64,128,256], independently checked before future HE.

### Plaintext visibility and role bounds

Values below are from one deterministic ordinary-Python run, not encrypted data.
Rows0/1 are identical between G2/G3 because of local-loss isolation.

| stage | final max W0 change | final max W1 change | final max U0 | final max U1 |
|---|---:|---:|---:|---:|
| 0 | .0686884673655 | .0677074109230 | .0765106096705 | .0763258361555 |
| 1 | .0356782848478 | .0375947788380 | .0395983718645 | .0421955328677 |
| 2 | .0160385672001 | .0180619796807 | .0177330657382 | .0202158127715 |

Every respective W/U role exceeds4E for E=2^-12; smallest signal is65.69E.
This is final role-wise maximum signal, not every coordinate/individual step.
Training-schedule maxima W=.381188467366, U=.076510609671, old hidden=.425659532397,
output=.168111179410 (rounded up), original inputs<=.75. Hidden maxima stage1/2
are .205584113973/.091441038866. Fix actual all-slot pre-bootstrap bounds W<=.4,
U<=.1, **training hidden<=.5** (universal contract<=1), matching established c2
envelopes. Original inputs and valid targets<=.75 are not bootstrapped. Invalid
target Y3 has magnitude<=.875; its representation is checked but it contributes
zero HE, refresh, denominator or gradient work. Bias bounds are inapplicable, not zero
evidence. Metadata cannot prove any encrypted range.

FIX1 explicitly includes the required post-final-update n3 inference. Its inputs
are X0/X1/X2 and its weights are the frozen final committed scalar weights above.
The revised JSON's `cases[].final_inference.stages[]` stores each input, preactivation
z, hidden vector and maximum, plus final output; no training value is changed.

| graph | inference stage | ideal maximum hidden | NI placements that bootstrap it |
|---|---:|---:|---|
| G2 | 0 | .5292030932466849 | AU/WN/AA |
| G2 | 1 | .2873361031574850 | AA only; no final AU/WN boundary |
| G3 | 0 | .5292030932466849 | AU/WN/AA |
| G3 | 1 | .2873361031574850 | AU/WN/AA |
| G3 | 2 | .1346734554156006 | AA only; no final AU/WN boundary |

Register **inference hidden<=.6**, separately from training hidden<=.5, for every
actual all512-complex-slot inference bootstrap input; universal contract<=1 stays.
Inference intermediates are also explicit I parity snapshots, although I has no
bootstrap-range gate. The old .5 inference bound would fail even ideally, because
the final committed W changes the forward inputs. The prospective .6 envelope has
approximately .0708 ideal margin; it is no actual-error proof, interval certificate
or calibration result. W<=.4, U<=.1, E=2^-12 and epsilon_boot=2^-20 stay fixed.
Changing arrays/weights/rates would alter the already-inspectable learning fixture;
omitting inference would remove required forward-only coverage. The selected repair
adds no bootstrap or experiment axis. Its cost is a wider explicitly phase-labelled
inference envelope; actual noise/replica/range/parity can still fail and is retained.
This is a pre-HE design correction, not a tolerance change after measurements.

Prospective numeric acceptance: every bootstrap incremental complex all-slot error
<=epsilon_boot=2^-20 versus its actual level0 input; every old/candidate/committed
W/U, pre/post hidden, handed input, output and final inference snapshot <=E=2^-12
versus the untouched scalar twin. Check all512 complex slots and record maximum,
worst slot, active/padding error, replica spread and imaginary leakage. Compare
state projection and bootstrap separately; AA post and handoff must be the selected
payload, not merely close. U stays in fixed-reference units. Separate V error bound
would be E/(.08*1.9). No averaging over seeds/events/samples or final-only gate.

These gates remain unvalidated for blocks. Cumulative polynomial/replica/NAG error
can fail E despite per-event epsilon_boot. No new interval certificate, assumed
amplification constant or tolerances changed after HE observations.

### Exact fixed seed map (prospective, no seed search)

One held-out key seed `[167u8;32]`, distinct from151/152/153. One constructed context
may serve cases, resetting every state/receipt/timer/counter origin. Case order is
G2 AU,WN,AA then G3 AU,WN,AA; I cases have separate phase identity.

All encryption RNG seeds are exactly the32-byte concatenation
`[195, domain, G, mode, update, stage, record, role] || [0;24]`.
domain1 is initial/fresh encryption. mode0=I,1=AU,2=WN,3=AA; update0..3,
stage0..G-1, record0..3. Initial W seeds use update255, record255, role0=W0 or1=W1.
Fresh inputs use stage255, role2; targets use real stage, role3. Inference fresh
inputs use update4, stage255, role2. No encrypted initial U or dummy bias.
I client refresh alone uses domain2: `[195,2,G,0] || LE64(client_call_index) || [0;20]`,
index0..115 per phase's deterministic global ordered log (do not restart within a
phase). NI refresh has no encryption seed and no RNG/client callback. Additional
structural tests use separate domain3, test ordinal recorded before any execution;
no unbounded real-bootstrap or random-seed axis is authorized by that namespace.

## 5. Finite phases, metadata freeze and exact maximum call cap

1. **Pure scalar/metadata**: this scalar calculation is complete; future Rust
   metadata-only test validates the exact Q/P/Delta/Encoder histories, current and
   known-next four-update plans, batch payload/partition/target failures, inference
   counts and32+16 denominator. Fake capabilities are metadata/failure controls,
   never numeric NI. Freeze source/descriptor and expected schedules before HE.
2. **Bounded I correctness**, only after separate authorization: G2/G3, four updates
   each and one n3 post-final inference each. Exactly107 training client refreshes
   (80 state +27 hidden) and9 inference handoffs =116; zero bootstraps. Restore
   canonical logical values/replica averages only, no clipping or model arithmetic.
   Client secret and deterministic encryption RNG remain in test-side code.
3. **True NI primary**, separately authorized: six cases above, four updates each,
   Lazy only; then one n3 inference after the final committed graph of each case.
   One attempt per registered case, stop the phase at first failed gate. Do not
   rerun c2 calibration84/main212. PerSample, disabled-lazy and failure injection
   are separate metadata/controlled-runtime obligations, not primary numeric claims.
4. **Production unrun**: full profile/batch/schedule/security/precision, memory,
   generic CPU/CUDA bridge, hardware parity and full dataset matrix remain gates.

Let G=2 or3 and sum(valid counts)=9. Every successful NAG state refresh contains
exactly4 roles/stage (W0,U0,W1,U1), including first update after U initialization.
Four updates therefore have at most16G state calls. WN/AU boundary calls are9(G-1);
AA within-stage hidden calls are9G and boundary extras zero.

| placement | G2 training max | G3 training max | post-final G2/G3 inference | total max |
|---|---:|---:|---:|---:|
| AU | 32+9=41 | 48+18=66 | 3+6=9 | 116 |
| WN | <=32+9=41 | <=48+18=66 | 3+6=9 | <=116 |
| AA | <=32+18=50 | <=48+27=75 | 6+9=15 | <=140 |
| all six cases | | | | **372 attempted bootstrap calls maximum** |

339 is the maximum training-only cap,33 inference-only. AU training107 and its9
inference calls are exact for successful execution. WN/AA state frequency is **not
predicted here**. Before execution, metadata must produce a complete exact schedule
and all7 predicted counts for each case/update/role and inference, whose call total
must be<=372. Primary accepts only actual attempted/completed/validated counts equal
to that frozen successful schedule (with complete coverage); cap compliance alone
is not success. The runtime guard refuses call373 before invoking bootstrap.
Failures may have fewer completions; retain attempted roles and partial intervals.
Negative-path real bootstraps are excluded from this cap and not implicitly released;
use fault/fake controls, or obtain a separately bounded assignment if real work needed.

## 6. Required structural controls and diagnostic protocol

The actual NI B capability remains public-key-only: a test recording wrapper may
clone encrypted input/output payloads and snapshot counters, but stores no secret,
decryptor, diagnostic callback or plaintext model and delegates the real CPU
Bootstrapper without repairing outputs. Only after prepare returns success/failure
does the external test harness decrypt retained observations and check each actual
pre-bootstrap input range retrospectively. Stop before another update/case on a
failed check; do not claim range was proven before the bootstrap. On a failed call
retain its captured encrypted input and available output. No decrypt diagnostic is
invoked from inside NI and no maximum/scale/array is fed back into the server.
Log all512 decoded
complex slots (or a byte-stable full-slot artifact referenced by hash), not only a
single scalar maximum; preserve exact source/input/output parameter identity,
levels, Delta bits, components, stage/record/role, event ordinal and7-count intervals.
Capture old graph, pre-refresh candidates, pre/post/projection, final whole graph,
and nontrivial U at every update. Flags include local-head affine count, loss calls,
outer-sum calls, optimizer calls, generation changes and graph commits.

Required controls, not an invitation to new experiment axes:

- Alter only last-stage target: earlier W/U and all old handoffs unchanged; final
  update changes. Alter first local target: its W/U change, its same-update old
  hidden and all downstream updates unchanged. Alter backbone W: downstream input
  changes. These scalar controls already pass; encrypted controls remain future.
- Captured-old-forward reverse update equals sequential graph preparation on every
  scheduled update. Wrong averaging of2+1 group means must be distinguishable;
  scalar max gradient difference .00866062636487. Pure32+16 gives denominator48.
- Wrong stage target representation, missing/extra target, duplicate/out-of-order ID,
  overlapping/missing partition, wrong declared valid total, wrong adjacency/context,
  stale graph/batch identity and known-next index/rate fail before counted work.
- Exact AA post payload participates in affine2 and classifier outer1; boundary
  aliases that selected payload; additional mask/bootstrap count is zero. Changing
  the local head cannot alter backbone-only inference. No final boundary restoration.
- Fail during the last stage after prior candidates and handoffs exist; old whole
  graph and all generations/indices remain identical, counters remain advanced,
  failed role/completed events preserved, no candidate commit. Also replay/mismatch
  I hidden responses and NI wrong output metadata are rejected at the boundary.
- Inference record/test flags must show0 local-head calls on nonfinal stages,
  0 loss/outer/update calls,0 generation changes and no targets. Exact selected
  matrix counts detect a head accidentally evaluated and discarded. Numeric output
  matches the separate postcommit scalar predictor; AA has only planned seams.

Future test registration is normative; these names are prospective, not implemented:

| Exact name | Registration | Permitted default-test work |
|---|---|---|
| `block_metadata_schedule` | ordinary `#[test]` | parameter-only Encoder/metadata, no keys or HE |
| `block_partition_and_target_preflight` | ordinary `#[test]` | metadata rejection controls, no keys or HE |
| `block_failure_controls` | ordinary `#[test]` | fake/controlled metadata backend only, no real keygen/HE/bootstrap |
| `block_interactive_original_fixture` | `#[test]` and `#[ignore = "D4c3 bounded I primary: requires reviewed source, frozen metadata and explicit root I phase release"]` | discoverable but skipped |
| `block_ni_original_fixture` | `#[test]` and `#[ignore = "D4c3 true NI primary: requires reviewed source, frozen metadata and explicit root NI phase release"]` | discoverable but skipped |

No actual-arithmetic control may be hidden inside an ordinary fake-control test.
If real key/HE controls are needed later, they require a separately reviewed bounded
ignored registration and execution assignment. The ignore attributes preserve
ordinary workspace/CI discovery while preventing an unapproved broad run from
launching either primary. No silent early return when release is absent. An
environment-only guard was rejected as an unnecessary input; this follows c2's
discoverable ignored-primary separation. Omission remains detectable by exact
harness and phase-coverage assertions below. No phase is released now.

## 7. Commands and remaining gates

The only computation executed during design:

```text
python3 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d4c3-scalar.py > .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d4c3-scalar.json
```

After source assignment, the build/list step verifies compilation and discoverability
without running test bodies, generating keys or performing HE. Prospective commands,
not run now, are separate from any primary phase execution:

```text
CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --test packed_blocks --no-run
CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --test packed_blocks block_interactive_original_fixture -- --ignored --exact --list
CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --test packed_blocks block_ni_original_fixture -- --ignored --exact --list
CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --test packed_blocks block_metadata_schedule -- --exact --nocapture --test-threads=1
```

Each ignored-list command must select exactly its one registered test; a missing,
misspelled, ordinary/unignored or duplicate primary name fails the source gate.
The metadata execution is separate no-key/no-HE admission, not primary release.
After reviewed source, frozen exact successful metadata schedule/build identity and
explicit root release of that phase, use exactly the relevant ignored invocation:

```text
CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --test packed_blocks block_interactive_original_fixture -- --ignored --exact --nocapture --test-threads=1
CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --test packed_blocks block_ni_original_fixture -- --ignored --exact --nocapture --test-threads=1
```

The attempt runner must assert harness **exactly1 selected and executed**, success
exit0, passed1/failed0/ignored0, never a zero-test pass. Filtered-out tests are not
executed evidence. The I body must assert2 graphs,8 committed updates,2 final n3
inferences and116 attempted/completed/validated client calls. The NI body must
assert6 cases,24 committed updates,6 final n3 inferences and attempted/completed/
validated bootstrap counts equal to the complete frozen successful schedule<=372;
each event/role/count/diagnostic gate remains mandatory. Counts require both harness
and body audit, not only Cargo exit0. Stop on first failure without retry/retuning
or silent return; preserve partial attempts and no success stamp. As diagnostics
run after synchronous prepare, later events in that same failed update may already
have occurred; count them and stop before the next update/case. Build/list success
does not release I, I success does not release NI, and neither releases production.

Build identity, source hashes, fixture/seed hash, parameters/descriptor, schedule,
command/environment, attempt ledger/stdout/stderr/exit and actual coverage are frozen
before claiming execution evidence. A failure is preserved without retuning arrays,
seeds/rates/reference, tolerances, bounds or presets. Any revision is a new retained
proposal, not overwriting this registration after observations.

Independent design review and root ownership/API/fixture ratification are complete. Pending: bounded
source assignment; source review; full current/future metadata Encoder admission;
separate I/NI run release; all-slot precision/range and true event/counter success;
generic/backend/host-failure diagnostics; production pre-key memory including retained
old/candidate/snapshot buffers; actual GPU; secure profiles and scientific matrix.
The shared P1 capability is structural/conditional and does not remove bootstrap
internal encoding/EvalMod, secret provenance, range, precision, memory or backend
gates. Full real-dataset block campaign remains user-held even if tiny tests pass.
