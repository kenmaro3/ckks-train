# Task54 D7 resource producer — independent design review

SPEC: **REQUEST CHANGES**. QUALITY: **REQUEST CHANGES**.

Reviewed the frozen proposal as a design, not an implementation or validated resource model. The single phase-aware owner is a sound direction, but four bounded issues below must be resolved before ratification. This review grants **no source authority, model calibration, native execution, HE, installation or experiment release**. Existing all21 profiles, full2680+20+7 and quick268, Full/Quick separation, C1, scientific/security thresholds, full encrypted-block hold and baseline/GPU/external obligations remain unchanged.

## Inputs and method

Read `task-54-brief.md` first. Independently verified both requested byte counts and SHA256 values:

| Input | Bytes | SHA256 |
|---|---:|---|
| `task-54-d7-resource-producer-proposal.md` | 45323 | `80d6bdb917e33f4db7fde5233dac15339ea91c76903a403a588bcda757ecaf74` |
| `task-54-d7-typed-materializer-preallocation-map.md` | 44480 | `31504e7c3cda346de741448103ce1d9629003bd3fc4359d3d6916db9ac583f9c` |

Binding document hashes inspected:

| Document | SHA256 |
|---|---|
| `docs/study-gates-design.md` | `073f9fec30a4818ae78b990e02439948886573a65804f03bb21a47b2f37baf5d` |
| `docs/study-execution-design.md` | `c2a260dd299043b817bd6f7d1586918a3f48c800f8cc59f259ceafeac5b9a779` |
| `task-54-d7-gate-schema-proposal.md` | `1e54707536ee52c481cfa789c308d9f4605f096a1a720649c2eebe7a1dc6f50b` |
| `task-54-d7-gate-custody-proposal.md` | `fc93856110cb2e8c7382fe792b21cf0550a85007d9267d09f75d6e9637e8c73f` |
| `task-54-d7-live-gate-integration-proposal.md` | `c667176c41f1e5dda060e552c6fe9db1209446c76c3f2161ea949571cb539e5c` |

All code inspection used `git show`/`git grep` at BASE `71d3f0937741a4ff8d156b36540332ea82203d3c`. No moving worktree source was relied on. Inspected the actual runtime/materializer, reference read/recovery/snapshot/target, context helper, process stamp, shared publication primitive, supplied Resource arithmetic and CPU continuation seams. This is not a new exhaustive allocation audit; no complete 22-term proof is claimed. Used engineering:code-review for correctness, security, performance and maintainability review. Memory was consulted only for historical permissive-license/evidence-boundary context; current binding documents govern this review.

Root additionally supplied the current task constraint: no sudo/new administrator provisioning, ordinary key-authenticated server access, and no new generic permission loop. I have not inspected the server or inferred its current permissions/filesystems/namespace state.

## Findings

### 1. High — the mandatory Unix-root lock introduces an unnecessary deployment blocker

**Locations:** proposal lines 23–24, 28, 43, 114–120; binding live-gate proposal lines 89 and 139–143; gate-schema proposal lines 108–110. BASE `study_provenance/publication.rs:174–188` explicitly supports effective-user-owned private roots and root-or-effective-user-owned safe ancestors.

The proposal requires a newly provisioned `/run/hekit/task54-resource-v1/execution.lock`, Unix UID0 ownership, non-runner-writable parents, and a separate root-provisioning record. Its own text labels installation a separate authorization prerequisite. The binding design requires a trusted origin and a live cooperative claim, but does not require this new UID0 lock. In the task's ordinary-user/no-sudo environment this selected prerequisite can prevent every positive run for a reason unrelated to actual scientific or resource evidence. “Root” as the trusted task agent/launcher must not silently become Unix UID0. The root-owned lock also does not reserve RAM against other users; its principal exclusion guarantee remains cooperation by the defined runner domain.

**Bounded remedy:** choose one stable per-physical-host lock namespace under an existing trusted ordinary-user-owned root, shared by every participating campaign/checkout on that host. Specify its exact selection and owner/mode/ancestor/mount checks, exclusive initial establishment, fresh open-file-description flock, no replacement/unlink policy, and continued identity checks. Establish it outside a running claim using already authorized ordinary filesystem work; do not add an administrator dependency or a generic approval service. A campaign-local path is insufficient. State explicitly that the trusted account/launcher and its compliant runners must not replace this inode during use: this is cooperative exclusion, not protection against arbitrary same-UID malicious namespace mutation. Preserve refusal if other applicable cooperative reservations cannot be accounted for.

Keep **host origin proof separate**: a user-owned lock does not prove the initial host namespaces, global cgroup root or controller absence. Retain a genuine independently trusted boot-bound origin with ordinary-user-readable evidence; if unavailable, report that precise missing observation. Do not turn missing namespace evidence into zero limits or weaken it to a self-authored `host=true`. The bounded fix should explain the actual trust source without claiming that the reviewed host has already supplied it.

### 2. High — the first capacity decision depends on an anchor loaded only in the next phase

**Locations:** proposal lines 28, 34, 43 and 53–55, 103–106. BASE `study_provenance/process.rs:12–24,65–68` and `study_provenance/context.rs:867–928` show that existing process/context checks themselves allocate and may launch helpers.

Step 1 performs the initial trusted host/cgroup/disk capacity check before model/context/registry expansion. Step 2 then loads the host policy. Yet the initial observation is required to compare namespaces against `host_namespace_origin`, use supported kernel/allocator/filesystem facts and prove that cgroup traversal reaches the actual host root. Those facts live in that host policy/installation evidence. The ordering does not specify how the minimum trusted anchor is acquired before it is relied on. Calling the current context/ProcessStamp implementation as though it were a fixed-scratch bootstrap is also not justified by its existing Vec and helper behavior. The separately validated compiled bound is necessary but does not resolve where the initial trusted inputs come from.

**Bounded remedy:** freeze a small explicit bootstrap sequence: trusted root/FD origin -> bounded acquisition/verification of the minimal boot/namespace/kernel/filesystem/claim-domain anchor -> fixed-scratch topology/capacity probe -> admission of the larger metadata/model/context phase. Identify which checks use already established launcher state, which reads occur before the first comparison, their compiled scratch/lifetime coverage, and which exact retained bytes/handles become the later policy view. The minimum anchor can be a strictly bounded projection of the existing fixed policy; no new generic policy service is needed. Prove expected procfs/cgroupfs origin and the deliberate namespace-descriptor opening rules, including how necessary proc namespace links are handled without permitting arbitrary symlinks or artifact-selected paths.

Do not invent a provisional final Resource/physical identity, a caller permit or numeric bootstrap allowance. It is acceptable to leave the numeric compiled bound unvalidated and the positive constructor unavailable, but the source design must have an acyclic input/trust order. The larger model cannot authorize the parsing required to discover its own authority. Unknown/hidden ancestors remain refusal, not an empty cgroup list.

### 3. High — CLOEXEC permits live cooperative helper work to outlast its claim

**Locations:** proposal lines 54, 88, 114 and 120–122. BASE `study_provenance/context.rs:867–928,1074–1089` constructs and spawns the Python provenance helper; snapshot additionally names the Git executable. The proposal correctly includes helper overlap in the modeled work.

The selected lock description is CLOEXEC and never exported. After a controlled helper execs, it therefore has no hold on the parent's resource flock. If the parent dies while that helper or its Git descendant is still running, the parent lock may close while cooperative modeled work remains alive and can make further allocations. Another runner can acquire the same lock and claim `other_claimed_* = 0`. An ordinary fresh OS reading includes current usage but does not account for that surviving work's still-pending demand. This is not covered merely by acknowledging that old processes may survive: section 5 itself disallows relabeling outstanding cooperative reservations as noncooperative to obtain zero.

**Bounded remedy:** specify one actual crash-safe custody rule for the existing Python/Git subprocess tree. For example, a narrowly controlled inherited hold on the same open file description may survive the exact trusted helper/descendant lifetime, with an explicit exception to blanket CLOEXEC/nonexport and no child authority to resume the parent's frame or unlock the shared description. Alternatively remove the subprocess work from this phase through an equivalently verified bounded implementation, or use an already available proven child-lifetime mechanism that prevents any helper work outliving exclusion. Choose one concrete route and inventory its descendant/exec behavior; do not add a generic daemon, scheduler, reservation registry or arbitrary PID cleanup. A parent-only RAII guard, ordinary wait on success, or untested parent-death assertion does not cover parent crash.

Require later bounded controls for parent death during the actual helper and descendant stages, attempted competing acquisition, and eventual release after the whole covered tree terminates. No such control was executed in this review. If a helper route cannot provide this lifecycle, it must remain specifically unsupported; the design must not report the serial domain empty while that route's work survives.

### 4. Medium — mandatory standalone reference/target entrypoints lack a concrete work selection and claim path

**Locations:** proposal lines 32, 35, 54–60, 126; accepted map lines 164–165. BASE `study_reference/publication.rs:114–119,237–250,297–305`, `document/reader.rs:529–550`, `target.rs:192–197`; `study_runtime.rs:309–336` is the distinct Train entry.

The proposed model selects `(registration_id, Full/Quick consumption)` and the claim requires canonical registration and planning request. The sequence constructs these in the Train `run_owned` frame. However, the mandatory ReferenceStore read, recovery and snapshot paths take a `VerifiedStudyContext` and reference ID, and the target canonical helper takes context/profile/seed. They can acquire a large document and independently materialize outside a live HE planning request. Merely listing these callsites as future wiring does not specify who constructs their preallocation owner or how their real non-HE work matches the claim. Naively constructing another owner inside an already-held Train claim would also contend with its own serial lock. Fabricating a Quick/Full HE request for reference verification would misstate applicability.

**Bounded remedy:** give an explicit entrypoint table and ownership sequence for the named APIs. Nested Train target/reference reconstruction must borrow the SAME existing resource owner/envelope; standalone reference/target/recovery operations need an actual bounded preparation-only work selection that cannot finalize an HE gate. Specify the minimum closed source-format adjustment, if needed, before vectors/encoders, and how context/bootstrap/document buffers are covered before their first read/parse. Preserve seed0 and selected-seed independent reconstruction and receipt checks. Unsupported legacy wrappers must refuse before their uncovered acquisitions, with exact scope stated; this may be an intermediate source state, but blanket refusal of all real reference paths is not completion of the promised integration. Do not add a public generic permit or duplicate the materializer.

## Rulings and positive assessment

1. **Publication versus resource locks:** accept the conceptual distinction. The existing “no flock held” wording needs a narrow explicit refinement to publication/profile guards only; a separate resource claim must remain alive through the actual covered work. That clarification alone does not close findings 1 or 3. Preserve a single lock order (resource first, later short publication guards) and no reacquisition of the resource lock by nested readers.
2. **Structural cgroup v2 root:** accept the conceptual refinement, contingent on genuine initial-host origin and source-validated traversal. Emitting only applicable accounting interfaces avoids inventing a nonexistent root `memory.current=0` row. Namespace-root `/` is not proof of global root. Absence/error is not unlimited. Retain both finite and unlimited ancestor rows, including a finite ancestor above an unlimited leaf. V1/hybrid remains explicitly unsupported until its concrete normalization/profile is validated. `/proc/cgroups` alone is not v2 memory-controller origin evidence; the retained root/controller interfaces must establish that branch.
3. **Records and model selection:** closed retained records and independent literal vectors before encoders are appropriate. Selecting final exact model/validation artifacts through a trusted current policy after the built source is known avoids the proposed source/model self-hash cycle. That selection is provenance, not model validation. Resolve the work-selection/anchor changes above before freezing the vectors. Keep outer Resource and the other existing formats/caps unchanged unless a specific reviewed necessity is demonstrated.

The allocation approach correctly distinguishes requested payload, model-derived peak/incremental need and observed capacity. The existing Resource codec at BASE `document/evidence/tails.rs:436–531` validates supplied arithmetic and explicitly is not an OS producer. The proposal does not mistake it for one. The retained materializer plan covers raw plus F64/F32 arrays, decoder copies, hashing encodings, selected and seed0 reconstruction, plan preparation and post-start revalidation rather than authorizing these retrospectively. BASE `study_profile.rs:1478–1502` indeed has two acquisition waves, and `data.rs:559–596` indeed retains decoded NPY arrays while cloning images; the map correctly counts these until changed source removes them.

Checked four-phase sums, all22 components, exact owning-allocation assignment, conservative additional bytes when resident credit is unproved, and no guessed allocator margin are consistent with the binding schema. A later positive implementation still requires complete source/lifetime derivations plus actual applicable native checks. No present source excerpt, transport cap, historical payload count or empirical maximum establishes that completeness. Unknown allocator/kernel/helper costs remain missing.

Host/cgroup headroom and disk accounting are honestly prospective: the lock is advisory, MemAvailable is an estimate, nonparticipating work and administrative changes can race the observations, and later OOM is failure. The per-filesystem checks plus conservative scalar disk projection preserve the wire at a possible false-refusal cost. Retained immutable observation files and per-boundary publication add real I/O, inode, memory and storage demand; the selected work's finite observation/publication count and failure-path retention must be part of the actual model/test brief. No calibration or runtime cost is measured here.

## QUALITY and task alignment

| Dimension | Assessment |
|---|---|
| Correctness | Changes required: bootstrap trust ordering, subprocess lifetime and all materializer entrypoints. |
| Security/custody | Good closed origins, retained files, same-process ownership and no-clobber reuse; ordinary-user trust boundary and namespace-descriptor bootstrap rules need explicit resolution. |
| Performance | Single host serial claim and compiled rules are proportionate first mechanisms. No generic scheduler/model language is needed. Runtime and validation cost remain unmeasured. |
| Maintainability | Reusing the reader/publisher/Resource wire is preferable to parallel frameworks. Fix the four local design gaps before adding encoders or broader architecture. |
| Scope/operability | CPU-only support is honestly bounded, but the added admin installation is avoidable and conflicts with the task. Runnable paper tables still require actual CPU execution, remaining GPU/external/baseline adapters and complete study evidence. |

A multi-claim registry would add reservation arithmetic, stale-work handling and concurrency states without helping the immediate serial CPU path. A UID0 lock adds deployment cost without making RAM reserved; a cooperative user-owned stable domain is the appropriate alternative under the stated task constraints. An arbitrary expression language, fitted resource multiplier, reduced dataset, skipped independent reconstruction or fake positive permit would weaken the design and is not a remedy. Source-only byte-input factoring can remain bounded and fail closed while evidence is missing, but should progress toward the actual read/execute/table path rather than another indefinitely disconnected framework.

## Bounded return to designer

Revise only the producer design (leave the accepted map intact): ordinary-user stable host claim; explicit minimal-anchor bootstrap order; exact helper-tree crash lifetime; concrete standalone/nested reference entrypoint work and ownership. Then freeze the revised record fields/vectors and independently review that bounded delta. Do not fill unknown model terms with zero, guess validation numbers, change study thresholds or authorize execution from this review.

Only this review file was written. No source/Cargo edit, test, `--list`, helper/process execution, dataset/native/process scan, SSH, dependency install, spending, commit/push, publication or subagent was performed. The frozen proposal and map were not modified.
