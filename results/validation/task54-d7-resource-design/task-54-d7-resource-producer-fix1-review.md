# Task54 D7 resource producer — scoped DESIGN FIX1 review

SPEC: **PASS (scoped design rereview)**. QUALITY: **PASS (scoped design rereview)**.

The four findings in review `1081ab3e` are addressed at the design level. No additional blocking defect was found in their fixes or necessary consistency changes. This is not implementation acceptance, bootstrap/model validation, native evidence, source authority, runtime/HE authorization or Task54 completion. Root's full read and ratification remain separate. Missing physical model terms and host evidence remain missing.

## Scope and pins

Reviewed only the four prior findings and introduced breakage: ordinary-user stable host lock; minimal-anchor ordering; actual Python/Git inherited-description lifetime; standalone/nested preparation work. Used the same engineering:code-review criteria as the original review. No new broad design audit was performed.

Independently verified exact hashes and sizes for the FIX1 inputs:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `task-54-d7-resource-producer-fix1-report.md` | 11851 | `60d10ddd10968be155f3446a060f15938f48f7fc591b3fb7f0b372bcd7466403` |
| `task-54-d7-resource-producer-proposal.md` | 65108 | `e72b99a7a599460ea3f2c206a5ecaa002bfa3ad9b62925f02da9d5d07f63604a` |
| `task-54-d7-resource-producer-fix1.diff` | 73750 | `76f6cc3d2d7371ac1ead669c1737c5bd7fc9d93e59a324455be08936f4954de3` |

Also rehashed the preserved original proposal `80d6bdb917e33f4db7fde5233dac15339ea91c76903a403a588bcda757ecaf74`, accepted map `31504e7c3cda346de741448103ce1d9629003bd3fc4359d3d6916db9ac583f9c` and original review `1081ab3e1297582247995da5ac801638be896cf02b59a4bb4094f7f5a9a97575`; all match. A read-only document reconstruction checked every old/context line and hunk count in the supplied seven-hunk diff and reconstructed the exact 65108-byte revised proposal with SHA256 `e72b99a7…`. A system-diff rendering differed in hunk presentation, so reconstruction, rather than renderer byte identity, was used to verify the supplied patch. No repository/test helper was executed for this document check.

All source reads used committed blobs at BASE `71d3f0937741a4ff8d156b36540332ea82203d3c`. In particular, checked `context.rs:867–928,1074–1089`, `scripts/study_reference_build.py:379–423` and `scripts/study_reference_build_support.py:800–875,1153–1176`. No moving source or later `d1aa3af` source was inspected. The previous fixed-source reader/target entrypoint findings remain the baseline for finding 4.

## Per-finding disposition

### 1. High — unnecessary Unix-root deployment prerequisite: ADDRESSED

Revised proposal lines 17–28 and 148–152 replace mandatory UID0 `/run` provisioning with one stable ordinary-user-owned host inode under the trusted account anchor. Host-key selection is independent of boot, checkout, campaign and HOME/XDG overrides. Exact private modes, ancestor/mount/identity checks, exclusive establishment, durability, fresh flock descriptions and no replacement/unlink are specified. The physical host key is stable across reboots; the changing boot identity does not create an alternate lock path.

The trust limit is explicit: compliant runners under the participating account cooperate; malicious same-UID namespace replacement is not defeated by this lock. Other users can still consume RAM. Applicable external cooperative reservations cannot be ignored to produce zero competing demand. This is consistent with the original advisory claim semantics and the ordinary-user/no-sudo task constraint. No daemon, scheduler, administrator install or new generic permission workflow is introduced.

Host/global-cgroup origin remains independently required. Lines 43, 53 and 150 do not use lock ownership, namespace equality, hostname or namespace-visible `/` as proof of actual global origin. Absence of the trusted direct-host anchor is explicitly missing evidence. That evidence has not been observed on the server here; design acceptance must not be read as confirmation that it already exists.

### 2. High — initial observation depends on later-loaded anchor: ADDRESSED

Revised lines 49–57 and 63–65 specify an acyclic B0/B1/B2 sequence. B0 reads a fixed-layout anchor through existing trusted handles and verifies its independently expected identity using separately bounded fixed scratch. B1 selects compiled supported profiles from that anchor and performs the first capacity check under the resource lock. B2 admits the larger host-policy/model/context work and checks host.json against the already retained anchor hash/projection. Host.json has no anchor backlink, so this does not introduce an anchor-policy hash cycle or a source-model self-hash cycle.

The current allocating ProcessStamp parser and context/Python work are expressly excluded from B0. The new fixed-layout anchor is not a caller permit, numeric allowance or fabricated final physical descriptor. Its bounds and native opening behavior remain validation prerequisites. Necessary proc namespace-link traversal is narrowly described using own numeric PID, literal namespace names, procfs/nsfs checks, independently expected identities and before/after process/boot checks. Cgroup mount/type/origin checks remain separate. V2 uses the actual root controller interface rather than treating `/proc/cgroups` as authoritative for that branch.

The fix resolves the dependency ordering without declaring startup allocation-free or pretending to possess a validated bootstrap bound. Unknown profiles, missing origins and hidden ancestors refuse; they are not encoded as unlimited/zero-use observations.

### 3. High — helper work can outlive the claim after parent death: ADDRESSED

Revised lines 154–166 choose a concrete same-open-file-description inheritance mechanism for the actual runtime helper path. Rust makes only an internal File clone and maps it to child stdin through safe std process handling. The parent retains its own hold. Python starts with that inherited hold, validates the closed parent binding, duplicates to a raw FD retained until exit, and supplies that FD through pass_fds to each runtime Git invocation. Parent death therefore need not release exclusion while Python lives; Python death need not release it while an inheriting Git/allowed descendant lives. No child can reconstruct the parent's live gate/frame from this descriptor.

The BASE source confirms the stated route: Rust invokes only `hash-file` and `snapshot` here; snapshot reaches `verify_runtime_snapshot` -> `source_inventory` -> `git_state`/`bounded_run`. The five commands listed by the proposal match the committed calls: two rev-parse calls, diff, untracked ls-files, and staged ls-files. BASE currently uses DEVNULL at both Rust and Python subprocess boundaries, so this is accurately a future source change, not existing behavior. Build/capture/Cargo paths are not implicitly admitted by this runtime-only exception.

The proposal correctly acknowledges that possession of the shared description technically permits unlock/conversion. No-unlock and no-premature-close are obligations of the exact trusted Python/Git/descendant implementation, not a fictional kernel privilege boundary. Specific unproved descendants remain unsupported. This is an implementable cooperative design choice; it is not proof that an arbitrary Git binary preserves the FD or that the current host's process tree does so.

The inherited-hold exception replaces the old contradictory blanket CLOEXEC/nonexport rule and includes actual parent/Python-death, competing-acquisition and eventual-release controls for later bounded native validation. Source/build renewal, the exact executable/descendant proof and those controls remain necessary. They were not run here. Merely adding pass_fds without closing that evidence would not satisfy the design.

### 4. Medium — no real work/claim path for standalone readers: ADDRESSED

Revised lines 72–92 add closed Train, Reference and Target supported-work/claim branches. Preparation operations carry their actual IDs, operation, profile and seeds, with no invented HE registration/request. Only Train may finalize the outer Resource/HE gate. The document transport/parse phase is explicitly modeled before the document profile is known; afterward the same owner selects a typed-materialization envelope before dataset/preflight allocations. A transport cap is expressly not a physical bound.

The entrypoint table covers Reference read/recover/snapshot, the verification callback, target read/canonical/lock and Train nesting. Resource-first lock order precedes the profile guard. Nested calls borrow one owner and the outer source-derived envelope rather than reacquiring the serial lock or rewriting the immutable claim. Seed0 reference reconstruction and selected-seed independent reconstruction/receipt equality remain required.

The supplied-input option has an explicit noncircular meaning: Some only for an actual input already under the same owner; None admits its later construction with that additional work in the envelope. Target lock has no future target/input hash requirement. Large reference results and snapshots remain retained inside the outer resource lifetime, including snapshot-to-target-lock use; short scoped interior-state borrows address nested checks without self-referential storage or a second owner.

Legacy wrappers that cannot satisfy the new lifetime must fail before uncovered work. The proposal correctly calls that an intermediate state and explicitly requires real standalone and nested wiring for completion. This closes the design omission; neither the new branch schema nor a set of refusing wrappers alone completes integration.

## Introduced-breakage and scope checks

No required change to the frozen outer Resource/Requested/Requirements/Observation/Claim/FileRef or Full/Quick wire is introduced. The new bootstrap and supported-work/claim source formats must receive independently frozen literal vectors before encoders; the report does not claim those vectors or implementations exist. Bootstrap projection consistency, operation-option predicates, preparation nonpromotion and actual same-owner nesting have been added to the required controls.

The resource-lock/publication-guard distinction and structural cgroup-root refinement remain explicit design rulings for root ratification. No namespace-root-as-global shortcut or fake zero row was introduced. Missing allocator/kernel/helper coverage is still missing, all22 components remain required for actual applicable work, and finite observation/publication/storage/inode demand is now an explicit source-model term. Exact source-proven crypto inapplicability in a non-crypto preparation operation is not promoted into an HE model pass.

The later retained-acquisition commit `d1aa3af5f706c2f42a634f293749a7618cfb772d` is correctly disclosed as a future model delta: Rc backing, original/cloned guard FDs and pre/post mount checks must be included for that source. BASE map counts are not represented as complete for the later implementation. This review neither inspects nor accepts that newer source.

The fixes add one small bootstrap record, narrowly scoped helper FD handoff, and concrete preparation operations. Their complexity follows the four identified gaps; they do not add a general resource language, scheduler, signing service, reservation registry or new administrator dependency. Serialization and retained-evidence cost remain real and unmeasured. Existing all21/full2680+20+7/quick268, CPU/GPU/external/baseline scope, source/licensing rules, scientific thresholds and holds are unchanged.

## Remaining work and authority

No blocking design finding remains open within this FIX1 scope. Actual trusted host origin, the compiled B0/bootstrap bound, complete source-specific model, exact Python/Git/allocator/kernel behavior, native observer/lock controls, byte-input wiring, live gates and scientific execution remain unestablished by this review. Missing evidence must not be replaced by zero, synthetic positives or historical serialized-size counts.

Only this review file was written. No source/Cargo/public-document edit, project helper, test, --list, native/runtime/HE execution, SSH, process/data scan, dependency install, spending, commit/push or subagent was performed. Proposal, diff, report, original snapshot, accepted map and original review were not changed. This PASS supplies a scoped review result for root's full read; it grants no automatic authorization to implement or run.
