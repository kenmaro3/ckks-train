# Task54 D7 custody inventory FIX1 scoped independent review

Reviewed 2026-11-07. **SPEC: PASS. QUALITY: PASS. I1: ADDRESSED.** No remaining findings in this scoped document correction. Acceptance is of the inventory's accuracy and required test coverage, **not** acceptance that the current publisher provides production custody safety.

Read the same custody inventory brief, complete appended report, full FIX1 diff and pins. Re-reviewed only original I1 and possible breakage introduced by its correction. Unaffected C01–C10 architecture, C1 composition and fixture separation were not reopened.

Exact reviewed pins:

- Inventory: `9276aabb55edc10e11e0191e87d807ebb354b1fce8335242b2bf19aa13978aa5`.
- Appended report: `fa6d024c5de7afe8dbafcd6db974f5e82bc04cbae9253bfdbcccce645573356a`.
- Complete scoped diff: `edb89f39ac2e2cb36e4f10a55f876400444d363e024d618b091c8f80b6a3ce6a`.
- Pinned publication source, still unchanged: `8cfabba4dcf7daae93e90ce40a0bc1bcf13c59c713bb5c9d65964b97a4a2c004`.
- Original review remains `e662465b71a488266559958b8935a22a11c6b321d48af5b3502a117a24d65906`.

## I1 disposition

The current-source description now promises rejection only for a replacement visible at the inode comparison. It explicitly distinguishes that check from the later pathname operation, qualifies the original30 before-check and60 create-mode cases, and removes the unsupported unconditional preservation claim. Required preservation remains a separate unresolved source/proof prerequisite.

All six after-check/before-path-syscall cuts are concrete and agree with the pinned helper control flow:

| New suffix under `d7_custody_c04_postcheck_` | Outer Ops count | Current-code result correctly recorded |
|---|---:|---|
| `link` |6| linkat publishes replacement R; later U rejects, VisibleUnconfirmed, no D2 |
| `unlink_postlink` |7| Final still O; replacement T→R removed; primitive may return Ok |
| `unlink_write_cleanup` |3| W fails before progress; cleanup removes replacement; BeforeLink |
| `unlink_file_sync_cleanup` |4| Complete temp, sync fails; cleanup removes replacement; BeforeLink |
| `unlink_link_cleanup` |5| Atomic link failure has no final; cleanup removes replacement; BeforeLink |
| `unlink_fchmod_cleanup` |1| Internal cleanup within C removes replacement; no second outer U; BeforeLink |

The table separately records Native L/U invocations, successful comparison helper calls, actual path-call success/failure and actor rename1/create1. It retains both original checked descriptors, explicitly locates moved O at M outside the selected attempt directory, and distinguishes final F, temporary T and replacement R states. This avoids an accidental unknown-entry refusal masking the targeted window. The6×15 expansion correctly adds90 planned cases without silently replacing earlier counts.

The `link` cut does not turn a successful link side effect into an erroneous BeforeLink classification: L succeeds and the later U failure yields VisibleUnconfirmed. The normal postlink unlink case truthfully records current primitive Ok despite foreign-entry deletion and notes that later correct-final checks need not detect or repair the lost entry. No such result is permitted to count as successful custody safety or a live fixture capability. Pre-link cleanup cases retain their actual BeforeLink return while separately recording preservation failure; this is the correct distinction between return classification and mutation safety.

The six-schedule future native companion specifies barriers after comparison and before the path call, rather than a pre-helper barrier or timing-only probe. It is explicitly unexecuted. The required source prerequisite is a separately reviewed concrete mechanism/proof binding mutation to the verified inode or actually enforced exclusion over the entire window. Retained FDs, cooperative flock and optimistic inode-conditional fake behavior are explicitly insufficient. The inventory chooses no new architecture and does not claim that source prerequisite is resolved.

## Independent verification and limits

One read-only inline stdlib inspection, exit0, verified all5 pin-file entries, both complete diff bodies through their7 exact before/after hunks, both archived originals, the unchanged3470-byte report prefix, current publication source hash, six row counts and90-case arithmetic. Source behavior was assessed against the already-reviewed pinned call sequence; no project code was executed.

The correction changes only inventory/report text and review metadata. No format/cap/golden, Full/Quick scope, policy or authority change appears in the diff. Prior fixture and implementation dependencies remain explicit, including the independent coherent Full fixture and retained-FD/exclusive-attempt/multi-file recovery seams. Inventory acceptance can support subsequent authorized source work, but it cannot be cited as proof that post-check replacement preservation or native custody has passed.

No source edit, Cargo, compiler, test, --list, runtime, native operation, generator, HE, SSH, subagent, commit or push was performed. Only this review artifact was written. Current authority and all21/full2680+20+7/quick268, holds, original scientific/security thresholds, baselines/Task5 and MIT/Apache2/BSD/ISC plus Unicode-tables-only exception remain unchanged.
