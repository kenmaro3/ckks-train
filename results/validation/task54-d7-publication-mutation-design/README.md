# Shared publisher mutation design

Design ratified after independent SPEC/QUALITY PASS. This archive contains exact copies of the brief, proposal, full review diff and independent review. Source implementation and native validation are pending; the current pathname check/use defect is not declared fixed.

The chosen shared mechanism uses an owned anonymous O_TMPFILE inode, a validated retained procfs FD directory and no-clobber linkat, with drop-only cleanup. Reference, Diagnostic, Failure, Target and future D7 share the migration. The proposal records safe rustix APIs, the new direct Linux dependency edge, Linux 5.6/openat2 support floor, refusal without weaker fallback, five-operation durability contract and explicit revised test counts. Existing formats/caps, legacy recovery, scientific criteria and authority remain unchanged.

See [proposal](task-54-d7-publication-mutation-proposal.md) for reasons, alternatives, exact operations and native proof obligations; [independent review](task-54-d7-publication-mutation-review.md) is design evidence only. No Cargo, native scratch test, SSH or scientific run was authorized by this archive. Primary sources were consulted for semantics, without copying external implementation.

SHA256SUMS covers the other five files, including this README.
