# Post-FULL result collection map preparation

Documentation only, prepared 2026-11-08. No collector, server request, new experiment or measurement was executed by this preparation. The map in `docs/post-full-queue.md` binds the planned intake to the actual deployed queue and pinned producers, and keeps execution status, SKIP, failure and scientific acceptance separate.

Initial independent review requested one correction: the producer has a one-stream sequential baseline and two-/four-stream batch variants, not a separate one-stream batch measurement. The exact one-sentence fix passed independent scoped SPEC and QUALITY review. Final document SHA256: `73e0f902c2d7d45d342fdc8641f9508d32f4fec6196995a8f14abdb6310c5bd7`.

Reports and reviews are byte-for-byte copies of the original local records. The two raw diffs are gzip-compressed with mtime zero; decompressing reproduces their original bytes, including unified-diff context whitespace. The initial reviewed document SHA256 is recorded in its review; the final review verifies that reversing only the accepted sentence reproduces that exact digest. No executable tests were needed or rerun for this prose-only change. Root ran `git diff --check` on the document.

The referenced 14:22:14 JST FULL-2 observation is existing evidence, not a fresh observation by this task. It recorded 865/1000 updates, absent final JSON and an owner-only queue state. Actual collection remains a future separately registered bounded action. The published preview is unchanged.
