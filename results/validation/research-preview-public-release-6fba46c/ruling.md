# Research preview candidate update — 2026-11-08

Ruling: advance the existing unpublished preview draft from 0b2d17a to fixed commit
6fba46c38f1cdb5be88b9bf1eeecab0f4566b534, conditional on successful completion of its
own CI run 42433577945 and independent review of the prepared distribution.
This includes already-reviewed publication records, FULL-2 observation and Linux
focused evidence without including uncommitted Task54 work. The 0b2d17a bundle,
draft verification and historical notes remain unchanged in their original locations.

Why: the first public CI run at 684d7fb succeeded, while the draft's old exact SHA
had only an admission-blocked run. The latest committed candidate already has its
own executing CI; use that exact result rather than imply inherited success.
Alternative: rerun the old commit or publish the earlier green commit. Neither is
necessary while preparing the latest fixed archive and its already-running CI.
Cost if wrong: another candidate and asset rebuild; do not publish on pending or
failed CI. No source or scientific-result JSON edits are authorized by this ruling.

The user explicitly requested public GitHub code/release/PDF within two days.
Publishing this clearly labeled prerelease is within that scope. It is not stable
v1.0, an arXiv submission, or completion of the original research gates. Existing
release ID 437943352 is reused; no duplicate release. Keep FULL-2 pending and
unmeasured GPU/equal-security limits explicit. PDF and arXiv ZIP bytes may be
reused only after matching all manuscript input files to the new Git commit.
