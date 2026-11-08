# Task55 changelog publication-state correction

2026-11-08. CHANGELOG.md had retained the pre-publication statement that nothing
was published and compared Unreleased against the uncreated stable v1.0.0 tag.
It now records the actual fixed v1.0.0-preview.1 research preview, distinguishes
workspace version1.0.0 from stable publication, and compares against that preview.
The planned stable heading links to release guidance until the stable release exists.

The exact changed text and primary publication record passed independent
[documentation review](review.md); the reviewed CHANGELOG SHA256 is
9e29a3d5fa48cc7b0633e0ac5e4a7f14f4bfb5ea92ad972e0e611f2210f8275a.
Root checked the diff, whitespace, reviewed-file identity and exact report bytes.
No source behavior, result JSON, release tag or published asset changed.
No test/build was needed for this documentation correction. Earlier source-specific
CI and distribution evidence is not extended to current Rust work in progress.
