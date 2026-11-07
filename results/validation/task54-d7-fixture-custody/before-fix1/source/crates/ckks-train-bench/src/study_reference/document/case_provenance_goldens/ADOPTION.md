# Case component fixture adoption

Copied byte-exact from the root-adopted Task54 case-provenance-vectors package.
Manifest SHA256: 55aa509322d9e98a927d89e3c91d745faa1cce758c015b034ef8ae7f3b9d61aa.
Author report: 1393a1c2b4575ba867de1351e750c8bcfcebedabc3616af65de28b33217fd6ff.
Independent review: e1b065536ae83f4ad60775c19d82266c642f27a27d6dea76e941ce4dec6734fa.
Original SHA256SUMS: 660e1a9cf6d01fab6b1541c6b990986a9f15a16a219d0c2052bdbfff79346336.

Six positive components and67 negative literals test untrusted typed parsing,
canonical encoding and completed-claim consistency only. No actual captured
Train facts, checked D7 binding, standalone artifact envelope, production
admission/start/HE or native runtime approval is supplied by these files.
The original77 target goldens and Reference scalar1 policy remain unchanged.
Full copy hashes live in task-54-case-provenance-fixture-adoption.json in the ledger.

## Exact literal custody compaction (Task54 build prerequisite)

The original payload files, including manifest.json and SHA256SUMS, now reside byte-exact in literals.json under their original relative names. The outer JSON stores original UTF8 text, byte length and SHA256; the test-only cached accessor returns decoded text bytes without parsing/reserializing the inner document or normalizing whitespace, hex case or trailing bytes. This keeps every original independent expectation and malformed/duplicate-key negative. The adoption history above is unchanged.

Root prepared the bundle from original BASE abd6c3393b843d7d80fff4ebc890526866682c3b. Before removal, every member was independently compared against exact released BASE ec2d2656f21e9139d3dde12abdce13d770ea2757 Git objects and a frozen whole-family path/length/SHA256 inventory. The ledger task-54-d7-fixture-custody-original-inventory.json and task-54-d7-fixture-custody-check.py retain that check. These are fictional test documents, not scientific results. Both families remain inside source-root coverage; compaction preserves the unchanged512-record source cap.
