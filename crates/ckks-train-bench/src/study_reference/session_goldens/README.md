# D6c2b byte fixtures

`prefix_*.hex`, `state_*.hex`, and `fixture_inputs.json` are exact unmodified
copies of the independent accepted D6c2 oracle under the Task54 SDD workspace's
`d6c2-independent-goldens/` directory. They were not generated from Rust encoders.
The input JSON explicitly describes fictional synthetic values, not an execution,
approval, build or scientific result. The positive-zero variant changes only the
first weight scalar of the first layer, preserving other signed zeros.

`adversarial_reference.json` and `adversarial_captured_reference.json` copy the
existing D6c1b independent transport literals (`REFERENCE` and
`CAPTURED_REFERENCE` in `document_goldens.rs`), adding a final newline only.
Their role is adversarial untrusted parse claims. No test obtains a production
completion, VerifiedReference, publication, target or approval from these bytes.
Reader mutations recompute identity and envelope hashes to ensure that integrity
alone cannot bypass the canonical and approved-policy comparison components.
