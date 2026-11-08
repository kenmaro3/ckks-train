# Pre-edit ruling

Ruling: Serialize only the four integration tests sharing Fixtures.budget with a fixture-local Mutex held for the full test; retain cumulative budget enforcement and compute canonical assertions from per-test usage deltas. Reason: request() clones one Arc<PlanningBudget>; production counters are correctly cumulative, while canonical reads totals and inventory reads a before/after interval that parallel tests can contaminate. Cost if wrong: fixture tests might still depend on order or permit a missed counter assertion; default and explicit-two-thread full-target runs cover the existing direct failure trigger.

Callers: inventory, canonical, case projection, late finalization all use request() and Fixtures.budget; projection currently rejects before charges but remains protected. Bootstrap configuration binding constructs its own fixtures and budget and remains independently parallel. main_case and parameters only read shared immutable state. Iris OnceLock remains shared and materializes once.

Alternative rejected: independent per-test budgets would change the existing cumulative 148 geometries / 148 initial states / 8 updates / 0 handoffs / 8 wrapper-call limit. Whole-suite single-threading or increased caps are forbidden and unnecessary. Keep all numeric caps and all existing assertions. Existing workspace and target-parallel RED plus isolated canonical GREEN already demonstrate the exact defect; no additional redundant failing test is needed.

Original evidence is retained under research-preview-evidence/20261107T235505.137343Z-tests, 20261108T000402.372853Z-study-parallel and 20261108T000417.211472Z-study-isolated. This correction provides test isolation only and no new production authority or scientific qualification.
