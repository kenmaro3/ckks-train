# Read-only Iris Full checkpoint derivation for CW-04

This is a source derivation, not an executed BatchPlan, test or scientific result. No registry resolver, dataset, parameter, key, context, materializer or native function was invoked.

experiment_matrix.json:207–218 registers Iris train120/test30, primaryB20,400epochs; profile iris__emlp1 at435–479 independently specifies B20,logical_epochs400,drop_last=false and local_per_epoch order. study_profile.rs:750–755 selects those metadata values. build_plan at1650–1747 uses noncaptured local segments, per_pass=population120, total=120*400=48000, segments400, segment_samples120. Each inner loop consumes min(20,remaining), so six updates of20 per epoch, no remainder. next_boundary begins1; the crossed-boundary loop pushes each boundary only once, and the emitted checkpoint stores updates.len(), total processed and that crossed list.

Consequently planned Full checkpoint k, for k=1..400, has completed_updates=6*k, processed_samples=120*k, crossed_logical_boundaries=[k], exact_logical_boundary=true and interval_updates=6/interval_samples120. First=(6,120,[1]); second=(12,240,[2]); last=(2400,48000,[400]). The complete Full result must use the full schedule. Failed/incomplete records retain only actually completed checkpoint prefix observations; this derivation does not authorize fabricating a missing evaluation after failure. Runtime still obtains/compares the actual registered plan and held source input, not this note.

This independently confirms the registered coordinate in CW-04/R-WIRE-3. An endpoint atupdate1 or crossed=[0] is not the first registered Full checkpoint even if target/time fields are changed consistently. No other dataset/model/packing schedule is generalized from these constants.

Exact read source pins:
- crates/ckks-train-bench/src/study_profile.rs: 79141B SHA256eaaf3ad085f1a347b664099f720a1ad2fb068ada66939baf60c3c6bbefe3670f
- crates/ckks-train-bench/configs/experiment_matrix.json: 96937B SHA256164dbc67fddb87bed814bfa9bcbad1f97508a4a2de265b7a8ca899d8fab1b4cc
