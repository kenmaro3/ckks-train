# Actual typed-materializer follow-through at d1aa3af

Source inspection only; not an implementation/runtime release. Four parser/profile/materializer modules are byte-identical to map BASE71d3f09 (pins below). The retained-reader change therefore has not silently fixed the earlier path reads.

Confirmed current run_owned still loads context/registry/resolution then calls materialize before selecting parameters/planning. ResolvedProfile.prepare_dataset does captured preflight, locked dataset pre-read, then pathname load_dataset and post-load digest comparison. profile_data.load_dataset separately reads/hashes CSV/partition/IDX/NPY paths, and preflight_seed reads captured arrays/order/schedule. These real paths need SAME retained input slices and one actual resource owner; adding a parser used only by tests is insufficient.

Existing parse_npy_f64 already checks payload product/length before collecting values; captured ArrayInput.shape and finite-value checks currently occur after decoded allocation. Preserve canonical dtype/Fortran/value behavior while introducing the actual expected-shape sizing/check before the affected allocation. Preserve IDX dimensions, CSV splits, WDBC digest, exact train/test populations, resize/minmax/RNG, initialization/order/schedule and derived F32 values. No baseline code/excerpts are needed.

The real reference reader and target canonical reconstruction independently call materialize. Nested operations must borrow the SAME resource owner and standalone preparation must use actual non-HE work selection, as resource design FIX1 requires. No fake Full/Quick request or post-materialization permission. The d1aa3af store Rc/original-guard FD and extra mount/proc checks add real costs to the old map; final source-specific model must include them. No generic budget token, resource model number or approval flow is authorized here.

```json
{
  "head": "d1aa3af5f706c2f42a634f293749a7618cfb772d",
  "files": [
    {
      "path": "crates/ckks-train-bench/src/data.rs",
      "bytes": 58637,
      "sha256": "548c647d832e860d6daadd48ac8832fb185c788365798b32941ee650ee752225",
      "unchanged_from": "71d3f0937741a4ff8d156b36540332ea82203d3c"
    },
    {
      "path": "crates/ckks-train-bench/src/profile_data.rs",
      "bytes": 28565,
      "sha256": "1de6c85f1baaf4a292a66598e679125ba511f4cf50b340d3ae92693a54d957dd",
      "unchanged_from": "71d3f0937741a4ff8d156b36540332ea82203d3c"
    },
    {
      "path": "crates/ckks-train-bench/src/study_profile.rs",
      "bytes": 79141,
      "sha256": "eaaf3ad085f1a347b664099f720a1ad2fb068ada66939baf60c3c6bbefe3670f",
      "unchanged_from": "71d3f0937741a4ff8d156b36540332ea82203d3c"
    },
    {
      "path": "crates/ckks-train-bench/src/study_inputs.rs",
      "bytes": 66327,
      "sha256": "fd641f27fc454ad85eb3c1ef67db759459a1ad22dc2a42dff15373c9650063a9",
      "unchanged_from": "71d3f0937741a4ff8d156b36540332ea82203d3c"
    }
  ]
}
```
