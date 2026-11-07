# ckks-train-bench

Part of **ckks-train** (<https://github.com/kenmaro3/ckks-train>). Not published.

Benchmarks and the experiment binaries of the paper:

- `iris`: the 2020 reproduction (Iris, `seal_2020`);
- `mnist`: MNIST / Fashion-MNIST training, interactive or non-interactive, on the CPU or the
  GPU (`--features cuda`), including ReBoot's eMLP-1/2/3 configurations (`configs/`);
- `mnist_reboot_layout`: the ReBoot slot layout as a baseline;
- `plain`: plaintext twins of the encrypted runs;
- `ablation`: the RQ3 ablation of hoisting, BSGS and lazy relinearization;
- criterion benches `ckks_ops` and `cuda_ops`.

Every binary writes its own provenance (commit, command line, threads, load, memory) into its
result JSON. See `scripts/run_all.sh` and `results/README.md` in the repository for
reproduction. Data credits are in `data/README.md` and `THIRD_PARTY_NOTICES`.

### Explicit StudyTrain planning inspection

`study_train` still refuses execution: live gates, admission, durable start and scientific
publication are unavailable. Existing invocations without planning options retain their
provenance inspection behavior. To inspect a canonical CPU Main/BatchedDiagonal plan,
supply **all** of the following options together with the existing authority, case and
runtime options; a partial group, duplicate or unknown option is rejected:

```text
--consumption full|quick
--physical-lanes auto|<positive usize>
--transpose dedicated|reuse
--transpose-pool none|<ordered comma-separated i32 pool>
--eval-hoist 0|1 --eval-bsgs 0|1 --eval-lazy-relin 0|1
--partitions contiguous
--receipt-max-bytes <usize> --receipt-max-records <usize>
--work-geometries <usize> --work-initial-states <usize> --work-updates <usize>
--work-handoffs <usize> --work-wrappers <usize>
```

Integers use canonical decimal spelling: no leading zeroes, plus signs, spaces or
negative zero. `auto` explicitly requests the registered batch's existing next-power-of-two
lane rule. Dedicated transpose requires `none`; reuse requires an explicit nonempty pool
in the supplied order, with reachability checked by the existing geometry validator.
`quick` consumes exactly the seed0 two-update prefix while `--scope full` still names the
full registration. Receipt/work caps are the original unchanged request limits, including
zero caps; no defaults enlarge them. Eval flags must match the sealed registration.
These requests do not clear any gates or the encrypted-block hold, enable unsupported
baseline/CUDA/packed/external adapters, or invoke the CPU continuation.
