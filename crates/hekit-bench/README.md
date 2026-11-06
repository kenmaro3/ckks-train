# hekit-bench

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
