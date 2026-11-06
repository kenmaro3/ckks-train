# R-REB-6-OBS: current producer environment, read-only

Observed 2026-11-07T04:32:19.620754+09:00. Original stdlib observer supplied over key-only SSH and run with nice10; exit0, controller wall 40.113029s including connection. No scientific/native/baseline imports, builds, training or queued workload. No native performance/RAM measurement.

`inventory.json` is byte-exact observer stdout, not an accepted scientific result or historical runtime attestation. `meta.json` binds command/sourceSHA and wall observations; `observer.py` is the exact original supplied source. Large native files are stat-only. Current versions include torch2.15.0+cpu / torchvision0.29.2+cpu; the three torchvision transform files, baseline data loader and two historical wrapper scripts match the locally inspected source hashes. This does not establish historical native identity or prove the cause of Mac image mismatch.

Two observer path limitations are retained without modifying raw output:

- `full2_final_result_exists` checks an incorrect generic `out/R-41-FULL-2/result.json` path. Ignore this field for terminal status. Authoritative producer output is `/home/kmihara/hekit-r41full/results/quadruped/mnist196_ni_b48_full_seed0.json`, with runner post/meta as separate terminal evidence.
- The guessed `experiments/4_Scalability/precision_analysis.py` source path is absent. Actual exporter uses `experiments/3_training_encrypted/precision_analysis.py`; that file was not inventoried here. The absent entry is not proof of missing baseline configuration.

The authoritative FULL-2 log tail is retained as observed: zero-based update28 (29 completed of1000), elapsed4199s, recent updates1369KS/4bootstraps and predictions true. This log is progress evidence, not a current-process liveness or final accuracy result. No restart/termination/queue mutation occurred. The inventory is an untimed read-only observation overlapping FULL-2 and must not be represented as zero overhead.
