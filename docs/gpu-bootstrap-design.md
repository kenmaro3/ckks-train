# Task 49: GPU bootstrap implementation decision (2026-11-06)

Before implementation, the controller approved this design:

- Share the existing level-optimal Chebyshev/EvalMod executor through a narrow
  `BootstrapArithmetic: HeOps` extension. Preserve the CPU operation order,
  integer rounding, exact scale labels, and counter rules.
- Reuse CPU bootstrap encoding/key generation through read-only accessors.
  Upload public diagonals and keys once; ciphertexts remain on the CUDA device
  through SSE, centered mod-raise, CtS, EvalMod, and StC.
- Use the existing CUDA basis-conversion and pointwise kernels. A level-zero SSE
  key needs a narrowly checked upload path; ordinary typed keys keep their checks.
- Merge counters only after the complete operation and device synchronization
  succeed. Device allocation failures remain fallible `Device` errors.

Alternatives considered: copying the CPU polynomial evaluator risks divergence
in delicate rounding and scale logic; genericizing the entire CPU bootstrap
would create a larger regression surface. Host fallback would not implement GPU
bootstrapping and is not used. No new dependencies or external implementation
source are needed. No changes to training interfaces or API names are planned.

The current parameters are the implemented SSE h'=32, q0=54-bit, CtS=56-bit,
EvalMod=7 levels without double angle, and 16 remaining compute levels; the
superseded original plan constants are not used.

Local evidence is CPU execution/replay and CUDA compilation only. Full-slot GPU
precision, CPU/GPU bit parity, GPU memory peak and latency remain unmeasured.
The ignored n16 gate must preflight free GPU memory before generating large
host keys and report a skip as unmeasured. No remote job is authorized here.

## External timing reference and comparison boundary

The controller verified the primary [HEonGPU performance appendix](https://heongpu.readthedocs.io/en/latest/appendix.html#performance-benchmarks)
on 2026-11-06: the CKKSBootstrapping table reports RTX 4090, Regular,
N=2^16, 2^15 slots, **9 remaining levels**, median of 5 runs, **109.28 ms**.
Its surrounding text specifies main-secret h=192 and temporary-secret h=32.
Our implemented preset has a dense main secret and **16 remaining levels**;
these are different workloads, so the plan's fixed 220 ms target is a target,
not a normalized performance claim. No HEonGPU implementation source was read
or incorporated for this task. GPU timing here remains unmeasured.

## Local verification of this implementation

The full-slot CPU n16_boot gate was rerun on 2026-11-06: 17.46 bits for all
32768 random real slots in [-1,1], level 16, 111 key switches (69 rotations
including conjugations, 40 relins, 2 SSE). The miniature N=1024 device-pipeline
orchestration run on the CPU matched CPU residues and every counter exactly,
with 26.73-bit precision. CPU kernel replay also checked centered mod-raise
half-modulus boundaries and both SSE key switches. Injected work/sync errors
left caller counters unchanged. These are CPU results, not measured GPU parity.

The ignored GPU gate compiles through the existing generic BootstrapRefresher.
It preflights 20 GiB free device memory for n16 before generating host keys.
Insufficient memory prints `SKIP UNMEASURED`; a successful test process with
that message is not evidence that the precision or timing gate was executed.
Keys and transform diagonals are uploaded once. Public integer constants are
currently expanded on the host and uploaded during EvalMod; all arithmetic on
encrypted values executes on the device. Their transfer overhead is part of
any future timing and has not been optimized or measured here.
