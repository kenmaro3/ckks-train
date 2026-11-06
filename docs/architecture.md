# ckks-train architecture

A short map of the crates, the types that matter and one training step. The detailed design
documents are `docs/layout.md` (slot layout, cost model), `docs/bootstrap.md`,
`docs/precision.md`, `docs/security.md` and `docs/gpu-bootstrap-design.md`. The workspace uses the
`ckks-train-*` crate names.

## Crate graph

```
ckks-train-core            RNS / NTT arithmetic, base conversion, ChaCha20 sampling
    ^
ckks-train-ckks            RNS-CKKS, key switching, BSGS, bootstrapping, HeOps / BootstrapOps
    ^        ^
    |     ckks-train-cuda  CUDA backend: impl HeOps (CudaEvaluator), BootstrapOps (CudaBootstrapper)
    |        .
ckks-train-nn  .  .  .  .  (dev-dependency of ckks-train-cuda only: GPU training tests)
    ^        ^
ckks-train-bench           binaries and benches (not published); feature `cuda` adds ckks-train-cuda
```

`ckks-train-nn` does not depend on `ckks-train-cuda`: the trainers are generic over the `HeOps` trait of
`ckks-train-ckks`, and the backend is chosen by the caller (`ckks-train-bench`).

## Main types

| type | crate | role |
|---|---|---|
| `CkksParams` | ckks | ring degree, RNS chain `q_0..q_L`, special primes `P`, `dnum`, secret distribution; presets `seal_2020`, `n16_train`, `n16_boot`. Shared as one `Arc<CkksParams>` |
| `Encoder`, `Encryptor`, `Decryptor`, `KeyGenerator` | ckks | encoding (bit-exact with SEAL 4.1.2), encryption, key generation |
| `Evaluator` | ckks | CPU evaluation with public keys only: arithmetic, rescale, hybrid key switching, hoisted rotations, BSGS `matvec_enc`; counts every operation in an `OpCounter` |
| `HeOps` (trait) | ckks | the evaluator interface the trainers use; `E::Ct` is the backend's ciphertext. Implemented by `Evaluator` and `CudaEvaluator`. `EvalConfig` switches hoisting, BSGS and lazy relinearization for the ablation |
| `Bootstrapper` | ckks | CKKS bootstrapping on `n16_boot`: SSE switch to a sparse key, ModRaise, CoeffToSlot, EvalMod, SlotToCoeff, switch back. Implements `BootstrapOps<Evaluator>` |
| `CudaEvaluator` | cuda | `HeOps` on device ciphertexts (`CudaCiphertext`), bit-exact with `Evaluator`, same counts and errors |
| `CudaBootstrapper` | cuda | `BootstrapOps<CudaEvaluator>`: the same bootstrapping on the GPU |
| `Mlp`, `Dense`, `Layout` | nn | encrypted weights as diagonals, feature-major batched slot layout |
| `Trainer` | nn | interactive training: after each update a `Refresher` (`ClientRefresher`: the key owner decrypts and re-encrypts) restores the levels |
| `NonInteractiveTrainer` | nn | training without the key owner: compact repacking (`CompactLayout`) and bootstrapping at the `BootPlacement`. Holds no secret key by construction |
| `BlockMlp` | nn | block MLPs with local classifiers (ReBoot's eMLP-2/3), trained stage by stage by `BlockTrainer` / `BlockNiTrainer` |

## One training step

Non-interactive mode (`NonInteractiveTrainer::train_batch`), one update of `k` micro-batches
of `B` samples each; the interactive `Trainer` runs steps 1 to 3 the same way and then
refreshes through the client instead of steps 4 to 6.

```
client: encode + encrypt batch x, labels y  --upload-->  server (HeOps backend: CPU or GPU)

server, per micro-batch:
  1. forward     z_l = matvec_enc(W_l, a_{l-1}) (BSGS, hoisted rotations), a_l = act(z_l)
  2. backward    delta_L = a_L - y; per layer dz = delta * act'(z) (one rescale),
                 delta_{l-1} = W_l^T dz (transpose by rotations only)
  3. gradients   dW summed over the micro-batches, reduced over the batch, masked and
                 shifted into compact form in one plaintext multiply
server, per update:
  4. optimizer   NAG on the compact weights and velocity, SGD on the biases (exact scales)
  5. bootstrap   each compact ciphertext when the placement asks for it
                 (Bootstrapper / CudaBootstrapper: 111 key switches, 16 levels after)
  6. unpack      compact weights -> diagonals for the next forward pass

client (evaluation only): download + decrypt the weights or predictions
```

Every operation goes through `HeOps`, so the same step runs on `Evaluator` or `CudaEvaluator`,
and the `OpCounter` difference of the step equals the closed-form prediction
(`trainer_ni::predict_update`), which the benchmarks check on every update.
