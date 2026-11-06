# ckks-train-nn

Part of **ckks-train**, an MIT-licensed Rust RNS-CKKS library for encrypted neural-network
training (repository: <https://github.com/kenmaro3/ckks-train>). The workspace uses the
`ckks-train-*` crate names.

Encrypted MLP training on `ckks-train-ckks`, generic over the backend (`HeOps`, CPU or
`ckks-train-cuda`):

- `layout`: the slot layout of weights and activations (feature-major, batched);
- `dense`, `activation`, `loss`, `optim`: encrypted forward and backward passes, polynomial
  activations, squared-error loss, SGD and Nesterov momentum;
- `trainer::Trainer`: **interactive** training; the key owner refreshes the levels
  (`refresh::ClientRefresher`);
- `trainer_ni::NonInteractiveTrainer`: **non-interactive** training; the server restores the
  levels by bootstrapping, with compact repacking and the bootstrap placements
  `AfterUpdate`, `WhenNeeded` and `AfterActivation`;
- `block::BlockMlp`: block MLPs with local classifiers (the structure of ReBoot's eMLP-2/3);
- `sim`: a plaintext simulator of levels and operation counts.

Research code, not audited: see the security note in the
[repository README](https://github.com/kenmaro3/ckks-train#security).

License: MIT.
