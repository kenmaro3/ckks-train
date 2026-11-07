# Frozen independent wire vectors

Synthetic format examples only. No repository encoders, parameter/planner constructors, HE, native calls or evidence sources were used.
The complete inputs are in inputs.json; complete canonical bytes in each named .hex; manifest.json fixes offsets, byte lengths and SHA256.
All a/b hashes are literal placeholders. Positive means a byte-oracle example, not a production-valid object. Every production producer still derives fields from actual retained objects.
Security Some(DenseTernary), fixed main, and non-default ephemeral examples explicitly test transport branches unsupported by current production policy. Layout values are hand-written format specimens, not derived StageGeometry evidence.

| Vector | Bytes | SHA256 |
|---|---:|---|
| parameter_dense | 191 | `1610674445f652ed0bb3850e691281dcfad0787c90d424d1fbd5cb7a96a86bab` |
| parameter_fixed | 199 | `5b4de5932e89676278f2395e00af59de77c0a713376d67960a24491c5da1f142` |
| layout_i_single | 651 | `899d964853d8095e353032a6ebce4568d63e1971cd8a8ea4a89af231a4a88a71` |
| layout_ni_two | 1582 | `2c8b8ce9a90cfa20d61b18a36e2b61542c5472888e6ba85942e1e50408220bff` |
| layout_i_three | 1529 | `89b1749ddec7f6fa8c3aa690ef9f21b9003e96d3a7d959dae24d09cfab6d2a41` |
| keys_i_dedicated | 206 | `3ff5a111fe2548fb4f530645bff2f6c7edc4a77a3ff5585a307f89277c8630d2` |
| keys_i_reuse | 226 | `20aae2cb808e1022df0b97c0c86b57f7b1d4a37c00a358428067505fffe915f8` |
| keys_ni_dedicated | 298 | `05457cdf5d5d2d707823b3b669cda73b84173a2a4f1ef5b521f6471ffef3887c` |
| keys_ni_reuse | 318 | `d7769613377a0bcb3acbbd6d622d70d2399beb0e7ecf2f0b08e0ea62b48140c0` |
| security_i_dense | 170 | `7edfd7f62bd72a61d4704eceb723b7a592f2a3e0840ec0f8f2a1ffc4934a0890` |
| security_i_fixed | 178 | `89be3423d6819f280242c5eee14113b3ac8563fcfd001072a82700a1b88f1f6e` |
| security_ni_absent | 252 | `2fa566e810e32df557c93899038e98de86fa468ddd820ef7ee7c3c049e15472a` |
| security_ni_fixed | 261 | `88665a13d3aad0a2f5fe83d62ecac96f0e50130194372afb2cf1a7c60adb4498` |
| security_ni_dense | 253 | `c1bb966c2d6c0351dd702580d83d18c280b96eabb1b47e5c8eee382e3195fed5` |

negative-vectors.json supplies 56 complete mutation inputs with exact bytes/length/digest and distinguishes wire failures from live/source binding failures. It additionally fixes every proper truncation, exact A_MAX boundary checks, and checked-arithmetic/conversion failures without allocating multi-megabyte filler artifacts.
A digest change alone is never a rejection oracle: semantic mutations must be refused against the unchanged retained source/route facts even if every outer digest is recomputed.

Reproduce: `python3 generate.py`. This writes only files beside this script; no project imports or constructors. Independent review must freeze these artifacts before hash-producing source. These are not test execution results or ratification.
