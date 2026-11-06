# Outer budget derivation (symbolic only)

Ratified C=2,613,526, R=31,584,142, A=2,097,152 are unchanged. Enforce both independently capped decoded blobs, total file bytes, and outer raw bytes before allocating from input claims. Enforce canonical outer budget independently while appending. No cap is selected by a claimed profile. These are logical byte bounds, not measured RSS or allocator capacity.

| Layout | Only excluded JSON raw string contents | File cap | Canonical cap |
|---|---|---:|---:|
| lock | `contract.canonical_bytes` | 2C+A = 7,324,204 | C+A = 4,710,678 |
| admission | `contract.canonical_bytes`; `selected_input.seed_receipt.canonical_bytes` | 2C+2R+A = 70,492,488 | C+R+A = 36,294,820 |
| start | none | A = 2,097,152 | A = 2,097,152 |

Quotes around those hex strings, keys, separators, whitespace, all other strings, and all other escapes consume A. Only the exact decoded object paths qualify; no suffix/prefix match or key-substring detection. Hex contents are unescaped lowercase and even length; each byte contributes exactly two raw hex bytes. Escaped spelling at an exempt path is rejected, not deducted from the budget. The ordinary RawOsString.bytes field is never a large-blob exemption. Canonical subtraction removes only decoded identity contents; their u64 length prefixes remain in A. Header domain and body schema are both charged.

For ordinary Text(n), canonical bound is 8+n; raw token bound is 6n+2. Git40 and Hash64 canonically cost48 and72; escaped raw ceilings242 and386. Enum tags cost1 canonically (raw JSON spelling uses at most6*name-length+2). U64 costs8 canonical and at most20 JSON digits; F64 costs8 canonical and at most64 raw number bytes. Bool costs1 canonical and at most5 raw bytes. Option costs1 plus the present value; null raw spelling costs4. List(n,T) costs8+n*canonical(T); compact JSON at most2+n*raw(T)+(n-1). Object JSON contributes2 braces plus each escaped key token, colon, value and commas. A further amount W of arbitrary valid whitespace is charged exactly to A: it is not bounded by the no-whitespace estimates below.

Ordinary decoded text limit4096 gives raw token24,578; Text128 gives770; compiler16384 gives98,306. Unix bytes4096 are transported as hex of decoded text length8192, so ordinary raw-token ceiling49,154 (12*4096+2); canonical cost4104 plus encoding tag1. The field bound for Bytes at large identity paths is instead2*blob+2 raw and8+blob canonical, with no escaping. Integer token length64 preflight remains a separate conservative lexical cap; typed U64 rejects anything outside its narrower grammar/range. Depth16 and decoded keys64 are independent refusal conditions and are not allocation budgets.

Inherited lists remain manifest<=512, features<=64, argv<=256, environment transport<=32 but exact seven unique named keys, exact five design documents, endpoint crossed boundaries<=400. One crossed-boundary vector alone costs8+400*8=3,208 canonical bytes, compact maximum raw8,401 bytes (400 twenty-digit integers and separators). There is one Endpoint in a lock, no duplicated endpoint sequence. SourceFile(path4096,Hash64) canonical size4176; the maximum 512-entry manifest list is8+512*4176=2,138,120, ALREADY above A before other fields. Thus individual maxima do not imply global admissibility. No huge list/blob was allocated to establish this arithmetic witness.

`expected/cap-accounting.json` is the machine-readable recursive inventory of EVERY new and inherited field and nested record/list contribution, using the explicit schema table in oracle.py. Header accounting uses the exact production domain lengths (one byte longer than the private test domains). Bounds intentionally use enclosing transport text caps even for fields whose exact approved constant/path is shorter, and use six-byte escaping even where stronger downstream syntax disallows some escapes. These are conservative per-field upper ceilings, not a claim that every combination or exact maximal string is semantically admissible. Actual fixed constants are encoded at their exact observed lengths in the literals. Array child rows give per-item contributions; parent rows give the cap-multiplied contribution. Do not sum recursive parent and child rows together.

| Independent field ceilings combined (before W whitespace) | Canonical outer upper ceiling | JSON outer upper ceiling |
|---|---:|---:|
| lock | 3,623,872 | 28,117,273 |
| admission | 3,626,634 | 28,145,835 |
| start | 3,607,570 | 28,027,254 |

These intentionally exceed A and are NOT new accepted caps. Captured inherited provenance contributes3,580,613 canonical bytes for reference type /3,580,614 case type at these independent maxima; outer non-provenance additions are small relative to A but still charged to it. Source list512 and argv256 can consume too much in combination or separately. If those valid-shaped individual values consume A, reject globally. Increasing A or truncating fields is forbidden.

| Actual frozen small fixture | JSON outer bytes | Canonical outer bytes |
|---|---:|---:|
| lock | 9,116 | 4,093 |
| initial admission | 10,456 | 5,247 |
| start-initial | 7,690 | 3,761 |

The ledger has compact arithmetic limit/limit+1 descriptors in cap-accounting.json: outer A with maximum individual blobs has the ratified totals; outer A+1 is refused even with zero blobs and a small total admission file; C+1 or R+1 is refused separately. These are symbolic size-bound cases only, not valid completed artifacts or parser-run observations. Full Fashion blobs and large synthetic cap-sized buffers were not generated. All21-profile input inventory, memory/RSS, native platform preflight, and actual parser boundary execution remain outside this package.
