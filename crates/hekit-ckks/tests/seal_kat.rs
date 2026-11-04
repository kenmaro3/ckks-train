//! Known-answer interop with Microsoft SEAL 4.1.2 (seal_2020 preset).
//!
//! `#[ignore]`d: the KAT files (~80 MB, not committed) come from
//! `tools/seal-kat/run_kat.sh`, which builds `tools/seal-kat/interop`, writes
//! `kat/seal_2020/`, runs `cargo test -p hekit-ckks --test seal_kat -- --ignored` and then
//! lets SEAL decrypt what these tests wrote. Without the files the tests print a message and
//! return. `HEKIT_SEAL_KAT_DIR` overrides the directory.
//!
//! * `seal_to_hekit`: hekit loads SEAL's secret key and ciphertexts, decrypts them, and the
//!   decrypted plaintexts are **bit-identical** to SEAL's own `Decryptor::decrypt` output;
//!   the decoded values match SEAL's decoding and the exact values in `values.json` (bounds
//!   in [`bound_log2`]); re-serializing every ciphertext reproduces SEAL's bytes; hekit's
//!   `add`, `mul_no_relin` and `rescale` on SEAL ciphertexts are bit-identical to SEAL's.
//! * `hekit_to_seal`: hekit encrypts with SEAL's keys (symmetric with the loaded secret key,
//!   public-key with the loaded public key) and computes on SEAL ciphertexts, then saves
//!   `hekit_*.bin`; `seal_interop_kat verify` decrypts them with SEAL (exit 0 iff within the
//!   same bounds). This test checks the files with hekit first.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use hekit_ckks::seal_compat::{
    ciphertext_to_bytes, data_parms_id, key_parms_id, load_ciphertext, load_plaintext,
    load_public_key, load_secret_key, save_ciphertext,
};
use hekit_ckks::{Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator};
use hekit_core::sampler::Rng;

fn kat_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("HEKIT_SEAL_KAT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kat/seal_2020"));
    if dir.join("values.json").exists() {
        Some(dir)
    } else {
        eprintln!(
            "SKIPPED: no SEAL KAT files in {} (run tools/seal-kat/run_kat.sh)",
            dir.display()
        );
        None
    }
}

/// Bound (log2 of the max absolute slot error against the exact values) per object.
///
/// * `-25` (the brief's bound): symmetric encryption (noise `e`, measured ~2^-29), `add`,
///   the tensor product and relinearization at scale `Delta^2` (key-switching noise is
///   ~2^-57 relative there).
/// * `-23`: objects whose noise is a rounding `r_0 + r_1 s` (`|r_i| <= 1/2`): SEAL's
///   public-key encryption (divide by the special prime) and `rescale_to_next`. With SEAL's
///   uniform ternary `s` (variance 2/3) the max over 2^14 slots has median 2^-24.76 (400
///   simulated draws: 2^-25.13 .. 2^-24.09; measured here 2^-25.0 and 2^-24.8), so 2^-25 is
///   below the median and fails SEAL itself; 2^-23 is 1.7 bits above the median. Same
///   effect as hekit's own public-key bound (`encrypt.rs`, 2^-24 for its variance-1/2 key).
/// * `-15`: SEAL's `rotate_vector`. SEAL 4.1 key switching decomposes by prime with the
///   representative in `[0, q_j)` (not centered), so digit 0 (the 60-bit `q_0`, as large as
///   the special prime `P`) contributes `(q_0 / 2P) S(X) e_0`, `S = 1 + X + ... + X^{N-1}`,
///   `e_0` the Galois-key error: at the slot of `zeta^{+-1}` (slot 0), `|S| ~ 2N/pi`, so the
///   error is `~ 0.5 * 20861 * |e_0(zeta)| / Delta` with `|e_0(zeta)|` Rayleigh of scale
///   `3.2 sqrt(N/2)`: typical 2^-17.5 (measured 2^-17.35), 1e-6 quantile 2^-15.6. This is
///   SEAL's own error; hekit's decryption is checked bit-exact against SEAL's regardless.
fn bound_log2(object: &str) -> f64 {
    match object {
        "ct_x_pk" | "ct_x_times_y_rescaled" => -23.0,
        "ct_rot1" => -15.0,
        _ => -25.0,
    }
}

/// hekit's decoding vs SEAL's decoding of the same (bit-identical) plaintext: both are
/// floating-point FFTs of 2^15 values of size up to ~2^60 (scale 2^80 after the tensor
/// product: values / scale), so they agree to ~1e-15 relative to the scale-normalized
/// coefficients; 2^-40 absolute leaves ample room.
const DECODE_AGREEMENT_LOG2: f64 = -40.0;

fn max_err(a: &[f64], b: &[f64]) -> (f64, usize) {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .enumerate()
        .fold(
            (0.0, 0),
            |(m, i), (j, e)| if e > m { (e, j) } else { (m, i) },
        )
}

fn assert_same(a: &Ciphertext, b: &Ciphertext, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: size");
    assert_eq!(a.level(), b.level(), "{what}: level");
    assert_eq!(a.scale().to_bits(), b.scale().to_bits(), "{what}: scale");
    for (x, y) in a.polys().iter().zip(b.polys()) {
        assert!(x.limbs() == y.limbs(), "{what}: residues differ");
    }
}

struct Kat {
    dir: PathBuf,
    params: Arc<CkksParams>,
    v: Json,
}

impl Kat {
    fn load() -> Option<Kat> {
        let dir = kat_dir()?;
        let text = std::fs::read_to_string(dir.join("values.json")).unwrap();
        let v = Json::parse(&text);
        let params = Arc::new(CkksParams::seal_2020());
        // The parameter sets are the same: N, the primes in SEAL's order, parms_ids.
        assert_eq!(v.get("seal_version").str(), "4.1.2");
        assert_eq!(v.get("poly_modulus_degree").num() as usize, params.n());
        assert_eq!(v.get("scale").num(), params.scale());
        let all: Vec<u64> = params.q().iter().chain(params.p()).copied().collect();
        assert_eq!(v.get("coeff_modulus").u64s(), all, "SEAL's coeff_modulus");
        assert_eq!(v.get("key_parms_id").u64s(), key_parms_id(&params));
        let ids = v.get("data_parms_ids").arr();
        assert_eq!(ids.len(), params.max_level() + 1);
        for (l, id) in ids.iter().enumerate() {
            assert_eq!(id.u64s(), data_parms_id(&params, l), "parms_id level {l}");
        }
        Some(Kat { dir, params, v })
    }

    fn ct(&self, name: &str) -> Ciphertext {
        load_ciphertext(self.dir.join(format!("{name}.bin")), &self.params).unwrap()
    }

    fn values(&self, name: &str) -> Vec<f64> {
        self.v.get(name).f64s()
    }
}

#[test]
#[ignore = "needs kat/seal_2020 from tools/seal-kat/run_kat.sh"]
fn seal_to_hekit() {
    let Some(k) = Kat::load() else { return };
    let p = &k.params;
    let sk = load_secret_key(k.dir.join("secret_key.bin"), p).unwrap();
    let dec = Decryptor::new(Arc::clone(p), sk);
    let enc = Encoder::new(Arc::clone(p));

    let objects = k.v.get("objects").obj();
    assert_eq!(objects.len(), 8);
    for (name, o) in objects {
        let file = k.dir.join(o.get("file").str());
        let ct = load_ciphertext(&file, p).unwrap();
        assert_eq!(ct.level(), o.get("level").num() as usize, "{name}");
        assert_eq!(ct.len(), o.get("size").num() as usize, "{name}");
        assert_eq!(ct.scale(), o.get("scale").num(), "{name}");
        // Format: re-serializing reproduces SEAL's bytes exactly.
        assert!(
            ciphertext_to_bytes(&ct) == std::fs::read(&file).unwrap(),
            "{name}: re-serialized bytes differ from SEAL's"
        );
        // Decryption: bit-identical to SEAL's decryption with the same key.
        let pt = dec.decrypt(&ct);
        let seal_pt = load_plaintext(k.dir.join(o.get("plain").str()), p).unwrap();
        assert_eq!(pt.level(), seal_pt.level(), "{name}");
        assert_eq!(pt.scale().to_bits(), seal_pt.scale().to_bits(), "{name}");
        assert!(
            pt.poly().limbs() == seal_pt.poly().limbs(),
            "{name}: hekit's decryption differs from SEAL's"
        );
        // Decoding vs SEAL's decoding and vs the exact values.
        let got = enc.decode(&pt);
        let (agree, _) = max_err(&got, &o.get("seal_decoded").f64s());
        let want = k.values(o.get("expected").str());
        let (err, at) = max_err(&got, &want);
        let seal_err = o.get("seal_max_err").num();
        let bound = bound_log2(name);
        eprintln!(
            "SEAL->hekit {name:<22} ({:<38}) level {} size {}: max error 2^{:.2} at slot {at} \
             (SEAL's own decode 2^{:.2}; hekit vs SEAL decode 2^{:.1}); bound 2^{bound}",
            o.get("op").str(),
            ct.level(),
            ct.len(),
            err.log2(),
            seal_err.log2(),
            agree.log2(),
        );
        assert!(
            agree < DECODE_AGREEMENT_LOG2.exp2(),
            "{name}: decode differs from SEAL's by {agree:e}"
        );
        assert!(
            (err - seal_err).abs() < DECODE_AGREEMENT_LOG2.exp2(),
            "{name}"
        );
        assert!(err < bound.exp2(), "{name}: max error {err:e} >= 2^{bound}");
    }

    // hekit's evaluator on SEAL ciphertexts is bit-identical to SEAL's evaluator.
    let ev = Evaluator::new(Arc::clone(p), None, None);
    let (x, y) = (k.ct("ct_x"), k.ct("ct_y"));
    assert_same(&ev.add(&x, &y).unwrap(), &k.ct("ct_x_plus_y"), "add");
    assert_same(
        &ev.mul_no_relin(&x, &y).unwrap(),
        &k.ct("ct_x_times_y_3"),
        "multiply (no relinearization)",
    );
    assert_same(
        &ev.rescale(&k.ct("ct_x_times_y")).unwrap(),
        &k.ct("ct_x_times_y_rescaled"),
        "rescale_to_next",
    );
    eprintln!("SEAL->hekit add / multiply / rescale_to_next: bit-identical to SEAL");
}

#[test]
#[ignore = "needs kat/seal_2020 from tools/seal-kat/run_kat.sh"]
fn hekit_to_seal() {
    let Some(k) = Kat::load() else { return };
    let p = &k.params;
    let sk = load_secret_key(k.dir.join("secret_key.bin"), p).unwrap();
    let pk = load_public_key(k.dir.join("public_key.bin"), p).unwrap();
    let enc = Encoder::new(Arc::clone(p));
    let dec = Decryptor::new(Arc::clone(p), sk.clone());
    let ev = Evaluator::new(Arc::clone(p), None, None);
    let x = k.values("x");
    let mut rng = Rng::from_seed([23; 32]);
    let sym = Encryptor::with_secret_key(Arc::clone(p), sk);
    let asym = Encryptor::new(Arc::clone(p), pk);
    let pt = |level| enc.encode(&x, p.scale(), level).unwrap();
    let (cx, cy) = (k.ct("ct_x"), k.ct("ct_y"));

    let outputs: Vec<(&str, Ciphertext, &str, f64)> = vec![
        ("hekit_ct_x_sym", sym.encrypt(&pt(9), &mut rng), "x", -25.0),
        (
            "hekit_ct_x_sym_l3",
            sym.encrypt(&pt(3), &mut rng),
            "x",
            -25.0,
        ),
        // Rounding noise with SEAL's dense key, see `bound_log2`.
        ("hekit_ct_x_pk", asym.encrypt(&pt(9), &mut rng), "x", -23.0),
        (
            "hekit_ct_x_plus_y",
            ev.add(&cx, &cy).unwrap(),
            "x_plus_y",
            -25.0,
        ),
        (
            "hekit_ct_x_times_y_3",
            ev.mul_no_relin(&cx, &cy).unwrap(),
            "x_times_y",
            -25.0,
        ),
        (
            "hekit_ct_x_times_y_rescaled",
            ev.rescale(&k.ct("ct_x_times_y")).unwrap(),
            "x_times_y",
            -23.0,
        ),
    ];
    for (name, ct, expected, bound) in &outputs {
        let path = k.dir.join(format!("{name}.bin"));
        save_ciphertext(ct, &path).unwrap();
        // hekit's own view of what it wrote (SEAL's view: `seal_interop_kat verify`).
        let back = load_ciphertext(&path, p).unwrap();
        assert_same(&back, ct, name);
        let (err, _) = max_err(&enc.decode(&dec.decrypt(&back)), &k.values(expected));
        eprintln!(
            "hekit->SEAL {name:<28} level {} size {}: hekit decrypts to max error 2^{:.2} \
             (bound 2^{bound}); wrote {}",
            ct.level(),
            ct.len(),
            err.log2(),
            path.display()
        );
        assert!(err < bound.exp2(), "{name}: {err:e}");
    }
}

// ----- minimal JSON reader for values.json (numbers, strings, arrays, objects) -----

enum Json {
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    fn parse(s: &str) -> Json {
        let b = s.as_bytes();
        let mut i = 0;
        let v = Self::value(b, &mut i);
        Self::ws(b, &mut i);
        assert_eq!(i, b.len(), "trailing data in values.json");
        v
    }
    fn ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && b[*i].is_ascii_whitespace() {
            *i += 1;
        }
    }
    fn value(b: &[u8], i: &mut usize) -> Json {
        Self::ws(b, i);
        match b[*i] {
            b'{' => {
                *i += 1;
                let mut m = BTreeMap::new();
                loop {
                    Self::ws(b, i);
                    if b[*i] == b'}' {
                        *i += 1;
                        return Json::Obj(m);
                    }
                    let Json::Str(key) = Self::value(b, i) else {
                        panic!("object key")
                    };
                    Self::ws(b, i);
                    assert_eq!(b[*i], b':');
                    *i += 1;
                    m.insert(key, Self::value(b, i));
                    Self::ws(b, i);
                    if b[*i] == b',' {
                        *i += 1;
                    }
                }
            }
            b'[' => {
                *i += 1;
                let mut v = Vec::new();
                loop {
                    Self::ws(b, i);
                    if b[*i] == b']' {
                        *i += 1;
                        return Json::Arr(v);
                    }
                    v.push(Self::value(b, i));
                    Self::ws(b, i);
                    if b[*i] == b',' {
                        *i += 1;
                    }
                }
            }
            b'"' => {
                let start = *i + 1;
                *i = start;
                while b[*i] != b'"' {
                    assert_ne!(b[*i], b'\\', "escapes are not used in values.json");
                    *i += 1;
                }
                *i += 1;
                Json::Str(String::from_utf8(b[start..*i - 1].to_vec()).unwrap())
            }
            _ => {
                let start = *i;
                while *i < b.len()
                    && !matches!(b[*i], b',' | b']' | b'}')
                    && !b[*i].is_ascii_whitespace()
                {
                    *i += 1;
                }
                Json::Num(String::from_utf8(b[start..*i].to_vec()).unwrap())
            }
        }
    }
    fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(m) => m.get(key).unwrap_or_else(|| panic!("missing key {key}")),
            _ => panic!("not an object"),
        }
    }
    fn obj(&self) -> &BTreeMap<String, Json> {
        match self {
            Json::Obj(m) => m,
            _ => panic!("not an object"),
        }
    }
    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => panic!("not an array"),
        }
    }
    fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("not a string"),
        }
    }
    fn num(&self) -> f64 {
        match self {
            Json::Num(s) => s.parse().unwrap(),
            _ => panic!("not a number"),
        }
    }
    fn f64s(&self) -> Vec<f64> {
        self.arr().iter().map(Json::num).collect()
    }
    fn u64s(&self) -> Vec<u64> {
        self.arr()
            .iter()
            .map(|j| match j {
                Json::Num(s) => s.parse().unwrap(),
                _ => panic!("not an integer"),
            })
            .collect()
    }
}
