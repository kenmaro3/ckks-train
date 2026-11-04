//! Reading and writing Microsoft SEAL 4.1 serialized CKKS objects (`compr_mode_type::none`).
//!
//! Interop requires the parameters to be SEAL's: the same ring degree and the same primes in
//! the same order ([`CkksParams::seal_2020`] is `CoeffModulus::Create(32768, {60, 40 x 9,
//! 60})`), with SEAL's last prime as hekit's single special prime. hekit and SEAL 4.1 then
//! agree on everything else: NTT form with the same root and bit-reversed order (so NTT-form
//! residues are copied as they are), the slot order of the encoder, and Galois element `3^k`
//! for a rotation by `k`. The known-answer test `tests/seal_kat.rs` (driven by
//! `tools/seal-kat/run_kat.sh`) checks this against a real SEAL 4.1.2 build.
//!
//! # Byte layout (verified against SEAL 4.1.2 `serialization.h`, `dynarray.h`, `ciphertext.cpp`, `plaintext.cpp`)
//!
//! All integers are little-endian (SEAL writes host byte order; every platform SEAL supports
//! is little-endian), `f64` is IEEE-754 binary64.
//!
//! ```text
//! SEALHeader (16 bytes)        magic u16 = 0xA15E | header_size u8 = 16 | version_major u8 (4)
//!                              | version_minor u8 | compr_mode u8 (0 = none) | reserved u16
//!                              | size u64 = byte length of the whole object incl. this header
//! Ciphertext body              parms_id [u64; 4] | is_ntt_form u8 | size u64 (components)
//!                              | poly_modulus_degree u64 | coeff_modulus_size u64 | scale f64
//!                              | correction_factor u64 (SEAL >= 4.0; 1 for CKKS)
//!                              | DynArray<u64>
//! Plaintext body               parms_id [u64; 4] | coeff_count u64 | scale f64 | DynArray<u64>
//! DynArray<u64>                SEALHeader (size = 24 + 8 count) | count u64 | count x u64
//! SecretKey  = SEALHeader + Plaintext body at the key level (all primes, Q then P), NTT form
//! PublicKey  = SEALHeader + Ciphertext body at the key level, 2 components, NTT form
//! ```
//!
//! Data order: component-major, then prime, then coefficient (`data[(c * L + i) * N + j]` is
//! residue `j` mod prime `i` of component `c`, `L = coeff_modulus_size`), NTT form.
//!
//! `parms_id` is SEAL's hash of the parameters at a level: BLAKE2b with a 32-byte digest of
//! the little-endian `u64` words `[scheme = 2 (CKKS), N, q_0, ..., q_{L-1}, plain_modulus =
//! 0]` ([`parms_id`]). A ciphertext or plaintext at hekit level `l` carries the id of
//! `q_0..=q_l`; keys carry the id of all primes `q_0..q_L, p_0..p_{k-1}` (SEAL's
//! `key_parms_id`). Loading looks the id up among the levels of `params`, so moduli or `N`
//! that differ from SEAL's are rejected (`InvalidParams`) even though the file only stores
//! the hash.
//!
//! # Untrusted input
//!
//! Every length is checked against the bytes actually present (and file sizes against the
//! largest object the parameters allow) before anything is allocated; every residue must be
//! reduced modulo its prime; trailing bytes are rejected. Not supported (rejected with
//! `InvalidParams`): compressed objects (zlib / zstd), seeded ciphertexts (SEAL's
//! `Serializable<Ciphertext>` from symmetric encryption; save an expanded `Ciphertext`),
//! coefficient-form objects, SEAL 3.x headers. All failures, including I/O errors, are
//! `HekitError::InvalidParams` with a message.

use std::path::Path;

use hekit_core::error::{HekitError, Result};
use hekit_core::poly::RnsPoly;
use hekit_core::rns::RnsBasis;
use std::sync::Arc;

use crate::ciphertext::Ciphertext;
use crate::keys::{PublicKey, QpPoly, SecretKey};
use crate::params::CkksParams;
use crate::plaintext::Plaintext;

/// SEAL's serialization magic number.
pub const SEAL_MAGIC: u16 = 0xA15E;
/// Size of a `SEALHeader` in bytes.
pub const SEAL_HEADER_SIZE: usize = 16;
/// The SEAL major version this module reads and writes.
pub const SEAL_VERSION_MAJOR: u8 = 4;
/// The SEAL minor version written by [`save_ciphertext`].
pub const SEAL_VERSION_MINOR: u8 = 1;
/// SEAL's `SEAL_CIPHERTEXT_SIZE_MAX`.
pub const SEAL_CIPHERTEXT_SIZE_MAX: usize = 16;
/// SEAL's `scheme_type::ckks`.
const SCHEME_CKKS: u64 = 2;
/// Ciphertext body bytes before the `DynArray`: parms_id, is_ntt_form, size, N,
/// coeff_modulus_size, scale, correction_factor.
const CT_META: usize = 32 + 1 + 8 * 5;
/// `DynArray` overhead: its header and the element count.
const DYN_META: usize = SEAL_HEADER_SIZE + 8;

/// A SEAL `parms_id`: four `u64` words (32 bytes of BLAKE2b output).
pub type ParmsId = [u64; 4];

fn bad<T>(msg: impl Into<String>) -> Result<T> {
    Err(HekitError::InvalidParams(msg.into()))
}

/// SEAL's `parms_id` of CKKS parameters with ring degree `n` and coefficient moduli
/// `moduli` (in SEAL's order): BLAKE2b-256 of `[2, n, moduli..., 0]` as little-endian `u64`s.
pub fn parms_id(n: usize, moduli: &[u64]) -> ParmsId {
    let mut data = Vec::with_capacity(8 * (moduli.len() + 3));
    for w in [SCHEME_CKKS, n as u64]
        .into_iter()
        .chain(moduli.iter().copied())
        .chain([0])
    {
        data.extend_from_slice(&w.to_le_bytes());
    }
    let h = blake2b(32, &data);
    let mut id = [0u64; 4];
    for (w, c) in id.iter_mut().zip(h.chunks_exact(8)) {
        *w = u64::from_le_bytes(c.try_into().expect("8 bytes"));
    }
    id
}

/// The `parms_id` of data level `level` (primes `q_0..=q_level`) of `params`.
///
/// # Panics
/// If `level > params.max_level()`.
pub fn data_parms_id(params: &CkksParams, level: usize) -> ParmsId {
    parms_id(params.n(), &params.q()[..=level])
}

/// SEAL's `key_parms_id` of `params` (all primes, `Q` then `P`).
pub fn key_parms_id(params: &CkksParams) -> ParmsId {
    let all: Vec<u64> = params.q().iter().chain(params.p()).copied().collect();
    parms_id(params.n(), &all)
}

/// Loads a SEAL `SecretKey` saved with `compr_mode_type::none` for SEAL parameters equal to
/// `params` (see the module docs).
pub fn load_secret_key(path: impl AsRef<Path>, params: &CkksParams) -> Result<SecretKey> {
    secret_key_from_bytes(&read_file(path.as_ref(), params)?, params)
}

/// Loads a SEAL `PublicKey` saved with `compr_mode_type::none` for SEAL parameters equal to
/// `params`.
pub fn load_public_key(path: impl AsRef<Path>, params: &CkksParams) -> Result<PublicKey> {
    public_key_from_bytes(&read_file(path.as_ref(), params)?, params)
}

/// Loads a SEAL `Ciphertext` (NTT form, at a data level of `params`) saved with
/// `compr_mode_type::none`.
pub fn load_ciphertext(path: impl AsRef<Path>, params: &CkksParams) -> Result<Ciphertext> {
    ciphertext_from_bytes(&read_file(path.as_ref(), params)?, params)
}

/// Loads a SEAL NTT-form `Plaintext` at a data level of `params` (e.g. the output of SEAL's
/// `Decryptor::decrypt`) saved with `compr_mode_type::none`.
pub fn load_plaintext(path: impl AsRef<Path>, params: &CkksParams) -> Result<Plaintext> {
    plaintext_from_bytes(&read_file(path.as_ref(), params)?, params)
}

/// Saves `ct` as a SEAL 4.1 `Ciphertext` (`compr_mode_type::none`) that SEAL's
/// `Ciphertext::load` accepts for the SEAL parameters equal to the ciphertext's.
pub fn save_ciphertext(ct: &Ciphertext, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, ciphertext_to_bytes(ct))
        .or_else(|e| bad(format!("{}: {e}", path.display())))
}

/// [`load_secret_key`] from bytes.
pub fn secret_key_from_bytes(bytes: &[u8], params: &CkksParams) -> Result<SecretKey> {
    let mut r = object(bytes)?;
    let (pid, coeff_count, _scale) = plaintext_meta(&mut r)?;
    if pid != key_parms_id(params) {
        return bad("secret key parms_id is not the key level of these parameters");
    }
    let limbs = params.q().len() + params.p().len();
    if coeff_count != (limbs * params.n()) as u64 {
        return bad("secret key coeff_count does not match N x (number of primes)");
    }
    let data = dyn_array(&mut r, limbs * params.n())?;
    r.finish()?;
    Ok(SecretKey::from_qp(key_poly(params, &data)?))
}

/// [`load_public_key`] from bytes.
pub fn public_key_from_bytes(bytes: &[u8], params: &CkksParams) -> Result<PublicKey> {
    let mut r = object(bytes)?;
    let meta = ciphertext_meta(&mut r, params)?;
    if meta.level.is_some() {
        return bad("public key parms_id is not the key level of these parameters");
    }
    if meta.size != 2 {
        return bad("a public key has 2 components");
    }
    let limbs = params.q().len() + params.p().len();
    let per = limbs * params.n();
    let data = dyn_array(&mut r, 2 * per)?;
    r.finish()?;
    let b = key_poly(params, &data[..per])?;
    let a = key_poly(params, &data[per..])?;
    Ok(PublicKey::from_parts(b, a))
}

/// [`load_ciphertext`] from bytes.
pub fn ciphertext_from_bytes(bytes: &[u8], params: &CkksParams) -> Result<Ciphertext> {
    let mut r = object(bytes)?;
    let meta = ciphertext_meta(&mut r, params)?;
    let Some(level) = meta.level else {
        return bad("ciphertext parms_id is the key level, not a data level");
    };
    let per = (level + 1) * params.n();
    let data = dyn_array(&mut r, meta.size * per)?;
    r.finish()?;
    let c = data
        .chunks_exact(per)
        .map(|d| data_poly(params.basis_q(), level, d))
        .collect::<Result<Vec<_>>>()?;
    Ciphertext::new(c, meta.scale)
}

/// [`load_plaintext`] from bytes.
pub fn plaintext_from_bytes(bytes: &[u8], params: &CkksParams) -> Result<Plaintext> {
    let mut r = object(bytes)?;
    let (pid, coeff_count, scale) = plaintext_meta(&mut r)?;
    if pid == [0; 4] {
        return bad("coefficient-form plaintexts are not supported (CKKS uses NTT form)");
    }
    let Some(level) = data_level(params, &pid) else {
        return bad("plaintext parms_id is not a data level of these parameters");
    };
    let per = (level + 1) * params.n();
    if coeff_count != per as u64 {
        return bad("plaintext coeff_count does not match its level");
    }
    let data = dyn_array(&mut r, per)?;
    r.finish()?;
    Plaintext::new(data_poly(params.basis_q(), level, &data)?, scale)
}

/// Serializes `ct` as a SEAL 4.1 `Ciphertext` with `compr_mode_type::none` (byte-identical
/// to SEAL's own `save` of the same ciphertext).
pub fn ciphertext_to_bytes(ct: &Ciphertext) -> Vec<u8> {
    let basis = ct.polys()[0].basis();
    let n = basis.n();
    let moduli: Vec<u64> = basis.moduli()[..=ct.level()]
        .iter()
        .map(|m| m.value())
        .collect();
    let count = ct.len() * moduli.len() * n;
    let total = SEAL_HEADER_SIZE + CT_META + DYN_META + 8 * count;
    let mut out = Vec::with_capacity(total);
    header(&mut out, total as u64);
    for w in parms_id(n, &moduli) {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.push(1); // is_ntt_form
    for w in [ct.len(), n, moduli.len()] {
        out.extend_from_slice(&(w as u64).to_le_bytes());
    }
    out.extend_from_slice(&ct.scale().to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes()); // correction_factor
    header(&mut out, (DYN_META + 8 * count) as u64);
    out.extend_from_slice(&(count as u64).to_le_bytes());
    for p in ct.polys() {
        for limb in p.limbs() {
            for &x in limb {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    debug_assert_eq!(out.len(), total);
    out
}

fn header(out: &mut Vec<u8>, size: u64) {
    out.extend_from_slice(&SEAL_MAGIC.to_le_bytes());
    out.push(SEAL_HEADER_SIZE as u8);
    out.push(SEAL_VERSION_MAJOR);
    out.push(SEAL_VERSION_MINOR);
    out.push(0); // compr_mode_type::none
    out.extend_from_slice(&0u16.to_le_bytes()); // reserved
    out.extend_from_slice(&size.to_le_bytes());
}

/// Reads a file after checking that its size is plausible for `params`: at most the
/// largest object (a 16-component ciphertext at the key level) plus metadata.
fn read_file(path: &Path, params: &CkksParams) -> Result<Vec<u8>> {
    let limbs = params.q().len() + params.p().len();
    let max = (SEAL_HEADER_SIZE + CT_META + DYN_META) as u64
        + 8 * (SEAL_CIPHERTEXT_SIZE_MAX * limbs * params.n()) as u64;
    let io = |e: std::io::Error| HekitError::InvalidParams(format!("{}: {e}", path.display()));
    let len = std::fs::metadata(path).map_err(io)?.len();
    if len > max {
        return bad(format!(
            "{}: {len} bytes, more than any SEAL object for these parameters ({max})",
            path.display()
        ));
    }
    std::fs::read(path).map_err(io)
}

/// Bounds-checked little-endian reader.
struct Reader<'a> {
    buf: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.buf.len() {
            return bad("SEAL object truncated");
        }
        let (head, rest) = self.buf.split_at(n);
        self.buf = rest;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("2")))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }
    fn parms_id(&mut self) -> Result<ParmsId> {
        Ok([self.u64()?, self.u64()?, self.u64()?, self.u64()?])
    }
    fn finish(&self) -> Result<()> {
        if self.buf.is_empty() {
            Ok(())
        } else {
            bad(format!(
                "{} trailing bytes after the SEAL object",
                self.buf.len()
            ))
        }
    }
}

/// Parses a `SEALHeader` and returns `size`, the object's total length.
fn seal_header(r: &mut Reader<'_>) -> Result<u64> {
    if r.u16()? != SEAL_MAGIC {
        return bad("not a SEAL object (bad magic)");
    }
    if r.u8()? as usize != SEAL_HEADER_SIZE {
        return bad("unsupported SEALHeader size");
    }
    let (major, _minor) = (r.u8()?, r.u8()?);
    if major != SEAL_VERSION_MAJOR {
        return bad(format!(
            "SEAL {major}.x serialization is not supported (need 4.x)"
        ));
    }
    match r.u8()? {
        0 => {}
        1 | 2 => {
            return bad(
                "compressed SEAL objects are not supported (save with compr_mode_type::none)",
            );
        }
        m => return bad(format!("unknown SEAL compression mode {m}")),
    }
    let _reserved = r.u16()?;
    r.u64()
}

/// Checks the outer header (its size must be the whole buffer) and returns a reader over
/// the object body.
fn object(bytes: &[u8]) -> Result<Reader<'_>> {
    let mut r = Reader { buf: bytes };
    let size = seal_header(&mut r)?;
    if size != bytes.len() as u64 {
        return bad(format!(
            "SEALHeader size {size} != object length {}",
            bytes.len()
        ));
    }
    Ok(r)
}

/// Parses a `DynArray<u64>` that must hold exactly `expected` elements.
fn dyn_array(r: &mut Reader<'_>, expected: usize) -> Result<Vec<u64>> {
    let size = seal_header(r)?;
    let count = r.u64()?;
    if count != expected as u64 {
        return bad(format!(
            "SEAL data has {count} words, expected {expected} (seeded ciphertexts are not supported)"
        ));
    }
    let bytes = expected
        .checked_mul(8)
        .ok_or_else(|| HekitError::InvalidParams("SEAL data size overflows".into()))?;
    if size != (DYN_META + bytes) as u64 {
        return bad("SEAL DynArray header size does not match its count");
    }
    let raw = r.take(bytes)?; // checked against the buffer before allocating
    Ok(raw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8")))
        .collect())
}

fn data_level(params: &CkksParams, pid: &ParmsId) -> Option<usize> {
    (0..=params.max_level()).find(|&l| data_parms_id(params, l) == *pid)
}

struct CtMeta {
    /// `None` for the key level.
    level: Option<usize>,
    size: usize,
    scale: f64,
}

fn ciphertext_meta(r: &mut Reader<'_>, params: &CkksParams) -> Result<CtMeta> {
    let pid = r.parms_id()?;
    let is_ntt = r.u8()?;
    let size = r.u64()?;
    let n = r.u64()?;
    let coeff_modulus_size = r.u64()?;
    let scale = r.f64()?;
    let correction_factor = r.u64()?;
    let (level, limbs) = if pid == key_parms_id(params) {
        (None, params.q().len() + params.p().len())
    } else if let Some(l) = data_level(params, &pid) {
        (Some(l), l + 1)
    } else {
        return bad("SEAL parms_id matches no level of these parameters (different N or moduli)");
    };
    if n != params.n() as u64 || coeff_modulus_size != limbs as u64 {
        return bad("SEAL ring degree / modulus count do not match the parms_id level");
    }
    if is_ntt != 1 {
        return bad("SEAL ciphertext is not in NTT form");
    }
    if !(2..=SEAL_CIPHERTEXT_SIZE_MAX as u64).contains(&size) {
        return bad(format!("SEAL ciphertext size {size} outside 2..=16"));
    }
    if correction_factor != 1 {
        return bad("CKKS ciphertexts have correction_factor 1");
    }
    if !(scale.is_finite() && scale > 0.0) {
        return bad("SEAL ciphertext scale must be positive and finite");
    }
    Ok(CtMeta {
        level,
        size: size as usize,
        scale,
    })
}

fn plaintext_meta(r: &mut Reader<'_>) -> Result<(ParmsId, u64, f64)> {
    Ok((r.parms_id()?, r.u64()?, r.f64()?))
}

/// `level + 1` limbs of `n` residues each, NTT form, over `basis` (residues checked reduced
/// by `RnsPoly::from_limbs`).
fn data_poly(basis: &Arc<RnsBasis>, level: usize, data: &[u64]) -> Result<RnsPoly> {
    let limbs = data.chunks_exact(basis.n()).map(<[u64]>::to_vec).collect();
    let p = RnsPoly::from_limbs(basis, limbs, true)?;
    debug_assert_eq!(p.level(), level);
    Ok(p)
}

/// A key-level polynomial (`Q` limbs then `P` limbs) as a [`QpPoly`].
fn key_poly(params: &CkksParams, data: &[u64]) -> Result<QpPoly> {
    let (dq, dp) = data.split_at(params.q().len() * params.n());
    let q = data_poly(params.basis_q(), params.max_level(), dq)?;
    let p = data_poly(params.basis_p(), params.p().len() - 1, dp)?;
    Ok(QpPoly::new(q, p))
}

// ----- BLAKE2b (RFC 7693), unkeyed, used only for `parms_id` -----

const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const BLAKE2B_SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

fn blake2b_compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for (w, c) in m.iter_mut().zip(block.chunks_exact(8)) {
        *w = u64::from_le_bytes(c.try_into().expect("8"));
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&BLAKE2B_IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    let g = |v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64| {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    };
    for round in 0..12 {
        let s = &BLAKE2B_SIGMA[round % 10];
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// Unkeyed BLAKE2b with an `out_len`-byte digest (`1..=64`).
fn blake2b(out_len: usize, data: &[u8]) -> Vec<u8> {
    assert!((1..=64).contains(&out_len), "BLAKE2b digest length");
    let mut h = BLAKE2B_IV;
    h[0] ^= 0x0101_0000 ^ out_len as u64;
    let blocks = data.len().div_ceil(128).max(1);
    for i in 0..blocks {
        let chunk = &data[i * 128..((i + 1) * 128).min(data.len())];
        let mut block = [0u8; 128];
        block[..chunk.len()].copy_from_slice(chunk);
        let last = i + 1 == blocks;
        let t = if last { data.len() } else { (i + 1) * 128 } as u128;
        blake2b_compress(&mut h, &block, t, last);
    }
    h.iter()
        .flat_map(|w| w.to_le_bytes())
        .take(out_len)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::keys::KeyGenerator;
    use hekit_core::sampler::Rng;

    /// Plaintext body bytes before the `DynArray`: parms_id, coeff_count, scale.
    const PT_META: usize = 32 + 8 + 8;

    fn hex(b: &[u8]) -> String {
        use std::fmt::Write;
        b.iter().fold(String::new(), |mut s, x| {
            write!(s, "{x:02x}").expect("writing to a String");
            s
        })
    }

    /// RFC 7693 Appendix A ("abc") and the BLAKE2 reference empty-input digest.
    #[test]
    fn blake2b_test_vectors() {
        assert_eq!(
            hex(&blake2b(64, b"abc")),
            "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
             7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
        );
        assert_eq!(
            hex(&blake2b(64, b"")),
            "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419\
             d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
        );
        // Multi-block input: two full blocks plus one byte must differ from two blocks.
        assert_ne!(blake2b(32, &[7; 257]), blake2b(32, &[7; 256]));
    }

    /// `parms_id`s printed by SEAL 4.1.2 for seal_2020 (`tools/seal-kat/interop`,
    /// `values.json`): the key level and data levels 0 and 9.
    #[test]
    fn seal_2020_parms_ids_match_seal() {
        let p = CkksParams::seal_2020();
        assert_eq!(
            key_parms_id(&p),
            [
                7191370043490053719,
                62418774019697152,
                17097305126088585189,
                18375298844209535223
            ]
        );
        assert_eq!(
            data_parms_id(&p, 0),
            [
                1256757360063172209,
                3701657226246155154,
                6506013794352499262,
                1216675072159714914
            ]
        );
        assert_eq!(
            data_parms_id(&p, 9),
            [
                16169145876469523575,
                11660861062887149260,
                12362353866312426611,
                17077531305736521100
            ]
        );
    }

    struct Ctx {
        params: Arc<CkksParams>,
        sk: SecretKey,
        ct: Ciphertext,
    }

    fn ctx() -> Ctx {
        let params = Arc::new(CkksParams::new(6, &[60, 40, 40], &[60], 40, 3).unwrap());
        let kg = KeyGenerator::new(Arc::clone(&params), Rng::from_seed([1; 32])).unwrap();
        let sk = kg.secret_key();
        let enc = Encoder::new(Arc::clone(&params));
        let pt = enc.encode(&[0.25, -0.5, 0.75], params.scale(), 2).unwrap();
        let ct = Encryptor::with_secret_key(Arc::clone(&params), sk.clone())
            .encrypt(&pt, &mut Rng::from_seed([2; 32]));
        Ctx { params, sk, ct }
    }

    #[test]
    fn ciphertext_roundtrip() {
        let c = ctx();
        let bytes = ciphertext_to_bytes(&c.ct);
        // header + metadata + DynArray(2 components x 3 primes x 64)
        assert_eq!(bytes.len(), 16 + 73 + 24 + 8 * 2 * 3 * 64);
        assert_eq!(&bytes[..8], &[0x5e, 0xa1, 16, 4, 1, 0, 0, 0]);
        let back = ciphertext_from_bytes(&bytes, &c.params).unwrap();
        assert_eq!(back.level(), c.ct.level());
        assert_eq!(back.scale(), c.ct.scale());
        for (a, b) in back.polys().iter().zip(c.ct.polys()) {
            assert_eq!(a.limbs(), b.limbs());
        }
        assert_eq!(ciphertext_to_bytes(&back), bytes);
        let dec = Decryptor::new(Arc::clone(&c.params), c.sk.clone());
        let v = Encoder::new(Arc::clone(&c.params)).decode(&dec.decrypt(&back));
        assert!((v[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_malformed_ciphertexts() {
        let c = ctx();
        let good = ciphertext_to_bytes(&c.ct);
        let p = &c.params;
        let check = |bytes: &[u8], what: &str| {
            let err = ciphertext_from_bytes(bytes, p).unwrap_err();
            assert!(
                matches!(err, HekitError::InvalidParams(_)),
                "{what}: {err:?}"
            );
        };
        let patched = |at: usize, b: &[u8]| {
            let mut v = good.clone();
            v[at..at + b.len()].copy_from_slice(b);
            v
        };
        check(&[], "empty");
        check(&good[..good.len() - 1], "truncated");
        let mut long = good.clone();
        long.push(0);
        check(&long, "trailing byte");
        check(&patched(0, &[0, 0]), "magic");
        check(&patched(2, &[17]), "header size");
        check(&patched(3, &[3]), "version 3");
        check(&patched(5, &[2]), "zstd");
        check(&patched(8, &u64::MAX.to_le_bytes()), "outer size");
        check(&patched(16, &[0xff]), "parms_id");
        check(&patched(48, &[0]), "coefficient form");
        check(&patched(49, &1u64.to_le_bytes()), "size 1");
        check(&patched(49, &17u64.to_le_bytes()), "size 17");
        check(&patched(49, &u64::MAX.to_le_bytes()), "size huge");
        check(&patched(57, &128u64.to_le_bytes()), "N");
        check(&patched(65, &2u64.to_le_bytes()), "coeff_modulus_size");
        check(&patched(73, &0f64.to_le_bytes()), "scale 0");
        check(&patched(73, &f64::NAN.to_le_bytes()), "scale NaN");
        check(&patched(81, &2u64.to_le_bytes()), "correction factor");
        check(&patched(89, &[0, 0]), "inner magic");
        check(&patched(97, &u64::MAX.to_le_bytes()), "inner size");
        check(&patched(105, &(3 * 64u64).to_le_bytes()), "seeded count");
        check(&patched(105, &u64::MAX.to_le_bytes()), "huge count");
        let q0 = p.q()[0];
        check(&patched(113, &q0.to_le_bytes()), "unreduced residue");
        // A parms_id of other moduli (the first prime replaced) is rejected.
        let other = parms_id(p.n(), &[p.q()[0] - 2 * p.n() as u64, p.q()[1], p.q()[2]]);
        let mut v = good.clone();
        for (i, w) in other.iter().enumerate() {
            v[16 + 8 * i..24 + 8 * i].copy_from_slice(&w.to_le_bytes());
        }
        check(&v, "other moduli");
        // The key level is not a ciphertext level.
        let mut v = good.clone();
        for (i, w) in key_parms_id(p).iter().enumerate() {
            v[16 + 8 * i..24 + 8 * i].copy_from_slice(&w.to_le_bytes());
        }
        check(&v, "key level");
        // A different parameter set (other N) rejects the same bytes.
        let p2 = CkksParams::new(7, &[60, 40, 40], &[60], 40, 3).unwrap();
        assert!(ciphertext_from_bytes(&good, &p2).is_err());
    }

    #[test]
    fn file_io_errors_and_size_cap() {
        let c = ctx();
        let dir = std::env::temp_dir().join(format!("hekit-seal-compat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ct.bin");
        save_ciphertext(&c.ct, &path).unwrap();
        let back = load_ciphertext(&path, &c.params).unwrap();
        assert_eq!(back.polys()[1].limbs(), c.ct.polys()[1].limbs());
        assert!(load_ciphertext(dir.join("missing.bin"), &c.params).is_err());
        assert!(load_secret_key(&path, &c.params).is_err());
        let big = dir.join("big.bin");
        let f = std::fs::File::create(&big).unwrap();
        f.set_len(1 << 30).unwrap(); // sparse; must be rejected before reading
        assert!(load_ciphertext(&big, &c.params).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A secret key in SEAL's layout (Q limbs then P limbs, key-level parms_id) loads back
    /// to the same `s`.
    #[test]
    fn secret_key_layout() {
        let c = ctx();
        let p = &c.params;
        let (q, pp) = (c.sk.qp().q(), c.sk.qp().p());
        let words: Vec<u64> = q
            .limbs()
            .iter()
            .chain(pp.limbs())
            .flatten()
            .copied()
            .collect();
        let mut out = Vec::new();
        let total = SEAL_HEADER_SIZE + PT_META + DYN_META + 8 * words.len();
        header(&mut out, total as u64);
        for w in key_parms_id(p) {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&(words.len() as u64).to_le_bytes());
        out.extend_from_slice(&1f64.to_le_bytes());
        header(&mut out, (DYN_META + 8 * words.len()) as u64);
        out.extend_from_slice(&(words.len() as u64).to_le_bytes());
        for w in &words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        let sk = secret_key_from_bytes(&out, p).unwrap();
        assert_eq!(sk.qp().q().limbs(), q.limbs());
        assert_eq!(sk.qp().p().limbs(), pp.limbs());
        // As a plaintext it is at the key level, not a data level.
        assert!(plaintext_from_bytes(&out, p).is_err());
        // Wrong coeff_count.
        let mut v = out.clone();
        v[48..56].copy_from_slice(&((words.len() - 64) as u64).to_le_bytes());
        assert!(secret_key_from_bytes(&v, p).is_err());
    }
}
