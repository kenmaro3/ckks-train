//! Data sets for the training benchmarks: the MNIST IDX loader (Task 32), also used for
//! Fashion-MNIST (same format; `scripts/fetch_fashion_mnist.sh`), ReBoot's 14 x 14 resize and
//! min-max scaling, and the Breast Cancer (UCI WDBC) table with its split and standardization
//! (Task 51).
//!
//! MNIST is **not** in the repository (licence and size): `scripts/fetch_mnist.sh` downloads
//! the four IDX files into the gitignored `data/mnist/` and unpacks them. The IDX format
//! (LeCun et al.) is a big-endian header `00 00 <type> <ndim>`, `ndim` big-endian `u32`
//! dimensions, then the row-major data; MNIST uses type `0x08` (unsigned byte), images as
//! `n x 28 x 28` and labels as `n`.

use std::fs;
use std::io;
use std::path::Path;

use ckks_train_core::sampler::Rng;
use ndarray::Array2;

/// An IDX array of unsigned bytes: its dimensions and row-major data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdxU8 {
    /// The dimensions, outermost first.
    pub dims: Vec<usize>,
    /// `dims.iter().product()` bytes, row-major.
    pub data: Vec<u8>,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Parses an IDX file of type `0x08` (unsigned byte).
///
/// Errors (`InvalidData`): a short header, a nonzero magic prefix, another data type, zero
/// dimensions, or a payload whose length is not the product of the dimensions.
pub fn parse_idx_u8(bytes: &[u8]) -> io::Result<IdxU8> {
    if bytes.len() < 4 {
        return Err(invalid(format!(
            "IDX header too short ({} bytes)",
            bytes.len()
        )));
    }
    if bytes[0] != 0 || bytes[1] != 0 {
        return Err(invalid("IDX magic must start with two zero bytes".into()));
    }
    if bytes[2] != 0x08 {
        return Err(invalid(format!(
            "IDX data type 0x{:02x} is not unsigned byte (0x08)",
            bytes[2]
        )));
    }
    let ndim = bytes[3] as usize;
    if ndim == 0 {
        return Err(invalid("IDX file with zero dimensions".into()));
    }
    let header = 4 + 4 * ndim;
    if bytes.len() < header {
        return Err(invalid(format!(
            "IDX header of {ndim} dimensions needs {header} bytes, file has {}",
            bytes.len()
        )));
    }
    let dims: Vec<usize> = bytes[4..header]
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let len = dims
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| invalid(format!("IDX dimensions {dims:?} overflow")))?;
    if bytes.len() - header != len {
        return Err(invalid(format!(
            "IDX dimensions {dims:?} need {len} data bytes, file has {}",
            bytes.len() - header
        )));
    }
    Ok(IdxU8 {
        dims,
        data: bytes[header..].to_vec(),
    })
}

/// Serializes an unsigned-byte IDX array (the inverse of [`parse_idx_u8`]; synthetic data
/// sets for tests and smoke runs).
///
/// # Panics
/// If `dims` is empty or longer than 255, a dimension exceeds `u32::MAX`, or `data` does not
/// have `dims.iter().product()` bytes.
pub fn write_idx_u8(dims: &[usize], data: &[u8]) -> Vec<u8> {
    assert!(!dims.is_empty() && dims.len() <= 255, "1..=255 dimensions");
    assert_eq!(dims.iter().product::<usize>(), data.len(), "data length");
    let mut out = vec![0, 0, 0x08, dims.len() as u8];
    for &d in dims {
        out.extend(u32::try_from(d).expect("dimension fits u32").to_be_bytes());
    }
    out.extend_from_slice(data);
    out
}

/// Reads and parses an unsigned-byte IDX file.
///
/// Errors: the file's I/O errors and those of [`parse_idx_u8`].
pub fn read_idx_u8(path: &Path) -> io::Result<IdxU8> {
    let bytes =
        fs::read(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    parse_idx_u8(&bytes).map_err(|e| invalid(format!("{}: {e}", path.display())))
}

/// The MNIST split to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// 60 000 training images (`train-*-idx?-ubyte`).
    Train,
    /// 10 000 test images (`t10k-*-idx?-ubyte`).
    Test,
}

impl Split {
    /// The `(images, labels)` file names of this split (unpacked, as `scripts/fetch_mnist.sh`
    /// leaves them).
    pub fn files(self) -> (&'static str, &'static str) {
        match self {
            Split::Train => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            Split::Test => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        }
    }
}

/// Images and labels of an MNIST-format data set.
#[derive(Debug, Clone, PartialEq)]
pub struct Mnist {
    /// `n x (rows * cols)`, pixels scaled to `[0, 1]` (`byte / 255`), row-major per image.
    pub images: Array2<f64>,
    /// `n` labels.
    pub labels: Vec<u8>,
    /// Image height.
    pub rows: usize,
    /// Image width.
    pub cols: usize,
}

impl Mnist {
    /// The number of samples.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Pixels per image, `rows * cols`.
    pub fn features(&self) -> usize {
        self.rows * self.cols
    }
}

/// Builds an [`Mnist`] from parsed image (`n x rows x cols`) and label (`n`) arrays, keeping
/// the first `limit` samples (all if `None`).
///
/// Errors (`InvalidData`): images not 3-dimensional, labels not 1-dimensional, different
/// sample counts, an array whose data length is not the product of its dimensions (possible
/// for a hand-built [`IdxU8`]), or a label outside the MNIST classes `0..=9`.
pub fn mnist_from_idx(images: &IdxU8, labels: &IdxU8, limit: Option<usize>) -> io::Result<Mnist> {
    if images.dims.len() != 3 || labels.dims.len() != 1 {
        return Err(invalid(format!(
            "MNIST images must be n x rows x cols and labels n (got {:?} / {:?})",
            images.dims, labels.dims
        )));
    }
    let (n, rows, cols) = (images.dims[0], images.dims[1], images.dims[2]);
    if labels.dims[0] != n {
        return Err(invalid(format!("{n} images but {} labels", labels.dims[0])));
    }
    for (what, idx) in [("images", images), ("labels", labels)] {
        if idx.dims.iter().product::<usize>() != idx.data.len() {
            return Err(invalid(format!(
                "{what}: dimensions {:?} but {} data bytes",
                idx.dims,
                idx.data.len()
            )));
        }
    }
    if let Some(l) = labels.data.iter().find(|&&l| l >= 10) {
        return Err(invalid(format!("label {l} is not an MNIST class (0..=9)")));
    }
    let n = limit.map_or(n, |l| l.min(n));
    let f = rows * cols;
    let pixels = Array2::from_shape_fn((n, f), |(i, j)| images.data[i * f + j] as f64 / 255.0);
    Ok(Mnist {
        images: pixels,
        labels: labels.data[..n].to_vec(),
        rows,
        cols,
    })
}

/// Loads `split` from `dir` (the unpacked IDX files of [`Split::files`]), keeping the first
/// `limit` samples.
///
/// Errors: the I/O errors (a missing file: run `scripts/fetch_mnist.sh`) and those of
/// [`read_idx_u8`] and [`mnist_from_idx`].
pub fn load_mnist(dir: &Path, split: Split, limit: Option<usize>) -> io::Result<Mnist> {
    let (img, lab) = split.files();
    let images = fs::read(dir.join(img))?;
    let labels = fs::read(dir.join(lab))?;
    mnist_from_bytes(&images, &labels, limit)
}

/// Decodes the exact borrowed IDX image/label bytes with the same scaling and limits as
/// [`load_mnist`]. This pure decoder performs no pathname reads.
pub fn mnist_from_bytes(images: &[u8], labels: &[u8], limit: Option<usize>) -> io::Result<Mnist> {
    mnist_from_idx(&parse_idx_u8(images)?, &parse_idx_u8(labels)?, limit)
}

/// One-hot targets: row `i` has a 1 in column `labels[i]`.
///
/// # Panics
/// If a label is `>= classes`.
pub fn one_hot(labels: &[u8], classes: usize) -> Array2<f64> {
    let mut y = Array2::zeros((labels.len(), classes));
    for (i, &l) in labels.iter().enumerate() {
        assert!((l as usize) < classes, "label {l} >= {classes} classes");
        y[[i, l as usize]] = 1.0;
    }
    y
}

// ---------------------------------------------------------------------------------------------
// Minimal .npy reader (Task 41, R-M3-12: ReBoot's initial weights as plain data).
// ---------------------------------------------------------------------------------------------

/// The value of `'key': value` in a .npy header dict (up to the next `,` at depth 0 or `}`).
fn npy_header_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("'{key}':");
    let start = header.find(&pat)? + pat.len();
    let rest = &header[start..];
    let mut depth = 0i32;
    for (i, c) in rest.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' | '}' if depth == 0 => return Some(rest[..i].trim()),
            _ => {}
        }
    }
    None
}

/// Parses a 2-D `.npy` array (format versions 1.0, 2.0 and 3.0) of little-endian `f8` / `f4`, or
/// of 1- to 8-byte integers (labels), into an `f64` matrix of its shape. C order is read as is; Fortran order is transposed back
/// into row-major. A 1-D array is a `1 x n` matrix.
///
/// Errors (`InvalidData`): a bad magic or version, a header without `descr` / `fortran_order` /
/// `shape`, a dtype other than `<f8` / `<f4`, more than 2 dimensions, or a data length that
/// does not match the shape.
pub fn parse_npy_f64(bytes: &[u8]) -> io::Result<Array2<f64>> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(invalid("not a .npy file (magic)".into()));
    }
    let (hlen, hstart) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => return Err(invalid(format!(".npy version {v} not supported"))),
    };
    let header = bytes
        .get(hstart..hstart + hlen)
        .ok_or_else(|| invalid("short .npy header".into()))
        .and_then(|h| {
            std::str::from_utf8(h).map_err(|_| invalid(".npy header is not UTF-8".into()))
        })?;
    let descr = npy_header_field(header, "descr")
        .ok_or_else(|| invalid(".npy header without descr".into()))?
        .trim_matches(|c| c == '\'' || c == '"');
    // (byte width, decoder) per supported dtype; every value is returned as f64.
    let width = match descr {
        "<f8" | "<i8" | "<u8" => 8,
        "<f4" | "<i4" | "<u4" => 4,
        "<i2" | "<u2" => 2,
        "|u1" | "|i1" | "<u1" | "<i1" | "|b1" => 1,
        d => {
            return Err(invalid(format!(
                ".npy dtype {d} not supported (little-endian f8/f4, i/u 1-8 bytes)"
            )));
        }
    };
    let fortran = match npy_header_field(header, "fortran_order") {
        Some("False") => false,
        Some("True") => true,
        _ => return Err(invalid(".npy header without fortran_order".into())),
    };
    let shape_txt = npy_header_field(header, "shape")
        .ok_or_else(|| invalid(".npy header without shape".into()))?;
    let shape = shape_txt
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .map_err(|_| invalid(format!("bad shape {shape_txt}")))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let (rows, cols) = match shape[..] {
        [n] => (1, n),
        [r, c] => (r, c),
        _ => {
            return Err(invalid(format!(
                "{}-D .npy array (1-D or 2-D only)",
                shape.len()
            )));
        }
    };
    let data = &bytes[hstart + hlen..];
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| invalid(format!(".npy shape {shape:?} overflows")))?;
    if data.len() != expected {
        return Err(invalid(format!(
            ".npy data of {} bytes for shape {shape:?} and {descr}",
            data.len()
        )));
    }
    let kind = descr.as_bytes()[1];
    let values: Vec<f64> = data
        .chunks_exact(width)
        .map(|c| match (kind, width) {
            (b'f', 8) => f64::from_le_bytes(c.try_into().expect("8 bytes")),
            (b'f', _) => f32::from_le_bytes(c.try_into().expect("4 bytes")) as f64,
            (b'i', 8) => i64::from_le_bytes(c.try_into().expect("8 bytes")) as f64,
            (b'i', 4) => i32::from_le_bytes(c.try_into().expect("4 bytes")) as f64,
            (b'i', 2) => i16::from_le_bytes(c.try_into().expect("2 bytes")) as f64,
            (b'i', _) => c[0] as i8 as f64,
            (_, 8) => u64::from_le_bytes(c.try_into().expect("8 bytes")) as f64,
            (_, 4) => u32::from_le_bytes(c.try_into().expect("4 bytes")) as f64,
            (_, 2) => u16::from_le_bytes(c.try_into().expect("2 bytes")) as f64,
            _ => c[0] as f64,
        })
        .collect();
    let m = if fortran {
        Array2::from_shape_vec((cols, rows), values)
            .expect("checked length")
            .reversed_axes()
            .as_standard_layout()
            .to_owned()
    } else {
        Array2::from_shape_vec((rows, cols), values).expect("checked length")
    };
    Ok(m)
}

/// [`parse_npy_f64`] of a file.
///
/// Errors: the file's I/O errors and those of [`parse_npy_f64`].
pub fn read_npy_f64(path: &Path) -> io::Result<Array2<f64>> {
    parse_npy_f64(&fs::read(path)?)
}

// ---------------------------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), for checking data files against a manifest without a dependency.
// ---------------------------------------------------------------------------------------------

/// The SHA-256 digest of `data` as 64 lowercase hex characters.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut state = Sha256State::new();
    state.update(data);
    state.finish()
}

// D7 callers reserve their accounting payload before this fallible allocation.
// Keep legacy output semantics; both paths share finalization/compression below.
fn reserved_digest_output(
    bytes: usize,
) -> std::result::Result<String, std::collections::TryReserveError> {
    let mut output = String::new();
    output.try_reserve_exact(bytes)?;
    Ok(output)
}
pub(crate) fn try_sha256_hex(
    data: &[u8],
) -> std::result::Result<String, std::collections::TryReserveError> {
    let output = reserved_digest_output(64)?;
    let mut state = Sha256State::new();
    state.update(data);
    Ok(state.finish_with_output(output))
}

// Original compression algorithm, now retaining only one partial block.
pub(crate) struct Sha256State {
    h: [u32; 8],
    block: [u8; 64],
    used: usize,
    bytes: u64,
}
impl Sha256State {
    pub(crate) fn new() -> Self {
        let h = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        Self {
            h,
            block: [0; 64],
            used: 0,
            bytes: 0,
        }
    }
    pub(crate) fn update(&mut self, mut data: &[u8]) {
        self.bytes = self.bytes.wrapping_add(data.len() as u64);
        if self.used > 0 {
            let take = (64 - self.used).min(data.len());
            self.block[self.used..self.used + take].copy_from_slice(&data[..take]);
            self.used += take;
            data = &data[take..];
            if self.used < 64 {
                return;
            }
            let block = self.block;
            self.compress(&block);
            self.used = 0;
        }
        let mut chunks = data.chunks_exact(64);
        for chunk in &mut chunks {
            self.compress(chunk);
        }
        let tail = chunks.remainder();
        self.block[..tail.len()].copy_from_slice(tail);
        self.used = tail.len();
    }
    pub(crate) fn finish(self) -> String {
        self.finish_with_output(String::with_capacity(64))
    }
    fn finish_with_output(self, mut output: String) -> String {
        debug_assert!(output.is_empty() && output.capacity() >= 64);
        for value in self.finish_fixed() {
            output.push(char::from(value));
        }
        output
    }
    /// Finish into fixed storage for the pre-context approval scanner.
    pub(crate) fn finish_fixed(mut self) -> [u8; 64] {
        self.block[self.used] = 0x80;
        self.block[self.used + 1..].fill(0);
        if self.used >= 56 {
            let block = self.block;
            self.compress(&block);
            self.block = [0; 64];
        }
        self.block[56..].copy_from_slice(&self.bytes.wrapping_mul(8).to_be_bytes());
        let block = self.block;
        self.compress(&block);
        let mut out = [0; 64];
        for (word, chunk) in self.h.iter().zip(out.chunks_exact_mut(8)) {
            for (index, byte) in chunk.iter_mut().enumerate() {
                *byte = b"0123456789abcdef"[((word >> (28 - 4 * index)) & 15) as usize];
            }
        }
        out
    }
    fn compress(&mut self, block: &[u8]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for (t, c) in block.chunks_exact(4).enumerate() {
            w[t] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let mut v = self.h;
        for t in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (hi, vi) in self.h.iter_mut().zip(v) {
            *hi = hi.wrapping_add(vi);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// A data set given as .npy arrays (Task 41b, R-M3-13: ReBoot's preprocessed 14 x 14 MNIST).
// ---------------------------------------------------------------------------------------------

/// The file names of a [`load_npy_dataset`] directory.
pub const NPY_DATASET_FILES: [&str; 4] = [
    "train_images_14x14.npy",
    "train_labels.npy",
    "test_images_14x14.npy",
    "test_labels.npy",
];

/// The 64-hex SHA-256 listed for `file` in a manifest's text: the first run of 64 hex
/// characters after the first occurrence of `file` (so any JSON or text layout that names the
/// file before its digest works).
pub fn manifest_sha256(manifest: &str, file: &str) -> Option<String> {
    let at = manifest.find(file)? + file.len();
    let tail = manifest[at..].as_bytes();
    let mut run = 0usize;
    for (i, &c) in tail.iter().enumerate() {
        if c.is_ascii_hexdigit() {
            run += 1;
            if run == 64 && !tail.get(i + 1).is_some_and(|n| n.is_ascii_hexdigit()) {
                return Some(String::from_utf8_lossy(&tail[i + 1 - 64..=i]).to_ascii_lowercase());
            }
        } else {
            run = 0;
        }
    }
    None
}

/// Labels from a `.npy` array (any supported dtype): every value must be an integer `< 256`.
fn npy_labels(m: &Array2<f64>, what: &str) -> io::Result<Vec<u8>> {
    m.iter()
        .map(|&v| {
            if v.fract() == 0.0 && (0.0..256.0).contains(&v) {
                Ok(v as u8)
            } else {
                Err(invalid(format!(
                    "{what}: label {v} is not an integer in 0..256"
                )))
            }
        })
        .collect()
}

/// A training and a test set from `dir`: [`NPY_DATASET_FILES`] (images `n x features`, used as
/// given: no rescaling; labels `n`), each checked against its SHA-256 in
/// `dir/data_manifest.json` ([`manifest_sha256`]). The image side is `sqrt(features)` when that
/// is an integer (14 for 196 features), else `1 x features`.
///
/// Errors: `NotFound` / I/O errors for a missing file; `InvalidData` for a missing digest, a
/// digest mismatch, a malformed array, image and label counts that differ, or train and test
/// widths that differ.
pub fn load_npy_dataset(dir: &Path) -> io::Result<(Mnist, Mnist)> {
    let manifest = fs::read(dir.join("data_manifest.json"))?;
    let arrays = [
        fs::read(dir.join(NPY_DATASET_FILES[0]))?,
        fs::read(dir.join(NPY_DATASET_FILES[1]))?,
        fs::read(dir.join(NPY_DATASET_FILES[2]))?,
        fs::read(dir.join(NPY_DATASET_FILES[3]))?,
    ];
    npy_dataset_from_bytes(
        &manifest,
        arrays.each_ref().map(Vec::as_slice),
        NpyManifestPolicy::LegacyText,
    )
}

// The standalone loader keeps its documented legacy text format. Profile loading adds
// the independent structured file-specific defense and label-vector check, while sharing
// this one decoder and numerical assembly path.
pub(crate) enum NpyManifestPolicy {
    LegacyText,
    Structured,
}

pub(crate) fn npy_dataset_from_bytes(
    manifest: &[u8],
    sources: [&[u8]; 4],
    policy: NpyManifestPolicy,
) -> io::Result<(Mnist, Mnist)> {
    let text =
        std::str::from_utf8(manifest).map_err(|_| invalid("NPY manifest is not UTF-8".into()))?;
    let structured = match policy {
        NpyManifestPolicy::LegacyText => None,
        NpyManifestPolicy::Structured => Some(
            serde_json::from_slice::<serde_json::Value>(manifest)
                .map_err(|e| invalid(format!("invalid NPY manifest: {e}")))?,
        ),
    };
    let mut arrays = Vec::with_capacity(4);
    for (i, (file, bytes)) in NPY_DATASET_FILES.into_iter().zip(sources).enumerate() {
        let got = sha256_hex(bytes);
        if let Some(manifest) = &structured {
            let digest = manifest
                .get(file)
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    manifest
                        .get("files")?
                        .as_array()?
                        .iter()
                        .find(|v| v.get("file").and_then(serde_json::Value::as_str) == Some(file))?
                        .get("sha256_npy_file")?
                        .as_str()
                })
                .ok_or_else(|| {
                    invalid(format!("manifest missing file-specific sha256 for {file}"))
                })?;
            if digest.len() != 64
                || !digest.bytes().all(|c| c.is_ascii_hexdigit())
                || !got.eq_ignore_ascii_case(digest)
            {
                return Err(invalid(format!("{file}: manifest sha256 mismatch")));
            }
        }
        let want = manifest_sha256(text, file)
            .ok_or_else(|| invalid(format!("data_manifest.json lists no sha256 for {file}")))?;
        if got != want {
            return Err(invalid(format!("{file}: sha256 {got}, manifest {want}")));
        }
        let array = parse_npy_f64(bytes)?;
        if structured.is_some() && (i == 1 || i == 3) && array.nrows() != 1 && array.ncols() != 1 {
            return Err(invalid("NPY labels must be a vector".into()));
        }
        arrays.push(array);
    }
    let set = |images: &Array2<f64>, labels: &Array2<f64>, what: &str| -> io::Result<Mnist> {
        let labels = npy_labels(labels, what)?;
        if labels.len() != images.nrows() {
            return Err(invalid(format!(
                "{what}: {} images and {} labels",
                images.nrows(),
                labels.len()
            )));
        }
        let f = images.ncols();
        let side = (f as f64).sqrt().round() as usize;
        let (rows, cols) = if side * side == f {
            (side, side)
        } else {
            (1, f)
        };
        Ok(Mnist {
            images: images.clone(),
            labels,
            rows,
            cols,
        })
    };
    let train = set(&arrays[0], &arrays[1], "train")?;
    let test = set(&arrays[2], &arrays[3], "test")?;
    if train.features() != test.features() {
        return Err(invalid("train and test images differ in width".into()));
    }
    Ok((train, test))
}

// ---------------------------------------------------------------------------------------------
// Task 51: ReBoot's 14 x 14 preprocessing of the MNIST family (MNIST, Fashion-MNIST).
// ---------------------------------------------------------------------------------------------

/// The bicubic convolution kernel with `a = -0.5` (Keys; the value PIL and torch's
/// antialiased resize use).
fn cubic_kernel(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// The `n_out x n_in` matrix of the antialiased bicubic resize along one axis: output `i` has
/// its center at `c = s (i + 1/2)` with `s = n_in / n_out`; for a downscale (`s >= 1`) the
/// kernel is stretched by `s` (support `2 s`), input `j` (center `j + 1/2`) gets
/// `cubic((j + 1/2 - c) / s)`, and each row is normalized to sum 1 over the inputs in
/// `[int(c - 2s + 1/2), int(c + 2s + 1/2))` clipped to the image. This is the separable filter
/// of PIL's `Image.resize(BICUBIC)` and of `torch.nn.functional.interpolate(mode="bicubic",
/// antialias=True)`, written from that description (no third-party code). For an upscale
/// (`s < 1`) the kernel is not stretched.
pub fn bicubic_aa_weights(n_in: usize, n_out: usize) -> Array2<f64> {
    assert!(n_in > 0 && n_out > 0, "nonempty axes");
    let s = n_in as f64 / n_out as f64;
    let (support, inv) = if s >= 1.0 {
        (2.0 * s, 1.0 / s)
    } else {
        (2.0, 1.0)
    };
    let mut m = Array2::zeros((n_out, n_in));
    for i in 0..n_out {
        let c = s * (i as f64 + 0.5);
        // `as usize` truncates like C's int(); the lower end is clamped at 0 first.
        let lo = (c - support + 0.5).max(0.0) as usize;
        let hi = ((c + support + 0.5) as usize).min(n_in);
        let w: Vec<f64> = (lo..hi)
            .map(|j| cubic_kernel((j as f64 - c + 0.5) * inv))
            .collect();
        let total: f64 = w.iter().sum();
        for (k, wk) in w.iter().enumerate() {
            m[[i, lo + k]] = wk / total;
        }
    }
    m
}

/// Resizes every row of `images` (an image of `rows x cols`, row-major) to
/// `out_rows x out_cols` with the separable [`bicubic_aa_weights`] filter, in f64 and without
/// clamping (so values overshoot the input range, as in a float-tensor resize).
pub fn resize_bicubic_aa(
    images: &Array2<f64>,
    rows: usize,
    cols: usize,
    out_rows: usize,
    out_cols: usize,
) -> Array2<f64> {
    assert_eq!(images.ncols(), rows * cols, "image width");
    let wr = bicubic_aa_weights(rows, out_rows);
    let wc = bicubic_aa_weights(cols, out_cols);
    let mut out = Array2::zeros((images.nrows(), out_rows * out_cols));
    for (src, mut dst) in images.rows().into_iter().zip(out.rows_mut()) {
        let img = src
            .to_shape((rows, cols))
            .expect("row-major image")
            .to_owned();
        let r = wr.dot(&img).dot(&wc.t());
        dst.iter_mut().zip(r.iter()).for_each(|(d, v)| *d = *v);
    }
    out
}

/// Min-max scaling with the training set's range: `x -> (x - min) / (max - min)` over all
/// values of `train`, applied to both sets (test values may leave `[0, 1]`). Returns
/// `(min, max)`.
///
/// # Panics
/// If `train` is empty or constant.
pub fn min_max_scale(train: &mut Array2<f64>, test: &mut Array2<f64>) -> (f64, f64) {
    let lo = train.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = train.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(hi > lo, "training values must span a range");
    let span = hi - lo;
    train.mapv_inplace(|v| (v - lo) / span);
    test.mapv_inplace(|v| (v - lo) / span);
    (lo, hi)
}

/// ReBoot's input pipeline for the MNIST family as recorded by gpu-runner (R-REB-3: "resize to
/// 14, then min-max scaling over the training set"): each image resized to `side x side` with
/// [`resize_bicubic_aa`], then [`min_max_scale`] with the training range. Our pixels are
/// `byte / 255`; the resize is linear and the scaling affine-invariant, so the result equals
/// that of resizing the raw bytes. It matches ReBoot's float32 14 x 14 MNIST arrays to < 1e-6
/// (test `resize_matches_reboot_14x14_mnist`, run on the gitignored R-REB-3 arrays).
pub fn reboot_preprocess(train: &Mnist, test: &Mnist, side: usize) -> (Mnist, Mnist) {
    let resize = |m: &Mnist| resize_bicubic_aa(&m.images, m.rows, m.cols, side, side);
    let (mut tr, mut te) = (resize(train), resize(test));
    min_max_scale(&mut tr, &mut te);
    let set = |images, m: &Mnist| Mnist {
        images,
        labels: m.labels.clone(),
        rows: side,
        cols: side,
    };
    (set(tr, train), set(te, test))
}

// ---------------------------------------------------------------------------------------------
// Task 51: Breast Cancer Wisconsin (Diagnostic), UCI WDBC.
// ---------------------------------------------------------------------------------------------

/// The SHA-256 of `wdbc.data` as distributed by the UCI Machine Learning Repository
/// (committed verbatim as `crates/ckks-train-bench/data/breast_cancer.csv`; see the README there).
pub const WDBC_SHA256: &str = "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a";

/// A tabular data set.
#[derive(Debug, Clone, PartialEq)]
pub struct Tabular {
    /// `n x features`.
    pub features: Array2<f64>,
    /// `n` class labels.
    pub labels: Vec<u8>,
    /// `n` record ids (WDBC's patient ids).
    pub ids: Vec<u64>,
}

/// Parses UCI's `wdbc.data`: one comma-separated record per line, `id, diagnosis (M | B),
/// 30 real features`; blank lines and `\r` are ignored. Labels follow scikit-learn's
/// `load_breast_cancer`: malignant `M` = 0, benign `B` = 1.
///
/// Errors (`InvalidData`): no records, a record without 32 fields, a non-integer id, another
/// diagnosis, or a feature that is not a finite number.
pub fn parse_wdbc(text: &str) -> io::Result<Tabular> {
    const FEATURES: usize = 30;
    let (mut x, mut labels, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    for (n, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() != FEATURES + 2 {
            return Err(invalid(format!(
                "WDBC line {}: {} fields, want {}",
                n + 1,
                fields.len(),
                FEATURES + 2
            )));
        }
        ids.push(
            fields[0]
                .parse::<u64>()
                .map_err(|e| invalid(format!("WDBC line {}: id {}: {e}", n + 1, fields[0])))?,
        );
        labels.push(match fields[1] {
            "M" => 0,
            "B" => 1,
            d => {
                return Err(invalid(format!(
                    "WDBC line {}: diagnosis {d} (M | B)",
                    n + 1
                )));
            }
        });
        for f in &fields[2..] {
            let v = f
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| invalid(format!("WDBC line {}: feature {f}", n + 1)))?;
            x.push(v);
        }
    }
    if labels.is_empty() {
        return Err(invalid("WDBC: no records".into()));
    }
    Ok(Tabular {
        features: Array2::from_shape_vec((labels.len(), FEATURES), x).expect("30 per record"),
        labels,
        ids,
    })
}

/// Reads `path`, checks it against [`WDBC_SHA256`] and parses it ([`parse_wdbc`]).
///
/// Errors: the file's I/O errors, `InvalidData` for a digest mismatch, and those of
/// [`parse_wdbc`].
pub fn load_wdbc(path: &Path) -> io::Result<Tabular> {
    let bytes =
        fs::read(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    wdbc_from_bytes(&bytes).map_err(|e| invalid(format!("{}: {e}", path.display())))
}

/// Checks the mandatory original UCI [`WDBC_SHA256`] before UTF-8 and WDBC decoding.
/// Consumes only these borrowed bytes; it cannot bypass the production source pin.
pub fn wdbc_from_bytes(bytes: &[u8]) -> io::Result<Tabular> {
    let got = sha256_hex(bytes);
    if got != WDBC_SHA256 {
        return Err(invalid(format!(
            "sha256 {got}, want UCI's wdbc.data {WDBC_SHA256}"
        )));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("WDBC: not UTF-8".into()))?;
    parse_wdbc(text)
}

/// A stratified train/test split: class `c` with `n_c` of the `n` samples puts
/// `floor(n_c n_test / n)` samples in the test set, the remaining test places going to the
/// largest fractional parts (ties to the lower class); the test samples of each class are the
/// first ones of a Fisher-Yates shuffle of that class's indices (classes in ascending order,
/// one `rng` stream). Returns `(train, test)`, each in ascending index order.
///
/// # Panics
/// If `n_test > labels.len()`.
pub fn stratified_split(labels: &[u8], n_test: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    let n = labels.len();
    assert!(n_test <= n, "{n_test} test samples of {n}");
    let mut classes: Vec<u8> = labels.to_vec();
    classes.sort_unstable();
    classes.dedup();
    let members: Vec<Vec<usize>> = classes
        .iter()
        .map(|&c| (0..n).filter(|&i| labels[i] == c).collect())
        .collect();
    // Quotas by the largest-remainder method, in exact integer arithmetic.
    let mut quota: Vec<usize> = members.iter().map(|m| m.len() * n_test / n).collect();
    let mut left = n_test - quota.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..members.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(members[k].len() * n_test % n));
    for &k in &order {
        if left == 0 {
            break;
        }
        quota[k] += 1;
        left -= 1;
    }
    let (mut train, mut test) = (Vec::with_capacity(n - n_test), Vec::with_capacity(n_test));
    for (m, &q) in members.iter().zip(&quota) {
        let mut idx = m.clone();
        for i in (1..idx.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            idx.swap(i, j);
        }
        test.extend_from_slice(&idx[..q]);
        train.extend_from_slice(&idx[q..]);
    }
    train.sort_unstable();
    test.sort_unstable();
    (train, test)
}

/// Standardization with the training statistics: each column of both sets becomes
/// `(x - mean) / std` with the training mean and population standard deviation (`ddof = 0`,
/// scikit-learn's `StandardScaler`); a constant training column is only centered.
pub fn standardize(train: &mut Array2<f64>, test: &mut Array2<f64>) {
    let n = train.nrows() as f64;
    for j in 0..train.ncols() {
        let mean = train.column(j).sum() / n;
        let var = train
            .column(j)
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n;
        let sd = if var > 0.0 { var.sqrt() } else { 1.0 };
        train.column_mut(j).mapv_inplace(|v| (v - mean) / sd);
        test.column_mut(j).mapv_inplace(|v| (v - mean) / sd);
    }
}

/// Per-column min-max scaling with the training ranges: each column of both sets becomes
/// `(x - min) / (max - min)` with the training column's minimum and maximum (scikit-learn's
/// `MinMaxScaler`; test values may leave `[0, 1]`); a constant training column is only shifted.
pub fn min_max_columns(train: &mut Array2<f64>, test: &mut Array2<f64>) {
    for j in 0..train.ncols() {
        let lo = train
            .column(j)
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let hi = train
            .column(j)
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let span = if hi > lo { hi - lo } else { 1.0 };
        train.column_mut(j).mapv_inplace(|v| (v - lo) / span);
        test.column_mut(j).mapv_inplace(|v| (v - lo) / span);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny MNIST-format data set: 3 images of 2 x 3 pixels.
    fn tiny() -> (Vec<u8>, Vec<u8>) {
        let pixels: Vec<u8> = (0..18).map(|i| (i * 15) as u8).collect();
        (
            write_idx_u8(&[3, 2, 3], &pixels),
            write_idx_u8(&[3], &[7, 0, 9]),
        )
    }

    #[test]
    fn idx_header_layout() {
        let bytes = write_idx_u8(&[2, 1], &[5, 6]);
        assert_eq!(bytes, [0, 0, 8, 2, 0, 0, 0, 2, 0, 0, 0, 1, 5, 6]);
        let idx = parse_idx_u8(&bytes).unwrap();
        assert_eq!(idx.dims, [2, 1]);
        assert_eq!(idx.data, [5, 6]);
    }

    #[test]
    fn idx_rejects_malformed_files() {
        let good = write_idx_u8(&[2, 2], &[1, 2, 3, 4]);
        assert!(parse_idx_u8(&good[..3]).is_err(), "short header");
        assert!(parse_idx_u8(&good[..10]).is_err(), "truncated dimensions");
        assert!(
            parse_idx_u8(&good[..good.len() - 1]).is_err(),
            "short payload"
        );
        let mut long = good.clone();
        long.push(0);
        assert!(parse_idx_u8(&long).is_err(), "trailing bytes");
        let mut magic = good.clone();
        magic[1] = 1;
        assert!(parse_idx_u8(&magic).is_err(), "magic");
        let mut ty = good.clone();
        ty[2] = 0x0d; // float
        assert!(parse_idx_u8(&ty).is_err(), "not unsigned byte");
        assert!(parse_idx_u8(&[0, 0, 8, 0]).is_err(), "zero dimensions");
        assert!(parse_idx_u8(&good).is_ok(), "the good file parses");
    }

    #[test]
    fn mnist_from_synthetic_idx_files() {
        let dir = std::env::temp_dir().join(format!("hekit-mnist-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let (img, lab) = tiny();
        let (fi, fl) = Split::Test.files();
        fs::write(dir.join(fi), &img).unwrap();
        fs::write(dir.join(fl), &lab).unwrap();
        let m = load_mnist(&dir, Split::Test, None).unwrap();
        assert_eq!((m.len(), m.rows, m.cols, m.features()), (3, 2, 3, 6));
        assert_eq!(m.labels, [7, 0, 9]);
        assert_eq!(m.images[[0, 0]], 0.0);
        assert_eq!(m.images[[1, 2]], (8 * 15) as f64 / 255.0);
        assert_eq!(m.images[[2, 5]], 1.0);
        let first2 = load_mnist(&dir, Split::Test, Some(2)).unwrap();
        assert_eq!(first2.len(), 2);
        assert_eq!(first2.images.row(1), m.images.row(1));
        assert_eq!(load_mnist(&dir, Split::Test, Some(10)).unwrap().len(), 3);
        assert!(
            load_mnist(&dir, Split::Train, None).is_err(),
            "missing files"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mnist_rejects_mismatched_arrays() {
        let (img, _) = tiny();
        let img = parse_idx_u8(&img).unwrap();
        let two = parse_idx_u8(&write_idx_u8(&[2], &[1, 2])).unwrap();
        assert!(
            mnist_from_idx(&img, &two, None).is_err(),
            "3 images, 2 labels"
        );
        assert!(mnist_from_idx(&two, &two, None).is_err(), "1-d images");
        let bad_label = parse_idx_u8(&write_idx_u8(&[3], &[1, 10, 2])).unwrap();
        assert!(
            mnist_from_idx(&img, &bad_label, None).is_err(),
            "label >= 10"
        );
        let short = IdxU8 {
            dims: vec![3, 2, 3],
            data: vec![0; 17],
        };
        let labels = parse_idx_u8(&write_idx_u8(&[3], &[1, 2, 3])).unwrap();
        assert!(
            mnist_from_idx(&short, &labels, None).is_err(),
            "hand-built IdxU8 with too little data"
        );
        let short_labels = IdxU8 {
            dims: vec![3],
            data: vec![1, 2],
        };
        assert!(mnist_from_idx(&img, &short_labels, None).is_err());
    }

    #[test]
    fn one_hot_rows() {
        let y = one_hot(&[2, 0], 3);
        assert_eq!(y.row(0).to_vec(), [0.0, 0.0, 1.0]);
        assert_eq!(y.row(1).to_vec(), [1.0, 0.0, 0.0]);
    }

    /// A handmade .npy (version 1.0): `header` padded to a multiple of 64 with `\n`.
    fn npy(descr: &str, fortran: bool, shape: &str, data: &[u8]) -> Vec<u8> {
        let mut h = format!(
            "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape}, }}",
            if fortran { "True" } else { "False" }
        );
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(' ');
        }
        h.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((h.len() as u16).to_le_bytes());
        out.extend(h.as_bytes());
        out.extend(data);
        out
    }

    #[test]
    fn npy_reader_handles_f8_f4_c_and_fortran_order() {
        let vals = [1.0f64, -2.5, 3.25, 0.125, 1e-3, -7.0];
        let f8: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let m = parse_npy_f64(&npy("<f8", false, "(2, 3)", &f8)).unwrap();
        assert_eq!(m, ndarray::arr2(&[[1.0, -2.5, 3.25], [0.125, 1e-3, -7.0]]));
        // Fortran order: the same bytes are column-major.
        let m = parse_npy_f64(&npy("<f8", true, "(2, 3)", &f8)).unwrap();
        assert_eq!(m, ndarray::arr2(&[[1.0, 3.25, 1e-3], [-2.5, 0.125, -7.0]]));
        let f4: Vec<u8> = vals
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect();
        let m = parse_npy_f64(&npy("<f4", false, "(3, 2)", &f4)).unwrap();
        assert_eq!(m[[2, 1]], -7.0);
        assert_eq!(m[[1, 1]], 0.125);
        let m = parse_npy_f64(&npy("<f8", false, "(6,)", &f8)).unwrap();
        assert_eq!(m.dim(), (1, 6));
        // Errors.
        assert!(parse_npy_f64(b"NUMPY").is_err());
        assert!(parse_npy_f64(&npy(">f8", false, "(2, 3)", &f8)).is_err());
        assert_eq!(
            parse_npy_f64(&npy("<i8", false, "(2, 3)", &f8))
                .unwrap()
                .dim(),
            (2, 3)
        );
        assert!(parse_npy_f64(&npy("<f8", false, "(2, 4)", &f8)).is_err());
        assert!(parse_npy_f64(&npy("<f8", false, "(1, 2, 3)", &f8)).is_err());
    }

    #[test]
    fn d7_sha_fallible_known_vectors() {
        for (input, expected) in [
            (
                &b""[..],
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                &b"abc"[..],
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
        ] {
            let digest = try_sha256_hex(input).unwrap();
            assert_eq!(digest, expected);
            assert_eq!(digest, sha256_hex(input));
        }
    }
    #[test]
    fn d7_sha_output_reservation_boundary() {
        let output = reserved_digest_output(64).unwrap();
        assert!(output.is_empty());
        assert!(output.capacity() >= 64);
        assert!(reserved_digest_output(usize::MAX).is_err());
    }
    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            sha256_hex(long),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn npy_dataset_with_manifest() {
        let dir = std::env::temp_dir().join(format!("hekit-npy-ds-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        // 3 training and 2 test images of 4 features (2 x 2), f4 images, i8 / u1 labels.
        let img = |n: usize, off: f32| -> Vec<u8> {
            (0..n * 4)
                .flat_map(|i| (off + i as f32 * 0.25).to_le_bytes())
                .collect()
        };
        let files: [(&str, Vec<u8>); 4] = [
            (
                NPY_DATASET_FILES[0],
                npy("<f4", false, "(3, 4)", &img(3, 0.0)),
            ),
            (
                NPY_DATASET_FILES[1],
                npy(
                    "<i8",
                    false,
                    "(3,)",
                    &[7i64, 0, 9]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                NPY_DATASET_FILES[2],
                npy("<f4", false, "(2, 4)", &img(2, 1.0)),
            ),
            (NPY_DATASET_FILES[3], npy("|u1", false, "(2,)", &[3u8, 5])),
        ];
        let mut manifest = String::from("{\"files\": [\n");
        for (name, bytes) in &files {
            fs::write(dir.join(name), bytes).unwrap();
            manifest.push_str(&format!(
                "  {{\"file\": \"{name}\", \"sha256\": \"{}\"}},\n",
                sha256_hex(bytes)
            ));
        }
        manifest.push_str("]}\n");
        fs::write(dir.join("data_manifest.json"), &manifest).unwrap();
        let (train, test) = load_npy_dataset(&dir).unwrap();
        assert_eq!((train.len(), train.features(), train.rows), (3, 4, 2));
        assert_eq!(train.labels, vec![7, 0, 9]);
        assert_eq!(test.labels, vec![3, 5]);
        assert_eq!(train.images[[1, 0]], 1.0);
        assert_eq!(test.images[[0, 1]], 1.25);
        // A tampered file is refused.
        let mut bad = files[2].1.clone();
        *bad.last_mut().unwrap() ^= 1;
        fs::write(dir.join(NPY_DATASET_FILES[2]), &bad).unwrap();
        assert!(load_npy_dataset(&dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    // ---- Task 51: resize, scaling, WDBC, split ----

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-14
    }

    /// Reference weights printed by an independent NumPy transcription of the antialiased
    /// bicubic filter (a = -0.5, support scaled by the downscale factor, weights normalized).
    #[test]
    fn bicubic_aa_weights_match_reference() {
        let m = bicubic_aa_weights(28, 14);
        assert_eq!(m.dim(), (14, 28));
        let row0 = [
            0.46443514644351463,
            0.46443514644351463,
            0.12133891213389121,
            -0.03765690376569038,
            -0.01255230125523013,
            0.0,
        ];
        for (j, &w) in row0.iter().enumerate() {
            assert!(close(m[[0, j]], w), "row 0 col {j}: {}", m[[0, j]]);
        }
        let row7 = [
            0.0,
            -0.01171875,
            -0.03515625,
            0.11328125,
            0.43359375,
            0.43359375,
            0.11328125,
            -0.03515625,
            -0.01171875,
            0.0,
        ];
        for (k, &w) in row7.iter().enumerate() {
            assert!(
                close(m[[7, 10 + k]], w),
                "row 7 col {}: {}",
                10 + k,
                m[[7, 10 + k]]
            );
        }
        for row in m.rows() {
            assert!(close(row.sum(), 1.0));
        }
        // Last row mirrors the first.
        for j in 0..28 {
            assert!(close(m[[13, 27 - j]], m[[0, j]]));
        }
    }

    #[test]
    fn resize_bicubic_aa_reference_image() {
        // A 6 x 6 image (i^2 for i = 0..36, row-major) resized to 3 x 3 (NumPy reference).
        let img = Array2::from_shape_fn((1, 36), |(_, i)| (i * i) as f64);
        let out = resize_bicubic_aa(&img, 6, 6, 3, 3);
        let want = [
            18.72190612909438,
            37.61003225909483,
            63.81784282488051,
            251.82727011402474,
            316.6297709923664,
            388.75195630649335,
            748.4412737872235,
            859.1581494139065,
            977.1947094763747,
        ];
        assert_eq!(out.dim(), (1, 9));
        for (j, &w) in want.iter().enumerate() {
            assert!(
                (out[[0, j]] - w).abs() < 1e-10,
                "{j}: {} vs {w}",
                out[[0, j]]
            );
        }
        // A constant image stays constant; rows are resized independently.
        let two = Array2::from_shape_fn((2, 16), |(r, _)| r as f64 + 0.5);
        let out = resize_bicubic_aa(&two, 4, 4, 2, 2);
        assert!(out.row(0).iter().all(|&v| close(v, 0.5)));
        assert!(out.row(1).iter().all(|&v| close(v, 1.5)));
    }

    #[test]
    fn min_max_scaling_uses_the_training_range() {
        let mut train = ndarray::arr2(&[[-1.0, 1.0], [3.0, 0.0]]);
        let mut test = ndarray::arr2(&[[5.0, -3.0]]);
        let (lo, hi) = min_max_scale(&mut train, &mut test);
        assert_eq!((lo, hi), (-1.0, 3.0));
        assert_eq!(train, ndarray::arr2(&[[0.0, 0.5], [1.0, 0.25]]));
        assert_eq!(test, ndarray::arr2(&[[1.5, -0.5]]));
    }

    /// Two WDBC rows in the UCI file's layout (id, diagnosis, 30 features).
    fn wdbc_rows() -> String {
        let f = |k: usize| {
            (0..30)
                .map(|j| format!("{}", (k * 100 + j) as f64 / 8.0))
                .collect::<Vec<_>>()
                .join(",")
        };
        format!("842302,M,{}\n8510426,B,{}\n", f(1), f(2))
    }

    #[test]
    fn wdbc_parses_the_uci_layout() {
        let t = parse_wdbc(&wdbc_rows()).unwrap();
        assert_eq!(t.features.dim(), (2, 30));
        assert_eq!(t.ids, [842302, 8510426]);
        // sklearn's convention: malignant = 0, benign = 1.
        assert_eq!(t.labels, [0, 1]);
        assert_eq!(t.features[[0, 0]], 100.0 / 8.0);
        assert_eq!(t.features[[1, 29]], 229.0 / 8.0);
        // No trailing newline, CRLF and blank lines are accepted.
        let crlf = wdbc_rows().replace('\n', "\r\n") + "\r\n";
        assert_eq!(parse_wdbc(&crlf).unwrap().features, t.features);
        assert_eq!(parse_wdbc(wdbc_rows().trim_end()).unwrap().labels, [0, 1]);
    }

    #[test]
    fn wdbc_rejects_malformed_rows() {
        let good = wdbc_rows();
        assert!(
            parse_wdbc(&good.replacen(",M,", ",X,", 1)).is_err(),
            "diagnosis"
        );
        assert!(
            parse_wdbc(&good.replacen(",12.5,", ",abc,", 1)).is_err(),
            "number"
        );
        let short = good
            .lines()
            .next()
            .unwrap()
            .rsplit_once(',')
            .unwrap()
            .0
            .to_string();
        assert!(parse_wdbc(&short).is_err(), "29 features");
        assert!(parse_wdbc("").is_err(), "empty");
        assert!(parse_wdbc(&good.replacen("842302", "x1", 1)).is_err(), "id");
    }

    #[test]
    fn stratified_split_keeps_class_proportions() {
        // 212 malignant (0), 357 benign (1), as WDBC; 114 test samples.
        let labels: Vec<u8> = (0..569).map(|i| u8::from(i % 569 >= 212)).collect();
        let (train, test) = stratified_split(&labels, 114, &mut Rng::from_seed([7; 32]));
        assert_eq!((train.len(), test.len()), (455, 114));
        let test_m = test.iter().filter(|&&i| labels[i] == 0).count();
        assert_eq!(test_m, 42, "round(212 * 114 / 569) = 42");
        let mut all: Vec<usize> = train.iter().chain(&test).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..569).collect::<Vec<_>>(), "a partition");
        assert!(train.windows(2).all(|w| w[0] < w[1]) && test.windows(2).all(|w| w[0] < w[1]));
        let again = stratified_split(&labels, 114, &mut Rng::from_seed([7; 32]));
        assert_eq!(again, (train.clone(), test.clone()), "deterministic");
        let other = stratified_split(&labels, 114, &mut Rng::from_seed([8; 32]));
        assert_ne!(other.1, test, "seed-dependent");
    }

    #[test]
    fn stratified_split_largest_remainder() {
        // 3 classes of 5 samples, 7 test: quotas 2.33 each -> 2, 2, 2 plus one by remainder.
        let labels: Vec<u8> = (0..15).map(|i| (i / 5) as u8).collect();
        let (train, test) = stratified_split(&labels, 7, &mut Rng::from_seed([1; 32]));
        assert_eq!((train.len(), test.len()), (8, 7));
        for c in 0..3u8 {
            let k = test.iter().filter(|&&i| labels[i] == c).count();
            assert!((2..=3).contains(&k), "class {c}: {k}");
        }
    }

    #[test]
    fn standardize_uses_training_statistics() {
        let mut train = ndarray::arr2(&[[1.0, 5.0], [3.0, 5.0]]);
        let mut test = ndarray::arr2(&[[5.0, 6.0]]);
        standardize(&mut train, &mut test);
        // Column 0: mean 2, population std 1; column 1: constant (std 0 -> only centered).
        assert_eq!(train, ndarray::arr2(&[[-1.0, 0.0], [1.0, 0.0]]));
        assert_eq!(test, ndarray::arr2(&[[3.0, 1.0]]));
    }

    #[test]
    fn min_max_columns_uses_training_ranges() {
        let mut train = ndarray::arr2(&[[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]]);
        let mut test = ndarray::arr2(&[[4.0, 6.0]]);
        min_max_columns(&mut train, &mut test);
        // Column 0: range [1, 3]; column 1: constant (only shifted).
        assert_eq!(train, ndarray::arr2(&[[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]));
        assert_eq!(test, ndarray::arr2(&[[1.5, 1.0]]));
    }

    /// The committed WDBC file is the UCI original (SHA-256) and has 569 rows, 212 / 357.
    #[test]
    fn committed_wdbc_file() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/breast_cancer.csv");
        let t = load_wdbc(&path).unwrap();
        assert_eq!(t.features.dim(), (569, 30));
        assert_eq!(t.labels.iter().filter(|&&l| l == 0).count(), 212);
        assert_eq!(t.features[[0, 0]], 17.99);
        let bad = std::env::temp_dir().join(format!("hekit-wdbc-{}", std::process::id()));
        fs::write(&bad, wdbc_rows()).unwrap();
        assert!(load_wdbc(&bad).is_err(), "sha256 mismatch");
        fs::remove_file(&bad).unwrap();
    }

    /// Fashion-MNIST is read by the MNIST IDX loader: 60 000 / 10 000 images of 28 x 28 and
    /// 10 balanced classes (needs `scripts/fetch_fashion_mnist.sh`).
    #[test]
    #[ignore = "needs data/fashion_mnist (scripts/fetch_fashion_mnist.sh)"]
    fn fashion_mnist_idx_files() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/fashion_mnist");
        for (split, n) in [(Split::Train, 60_000), (Split::Test, 10_000)] {
            let m = load_mnist(&dir, split, None).expect("scripts/fetch_fashion_mnist.sh");
            assert_eq!((m.len(), m.rows, m.cols), (n, 28, 28));
            for c in 0..10u8 {
                assert_eq!(m.labels.iter().filter(|&&l| l == c).count(), n / 10);
            }
        }
    }

    /// The resize plus min-max scaling reproduces ReBoot's 14 x 14 MNIST arrays (R-REB-3,
    /// gitignored): needs `data/mnist` and `results/quadruped/data/reboot_emlp1_paper`.
    #[test]
    #[ignore = "needs data/mnist and the gitignored R-REB-3 arrays"]
    fn resize_matches_reboot_14x14_mnist() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mnist = root.join("data/mnist");
        let (tr, te) = reboot_preprocess(
            &load_mnist(&mnist, Split::Train, None).expect("scripts/fetch_mnist.sh"),
            &load_mnist(&mnist, Split::Test, None).expect("scripts/fetch_mnist.sh"),
            14,
        );
        let d = root.join("results/quadruped/data/reboot_emlp1_paper");
        let rd = |f: &str| read_npy_f64(&d.join(f)).expect("R-REB-3 arrays");
        let rtr = rd("train_images_14x14.npy");
        let rte = rd("test_images_14x14.npy");
        let ltr: Vec<u8> = rd("train_labels.npy").iter().map(|&v| v as u8).collect();
        let lte: Vec<u8> = rd("test_labels.npy").iter().map(|&v| v as u8).collect();
        for (ours, theirs, labels) in [(&tr, &rtr, &ltr), (&te, &rte, &lte)] {
            assert_eq!((ours.rows, ours.cols), (14, 14));
            assert_eq!(&ours.labels, labels);
            let e = ours
                .images
                .iter()
                .zip(theirs)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("max |ours - ReBoot| = {e:.3e} over {} images", ours.len());
            // ReBoot's arrays are float32.
            assert!(e < 1e-6, "max |ours - ReBoot| = {e}");
        }
    }
}

#[cfg(test)]
mod streaming_hash_tests {
    use super::*;
    // Frozen Python hashlib vectors, independently crosschecked with shasum before implementation.
    #[test]
    fn streaming_sha_independent_vectors_and_empty_updates() {
        let cases = [
            (
                0,
                "cycle_u8_0_through_255",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                1,
                "cycle_u8_0_through_255",
                "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            ),
            (
                55,
                "cycle_u8_0_through_255",
                "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59",
            ),
            (
                56,
                "cycle_u8_0_through_255",
                "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562",
            ),
            (
                63,
                "cycle_u8_0_through_255",
                "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488",
            ),
            (
                64,
                "cycle_u8_0_through_255",
                "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108",
            ),
            (
                65,
                "cycle_u8_0_through_255",
                "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781",
            ),
            (
                127,
                "cycle_u8_0_through_255",
                "92ca0fa6651ee2f97b884b7246a562fa71250fedefe5ebf270d31c546bfea976",
            ),
            (
                128,
                "cycle_u8_0_through_255",
                "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5",
            ),
            (
                129,
                "cycle_u8_0_through_255",
                "5099c6a56203f9687f7d33f4bfdf576d31dc91f6b695ecea38b2770c87631135",
            ),
            (
                1048575,
                "cycle_u8_0_through_255",
                "d8222d77ac43a9391cb348ce3a3bb3c0221a4a666e907b5bc77de56a41f37848",
            ),
            (
                1048576,
                "cycle_u8_0_through_255",
                "fbbab289f7f94b25736c58be46a994c441fd02552cc6022352e3d86d2fab7c83",
            ),
            (
                1048577,
                "cycle_u8_0_through_255",
                "607deb6eccbc844880b9d7b523751a4cdba0452727b885c74264bfe1fb7843e2",
            ),
            (
                3,
                "literal_hex",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                1000000,
                "ascii_repeat",
                "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            ),
        ];
        for (len, kind, expected) in cases {
            let data: Vec<u8> = match kind {
                "literal_hex" => b"abc".to_vec(),
                "ascii_repeat" => vec![b'a'; len],
                _ => (0..len).map(|n| (n % 256) as u8).collect(),
            };
            assert_eq!(sha256_hex(&data), expected);
            for schedule in [
                &[1][..],
                &[7, 31, 64][..],
                &[55, 56, 63, 64, 65][..],
                &[1048576][..],
            ] {
                let mut h = Sha256State::new();
                let mut pos = 0;
                let mut i = 0;
                h.update(&[]);
                while pos < data.len() {
                    let end = (pos + schedule[i % schedule.len()]).min(data.len());
                    h.update(&data[pos..end]);
                    h.update(&[]);
                    pos = end;
                    i += 1;
                }
                assert_eq!(h.finish(), expected, "length={len} schedule={schedule:?}");
            }
        }
    }
}
