//! Data sets for the training benchmarks: the MNIST IDX loader (Task 32).
//!
//! MNIST is **not** in the repository (licence and size): `scripts/fetch_mnist.sh` downloads
//! the four IDX files into the gitignored `data/mnist/` and unpacks them. The IDX format
//! (LeCun et al.) is a big-endian header `00 00 <type> <ndim>`, `ndim` big-endian `u32`
//! dimensions, then the row-major data; MNIST uses type `0x08` (unsigned byte), images as
//! `n x 28 x 28` and labels as `n`.

use std::fs;
use std::io;
use std::path::Path;

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
    let images = read_idx_u8(&dir.join(img))?;
    let labels = read_idx_u8(&dir.join(lab))?;
    mnist_from_idx(&images, &labels, limit)
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
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend(bits.to_be_bytes());
    for block in msg.chunks_exact(64) {
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
        let mut v = h;
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
        for (hi, vi) in h.iter_mut().zip(v) {
            *hi = hi.wrapping_add(vi);
        }
    }
    h.iter().fold(String::with_capacity(64), |mut out, x| {
        use std::fmt::Write as _;
        write!(out, "{x:08x}").expect("writing to a String");
        out
    })
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
    let manifest = fs::read_to_string(dir.join("data_manifest.json"))?;
    let mut arrays = Vec::with_capacity(4);
    for file in NPY_DATASET_FILES {
        let bytes = fs::read(dir.join(file))?;
        let want = manifest_sha256(&manifest, file)
            .ok_or_else(|| invalid(format!("data_manifest.json lists no sha256 for {file}")))?;
        let got = sha256_hex(&bytes);
        if got != want {
            return Err(invalid(format!("{file}: sha256 {got}, manifest {want}")));
        }
        arrays.push(parse_npy_f64(&bytes)?);
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
}
