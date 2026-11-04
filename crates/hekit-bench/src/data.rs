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
}
