//! Closed read custody. No publication, gate, or caller-authored FileRef authority.
#![forbid(unsafe_code)]
use super::*;
use crate::study_profile::{StudySource, StudySourceRole};
use std::os::unix::fs::FileExt;

pub(super) enum ReadBound {
    AtMost(usize),
    Exact(usize),
}
impl ReadBound {
    pub(super) fn check(&self, length: u64) -> Result<usize> {
        let length = usize::try_from(length).map_err(|_| ProvenanceError("file size overflow"))?;
        ensure(
            match self {
                Self::AtMost(cap) => length <= *cap,
                Self::Exact(size) => length == *size,
            },
            "retained read size policy",
        )?;
        Ok(length)
    }
}
pub(super) enum ReadRole {
    Published(Kind),
    Target,
    Source(StudySourceRole),
}
/// Final FD and every ancestor remain held, independently of the short store borrow.
/// Owned bytes are immutable; parsing is always transient under a borrow.
pub(crate) struct HeldRead {
    file: File,
    anchors: Anchors,
    name: String,
    stamp: ReadStamp,
    bytes: Box<[u8]>,
    role: ReadRole,
    _not_send_sync: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl HeldRead {
    pub(super) fn acquire(
        anchors: Anchors,
        name: String,
        bound: ReadBound,
        role: ReadRole,
    ) -> Result<Self> {
        Self::acquire_on(anchors, name, bound, role, |_| Ok(()))
    }
    // Per-invocation seam only. No global mutation, production callers or gate constructor.
    fn acquire_on(
        anchors: Anchors,
        name: String,
        bound: ReadBound,
        role: ReadRole,
        mut phase: impl FnMut(ReadPhase) -> Result<()>,
    ) -> Result<Self> {
        anchors.recheck()?;
        let mut file = openat(anchors.last().as_raw_fd(), OsStr::new(&name), FILE_FLAGS, 0)?;
        let metadata = file.metadata()?;
        bound.check(metadata.len())?; // Before raw allocation.
        validate_file(&file, &role)?;
        same_publication_mount(anchors.root_mount, mount(&file)?)?;
        let stamp = ReadStamp {
            file: identity(&metadata),
            parent: inode(anchors.last())?,
        };
        let length = bound.check(metadata.len())?;
        phase(ReadPhase::Opened)?;
        let bytes =
            read_exact_on(&mut file, length, || phase(ReadPhase::DuringRead))?.into_boxed_slice();
        phase(ReadPhase::AfterRead)?;
        ensure(
            identity(&file.metadata()?) == stamp.file,
            "retained acquisition changed",
        )?;
        let result = Self {
            file,
            anchors,
            name,
            stamp,
            bytes,
            role,
            _not_send_sync: std::marker::PhantomData,
        };
        result.recheck()?;
        Ok(result)
    }
    pub(super) fn recheck_target(&self, parent: &File) -> Result<()> {
        ensure(
            matches!(self.role, ReadRole::Target)
                && self.name == "target.lock.json"
                && self.stamp.parent == inode(parent)?,
            "target read origin changed",
        )?;
        self.recheck()
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub(crate) fn recheck(&self) -> Result<()> {
        self.anchors.recheck()?;
        ensure(
            inode(self.anchors.last())? == self.stamp.parent,
            "retained parent changed",
        )?;
        validate_file(&self.file, &self.role)?;
        same_publication_mount(self.anchors.root_mount, mount(&self.file)?)?;
        ensure(
            identity(&self.file.metadata()?) == self.stamp.file,
            "retained file changed",
        )?;
        let current = openat(
            self.anchors.last().as_raw_fd(),
            OsStr::new(&self.name),
            FILE_FLAGS,
            0,
        )?;
        validate_file(&current, &self.role)?;
        same_publication_mount(self.anchors.root_mount, mount(&current)?)?;
        ensure(
            identity(&current.metadata()?) == self.stamp.file,
            "retained file namespace changed",
        )?;
        // Compare using the original FD. Namespace reopen is identity checking only.
        let mut scratch = [0u8; 65536];
        let mut offset = 0;
        while offset < self.bytes.len() {
            let n = (self.bytes.len() - offset).min(scratch.len());
            self.file.read_exact_at(&mut scratch[..n], offset as u64)?;
            ensure(
                scratch[..n] == self.bytes[offset..offset + n],
                "retained bytes changed",
            )?;
            offset += n;
        }
        ensure(
            self.file.read_at(&mut scratch[..1], offset as u64)? == 0,
            "retained file grew",
        )?;
        ensure(
            identity(&self.file.metadata()?) == self.stamp.file
                && identity(&current.metadata()?) == self.stamp.file,
            "retained read changed",
        )?;
        // check_child is directory-only (O_DIRECTORY); final files use FILE_FLAGS.
        let after = openat(
            self.anchors.last().as_raw_fd(),
            OsStr::new(&self.name),
            FILE_FLAGS,
            0,
        )?;
        validate_file(&after, &self.role)?;
        same_publication_mount(self.anchors.root_mount, mount(&after)?)?;
        ensure(
            identity(&after.metadata()?) == self.stamp.file,
            "retained final namespace changed",
        )?;
        self.anchors.recheck()
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadPhase {
    Opened,
    DuringRead,
    AfterRead,
}
#[cfg(test)]
fn read_exact_bounded(reader: &mut impl Read, length: usize) -> Result<Vec<u8>> {
    read_exact_on(reader, length, || Ok(()))
}
fn read_exact_on(
    reader: &mut impl Read,
    length: usize,
    halfway: impl FnOnce() -> Result<()>,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| ProvenanceError("retained raw allocation unavailable"))?;
    bytes.resize(length, 0);
    let half = length / 2;
    reader.read_exact(&mut bytes[..half])?;
    halfway()?;
    reader.read_exact(&mut bytes[half..])?;
    let mut extra = [0];
    ensure(
        reader.read(&mut extra)? == 0,
        "retained oversized/growing read",
    )?;
    Ok(bytes)
}
fn validate_file(file: &File, role: &ReadRole) -> Result<()> {
    match role {
        ReadRole::Published(kind) => {
            let _ = kind.directory();
            safe_file(file)
        }
        ReadRole::Target => safe_file(file),
        ReadRole::Source(role) => {
            let _ = role;
            let m = file.metadata()?;
            // Source checkout permits read/executable modes, never group/world writable
            // or special permission bits. Publication records retain their exact0600 rule.
            ensure(
                source_metadata(m.is_file(), m.uid(), m.mode(), effective_uid()),
                "source file owner/mode/type",
            )
        }
    }
}
fn source_metadata(regular: bool, uid: u32, mode: u32, effective: u32) -> bool {
    regular && (uid == 0 || uid == effective) && mode & 0o7022 == 0
}
pub(super) fn dataset_size(path: &str) -> Result<usize> {
    match path {
        "crates/ckks-train-bench/data/iris.csv" => Ok(2756),
        "crates/ckks-train-bench/data/iris_split_seed2020.txt" => Ok(705),
        "crates/ckks-train-bench/data/breast_cancer.csv" => Ok(124103),
        "results/quadruped/data/reboot_emlp1_full/train_images_14x14.npy" => Ok(3763328),
        "results/quadruped/data/reboot_emlp1_full/train_labels.npy" => Ok(4928),
        "results/quadruped/data/reboot_emlp1_full/test_images_14x14.npy" => Ok(7840128),
        "results/quadruped/data/reboot_emlp1_full/test_labels.npy" => Ok(10128),
        "data/fashion_mnist/train-images-idx3-ubyte" => Ok(47040016),
        "data/fashion_mnist/train-labels-idx1-ubyte" => Ok(60008),
        "data/fashion_mnist/t10k-images-idx3-ubyte" => Ok(7840016),
        "data/fashion_mnist/t10k-labels-idx1-ubyte" => Ok(10008),
        _ => Err(ProvenanceError("unregistered DatasetInput representation")),
    }
}
fn check_source_digest(bytes: &[u8], expected: &str) -> Result<()> {
    ensure(
        sha256_hex(bytes) == expected,
        "selected retained source digest",
    )
}
/// Only the actual context's complete resolver-selected inventory can call this.
/// This acquisition does not retroactively bound earlier materialization work.
pub(in crate::study_provenance) fn selected_sources(
    root: &Path,
    sources: &[StudySource],
) -> Result<Vec<HeldRead>> {
    let root = Anchors::open_source(root)?;
    root.recheck()?;
    let mut held = Vec::new();
    held.try_reserve_exact(sources.len())
        .map_err(|_| ProvenanceError("source custody allocation"))?;
    for source in sources {
        let path = Path::new(source.path());
        ensure(
            !path.is_absolute() && path.components().all(|c| matches!(c, Component::Normal(_))),
            "source relative path",
        )?;
        let mut anchors = root.duplicate()?;
        let mut names = path.components().peekable();
        let mut final_name = None;
        while let Some(Component::Normal(name)) = names.next() {
            if names.peek().is_none() {
                final_name = Some(
                    name.to_str()
                        .ok_or(ProvenanceError("source path encoding"))?
                        .to_owned(),
                );
            } else {
                anchors.source_child(name)?;
            }
        }
        let bound = if source.role() == StudySourceRole::DatasetInput {
            ReadBound::Exact(dataset_size(source.path())?)
        } else {
            ReadBound::AtMost(8 * 1024 * 1024)
        };
        let file = HeldRead::acquire(
            anchors,
            final_name.ok_or(ProvenanceError("source filename"))?,
            bound,
            ReadRole::Source(source.role()),
        )?;
        check_source_digest(file.bytes(), source.sha256())?;
        held.push(file);
    }
    for file in &held {
        file.recheck()?;
    }
    root.recheck()?;
    Ok(held)
}
impl Anchors {
    fn open_source(path: &Path) -> Result<Self> {
        // Reuse the same descriptor walk but source roots need not be private0700.
        let mut result = Self::open_mode(path, false)?;
        result.private_from = usize::MAX;
        result.recheck()?;
        Ok(result)
    }
    fn source_child(&mut self, name: &OsStr) -> Result<()> {
        self.recheck()?;
        let file = openat(self.last().as_raw_fd(), name, DIR_FLAGS, 0)?;
        safe_directory(&file, false)?;
        same_publication_mount(self.root_mount, mount(&file)?)?;
        self.files.push(file);
        self.names.push(name.into());
        self.recheck()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_bounded_reads_short_truncated_growing_and_complete() {
        for n in [0, 1, 3, 4, 5] {
            let data = vec![9; n];
            let result = read_exact_bounded(&mut data.as_slice(), 4);
            assert_eq!(result.is_ok(), n == 4);
        }
        // One-byte reads cover a complete bounded read independently of Read's chunking.
        struct Short<'a>(&'a [u8]);
        impl Read for Short<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                if output.is_empty() || self.0.is_empty() {
                    return Ok(0);
                }
                output[0] = self.0[0];
                self.0 = &self.0[1..];
                Ok(1)
            }
        }
        assert_eq!(read_exact_bounded(&mut Short(b"data"), 4).unwrap(), b"data");
        assert!(
            read_exact_on(&mut b"data".as_slice(), 4, || Err(ProvenanceError(
                "mid-read failure"
            )))
            .is_err()
        );
    }
    #[test]
    fn retained_source_type_owner_and_mode_policy() {
        assert!(source_metadata(true, 42, 0o100644, 42));
        assert!(source_metadata(true, 0, 0o100755, 42));
        assert!(!source_metadata(true, 43, 0o100644, 42));
        assert!(!source_metadata(false, 42, 0o100644, 42));
        for mode in [0o666, 0o664, 0o4755, 0o2755, 0o1755] {
            assert!(!source_metadata(true, 42, mode, 42));
        }
    }
    #[test]
    fn retained_same_size_wrong_digest_refuses() {
        const HASH: &str = "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7";
        check_source_digest(b"data", HASH).unwrap();
        assert!(check_source_digest(b"date", HASH).is_err());
    }
    #[cfg(target_os = "linux")]
    struct Scratch(PathBuf);
    #[cfg(target_os = "linux")]
    impl Scratch {
        fn new(label: &str) -> Self {
            use std::os::unix::fs::PermissionsExt;
            let root = PathBuf::from(
                std::env::var_os("HEKIT_PUBLICATION_NATIVE_SCRATCH")
                    .expect("separately released safe ext4/XFS scratch required"),
            );
            let anchors = Anchors::open(&root).unwrap();
            assert_ne!(
                anchors.last().metadata().unwrap().uid(),
                0,
                "unprivileged native actor required"
            );
            let path = root.join(format!("retained-{label}-{}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
            Self(path)
        }
        fn write(&self, name: &str, bytes: &[u8]) {
            use std::os::unix::fs::PermissionsExt;
            std::fs::write(self.0.join(name), bytes).unwrap();
            std::fs::set_permissions(self.0.join(name), std::fs::Permissions::from_mode(0o600))
                .unwrap();
        }
        fn acquire(&self) -> Result<HeldRead> {
            HeldRead::acquire(
                Anchors::open(&self.0)?,
                "input".into(),
                ReadBound::Exact(4),
                ReadRole::Target,
            )
        }
    }
    #[cfg(target_os = "linux")]
    impl Drop for Scratch {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn retained_native_fd_ancestors_survive_store_scope_and_replacement_refuses() {
        let d = Scratch::new("original-fd");
        d.write("input", b"same");
        let held = d.acquire().unwrap();
        let original = held.file.as_raw_fd();
        held.recheck().unwrap();
        assert_eq!(original, held.file.as_raw_fd());
        std::fs::rename(d.0.join("input"), d.0.join("old")).unwrap();
        d.write("input", b"same");
        assert!(held.recheck().is_err());
        let mut bytes = [0; 4];
        held.file.read_exact_at(&mut bytes, 0).unwrap();
        assert_eq!(&bytes, b"same");
        assert_ne!(
            inode(&held.file).unwrap(),
            inode(&File::open(d.0.join("input")).unwrap()).unwrap()
        );
    }
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn retained_native_before_during_after_equal_byte_namespace_races() {
        for (index, selected) in [
            ReadPhase::Opened,
            ReadPhase::DuringRead,
            ReadPhase::AfterRead,
        ]
        .into_iter()
        .enumerate()
        {
            let d = Scratch::new(&format!("race-{index}"));
            d.write("input", b"same");
            let result = HeldRead::acquire_on(
                Anchors::open(&d.0).unwrap(),
                "input".into(),
                ReadBound::Exact(4),
                ReadRole::Target,
                |phase| {
                    if phase == selected {
                        std::fs::rename(d.0.join("input"), d.0.join("old"))?;
                        d.write("input", b"same");
                    }
                    Ok(())
                },
            );
            assert!(result.is_err());
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn retained_native_truncated_growing_and_equal_size_inplace_reads() {
        for (index, bytes) in [&b"x"[..], &b"extra"[..], &b"else"[..]]
            .into_iter()
            .enumerate()
        {
            let d = Scratch::new(&format!("size-race-{index}"));
            d.write("input", b"data");
            let result = HeldRead::acquire_on(
                Anchors::open(&d.0).unwrap(),
                "input".into(),
                ReadBound::Exact(4),
                ReadRole::Target,
                |phase| {
                    if phase == ReadPhase::DuringRead {
                        d.write("input", bytes);
                    }
                    Ok(())
                },
            );
            assert!(result.is_err());
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires separately released Linux ext4/XFS scratch and native execution"]
    fn retained_native_wrong_size_type_mode_and_ancestor_replacement() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let d = Scratch::new("invalid");
        for bytes in [&b"abc"[..], &b"abcde"[..]] {
            d.write("input", bytes);
            assert!(d.acquire().is_err());
        }
        d.write("input", b"data");
        std::fs::set_permissions(d.0.join("input"), std::fs::Permissions::from_mode(0o644))
            .unwrap();
        assert!(d.acquire().is_err());
        std::fs::remove_file(d.0.join("input")).unwrap();
        std::fs::create_dir(d.0.join("input")).unwrap();
        assert!(d.acquire().is_err());
        std::fs::remove_dir(d.0.join("input")).unwrap();
        d.write("other", b"data");
        symlink(d.0.join("other"), d.0.join("input")).unwrap();
        assert!(d.acquire().is_err());
        std::fs::remove_file(d.0.join("input")).unwrap();
        d.write("input", b"data");
        let held = d.acquire().unwrap();
        let moved = d.0.with_extension("moved");
        std::fs::rename(&d.0, &moved).unwrap();
        std::fs::create_dir(&d.0).unwrap();
        std::fs::set_permissions(&d.0, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(held.recheck().is_err());
        std::fs::remove_dir_all(moved).unwrap();
        std::fs::set_permissions(&d.0, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(Anchors::open(&d.0).is_err());
    }
}
