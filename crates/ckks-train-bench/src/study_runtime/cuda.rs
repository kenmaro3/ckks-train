//! Retained top-level CUDA module descriptors, not complete native-linker attestation.
use super::{ProvenanceError, Result, check_runtime_environment, require};
use crate::data::{Sha256State, sha256_hex};
use ckks_train_cuda::{CudaLibraryKind, CudaRuntimeObservation};
use std::{
    ffi::{CStr, c_void},
    fs::{File, Metadata, OpenOptions},
    io::{ErrorKind, Read},
    os::unix::{
        ffi::OsStrExt,
        fs::{FileExt, MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
};
pub(super) const MODULE_CAP: u64 = 1024 * 1024 * 1024;
const CHUNK: usize = 1024 * 1024;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ModuleOwner {
    pub(super) path: PathBuf,
    loaded_path: PathBuf,
    pub(super) base: usize,
}
pub(super) fn require_same_owner(a: &ModuleOwner, b: &ModuleOwner) -> Result<()> {
    require(a == b, "native exported function owner changed")
}
pub(super) fn resolve_owner(address: *const c_void) -> Result<ModuleOwner> {
    require(!address.is_null(), "native export missing")?;
    // SAFETY: dladdr inspects the address without calling it. Only real retained
    // cudarc export addresses reach this function in production.
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    require(
        unsafe { libc::dladdr(address, &mut info) } != 0,
        "native export owner unavailable",
    )?;
    require(
        !info.dli_fname.is_null() && !info.dli_fbase.is_null(),
        "native export owner missing",
    )?;
    // SAFETY: successful dladdr supplies a loader-owned NUL-terminated name.
    let bytes = unsafe { CStr::from_ptr(info.dli_fname) }.to_bytes();
    require(
        !bytes.is_empty() && bytes.len() <= 4096,
        "native module path bounds",
    )?;
    let loaded_path = PathBuf::from(std::ffi::OsStr::from_bytes(bytes));
    require(
        loaded_path.is_absolute(),
        "native module absolute path required",
    )?;
    let path = loaded_path.canonicalize()?;
    Ok(ModuleOwner {
        path,
        loaded_path,
        base: info.dli_fbase as usize,
    })
}
#[derive(Debug, PartialEq, Eq)]
struct Identity {
    dev: u64,
    ino: u64,
    size: u64,
    mtime: i64,
    mtime_nsec: i64,
    ctime: i64,
    ctime_nsec: i64,
}
fn identity(m: &Metadata) -> Result<Identity> {
    require(m.is_file(), "native module must be regular")?;
    require(m.len() <= MODULE_CAP, "native module byte cap")?;
    Ok(Identity {
        dev: m.dev(),
        ino: m.ino(),
        size: m.len(),
        mtime: m.mtime(),
        mtime_nsec: m.mtime_nsec(),
        ctime: m.ctime(),
        ctime_nsec: m.ctime_nsec(),
    })
}
struct DescriptorReader<'a> {
    file: &'a File,
    offset: u64,
}
impl Read for DescriptorReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file.read_at(buffer, self.offset)?;
        self.offset = self
            .offset
            .checked_add(n as u64)
            .ok_or(std::io::Error::other("module offset overflow"))?;
        Ok(n)
    }
}
pub(super) fn hash_bounded(reader: &mut impl Read, cap: u64) -> Result<String> {
    let mut hash = Sha256State::new();
    let mut total = 0u64;
    let mut chunk = vec![0; CHUNK];
    loop {
        // Read one extra byte at the bound; never return a prefix hash.
        let take = usize::try_from((cap - total).saturating_add(1).min(CHUNK as u64))
            .map_err(|_| ProvenanceError("native module read size"))?;
        let n = match reader.read(&mut chunk[..take]) {
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            result => result?,
        };
        if n == 0 {
            break;
        }
        total = total
            .checked_add(n as u64)
            .ok_or(ProvenanceError("native module length overflow"))?;
        require(total <= cap, "native module grew beyond byte cap")?;
        hash.update(&chunk[..n]);
    }
    Ok(hash.finish())
}
pub(super) struct RetainedModule {
    path: PathBuf,
    file: File,
    identity: Identity,
    hash: String,
}
impl RetainedModule {
    pub(super) fn open(path: &Path) -> Result<Self> {
        require(
            path.is_absolute() && path.canonicalize()? == path,
            "canonical native module path required",
        )?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(path)?;
        let before = identity(&file.metadata()?)?; // Cap BEFORE any hashing.
        let mut reader = DescriptorReader {
            file: &file,
            offset: 0,
        };
        let hash = hash_bounded(&mut reader, MODULE_CAP)?;
        require(
            reader.offset == before.size,
            "native module incomplete read",
        )?;
        require(
            before == identity(&file.metadata()?)?,
            "native module changed during hash",
        )?;
        require(
            before == identity(&std::fs::symlink_metadata(path)?)?,
            "native module path replaced",
        )?;
        Ok(Self {
            path: path.to_owned(),
            file,
            identity: before,
            hash,
        })
    }
    pub(super) fn recheck(&self) -> Result<()> {
        require(
            self.path.canonicalize()? == self.path,
            "native module path redirected",
        )?;
        require(
            self.identity == identity(&self.file.metadata()?)?,
            "held native module changed",
        )?;
        require(
            self.identity == identity(&std::fs::symlink_metadata(&self.path)?)?,
            "native module path replaced",
        )?;
        let mut reader = DescriptorReader {
            file: &self.file,
            offset: 0,
        };
        let hash = hash_bounded(&mut reader, MODULE_CAP)?;
        require(
            reader.offset == self.identity.size,
            "native module incomplete read",
        )?;
        require(
            hash == self.hash && self.identity == identity(&self.file.metadata()?)?,
            "held native module bytes changed",
        )?;
        require(
            self.identity == identity(&std::fs::symlink_metadata(&self.path)?)?,
            "native module path changed during hash",
        )
    }
}
struct BoundExport {
    kind: CudaLibraryKind,
    name: &'static str,
    address: usize,
    owner: ModuleOwner,
}
pub(super) struct CudaBinding {
    observation: CudaRuntimeObservation,
    driver: RetainedModule,
    compiler: RetainedModule,
    exports: Vec<BoundExport>,
    // Exact owned source identities; no alternate runtime-policy encoding.
    kernel_hashes: [String; 4],
}
// Borrowed observations only: no exports, addresses, owners or file handles.
pub(crate) struct CudaRuntimeFacts<'a> {
    binding: &'a CudaBinding,
}
impl CudaRuntimeFacts<'_> {
    pub(crate) fn driver_path(&self) -> &[u8] {
        self.binding.driver.path.as_os_str().as_bytes()
    }
    pub(crate) fn driver_sha256(&self) -> &str {
        &self.binding.driver.hash
    }
    pub(crate) fn nvrtc_path(&self) -> &[u8] {
        self.binding.compiler.path.as_os_str().as_bytes()
    }
    pub(crate) fn nvrtc_sha256(&self) -> &str {
        &self.binding.compiler.hash
    }
    pub(crate) fn driver_version(&self) -> u32 {
        self.binding.observation.driver_version()
    }
    pub(crate) fn nvrtc_version(&self) -> (u32, u32) {
        self.binding.observation.nvrtc_version()
    }
    pub(crate) fn physical_identity(&self) -> &ckks_train_cuda::CudaPhysicalIdentity {
        self.binding.observation.physical_identity()
    }
    pub(crate) fn kernel_hashes(&self) -> &[String; 4] {
        &self.binding.kernel_hashes
    }
    pub(crate) fn intended_options(&self) -> &[String; 2] {
        self.binding.observation.intended_compile_options()
    }
}
impl CudaBinding {
    pub(super) fn facts(&self) -> Result<CudaRuntimeFacts<'_>> {
        self.recheck()?;
        let facts = CudaRuntimeFacts { binding: self };
        self.recheck()?;
        Ok(facts)
    }

    pub(super) fn observe(ordinal: usize) -> Result<Self> {
        check_runtime_environment()?; // MUST precede CUDA singleton/library discovery.
        require(
            cfg!(target_os = "linux"),
            "CUDA study observation requires native Linux",
        )?;
        let observation = CudaRuntimeObservation::observe(ordinal)
            .map_err(|_| ProvenanceError("CUDA runtime observation unavailable"))?;
        let mut exports = Vec::new();
        for export in observation.exported_functions() {
            let owner = resolve_owner(export.address())?;
            exports.push(BoundExport {
                kind: export.library_kind(),
                name: export.symbol_name(),
                address: export.address() as usize,
                owner,
            });
        }
        let module = |kind| -> Result<RetainedModule> {
            let mut selected = exports.iter().filter(|export| export.kind == kind);
            let first = selected
                .next()
                .ok_or(ProvenanceError("CUDA module exports missing"))?;
            for export in selected {
                require_same_owner(&first.owner, &export.owner)?;
            }
            RetainedModule::open(&first.owner.path)
        };
        let driver = module(CudaLibraryKind::Driver)?;
        let compiler = module(CudaLibraryKind::Nvrtc)?;
        require(
            driver.path != compiler.path,
            "driver/compiler module owners not distinct",
        )?;
        let kernel_hashes = [
            ckks_train_cuda::NTT_KERNEL_SRC,
            ckks_train_cuda::POINTWISE_KERNEL_SRC,
            ckks_train_cuda::BASIS_KERNEL_SRC,
            ckks_train_cuda::KEYSWITCH_KERNEL_SRC,
        ]
        .map(|source| sha256_hex(source.as_bytes()));
        let binding = Self {
            observation,
            driver,
            compiler,
            exports,
            kernel_hashes,
        };
        binding.recheck()?;
        Ok(binding)
    }
    pub(super) fn recheck(&self) -> Result<()> {
        check_runtime_environment()?;
        self.observation
            .recheck()
            .map_err(|_| ProvenanceError("CUDA runtime observation changed"))?;
        self.recheck_exports()?;
        self.driver.recheck()?;
        self.compiler.recheck()?;
        self.recheck_exports()?;
        self.observation
            .recheck()
            .map_err(|_| ProvenanceError("CUDA runtime changed during module recheck"))
    }
    fn recheck_exports(&self) -> Result<()> {
        let actual = self.observation.exported_functions();
        require(
            actual.len() == self.exports.len(),
            "CUDA export inventory changed",
        )?;
        for (now, held) in actual.iter().zip(&self.exports) {
            require(
                now.library_kind() == held.kind
                    && now.symbol_name() == held.name
                    && now.address() as usize == held.address,
                "CUDA export changed",
            )?;
            require_same_owner(&held.owner, &resolve_owner(now.address())?)?;
        }
        Ok(())
    }
}
