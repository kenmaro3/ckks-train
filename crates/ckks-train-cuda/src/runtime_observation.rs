//! Context-free CUDA runtime inspection; not study admission or GPU execution.

use std::ffi::{c_char, c_int, c_void};
use std::marker::PhantomData;
use std::rc::Rc;

use ckks_train_core::{HekitError, Result};
use cudarc::driver::sys;
use cudarc::nvrtc::{CompileOptions, sys as nvrtc};

use crate::device::guarded;

/// One of the two retained top-level native libraries (not a transitive closure).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaLibraryKind {
    /// CUDA driver library.
    Driver,
    /// CUDA runtime compiler library.
    Nvrtc,
}

/// An immutable native export inspected from a retained cudarc singleton.
///
/// A returned address is for owner inspection, not a study capability. Consumers
/// must obtain entries from the opaque observation, not accept caller pointer lists.
///
/// ```compile_fail
/// use ckks_train_cuda::{CudaExportedFunction, CudaLibraryKind};
/// let forged = CudaExportedFunction {
///     library: CudaLibraryKind::Driver, name: "cuInit", address: std::ptr::null(),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaExportedFunction {
    library: CudaLibraryKind,
    name: &'static str,
    address: *const c_void,
}

impl CudaExportedFunction {
    /// The singleton from which this export was resolved.
    pub fn library_kind(&self) -> CudaLibraryKind {
        self.library
    }
    /// Exact native export name, not a Rust wrapper name.
    pub fn symbol_name(&self) -> &'static str {
        self.name
    }
    /// Actual native function address for platform owner inspection; do not call it.
    pub fn address(&self) -> *const c_void {
        self.address
    }
}

/// Immutable identity queried from an actual CUDA device handle.
#[derive(Debug, PartialEq, Eq)]
pub struct CudaPhysicalIdentity {
    ordinal: usize,
    uuid: [u8; 16],
    pci: String,
    capability: (u32, u32),
}

impl CudaPhysicalIdentity {
    /// Selected CUDA ordinal; zero is valid.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
    /// Exact UUID_v2 bytes, with no text conversion.
    pub fn uuid_bytes(&self) -> &[u8; 16] {
        &self.uuid
    }
    /// Driver's PCI bus identifier spelling.
    pub fn pci_bus_id(&self) -> &str {
        &self.pci
    }
    /// Compute capability (major, minor); minor zero is valid.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.capability
    }
}

/// Retained, context-free CUDA observation. This is not admission or JIT evidence.
///
/// The caller must reject study loader/environment overrides **before** observing.
/// Bench owns module-owner resolution and retained descriptor hashing. Native loading
/// and initialization may allocate infrastructure; no CUDA context, HE object, JIT
/// program or capacity allocation is created here. A normal panic hook may print a
/// dependency diagnostic even though returned errors omit loader-repair advice.
///
/// The inspection API borrows real observations; this example does not load CUDA:
/// ```
/// use ckks_train_cuda::{CudaDevice, CudaRuntimeObservation};
/// fn inspect(observation: &CudaRuntimeObservation, created: &CudaDevice) {
///     let _ = (observation.driver_version(), observation.nvrtc_version(), observation.device_count());
///     let physical = observation.physical_identity();
///     let _ = (physical.ordinal(), physical.uuid_bytes(), physical.pci_bus_id(), physical.compute_capability());
///     for export in observation.exported_functions() {
///         let _ = (export.library_kind(), export.symbol_name(), export.address());
///     }
///     let _ = observation.intended_compile_options();
///     let _ = (created.physical_identity(), created.intended_compile_options());
///     let _ = observation.recheck();
/// }
/// let _observer = CudaRuntimeObservation::observe;
/// ```
///
/// No evidence or arbitrary handle constructor is available:
/// ```compile_fail
/// use ckks_train_cuda::CudaRuntimeObservation;
/// let forged = CudaRuntimeObservation::default();
/// ```
/// The retained observation stays in its owner frame:
/// ```compile_fail
/// use ckks_train_cuda::CudaRuntimeObservation;
/// fn needs_send<T: Send>() {}
/// needs_send::<CudaRuntimeObservation>();
/// ```
/// ```compile_fail
/// use ckks_train_cuda::CudaRuntimeObservation;
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<CudaRuntimeObservation>();
/// ```
pub struct CudaRuntimeObservation {
    driver: &'static sys::Lib,
    compiler: &'static nvrtc::Lib,
    identity: RuntimeIdentity,
    facts: Facts,
    options: [String; 2],
    _owner: PhantomData<Rc<()>>,
}

impl CudaRuntimeObservation {
    /// Observe the selected device using actual retained driver/compiler singletons.
    /// No caller-provided paths, symbols, handles or synthetic facts are accepted.
    pub fn observe(ordinal: usize) -> Result<Self> {
        checked_ordinal(ordinal)?;
        let (driver, compiler) = load_libraries()?;
        let identity = runtime_identity(driver, compiler)?;
        let facts = study_guarded("device queries", || {
            query_facts(&NativeQueries { driver, compiler }, ordinal)
        })?;
        let (major, minor) = facts.physical.capability;
        // Both values originated as nonnegative c_int, so conversion is exact.
        let options = compile_option_projection(
            i32::try_from(major).map_err(|_| unavailable("compute major"))?,
            i32::try_from(minor).map_err(|_| unavailable("compute minor"))?,
        );
        Ok(Self {
            driver,
            compiler,
            identity,
            facts,
            options,
            _owner: PhantomData,
        })
    }

    /// Refuse changed singleton, export, version, count or device facts.
    /// Does not refresh the original snapshot or inspect module files for the caller.
    pub fn recheck(&self) -> Result<()> {
        let (driver, compiler) = load_libraries()?;
        require_same_runtime(&self.identity, &runtime_identity(driver, compiler)?)?;
        // Keep querying through the retained owners after singleton identity checks.
        let current = study_guarded("device recheck", || {
            query_facts(
                &NativeQueries {
                    driver: self.driver,
                    compiler: self.compiler,
                },
                self.facts.physical.ordinal,
            )
        })?;
        require_same_facts(&self.facts, &current)
    }
    /// Driver's actual positive encoded version integer.
    pub fn driver_version(&self) -> u32 {
        self.facts.driver_version
    }
    /// Actual NVRTC (major, minor), allowing minor zero.
    pub fn nvrtc_version(&self) -> (u32, u32) {
        self.facts.nvrtc_version
    }
    /// Actual device count at observation.
    pub fn device_count(&self) -> usize {
        self.facts.device_count
    }
    /// Physical identity of the actual selected device.
    pub fn physical_identity(&self) -> &CudaPhysicalIdentity {
        &self.facts.physical
    }
    /// Fixed ordered native export inventory, borrowed from this real observation.
    pub fn exported_functions(&self) -> &[CudaExportedFunction] {
        &self.identity.exports
    }
    /// Intended NVRTC flags, in pinned cudarc rendering order; not compilation evidence.
    pub fn intended_compile_options(&self) -> &[String; 2] {
        &self.options
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Facts {
    driver_version: u32,
    nvrtc_version: (u32, u32),
    device_count: usize,
    device: sys::CUdevice,
    physical: CudaPhysicalIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeIdentity {
    // These are Rust singleton-object addresses, NOT loaded native module bases.
    driver: *const c_void,
    nvrtc: *const c_void,
    exports: Vec<CudaExportedFunction>,
}

fn unavailable(stage: &str) -> HekitError {
    HekitError::Device(format!("CUDA study observation unavailable: {stage}"))
}

fn study_guarded<T>(stage: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    guarded(stage, f).map_err(|_| unavailable(stage))
}

fn load_libraries() -> Result<(&'static sys::Lib, &'static nvrtc::Lib)> {
    study_guarded("driver/compiler libraries", || {
        // SAFETY: cudarc owns both libraries for process lifetime. No raw handle
        // escapes and no function is called while obtaining these references.
        Ok(unsafe { (sys::culib(), nvrtc::culib()) })
    })
}

fn runtime_identity(driver: &sys::Lib, compiler: &nvrtc::Lib) -> Result<RuntimeIdentity> {
    let mut exports = Vec::with_capacity(15);
    macro_rules! export {
        ($kind:ident, $lib:ident, $field:ident) => {
            exports.push(CudaExportedFunction {
                library: CudaLibraryKind::$kind,
                name: stringify!($field),
                address: $lib.$field as *const () as *const c_void,
            });
        };
    }
    export!(Driver, driver, cuInit);
    export!(Driver, driver, cuDriverGetVersion);
    export!(Driver, driver, cuDeviceGetCount);
    export!(Driver, driver, cuDeviceGet);
    export!(Driver, driver, cuDeviceGetUuid_v2);
    export!(Driver, driver, cuDeviceGetPCIBusId);
    export!(Driver, driver, cuDeviceGetAttribute);
    export!(Nvrtc, compiler, nvrtcVersion);
    export!(Nvrtc, compiler, nvrtcCreateProgram);
    export!(Nvrtc, compiler, nvrtcCompileProgram);
    export!(Nvrtc, compiler, nvrtcDestroyProgram);
    export!(Nvrtc, compiler, nvrtcGetPTXSize);
    export!(Nvrtc, compiler, nvrtcGetPTX);
    export!(Nvrtc, compiler, nvrtcGetProgramLogSize);
    export!(Nvrtc, compiler, nvrtcGetProgramLog);
    validate_exports(&exports)?;
    Ok(RuntimeIdentity {
        driver: (driver as *const sys::Lib).cast(),
        nvrtc: (compiler as *const nvrtc::Lib).cast(),
        exports,
    })
}

fn validate_exports(exports: &[CudaExportedFunction]) -> Result<()> {
    if exports.iter().any(|entry| entry.address.is_null()) {
        return Err(unavailable("null native export"));
    }
    Ok(())
}

fn require_same_runtime(before: &RuntimeIdentity, after: &RuntimeIdentity) -> Result<()> {
    if before != after {
        return Err(unavailable("changed runtime owner/export"));
    }
    Ok(())
}

fn require_same_facts(before: &Facts, after: &Facts) -> Result<()> {
    if before != after {
        return Err(unavailable("changed device/version facts"));
    }
    Ok(())
}

fn checked_ordinal(ordinal: usize) -> Result<c_int> {
    c_int::try_from(ordinal).map_err(|_| unavailable("ordinal exceeds c_int"))
}

// There is deliberately no context/JIT/memory operation in this private query seam.
trait DeviceQueries {
    fn init(&self) -> Result<()>;
    fn driver_version(&self) -> Result<i32>;
    fn nvrtc_version(&self) -> Result<(i32, i32)>;
    fn device_count(&self) -> Result<i32>;
    fn device(&self, ordinal: c_int) -> Result<sys::CUdevice>;
    fn uuid(&self, device: sys::CUdevice) -> Result<[c_char; 16]>;
    fn pci(&self, device: sys::CUdevice) -> Result<[c_char; 32]>;
    fn attribute(&self, device: sys::CUdevice, attribute: sys::CUdevice_attribute) -> Result<i32>;
}

struct NativeQueries<'a> {
    driver: &'a sys::Lib,
    compiler: &'a nvrtc::Lib,
}

impl DeviceQueries for NativeQueries<'_> {
    fn init(&self) -> Result<()> {
        // SAFETY: retained native function, specified zero flags, no out pointers.
        unsafe { (self.driver.cuInit)(0) }
            .result()
            .map_err(|_| unavailable("cuInit"))
    }
    fn driver_version(&self) -> Result<i32> {
        let mut version = 0;
        // SAFETY: valid initialized out pointer across this call.
        unsafe { (self.driver.cuDriverGetVersion)(&mut version) }
            .result()
            .map_err(|_| unavailable("cuDriverGetVersion"))?;
        Ok(version)
    }
    fn nvrtc_version(&self) -> Result<(i32, i32)> {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: valid distinct initialized out pointers across this call.
        unsafe { (self.compiler.nvrtcVersion)(&mut major, &mut minor) }
            .result()
            .map_err(|_| unavailable("nvrtcVersion"))?;
        Ok((major, minor))
    }
    fn device_count(&self) -> Result<i32> {
        let mut count = 0;
        // SAFETY: valid initialized out pointer after cuInit success.
        unsafe { (self.driver.cuDeviceGetCount)(&mut count) }
            .result()
            .map_err(|_| unavailable("cuDeviceGetCount"))?;
        Ok(count)
    }
    fn device(&self, ordinal: c_int) -> Result<sys::CUdevice> {
        let mut device = 0;
        // SAFETY: valid out pointer; caller checked ordinal and initialized driver.
        unsafe { (self.driver.cuDeviceGet)(&mut device, ordinal) }
            .result()
            .map_err(|_| unavailable("cuDeviceGet"))?;
        Ok(device)
    }
    fn uuid(&self, device: sys::CUdevice) -> Result<[c_char; 16]> {
        let mut uuid = sys::CUuuid { bytes: [0; 16] };
        // SAFETY: device came from driver or live context; initialized UUID out pointer.
        unsafe { (self.driver.cuDeviceGetUuid_v2)(&mut uuid, device) }
            .result()
            .map_err(|_| unavailable("cuDeviceGetUuid_v2"))?;
        Ok(uuid.bytes)
    }
    fn pci(&self, device: sys::CUdevice) -> Result<[c_char; 32]> {
        let mut pci = [0x7f; 32];
        let length = c_int::try_from(pci.len()).map_err(|_| unavailable("PCI buffer length"))?;
        // SAFETY: fixed buffer has exactly length writable bytes; live device handle.
        unsafe { (self.driver.cuDeviceGetPCIBusId)(pci.as_mut_ptr(), length, device) }
            .result()
            .map_err(|_| unavailable("cuDeviceGetPCIBusId"))?;
        Ok(pci)
    }
    fn attribute(&self, device: sys::CUdevice, attribute: sys::CUdevice_attribute) -> Result<i32> {
        let mut value = 0;
        // SAFETY: valid initialized out pointer and live device handle.
        unsafe { (self.driver.cuDeviceGetAttribute)(&mut value, attribute, device) }
            .result()
            .map_err(|_| unavailable("cuDeviceGetAttribute"))?;
        Ok(value)
    }
}

fn positive(value: i32, stage: &str) -> Result<u32> {
    if value <= 0 {
        return Err(unavailable(stage));
    }
    u32::try_from(value).map_err(|_| unavailable(stage))
}

fn version_pair(major: i32, minor: i32, stage: &str) -> Result<(u32, u32)> {
    Ok((
        positive(major, stage)?,
        u32::try_from(minor).map_err(|_| unavailable(stage))?,
    ))
}

fn query_facts(queries: &impl DeviceQueries, ordinal: usize) -> Result<Facts> {
    let native_ordinal = checked_ordinal(ordinal)?;
    queries.init()?;
    let driver_version = positive(queries.driver_version()?, "driver version")?;
    let count = queries.device_count()?;
    if count <= 0 {
        return Err(unavailable("device count"));
    }
    let device_count = usize::try_from(count).map_err(|_| unavailable("device count"))?;
    if ordinal >= device_count {
        return Err(unavailable("ordinal outside device count"));
    }
    let device = queries.device(native_ordinal)?;
    let (major, minor) = queries.nvrtc_version()?;
    let nvrtc_version = version_pair(major, minor, "NVRTC version")?;
    let physical = query_physical(queries, ordinal, device)?;
    Ok(Facts {
        driver_version,
        nvrtc_version,
        device_count,
        device,
        physical,
    })
}

fn query_physical(
    queries: &impl DeviceQueries,
    ordinal: usize,
    device: sys::CUdevice,
) -> Result<CudaPhysicalIdentity> {
    let uuid = queries.uuid(device)?.map(|byte| byte.to_ne_bytes()[0]);
    if uuid.iter().all(|byte| *byte == 0) {
        return Err(unavailable("zero device UUID"));
    }
    let raw_pci = queries.pci(device)?.map(|byte| byte.to_ne_bytes()[0]);
    let end = raw_pci
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(|| unavailable("unterminated PCI identity"))?;
    let bytes = &raw_pci[..end];
    if bytes.is_empty()
        || bytes
            .iter()
            .any(|byte| !byte.is_ascii() || byte.is_ascii_control())
    {
        return Err(unavailable("invalid PCI identity"));
    }
    let pci = std::str::from_utf8(bytes)
        .map_err(|_| unavailable("PCI identity encoding"))?
        .to_owned();
    let major = queries.attribute(
        device,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )?;
    let minor = queries.attribute(
        device,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )?;
    let capability = version_pair(major, minor, "compute capability")?;
    Ok(CudaPhysicalIdentity {
        ordinal,
        uuid,
        pci,
        capability,
    })
}

fn query_created_device(
    queries: &impl DeviceQueries,
    ordinal: usize,
    device: sys::CUdevice,
    compiled_capability: (i32, i32),
) -> Result<CudaPhysicalIdentity> {
    checked_ordinal(ordinal)?;
    let physical = query_physical(queries, ordinal, device)?;
    let expected = version_pair(
        compiled_capability.0,
        compiled_capability.1,
        "compiled capability",
    )?;
    if physical.capability != expected {
        return Err(unavailable("changed compiled capability"));
    }
    Ok(physical)
}

pub(crate) fn created_device_identity(
    ordinal: usize,
    device: sys::CUdevice,
    compiled_capability: (i32, i32),
) -> Result<CudaPhysicalIdentity> {
    let (driver, compiler) = load_libraries()?;
    study_guarded("created device identity", || {
        query_created_device(
            &NativeQueries { driver, compiler },
            ordinal,
            device,
            compiled_capability,
        )
    })
}

pub(crate) fn kernel_compile_options(major: i32, minor: i32) -> CompileOptions {
    CompileOptions {
        options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
        fmad: Some(false),
        ..Default::default()
    }
}

pub(crate) fn compile_option_projection(major: i32, minor: i32) -> [String; 2] {
    let options = kernel_compile_options(major, minor);
    // Pinned cudarc0.19's private build() emits fmad before its options vector.
    [
        format!("--fmad={}", options.fmad.expect("fixed compile policy")),
        options.options[0].clone(),
    ]
}

#[cfg(test)]
#[path = "runtime_observation_tests.rs"]
mod tests;
