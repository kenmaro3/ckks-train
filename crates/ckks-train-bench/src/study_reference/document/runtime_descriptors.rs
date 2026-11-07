//! Private D7 inspection codecs. None of these values is a checked route or gate.
#![allow(dead_code)] // Runtime policy awaits the genuine checked implementation-route adapter.
use super::*;
use crate::study_profile::{CaseBackend, CaseMode, CasePlacement};
use crate::study_provenance::{
    Host as CapturedHost, Observation as CapturedObservation, role::Recipe,
};
use crate::study_runtime::RuntimeDescriptorFacts;
const PHYSICAL: &str = "hekit-study-physical-binding-v1";
const POLICY: &str = "hekit-study-runtime-policy-v1";
const RECIPES: [&str; 3] = [
    "study-train-cpu-serial-linux-clean-v1",
    "study-train-cpu-rayon-linux-clean-v1",
    "study-train-cuda-host-linux-clean-v1",
];
fn valid(ok: bool, p: &str) -> Result<()> {
    ensure(ok, DocumentErrorCode::CrossLinkMismatch, p)
}
fn text_valid<const N: usize>(t: &Text<N>) -> Result<()> {
    ensure(
        t.0.len() <= N,
        DocumentErrorCode::TooLarge,
        "descriptor.text",
    )
}
fn hash_valid(h: &Hash) -> Result<()> {
    ensure(
        h.0.len() == 64 && lower_hex(&h.0),
        DocumentErrorCode::BadHex,
        "descriptor.hash",
    )
}
fn positive_i32(x: u64) -> bool {
    x > 0 && x <= i32::MAX as u64
}
fn header(domain: &str) -> Result<Encoder> {
    let mut e = Encoder::new(DocumentLayout::Reference);
    Text::<4096>(ENCODING_VERSION.into()).encode(&mut e)?;
    Text::<4096>(domain.into()).encode(&mut e)?;
    Ok(e)
}
/// Immutable bytes/hash inspection only; no gate/start/Ready conversion.
pub(crate) struct DescriptorBytes {
    bytes: Vec<u8>,
    hash: String,
}
impl DescriptorBytes {
    fn new(bytes: Vec<u8>) -> Result<Self> {
        ensure(
            bytes.len() <= A_MAX,
            DocumentErrorCode::TooLarge,
            "descriptor",
        )?;
        Ok(Self {
            hash: sha256_hex(&bytes),
            bytes,
        })
    }
    pub(crate) fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub(crate) fn sha256(&self) -> &str {
        &self.hash
    }
}
#[derive(Clone)]
pub(super) struct DeviceIdentity {
    pub(super) ordinal: u64,
    pub(super) uuid: [u8; 16],
    pub(super) pci_bus_id: Text<4096>,
    pub(super) compute_major: u64,
    pub(super) compute_minor: u64,
}
impl DeviceIdentity {
    fn validate(&self) -> Result<()> {
        let pci = self.pci_bus_id.0.as_bytes();
        valid(
            self.ordinal <= i32::MAX as u64
                && self.uuid.iter().any(|b| *b != 0)
                && (1..=31).contains(&pci.len())
                && pci.iter().all(|b| b.is_ascii() && !b.is_ascii_control())
                && positive_i32(self.compute_major)
                && self.compute_minor <= i32::MAX as u64,
            "physical.cuda",
        )
    }
    #[cfg(feature = "cuda")]
    fn from_facts(f: &crate::study_runtime::CudaRuntimeFacts<'_>) -> Self {
        let p = f.physical_identity();
        let (major, minor) = p.compute_capability();
        Self {
            ordinal: p.ordinal() as u64,
            uuid: *p.uuid_bytes(),
            pci_bus_id: Text(p.pci_bus_id().into()),
            compute_major: major.into(),
            compute_minor: minor.into(),
        }
    }
}
impl Encode for DeviceIdentity {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        self.ordinal.encode(e)?;
        e.u64(16)?;
        e.append(&self.uuid, false)?;
        self.pci_bus_id.encode(e)?;
        self.compute_major.encode(e)?;
        self.compute_minor.encode(e)
    }
}
#[derive(Clone)]
pub(super) struct PhysicalBody {
    pub(super) cpu_model: Observation<Text<4096>>,
    pub(super) os: Observation<Text<4096>>,
    pub(super) hostname: Observation<Text<4096>>,
    pub(super) cuda_device: Option<DeviceIdentity>,
}
fn captured_text(value: &CapturedObservation<String>) -> Result<Observation<Text<4096>>> {
    match value {
        CapturedObservation::Available(s) => Ok(Observation::Available(Text(s.clone()))),
        _ => Err(err(
            DocumentErrorCode::CrossLinkMismatch,
            "physical.host.unavailable",
        )),
    }
}
impl PhysicalBody {
    pub(super) fn host(host: &CapturedHost, cuda_device: Option<DeviceIdentity>) -> Result<Self> {
        Ok(Self {
            cpu_model: captured_text(host.cpu_model())?,
            os: captured_text(host.os())?,
            hostname: captured_text(host.hostname())?,
            cuda_device,
        })
    }
    fn from_facts(facts: &RuntimeDescriptorFacts<'_>) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let cuda = facts.cuda().map(DeviceIdentity::from_facts);
        #[cfg(not(feature = "cuda"))]
        let cuda = None;
        valid(
            cuda.is_some() == (facts.recipe() == Recipe::TrainCudaHost),
            "physical.recipe",
        )?;
        Self::host(facts.host(), cuda)
    }
    fn validate(&self) -> Result<()> {
        for observation in [&self.cpu_model, &self.os, &self.hostname] {
            let t = observation
                .available()
                .ok_or_else(|| err(DocumentErrorCode::CrossLinkMismatch, "physical.host"))?;
            text_valid(t)?;
            valid(!t.0.is_empty(), "physical.host")?;
        }
        if let Some(device) = &self.cuda_device {
            device.validate()?;
        }
        Ok(())
    }
    pub(super) fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut e = header(PHYSICAL)?;
        self.cpu_model.encode(&mut e)?;
        self.os.encode(&mut e)?;
        self.hostname.encode(&mut e)?;
        self.cuda_device.encode(&mut e)?;
        Ok(e.bytes)
    }
}
pub(crate) struct PhysicalDescriptor {
    pub(super) body: PhysicalBody,
    value: DescriptorBytes,
}
impl PhysicalDescriptor {
    pub(super) fn from_body(body: PhysicalBody) -> Result<Self> {
        let value = DescriptorBytes::new(body.encode()?)?;
        Ok(Self { body, value })
    }
    pub(crate) fn canonical_bytes(&self) -> &[u8] {
        self.value.canonical_bytes()
    }
    pub(crate) fn sha256(&self) -> &str {
        self.value.sha256()
    }
    pub(super) fn parse(bytes: &[u8]) -> Result<Self> {
        let mut r = Cursor::new(bytes, PHYSICAL)?;
        let body = PhysicalBody {
            cpu_model: r.observation()?,
            os: r.observation()?,
            hostname: r.observation()?,
            cuda_device: r.option(|r| r.device())?,
        };
        r.end()?;
        let parsed = Self::from_body(body)?;
        valid(parsed.canonical_bytes() == bytes, "physical.canonical")?;
        Ok(parsed)
    }
}
impl crate::study_provenance::RecheckedTrainCapture<'_> {
    pub(crate) fn physical_descriptor(&self) -> Result<PhysicalDescriptor> {
        let facts = self
            .runtime_descriptor_facts()
            .map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "physical.retained"))?;
        valid(
            std::ptr::eq(facts.host(), self.host()),
            "physical.same_host",
        )?;
        let descriptor = PhysicalDescriptor::from_body(PhysicalBody::from_facts(&facts)?)?;
        let after = self
            .runtime_descriptor_facts()
            .map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "physical.recheck"))?;
        valid(
            std::ptr::eq(facts.host(), after.host())
                && descriptor.canonical_bytes() == PhysicalBody::from_facts(&after)?.encode()?,
            "physical.retained_changed",
        )?;
        Ok(descriptor)
    }
}
#[derive(Clone)]
pub(super) struct Kernel {
    pub(super) kind: u8,
    pub(super) source_sha256: Hash,
}
impl Encode for Kernel {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.byte(self.kind)?;
        self.source_sha256.encode(e)
    }
}
#[derive(Clone)]
pub(super) struct CudaRuntime {
    pub(super) driver_path: RawOs,
    pub(super) driver_sha256: Hash,
    pub(super) driver_version: u64,
    pub(super) nvrtc_path: RawOs,
    pub(super) nvrtc_sha256: Hash,
    pub(super) nvrtc_major: u64,
    pub(super) nvrtc_minor: u64,
    pub(super) kernels: List<Kernel, 4>,
    pub(super) intended_options: List<Text<4096>, 2>,
}
fn path_valid(path: &RawOs) -> Result<()> {
    // A canonical observed Unix spelling has no NUL, empty, dot or parent component.
    // No filesystem access or UTF8 conversion is used to validate inspection bytes.
    let b = &path.bytes;
    valid(
        matches!(path.encoding, OsEncoding::Unix)
            && b.len() <= 4096
            && b.first() == Some(&b'/')
            && b.len() > 1
            && !b.contains(&0)
            && b[1..]
                .split(|b| *b == b'/')
                .all(|part| !part.is_empty() && part != b"." && part != b".."),
        "runtime.module_path",
    )
}
impl CudaRuntime {
    fn validate(&self, device: &DeviceIdentity) -> Result<()> {
        path_valid(&self.driver_path)?;
        path_valid(&self.nvrtc_path)?;
        hash_valid(&self.driver_sha256)?;
        hash_valid(&self.nvrtc_sha256)?;
        valid(
            self.driver_path.bytes != self.nvrtc_path.bytes
                && positive_i32(self.driver_version)
                && positive_i32(self.nvrtc_major)
                && self.nvrtc_minor <= i32::MAX as u64,
            "runtime.modules",
        )?;
        valid(
            self.kernels.0.len() == 4
                && self
                    .kernels
                    .0
                    .iter()
                    .enumerate()
                    .all(|(i, k)| k.kind == i as u8),
            "runtime.kernels",
        )?;
        for kernel in &self.kernels.0 {
            hash_valid(&kernel.source_sha256)?;
        }
        valid(
            self.intended_options.0.len() == 2
                && self.intended_options.0[0].0 == "--fmad=false"
                && self.intended_options.0[1].0
                    == format!(
                        "--gpu-architecture=compute_{}{}",
                        device.compute_major, device.compute_minor
                    ),
            "runtime.options",
        )
    }
    // Called by retained-fact validation; fixture tests supply only synthetic snapshots.
    pub(super) fn matches_observed(&self, expected: &Self) -> Result<()> {
        let mut a = Encoder::new(DocumentLayout::Reference);
        self.encode(&mut a)?;
        let mut b = Encoder::new(DocumentLayout::Reference);
        expected.encode(&mut b)?;
        valid(a.bytes == b.bytes, "runtime.cuda_retained")
    }
    #[cfg(feature = "cuda")]
    fn from_facts(f: &crate::study_runtime::CudaRuntimeFacts<'_>) -> Self {
        let raw = |bytes: &[u8]| RawOs {
            encoding: OsEncoding::Unix,
            bytes: bytes.to_vec(),
        };
        let (major, minor) = f.nvrtc_version();
        Self {
            driver_path: raw(f.driver_path()),
            driver_sha256: Hash(f.driver_sha256().into()),
            driver_version: f.driver_version().into(),
            nvrtc_path: raw(f.nvrtc_path()),
            nvrtc_sha256: Hash(f.nvrtc_sha256().into()),
            nvrtc_major: major.into(),
            nvrtc_minor: minor.into(),
            kernels: List(
                f.kernel_hashes()
                    .iter()
                    .enumerate()
                    .map(|(i, h)| Kernel {
                        kind: i as u8,
                        source_sha256: Hash(h.clone()),
                    })
                    .collect(),
            ),
            intended_options: List(
                f.intended_options()
                    .iter()
                    .map(|s| Text(s.clone()))
                    .collect(),
            ),
        }
    }
}
impl Encode for CudaRuntime {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        self.driver_path.encode(e)?;
        self.driver_sha256.encode(e)?;
        self.driver_version.encode(e)?;
        self.nvrtc_path.encode(e)?;
        self.nvrtc_sha256.encode(e)?;
        self.nvrtc_major.encode(e)?;
        self.nvrtc_minor.encode(e)?;
        self.kernels.encode(e)?;
        self.intended_options.encode(e)
    }
}
// Selection claims for inspection validation, explicitly NOT a checked route.
#[derive(Clone, Copy)]
pub(super) struct PolicySelection {
    pub(super) backend: CaseBackend,
    pub(super) mode: CaseMode,
    pub(super) placement: CasePlacement,
    pub(super) bootstrap_engine: u8,
}
#[derive(Clone)]
pub(super) struct RuntimePolicy {
    pub(super) form: u8,
    pub(super) recipe: Text<4096>,
    pub(super) backend: u8,
    pub(super) execution_threads: u64,
    pub(super) matvec_order: u8,
    pub(super) stream_count: Option<u64>,
    pub(super) refresh_route: u8,
    pub(super) physical_binding_sha256: Hash,
    pub(super) cuda_runtime: Option<CudaRuntime>,
}
impl RuntimePolicy {
    fn validate(&self, p: &PhysicalDescriptor, s: PolicySelection) -> Result<()> {
        hash_valid(&self.physical_binding_sha256)?;
        let cuda = self.form == 2;
        valid(
            self.form <= 2
                && self.recipe.0 == RECIPES[self.form as usize]
                && self.backend == u8::from(cuda)
                && s.backend
                    == if cuda {
                        CaseBackend::Cuda
                    } else {
                        CaseBackend::Cpu
                    }
                && self.execution_threads > 0
                && (self.form != 0 || self.execution_threads == 1)
                && self.matvec_order == 0
                && self.stream_count == if cuda { Some(1) } else { None }
                && p.body.cuda_device.is_some() == cuda
                && self.cuda_runtime.is_some() == cuda
                && self.physical_binding_sha256.0 == p.sha256(),
            "runtime.alignment",
        )?;
        let ni = s.mode == CaseMode::Noninteractive;
        valid(
            s.bootstrap_engine == u8::from(ni)
                && self.refresh_route == u8::from(cuda) * 2 + u8::from(ni)
                && if ni {
                    matches!(
                        s.placement,
                        CasePlacement::WhenNeeded
                            | CasePlacement::AfterUpdate
                            | CasePlacement::AfterActivation
                    )
                } else {
                    s.placement == CasePlacement::ClientRefresh
                },
            "runtime.route_claims",
        )?;
        if let (Some(c), Some(d)) = (&self.cuda_runtime, &p.body.cuda_device) {
            c.validate(d)?;
        }
        Ok(())
    }
    fn bytes(&self) -> Result<Vec<u8>> {
        let mut e = header(POLICY)?;
        e.byte(self.form)?;
        self.recipe.encode(&mut e)?;
        e.byte(self.backend)?;
        self.execution_threads.encode(&mut e)?;
        e.byte(self.matvec_order)?;
        self.stream_count.encode(&mut e)?;
        e.byte(self.refresh_route)?;
        self.physical_binding_sha256.encode(&mut e)?;
        self.cuda_runtime.encode(&mut e)?;
        Ok(e.bytes)
    }
    pub(super) fn encode(
        &self,
        p: &PhysicalDescriptor,
        s: PolicySelection,
    ) -> Result<DescriptorBytes> {
        self.validate(p, s)?;
        DescriptorBytes::new(self.bytes()?)
    }
    fn validate_retained(
        &self,
        p: &PhysicalDescriptor,
        s: PolicySelection,
        f: &RuntimeDescriptorFacts<'_>,
    ) -> Result<()> {
        self.validate(p, s)?;
        let form = match f.recipe() {
            Recipe::TrainCpuSerial => 0,
            Recipe::TrainCpuRayon => 1,
            Recipe::TrainCudaHost => 2,
            Recipe::Reference => {
                return Err(err(DocumentErrorCode::CrossLinkMismatch, "runtime.recipe"));
            }
        };
        valid(
            self.form == form
                && self.execution_threads == f.execution_threads()
                && p.canonical_bytes() == PhysicalBody::from_facts(f)?.encode()?,
            "runtime.retained",
        )?;
        #[cfg(feature = "cuda")]
        if let Some(f) = f.cuda() {
            let expected = CudaRuntime::from_facts(f);
            self.cuda_runtime
                .as_ref()
                .ok_or_else(|| err(DocumentErrorCode::CrossLinkMismatch, "runtime.cuda"))?
                .matches_observed(&expected)?;
        }
        Ok(())
    }
    pub(super) fn parse(bytes: &[u8], p: &PhysicalDescriptor, s: PolicySelection) -> Result<Self> {
        let mut r = Cursor::new(bytes, POLICY)?;
        let policy = Self {
            form: r.byte()?,
            recipe: r.text()?,
            backend: r.byte()?,
            execution_threads: r.u64()?,
            matvec_order: r.byte()?,
            stream_count: r.option(|r| r.u64())?,
            refresh_route: r.byte()?,
            physical_binding_sha256: r.hash()?,
            cuda_runtime: r.option(|r| r.cuda())?,
        };
        r.end()?;
        valid(
            policy.encode(p, s)?.canonical_bytes() == bytes,
            "runtime.canonical",
        )?;
        Ok(policy)
    }
}
// A narrow bounded decoder for these two fixed bodies. Nested values are the
// existing document primitives, and all output goes through their Encode impls.
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], domain: &str) -> Result<Self> {
        ensure(
            bytes.len() <= A_MAX,
            DocumentErrorCode::TooLarge,
            "descriptor",
        )?;
        let mut r = Self { bytes, offset: 0 };
        valid(
            r.text::<4096>()?.0 == ENCODING_VERSION && r.text::<4096>()?.0 == domain,
            "descriptor.domain",
        )?;
        Ok(r)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(n)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| err(DocumentErrorCode::OutOfRange, "descriptor.truncated"))?;
        let b = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(b)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        b.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(b))
    }
    fn blob(&mut self, cap: usize) -> Result<Vec<u8>> {
        let n = usize::try_from(self.u64()?)
            .map_err(|_| err(DocumentErrorCode::TooLarge, "descriptor.length"))?;
        ensure(n <= cap, DocumentErrorCode::TooLarge, "descriptor.length")?;
        Ok(self.take(n)?.to_vec())
    }
    fn text<const N: usize>(&mut self) -> Result<Text<N>> {
        let b = self.blob(N)?;
        let s = String::from_utf8(b)
            .map_err(|_| err(DocumentErrorCode::OutOfRange, "descriptor.utf8"))?;
        Ok(Text(s))
    }
    fn hash(&mut self) -> Result<Hash> {
        let h = Hash(self.text::<64>()?.0);
        hash_valid(&h)?;
        Ok(h)
    }
    fn option<T>(&mut self, read: impl FnOnce(&mut Self) -> Result<T>) -> Result<Option<T>> {
        match self.byte()? {
            0 => Ok(None),
            1 => read(self).map(Some),
            _ => Err(err(DocumentErrorCode::UnsupportedTag, "descriptor.option")),
        }
    }
    fn observation(&mut self) -> Result<Observation<Text<4096>>> {
        match self.byte()? {
            0 => Ok(Observation::Available(self.text()?)),
            1 => Ok(Observation::Unavailable(self.text()?)),
            _ => Err(err(
                DocumentErrorCode::UnsupportedTag,
                "descriptor.observation",
            )),
        }
    }
    fn raw(&mut self) -> Result<RawOs> {
        ensure(
            self.byte()? == 0,
            DocumentErrorCode::UnsupportedTag,
            "descriptor.raw_os",
        )?;
        Ok(RawOs {
            encoding: OsEncoding::Unix,
            bytes: self.blob(4096)?,
        })
    }
    fn device(&mut self) -> Result<DeviceIdentity> {
        let ordinal = self.u64()?;
        let b = self.blob(16)?;
        valid(b.len() == 16, "physical.uuid")?;
        let mut uuid = [0; 16];
        uuid.copy_from_slice(&b);
        Ok(DeviceIdentity {
            ordinal,
            uuid,
            pci_bus_id: self.text()?,
            compute_major: self.u64()?,
            compute_minor: self.u64()?,
        })
    }
    fn cuda(&mut self) -> Result<CudaRuntime> {
        let driver_path = self.raw()?;
        let driver_sha256 = self.hash()?;
        let driver_version = self.u64()?;
        let nvrtc_path = self.raw()?;
        let nvrtc_sha256 = self.hash()?;
        let nvrtc_major = self.u64()?;
        let nvrtc_minor = self.u64()?;
        valid(self.u64()? == 4, "runtime.kernel_count")?;
        let mut kernels = Vec::with_capacity(4);
        for _ in 0..4 {
            kernels.push(Kernel {
                kind: self.byte()?,
                source_sha256: self.hash()?,
            });
        }
        valid(self.u64()? == 2, "runtime.option_count")?;
        let intended_options = List(vec![self.text()?, self.text()?]);
        Ok(CudaRuntime {
            driver_path,
            driver_sha256,
            driver_version,
            nvrtc_path,
            nvrtc_sha256,
            nvrtc_major,
            nvrtc_minor,
            kernels: List(kernels),
            intended_options,
        })
    }
    fn end(&self) -> Result<()> {
        valid(self.offset == self.bytes.len(), "descriptor.trailing")
    }
}
