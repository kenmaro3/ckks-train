//! Closed inert D7 resource sources. Parsed bytes prove neither live applicability
//! nor custody, physical bounds, a supported model/profile or permission to allocate.
#![allow(dead_code)] // No live resource owner/acceptance constructor exists.
use super::*;
use evidence::Mode;
mod bootstrap;
use bootstrap::Bootstrap;
mod links;
#[cfg(test)]
use links::Baseline;
#[cfg(test)]
mod tests;

const SOURCE_MAX: usize = 8_388_608;
#[derive(Debug, PartialEq, Eq)]
struct SourceError {
    reason: &'static str,
    field: std::borrow::Cow<'static, str>,
}
type SResult<T> = std::result::Result<T, SourceError>;
fn refusal(reason: &'static str, field: &str) -> SourceError {
    SourceError {
        reason,
        field: std::borrow::Cow::Owned(field.into()),
    }
}
fn require(ok: bool, reason: &'static str, field: &str) -> SResult<()> {
    if ok {
        Ok(())
    } else {
        Err(refusal(reason, field))
    }
}
// Reuse the strict document field/primitive traversal while preserving a resource
// refusal code and logical field for resource-specific primitive constraints.
fn bad(reason: &str, p: &str) -> DocumentError {
    err(
        DocumentErrorCode::CrossLinkMismatch,
        &format!("{reason}:{p}"),
    )
}
impl From<DocumentError> for SourceError {
    fn from(e: DocumentError) -> Self {
        if e.code == DocumentErrorCode::CrossLinkMismatch {
            if let Some((r, p)) = e.path.split_once(':') {
                let reason = match r {
                    "tag" => "tag",
                    "path" => "fileref-path",
                    "integer" => "integer",
                    "bytes-spelling" => "bytes-spelling",
                    "argv" => "argv",
                    "uuid" => "uuid",
                    "identifier" => "identifier",
                    _ => "primitive",
                };
                return refusal(reason, p);
            }
        }
        refusal(
            match e.code {
                DocumentErrorCode::DuplicateField => "duplicate-key",
                DocumentErrorCode::UnknownField | DocumentErrorCode::MissingField => "keys",
                DocumentErrorCode::OutOfRange => "integer",
                DocumentErrorCode::BadHex
                    if e.path == "work.registration"
                        || e.path == "work.request"
                        || e.path.ends_with(".argv.*") =>
                {
                    "bytes-spelling"
                }
                DocumentErrorCode::BadHex => "hex",
                DocumentErrorCode::TooLarge | DocumentErrorCode::TokenLimit => "raw-cap",
                _ => "json",
            },
            if e.path.is_empty() { "source" } else { &e.path },
        )
    }
}
fn check_source_length(n: usize) -> SResult<()> {
    require(n <= SOURCE_MAX, "raw-cap", "source.length")
}
fn checked_disk_sum(values: impl IntoIterator<Item = u64>) -> SResult<u64> {
    values.into_iter().try_fold(0u64, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| refusal("arithmetic-overflow", "disk.filesystems.required_bytes"))
    })
}
fn ordered<T: Ord>(v: &[T]) -> bool {
    v.windows(2).all(|p| p[0] < p[1])
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Tag<const N: u8>(u8);
impl<const N: u8> Decode for Tag<N> {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let n = u64::decode(r, p)?;
        if n > N as u64 {
            return Err(bad("tag", p));
        }
        Ok(Self(n as u8))
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Id(String);
impl Decode for Id {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let s = Text::<128>::decode(r, p)?.0;
        ensure(
            !s.is_empty()
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
            DocumentErrorCode::CrossLinkMismatch,
            &format!("identifier:{p}"),
        )?;
        Ok(Self(s))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Git(String);
impl Decode for Git {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let s = Text::<40>::decode(r, p)?.0;
        ensure(s.len() == 40 && lower_hex(&s), DocumentErrorCode::BadHex, p)?;
        Ok(Self(s))
    }
}
fn valid_uuid(s: &str) -> bool {
    s.len() == 36
        && s.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
            }
        })
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Uuid(String);
impl Decode for Uuid {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let s = Text::<36>::decode(r, p)?.0;
        if !valid_uuid(&s) {
            return Err(bad("uuid", p));
        }
        Ok(Self(s))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Int(i32);
impl Decode for Int {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let s = r.get();
        let v = s.strip_prefix('-').unwrap_or(s);
        if v.is_empty()
            || !v.bytes().all(|b| b.is_ascii_digit())
            || (v.starts_with('0') && v != "0")
            || s == "-0"
        {
            return Err(bad("integer", p));
        }
        s.parse().map(Self).map_err(|_| bad("integer", p))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Blob<const N: usize>(Vec<u8>);
impl<const N: usize> Decode for Blob<N> {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let s = r.get().as_bytes();
        if s.len() < 2 || s[0] != b'"' || s[s.len() - 1] != b'"' {
            return Err(bad("bytes-spelling", p));
        }
        let h = &s[1..s.len() - 1];
        if h.len() % 2 != 0
            || h.len() / 2 > N
            || !h
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
        {
            return Err(bad("bytes-spelling", p));
        }
        let mut out = Vec::new();
        out.try_reserve_exact(h.len() / 2)
            .map_err(|_| err(DocumentErrorCode::TooLarge, p))?;
        fn nib(b: u8) -> u8 {
            if b <= b'9' { b - b'0' } else { b - b'a' + 10 }
        }
        for pair in h.chunks_exact(2) {
            out.push(nib(pair[0]) * 16 + nib(pair[1]));
        }
        Ok(Self(out))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Items<T, const N: usize>(Vec<T>);
impl<T: Decode, const N: usize> Decode for Items<T, N> {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        List::<T, N>::decode(r, p).map(|v| Self(v.0))
    }
}
// Fixed closed field records, using existing exact-field support. No generic
// value tree, serde defaults or extensible source metadata schema is retained.
macro_rules! source_record {($n:ident {$($f:ident:$t:ty),*$(,)?})=>{
#[derive(Debug,Clone,PartialEq,Eq)] struct $n {$($f:$t),*}
impl Decode for $n {fn decode(r:&RawValue,p:&str)->Result<Self>{let mut f=Fields::new(r,&[$(stringify!($f)),*],p)?;Ok(Self{$($f:f.field(stringify!($f),p)?),*})}}
};}
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileRef {
    root: Tag<5>,
    path: Text<4096>,
    length: u64,
    sha256: Hash,
}
impl Decode for FileRef {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(r, &["root", "path", "length", "sha256"], p)?;
        let v = Self {
            root: f.field("root", p)?,
            path: f.field("path", p)?,
            length: f.field("length", p)?,
            sha256: f.field("sha256", p)?,
        };
        v.validate(&path(p, "path"))
            .map_err(|_| bad("path", &path(p, "path")))?;
        Ok(v)
    }
}
impl FileRef {
    fn validate(&self, p: &str) -> SResult<()> {
        let s = &self.path.0;
        require(
            !s.is_empty()
                && !s.starts_with('/')
                && !s.contains(['\0', '\\'])
                && s.split('/').all(|c| !matches!(c, "" | "." | "..")),
            "path",
            p,
        )
    }
    fn key(&self) -> (u8, &str) {
        (self.root.0, &self.path.0)
    }
}
// FileRef validation occurs immediately in Decode, including every nested use.
source_record!(ClockValue {wall_source:Tag<0>,unix_ns:u64,monotonic_source:Tag<0>,elapsed_since_operation_start_ns:u64});
fn clocks(a: &ClockValue, b: &ClockValue, p: &str) -> SResult<()> {
    require(
        a.elapsed_since_operation_start_ns <= b.elapsed_since_operation_start_ns,
        "clock",
        p,
    )
}
source_record!(ModelComponent {kind:Tag<21>,rule:Tag<21>,derivation:FileRef});
source_record!(Coverage {kind:Tag<21>,source_derivation:FileRef,validation_runs:Items<Id,65536>});
source_record!(Allocator {implementation:Tag<0>,library_files:Items<FileRef,512>,configuration:FileRef,derivation:FileRef});
source_record!(Kernel {release:Text<4096>,version:Text<4096>,page_size:u64,source:FileRef,configuration:FileRef});
source_record!(BootstrapCoverage {source_derivation:FileRef,validation_runs:Items<Id,65536>});
source_record!(Environment {
    target: Id,
    toolchain: FileRef,
    cargo_lock: FileRef,
    build_receipt: FileRef,
    allocator: Allocator,
    kernel: Kernel,
    workers: u64,
    bootstrap: BootstrapCoverage
});
source_record!(Measurement {kind:Tag<2>,scope:Tag<2>,bytes:u64,raw:FileRef});
source_record!(RunMetadata {source_manifest:Hash,source_revision:Git,executable:Hash,environment:Environment,argv:Items<Blob<4096>,256>,exit_code:Option<Int>,timed_out:bool,started:ClockValue,finished:ClockValue,measurements:Items<Measurement,65536>});
source_record!(Run {id:Id,command:FileRef,stdout:FileRef,stderr:FileRef,metadata:RunMetadata,covered_components:Items<Tag<21>,65536>});
source_record!(NamespaceIdentity {
    device: u64,
    inode: u64
});
source_record!(NamespaceOrigin {
    boot_id: Uuid,
    mount_ns: NamespaceIdentity,
    cgroup_ns: NamespaceIdentity,
    pid_ns: NamespaceIdentity,
    user_ns: NamespaceIdentity,
    installation_record: FileRef
});
source_record!(Topology {
    membership: FileRef,
    mountinfo: FileRef,
    controllers: FileRef,
    namespaces: NamespaceOrigin
});
#[derive(Debug, Clone, PartialEq, Eq)]
struct FilesystemIdentity {
    device: u64,
    mount_id: u64,
    r#type: u64,
}
impl Decode for FilesystemIdentity {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(r, &["device", "mount_id", "type"], p)?;
        Ok(Self {
            device: f.field("device", p)?,
            mount_id: f.field("mount_id", p)?,
            r#type: f.field("type", p)?,
        })
    }
}
impl FilesystemIdentity {
    fn key(&self) -> (u64, u64, u64) {
        (self.device, self.mount_id, self.r#type)
    }
}
source_record!(StructuralRoot {version:Tag<1>,domain_path:Text<4096>,mount_identity:FilesystemIdentity,disposition:Tag<1>,controller_raw:Items<FileRef,65536>});
source_record!(ControllerRaw {version:Tag<1>,path:Text<4096>,limit_raw:FileRef,current_raw:FileRef});
source_record!(FilesystemProfile {identity:FilesystemIdentity,roots:Items<Tag<5>,65536>,mount_evidence:FileRef,quota_evidence:FileRef,derivation:FileRef});
source_record!(DiskFilesystem {identity:FilesystemIdentity,roots:Items<Tag<5>,65536>,frsize:u64,bavail:u64,favail:u64,required_bytes:u64,required_files:u64,quota_evidence:FileRef});
source_record!(Component {kind:Tag<21>,phase_mask:Tag<15>,resident_bytes:u64,additional_bytes:u64});
source_record!(Requirements {components:Items<Component,22>,peak_host_bytes:u64,incremental_host_bytes:u64,required_disk_bytes:u64});
source_record!(Model {schema:Text<128>,synthetic:bool,revision:Id,source_manifest:Hash,source_revision:Git,source_files:Items<FileRef,512>,registry:Hash,build_environment:FileRef,rule_revision:Id,supported_work:Items<SupportedWork,65536>,components:Items<ModelComponent,22>,bootstrap_derivation:FileRef});
source_record!(Validation {schema:Text<128>,synthetic:bool,model:FileRef,source_manifest:Hash,source_revision:Git,environment:Environment,coverage:Items<Coverage,22>,runs:Items<Run,65536>});
source_record!(HostPolicy {schema:Text<128>,synthetic:bool,boot_id:Uuid,physical_host:Hash,host_namespace_origin:NamespaceOrigin,allocator_environment:Allocator,kernel_environment:Kernel,filesystem_environment:Items<FilesystemProfile,65536>,claim_domain:Id,model:FileRef,validation:FileRef});
source_record!(ClaimRecord {schema:Text<128>,synthetic:bool,claim_id:Id,claim_domain:Id,physical_host:Hash,boot_id:Uuid,model:FileRef,validation:FileRef,work:ClaimWork,approval:Hash,source_manifest:Hash,source_revision:Git,created:ClockValue,modeled_envelope:Requirements});
source_record!(HostObservation {schema:Text<128>,synthetic:bool,claim_id:Id,sequence:u64,physical:Hash,started:ClockValue,finished:ClockValue,topology_before:Topology,topology_after:Topology,meminfo_before:FileRef,meminfo_after:FileRef,controllers:Items<ControllerRaw,65536>,structural_root:StructuralRoot});
source_record!(DiskObservation {schema:Text<128>,synthetic:bool,claim_id:Id,sequence:u64,started:ClockValue,finished:ClockValue,filesystems:Items<DiskFilesystem,65536>});
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum SupportedWork {
    Train {
        registration_id: String,
        consumption: u8,
    },
    Reference {
        operation: u8,
        profile_id: String,
        seeds: Vec<u64>,
    },
    Target {
        operation: u8,
        profile_id: String,
        seeds: Vec<u64>,
    },
}
fn branch(r: &RawValue, p: &str) -> Result<u8> {
    // Select only among these two closed work records, then recheck the exact
    // selected branch fields. Keep typed integer/tag diagnostics intact.
    let mut fields = Fields::new(
        r,
        &[
            "kind",
            "registration_id",
            "consumption",
            "operation",
            "profile_id",
            "seeds",
            "registration",
            "request",
            "reference_id",
            "expected_file",
            "seed",
            "document",
            "campaign_id",
            "selected_seed",
            "target",
            "reference",
            "supplied_input",
        ],
        p,
    )?;
    Ok(fields.field::<Tag<2>>("kind", p)?.0)
}
impl Decode for SupportedWork {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let kind = branch(r, p)?;
        let mut f = Fields::new(
            r,
            if kind == 0 {
                &["kind", "registration_id", "consumption"]
            } else {
                &["kind", "operation", "profile_id", "seeds"]
            },
            p,
        )?;
        let _: Tag<2> = f.field("kind", p)?;
        if kind == 0 {
            Ok(Self::Train {
                registration_id: f.field::<Hash>("registration_id", p)?.0,
                consumption: f.field::<Tag<1>>("consumption", p)?.0,
            })
        } else {
            let operation = f.field::<Tag<2>>("operation", p)?.0;
            if kind == 2 && operation > 1 {
                return Err(bad("tag", &path(p, "operation")));
            }
            let profile_id = f.field::<Id>("profile_id", p)?.0;
            let seeds = f.field::<Items<u64, 65536>>("seeds", p)?.0;
            Ok(if kind == 1 {
                Self::Reference {
                    operation,
                    profile_id,
                    seeds,
                }
            } else {
                Self::Target {
                    operation,
                    profile_id,
                    seeds,
                }
            })
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
enum ClaimWork {
    Train {
        registration: Blob<A_MAX>,
        request: Blob<A_MAX>,
    },
    Reference {
        operation: u8,
        reference_id: Id,
        expected_file: Option<Hash>,
        profile_id: Id,
        seed: u64,
        document: FileRef,
    },
    Target {
        operation: u8,
        campaign_id: Id,
        profile_id: Id,
        selected_seed: u64,
        target: Option<FileRef>,
        reference: FileRef,
        supplied_input: Option<Hash>,
    },
}
impl Decode for ClaimWork {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let kind = branch(r, p)?;
        let mut f = Fields::new(
            r,
            match kind {
                0 => &["kind", "registration", "request"],
                1 => &[
                    "kind",
                    "operation",
                    "reference_id",
                    "expected_file",
                    "profile_id",
                    "seed",
                    "document",
                ],
                _ => &[
                    "kind",
                    "operation",
                    "campaign_id",
                    "profile_id",
                    "selected_seed",
                    "target",
                    "reference",
                    "supplied_input",
                ],
            },
            p,
        )?;
        let _: Tag<2> = f.field("kind", p)?;
        Ok(match kind {
            0 => Self::Train {
                registration: f.field("registration", p)?,
                request: f.field("request", p)?,
            },
            1 => Self::Reference {
                operation: f.field::<Tag<2>>("operation", p)?.0,
                reference_id: f.field("reference_id", p)?,
                expected_file: f.field("expected_file", p)?,
                profile_id: f.field("profile_id", p)?,
                seed: f.field("seed", p)?,
                document: f.field("document", p)?,
            },
            _ => Self::Target {
                operation: f.field::<Tag<1>>("operation", p)?.0,
                campaign_id: f.field("campaign_id", p)?,
                profile_id: f.field("profile_id", p)?,
                selected_seed: f.field("selected_seed", p)?,
                target: f.field("target", p)?,
                reference: f.field("reference", p)?,
                supplied_input: f.field("supplied_input", p)?,
            },
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
enum Body {
    Model(Model),
    Validation(Validation),
    Policy(HostPolicy),
    Claim(ClaimRecord),
    Host(HostObservation),
    Disk(DiskObservation),
}
#[derive(Debug, Clone)]
struct Source<'a> {
    raw: &'a [u8],
    mode: Mode,
    body: Body,
}
impl<'a> Source<'a> {
    fn sha256(&self) -> String {
        sha256_hex(self.raw)
    }
    fn parse(raw: &'a [u8], mode: Mode) -> SResult<Self> {
        check_source_length(raw.len())?;
        preflight_layout(raw, DocumentLayout::Resource)?;
        let value: &RawValue =
            serde_json::from_slice(raw).map_err(|e| SourceError::from(json_err(e, "source")))?;
        // Domain selection borrows the raw top object and uses only existing field
        // traversal; each selected record then enforces its complete exact field set.
        let schema: String = {
            struct Schema;
            impl<'de> Visitor<'de> for Schema {
                type Value = String;
                fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    f.write_str("source object")
                }
                fn visit_map<M: MapAccess<'de>>(
                    self,
                    mut m: M,
                ) -> std::result::Result<String, M::Error> {
                    let mut s = None;
                    while let Some(k) = m.next_key::<String>()? {
                        let r = m.next_value::<&RawValue>()?;
                        if k == "schema" {
                            if s.is_some() {
                                return Err(de::Error::custom("DuplicateField"));
                            }
                            s = Some(serde_json::from_str(r.get()).map_err(de::Error::custom)?);
                        }
                    }
                    s.ok_or_else(|| de::Error::custom("MissingField"))
                }
            }
            let mut d = serde_json::Deserializer::from_str(value.get());
            de::Deserializer::deserialize_map(&mut d, Schema)
                .map_err(|e| SourceError::from(json_err(e, "schema")))?
        };
        let names = [
            "resource-model",
            "resource-validation",
            "resource-host-policy",
            "resource-claim",
            "resource-host-observation",
            "resource-disk-observation",
        ];
        let i = names
            .iter()
            .position(|s| schema == mode.domain(s))
            .ok_or_else(|| refusal("domain", "schema"))?;
        let body = match i {
            0 => Body::Model(Model::decode(value, "model")?),
            1 => Body::Validation(Validation::decode(value, "validation")?),
            2 => Body::Policy(HostPolicy::decode(value, "policy")?),
            3 => Body::Claim(ClaimRecord::decode(value, "claim")?),
            4 => Body::Host(HostObservation::decode(value, "host")?),
            _ => Body::Disk(DiskObservation::decode(value, "disk")?),
        };
        let s = Self { raw, mode, body };
        s.validate()?;
        Ok(s)
    }
}
fn unique_files(v: &[FileRef], sort: bool, p: &str) -> SResult<()> {
    if sort {
        return require(
            v.windows(2).all(|w| w[0].key() < w[1].key()),
            "source-order",
            p,
        );
    }
    let mut seen = std::collections::BTreeSet::new();
    for file in v {
        require(seen.insert(file.key()), "source-order", p)?;
    }
    Ok(())
}
fn roots(v: &Items<Tag<5>, 65536>, p: &str) -> SResult<()> {
    require(!v.0.is_empty() && ordered(&v.0), "root-order", p)
}
fn environment(v: &Environment, p: &str) -> SResult<()> {
    unique_files(
        &v.allocator.library_files.0,
        false,
        &path(p, "allocator.library_files"),
    )?;
    require(
        !v.kernel.release.0.is_empty() && !v.kernel.version.0.is_empty() && v.kernel.page_size > 0,
        "kernel",
        &path(p, "kernel"),
    )?;
    require(
        ordered(&v.bootstrap.validation_runs.0),
        "run-order",
        &path(p, "bootstrap.validation_runs"),
    )
}
impl Requirements {
    fn validate(&self) -> SResult<()> {
        require(
            self.components.0.len() == 22
                && self
                    .components
                    .0
                    .iter()
                    .enumerate()
                    .all(|(i, c)| c.kind.0 == i as u8),
            "component-order",
            "claim.modeled_envelope.components",
        )?;
        let mut total = [0u64; 4];
        let mut additional = [0u64; 4];
        for c in &self.components.0 {
            require(
                c.phase_mask.0 != 0 || (c.resident_bytes == 0 && c.additional_bytes == 0),
                "inapplicable-bytes",
                "claim.modeled_envelope.components",
            )?;
            let b = c
                .resident_bytes
                .checked_add(c.additional_bytes)
                .ok_or_else(|| {
                    refusal("arithmetic-overflow", "claim.modeled_envelope.components")
                })?;
            for i in 0..4 {
                if c.phase_mask.0 & (1 << i) != 0 {
                    total[i] = total[i].checked_add(b).ok_or_else(|| {
                        refusal("arithmetic-overflow", "claim.modeled_envelope.components")
                    })?;
                    additional[i] =
                        additional[i]
                            .checked_add(c.additional_bytes)
                            .ok_or_else(|| {
                                refusal("arithmetic-overflow", "claim.modeled_envelope.components")
                            })?;
                }
            }
        }
        require(
            self.peak_host_bytes == *total.iter().max().unwrap(),
            "peak-identity",
            "claim.modeled_envelope.peak_host_bytes",
        )?;
        require(
            self.incremental_host_bytes == *additional.iter().max().unwrap(),
            "incremental-identity",
            "claim.modeled_envelope.incremental_host_bytes",
        )
    }
}
impl Source<'_> {
    fn validate(&self) -> SResult<()> {
        let synthetic = match &self.body {
            Body::Model(v) => v.synthetic,
            Body::Validation(v) => v.synthetic,
            Body::Policy(v) => v.synthetic,
            Body::Claim(v) => v.synthetic,
            Body::Host(v) => v.synthetic,
            Body::Disk(v) => v.synthetic,
        };
        require(synthetic == self.mode.synthetic(), "domain", "synthetic")?;
        match &self.body {
            Body::Model(v) => {
                require(
                    v.rule_revision.0 == "d7-cpu-allocation-v1",
                    "rule-revision",
                    "model.rule_revision",
                )?;
                require(
                    v.components.0.len() == 22
                        && v.components
                            .0
                            .iter()
                            .enumerate()
                            .all(|(i, c)| c.kind.0 == i as u8 && c.rule.0 == c.kind.0),
                    "component-order",
                    "model.components",
                )?;
                require(
                    !v.source_files.0.is_empty(),
                    "source-order",
                    "model.source_files",
                )?;
                unique_files(&v.source_files.0, true, "model.source_files")?;
                require(
                    !v.supported_work.0.is_empty(),
                    "work-order",
                    "model.supported_work",
                )?;
                let mut train_selectors = std::collections::BTreeSet::new();
                let mut other_selectors = std::collections::BTreeSet::new();
                for w in &v.supported_work.0 {
                    let unique = match w {
                        SupportedWork::Train {
                            registration_id,
                            consumption,
                        } => train_selectors.insert((registration_id.as_str(), *consumption)),
                        SupportedWork::Reference {
                            operation,
                            profile_id,
                            ..
                        } => other_selectors.insert((1, *operation, profile_id.as_str())),
                        SupportedWork::Target {
                            operation,
                            profile_id,
                            ..
                        } => other_selectors.insert((2, *operation, profile_id.as_str())),
                    };
                    require(unique, "work-duplicate", "model.supported_work")?;
                    match w {
                        SupportedWork::Reference { seeds, .. } => {
                            require(seeds == &[0], "work-seeds", "model.supported_work.seeds")?
                        }
                        SupportedWork::Target {
                            operation, seeds, ..
                        } => require(
                            !seeds.is_empty()
                                && ordered(seeds)
                                && (*operation != 1 || seeds == &[0]),
                            "work-seeds",
                            "model.supported_work.seeds",
                        )?,
                        _ => (),
                    }
                }
                require(
                    ordered(&v.supported_work.0),
                    "work-order",
                    "model.supported_work",
                )?;
            }
            Body::Validation(v) => {
                environment(&v.environment, "validation.environment")?;
                require(
                    v.coverage.0.len() == 22
                        && v.coverage
                            .0
                            .iter()
                            .enumerate()
                            .all(|(i, c)| c.kind.0 == i as u8),
                    "component-order",
                    "validation.coverage",
                )?;
                require(
                    v.runs.0.windows(2).all(|w| w[0].id < w[1].id),
                    "run-order",
                    "validation.runs",
                )?;
                for c in &v.coverage.0 {
                    require(
                        ordered(&c.validation_runs.0),
                        "run-order",
                        "validation.coverage.validation_runs",
                    )?;
                }
                // All references resolve once in this finite array; no recursive
                // expansion through repeated metadata environments is performed.
                let find = |id: &Id| v.runs.0.binary_search_by(|r| r.id.cmp(id)).ok();
                for id in &v.environment.bootstrap.validation_runs.0 {
                    require(
                        find(id).is_some(),
                        "run-reference",
                        "validation.environment.bootstrap.validation_runs",
                    )?;
                }
                for c in &v.coverage.0 {
                    for id in &c.validation_runs.0 {
                        let i = find(id).ok_or_else(|| {
                            refusal("run-reference", "validation.coverage.validation_runs")
                        })?;
                        require(
                            v.runs.0[i]
                                .covered_components
                                .0
                                .iter()
                                .any(|k| *k == c.kind),
                            "reciprocal-coverage",
                            "validation.coverage.validation_runs",
                        )?;
                    }
                }
                for r in &v.runs.0 {
                    environment(
                        &r.metadata.environment,
                        "validation.runs.metadata.environment",
                    )?;
                    require(
                        !r.metadata.argv.0.is_empty()
                            && r.metadata.argv.0.iter().all(|a| !a.0.contains(&0)),
                        "argv",
                        "validation.runs.metadata.argv",
                    )?;
                    clocks(
                        &r.metadata.started,
                        &r.metadata.finished,
                        "validation.runs.metadata.finished",
                    )?;
                    require(
                        ordered(&r.covered_components.0),
                        "component-order",
                        "validation.runs.covered_components",
                    )?;
                    for kind in &r.covered_components.0 {
                        require(
                            v.coverage.0[kind.0 as usize]
                                .validation_runs
                                .0
                                .binary_search(&r.id)
                                .is_ok(),
                            "reciprocal-coverage",
                            "validation.runs.covered_components",
                        )?;
                    }
                    require(
                        !r.covered_components.0.is_empty()
                            || v.environment
                                .bootstrap
                                .validation_runs
                                .0
                                .binary_search(&r.id)
                                .is_ok(),
                        "unreferenced-run",
                        "validation.runs",
                    )?;
                }
            }
            Body::Policy(v) => {
                require(
                    v.claim_domain.0 == "task54-host-serial-v1",
                    "claim-domain",
                    "policy.claim_domain",
                )?;
                require(
                    v.boot_id == v.host_namespace_origin.boot_id,
                    "bootstrap-boot",
                    "policy.host_namespace_origin.boot_id",
                )?;
                unique_files(
                    &v.allocator_environment.library_files.0,
                    false,
                    "policy.allocator_environment.library_files",
                )?;
                require(
                    !v.kernel_environment.release.0.is_empty()
                        && !v.kernel_environment.version.0.is_empty()
                        && v.kernel_environment.page_size > 0,
                    "kernel",
                    "policy.kernel_environment",
                )?;
                require(
                    v.filesystem_environment
                        .0
                        .windows(2)
                        .all(|w| w[0].identity.key() < w[1].identity.key()),
                    "filesystem-order",
                    "policy.filesystem_environment",
                )?;
                for f in &v.filesystem_environment.0 {
                    roots(&f.roots, "policy.filesystem_environment.roots")?;
                }
            }
            Body::Claim(v) => {
                require(
                    v.claim_domain.0 == "task54-host-serial-v1",
                    "claim-domain",
                    "claim.claim_domain",
                )?;
                match &v.work {
                    ClaimWork::Train {
                        registration,
                        request,
                    } => {
                        let case = admission::parse_case_binary(&registration.0)
                            .map_err(|_| refusal("registration", "claim.work.registration"))?;
                        let consumption =
                            evidence::resource_request_consumption(&request.0, self.mode)
                                .map_err(|_| refusal("request-domain", "claim.work.request"))?;
                        require(
                            consumption != 1 || case.seed == 0,
                            "quick-seed",
                            "claim.work.registration.seed",
                        )?;
                    }
                    ClaimWork::Reference {
                        operation,
                        expected_file,
                        seed,
                        document,
                        ..
                    } => {
                        require(
                            (*operation == 1) == expected_file.is_some(),
                            "reference-option",
                            "claim.work.expected_file",
                        )?;
                        require(
                            expected_file.as_ref().is_none_or(|h| *h == document.sha256),
                            "reference-hash",
                            "claim.work.expected_file",
                        )?;
                        require(*seed == 0, "reference-seed", "claim.work.seed")?;
                    }
                    ClaimWork::Target {
                        operation,
                        selected_seed,
                        target,
                        supplied_input,
                        ..
                    } => {
                        require(
                            (*operation == 0) == target.is_some(),
                            "target-option",
                            "claim.work.target",
                        )?;
                        require(
                            *operation != 1 || (*selected_seed == 0 && supplied_input.is_none()),
                            "lock-options",
                            "claim.work.supplied_input",
                        )?;
                    }
                }
                v.modeled_envelope.validate()?;
            }
            Body::Host(v) => {
                clocks(&v.started, &v.finished, "host.finished")?;
                let s = &v.structural_root;
                require(
                    !s.domain_path.0.is_empty()
                        && (s.disposition.0 != 1 || (s.version.0 == 1 && s.domain_path.0 == "/")),
                    "structural-root",
                    "host.structural_root",
                )?;
                unique_files(
                    &s.controller_raw.0,
                    false,
                    "host.structural_root.controller_raw",
                )?;
                let mut controllers = std::collections::BTreeSet::new();
                for c in &v.controllers.0 {
                    require(
                        !c.path.0.is_empty()
                            && controllers.insert((c.version.0, c.path.0.as_str())),
                        "controller-order",
                        "host.controllers",
                    )?;
                    require(
                        !(s.disposition.0 == 1
                            && c.version == s.version
                            && c.path == s.domain_path),
                        "structural-root-row",
                        "host.controllers.path",
                    )?;
                }
            }
            Body::Disk(v) => {
                clocks(&v.started, &v.finished, "disk.finished")?;
                require(
                    v.filesystems
                        .0
                        .windows(2)
                        .all(|w| w[0].identity.key() < w[1].identity.key()),
                    "filesystem-order",
                    "disk.filesystems",
                )?;
                for f in &v.filesystems.0 {
                    roots(&f.roots, "disk.filesystems.roots")?;
                    require(f.frsize > 0, "disk-frsize", "disk.filesystems.frsize")?;
                    let available = f
                        .frsize
                        .checked_mul(f.bavail)
                        .ok_or_else(|| refusal("arithmetic-overflow", "disk.filesystems.bavail"))?;
                    require(
                        f.required_bytes <= available,
                        "disk-bytes",
                        "disk.filesystems.required_bytes",
                    )?;
                    require(
                        f.required_files <= f.favail,
                        "disk-inodes",
                        "disk.filesystems.required_files",
                    )?;
                }
                checked_disk_sum(v.filesystems.0.iter().map(|f| f.required_bytes))?;
            }
        }
        Ok(())
    }
}
