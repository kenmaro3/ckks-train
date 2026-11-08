//! Narrow owner-facing access to the adopted inert wire, never an authority constructor.
// The live consumers are Linux-only. Other hosts compile the inert formats and
// pure tests; the Linux strict-Clippy pass still checks live dead code/imports.
#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]
use super::evidence::{
    Mode,
    observed::*,
    wire::{Bytes, H, Id, Str, Tag},
};
use super::*;
mod sources;
pub(crate) use sources::filesystem_applicability;
pub(crate) use sources::validate_filesystem_sources;
pub(crate) use sources::validate_native_sources;
pub(crate) use sources::{FilesystemFacts, filesystem_account};
pub(crate) fn policy_native(raw: &[u8], bootstrap: &[u8]) -> Result<FileReference> {
    let b = ObservedRecord::decode(RecordKind::Bootstrap, bootstrap, Mode::Production)?;
    let policy = evidence::observed::policy::ObservedPolicy::decode(raw, Mode::Production)?;
    policy.check_bootstrap(&b)?;
    let f = policy.native_validation;
    Ok(FileReference {
        root: f.root.0,
        path: f.path.0,
        length: f.length,
        hash: f.sha256.0,
    })
}
pub(crate) struct Requested {
    bytes: FixedWork,
}
impl Requested {
    pub fn train(noninteractive: bool, quick: bool) -> Result<Self> {
        let mut request = [0; 62];
        request[..6].copy_from_slice(&[u8::from(quick), 0, 0, 1, 1, 1]);
        for (slot, value) in request[6..].chunks_exact_mut(8).zip([
            67_108_864u64,
            1_000_000,
            2,
            2,
            if quick { 2 } else { 2400 },
            0,
            1,
        ]) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        let work = RequestedWork {
            mode: Mode::Production,
            body: Work::Train(Train {
                registration: Id(if noninteractive { "main_ni" } else { "main_i" }),
                profile: Id("iris__emlp1"),
                mode: u8::from(noninteractive),
                request: &request,
            }),
        };
        Ok(Self {
            bytes: work.encode_fixed()?,
        })
    }
    pub fn preparation(operation: u8, profile: &str, seeds: &[u64]) -> Result<Self> {
        ensure(
            seeds.len() <= 10,
            DocumentErrorCode::TooLarge,
            "preparation seeds",
        )?;
        let mut held = [0; 10];
        held[..seeds.len()].copy_from_slice(seeds);
        Ok(Self {
            bytes: RequestedWork {
                mode: Mode::Production,
                body: Work::Preparation {
                    operation,
                    profile: Id(profile),
                    seeds: held,
                    count: seeds.len(),
                },
            }
            .encode_fixed()?,
        })
    }
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }
    pub fn request_matches(&self, request: &crate::study_admission::DiagonalRequest) -> Result<()> {
        let mut e = Encoder::new(DocumentLayout::Start);
        crate::study_admission::encode_request_body(&mut e, request.into())?;
        let work = RequestedWork::decode(self.bytes(), Mode::Production)?;
        let Work::Train(t) = work.body else {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "train request required",
            ));
        };
        ensure(
            t.request == e.bytes && request.partitions.is_none(),
            DocumentErrorCode::CrossLinkMismatch,
            "compiled immutable request",
        )
    }
    pub fn bind_train(
        &self,
        case: &crate::study_profile::RegisteredCase,
        request: &crate::study_admission::DiagonalRequest,
    ) -> Result<()> {
        let mut e = Encoder::new(DocumentLayout::Start);
        crate::study_admission::encode_request_body(&mut e, request.into())?;
        let partitions = request
            .partitions
            .as_ref()
            .map(|p| u64::try_from(p.len()))
            .transpose()
            .map_err(|_| err(DocumentErrorCode::TooLarge, "request partitions"))?;
        RequestedWork::decode(self.bytes(), Mode::Production)?.bind_train(
            &registered_case_canonical(case)?,
            "study-train-cpu-serial-linux-clean-v2",
            &e.bytes,
            partitions,
        )
    }
    #[allow(dead_code)] // Closed live-borrow consumers connect in Step3; pure graph tests already use it.
    pub fn child(&self, parent: u8, child: u8, profile: &str, selected: u64) -> Result<()> {
        RequestedWork::decode(self.bytes(), Mode::Production)?
            .check_child(parent, child, profile, selected)
    }
    pub fn bound_request(&self) -> Result<Vec<u8>> {
        Ok(RequestedWork::decode(self.bytes(), Mode::Production)?
            .train_request()?
            .to_vec())
    }
    pub fn bind_preparation(&self, operation: u8, profile: &str, seeds: &[u64]) -> Result<()> {
        RequestedWork::decode(self.bytes(), Mode::Production)?
            .bind_preparation_root(operation, profile, seeds)
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileReference {
    pub root: u8,
    pub path: String,
    pub length: u64,
    pub hash: String,
}
impl FileReference {
    fn wire(&self) -> evidence::FileRef<'_> {
        evidence::FileRef {
            root: Tag(self.root),
            path: Str(&self.path),
            length: self.length,
            sha256: H(&self.hash),
        }
    }
    fn from_wire(r: &evidence::FileRef<'_>) -> Self {
        Self {
            root: r.root.0,
            path: r.path.0.into(),
            length: r.length,
            hash: r.sha256.0.into(),
        }
    }
}
pub(crate) struct BootstrapView<'a> {
    pub policy: (u8, &'a str, u64, &'a str),
    pub host: (u8, &'a str, u64, &'a str),
    pub physical: &'a str,
    pub boot: &'a str,
    pub source: &'a str,
    pub revision: &'a str,
    pub executable: &'a str,
    pub recipe: &'a str,
}
pub(crate) fn bootstrap(bytes: &[u8]) -> Result<BootstrapView<'_>> {
    let record = ObservedRecord::decode(RecordKind::Bootstrap, bytes, Mode::Production)?;
    let RecordBody::Bootstrap(b) = record.body else {
        return Err(err(DocumentErrorCode::UnsupportedTag, "bootstrap"));
    };
    Ok(BootstrapView {
        policy: (
            b.policy.root.0,
            b.policy.path.0,
            b.policy.length,
            b.policy.sha256.0,
        ),
        host: (
            b.namespace_origin.root.0,
            b.namespace_origin.path.0,
            b.namespace_origin.length,
            b.namespace_origin.sha256.0,
        ),
        physical: b.physical_host.0,
        boot: b.boot_uuid.0,
        source: b.source_manifest.0,
        revision: b.source_revision.0,
        executable: b.executable.0,
        recipe: b.recipe.0,
    })
}
pub(crate) struct Stamp<'a> {
    pub id: &'a str,
    pub boot: &'a str,
    pub pid: u64,
    pub start: u64,
    pub unix_ns: u64,
    pub elapsed_ns: u64,
}
fn clock(unix_ns: u64, elapsed: u64) -> Clock {
    Clock {
        wall_source: WallSource::Unix,
        unix_ns,
        monotonic_source: MonotonicSource::Instant,
        elapsed_since_operation_start_ns: elapsed,
    }
}
pub(crate) fn host_pending(s: &Stamp<'_>, root: &[u8], identity: (u64, u64)) -> Result<Vec<u8>> {
    ObservedRecord {
        mode: Mode::Production,
        kind: RecordKind::HostPending,
        body: RecordBody::HostPending(HostPending {
            claim_id: Id(s.id),
            boot: Str(s.boot),
            supervisor_pid: s.pid,
            supervisor_start_ticks: s.start,
            observation_root: Bytes(root),
            root_device: identity.0,
            root_inode: identity.1,
        }),
    }
    .canonical()
}
pub(crate) fn pending(s: &Stamp<'_>, policy: &FileReference, physical: &str) -> Result<Vec<u8>> {
    ObservedRecord {
        mode: Mode::Production,
        kind: RecordKind::Pending,
        body: RecordBody::Pending(Pending {
            claim_id: Id(s.id),
            policy: policy.wire(),
            physical_host: H(physical),
            boot: Str(s.boot),
            supervisor_pid: s.pid,
            supervisor_start_ticks: s.start,
            created: clock(s.unix_ns, s.elapsed_ns),
        }),
    }
    .canonical()
}
pub(crate) fn closed(pending_hash: &str, end: &FileReference) -> Result<Vec<u8>> {
    ObservedRecord {
        mode: Mode::Production,
        kind: RecordKind::HostClosed,
        body: RecordBody::HostClosed(HostClosed {
            pending_sha256: H(pending_hash),
            supervisor_end: end.wire(),
        }),
    }
    .canonical()
}
pub(crate) fn closed_reference(bytes: &[u8], expected_pending: &str) -> Result<FileReference> {
    let record = ObservedRecord::decode(RecordKind::HostClosed, bytes, Mode::Production)?;
    let RecordBody::HostClosed(v) = record.body else {
        return Err(err(DocumentErrorCode::UnsupportedTag, "closed"));
    };
    ensure(
        v.pending_sha256.0 == expected_pending,
        DocumentErrorCode::CrossLinkMismatch,
        "closed.pending",
    )?;
    Ok(FileReference::from_wire(&v.supervisor_end))
}
pub(crate) struct SupervisorFacts<'a> {
    pub stamp: Stamp<'a>,
    pub physical: &'a str,
    pub worker_pid: u64,
    pub worker_start: u64,
}
fn binding<'a>(s: &SupervisorFacts<'a>, initial: &'a FileReference) -> SupervisorBinding<'a> {
    SupervisorBinding {
        mechanism: Id("linux-pidfd-broker-v1"),
        physical_host: H(s.physical),
        boot: Str(s.stamp.boot),
        supervisor_pid: s.stamp.pid,
        supervisor_start_ticks: s.stamp.start,
        worker_pid: s.worker_pid,
        worker_start_ticks: s.worker_start,
        claim_id: Id(s.stamp.id),
        initial_sample: initial.wire(),
        rss_limit_kib: 94_371_840,
        sample_interval_ns: 10_000_000_000,
        term_grace_ns: 5_000_000_000,
        max_processes: 4,
    }
}
pub(crate) struct ClaimInputs<'a> {
    pub facts: SupervisorFacts<'a>,
    pub pending: &'a FileReference,
    pub initial: &'a FileReference,
    pub work: &'a Requested,
    pub approval: &'a str,
    pub source: &'a str,
    pub revision: &'a str,
}
pub(crate) fn claim(v: &ClaimInputs<'_>) -> Result<Vec<u8>> {
    ObservedRecord {
        mode: Mode::Production,
        kind: RecordKind::Claim,
        body: RecordBody::Claim(ClaimRecord {
            pending: v.pending.wire(),
            work: Bytes(v.work.bytes()),
            approval: H(v.approval),
            source_manifest: H(v.source),
            source_revision: evidence::wire::Git(v.revision),
            created: clock(v.facts.stamp.unix_ns, v.facts.stamp.elapsed_ns),
            supervisor: binding(&v.facts, v.initial),
        }),
    }
    .canonical()
}
#[derive(Clone)]
pub(crate) struct SampleValues {
    pub sequence: u64,
    pub started: (u64, u64),
    pub finished: (u64, u64),
    pub phase: u8,
    pub processes: Vec<(u64, u64, u64)>,
    pub valid: bool,
    pub status: u8,
}
fn sample(v: &SampleValues) -> Result<Sample> {
    let sum = v.processes.iter().try_fold(0u64, |n, p| {
        n.checked_add(p.2)
            .ok_or_else(|| err(DocumentErrorCode::TooLarge, "sample.sum"))
    })?;
    Ok(Sample {
        seq: v.sequence,
        started: clock(v.started.0, v.started.1),
        finished: clock(v.finished.0, v.finished.1),
        phase: Tag(v.phase),
        processes: v
            .processes
            .iter()
            .map(|p| ProcessSample {
                pid: p.0,
                start_ticks: p.1,
                rss_kib: p.2,
            })
            .collect(),
        sum_kib: sum,
        valid: Tag(u8::from(v.valid)),
        status: Tag(v.status),
    })
}
pub(crate) fn initial_sample(v: &SampleValues, pending: &FileReference) -> Result<Vec<u8>> {
    ObservedRecord {
        mode: Mode::Production,
        kind: RecordKind::InitialSample,
        body: RecordBody::InitialSample(InitialSample {
            sample: sample(v)?,
            pending: pending.wire(),
        }),
    }
    .canonical()
}
pub(crate) fn sample_header(claim: &FileReference) -> Result<Vec<u8>> {
    use evidence::wire::Wire;
    let mut e = Encoder::new(DocumentLayout::Start);
    e.text(ENCODING_VERSION)?;
    e.text("hekit-study-resource-observed-samples-v1")?;
    e.byte(0)?;
    claim.wire().write(&mut e)?;
    Ok(e.bytes)
}
pub(crate) fn sample_frame(v: &SampleValues) -> Result<Vec<u8>> {
    use evidence::wire::Wire;
    let mut e = Encoder::new(DocumentLayout::Start);
    sample(v)?.write(&mut e)?;
    let mut bytes = (e.bytes.len() as u32).to_le_bytes().to_vec();
    bytes.extend(e.bytes);
    Ok(bytes)
}
pub(crate) struct EndInputs<'a> {
    pub pending: &'a FileReference,
    pub claim: Option<&'a FileReference>,
    pub start: Option<&'a FileReference>,
    pub result: Option<&'a FileReference>,
    pub worker_end: Option<&'a FileReference>,
    pub samples: Option<&'a FileReference>,
    pub last_sequence: u64,
    pub peak: Option<u64>,
    pub reason: u8,
    pub wait_status: Option<u64>,
    pub resolved: bool,
    pub ended: (u64, u64),
}
pub(crate) fn supervisor_end(v: &EndInputs<'_>) -> Result<Vec<u8>> {
    ObservedRecord {
        mode: Mode::Production,
        kind: RecordKind::SupervisorEnd,
        body: RecordBody::SupervisorEnd(SupervisorEnd {
            pending: v.pending.wire(),
            claim: v.claim.map(FileReference::wire),
            start: v.start.map(FileReference::wire),
            result: v.result.map(FileReference::wire),
            worker_end: v.worker_end.map(FileReference::wire),
            samples: v.samples.map(FileReference::wire),
            last_sample_sequence: v.last_sequence,
            metric: Id("peak_rss_kb_tree_sampled_10s"),
            peak_sampled_rss_kib: v.peak,
            reason: Tag(v.reason),
            wait_status: v.wait_status,
            custody_resolved: Tag(u8::from(v.resolved)),
            ended: clock(v.ended.0, v.ended.1),
        }),
    }
    .canonical()
}
pub(crate) fn prior_marker(bytes: &[u8]) -> Result<(String, Vec<u8>, (u64, u64))> {
    let record = ObservedRecord::decode(RecordKind::HostPending, bytes, Mode::Production)?;
    let RecordBody::HostPending(v) = record.body else {
        return Err(err(DocumentErrorCode::UnsupportedTag, "host pending"));
    };
    Ok((
        v.claim_id.0.into(),
        v.observation_root.0.into(),
        (v.root_device, v.root_inode),
    ))
}
pub(crate) fn prior_end_resolved(bytes: &[u8]) -> Result<()> {
    let record = ObservedRecord::decode(RecordKind::SupervisorEnd, bytes, Mode::Production)?;
    let RecordBody::SupervisorEnd(v) = record.body else {
        return Err(err(DocumentErrorCode::UnsupportedTag, "supervisor end"));
    };
    ensure(
        v.custody_resolved.0 == 1,
        DocumentErrorCode::CrossLinkMismatch,
        "unresolved prior custody",
    )
}
pub(crate) fn prior_end_pending(bytes: &[u8]) -> Result<FileReference> {
    prior_end_resolved(bytes)?;
    let record = ObservedRecord::decode(RecordKind::SupervisorEnd, bytes, Mode::Production)?;
    let RecordBody::SupervisorEnd(v) = record.body else {
        return Err(err(DocumentErrorCode::UnsupportedTag, "supervisor end"));
    };
    Ok(FileReference::from_wire(&v.pending))
}
pub(crate) fn prior_pending_matches(marker: &[u8], observed: &[u8]) -> Result<()> {
    let m = ObservedRecord::decode(RecordKind::HostPending, marker, Mode::Production)?;
    let p = ObservedRecord::decode(RecordKind::Pending, observed, Mode::Production)?;
    let (RecordBody::HostPending(m), RecordBody::Pending(p)) = (m.body, p.body) else {
        return Err(err(DocumentErrorCode::UnsupportedTag, "pending pair"));
    };
    ensure(
        m.claim_id.0 == p.claim_id.0
            && m.boot.0 == p.boot.0
            && m.supervisor_pid == p.supervisor_pid
            && m.supervisor_start_ticks == p.supervisor_start_ticks,
        DocumentErrorCode::CrossLinkMismatch,
        "host/Observation pending",
    )
}
#[allow(clippy::too_many_arguments)] // Exact authenticated wire projection, not a general caller provider.
pub(crate) fn run_claim_matches(
    raw: &[u8],
    work: &Requested,
    approval: &str,
    source: &str,
    id: &str,
    boot: &str,
    parent: (u64, u64),
    worker: (u64, u64),
) -> Result<()> {
    let record = ObservedRecord::decode(RecordKind::Claim, raw, Mode::Production)?;
    let RecordBody::Claim(c) = record.body else {
        return Err(err(DocumentErrorCode::UnsupportedTag, "Run claim"));
    };
    ensure(
        c.work.0 == work.bytes()
            && c.approval.0 == approval
            && c.source_manifest.0 == source
            && c.supervisor.claim_id.0 == id
            && c.supervisor.boot.0 == boot
            && (
                c.supervisor.supervisor_pid,
                c.supervisor.supervisor_start_ticks,
            ) == parent
            && (c.supervisor.worker_pid, c.supervisor.worker_start_ticks) == worker,
        DocumentErrorCode::CrossLinkMismatch,
        "live Run claim binding",
    )
}

#[cfg(test)]
mod probe_tests;

pub(crate) mod probe_output;

pub(crate) use sources::NATIVE_CHECKS;
