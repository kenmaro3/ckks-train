//! Fixed ordered observed-resource/supervisor records, never operating-system facts.
use super::*;
wire_record!(Bootstrap {policy:FileRef<'a>,physical_host:H<'a>,boot_uuid:Str<'a,36>,namespace_origin:FileRef<'a>,claim_domain:Id<'a>,source_manifest:H<'a>,source_revision:Git<'a>,executable:H<'a>,recipe:Id<'a>});
wire_record!(Pending {claim_id:Id<'a>,policy:FileRef<'a>,physical_host:H<'a>,boot:Str<'a,36>,supervisor_pid:u64,supervisor_start_ticks:u64,created:Clock});
wire_record!(SupervisorBinding {mechanism:Id<'a>,physical_host:H<'a>,boot:Str<'a,36>,supervisor_pid:u64,supervisor_start_ticks:u64,worker_pid:u64,worker_start_ticks:u64,claim_id:Id<'a>,initial_sample:FileRef<'a>,rss_limit_kib:u64,sample_interval_ns:u64,term_grace_ns:u64,max_processes:u64});
impl SupervisorBinding<'_> {
    pub(in crate::study_reference::document::evidence) fn validate(&self) -> Result<()> {
        uuid(self.boot.0)?;
        self.initial_sample.validate()?;
        same(
            self.mechanism.0 == "linux-pidfd-broker-v1"
                && self.rss_limit_kib == RSS_LIMIT
                && self.sample_interval_ns == SAMPLE_INTERVAL
                && self.term_grace_ns == TERM_GRACE
                && self.max_processes == 4,
            "supervisor.constants",
        )?;
        same(
            self.supervisor_pid > 0
                && self.worker_pid > 0
                && self.supervisor_pid != self.worker_pid
                && self.supervisor_start_ticks > 0
                && self.worker_start_ticks > 0,
            "supervisor.identities",
        )
    }
}
wire_record!(ClaimRecord {pending:FileRef<'a>,work:Bytes<'a>,approval:H<'a>,source_manifest:H<'a>,source_revision:Git<'a>,created:Clock,supervisor:SupervisorBinding<'a>});
wire_record!(HostPending {claim_id:Id<'a>,boot:Str<'a,36>,supervisor_pid:u64,supervisor_start_ticks:u64,observation_root:Bytes<'a>,root_device:u64,root_inode:u64});
wire_record!(HostClosed {pending_sha256:H<'a>,supervisor_end:FileRef<'a>});
wire_record!(Allocation {site:Id<'a>,count:u64,element_bytes:u64});
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct SourceDemand<'a> {
    pub(in crate::study_reference::document::evidence) phase: Tag<4>,
    pub(in crate::study_reference::document::evidence) retained_payload_bytes: u64,
    pub(in crate::study_reference::document::evidence) additional_payload_bytes: u64,
    pub(in crate::study_reference::document::evidence) allocations: Vec<Allocation<'a>>,
}
impl<'a> Wire<'a> for SourceDemand<'a> {
    const MIN: usize = 25;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let v = Self {
            phase: Tag::read(c)?,
            retained_payload_bytes: c.u64()?,
            additional_payload_bytes: c.u64()?,
            allocations: wire::read_vec(c, 512)?,
        };
        v.validate()?;
        Ok(v)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.validate()?;
        self.phase.write(e)?;
        self.retained_payload_bytes.write(e)?;
        self.additional_payload_bytes.write(e)?;
        self.allocations.write(e)
    }
}
impl SourceDemand<'_> {
    pub(in crate::study_reference::document::evidence) fn validate(&self) -> Result<()> {
        same(
            self.phase.0 <= 4 && self.allocations.len() <= 512,
            "demand.bounds",
        )?;
        let mut prior = None;
        let mut total = 0;
        for a in &self.allocations {
            let s = a.site.0;
            let family = [
                "raw",
                "decoded_f64",
                "cast_f32",
                "order",
                "initial",
                "plan",
                "params",
                "keys",
                "ciphertext",
                "fp32",
                "snapshot",
                "result",
            ]
            .iter()
            .any(|f| {
                s == *f
                    || s.strip_prefix(f)
                        .and_then(|s| s.strip_prefix('_'))
                        .is_some_and(|v| {
                            !v.is_empty()
                                && v.bytes().all(|b| b.is_ascii_digit())
                                && (v == "0" || !v.starts_with('0'))
                                && v.parse::<u64>().is_ok()
                        })
            });
            same(family && prior.is_none_or(|p| p < s), "demand.sites")?;
            prior = Some(s);
            total = add(
                total,
                a.count
                    .checked_mul(a.element_bytes)
                    .ok_or_else(|| err(DocumentErrorCode::TooLarge, "demand.product"))?,
            )?;
        }
        same(
            total == add(self.retained_payload_bytes, self.additional_payload_bytes)?,
            "demand.sum",
        )
    }
    /// Necessary incremental payload comparison only; never a physical-fit proof.
    pub(in crate::study_reference::document::evidence) fn check_supplied_capacity(
        &self,
        o: &tails::Observation<'_>,
    ) -> Result<()> {
        self.validate()?;
        let mut available = o.host_mem_available;
        for c in &o.cgroups {
            if let Some(limit) = c.limit {
                available = available.min(limit.saturating_sub(c.current));
            }
        }
        same(
            self.additional_payload_bytes <= available,
            "demand.necessary_capacity",
        )
    }
}
wire_record!(OperationalResource {applicability:Applicability<'a>,physical:H<'a>,policy:FileRef<'a>,native_validation:FileRef<'a>,sizing_revision:Id<'a>,demand:SourceDemand<'a>,requested:Requested,observation:tails::Observation<'a>,supervisor:SupervisorBinding<'a>,claim:FileRef<'a>});
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct ProcessSample {
    pub(in crate::study_reference::document::evidence) pid: u64,
    pub(in crate::study_reference::document::evidence) start_ticks: u64,
    pub(in crate::study_reference::document::evidence) rss_kib: u64,
}
impl<'a> Wire<'a> for ProcessSample {
    const MIN: usize = 24;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            pid: c.u64()?,
            start_ticks: c.u64()?,
            rss_kib: c.u64()?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.pid.write(e)?;
        self.start_ticks.write(e)?;
        self.rss_kib.write(e)
    }
}
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct Sample {
    pub(in crate::study_reference::document::evidence) seq: u64,
    pub(in crate::study_reference::document::evidence) started: Clock,
    pub(in crate::study_reference::document::evidence) finished: Clock,
    pub(in crate::study_reference::document::evidence) phase: Tag<4>,
    pub(in crate::study_reference::document::evidence) processes: Vec<ProcessSample>,
    pub(in crate::study_reference::document::evidence) sum_kib: u64,
    pub(in crate::study_reference::document::evidence) valid: Tag<1>,
    pub(in crate::study_reference::document::evidence) status: Tag<4>,
}
impl<'a> Wire<'a> for Sample {
    const MIN: usize = 63;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let v = Self {
            seq: c.u64()?,
            started: Clock::read(c)?,
            finished: Clock::read(c)?,
            phase: Tag::read(c)?,
            processes: wire::read_vec(c, 4)?,
            sum_kib: c.u64()?,
            valid: Tag::read(c)?,
            status: Tag::read(c)?,
        };
        v.validate()?;
        Ok(v)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.validate()?;
        self.seq.write(e)?;
        self.started.write(e)?;
        self.finished.write(e)?;
        self.phase.write(e)?;
        self.processes.write(e)?;
        self.sum_kib.write(e)?;
        self.valid.write(e)?;
        self.status.write(e)
    }
}
impl Sample {
    pub(in crate::study_reference::document::evidence) fn validate(&self) -> Result<()> {
        clock_order(&self.started, &self.finished)?;
        same(self.processes.len() <= 4, "sample.processes")?;
        let mut total = 0;
        for (i, p) in self.processes.iter().enumerate() {
            same(
                p.pid > 0
                    && p.start_ticks > 0
                    && !self.processes[..i].iter().any(|q| q.pid == p.pid),
                "sample.identity",
            )?;
            total = add(total, p.rss_kib)?;
        }
        same(total == self.sum_kib, "sample.sum")?;
        same(
            if self.valid.0 == 1 {
                matches!(self.status.0, 0 | 1 | 4)
                    && ((self.sum_kib > RSS_LIMIT) == (self.status.0 == 1))
            } else {
                matches!(self.status.0, 2 | 3)
            },
            "sample.status",
        )
    }
}
wire_record!(InitialSample {sample:Sample,pending:FileRef<'a>});
wire_record!(SupervisorEnd {pending:FileRef<'a>,claim:Option<FileRef<'a>>,start:Option<FileRef<'a>>,result:Option<FileRef<'a>>,worker_end:Option<FileRef<'a>>,samples:Option<FileRef<'a>>,last_sample_sequence:u64,metric:Id<'a>,peak_sampled_rss_kib:Option<u64>,reason:Tag<7>,wait_status:Option<u64>,custody_resolved:Tag<1>,ended:Clock});
wire_record!(FullEnd {attempt:Id<'a>,admission:FileRef<'a>,start:FileRef<'a>,result:Option<FileRef<'a>>,result_json:Option<FileRef<'a>>,result_export:Option<FileRef<'a>>,outcome:Tag<2>,completed_updates:u64,processed_samples:u64,result_publication_ns:u64,ended:Clock});
#[derive(Debug, Clone, Copy)]
pub(in crate::study_reference::document::evidence) enum RecordKind {
    Bootstrap,
    Pending,
    Claim,
    HostPending,
    HostClosed,
    InitialSample,
    SupervisorEnd,
    FullEnd,
    Result,
}
impl RecordKind {
    fn name(self) -> &'static str {
        match self {
            Self::Bootstrap => "resource-observed-bootstrap",
            Self::Pending => "resource-observed-pending",
            Self::Claim => "resource-observed-claim",
            Self::HostPending => "host-claim-pending",
            Self::HostClosed => "host-claim-closed",
            Self::InitialSample => "resource-observed-initial-sample",
            Self::SupervisorEnd => "supervisor-end",
            Self::FullEnd => "he-end",
            Self::Result => "cpu-result",
        }
    }
}
// Keep inert fixed-buffer bootstrap parsing free of a new heap allocation.
// Variant stack size is not a physical memory bound.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) enum RecordBody<'a> {
    Bootstrap(Bootstrap<'a>),
    Pending(Pending<'a>),
    Claim(ClaimRecord<'a>),
    HostPending(HostPending<'a>),
    HostClosed(HostClosed<'a>),
    InitialSample(InitialSample<'a>),
    SupervisorEnd(SupervisorEnd<'a>),
    FullEnd(FullEnd<'a>),
    Result(CpuResult<'a>),
}
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct ObservedRecord<'a> {
    pub(in crate::study_reference::document::evidence) mode: Mode,
    pub(in crate::study_reference::document::evidence) kind: RecordKind,
    pub(in crate::study_reference::document::evidence) body: RecordBody<'a>,
}
impl<'a> ObservedRecord<'a> {
    pub(in crate::study_reference::document::evidence) fn decode(
        kind: RecordKind,
        b: &'a [u8],
        mode: Mode,
    ) -> Result<Self> {
        let mut c = cursor(
            b,
            kind.name(),
            1,
            mode,
            if matches!(
                kind,
                RecordKind::Bootstrap | RecordKind::HostPending | RecordKind::HostClosed
            ) {
                4096
            } else {
                A_MAX
            },
        )?;
        let body = match kind {
            RecordKind::Bootstrap => RecordBody::Bootstrap(Bootstrap::read(&mut c)?),
            RecordKind::Pending => RecordBody::Pending(Pending::read(&mut c)?),
            RecordKind::Claim => RecordBody::Claim(ClaimRecord::read(&mut c)?),
            RecordKind::HostPending => RecordBody::HostPending(HostPending::read(&mut c)?),
            RecordKind::HostClosed => RecordBody::HostClosed(HostClosed::read(&mut c)?),
            RecordKind::InitialSample => RecordBody::InitialSample(InitialSample::read(&mut c)?),
            RecordKind::SupervisorEnd => RecordBody::SupervisorEnd(SupervisorEnd::read(&mut c)?),
            RecordKind::FullEnd => RecordBody::FullEnd(FullEnd::read(&mut c)?),
            RecordKind::Result => RecordBody::Result(CpuResult::read(&mut c)?),
        };
        c.finish()?;
        let v = Self { mode, kind, body };
        v.validate()?;
        Ok(v)
    }
    pub(in crate::study_reference::document::evidence) fn validate(&self) -> Result<()> {
        match &self.body {
            RecordBody::Bootstrap(v) => {
                v.policy.validate()?;
                v.namespace_origin.validate()?;
                uuid(v.boot_uuid.0)?;
                same(
                    v.claim_domain.0 == "task54-host-serial-v1"
                        && [TRAIN_RECIPE, "study-reference-linux-clean-v2"].contains(&v.recipe.0),
                    "bootstrap.constants",
                )?;
            }
            RecordBody::Pending(v) => {
                v.policy.validate()?;
                uuid(v.boot.0)?;
                same(
                    v.supervisor_pid > 0 && v.supervisor_start_ticks > 0,
                    "pending.identity",
                )?;
            }
            RecordBody::Claim(v) => {
                v.pending.validate()?;
                RequestedWork::decode(v.work.0, self.mode)?;
                v.supervisor.validate()?;
            }
            RecordBody::HostPending(v) => {
                uuid(v.boot.0)?;
                same(
                    v.supervisor_pid > 0
                        && v.supervisor_start_ticks > 0
                        && v.observation_root.0.starts_with(b"/")
                        && v.observation_root.0.len() <= 4096
                        && !v.observation_root.0.contains(&0)
                        && v.root_inode > 0,
                    "host.pending.identity",
                )?;
            }
            RecordBody::HostClosed(v) => {
                v.supervisor_end.validate()?;
            }
            RecordBody::InitialSample(v) => {
                v.sample.validate()?;
                v.pending.validate()?;
                same(
                    v.sample.seq == 0 && v.sample.valid.0 == 1 && v.sample.status.0 == 0,
                    "initial.sample",
                )?;
            }
            RecordBody::SupervisorEnd(v) => {
                v.pending.validate()?;
                for f in [&v.claim, &v.start, &v.result, &v.worker_end, &v.samples]
                    .into_iter()
                    .flatten()
                {
                    f.validate()?;
                }
                same(
                    v.metric.0 == "peak_rss_kb_tree_sampled_10s",
                    "supervisor.metric",
                )?;
                if v.reason.0 == 0 {
                    same(
                        v.custody_resolved.0 == 1
                            && v.claim.is_some()
                            && v.start.is_some()
                            && v.result.is_some()
                            && v.worker_end.is_some()
                            && v.samples.is_some()
                            && v.wait_status == Some(0)
                            && v.peak_sampled_rss_kib.is_some(),
                        "supervisor.complete",
                    )?;
                }
                if v.reason.0 == 7 {
                    same(v.custody_resolved.0 == 0, "supervisor.unresolved")?;
                }
            }
            RecordBody::FullEnd(v) => {
                v.admission.validate()?;
                v.start.validate()?;
                for f in [&v.result, &v.result_json, &v.result_export]
                    .into_iter()
                    .flatten()
                {
                    f.validate()?;
                }
                same(
                    v.completed_updates <= 2400
                        && v.processed_samples
                            == v.completed_updates
                                .checked_mul(20)
                                .ok_or_else(|| err(DocumentErrorCode::TooLarge, "end.samples"))?,
                    "end.counts",
                )?;
                if v.outcome.0 == 0 {
                    same(
                        v.completed_updates == 2400
                            && v.result.is_some()
                            && v.result_json.is_some()
                            && v.result_export.is_some(),
                        "end.complete",
                    )?;
                }
            }
            RecordBody::Result(v) => v.validate(self.mode)?,
        }
        Ok(())
    }
    pub(in crate::study_reference::document::evidence) fn canonical(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut e = encoder(self.kind.name(), 1, self.mode)?;
        match &self.body {
            RecordBody::Bootstrap(v) => v.write(&mut e)?,
            RecordBody::Pending(v) => v.write(&mut e)?,
            RecordBody::Claim(v) => v.write(&mut e)?,
            RecordBody::HostPending(v) => v.write(&mut e)?,
            RecordBody::HostClosed(v) => v.write(&mut e)?,
            RecordBody::InitialSample(v) => v.write(&mut e)?,
            RecordBody::SupervisorEnd(v) => v.write(&mut e)?,
            RecordBody::FullEnd(v) => v.write(&mut e)?,
            RecordBody::Result(v) => v.write(&mut e)?,
        }
        let cap = if matches!(
            self.kind,
            RecordKind::Bootstrap | RecordKind::HostPending | RecordKind::HostClosed
        ) {
            4096
        } else {
            A_MAX
        };
        ensure(
            e.bytes.len() <= cap,
            DocumentErrorCode::TooLarge,
            "observed.length",
        )?;
        Ok(e.bytes)
    }
}
/// Borrowed framed traversal, bounded before each record, never allocates a stream.
pub(in crate::study_reference::document::evidence) fn check_sample_stream(
    b: &[u8],
    mode: Mode,
) -> Result<()> {
    let mut c = cursor(b, "resource-observed-samples", 1, mode, 64 * 1024 * 1024)?;
    let claim = FileRef::read(&mut c)?;
    claim.validate()?;
    let mut sequence = 0;
    let mut last = None;
    while c.remaining() > 0 {
        let mut len = [0; 4];
        len.copy_from_slice(c.take(4)?);
        let len = u32::from_le_bytes(len) as usize;
        same((63..=159).contains(&len), "samples.frame")?;
        let mut f = Cursor::new(c.take(len)?);
        let sample = Sample::read(&mut f)?;
        f.finish()?;
        same(sample.seq == sequence, "samples.sequence")?;
        if let Some(clock) = last {
            clock_order(&clock, &sample.started)?;
        }
        last = Some(sample.finished);
        sequence = add(sequence, 1)?;
    }
    same(sequence > 0, "samples.empty")
}
