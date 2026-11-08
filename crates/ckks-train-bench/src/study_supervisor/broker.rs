//! Borrowed live worker transport. Construction is confined to the private handshake.
use super::protocol::{Channel, Tag};
use crate::{
    study_provenance::{ProvenanceError, Result, ensure},
    study_reference::operational::Requested,
};
use std::{cell::RefCell, os::unix::ffi::OsStrExt, process::Command, rc::Rc};
struct State {
    channel: Channel,
    active: bool,
    #[cfg(target_os = "linux")]
    parent: super::handshake::ParentGuard,
    #[cfg(target_os = "linux")]
    probe_origin: Option<u64>,
}
#[derive(Clone)]
pub(crate) struct Broker {
    state: Rc<RefCell<State>>,
}
enum BoundWork {
    Train {
        canonical: Vec<u8>,
        digest: String,
        request: Vec<u8>,
        #[allow(dead_code)] // Retained for the Step3 bound Train subtree.
        profile: String,
        #[allow(dead_code)] // Retained for the Step3 bound Train subtree.
        selected: u64,
    },
    Preparation {
        operation: u8,
        profile: String,
        seeds: Vec<u64>,
    },
}
pub(crate) struct WorkerOwner {
    broker: Broker,
    work: Requested,
    approval: String,
    source: String,
    _lock: std::fs::File,
    context_bound: std::cell::Cell<bool>,
    bound: std::cell::OnceCell<BoundWork>,
    roots: Vec<u8>,
}
impl WorkerOwner {
    // Called only after the private child handshake verifies parent process,
    // executable, boot, lock description, common clock and durable claim binding.
    #[cfg(target_os = "linux")]
    pub(super) fn authenticated(
        channel: Channel,
        parent: super::handshake::ParentGuard,
        lock: std::fs::File,
        work: Requested,
        approval: String,
        source: String,
        roots: Vec<u8>,
    ) -> Self {
        Self {
            broker: Broker {
                state: Rc::new(RefCell::new(State {
                    channel,
                    active: true,
                    parent,
                    probe_origin: None,
                })),
            },
            work,
            approval,
            source,
            _lock: lock,
            context_bound: std::cell::Cell::new(false),
            bound: std::cell::OnceCell::new(),
            roots,
        }
    }
    pub(crate) fn broker(&self) -> Result<Broker> {
        self.check()?;
        Ok(self.broker.clone())
    }
    pub(crate) fn check(&self) -> Result<()> {
        self.broker.check()
    }
    pub(crate) fn bind_context(&self, approval: &str, source: &str) -> Result<()> {
        self.check()?;
        ensure(
            !self.context_bound.replace(true) && approval == self.approval && source == self.source,
            "original operational context binding",
        )
    }
    pub(crate) fn bind_train(
        &self,
        case: &crate::study_profile::RegisteredCase,
        request: &crate::study_admission::DiagonalRequest,
    ) -> Result<()> {
        self.check()?;
        ensure(
            self.context_bound.get() && self.bound.get().is_none(),
            "operational case binding order",
        )?;
        self.work
            .bind_train(case, request)
            .map_err(|_| ProvenanceError("operational actual case/request mismatch"))?;
        self.bound
            .set(BoundWork::Train {
                canonical: case.canonical_bytes().to_vec(),
                digest: case.digest().into(),
                request: self
                    .work
                    .bound_request()
                    .map_err(|_| ProvenanceError("bound Train request encoding"))?,
                profile: case.selector().profile_id.clone(),
                selected: case.selector().seed,
            })
            .map_err(|_| ProvenanceError("duplicate operational binding"))?;
        Ok(())
    }
    pub(crate) fn bind_preparation(
        &self,
        operation: u8,
        profile: &str,
        seeds: &[u64],
    ) -> Result<()> {
        self.check()?;
        ensure(
            self.context_bound.get() && self.bound.get().is_none(),
            "operational preparation binding order",
        )?;
        self.work
            .bind_preparation(operation, profile, seeds)
            .map_err(|_| ProvenanceError("operational preparation mismatch"))?;
        self.bound
            .set(BoundWork::Preparation {
                operation,
                profile: profile.into(),
                seeds: seeds.to_vec(),
            })
            .map_err(|_| ProvenanceError("duplicate operational preparation binding"))?;
        Ok(())
    }
    /// The actual registry-derived bytes and digest remain attached to this live
    /// owner alongside its exact immutable request, for Step3/4 producers.
    pub(crate) fn bound_train(&self) -> Result<(&[u8], &str, &[u8])> {
        self.check()?;
        let Some(BoundWork::Train {
            canonical,
            digest,
            request,
            ..
        }) = self.bound.get()
        else {
            return Err(ProvenanceError("live bound Train required"));
        };
        ensure(
            crate::data::sha256_hex(canonical) == *digest,
            "bound registry case digest changed",
        )?;
        Ok((canonical, digest, request))
    }
    pub(crate) fn preparation_scope(&self) -> Result<ResourceBorrow<'_>> {
        self.check()?;
        let Some(BoundWork::Preparation {
            operation,
            profile,
            seeds,
        }) = self.bound.get()
        else {
            return Err(ProvenanceError("bound preparation root required"));
        };
        self.work
            .bind_preparation(*operation, profile, seeds)
            .map_err(|_| ProvenanceError("bound preparation changed"))?;
        Ok(ResourceBorrow {
            owner: self,
            operation: *operation,
            parent: None,
            profile,
            selected: *seeds.last().ok_or(ProvenanceError("bound seeds absent"))?,
        })
    }
    #[allow(dead_code)] // Step3 retained Target reader entry, not a completed I/O route.
    pub(crate) fn train_target_scope(&self) -> Result<ResourceBorrow<'_>> {
        self.bound_train()?;
        let Some(BoundWork::Train {
            profile, selected, ..
        }) = self.bound.get()
        else {
            return Err(ProvenanceError("bound Train required"));
        };
        self.work
            .child(6, 3, profile, *selected)
            .map_err(|_| ProvenanceError("Quick/out-of-subtree target borrow"))?;
        Ok(ResourceBorrow {
            owner: self,
            operation: 3,
            parent: Some(6),
            profile,
            selected: *selected,
        })
    }
    pub(crate) fn before_data(&self) -> Result<()> {
        self.check()?;
        ensure(
            self.bound.get().is_some(),
            "data expansion before actual request binding",
        )?;
        ensure(
            self.roots.contains(&0) && self.roots.contains(&2),
            "Input/Reviewed retained roots not yet acquired under this owner",
        )
    }
}
/// Private live borrow only: no constructor from records, bytes, or caller IDs.
/// Step3 connects these borrows to actual retained readers; no data bypass here.
pub(crate) struct ResourceBorrow<'a> {
    owner: &'a WorkerOwner,
    operation: u8,
    #[allow(dead_code)] // Step3 snapshot-origin association.
    parent: Option<u8>,
    profile: &'a str,
    selected: u64,
}
impl<'a> ResourceBorrow<'a> {
    pub(crate) fn before_read(
        &self,
        context: &crate::study_provenance::VerifiedStudyContext,
        operation: u8,
        profile: &str,
        selected: u64,
    ) -> Result<()> {
        self.owner.before_data()?;
        ensure(
            self.operation == operation && self.profile == profile && self.selected == selected,
            "borrowed operation/profile/seed mismatch",
        )?;
        context.recheck_resource_broker(&self.owner.broker)
    }
    #[allow(dead_code)] // Step3 private reader child edge.
    pub(crate) fn snapshot(&self) -> Result<ResourceBorrow<'a>> {
        self.owner.check()?;
        self.owner
            .work
            .child(self.operation, 2, self.profile, self.selected)
            .map_err(|_| ProvenanceError("prohibited borrowed snapshot edge"))?;
        Ok(Self {
            owner: self.owner,
            operation: 2,
            parent: Some(self.operation),
            profile: self.profile,
            selected: 0,
        })
    }
    #[allow(dead_code)] // Step3 consuming lock_target association.
    pub(crate) fn same_lock_snapshot(&self, snapshot: &ResourceBorrow<'_>) -> Result<()> {
        self.owner.check()?;
        snapshot.owner.check()?;
        ensure(
            self.operation == 5
                && snapshot.operation == 2
                && snapshot.parent == Some(5)
                && std::ptr::eq(self.owner, snapshot.owner)
                && self.profile == snapshot.profile
                && self.selected == snapshot.selected,
            "foreign/out-of-subtree root5 snapshot",
        )
    }
}
impl Drop for WorkerOwner {
    fn drop(&mut self) {
        if let Ok(mut state) = self.broker.state.try_borrow_mut() {
            state.active = false;
        }
    }
}
fn field(out: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    ensure(
        bytes.len() <= 4096 && !bytes.contains(&0),
        "broker argument bounds",
    )?;
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}
#[cfg(target_os = "linux")]
impl State {
    fn check_parent(&self) -> Result<()> {
        self.parent.recheck()?;
        if let Some(origin) = self.probe_origin {
            ensure(
                super::protocol::elapsed(origin, super::native::monotonic()?)? <= 60_000_000_000,
                "native broker deadline",
            )?;
        }
        Ok(())
    }
}
impl Broker {
    pub(crate) fn same_origin(&self, other: &Self) -> Result<()> {
        self.check()?;
        other.check()?;
        ensure(
            Rc::ptr_eq(&self.state, &other.state),
            "foreign live operational broker",
        )
    }
    #[cfg(target_os = "linux")]
    pub(super) fn authenticated_probe(
        channel: Channel,
        parent: super::handshake::ParentGuard,
        origin: u64,
    ) -> Self {
        Self {
            state: Rc::new(RefCell::new(State {
                channel,
                active: true,
                parent,
                probe_origin: Some(origin),
            })),
        }
    }
    #[cfg(target_os = "linux")]
    pub(super) fn into_probe_transport(self) -> Result<(Channel, super::handshake::ParentGuard)> {
        let state = Rc::try_unwrap(self.state)
            .map_err(|_| ProvenanceError("native broker unexpectedly shared"))?
            .into_inner();
        ensure(
            state.active && state.probe_origin.is_some(),
            "non-probe transport extraction forbidden",
        )?;
        state.check_parent()?;
        Ok((state.channel, state.parent))
    }
    pub(crate) fn check(&self) -> Result<()> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| ProvenanceError("nested broker operation"))?;
        ensure(state.active, "operational owner expired")?;
        #[cfg(target_os = "linux")]
        state.check_parent()?;
        let frames = state.channel.turn()?;
        if state.channel.eof || !frames.is_empty() {
            state.active = false;
            return Err(ProvenanceError(
                "supervisor EOF/stop or unsolicited control",
            ));
        }
        Ok(())
    }
    pub(crate) fn command(&self, command: Command, cap: usize) -> Result<Vec<u8>> {
        ensure(cap <= super::protocol::PAYLOAD_CAP, "broker output bound")?;
        let mut payload = vec![0]; // Worker requests only pinned Python; Python requests Git kind1.
        field(
            &mut payload,
            command
                .get_current_dir()
                .ok_or(ProvenanceError("helper fixed root"))?
                .as_os_str()
                .as_bytes(),
        )?;
        let args = std::iter::once(command.get_program())
            .chain(command.get_args())
            .collect::<Vec<_>>();
        ensure(args.len() <= 256, "broker argv count")?;
        payload.extend_from_slice(&(args.len() as u64).to_le_bytes());
        for arg in args {
            field(&mut payload, arg.as_bytes())?;
        }
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| ProvenanceError("nested broker request"))?;
        ensure(state.active, "expired broker")?;
        state.channel.send(Tag::HelperRequest, payload)?;
        loop {
            #[cfg(target_os = "linux")]
            state.check_parent()?;
            let frames = state.channel.turn()?;
            if state.channel.eof {
                state.active = false;
                return Err(ProvenanceError("supervisor lost during helper"));
            }
            if let Some(frame) = frames.into_iter().next() {
                if frame.tag == Tag::Stop {
                    state.active = false;
                    return Err(ProvenanceError("supervisor stop"));
                }
                ensure(
                    frame.tag == Tag::HelperReply && frame.payload.len() >= 8,
                    "broker helper reply order",
                )?;
                let status = u64::from_le_bytes(
                    frame.payload[..8]
                        .try_into()
                        .map_err(|_| ProvenanceError("helper status"))?,
                );
                ensure(
                    status == 0 && frame.payload.len() - 8 <= cap,
                    "broker helper failed/oversized",
                )?;
                return Ok(frame.payload[8..].to_vec());
            }
            wait_channel(&state.channel)?;
        }
    }
}
#[cfg(target_os = "linux")]
pub(super) fn wait_channel(channel: &Channel) -> Result<()> {
    use rustix::event::{PollFd, PollFlags};
    let mut flags = PollFlags::IN;
    if channel.pending() {
        flags |= PollFlags::OUT;
    }
    let mut fds = [PollFd::new(&channel.stream, flags)];
    super::native::io(rustix::event::poll(
        &mut fds,
        Some(&rustix::time::Timespec {
            tv_sec: 0,
            tv_nsec: 100_000_000,
        }),
    ))?;
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn wait_channel(_: &Channel) -> Result<()> {
    Err(ProvenanceError("live broker requires native Linux"))
}
