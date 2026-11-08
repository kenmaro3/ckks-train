//! Closed native-only P->M->L transfer. No production entry consumes rights.
use super::{handshake::Projection, schedule::Token};
use crate::study_provenance::{ProvenanceError, Result, ensure};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Boundary {
    RescueToMiddle,
    RescueToLeaf,
    CustodyToOuter,
}
impl Boundary {
    pub fn rights(self) -> usize {
        if self == Self::CustodyToOuter { 2 } else { 1 }
    }
    pub fn marker(self) -> u8 {
        match self {
            Self::RescueToMiddle => b'M',
            Self::RescueToLeaf => b'L',
            Self::CustodyToOuter => b'C',
        }
    }
    pub fn validate(self, scenario: u8, marker: u8, count: usize, truncated: bool) -> Result<()> {
        ensure(
            marker == self.marker()
                && count == self.rights()
                && !truncated
                && (scenario == 6 || (scenario == 1 && self == Self::CustodyToOuter)),
            "native transfer kind/direction/scenario/count/truncation",
        )
    }
}
pub(super) fn rescue_record(
    boundary: Boundary,
    outer: &Projection,
    inner: Option<&Projection>,
) -> Result<Vec<u8>> {
    ensure(
        boundary != Boundary::CustodyToOuter
            && (boundary == Boundary::RescueToLeaf) == inner.is_some(),
        "rescue fixed direction/projection",
    )?;
    let mut raw = if boundary == Boundary::RescueToMiddle {
        b"HKRSM1".to_vec()
    } else {
        b"HKRSL1".to_vec()
    };
    raw.push(6);
    raw.extend(outer.encode());
    if let Some(inner) = inner {
        ensure(
            inner.parent == outer.child
                && inner.child.pid != outer.parent.pid
                && inner.lock == outer.lock
                && inner.origin == outer.origin
                && inner.boot == outer.boot
                && inner.executable == outer.executable,
            "rescue authenticated chain",
        )?;
        raw.extend(inner.encode());
    }
    Ok(raw)
}
pub(super) fn validate_rescue(
    raw: &[u8],
    boundary: Boundary,
    outer: &Projection,
    inner: Option<&Projection>,
) -> Result<()> {
    ensure(
        raw == rescue_record(boundary, outer, inner)?,
        "rescue original session/type/direction",
    )
}
pub(super) struct TransferProjection {
    pub scenario: u8,
    pub rights: u8,
    pub outer: Token,
    pub inner: Projection,
    pub session: [u8; 64],
}
impl TransferProjection {
    pub const LENGTH: usize = 314;
    pub fn decode(bytes: &[u8], outer: &Projection, scenario: u8) -> Result<Self> {
        ensure(
            bytes.len() == Self::LENGTH
                && &bytes[..6] == b"HKFDS1"
                && [1, 6].contains(&scenario)
                && bytes[6] == scenario
                && bytes[7] == 2,
            "probe transfer exact schema/scenario/rights",
        )?;
        let pid = u64::from_le_bytes(
            bytes[8..16]
                .try_into()
                .map_err(|_| ProvenanceError("transfer PID"))?,
        );
        let start = u64::from_le_bytes(
            bytes[16..24]
                .try_into()
                .map_err(|_| ProvenanceError("transfer start"))?,
        );
        let inner = Projection::decode(&bytes[24..250])?;
        ensure(
            pid == u64::from(outer.parent.pid)
                && start == outer.parent.start
                && inner.parent == outer.child
                && inner.child.pid != outer.parent.pid
                && inner.lock == outer.lock
                && inner.origin == outer.origin
                && inner.boot == outer.boot
                && inner.executable == outer.executable
                && bytes[250..] == outer.invocation,
            "probe transfer actual session",
        )?;
        Ok(Self {
            scenario,
            rights: bytes[7],
            outer: outer.parent,
            inner,
            session: outer.invocation,
        })
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut b = b"HKFDS1".to_vec();
        b.extend([self.scenario, self.rights]);
        b.extend(u64::from(self.outer.pid).to_le_bytes());
        b.extend(self.outer.start.to_le_bytes());
        b.extend(self.inner.encode());
        b.extend(self.session);
        b
    }
}
#[cfg(test)]
pub(super) struct Handoff {
    phase: u8,
    failed: bool,
}
#[cfg(test)]
impl Handoff {
    pub fn new() -> Self {
        Self {
            phase: 0,
            failed: false,
        }
    }
    fn advance(&mut self, from: u8, valid: bool) -> Result<()> {
        if self.failed || self.phase != from || !valid {
            self.failed = true;
            return Err(ProvenanceError("native transfer custody transition"));
        }
        self.phase += 1;
        Ok(())
    }
    pub fn sent(&mut self) -> Result<()> {
        self.advance(0, true)
    }
    pub fn received(
        &mut self,
        count: usize,
        expected: usize,
        truncated: bool,
        duplicate: bool,
    ) -> Result<()> {
        self.advance(
            1,
            count == expected && expected == 2 && !truncated && !duplicate,
        )
    }
    pub fn ack(&mut self) -> Result<()> {
        self.advance(2, true)
    }
    pub fn adopted(&mut self, actual_ppid_transition_same_identity: bool) -> Result<()> {
        self.advance(3, actual_ppid_transition_same_identity)
    }
    pub fn can_wait(&self) -> bool {
        !self.failed && self.phase == 4
    }
    pub fn reaped(&mut self) -> Result<()> {
        self.advance(4, true)
    }
    pub fn custody_unresolved(&self) -> bool {
        self.failed || self.phase != 5
    }
}
#[cfg(target_os = "linux")]
pub(super) fn send_rights(
    stream: &std::os::unix::net::UnixStream,
    boundary: Boundary,
    scenario: u8,
    rights: &[std::os::fd::BorrowedFd<'_>],
) -> Result<bool> {
    use rustix::net::{SendAncillaryBuffer, SendAncillaryMessage, SendFlags};
    boundary.validate(scenario, boundary.marker(), rights.len(), false)?;
    let mut space = [std::mem::MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(3))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    ensure(
        control.push(SendAncillaryMessage::ScmRights(rights)),
        "probe ancillary buffer",
    )?;
    let marker = [boundary.marker()];
    match rustix::net::sendmsg(
        stream,
        &[std::io::IoSlice::new(&marker)],
        &mut control,
        SendFlags::DONTWAIT | SendFlags::NOSIGNAL,
    ) {
        Ok(1) => Ok(true),
        Ok(_) => Err(ProvenanceError(
            "partial probe rights send remains unresolved",
        )),
        Err(rustix::io::Errno::AGAIN) => Ok(false),
        Err(e) => Err(std::io::Error::from(e).into()),
    }
}
#[cfg(target_os = "linux")]
pub(super) fn receive_rights(
    stream: &std::os::unix::net::UnixStream,
    boundary: Boundary,
    scenario: u8,
) -> std::result::Result<Option<Vec<std::os::fd::OwnedFd>>, RightsFailure> {
    use rustix::net::{RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags};
    let expected = boundary.rights();
    boundary.validate(scenario, boundary.marker(), expected, false)?;
    let mut space = [std::mem::MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(3))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let mut byte = [0];
    let result = match rustix::net::recvmsg(
        stream,
        &mut [std::io::IoSliceMut::new(&mut byte)],
        &mut control,
        RecvFlags::DONTWAIT | RecvFlags::CMSG_CLOEXEC,
    ) {
        Ok(v) => v,
        Err(rustix::io::Errno::AGAIN) => return Ok(None),
        Err(e) => return Err(ProvenanceError::from(std::io::Error::from(e)).into()),
    };
    let mut rights = Vec::new();
    let mut messages = 0;
    let mut wrong_type = false;
    for message in control.drain() {
        match message {
            RecvAncillaryMessage::ScmRights(fds) => {
                messages += 1;
                rights.extend(fds);
            }
            _ => wrong_type = true,
        }
    }
    if result.bytes != 1
        || boundary
            .validate(
                scenario,
                byte[0],
                rights.len(),
                result
                    .flags
                    .intersects(ReturnFlags::CTRUNC | ReturnFlags::TRUNC),
            )
            .is_err()
        || messages != 1
        || rights.len() != expected
        || wrong_type
    {
        return Err(RightsFailure {
            error: ProvenanceError(
                "probe malformed/partial/truncated/duplicate rights; custody retained",
            ),
            rights,
        });
    }
    Ok(Some(rights))
}

#[cfg(target_os = "linux")]
pub(super) struct RightsFailure {
    pub error: ProvenanceError,
    pub rights: Vec<std::os::fd::OwnedFd>,
}
#[cfg(target_os = "linux")]
impl From<ProvenanceError> for RightsFailure {
    fn from(error: ProvenanceError) -> Self {
        Self {
            error,
            rights: Vec::new(),
        }
    }
}
