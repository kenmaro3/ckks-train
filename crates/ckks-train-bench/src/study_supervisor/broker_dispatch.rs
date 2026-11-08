//! Shared actual broker dispatch for production and isolated native probes.
use super::{
    monitor::{Kind, Monitor},
    protocol::{self, HelperRequest, Tag},
    schedule::{Event, Token},
};
use crate::study_provenance::{ProvenanceError, Result, RuntimeBrokerPolicy, ensure};
use std::os::unix::fs::MetadataExt;
struct Helper {
    token: Token,
    requester: Token,
    run: Vec<u8>,
}
pub(super) struct BrokerDispatch {
    helpers: Vec<Helper>,
}
impl BrokerDispatch {
    pub fn new() -> Self {
        Self {
            helpers: Vec::new(),
        }
    }
    pub fn active(&self) -> usize {
        self.helpers.len()
    }
    pub fn payload(&self, token: Token) -> Result<&[u8]> {
        self.helpers
            .iter()
            .find(|h| h.token == token)
            .map(|h| h.run.as_slice())
            .ok_or(ProvenanceError("helper Run payload missing"))
    }
    pub fn request(
        &mut self,
        monitor: &mut Monitor,
        build: &RuntimeBrokerPolicy,
        token: Token,
        payload: &[u8],
    ) -> Result<()> {
        let request = HelperRequest::decode(payload)?;
        let requester = monitor.child_mut(token)?;
        ensure(
            requester.phase == 4,
            "broker request before Run acknowledgement",
        )?;
        let kind = match requester.kind {
            Kind::Worker => 0,
            Kind::Python => 1,
            Kind::Git => 2,
        };
        protocol::broker_edge(kind, request.kind, self.helpers.len())?;
        let metadata = monitor.lock.metadata()?;
        let command = build.command(
            request.kind,
            &request.cwd,
            &request.argv,
            (metadata.dev(), metadata.ino()),
            monitor.process.own_token.pid,
        )?;
        let run = protocol::encode_exec(&command)?;
        let kind = if request.kind == 0 {
            Kind::Python
        } else {
            Kind::Git
        };
        let mode = if kind == Kind::Python {
            "python"
        } else {
            "git"
        };
        let child = monitor.spawn(
            kind,
            &std::env::current_exe()?,
            "--private-helper",
            &[mode.into()],
            &build.executable_hash,
        )?;
        self.helpers.push(Helper {
            token: child,
            requester: token,
            run,
        });
        monitor.hello(child)
    }
    pub fn result(&mut self, monitor: &mut Monitor, token: Token, payload: Vec<u8>) -> Result<()> {
        ensure(
            !self.helpers.iter().any(|h| h.requester == token),
            "Python result before nested helper completion",
        )?;
        let c = monitor.child_mut(token)?;
        ensure(
            c.kind == Kind::Python
                && c.phase == 4
                && payload.len() >= 8
                && payload.len() <= protocol::PAYLOAD_CAP
                && c.result.is_none(),
            "Python result bounds/order/duplicate",
        )?;
        c.result = Some(payload);
        c.phase = 5; // Validated terminal result; no further framed requests allowed.
        Ok(())
    }
    pub fn exited(&mut self, monitor: &mut Monitor, token: Token, status: u64) -> Result<()> {
        let c = monitor.child_mut(token)?;
        ensure(
            c.kind != Kind::Worker && c.drain.ready() && status == 0,
            "helper exit/drain/status",
        )?;
        let reply = if c.kind == Kind::Git {
            let mut v = 0u64.to_le_bytes().to_vec();
            v.append(&mut c.raw);
            v
        } else {
            c.result
                .take()
                .ok_or(ProvenanceError("Python exited without result"))?
        };
        ensure(
            reply.len() >= 8 && reply[..8] == [0; 8],
            "helper returned failure",
        )?;
        let index = self
            .helpers
            .iter()
            .position(|h| h.token == token)
            .ok_or(ProvenanceError("helper requester missing"))?;
        let helper = self.helpers.remove(index);
        monitor.event(Event::HelperComplete(token))?;
        monitor
            .child_mut(helper.requester)?
            .child
            .channel
            .send(Tag::HelperReply, reply)
    }
}
