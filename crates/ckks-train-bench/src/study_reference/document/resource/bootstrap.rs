//! Minimal pre-capacity anchor transport. Both successful and failed parsing use
//! borrowed text, fixed arrays and static diagnostics; no String/Vec/Encoder.
use super::*;
const HEADER: &[u8] = b"hekit-study-reference-encoding-v1";
const PRODUCTION: &[u8] = b"hekit-study-resource-bootstrap-v1";
const ORACLE: &[u8] = b"hekit-test-resource-bootstrap-v1";
fn error(reason: &'static str, field: &'static str) -> SourceError {
    SourceError {
        reason,
        field: std::borrow::Cow::Borrowed(field),
    }
}
fn check(ok: bool, reason: &'static str, field: &'static str) -> SResult<()> {
    if ok {
        Ok(())
    } else {
        Err(error(reason, field))
    }
}
struct Reader<'a> {
    bytes: &'a [u8],
    i: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> SResult<&'a [u8]> {
        let end = self
            .i
            .checked_add(n)
            .ok_or_else(|| error("bootstrap-length", "bootstrap"))?;
        let b = self
            .bytes
            .get(self.i..end)
            .ok_or_else(|| error("bootstrap-length", "bootstrap"))?;
        self.i = end;
        Ok(b)
    }
    fn number(&mut self) -> SResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| error("bootstrap-length", "bootstrap"))?,
        ))
    }
    fn literal(&mut self, s: &[u8]) -> SResult<()> {
        check(
            self.number()? == s.len() as u64,
            "bootstrap-domain",
            "bootstrap.domain",
        )?;
        check(
            self.take(s.len())? == s,
            "bootstrap-domain",
            "bootstrap.domain",
        )
    }
    fn hash(&mut self) -> SResult<&'a str> {
        check(
            self.number()? == 64,
            "bootstrap-hash-prefix",
            "bootstrap.hash",
        )?;
        let b = self.take(64)?;
        check(
            b.iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b)),
            "bootstrap-hash",
            "bootstrap.hash",
        )?;
        std::str::from_utf8(b).map_err(|_| error("bootstrap-hash", "bootstrap.hash"))
    }
    fn numbers<const N: usize>(&mut self) -> SResult<[u64; N]> {
        let mut v = [0; N];
        for n in &mut v {
            *n = self.number()?;
        }
        Ok(v)
    }
}
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Bootstrap<'a> {
    pub(super) mode: Mode,
    pub(super) generation: u64,
    pub(super) origin_evidence: &'a str,
    pub(super) policy: &'a str,
    pub(super) boot_uuid: &'a str,
    pub(super) account_uid: u64,
    pub(super) host_key: &'a str,
    pub(super) namespaces: [[u64; 2]; 4],
    pub(super) proc_root: [u64; 4],
    pub(super) cgroup_root: [u64; 4],
    pub(super) kernel_profile: &'a str,
    pub(super) allocator_profile: &'a str,
    pub(super) filesystem_profile: &'a str,
}
impl<'a> Bootstrap<'a> {
    pub(super) fn parse(bytes: &'a [u8], mode: Mode) -> SResult<Self> {
        let domain = if mode == Mode::Production {
            PRODUCTION
        } else {
            ORACLE
        };
        let len = if mode == Mode::Production { 704 } else { 703 };
        check(bytes.len() == len, "bootstrap-length", "bootstrap.length")?;
        let mut c = Reader { bytes, i: 0 };
        c.literal(HEADER)?;
        c.literal(domain)?;
        check(
            c.take(1)? == [u8::from(mode.synthetic())],
            "bootstrap-domain",
            "bootstrap.synthetic",
        )?;
        let generation = c.number()?;
        let origin_evidence = c.hash()?;
        let policy = c.hash()?;
        check(c.number()? == 36, "bootstrap-uuid", "bootstrap.boot_uuid")?;
        let boot_uuid = std::str::from_utf8(c.take(36)?)
            .map_err(|_| error("bootstrap-uuid", "bootstrap.boot_uuid"))?;
        check(
            valid_uuid(boot_uuid),
            "bootstrap-uuid",
            "bootstrap.boot_uuid",
        )?;
        let account_uid = c.number()?;
        let host_key = c.hash()?;
        let namespaces = [c.numbers()?, c.numbers()?, c.numbers()?, c.numbers()?];
        let proc_root = c.numbers()?;
        let cgroup_root = c.numbers()?;
        let kernel_profile = c.hash()?;
        let allocator_profile = c.hash()?;
        let filesystem_profile = c.hash()?;
        check(
            c.take(1)? == [0],
            "bootstrap-domain",
            "bootstrap.claim_domain",
        )?;
        check(c.i == bytes.len(), "bootstrap-length", "bootstrap.eof")?;
        Ok(Self {
            mode,
            generation,
            origin_evidence,
            policy,
            boot_uuid,
            account_uid,
            host_key,
            namespaces,
            proc_root,
            cgroup_root,
            kernel_profile,
            allocator_profile,
            filesystem_profile,
        })
    }
    pub(super) fn encode_fixed(&self) -> SResult<FixedBytes> {
        let mut out = FixedBytes {
            bytes: [0; 704],
            len: 0,
        };
        out.text(HEADER)?;
        out.text(if self.mode == Mode::Production {
            PRODUCTION
        } else {
            ORACLE
        })?;
        out.raw(&[u8::from(self.mode.synthetic())])?;
        out.number(self.generation)?;
        out.text(self.origin_evidence.as_bytes())?;
        out.text(self.policy.as_bytes())?;
        out.text(self.boot_uuid.as_bytes())?;
        out.number(self.account_uid)?;
        out.text(self.host_key.as_bytes())?;
        for p in self.namespaces {
            for n in p {
                out.number(n)?;
            }
        }
        for n in self.proc_root {
            out.number(n)?;
        }
        for n in self.cgroup_root {
            out.number(n)?;
        }
        out.text(self.kernel_profile.as_bytes())?;
        out.text(self.allocator_profile.as_bytes())?;
        out.text(self.filesystem_profile.as_bytes())?;
        out.raw(&[0])?;
        // Reapply all literal/primitive/EOF constraints without heap storage.
        Bootstrap::parse(out.as_slice(), self.mode)?;
        Ok(out)
    }
}
pub(super) struct FixedBytes {
    bytes: [u8; 704],
    len: usize,
}
impl FixedBytes {
    pub(super) fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
    fn raw(&mut self, b: &[u8]) -> SResult<()> {
        let end = self
            .len
            .checked_add(b.len())
            .filter(|n| *n <= 704)
            .ok_or_else(|| error("bootstrap-length", "bootstrap.encoding"))?;
        self.bytes[self.len..end].copy_from_slice(b);
        self.len = end;
        Ok(())
    }
    fn number(&mut self, n: u64) -> SResult<()> {
        self.raw(&n.to_le_bytes())
    }
    fn text(&mut self, s: &[u8]) -> SResult<()> {
        self.number(s.len() as u64)?;
        self.raw(s)
    }
}
