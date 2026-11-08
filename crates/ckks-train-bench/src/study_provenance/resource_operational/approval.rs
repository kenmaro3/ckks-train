//! Fixed-storage approval-v3 projection before allocating context acquisition.
use crate::data::Sha256State;
use crate::study_provenance::{ProvenanceError, Result, ensure};
use std::io::Read;

const CAP: usize = 8 * 1024 * 1024;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Fixed<const N: usize> {
    bytes: [u8; N],
    len: usize,
}
impl<const N: usize> Default for Fixed<N> {
    fn default() -> Self {
        Self {
            bytes: [0; N],
            len: 0,
        }
    }
}
impl<const N: usize> Fixed<N> {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
    pub(crate) fn text(&self) -> Result<&str> {
        std::str::from_utf8(self.as_bytes()).map_err(|_| ProvenanceError("origin UTF8"))
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct OriginRef {
    pub(crate) root: u8,
    pub(crate) path: Fixed<4096>,
    pub(crate) length: u64,
    pub(crate) hash: Fixed<64>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct OriginProjection {
    pub(crate) account_uid: u64,
    pub(crate) account_anchor: Fixed<4096>,
    pub(crate) account_identity: (u64, u64),
    pub(crate) policy_origin: Fixed<4096>,
    pub(crate) policy_identity: (u64, u64),
    pub(crate) observation_origin: Fixed<4096>,
    pub(crate) observation_identity: (u64, u64),
    pub(crate) bootstrap: Fixed<64>,
    pub(crate) physical_host: Fixed<64>,
    pub(crate) host_origin: OriginRef,
    pub(crate) namespace_origin: OriginRef,
    pub(crate) native_validation: OriginRef,
}
#[derive(Clone, Debug, Default)]
pub(crate) struct BuildPin {
    pub(crate) role: u8,
    pub(crate) recipe: Fixed<128>,
    pub(crate) algorithm: Fixed<128>,
    pub(crate) receipt: Fixed<64>,
    pub(crate) capsule: Fixed<64>,
    pub(crate) executable: Fixed<64>,
}
pub(crate) struct ScannedApproval {
    pub(crate) body_end: usize,
    pub(crate) source: Fixed<64>,
    pub(crate) revision: Fixed<40>,
    pub(crate) origin: OriginProjection,
    pub(crate) builds: [BuildPin; 64],
    pub(crate) build_count: usize,
}
pub(super) struct Scan<'a, R> {
    reader: &'a mut R,
    digest: Sha256State,
    used: usize,
}
impl<'a, R: Read> Scan<'a, R> {
    pub(super) fn new(reader: &'a mut R) -> Self {
        Self {
            reader,
            digest: Sha256State::new(),
            used: 0,
        }
    }
    pub(super) fn done(&mut self) -> Result<()> {
        let mut b = [0];
        ensure(self.reader.read(&mut b)? == 0, "origin trailing bytes")
    }

    fn read(&mut self, out: &mut [u8]) -> Result<()> {
        self.used = self
            .used
            .checked_add(out.len())
            .ok_or(ProvenanceError("origin size overflow"))?;
        ensure(self.used <= CAP, "origin approval cap")?;
        self.reader.read_exact(out)?;
        self.digest.update(out);
        Ok(())
    }
    pub(super) fn number(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        self.read(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    pub(super) fn tag(&mut self, max: u8) -> Result<u8> {
        let mut b = [0];
        self.read(&mut b)?;
        ensure(b[0] <= max, "origin tag")?;
        Ok(b[0])
    }
    pub(super) fn bytes<const N: usize>(&mut self) -> Result<Fixed<N>> {
        let len = usize::try_from(self.number()?)
            .map_err(|_| ProvenanceError("origin length overflow"))?;
        ensure(len <= N, "origin field cap")?;
        let mut value = Fixed { bytes: [0; N], len };
        self.read(&mut value.bytes[..len])?;
        Ok(value)
    }
    pub(super) fn text<const N: usize>(&mut self) -> Result<Fixed<N>> {
        let value = self.bytes()?;
        value.text()?;
        Ok(value)
    }
    pub(super) fn hash(&mut self) -> Result<Fixed<64>> {
        let v = self.text()?;
        ensure(hex(v.as_bytes(), 64), "origin SHA256")?;
        Ok(v)
    }
    pub(super) fn count(&mut self, cap: usize) -> Result<usize> {
        let n = usize::try_from(self.number()?)
            .map_err(|_| ProvenanceError("origin count overflow"))?;
        ensure(n <= cap, "origin count cap")?;
        Ok(n)
    }
    pub(super) fn path(&mut self) -> Result<Fixed<4096>> {
        let v = self.bytes()?;
        ensure(absolute(v.as_bytes()), "origin absolute path")?;
        Ok(v)
    }
    pub(super) fn identity(&mut self) -> Result<(u64, u64)> {
        let pair = (self.number()?, self.number()?);
        ensure(pair.0 > 0 && pair.1 > 0, "origin empty identity")?;
        Ok(pair)
    }
    pub(super) fn reference(&mut self) -> Result<OriginRef> {
        let root = self.tag(5)?;
        let path = self.text::<4096>()?;
        ensure(
            super::super::codec::relative(path.as_bytes()),
            "origin relative reference",
        )?;
        let length = self.number()?;
        ensure(length <= CAP as u64, "origin referenced cap")?;
        Ok(OriginRef {
            root,
            path,
            length,
            hash: self.hash()?,
        })
    }
}
pub(crate) fn absolute(b: &[u8]) -> bool {
    b.starts_with(b"/")
        && b.len() > 1
        && !b.contains(&0)
        && !b.contains(&b'\\')
        && b[1..]
            .split(|v| *v == b'/')
            .all(|v| !v.is_empty() && v != b"." && v != b"..")
}
fn hex(b: &[u8], n: usize) -> bool {
    b.len() == n
        && b.iter()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(v))
}
pub(crate) fn scan_approval(reader: &mut impl Read, expected: &str) -> Result<ScannedApproval> {
    ensure(
        hex(expected.as_bytes(), 64),
        "origin expected approval hash",
    )?;
    let mut s = Scan {
        reader,
        digest: Sha256State::new(),
        used: 0,
    };
    ensure(
        s.text::<128>()?.as_bytes() == b"hekit-study-reference-encoding-v1"
            && s.text::<128>()?.as_bytes() == b"hekit-study-local-approval-v3",
        "observed owner requires approval v3",
    )?;
    ensure(
        !s.text::<4096>()?.as_bytes().is_empty(),
        "origin approval id",
    )?;
    let mut prior = Fixed::<4096>::default();
    for _ in 0..s.count(32)? {
        let path = s.text::<4096>()?;
        ensure(prior.as_bytes() < path.as_bytes(), "origin design order")?;
        s.hash()?;
        prior = path;
    }
    let revision = s.text::<40>()?;
    ensure(hex(revision.as_bytes(), 40), "origin source revision")?;
    let source = s.hash()?;
    let build_count = s.count(64)?;
    ensure(build_count > 0, "origin empty build pins")?;
    let mut builds = std::array::from_fn(|_| BuildPin::default());
    let mut previous_receipt = Fixed::<64>::default();
    for pin in &mut builds[..build_count] {
        pin.role = s.tag(1)?;
        pin.recipe = s.text()?;
        pin.algorithm = s.text()?;
        let recipe = pin.recipe.text()?;
        let is_reference = matches!(
            recipe,
            "study-reference-linux-clean-v1" | "study-reference-linux-clean-v2"
        );
        ensure(
            is_reference
                || matches!(
                    recipe,
                    "study-train-cpu-serial-linux-clean-v1"
                        | "study-train-cpu-serial-linux-clean-v2"
                        | "study-train-cpu-rayon-linux-clean-v1"
                        | "study-train-cuda-host-linux-clean-v1"
                ),
            "origin build recipe",
        )?;
        ensure(
            pin.role == u8::from(!is_reference)
                && pin.algorithm.text()?
                    == if is_reference {
                        "study-reference-session-v1"
                    } else {
                        "study-train-session-v1"
                    },
            "origin build role/algorithm",
        )?;
        pin.receipt = s.hash()?;
        pin.capsule = s.hash()?;
        pin.executable = s.hash()?;
        ensure(
            previous_receipt.as_bytes() < pin.receipt.as_bytes(),
            "origin build order",
        )?;
        previous_receipt = pin.receipt.clone();
    }
    let body_end = s.used;
    let origin = OriginProjection {
        account_uid: s.number()?,
        account_anchor: s.path()?,
        account_identity: s.identity()?,
        policy_origin: s.path()?,
        policy_identity: s.identity()?,
        observation_origin: s.path()?,
        observation_identity: s.identity()?,
        bootstrap: s.hash()?,
        physical_host: s.hash()?,
        host_origin: s.reference()?,
        namespace_origin: s.reference()?,
        native_validation: s.reference()?,
    };
    ensure(
        origin.host_origin == origin.namespace_origin,
        "origin host/namespace join",
    )?;
    let mut extra = [0];
    ensure(s.reader.read(&mut extra)? == 0, "origin trailing bytes")?;
    ensure(
        s.digest.finish_fixed() == expected.as_bytes(),
        "origin approval hash mismatch",
    )?;
    Ok(ScannedApproval {
        body_end,
        source,
        revision,
        origin,
        builds,
        build_count,
    })
}
