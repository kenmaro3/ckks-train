//! Acyclic test-only operator origin. Never convertible into a production owner.
use super::approval::{Fixed, OriginRef, Scan};
use crate::study_provenance::{Result, ensure};
pub(crate) struct ProbeOrigin {
    pub source: Fixed<64>,
    pub revision: Fixed<40>,
    pub executable: Fixed<64>,
    pub recipe: Fixed<128>,
    pub receipt: OriginRef,
    pub host: OriginRef,
    pub account_uid: u64,
    pub account_path: Fixed<4096>,
    pub account_identity: (u64, u64),
    pub policy_path: Fixed<4096>,
    pub policy_identity: (u64, u64),
    pub observation_path: Fixed<4096>,
    pub observation_identity: (u64, u64),
    pub probe_id: Fixed<128>,
}
pub(crate) fn parse_probe(bytes: &[u8], oracle: bool) -> Result<ProbeOrigin> {
    ensure(
        bytes.len() <= 4096 && (!oracle || cfg!(test)),
        "test-only origin cap/domain",
    )?;
    let mut input = std::io::Cursor::new(bytes);
    let mut r = Scan::new(&mut input);
    ensure(
        r.text::<128>()?.as_bytes() == b"hekit-study-reference-encoding-v1"
            && r.text::<128>()?.as_bytes()
                == if oracle {
                    b"hekit-test-resource-supervisor-probe-origin-v1".as_slice()
                } else {
                    b"hekit-study-resource-supervisor-probe-origin-v1".as_slice()
                }
            && r.tag(1)? == u8::from(oracle),
        "test-only probe origin domain",
    )?;
    let source = r.hash()?;
    let revision = r.text::<40>()?;
    ensure(
        revision.as_bytes().len() == 40
            && revision
                .as_bytes()
                .iter()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c)),
        "probe source revision",
    )?;
    let executable = r.hash()?;
    let recipe = r.text::<128>()?;
    ensure(
        matches!(
            recipe.text()?,
            "study-train-cpu-serial-linux-clean-v2" | "study-reference-linux-clean-v2"
        ),
        "probe current recipe",
    )?;
    let receipt = r.reference()?;
    let host = r.reference()?;
    ensure(
        receipt.root == 3
            && receipt.path.as_bytes() == b"resource/linux-cpu-observed-v1/build-receipt.bin"
            && host.root == 3
            && host.path.as_bytes() == b"resource/linux-cpu-observed-v1/host-origin.bin",
        "probe fixed Policy receipt/host",
    )?;
    let account_uid = r.number()?;
    let account_path = r.path()?;
    let account_identity = r.identity()?;
    let policy_path = r.path()?;
    let policy_identity = r.identity()?;
    let observation_path = r.path()?;
    let observation_identity = r.identity()?;
    let probe_id = r.text::<128>()?;
    ensure(
        !probe_id.as_bytes().is_empty()
            && probe_id
                .as_bytes()
                .iter()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-')),
        "probe ID",
    )?;
    r.done()?;
    Ok(ProbeOrigin {
        source,
        revision,
        executable,
        recipe,
        receipt,
        host,
        account_uid,
        account_path,
        account_identity,
        policy_path,
        policy_identity,
        observation_path,
        observation_identity,
        probe_id,
    })
}
#[cfg(target_os = "linux")]
pub(crate) struct ProbeInputs {
    pub origin: ProbeOrigin,
    pub origin_file: super::retained::HeldFile,
    pub bytes: Vec<u8>,
    pub host_file: super::retained::HeldFile,
    pub host: super::host::HostOrigin,
    pub account: super::retained::HeldDirectory,
    pub policy: super::retained::HeldDirectory,
    pub observation: super::retained::HeldDirectory,
}
#[cfg(target_os = "linux")]
impl ProbeInputs {
    pub fn acquire(path: &std::path::Path, expected: &str) -> Result<Self> {
        use super::retained::{HeldDirectory, HeldFile, check_kernel_projection};
        use std::{os::unix::ffi::OsStrExt, path::Path};
        let mut origin_file = HeldFile::absolute(path, 4096)?;
        let bytes = origin_file.fixed_name_bytes(4096)?;
        ensure(
            crate::data::sha256_hex(&bytes) == expected,
            "probe origin selected hash",
        )?;
        let origin = parse_probe(&bytes, false)?;
        ensure(
            origin.account_uid == u64::from(rustix::process::geteuid().as_raw())
                && rustix::process::getuid() == rustix::process::geteuid(),
            "probe actual account",
        )?;
        let account = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
            origin.account_path.as_bytes(),
        )))?;
        let policy = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
            origin.policy_path.as_bytes(),
        )))?;
        let observation = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
            origin.observation_path.as_bytes(),
        )))?;
        ensure(
            account.identity()? == origin.account_identity
                && policy.identity()? == origin.policy_identity
                && observation.identity()? == origin.observation_identity,
            "probe retained origin identities",
        )?;
        let mut host_file = policy.retained_file(origin.host.path.text()?, 4096)?;
        let host_bytes = host_file.bytes((origin.host.length, origin.host.hash.text()?))?;
        let host = super::host::parse_host(&host_bytes, false)?;
        ensure(
            host.account_uid == origin.account_uid
                && host.account_identity == origin.account_identity,
            "probe host/account projection",
        )?;
        check_kernel_projection(&host)?;
        origin_file.recheck()?;
        host_file.recheck()?;
        Ok(Self {
            origin,
            origin_file,
            bytes,
            host_file,
            host,
            account,
            policy,
            observation,
        })
    }
    pub fn acquire_lock(&self) -> Result<(super::retained::HeldDirectory, std::fs::File)> {
        super::retained::acquire_account_lock(&self.account, self.host.physical.text()?)
    }
}

/// Exact byte-copy join also compares decoded receipts; no filename authority.
pub(crate) fn receipt_copy(original: &[u8], copy: &[u8]) -> Result<()> {
    use super::super::codec::Receipt;
    ensure(
        original.len() == copy.len()
            && crate::data::sha256_hex(original) == crate::data::sha256_hex(copy),
        "original/probe receipt bytes",
    )?;
    ensure(
        Receipt::decode(original)? == Receipt::decode(copy)?,
        "original/probe decoded receipt tuple",
    )
}
pub(crate) fn receipt_tuple(
    raw: &[u8],
    embedded: &[u8],
    source: &str,
    revision: &str,
    executable: &str,
    recipe: &str,
    executable_path: &[u8],
) -> Result<()> {
    use super::super::codec::{Capsule, Receipt};
    let r = Receipt::decode(raw)?;
    let c = Capsule::decode(&r.capsule)?;
    ensure(
        r.capsule == embedded
            && c.manifests[0] == source
            && c.commit == revision
            && r.executable_hash == executable
            && c.recipe == recipe
            && r.executable == executable_path,
        "probe actual embedded/current receipt tuple",
    )
}
