//! Closed explicit study requests. No checked case, admission or execution success.
use crate::{
    study_profile::{CaseBackend, CaseScope, CaseSelector},
    study_provenance::ContextRequest,
    study_runtime::{self, RuntimeRequest, StudyRunRequest},
};
use std::{collections::BTreeMap, ffi::OsString, fmt, os::unix::ffi::OsStrExt, path::PathBuf};

/// An owned refusal diagnostic; this frontend has no successful execution state.
#[derive(Debug, PartialEq, Eq)]
pub enum RunDiagnostic {
    /// Real D7 descriptors, gate leases and checked execution remain unavailable.
    MissingExecutionDependency,
}
impl fmt::Display for RunDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingExecutionDependency => f.write_str(
                "missing execution dependency: D7 checked binding, descriptors and gates",
            ),
        }
    }
}
/// Bounded error category. Supplied argument/path contents are never echoed.
#[derive(Debug)]
pub struct RequestError(&'static str);
impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for RequestError {}
struct ParsedRequest {
    root: PathBuf,
    approval: PathBuf,
    approval_sha256: String,
    receipts: Vec<PathBuf>,
    case: CaseSelector,
    runtime: RuntimeRequest,
}
fn require(ok: bool, why: &'static str) -> Result<(), RequestError> {
    if ok { Ok(()) } else { Err(RequestError(why)) }
}
fn text(value: &std::ffi::OsStr) -> Result<&str, RequestError> {
    value
        .to_str()
        .ok_or(RequestError("nonUTF8 semantic argument"))
}
fn integer(value: &str) -> Result<u64, RequestError> {
    require(
        !value.is_empty()
            && value.bytes().all(|b| b.is_ascii_digit())
            && (value == "0" || !value.starts_with('0')),
        "invalid integer argument",
    )?;
    value
        .parse()
        .map_err(|_| RequestError("integer argument overflow"))
}
fn identifier(value: &str) -> Result<String, RequestError> {
    require(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')),
        "invalid registration identifier",
    )?;
    Ok(value.into())
}
fn parse(args: Vec<OsString>) -> Result<ParsedRequest, RequestError> {
    // Existing runtime invocation limits include argv0; this facade excludes it.
    require(
        args.len() < 256
            && args
                .iter()
                .all(|v| v.as_bytes().len() <= 4096 && !v.as_bytes().contains(&0)),
        "argument transport bounds",
    )?;
    require(args.len() % 2 == 0, "missing option value")?;
    let mut fields = BTreeMap::new();
    let mut receipts = Vec::new();
    for pair in args.chunks_exact(2) {
        let key = text(&pair[0])?;
        require(
            matches!(
                key,
                "--root"
                    | "--approval"
                    | "--approval-sha256"
                    | "--receipt"
                    | "--origin"
                    | "--registration"
                    | "--profile"
                    | "--seed"
                    | "--scope"
                    | "--backend"
                    | "--mode"
                    | "--placement"
                    | "--packing"
                    | "--runtime"
                    | "--workers"
                    | "--device-ordinal"
            ),
            "unknown option",
        )?;
        if key == "--receipt" {
            require(receipts.len() < 64, "receipt count bounds")?;
            require(!pair[1].is_empty(), "empty receipt path")?;
            receipts.push(PathBuf::from(&pair[1]));
        } else {
            require(
                fields.insert(key.to_owned(), pair[1].clone()).is_none(),
                "duplicate option",
            )?;
        }
    }
    let get = |key: &str| -> Result<&std::ffi::OsStr, RequestError> {
        fields
            .get(key)
            .map(OsString::as_os_str)
            .ok_or(RequestError("missing explicit option"))
    };
    let semantic = |key: &str| -> Result<&str, RequestError> { text(get(key)?) };
    let closed = || RequestError("unknown case vocabulary");
    let case = CaseSelector {
        origin: semantic("--origin")?.parse().map_err(|_| closed())?,
        registration_id: identifier(semantic("--registration")?)?,
        profile_id: identifier(semantic("--profile")?)?,
        seed: integer(semantic("--seed")?)?,
        scope: semantic("--scope")?.parse().map_err(|_| closed())?,
        backend: semantic("--backend")?.parse().map_err(|_| closed())?,
        mode: semantic("--mode")?.parse().map_err(|_| closed())?,
        placement: semantic("--placement")?.parse().map_err(|_| closed())?,
        packing: semantic("--packing")?.parse().map_err(|_| closed())?,
    };
    require(
        case.seed <= 9 && case.scope == CaseScope::Full,
        "not a prospective full selector",
    )?;
    let form = semantic("--runtime")?;
    let runtime = match form {
        "cpu-serial" => {
            require(
                case.backend == CaseBackend::Cpu
                    && !fields.contains_key("--workers")
                    && !fields.contains_key("--device-ordinal"),
                "serial runtime options mismatch",
            )?;
            RuntimeRequest::SerialCpu
        }
        "cpu-rayon" | "cuda-host" => {
            let workers = integer(semantic("--workers")?)?;
            require(
                workers > 0 && usize::try_from(workers).is_ok(),
                "invalid worker count",
            )?;
            if form == "cpu-rayon" {
                require(
                    case.backend == CaseBackend::Cpu && !fields.contains_key("--device-ordinal"),
                    "CPU Rayon options mismatch",
                )?;
                RuntimeRequest::RayonCpu { workers }
            } else {
                require(
                    case.backend == CaseBackend::Cuda,
                    "CUDA runtime backend mismatch",
                )?;
                let ordinal = usize::try_from(integer(semantic("--device-ordinal")?)?)
                    .map_err(|_| RequestError("device ordinal overflow"))?;
                RuntimeRequest::CudaHost { workers, ordinal }
            }
        }
        _ => return Err(RequestError("unknown runtime form")),
    };
    let approval_sha256 = semantic("--approval-sha256")?;
    require(
        approval_sha256.len() == 64
            && approval_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f')),
        "invalid approval SHA256",
    )?;
    let root = PathBuf::from(get("--root")?);
    let approval = PathBuf::from(get("--approval")?);
    require(
        !root.as_os_str().is_empty() && !approval.as_os_str().is_empty() && !receipts.is_empty(),
        "missing authority locations",
    )?;
    Ok(ParsedRequest {
        root,
        approval,
        approval_sha256: approval_sha256.into(),
        receipts,
        case,
        runtime,
    })
}
/// Parse explicit arguments excluding argv0 and enter the one concrete dispatcher.
/// Returns only an owned refusal or error; emits no artifacts and grants no capability.
pub fn run(args: Vec<OsString>) -> Result<RunDiagnostic, RequestError> {
    let r = parse(args)?;
    let request = StudyRunRequest {
        locations: ContextRequest::new(r.root, r.approval, r.approval_sha256, r.receipts),
        runtime: r.runtime,
        case: r.case,
    };
    match study_runtime::dispatch(request).map_err(|_| {
        RequestError("study runtime/context/registration unavailable or inconsistent")
    })? {
        study_runtime::RunOutcome::MissingExecutionDependency => {
            Ok(RunDiagnostic::MissingExecutionDependency)
        }
    }
}
#[cfg(test)]
#[path = "study_run_tests.rs"]
mod tests;
