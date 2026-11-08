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
    /// Genuine live gate leases, admission/start and checked execution remain unavailable.
    MissingExecutionDependency,
}
impl fmt::Display for RunDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingExecutionDependency => f.write_str(
                "missing execution dependency: live gates, admission, durable start and scientific execution",
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
    planning: Option<crate::study_admission::DiagonalRequest>,
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
                    | "--consumption"
                    | "--physical-lanes"
                    | "--transpose"
                    | "--transpose-pool"
                    | "--eval-hoist"
                    | "--eval-bsgs"
                    | "--eval-lazy-relin"
                    | "--partitions"
                    | "--receipt-max-bytes"
                    | "--receipt-max-records"
                    | "--work-geometries"
                    | "--work-initial-states"
                    | "--work-updates"
                    | "--work-handoffs"
                    | "--work-wrappers"
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
    let planning = parse_planning(&fields, case.seed)?;
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
        planning,
    })
}
// The complete group is an untrusted immutable request, never gate authority.
fn parse_planning(
    fields: &BTreeMap<String, OsString>,
    seed: u64,
) -> Result<Option<crate::study_admission::DiagonalRequest>, RequestError> {
    use crate::study_admission::{
        Consumption, DiagonalRequest, PlanningBudget, PlanningUsage, ReceiptLimits,
    };
    use ckks_train_ckks::EvalConfig;
    use ckks_train_nn::dense::TransposeKeys;
    const GROUP: [&str; 15] = [
        "--consumption",
        "--physical-lanes",
        "--transpose",
        "--transpose-pool",
        "--eval-hoist",
        "--eval-bsgs",
        "--eval-lazy-relin",
        "--partitions",
        "--receipt-max-bytes",
        "--receipt-max-records",
        "--work-geometries",
        "--work-initial-states",
        "--work-updates",
        "--work-handoffs",
        "--work-wrappers",
    ];
    let supplied = GROUP.iter().filter(|k| fields.contains_key(**k)).count();
    if supplied == 0 {
        return Ok(None);
    }
    require(supplied == GROUP.len(), "partial planning option group")?;
    let get = |key: &str| text(fields.get(key).expect("complete planning group"));
    let number = |key: &str| -> Result<usize, RequestError> {
        usize::try_from(integer(get(key)?)?).map_err(|_| RequestError("planning integer overflow"))
    };
    let flag = |key: &str| -> Result<bool, RequestError> {
        match get(key)? {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(RequestError("invalid planning flag")),
        }
    };
    let consumption = match get("--consumption")? {
        "full" => Consumption::Full,
        "quick" => {
            require(seed == 0, "Quick consumption requires seed0")?;
            Consumption::Quick
        }
        _ => return Err(RequestError("unknown consumption")),
    };
    let physical_lanes = if get("--physical-lanes")? == "auto" {
        None
    } else {
        let n = number("--physical-lanes")?;
        require(n > 0, "physical lanes must be positive")?;
        Some(n)
    };
    let pool = get("--transpose-pool")?;
    let transpose = match get("--transpose")? {
        "dedicated" => {
            require(pool == "none", "dedicated transpose requires none")?;
            TransposeKeys::Dedicated
        }
        "reuse" => {
            require(
                !pool.is_empty() && pool != "none",
                "reuse requires explicit pool",
            )?;
            let pool = pool
                .split(',')
                .map(|s| {
                    let n = s
                        .parse::<i32>()
                        .map_err(|_| RequestError("invalid transpose step"))?;
                    require(n.to_string() == s, "noncanonical transpose step")?;
                    Ok(n)
                })
                .collect::<Result<Vec<_>, RequestError>>()?;
            TransposeKeys::Reuse(pool)
        }
        _ => return Err(RequestError("unknown transpose policy")),
    };
    require(
        get("--partitions")? == "contiguous",
        "unknown partition policy",
    )?;
    Ok(Some(DiagonalRequest {
        consumption,
        physical_lanes,
        transpose,
        eval_config: EvalConfig {
            hoist: flag("--eval-hoist")?,
            bsgs: flag("--eval-bsgs")?,
            lazy_relin: flag("--eval-lazy-relin")?,
        },
        partitions: None,
        limits: ReceiptLimits {
            max_bytes: number("--receipt-max-bytes")?,
            max_records: number("--receipt-max-records")?,
        },
        work_budget: std::sync::Arc::new(PlanningBudget::new(PlanningUsage {
            geometries: number("--work-geometries")?,
            initial_states: number("--work-initial-states")?,
            updates: number("--work-updates")?,
            handoffs: number("--work-handoffs")?,
            wrapper_calls: number("--work-wrappers")?,
        })),
    }))
}
fn observed_work(
    r: &ParsedRequest,
) -> Result<crate::study_reference::operational::Requested, RequestError> {
    use crate::study_profile::{CaseMode, CaseOrigin, CasePacking, CasePlacement};
    require(
        matches!(r.runtime, RuntimeRequest::SerialCpu)
            && r.case.origin == CaseOrigin::Family
            && r.case.scope == CaseScope::Full
            && r.case.backend == CaseBackend::Cpu
            && r.case.profile_id == "iris__emlp1"
            && r.case.seed == 0
            && r.case.packing == CasePacking::BatchedDiagonal,
        "outside observed CPU slice",
    )?;
    let noninteractive = r.case.mode == CaseMode::Noninteractive;
    require(
        if noninteractive {
            r.case.registration_id == "main_ni" && r.case.placement == CasePlacement::WhenNeeded
        } else {
            r.case.registration_id == "main_i" && r.case.placement == CasePlacement::ClientRefresh
        },
        "observed selector axes",
    )?;
    let planning = r.planning.as_ref().ok_or(RequestError(
        "observed route requires compiled explicit planning group",
    ))?;
    let work = crate::study_reference::operational::Requested::train(
        noninteractive,
        planning.consumption == crate::study_admission::Consumption::Quick,
    )
    .map_err(|_| RequestError("fixed work encoding"))?;
    work.request_matches(planning)
        .map_err(|_| RequestError("caller planning overrides compiled observed work"))?;
    Ok(work)
}
/// Parse bounded arguments excluding argv0 and enter the observed Linux supervisor.
/// Scientific continuation remains gated; a closed refusal is never execution success.
pub fn run(args: Vec<OsString>) -> Result<RunDiagnostic, RequestError> {
    if args.first().is_some_and(|v| v == "--private-helper") {
        crate::study_supervisor::helper(&args).map_err(|e| RequestError(e.0))?;
        return Err(RequestError("private exec returned unexpectedly"));
    }
    let private = args.first().is_some_and(|v| v == "--private-worker");
    let r = parse(if private {
        args[1..].to_vec()
    } else {
        args.clone()
    })?;
    let work = observed_work(&r)?;
    if !private {
        crate::study_supervisor::launch(
            &r.root,
            &r.approval,
            &r.approval_sha256,
            &r.receipts,
            &args,
            work,
        )
        .map_err(|e| RequestError(e.0))?;
        return Err(RequestError(
            "supervisor returned without scientific completion",
        ));
    }
    let owner = crate::study_supervisor::worker(&args, &r.root, &r.approval_sha256, work)
        .map_err(|e| RequestError(e.0))?;
    let request = StudyRunRequest {
        locations: ContextRequest::new(r.root, r.approval, r.approval_sha256, r.receipts),
        runtime: r.runtime,
        case: r.case,
        planning: r.planning,
    };
    match study_runtime::dispatch_observed(request, &owner).map_err(|_| {
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
/// Handle only the closed operational native-probe or authenticated private
/// probe selector. Success is operational process completion, never science.
/// Other arguments return `None` for the ordinary study frontend.
#[doc(hidden)]
pub fn native_probe_entry(args: &[OsString]) -> Option<Result<(), RequestError>> {
    crate::study_supervisor::probe_entry(args).map(|result| result.map_err(|e| RequestError(e.0)))
}
