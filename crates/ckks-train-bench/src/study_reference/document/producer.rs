//! Private adapter: only live session result types enter production serializers.
use super::*;
use crate::study_provenance::{BuildFacts, Observation as CapturedObservation};
use crate::study_reference::session::{
    CaptureFact, CapturedDiagnostic, CompletedFp32Reference, EndpointFact, IdentityFact,
    LocalQuickDiagnostic, LocalQuickF64Diagnostic, Record, SessionFailure, TimingFact,
};
use serde_json::{Value, json};
use std::os::unix::ffi::OsStrExt;
pub(super) const API: &str = "study-reference-api-v1";
pub(super) const ALGORITHM: &str = "study-reference-session-v1";
pub(super) const REDUCTION: &str = crate::study_reference::ARITHMETIC;
const REDUCTION64: &str = "study-plain-v1;actual-f64;canonical-f64;forward-row-output-input;gradient-row-output-input;separate-product-add-no-fma;mean-gradient-divide-after-sum;zero-velocity;nag-v=mu*v+g,step=g+mu*v;logical-output-loss;stage-gradient-barriers;pre-update-handoff";
pub(super) const CLASSIFICATION: &str = crate::study_reference::CLASSIFICATION;
pub(super) fn hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(H[(b >> 4) as usize] as char);
        out.push(H[(b & 15) as usize] as char);
    }
    out
}
fn available<T: serde::Serialize>(v: T) -> Value {
    json!({"status":"available","value":v})
}
fn unavailable(reason: &str) -> Value {
    json!({"status":"unavailable","reason":reason})
}
fn observed<T: serde::Serialize>(v: &CapturedObservation<T>) -> Value {
    match v {
        CapturedObservation::Available(v) => available(v),
        CapturedObservation::Unavailable(r) => unavailable(r),
    }
}
fn raw(bytes: &[u8]) -> Value {
    json!({"encoding":"unix_bytes","bytes":hex(bytes)})
}
fn identity(v: &IdentityFact) -> Value {
    json!({"profile_id":v.profile,"seed":v.receipt.seed(),"materialized_sha256":v.receipt.materialized_digest(),"contract":{"schema":CONTRACT_VERSION,"encoding":ENCODING_VERSION,"canonical_bytes":hex(v.contract.canonical_bytes()),"sha256":v.contract.digest()},"seed_receipt":{"schema":RECEIPT_VERSION,"encoding":ENCODING_VERSION,"canonical_bytes":hex(v.receipt.canonical_bytes()),"sha256":v.receipt.digest()}})
}
fn accuracy(v: crate::study_plain::PlainEvaluation) -> Value {
    json!({"correct":v.correct,"test_count":v.test_count,"value":v.accuracy(),"bits":format!("{:016x}",v.accuracy().to_bits())})
}
fn endpoint(v: &EndpointFact) -> Value {
    let c = &v.checkpoint;
    json!({"completed_updates":c.completed_updates,"processed_samples":c.processed_samples,"crossed_logical_boundaries":c.crossed_logical_boundaries,"exact_logical_boundary":c.exact_logical_boundary,"interval_updates":c.interval_updates,"interval_samples":c.interval_samples,"observation":accuracy(v.evaluation),"update_elapsed_ns":available(v.update_ns),"evaluation_elapsed_ns":available(v.evaluation_ns)})
}
fn time(t: &TimingFact) -> Value {
    json!({"wall_source":"unix_system_time","start_unix_ns":t.start,"end_unix_ns":t.end,"monotonic_source":"std_instant","scope":"after_materialization","end_to_end_ns":t.total,"setup_ns":available(t.setup),"update_ns":available(t.update),"evaluation_ns":available(t.evaluation),"finalization_ns":available(t.finalization)})
}
pub(super) fn build(f: &BuildFacts<'_>) -> Value {
    json!({"compiler_vv":f.compiler_vv(),"target":f.target(),"host":f.host(),"profile":f.profile(),"opt_level":f.opt_level(),"debug":f.debug(),"features":f.features(),"package_version":f.package_version(),"cargo_lock_sha256":f.cargo_lock_sha256(),"source_manifest_sha256":f.source_manifest_sha256(),"config_manifest_sha256":f.config_manifest_sha256(),"build_environment_sha256":f.build_environment_sha256(),"recipe_version":f.recipe_version(),"receipt_file_sha256":f.receipt_file_sha256()})
}
pub(super) fn captured(v: &CaptureFact<'_>) -> Result<Value> {
    let c = v.context;
    let start = &v.start;
    let end = &v.end;
    let f = c
        .build_facts(start.receipt_file_sha256())
        .map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "provenance.build"))?;
    let manifest = c
        .source_files()
        .map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "provenance.source"))?
        .into_iter()
        .map(|(path, sha256)| json!({"path":path,"sha256":sha256}))
        .collect::<Vec<_>>();
    let host = start.host();
    Ok(
        json!({"status":"captured","value":{"schema":"hekit-study-provenance-v1","build":build(&f),"source":{"root":raw(c.root().as_os_str().as_bytes()),"commit":available(c.source_commit()),"tracked":"clean","untracked":"clean","manifest":manifest},"executable":{"path":raw(f.executable_path()),"start_sha256":start.executable_start_sha256(),"end_sha256":end.executable_end_sha256()},"invocation":{"argv":start.argv().iter().map(|v|raw(v)).collect::<Vec<_>>(),"cwd":raw(start.cwd())},"host":{"cpu_model":observed(host.cpu_model()),"os":observed(host.os()),"hostname":observed(host.hostname()),"available_cpus":observed(host.available_cpus()),"process_id":observed(host.process_id()),"observed_threads":observed(host.observed_threads()),"execution_threads":host.execution_threads()},"thread_environment":start.thread_environment().iter().map(|(key,value)|json!({"key":key,"value":observed(value)})).collect::<Vec<_>>(),"recheck":{"source_manifest_sha256":end.source_manifest_sha256(),"build_receipt_file_sha256":end.receipt_file_sha256()}}}),
    )
}
fn common_value(
    schema: &str,
    kind: &str,
    status: &str,
    id: &str,
    dtype: u8,
    provenance: Value,
) -> Value {
    json!({"schema":schema,"kind":kind,"status":status,"run_id":id,"writer":{"api_version":API,"algorithm_revision":ALGORITHM},"synthetic":false,"arithmetic":{"dtype":if dtype==32{"f32"}else{"f64"},"accumulation":if dtype==32{"scalar_f32"}else{"scalar_f64"},"cast":if dtype==32{"f64_to_f32_once"}else{"canonical_f64"},"reduction":if dtype==32{REDUCTION}else{REDUCTION64},"classification":CLASSIFICATION},"provenance":provenance,"integrity":{"encoding":ENCODING_VERSION,"hash":"sha256","payload_sha256":"0".repeat(64)}})
}
fn diagnostic(r: &Record, provenance: Value) -> Result<Vec<u8>> {
    let mut v = common_value(
        DIAGNOSTIC,
        if r.full {
            "diagnostic_full"
        } else {
            "diagnostic_quick"
        },
        "complete",
        &r.run_id,
        r.dtype,
        provenance,
    );
    v["identity"] = identity(&r.identity);
    v["target_eligible"] = json!(false);
    v["scope"] = json!(if r.full {
        "full_diagnostic"
    } else {
        "quick_two_updates"
    });
    v["progress"] = json!({"origin_completed_updates":0,"origin_processed_samples":0,"planned_updates":r.plan.updates.len(),"consumed_updates":r.progress.completed_updates(),"planned_samples":r.plan.processed_samples,"consumed_samples":r.progress.processed_samples(),"planned_dropped_samples":r.plan.dropped_samples,"full_plan_sha256":r.full_hash,"consumed_plan_sha256":r.consumed});
    v["endpoints"] = json!(r.endpoints.iter().map(endpoint).collect::<Vec<_>>());
    v["last_observation"] = accuracy(r.observation);
    v["last_state_sha256"] = json!(r.state);
    v["timing"] = time(&r.timing);
    seal(v, Kind::Quick)
}
pub(in crate::study_reference) fn serialize_local_quick(
    r: &LocalQuickDiagnostic,
) -> Result<Vec<u8>> {
    diagnostic(
        r.record(),
        unavailable("local_diagnostic_no_production_build_receipt"),
    )
}
pub(in crate::study_reference) fn serialize_local_quick_f64(
    r: &LocalQuickF64Diagnostic,
) -> Result<Vec<u8>> {
    diagnostic(
        r.record(),
        unavailable("local_diagnostic_no_production_build_receipt"),
    )
}
pub(in crate::study_reference) fn serialize_diagnostic(
    r: &CapturedDiagnostic<'_>,
) -> Result<Vec<u8>> {
    diagnostic(r.record(), captured(r.capture())?)
}
pub(in crate::study_reference) fn serialize_reference(
    result: &CompletedFp32Reference<'_>,
) -> Result<Vec<u8>> {
    let r = result.record();
    let mut v = common_value(
        REFERENCE,
        "full_fp32_final",
        "complete",
        &r.run_id,
        32,
        captured(result.capture())?,
    );
    v["identity"] = identity(&r.identity);
    v["completion"] = json!({"origin_completed_updates":0,"origin_processed_samples":0,"planned_updates":r.plan.updates.len(),"actual_updates":r.progress.completed_updates(),"planned_samples":r.plan.processed_samples,"actual_samples":r.progress.processed_samples(),"planned_dropped_samples":r.plan.dropped_samples,"actual_dropped_samples":r.plan.dropped_samples,"full_plan_sha256":r.full_hash,"consumed_plan_sha256":r.consumed,"final_endpoint_index":r.endpoints.len().checked_sub(1).ok_or_else(||err(DocumentErrorCode::OuterProgressMismatch,"endpoints"))?,"final_state_sha256":r.state});
    v["endpoints"] = json!(r.endpoints.iter().map(endpoint).collect::<Vec<_>>());
    v["final_observation"] = accuracy(r.observation);
    v["timing"] = time(&r.timing);
    seal(v, Kind::Reference)
}
pub(in crate::study_reference) fn serialize_failure(r: &SessionFailure) -> Result<Vec<u8>> {
    let r = r.detail();
    let mut v = common_value(
        FAILURE,
        "reference_failure",
        "failed",
        &r.run_id,
        r.dtype,
        unavailable("session_failed_before_provenance_recheck"),
    );
    v["identity"] = r.identity.as_ref().map(identity).unwrap_or(Value::Null);
    v["target_eligible"] = json!(false);
    v["requested_scope"] = json!(r.scope);
    v["phase"] = json!(r.phase);
    v["error"] = json!({"code":r.error.to_string(),"message":r.error.to_string()});
    v["last_successful_endpoint"] = r.last.as_ref().map(endpoint).unwrap_or(Value::Null);
    v["progress"]=r.progress.map(|p|json!({"completed_updates":p.completed_updates(),"processed_samples":p.processed_samples()})).unwrap_or(Value::Null);
    v["timing"] = r.timing.as_ref().map(time).unwrap_or(Value::Null);
    seal(v, Kind::Failure)
}
// The unchanged bounded typed grammar and envelope encoder own every wire field.
// This function is private and receives only values assembled above, not caller JSON.
fn seal(mut v: Value, kind: Kind) -> Result<Vec<u8>> {
    let initial = serde_json::to_vec(&v).map_err(|e| json_err(e, ""))?;
    preflight(&initial)?;
    let raw = serde_json::from_slice::<&RawValue>(&initial).map_err(|e| json_err(e, ""))?;
    let keys: Vec<_> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
    // Fields requires static key inventories; these are the existing schema's keys.
    let allowed: &[&str] = match kind {
        Kind::Reference => &[
            "schema",
            "kind",
            "status",
            "run_id",
            "writer",
            "synthetic",
            "identity",
            "arithmetic",
            "completion",
            "endpoints",
            "final_observation",
            "timing",
            "provenance",
            "integrity",
        ],
        Kind::Failure => &[
            "schema",
            "kind",
            "status",
            "run_id",
            "writer",
            "synthetic",
            "identity",
            "arithmetic",
            "target_eligible",
            "requested_scope",
            "phase",
            "error",
            "last_successful_endpoint",
            "progress",
            "timing",
            "provenance",
            "integrity",
        ],
        _ => &[
            "schema",
            "kind",
            "status",
            "run_id",
            "writer",
            "synthetic",
            "identity",
            "arithmetic",
            "target_eligible",
            "scope",
            "progress",
            "endpoints",
            "last_observation",
            "last_state_sha256",
            "timing",
            "provenance",
            "integrity",
        ],
    };
    debug_assert_eq!(keys.len(), allowed.len());
    let mut f = Fields::new(raw, allowed, "")?;
    let mut c = common(
        &mut f,
        String::new(),
        match kind {
            Kind::Reference => REFERENCE,
            Kind::Failure => FAILURE,
            _ => DIAGNOSTIC,
        },
    )?;
    provenance(&mut c, kind != Kind::Failure)?;
    let encoded = match kind {
        Kind::Reference => reference_encoding(
            &c,
            &ReferenceBody {
                identity: f.field("identity", "")?,
                completion: f.field("completion", "")?,
                endpoints: f.field("endpoints", "")?,
                final_observation: f.field("final_observation", "")?,
                timing: f.field("timing", "")?,
            },
        )?,
        Kind::Failure => failure_encoding(
            &c,
            &FailureBody {
                identity: f.field("identity", "")?,
                target_eligible: f.field("target_eligible", "")?,
                requested_scope: f.field("requested_scope", "")?,
                phase: f.field("phase", "")?,
                error: f.field("error", "")?,
                last_successful_endpoint: f.field("last_successful_endpoint", "")?,
                progress: f.field("progress", "")?,
                timing: f.field("timing", "")?,
            },
        )?,
        _ => diagnostic_encoding(
            &c,
            &DiagnosticBody {
                identity: f.field("identity", "")?,
                target_eligible: f.field("target_eligible", "")?,
                scope: f.field("scope", "")?,
                progress: f.field("progress", "")?,
                endpoints: f.field("endpoints", "")?,
                last_observation: f.field("last_observation", "")?,
                last_state_sha256: f.field("last_state_sha256", "")?,
                timing: f.field("timing", "")?,
            },
        )?,
    };
    v["integrity"]["payload_sha256"] = json!(sha256_hex(&encoded.bytes));
    let bytes = serde_json::to_vec(&v).map_err(|e| json_err(e, ""))?;
    match kind {
        Kind::Reference => {
            parse_reference_document(&bytes)?;
        }
        Kind::Failure => {
            parse_failure_document(&bytes)?;
        }
        _ => {
            parse_diagnostic_document(&bytes)?;
        }
    }
    Ok(bytes)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_independent_outer_arithmetic_literals() {
        assert_eq!(
            REDUCTION,
            "study-plain-v1;actual-f32;round-f64-to-f32-once;forward-row-output-input;gradient-row-output-input;separate-product-add-no-fma;mean-gradient-divide-after-sum;zero-velocity;nag-v=mu*v+g,step=g+mu*v;logical-output-loss;stage-gradient-barriers;pre-update-handoff"
        );
        assert_eq!(
            REDUCTION64,
            "study-plain-v1;actual-f64;canonical-f64;forward-row-output-input;gradient-row-output-input;separate-product-add-no-fma;mean-gradient-divide-after-sum;zero-velocity;nag-v=mu*v+g,step=g+mu*v;logical-output-loss;stage-gradient-barriers;pre-update-handoff"
        );
        assert_eq!(
            CLASSIFICATION,
            "real-classes-only;last-total_cmp-maximum;signed-zero-total-order;all-logical-logits-finite"
        );
    }
}
