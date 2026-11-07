//! Fixed untrusted Case component, shared nested grammar, no execution authority.
use super::*;
tags!(CaseExecutableRole { StudyTrain=1=>"study_train" });
record!(CaseValue {
    schema: Text<128>,
    role: CaseExecutableRole,
    build: Build,
    source: Source,
    executable: Executable,
    invocation: Invocation,
    host: Host,
    thread_environment: List<ThreadEnvironment,32>,
    recheck: Recheck,
});
// Parsed claims cannot construct a context, checked case or captured runtime fact.
pub(super) struct ParsedCaseProvenance(CaseValue);
impl Decode for ParsedCaseProvenance {
    fn decode(raw: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(raw, &["status", "value"], p)?;
        let status: Text<128> = f.field("status", p)?;
        ensure(status.0 == "captured", DocumentErrorCode::UnsupportedTag, p)?;
        let mut value: CaseValue = f.field("value", p)?;
        validate(&mut value)?;
        Ok(Self(value))
    }
}
impl Encode for ParsedCaseProvenance {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.byte(0)?;
        self.0.encode(e)
    }
}

// These checks belong to this distinct claim type. Reference validation is
// intentionally unchanged, including its unavailable branch and scalar1 rule.
fn validate(p: &mut CaseValue) -> Result<()> {
    ensure(
        p.schema.0 == "hekit-study-case-provenance-v1",
        DocumentErrorCode::UnsupportedSchema,
        "provenance.schema",
    )?;
    for (raw, field) in [
        (&p.source.root, "provenance.source.root"),
        (&p.executable.path, "provenance.executable.path"),
        (&p.invocation.cwd, "provenance.invocation.cwd"),
    ] {
        ensure(
            raw.bytes.first() == Some(&b'/'),
            DocumentErrorCode::OutOfRange,
            field,
        )?;
    }
    nonempty(&p.build.recipe_version, "provenance.build.recipe_version")?;
    nonempty(&p.build.package_version, "provenance.build.package_version")?;
    unique_by(
        &mut p.build.features.0,
        |x| &x.0,
        "provenance.build.features",
    )?;
    unique_by(
        &mut p.source.manifest.0,
        |x| &x.path.0,
        "provenance.source.manifest",
    )?;
    for file in &p.source.manifest.0 {
        let value = &file.path.0;
        ensure(
            !value.contains(['\\', '\0'])
                && value
                    .split('/')
                    .all(|c| !c.is_empty() && c != "." && c != ".."),
            DocumentErrorCode::OutOfRange,
            "provenance.source.manifest.path",
        )?;
    }
    unique_by(
        &mut p.thread_environment.0,
        |x| &x.key.0,
        "provenance.thread_environment",
    )?;
    for env in &p.thread_environment.0 {
        ensure(
            matches!(
                env.key.0.as_str(),
                "RAYON_NUM_THREADS"
                    | "OMP_NUM_THREADS"
                    | "OPENBLAS_NUM_THREADS"
                    | "MKL_NUM_THREADS"
                    | "VECLIB_MAXIMUM_THREADS"
                    | "BLIS_NUM_THREADS"
                    | "NUMEXPR_NUM_THREADS"
            ),
            DocumentErrorCode::UnsupportedTag,
            "provenance.thread_environment.key",
        )?;
    }
    ensure(
        p.host.execution_threads > 0,
        DocumentErrorCode::OutOfRange,
        "provenance.host.execution_threads",
    )?;
    for obs in [&p.host.available_cpus, &p.host.observed_threads] {
        ensure(
            obs.available().is_none_or(|n| *n > 0),
            DocumentErrorCode::OutOfRange,
            "provenance.host.threads",
        )?;
    }
    ensure(
        p.executable.start_sha256 == p.executable.end_sha256
            && p.build.source_manifest_sha256 == p.recheck.source_manifest_sha256
            && p.build.receipt_file_sha256 == p.recheck.build_receipt_file_sha256,
        DocumentErrorCode::CrossLinkMismatch,
        "provenance.recheck",
    )
}
