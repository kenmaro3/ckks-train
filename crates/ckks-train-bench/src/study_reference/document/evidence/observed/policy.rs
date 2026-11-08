//! Strict observed-policy source, using the unchanged owned FileRef JSON primitive.
use super::*;
#[derive(Debug, Clone)]
pub(in crate::study_reference::document) struct ObservedPolicy<'a> {
    pub(in crate::study_reference::document) raw: &'a [u8],
    mode: Mode,
    host_origin: resource::FileRef,
    pub(in crate::study_reference::document) native_validation: resource::FileRef,
    source_manifest: Hash,
    source_revision: Text<40>,
    executable: Hash,
    recipe: Text<128>,
}
impl<'a> ObservedPolicy<'a> {
    pub(in crate::study_reference::document) fn decode(raw: &'a [u8], mode: Mode) -> Result<Self> {
        preflight_layout(raw, DocumentLayout::Resource)?;
        let r: &RawValue =
            serde_json::from_slice(raw).map_err(|e| json_err(e, "observed.policy"))?;
        let mut f = Fields::new(
            r,
            &[
                "schema",
                "synthetic",
                "host_origin",
                "native_validation",
                "source_manifest",
                "source_revision",
                "executable",
                "recipe",
                "sizing_revision",
                "claim_domain",
                "rss_limit_kib",
                "sample_interval_ns",
                "term_grace_ns",
            ],
            "observed.policy",
        )?;
        same(
            f.field::<Text<128>>("schema", "policy")?.0
                == domain(mode, "resource-observed-policy", 1)
                && f.field::<bool>("synthetic", "policy")? == mode.synthetic(),
            "observed.policy.schema",
        )?;
        let v = Self {
            raw,
            mode,
            host_origin: f.field("host_origin", "policy")?,
            native_validation: f.field("native_validation", "policy")?,
            source_manifest: f.field("source_manifest", "policy")?,
            source_revision: f.field("source_revision", "policy")?,
            executable: f.field("executable", "policy")?,
            recipe: f.field("recipe", "policy")?,
        };
        same(
            v.source_revision.0.len() == 40
                && lower_hex(&v.source_revision.0)
                && [TRAIN_RECIPE, "study-reference-linux-clean-v2"].contains(&v.recipe.0.as_str()),
            "observed.policy.source",
        )?;
        same(
            f.field::<Text<128>>("sizing_revision", "policy")?.0 == SIZING
                && f.field::<Text<128>>("claim_domain", "policy")?.0 == "task54-host-serial-v1"
                && f.field::<u64>("rss_limit_kib", "policy")? == RSS_LIMIT
                && f.field::<u64>("sample_interval_ns", "policy")? == SAMPLE_INTERVAL
                && f.field::<u64>("term_grace_ns", "policy")? == TERM_GRACE,
            "observed.policy.constants",
        )?;
        Ok(v)
    }
    pub(in crate::study_reference::document) fn check_bootstrap(
        &self,
        b: &ObservedRecord<'_>,
    ) -> Result<()> {
        same(self.mode == b.mode, "observed.policy.mode")?;
        let RecordBody::Bootstrap(b) = &b.body else {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "observed.policy.bootstrap",
            ));
        };
        b.policy.check_bytes(self.raw)?;
        same(
            b.source_manifest.0 == self.source_manifest.0
                && b.source_revision.0 == self.source_revision.0
                && b.executable.0 == self.executable.0
                && b.recipe.0 == self.recipe.0
                && same_file(&self.host_origin, &b.namespace_origin),
            "observed.policy.bootstrap.fields",
        )
    }
    pub(in crate::study_reference::document::evidence) fn check_evidence(
        &self,
        r: &OperationalResourceEvidence<'_>,
    ) -> Result<()> {
        same(self.mode == r.mode, "observed.policy.mode")?;
        r.resource.policy.check_bytes(self.raw)?;
        same(
            same_file(&self.native_validation, &r.resource.native_validation)
                && self.source_manifest.0 == r.common.source_manifest_sha256.0
                && self.source_revision.0 == r.common.source_revision.0
                && self.executable.0 == r.common.executable_sha256.0,
            "observed.policy.evidence",
        )
    }
}
fn same_file(a: &resource::FileRef, b: &FileRef<'_>) -> bool {
    a.root.0 == b.root.0 && a.path.0 == b.path.0 && a.length == b.length && a.sha256.0 == b.sha256.0
}
