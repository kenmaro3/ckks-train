//! Exact outer wire and retained actual CPU producer. Parsed claims are inert.
use super::execution_descriptors::traverse_domain;
use super::runtime_descriptors::{CpuPolicyProjection, DescriptorBytes, PhysicalDescriptor};
use super::*;
use crate::study_admission::GeometryWriter;
use crate::study_client::{CpuPreparationError, CpuPreparationResult};
use crate::study_execution::{ActualBindingDescriptors, PreparedCpuRoute};
use crate::study_provenance::RecheckedTrainCapture;
use ckks_train_ckks::EvalConfig;
const BINDING: &str = "hekit-study-case-binding-v1";
// Borrowed transport fields only. No DTO constructs an actual route or readiness.
pub(super) struct ExecutionFields<'a> {
    pub(super) implementation: u8,
    pub(super) revision: &'a str,
    pub(super) manifest: &'a str,
    pub(super) parameter: &'a str,
    pub(super) bootstrap: Option<&'a str>,
    pub(super) layout: &'a str,
    pub(super) trainer: &'a str,
    pub(super) eval: EvalConfig,
    pub(super) engine: u8,
    pub(super) keys: &'a str,
    pub(super) security: &'a str,
    pub(super) physical: &'a str,
    pub(super) runtime: &'a str,
}
impl ExecutionFields<'_> {
    fn write(&self, w: &mut execution_descriptors::Writer) -> Result<()> {
        ensure(
            self.implementation <= 4 && self.engine <= 3,
            DocumentErrorCode::UnsupportedTag,
            "binding.tags",
        )?;
        ensure(
            self.revision.len() <= 128,
            DocumentErrorCode::TooLarge,
            "binding.revision",
        )?;
        w.tag(self.implementation)?;
        w.text(self.revision)?;
        w.hash(self.manifest)?;
        w.hash(self.parameter)?;
        w.optional_hash(self.bootstrap)?;
        w.hash(self.layout)?;
        w.hash(self.trainer)?;
        w.boolean(self.eval.hoist)?;
        w.boolean(self.eval.bsgs)?;
        w.boolean(self.eval.lazy_relin)?;
        w.tag(self.engine)?;
        w.hash(self.keys)?;
        w.hash(self.security)?;
        w.hash(self.physical)?;
        w.hash(self.runtime)
    }
    #[cfg(test)]
    pub(super) fn traverse(&self, output: bool) -> Result<(usize, Vec<u8>)> {
        traverse_domain(None, output, |w| self.write(w))
    }
    fn binding_size(&self) -> Result<usize> {
        Ok(traverse_domain(Some(BINDING), false, |w| self.write(w))?.0)
    }
    pub(super) fn binding_bytes(&self) -> Result<Vec<u8>> {
        Ok(traverse_domain(Some(BINDING), true, |w| self.write(w))?.1)
    }
}
// Mutates only the descriptor tail of the complete retained route. This never
// returns a detached proof or success state, and cannot construct gate authority.
pub(crate) fn bind_actual_cpu(
    route: &mut PreparedCpuRoute<'_, '_>,
    capture: &RecheckedTrainCapture<'_>,
    physical: &PhysicalDescriptor,
) -> CpuPreparationResult<()> {
    let document = CpuPreparationError::Descriptor;
    let provenance = |_| crate::study_diagonal::mismatch("actual capture/source/runtime");
    let (materialized, case, request) = route.owners();
    route.require_owners(materialized, case, request)?;
    let (shared, selected) = crate::study_reference::identity_payload_sizes(materialized)
        .map_err(|_| crate::study_diagonal::mismatch("capture identity sizing"))?;
    // derive_identities retains two buffers and four hash strings; the existing
    // checked_materialized_facts then copies five hashes before those drop.
    let identity_recheck = crate::study_execution::sum([shared, selected, 4 * 64, 5 * 64])?;
    {
        let (_, _, accounting) = route.binding_parts();
        let _reservation = accounting.reserve(identity_recheck)?;
        capture
            .recheck_binding_owners(case, materialized)
            .map_err(provenance)?;
    }
    let before = capture.runtime_descriptor_facts().map_err(provenance)?;
    let projection = CpuPolicyProjection::actual(route, &before, physical).map_err(document)?;
    let engine = route.bootstrap_engine();
    let context = capture.artifact_context();
    let source = context.train_source_manifest_bytes().map_err(provenance)?;
    // Borrow exact source/P1/plan bytes. Sizing uses fixed-width placeholders;
    // they cannot reach output, and no full transcript is copied or hashed yet.
    let (descriptors, facts, accounting) = route.binding_parts();
    ensure(
        descriptors.actual_binding.is_none(),
        DocumentErrorCode::CrossLinkMismatch,
        "binding.already_retained",
    )
    .map_err(document)?;
    let eval = facts.geometry(0).eval_config();
    for stage in 1..facts.stage_count() {
        ensure(
            facts.geometry(stage).eval_config() == eval,
            DocumentErrorCode::CrossLinkMismatch,
            "binding.stage_eval",
        )
        .map_err(document)?;
    }
    const HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    let fields = |manifest, trainer, runtime| ExecutionFields {
        implementation: 0,
        revision: context.source_commit(),
        manifest,
        parameter: &descriptors.parameter_digest,
        bootstrap: descriptors.bootstrap_digest.as_deref(),
        layout: &descriptors.layout_digest,
        trainer,
        eval,
        engine,
        keys: &descriptors.keys_digest,
        security: &descriptors.security_digest,
        physical: physical.sha256(),
        runtime,
    };
    let runtime_size = projection.size().map_err(document)?;
    let outer_size = fields(HASH, HASH, HASH).binding_size().map_err(document)?;
    let (retained_payload, peak) = binding_payload(
        runtime_size,
        outer_size,
        projection.temporary_payload(),
        identity_recheck,
    )?;
    let mut reservation = accounting.reserve(peak)?;
    let digest = |bytes: &[u8]| {
        crate::data::try_sha256_hex(bytes)
            .map_err(|_| crate::study_execution::AccountingError::Allocation)
    };
    let manifest = digest(source)?;
    ensure(
        manifest == context.source_manifest_sha256(),
        DocumentErrorCode::CrossLinkMismatch,
        "binding.source_manifest",
    )
    .map_err(document)?;
    let trainer = digest(facts.plan().canonical_bytes())?;
    let runtime = projection.encode_retained(&before).map_err(document)?;
    let execution = DescriptorBytes::new(
        fields(&manifest, &trainer, runtime.sha256())
            .binding_bytes()
            .map_err(document)?,
    )
    .map_err(document)?;
    ensure(
        runtime.canonical_bytes().len() == runtime_size
            && execution.canonical_bytes().len() == outer_size,
        DocumentErrorCode::CrossLinkMismatch,
        "binding.size_changed",
    )
    .map_err(document)?;
    // Recheck the same original owners/capture/source/runtime after projection.
    // No fake gate recheck is implied by these actual provenance rechecks.
    let after = capture.runtime_descriptor_facts().map_err(provenance)?;
    ensure(
        std::ptr::eq(before.host(), after.host())
            && before.execution_threads() == after.execution_threads()
            && before.recipe() == after.recipe(),
        DocumentErrorCode::CrossLinkMismatch,
        "binding.runtime_changed",
    )
    .map_err(document)?;
    ensure(
        std::ptr::eq(
            source,
            context.train_source_manifest_bytes().map_err(provenance)?,
        ),
        DocumentErrorCode::CrossLinkMismatch,
        "binding.source_owner",
    )
    .map_err(document)?;
    // Free temporary hash strings before releasing that part of the reservation.
    drop(trainer);
    drop(manifest);
    capture
        .recheck_binding_owners(case, materialized)
        .map_err(provenance)?;
    reservation.retain(retained_payload)?;
    descriptors.actual_binding = Some(ActualBindingDescriptors { runtime, execution });
    drop(reservation);
    Ok(())
}

// Exact application payload live intervals: runtime encoding (policy strings),
// outer encoding (two temporary hashes), then capture identity recheck. This is
// requested payload, not allocator capacity, context/OS inventory work or RSS.
pub(super) fn binding_payload(
    runtime: usize,
    outer: usize,
    policy_strings: usize,
    identity_recheck: usize,
) -> crate::study_execution::AccountingResult<(usize, usize)> {
    use crate::study_execution::sum;
    let retained = sum([runtime, outer, 2 * 64])?;
    let runtime_phase = sum([runtime, 64, 2 * 64, policy_strings])?;
    let outer_phase = sum([retained, 2 * 64])?;
    let recheck_phase = sum([retained, identity_recheck])?;
    Ok((retained, runtime_phase.max(outer_phase).max(recheck_phase)))
}
