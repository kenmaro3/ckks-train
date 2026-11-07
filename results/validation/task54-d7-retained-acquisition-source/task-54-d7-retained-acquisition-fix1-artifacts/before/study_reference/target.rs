//! Verified lock publication only. No admission, start, key or HE authority.
use super::{
    DocumentError, PublicationError, ReferenceStore, SeedInputReceiptV1, VerifiedReference,
    VerifiedReferenceSnapshot, derive_identities, document::target as codec, session::CaptureFact,
};
use crate::{
    data::sha256_hex,
    study_inputs::{MaterializedProfile, materialize},
    study_profile::{ResolvedProfile, StudyRegistry},
    study_provenance::{
        VerifiedStudyContext,
        publication::{DurableBytes, ProfileGuard, PublishError, TargetStore},
    },
};
use std::{
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
type Result<T> = std::result::Result<T, TargetError>;
/// Closed target failures. Every error withholds durable success and execution authority.
#[derive(Debug)]
pub enum TargetError {
    /// Unsupported target policy or request identifier.
    UnsupportedPolicy,
    /// Production needs approved local Linux ext4/XFS.
    UnsupportedPlatform,
    /// The selected canonical input was not available; no fallback exists.
    MissingInput,
    /// Canonical profile or selected seed mismatch.
    CaseMismatch,
    /// Approved source, inputs, build or context changed.
    ContextChanged,
    /// Invalid or unavailable wall/monotonic observation.
    ClockInvalid,
    /// Reference/lock bytes, namespace or payload mismatch.
    ByteOrPayloadMismatch,
    /// Existing target, cases or unsupported profile state blocks a new lock.
    AlreadyExists,
    /// Validation or filesystem failure before a final link.
    BeforePublication,
    /// Bounded document rejection.
    Document(DocumentError),
    /// A final link is visible or may be visible, but no durable capability was granted.
    PublishedButDurabilityUnconfirmed {
        /// Exact attempted final path.
        path: PathBuf,
        /// SHA256 of exact attempted file bytes.
        file_sha256: String,
        /// SHA256 of attempted canonical payload.
        payload_sha256: String,
    },
}
impl std::fmt::Display for TargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Document(e) => write!(f, "target document: {e}"),
            Self::PublishedButDurabilityUnconfirmed { .. } => {
                f.write_str("target publication durability unconfirmed")
            }
            _ => write!(f, "target: {self:?}"),
        }
    }
}
impl std::error::Error for TargetError {}
impl From<DocumentError> for TargetError {
    fn from(e: DocumentError) -> Self {
        Self::Document(e)
    }
}
fn require(ok: bool, e: TargetError) -> Result<()> {
    if ok { Ok(()) } else { Err(e) }
}
fn reference_error(e: PublicationError) -> TargetError {
    match e {
        PublicationError::Document(d) if d.path() == "canonical_inputs_missing_or_invalid" => {
            TargetError::MissingInput
        }
        PublicationError::Document(d) => TargetError::Document(d),
        _ => TargetError::ByteOrPayloadMismatch,
    }
}
/// Validated bounded campaign selector, never a filesystem capability.
pub struct CampaignId(String);
impl CampaignId {
    /// Parse ASCII alphanumeric, underscore or hyphen, 1..=128 bytes.
    pub fn parse(request: &str) -> Result<Self> {
        require(
            super::session::valid_id(request).is_ok(),
            TargetError::UnsupportedPolicy,
        )?;
        Ok(Self(request.into()))
    }
}
/// Held descriptor-relative target profile store borrowing the approved context.
/// ```compile_fail
/// use ckks_train_bench::study_reference::ProfileTargetStore;
/// fn borrow(mut store:ProfileTargetStore<'_>) {
///     let claim=store.claim_new().unwrap(); drop(store); drop(claim);
/// }
/// ```
/// ```compile_fail
/// use ckks_train_bench::{study_reference::{ProfileTargetStore,CampaignId},study_provenance::VerifiedStudyContext,study_profile::ResolvedProfile};
/// fn borrow(context:VerifiedStudyContext,profile:&ResolvedProfile) {
///     let mut store=ProfileTargetStore::open(std::path::Path::new("/unused"),&CampaignId::parse("a").unwrap(),profile,&context).unwrap();
///     let claim=store.claim_new().unwrap(); drop(context); drop(claim);
/// }
/// ```
pub struct ProfileTargetStore<'c> {
    inner: TargetStore,
    context: &'c VerifiedStudyContext,
    campaign: String,
    profile: String,
}
/// Exclusive new-profile claim holding a fresh opened flock FD through publication.
/// ```compile_fail
/// use ckks_train_bench::study_reference::NewProfileTargetStore;
/// fn duplicate(claim:NewProfileTargetStore<'_,'_>) {let _=claim.clone();}
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_reference::NewProfileTargetStore;
/// let _:NewProfileTargetStore<'static,'static>=serde_json::from_str("{}").unwrap();
/// ```
pub struct NewProfileTargetStore<'s, 'c: 's> {
    store: &'s mut ProfileTargetStore<'c>,
    guard: ProfileGuard,
}
/// Confirmed durable lock or fully verified recovery, not HE readiness.
pub struct PublishedTargetLock {
    durable: DurableBytes,
    payload: String,
}
impl PublishedTargetLock {
    /// Exact dedicated final path.
    pub fn path(&self) -> &Path {
        self.durable.path()
    }
    /// Exact complete file digest.
    pub fn file_sha256(&self) -> &str {
        self.durable.file_sha256()
    }
    /// Canonical payload digest.
    pub fn payload_sha256(&self) -> &str {
        &self.payload
    }
    /// Recovery deliberately preserves orphan temporary files.
    pub fn temporary_cleanup_not_performed(&self) -> bool {
        self.durable.temporary_cleanup_not_performed()
    }
}
/// Owned verified snapshots and that seed's actual receipt; no admission authority.
/// ```compile_fail
/// use ckks_train_bench::study_reference::VerifiedTargetForSeed;
/// let _:VerifiedTargetForSeed=serde_json::from_str("{}").unwrap();
/// ```
#[allow(dead_code)] // Retained real reader custody; live gate/admission wiring is the next dependency.
pub struct VerifiedTargetForSeed {
    lock: crate::study_provenance::publication::HeldRead,
    reference: super::publication::OwnedReferenceSnapshot,
    context: crate::study_provenance::ContextOwnerStamp,
    store: usize,
    file_hash: String,
    receipt: SeedInputReceiptV1,
    threshold: u64,
    payload: String,
}
#[allow(dead_code)] // Retained real reader custody; live gate/admission wiring is the next dependency.
impl VerifiedTargetForSeed {
    // Retained source owner, SAME store/context and current approved policy are
    // required before the inert current+prior composition. No admission token.
    pub(crate) fn check_attempt_history(
        &self,
        store: &ProfileTargetStore<'_>,
        current: &[u8],
        prior: Option<(&[u8], Option<&[u8]>)>,
    ) -> Result<()> {
        require(
            self.store == store as *const ProfileTargetStore<'_> as usize,
            TargetError::ContextChanged,
        )?;
        store
            .context
            .recheck_owner(&self.context)
            .map_err(|_| TargetError::ContextChanged)?;
        store
            .context
            .recheck()
            .map_err(|_| TargetError::ContextChanged)?;
        store
            .inner
            .recheck_read(&self.lock)
            .map_err(|_| TargetError::ByteOrPayloadMismatch)?;
        self.reference
            .recheck(store.context)
            .map_err(reference_error)?;
        let lock = codec::parse_lock(self.lock.bytes())?;
        codec::verify_link(
            &lock,
            self.reference.reference(),
            store.context,
            &store.campaign,
            &store.profile,
        )?;
        super::document::check_verified_target_history(
            current,
            prior,
            &lock,
            self.reference.bytes(),
            self.reference.reference(),
        )?;
        store
            .inner
            .recheck_read(&self.lock)
            .map_err(|_| TargetError::ByteOrPayloadMismatch)?;
        self.reference
            .recheck(store.context)
            .map_err(reference_error)
    }
    /// Exact binary64 threshold bits.
    pub fn threshold_bits(&self) -> u64 {
        self.threshold
    }
    /// Independently derived selected-seed receipt.
    pub fn seed_receipt(&self) -> &SeedInputReceiptV1 {
        &self.receipt
    }
    /// Exact retained lock file digest.
    pub fn lock_file_sha256(&self) -> &str {
        // stored parsed digest is checked against these retained bytes
        &self.file_hash
    }
    /// Canonical lock payload digest.
    pub fn lock_payload_sha256(&self) -> &str {
        &self.payload
    }
}
fn compatible(a: &VerifiedStudyContext, b: &VerifiedStudyContext) -> Result<()> {
    require(
        std::ptr::eq(a, b)
            && a.approval_sha256() == b.approval_sha256()
            && a.source_manifest_sha256() == b.source_manifest_sha256()
            && a.source_commit() == b.source_commit()
            && a.root() == b.root(),
        TargetError::ContextChanged,
    )
}
fn canonical(context: &VerifiedStudyContext, id: &str, seed: u8) -> Result<MaterializedProfile> {
    let registry = StudyRegistry::load(context.root()).map_err(|_| TargetError::ContextChanged)?;
    let resolved = registry
        .resolve(id, seed)
        .map_err(|_| TargetError::MissingInput)?;
    materialize(&resolved).map_err(|_| TargetError::MissingInput)
}
fn selected_matches(
    reference: &VerifiedReference,
    p: &MaterializedProfile,
) -> Result<SeedInputReceiptV1> {
    selected_identity_matches(
        reference.profile_id(),
        reference.contract().canonical_bytes(),
        reference.receipt().canonical_bytes(),
        p,
    )
}
fn selected_identity_matches(
    profile: &str,
    reference_contract: &[u8],
    reference_receipt: &[u8],
    p: &MaterializedProfile,
) -> Result<SeedInputReceiptV1> {
    let (contract, receipt) = derive_identities(p).map_err(|_| TargetError::CaseMismatch)?;
    require(
        p.metadata().id == profile
            && receipt.seed() == p.metadata().seed
            && contract.canonical_bytes() == reference_contract,
        TargetError::CaseMismatch,
    )?;
    if receipt.seed() == 0 {
        require(
            receipt.canonical_bytes() == reference_receipt,
            TargetError::CaseMismatch,
        )?;
    } else {
        require(
            receipt.canonical_bytes() != reference_receipt,
            TargetError::CaseMismatch,
        )?;
    }
    Ok(receipt)
}
impl<'c> ProfileTargetStore<'c> {
    /// Open/create only the closed campaign/profile namespace, under approved Linux policy.
    pub fn open(
        root: &Path,
        campaign: &CampaignId,
        profile: &ResolvedProfile,
        context: &'c VerifiedStudyContext,
    ) -> Result<Self> {
        require(cfg!(target_os = "linux"), TargetError::UnsupportedPlatform)?;
        context.recheck().map_err(|_| TargetError::ContextChanged)?;
        let registry =
            StudyRegistry::load(context.root()).map_err(|_| TargetError::ContextChanged)?;
        registry
            .resolve(&profile.metadata().id, 0)
            .map_err(|_| TargetError::CaseMismatch)?;
        let slot = codec::profile_slot(&profile.metadata().id)?;
        let inner = TargetStore::open(root, &campaign.0, &slot, context)
            .map_err(|_| TargetError::BeforePublication)?;
        Ok(Self {
            inner,
            context,
            campaign: campaign.0.clone(),
            profile: profile.metadata().id.clone(),
        })
    }
    /// Lock a fresh FD and reject any existing target, cases or unsupported profile entry.
    pub fn claim_new<'s>(&'s mut self) -> Result<NewProfileTargetStore<'s, 'c>> {
        self.context
            .require_current_reference()
            .map_err(|_| TargetError::ContextChanged)?;
        self.context
            .recheck()
            .map_err(|_| TargetError::ContextChanged)?;
        let guard = self
            .inner
            .acquire()
            .map_err(|_| TargetError::BeforePublication)?;
        self.inner
            .claim_new(&guard)
            .map_err(|_| TargetError::AlreadyExists)?;
        Ok(NewProfileTargetStore { store: self, guard })
    }
    /// Verify exact lock/reference snapshots and the selected seed's own canonical materialization.
    pub fn read_target(
        &self,
        references: &ReferenceStore<'c>,
        seed_profile: &MaterializedProfile,
    ) -> Result<VerifiedTargetForSeed> {
        let mut clock = NativeClock::new();
        let mut time = ClockTracker::begin(&mut clock)?;
        compatible(self.context, references.context())?;
        self.context
            .recheck()
            .map_err(|_| TargetError::ContextChanged)?;
        let held = self
            .inner
            .read_held()
            .map_err(|_| TargetError::ByteOrPayloadMismatch)?;
        let lock = codec::parse_lock(held.bytes())?;
        let source = references
            .read_reference_snapshot(lock.run_id())
            .map_err(reference_error)?;
        codec::verify_link(
            &lock,
            source.reference(),
            self.context,
            &self.campaign,
            &self.profile,
        )?;
        let actual = canonical(self.context, &self.profile, seed_profile.metadata().seed)?;
        let receipt = selected_matches(source.reference(), &actual)?;
        let supplied = selected_matches(source.reference(), seed_profile)?;
        require(
            receipt.canonical_bytes() == supplied.canonical_bytes(),
            TargetError::CaseMismatch,
        )?;
        let checked = self
            .context
            .check_materialized_input_now(&actual)
            .map_err(|_| TargetError::ContextChanged)?;
        self.context
            .recheck_materialized_input_now(&actual, &checked)
            .map_err(|_| TargetError::ContextChanged)?;
        source.recheck().map_err(reference_error)?;
        time.sample(&mut clock)?;
        self.inner
            .recheck_read(&held)
            .map_err(|_| TargetError::ByteOrPayloadMismatch)?;
        let reference = source.into_owned();
        Ok(VerifiedTargetForSeed {
            lock: held,
            reference,
            context: self.context.owner_stamp(),
            store: self as *const Self as usize,
            receipt,
            threshold: lock.threshold_bits(),
            payload: lock.payload_sha256().into(),
            file_hash: lock.file_sha256().into(),
        })
    }
    /// Verify the exact expected lock and linked source before resync; preserve every temporary name.
    pub fn recover_lock(
        &mut self,
        references: &ReferenceStore<'c>,
        expected_file_sha256: &str,
    ) -> Result<PublishedTargetLock> {
        self.context
            .require_current_reference()
            .map_err(|_| TargetError::ContextChanged)?;
        references
            .context()
            .require_current_reference()
            .map_err(|_| TargetError::ContextChanged)?;
        let mut clock = NativeClock::new();
        let mut time = ClockTracker::begin(&mut clock)?;
        compatible(self.context, references.context())?;
        require(
            expected_file_sha256.len() == 64
                && expected_file_sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            TargetError::ByteOrPayloadMismatch,
        )?;
        let guard = self
            .inner
            .acquire()
            .map_err(|_| TargetError::BeforePublication)?;
        let bytes = self
            .inner
            .read()
            .map_err(|_| TargetError::ByteOrPayloadMismatch)?;
        require(
            sha256_hex(&bytes) == expected_file_sha256,
            TargetError::ByteOrPayloadMismatch,
        )?;
        let lock = codec::parse_lock(&bytes)?;
        let source = references
            .read_reference_snapshot(lock.run_id())
            .map_err(reference_error)?;
        codec::verify_link(
            &lock,
            source.reference(),
            self.context,
            &self.campaign,
            &self.profile,
        )?;
        source.recheck().map_err(reference_error)?;
        self.context
            .recheck()
            .map_err(|_| TargetError::ContextChanged)?;
        time.sample(&mut clock)?;
        let durable = self
            .inner
            .recover(&guard, expected_file_sha256, |current| {
                if current != bytes {
                    return Err(crate::study_provenance::ProvenanceError("lock changed"));
                }
                source
                    .recheck()
                    .map_err(|_| crate::study_provenance::ProvenanceError("reference changed"))?;
                self.context.recheck()
            })
            .map_err(|_| uncertain(&self.inner, &bytes, lock.payload_sha256()))?;
        let durable = confirm_publication(
            durable,
            || {
                source.recheck().map_err(reference_error)?;
                self.context
                    .recheck()
                    .map_err(|_| TargetError::ContextChanged)?;
                time.sample(&mut clock)?;
                Ok(())
            },
            || uncertain(&self.inner, &bytes, lock.payload_sha256()),
        )?;
        Ok(PublishedTargetLock {
            durable,
            payload: lock.payload_sha256().into(),
        })
    }
}
fn uncertain(store: &TargetStore, bytes: &[u8], payload: &str) -> TargetError {
    TargetError::PublishedButDurabilityUnconfirmed {
        path: store.path(),
        file_sha256: sha256_hex(bytes),
        payload_sha256: payload.into(),
    }
}
/// Consume the origin-bound seed0 snapshot and exclusive profile claim; only durable success returns.
/// ```compile_fail
/// use ckks_train_bench::study_reference::{VerifiedReference,NewProfileTargetStore,lock_target};
/// fn bare(reference:VerifiedReference,claim:NewProfileTargetStore<'_,'_>) {let _=lock_target(reference,claim);}
/// ```
pub fn lock_target<'r, 's, 'c: 'r + 's>(
    source: VerifiedReferenceSnapshot<'r, 'c>,
    new_store: NewProfileTargetStore<'s, 'c>,
) -> Result<PublishedTargetLock> {
    source
        .context()
        .require_current_reference()
        .map_err(|_| TargetError::ContextChanged)?;
    new_store
        .store
        .context
        .require_current_reference()
        .map_err(|_| TargetError::ContextChanged)?;
    let mut clock = NativeClock::new();
    let mut time = ClockTracker::begin(&mut clock)?;
    let store = new_store.store;
    compatible(source.context(), store.context)?;
    require(
        source.reference().receipt().seed() == 0
            && source.reference().profile_id() == store.profile,
        TargetError::CaseMismatch,
    )?;
    source.recheck().map_err(reference_error)?;
    let materialized = canonical(store.context, &store.profile, 0)?;
    selected_matches(source.reference(), &materialized)?;
    let start = store
        .context
        .capture(&materialized)
        .map_err(|_| TargetError::ContextChanged)?;
    let end = store
        .context
        .finish_capture(&materialized, &start)
        .map_err(|_| TargetError::ContextChanged)?;
    let capture = CaptureFact {
        context: store.context,
        start,
        end,
    };
    let (wall, elapsed) = time.sample(&mut clock)?;
    require(
        source.reference().verified_time().start_unix_ns()
            <= source.reference().verified_time().completion_unix_ns()
            && source.reference().verified_time().completion_unix_ns() <= wall,
        TargetError::ClockInvalid,
    )?;
    let bytes = codec::serialize_lock(
        source.reference(),
        source.bytes(),
        &capture,
        &store.campaign,
        wall,
        elapsed,
    )?;
    let lock = codec::parse_lock(&bytes)?;
    source.recheck().map_err(reference_error)?;
    store
        .context
        .finish_capture(&materialized, &capture.start)
        .map_err(|_| TargetError::ContextChanged)?;
    time.sample(&mut clock)?;
    let durable = store
        .inner
        .publish(&new_store.guard, &bytes)
        .map_err(|e| match e {
            PublishError::BeforeLink => TargetError::BeforePublication,
            PublishError::VisibleUnconfirmed => {
                uncertain(&store.inner, &bytes, lock.payload_sha256())
            }
        })?;
    let durable = confirm_publication(
        durable,
        || {
            source.recheck().map_err(reference_error)?;
            store
                .context
                .finish_capture(&materialized, &capture.start)
                .map_err(|_| TargetError::ContextChanged)?;
            time.sample(&mut clock)?;
            Ok(())
        },
        || uncertain(&store.inner, &bytes, lock.payload_sha256()),
    )?;
    Ok(PublishedTargetLock {
        durable,
        payload: lock.payload_sha256().into(),
    })
}
// The same final boundary is exercised by synthetic fault tests. It never grants a
// result after a late source/context/clock failure, even after the final link exists.
fn confirm_publication<T>(
    value: T,
    after: impl FnOnce() -> Result<()>,
    uncertainty: impl FnOnce() -> TargetError,
) -> Result<T> {
    after().map_err(|_| uncertainty())?;
    Ok(value)
}
trait Clock {
    fn wall(&mut self) -> Result<u64>;
    fn monotonic(&mut self) -> Result<u64>;
}
struct NativeClock {
    start: Instant,
}
impl NativeClock {
    fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}
impl Clock for NativeClock {
    fn wall(&mut self) -> Result<u64> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|d| u64::try_from(d.as_nanos()).ok())
            .ok_or(TargetError::ClockInvalid)
    }
    fn monotonic(&mut self) -> Result<u64> {
        Instant::now()
            .checked_duration_since(self.start)
            .and_then(|d| u64::try_from(d.as_nanos()).ok())
            .ok_or(TargetError::ClockInvalid)
    }
}
struct ClockTracker {
    wall: u64,
    monotonic: u64,
    start: u64,
    failed: bool,
}
impl ClockTracker {
    fn begin(clock: &mut impl Clock) -> Result<Self> {
        let wall = clock.wall()?;
        let monotonic = clock.monotonic()?;
        Ok(Self {
            wall,
            monotonic,
            start: monotonic,
            failed: false,
        })
    }
    fn sample(&mut self, clock: &mut impl Clock) -> Result<(u64, u64)> {
        require(!self.failed, TargetError::ClockInvalid)?;
        self.failed = true;
        let wall = clock.wall()?;
        let monotonic = clock.monotonic()?;
        require(
            wall >= self.wall && monotonic >= self.monotonic,
            TargetError::ClockInvalid,
        )?;
        let elapsed = monotonic
            .checked_sub(self.start)
            .ok_or(TargetError::ClockInvalid)?;
        self.wall = wall;
        self.monotonic = monotonic;
        self.failed = false;
        Ok((wall, elapsed))
    }
}
#[cfg(test)]
#[path = "target_tests.rs"]
mod tests;

// Compile-only proof that the installed outer context/input frame can own the
// returned retained snapshot after both shorter store objects are destroyed.
// Not a test function: no reader/materializer invocation occurs during tests.
#[cfg(test)]
#[allow(dead_code)]
fn retained_target_after_short_stores<'c>(
    target: ProfileTargetStore<'c>,
    references: ReferenceStore<'c>,
    profile: &MaterializedProfile,
) -> Result<VerifiedTargetForSeed> {
    let snapshot = target.read_target(&references, profile)?;
    drop(target);
    drop(references);
    Ok(snapshot)
}
