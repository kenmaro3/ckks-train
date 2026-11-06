//! Capabilities are granted only after the reviewed durable Store protocol succeeds.
use super::{
    DocumentError, VerifiedReference, document,
    session::{CapturedDiagnostic, CapturedF64Diagnostic, CompletedFp32Reference, SessionFailure},
};
use crate::study_provenance::{
    VerifiedStudyContext,
    publication::{DurableBytes, Kind, PublishError, Store},
};
use std::path::{Path, PathBuf};
const CAP: usize = 70_492_488;
/// Publication failure with no scientific durability capability.
#[derive(Debug)]
pub enum PublicationError {
    /// Context, path or recovery verification failed.
    Validation,
    /// Bounded schema serialization or verification failed.
    Document(DocumentError),
    /// No final link was published by this attempt.
    BeforeLink,
    /// The final file may be visible, but durability/cleanup is unconfirmed.
    PublishedButDurabilityUnconfirmed {
        /// Visible final path.
        path: PathBuf,
        /// Exact attempted byte digest.
        file_sha256: String,
        /// Canonical payload digest.
        payload_sha256: String,
    },
}

impl std::fmt::Display for PublicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Document(e) => write!(f, "document: {e}"),
            Self::Validation => f.write_str("publication validation failed"),
            Self::BeforeLink => f.write_str("publication failed before link"),
            Self::PublishedButDurabilityUnconfirmed { .. } => {
                f.write_str("visible publication durability unconfirmed")
            }
        }
    }
}
impl std::error::Error for PublicationError {}
impl From<DocumentError> for PublicationError {
    fn from(e: DocumentError) -> Self {
        Self::Document(e)
    }
}
/// Durable scientific reference publication, not a target lock or campaign admission.
pub struct PublishedReference {
    durable: DurableBytes,
    payload: String,
    approval: String,
}
impl PublishedReference {
    #[doc = "Dedicated final file path."]
    pub fn path(&self) -> &Path {
        self.durable.path()
    }
    #[doc = "SHA256 of the exact immutable file bytes."]
    pub fn file_sha256(&self) -> &str {
        self.durable.file_sha256()
    }
    #[doc = "SHA256 of the canonical typed document payload."]
    pub fn payload_sha256(&self) -> &str {
        &self.payload
    }
    #[doc = "Pinned local approval digest used for this operation."]
    pub fn approval_sha256(&self) -> &str {
        &self.approval
    }
    #[doc = "True for recovery, which deliberately preserves every temporary name."]
    pub fn temporary_cleanup_not_performed(&self) -> bool {
        self.durable.temporary_cleanup_not_performed()
    }
}
/// Diagnostic/failure durability never grants reference/target readiness.
pub struct PublishedNonReference {
    durable: DurableBytes,
}
impl PublishedNonReference {
    #[doc = "Dedicated final file path."]
    pub fn path(&self) -> &Path {
        self.durable.path()
    }
    #[doc = "SHA256 of the exact immutable file bytes."]
    pub fn file_sha256(&self) -> &str {
        self.durable.file_sha256()
    }
}
/// Dedicated root with approved descriptor-relative nofollow, bounded, local-fs policy.
pub struct ReferenceStore<'c> {
    inner: Store,
    context: &'c VerifiedStudyContext,
    root: PathBuf,
}
impl<'c> ReferenceStore<'c> {
    #[doc = "Open a prepared private local ext4/XFS root under the approved context."]
    pub fn open(
        root: impl AsRef<Path>,
        context: &'c VerifiedStudyContext,
    ) -> Result<Self, PublicationError> {
        let root = root.as_ref();
        let inner = Store::open(root, context, Kind::Reference)
            .map_err(|_| PublicationError::Validation)?;
        Ok(Self {
            inner,
            context,
            root: root.into(),
        })
    }
    #[doc = "Boundedly read and fully verify a reference against exact approved canonical inputs."]
    pub fn read_reference(&self, id: &str) -> Result<VerifiedReference, PublicationError> {
        let bytes = self
            .inner
            .read(id, CAP)
            .map_err(|_| PublicationError::Validation)?;
        Ok(document::verify_reference(&bytes, self.context, id)?)
    }
    #[doc = "Recheck provenance, then publish without overwrite; only durable success grants a capability."]
    pub fn publish_reference(
        &self,
        result: &CompletedFp32Reference<'_>,
    ) -> Result<PublishedReference, PublicationError> {
        if !std::ptr::eq(self.context, result.capture().context) {
            return Err(PublicationError::Validation);
        }
        let bytes = result.document_bytes()?;
        let parsed = super::parse_reference_document(&bytes)?;
        result.recheck().map_err(|_| PublicationError::Validation)?;
        let durable = self.publish(
            &self.inner,
            "references",
            "reference",
            &result.record().run_id,
            &bytes,
            parsed.payload_sha256(),
        )?;
        Ok(PublishedReference {
            durable,
            payload: parsed.payload_sha256().into(),
            approval: self.context.approval_sha256().into(),
        })
    }
    #[doc = "Durably publish a production-captured diagnostic; it remains reference-ineligible."]
    pub fn publish_diagnostic(
        &self,
        result: &CapturedDiagnostic<'_>,
    ) -> Result<PublishedNonReference, PublicationError> {
        if !std::ptr::eq(self.context, result.capture().context) {
            return Err(PublicationError::Validation);
        }
        let bytes = result.document_bytes()?;
        let parsed = super::parse_diagnostic_document(&bytes)?;
        let store = Store::open(&self.root, self.context, Kind::Diagnostic)
            .map_err(|_| PublicationError::Validation)?;
        result.recheck().map_err(|_| PublicationError::Validation)?;
        Ok(PublishedNonReference {
            durable: self.publish(
                &store,
                "diagnostics",
                "diagnostic",
                &result.record().run_id,
                &bytes,
                parsed.payload_sha256(),
            )?,
        })
    }
    /// Publish a separately typed F64 diagnostic without reference readiness.
    pub fn publish_f64_diagnostic(
        &self,
        result: &CapturedF64Diagnostic<'_>,
    ) -> Result<PublishedNonReference, PublicationError> {
        self.publish_diagnostic(result.inner())
    }
    #[doc = "Best-effort durable failure publication, independent of scientific completion."]
    pub fn publish_failure(
        &self,
        result: &SessionFailure,
    ) -> Result<PublishedNonReference, PublicationError> {
        let bytes = result.document_bytes()?;
        let parsed = super::parse_failure_document(&bytes)?;
        let store = Store::open(&self.root, self.context, Kind::Failure)
            .map_err(|_| PublicationError::Validation)?;
        self.context
            .recheck()
            .map_err(|_| PublicationError::Validation)?;
        Ok(PublishedNonReference {
            durable: self.publish(
                &store,
                "failures",
                "failure",
                &result.detail().run_id,
                &bytes,
                parsed.payload_sha256(),
            )?,
        })
    }
    fn publish(
        &self,
        store: &Store,
        directory: &str,
        extension: &str,
        id: &str,
        bytes: &[u8],
        payload: &str,
    ) -> Result<DurableBytes, PublicationError> {
        store.publish(id, bytes, CAP).map_err(|e| match e {
            PublishError::BeforeLink => PublicationError::BeforeLink,
            PublishError::VisibleUnconfirmed => {
                PublicationError::PublishedButDurabilityUnconfirmed {
                    path: self
                        .root
                        .join(directory)
                        .join(format!("{id}.{extension}.json")),
                    file_sha256: crate::data::sha256_hex(bytes),
                    payload_sha256: payload.into(),
                }
            }
        })
    }
    /// Reopens and normally verifies exact bytes before fsync; never cleans orphan temps.
    #[doc = "Verify exact expected file bytes before sync; preserve every orphan temporary name."]
    pub fn recover_reference(
        &self,
        id: &str,
        expected_file_sha256: &str,
    ) -> Result<PublishedReference, PublicationError> {
        let mut verified = None;
        let durable = self
            .inner
            .recover(id, expected_file_sha256, CAP, |bytes| {
                verified = Some(document::verify_reference(bytes, self.context, id).map_err(
                    |_| crate::study_provenance::ProvenanceError("reference recovery verification"),
                )?);
                Ok(())
            })
            .map_err(|_| PublicationError::Validation)?;
        let verified = verified.ok_or(PublicationError::Validation)?;
        Ok(PublishedReference {
            durable,
            payload: verified.payload_sha256().into(),
            approval: verified.approval_sha256().into(),
        })
    }
}

/// Exact single-read verified snapshot borrowing its originating store.
/// It cannot be attached to a different store or built from a bare verification.
/// ```compile_fail
/// use ckks_train_bench::study_reference::{VerifiedReference,VerifiedReferenceSnapshot};
/// fn forge(v: VerifiedReference) -> VerifiedReferenceSnapshot<'static,'static> { v.into() }
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_reference::VerifiedReferenceSnapshot;
/// fn duplicate(s: VerifiedReferenceSnapshot<'_,'_>) { let _ = s.clone(); }
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_reference::VerifiedReferenceSnapshot;
/// let _: VerifiedReferenceSnapshot<'static,'static> = serde_json::from_str("{}").unwrap();
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_reference::ReferenceStore;
/// fn lifetime(store: ReferenceStore<'_>) {
///     let snapshot = store.read_reference_snapshot("reference").unwrap();
///     drop(store); drop(snapshot);
/// }
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_reference::VerifiedReferenceSnapshot;
/// fn construct() { let _ = VerifiedReferenceSnapshot { origin: todo!(), bytes: todo!(), verified: todo!(), stamp: todo!() }; }
/// ```
pub struct VerifiedReferenceSnapshot<'r, 'c: 'r> {
    origin: &'r ReferenceStore<'c>,
    bytes: Box<[u8]>,
    verified: VerifiedReference,
    stamp: crate::study_provenance::publication::ReadStamp,
}
impl<'c> ReferenceStore<'c> {
    /// Read exactly once through the held origin, verify normally and retain inode/namespace stamps.
    pub fn read_reference_snapshot<'r>(
        &'r self,
        id: &str,
    ) -> Result<VerifiedReferenceSnapshot<'r, 'c>, PublicationError> {
        let (bytes, stamp) = self
            .inner
            .read_stamped(id, CAP)
            .map_err(|_| PublicationError::Validation)?;
        let verified = document::verify_reference(&bytes, self.context, id)?;
        if verified.receipt().seed() != 0 {
            return Err(PublicationError::Validation);
        }
        Ok(VerifiedReferenceSnapshot {
            origin: self,
            bytes: bytes.into_boxed_slice(),
            verified,
            stamp,
        })
    }
    pub(super) fn context(&self) -> &'c VerifiedStudyContext {
        self.context
    }
}
impl VerifiedReferenceSnapshot<'_, '_> {
    pub(super) fn recheck(&self) -> Result<(), PublicationError> {
        let (bytes, stamp) = self
            .origin
            .inner
            .read_stamped(self.verified.run_id(), CAP)
            .map_err(|_| PublicationError::Validation)?;
        if bytes.as_slice() != self.bytes.as_ref() || stamp != self.stamp {
            return Err(PublicationError::Validation);
        }
        self.origin
            .context
            .recheck()
            .map_err(|_| PublicationError::Validation)
    }
    pub(super) fn context(&self) -> &VerifiedStudyContext {
        self.origin.context
    }
    pub(super) fn reference(&self) -> &VerifiedReference {
        &self.verified
    }
    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub(super) fn into_owned(self) -> (Box<[u8]>, VerifiedReference) {
        (self.bytes, self.verified)
    }
}
