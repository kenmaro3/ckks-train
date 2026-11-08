//! Six closed reference/target roots, under the same observed operational owner.
use crate::{
    study_provenance::{ContextRequest, VerifiedStudyContext},
    study_reference::{
        CampaignId, ProfileTargetStore, ReferenceStore, lock_target, operational::Requested,
    },
};
use std::{collections::BTreeMap, ffi::OsString, os::unix::ffi::OsStrExt, path::PathBuf};
/// A bounded frontend refusal, with no scientific or gate capability.
#[derive(Debug)]
pub struct PreparationError(&'static str);
impl std::fmt::Display for PreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for PreparationError {}
type Result<T> = std::result::Result<T, PreparationError>;
fn require(ok: bool, why: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(PreparationError(why))
    }
}
fn identifier(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}
fn hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
struct Request {
    root: PathBuf,
    approval: PathBuf,
    approval_hash: String,
    receipts: Vec<PathBuf>,
    operation: u8,
    profile: String,
    seed: u64,
    reference_root: PathBuf,
    reference_id: String,
    target_root: Option<PathBuf>,
    campaign: Option<String>,
    expected: Option<String>,
}
impl Request {
    fn seeds(&self) -> Vec<u64> {
        if self.seed == 0 {
            vec![0]
        } else {
            vec![0, self.seed]
        }
    }
    fn work(&self) -> Result<Requested> {
        Requested::preparation(self.operation, &self.profile, &self.seeds())
            .map_err(|_| PreparationError("fixed preparation work invalid"))
    }
}
fn parse(args: Vec<OsString>) -> Result<Request> {
    require(
        args.len() < 256
            && args.len() % 2 == 0
            && args
                .iter()
                .all(|a| a.as_bytes().len() <= 4096 && !a.as_bytes().contains(&0)),
        "preparation argv bounds",
    )?;
    let mut fields = BTreeMap::new();
    let mut receipts = Vec::new();
    for pair in args.chunks_exact(2) {
        let key = pair[0]
            .to_str()
            .ok_or(PreparationError("preparation option UTF8"))?;
        require(
            matches!(
                key,
                "--root"
                    | "--approval"
                    | "--approval-sha256"
                    | "--receipt"
                    | "--operation"
                    | "--profile"
                    | "--seed"
                    | "--reference-root"
                    | "--reference-id"
                    | "--target-root"
                    | "--campaign"
                    | "--expected-sha256"
            ),
            "unknown preparation option",
        )?;
        if key == "--receipt" {
            require(
                receipts.len() < 64 && !pair[1].is_empty(),
                "preparation receipt bounds",
            )?;
            receipts.push(PathBuf::from(&pair[1]));
        } else {
            require(
                fields.insert(key.to_owned(), pair[1].clone()).is_none(),
                "duplicate preparation option",
            )?;
        }
    }
    let get = |key: &str| {
        fields
            .get(key)
            .ok_or(PreparationError("missing explicit preparation option"))
    };
    let text = |key: &str| {
        get(key)?
            .to_str()
            .ok_or(PreparationError("preparation semantic UTF8"))
    };
    let operation = match text("--operation")? {
        "reference-read" => 0,
        "reference-recover" => 1,
        "reference-snapshot" => 2,
        "target-read" => 3,
        "target-recover" => 4,
        "lock-target" => 5,
        _ => return Err(PreparationError("unknown preparation operation")),
    };
    let seed = text("--seed")?;
    require(
        seed.len() == 1 && seed.as_bytes()[0].is_ascii_digit(),
        "preparation seed grammar",
    )?;
    let seed = u64::from(seed.as_bytes()[0] - b'0');
    require(
        operation == 3 || seed == 0,
        "preparation operation requires seed0",
    )?;
    let profile = text("--profile")?.to_owned();
    let reference_id = text("--reference-id")?.to_owned();
    let approval_hash = text("--approval-sha256")?.to_owned();
    require(
        identifier(&profile)
            && identifier(&reference_id)
            && hash(&approval_hash)
            && !receipts.is_empty(),
        "preparation identity/pins",
    )?;
    let (target_root, campaign) = if operation >= 3 {
        let c = text("--campaign")?.to_owned();
        require(identifier(&c), "preparation campaign")?;
        (Some(PathBuf::from(get("--target-root")?)), Some(c))
    } else {
        require(
            !fields.contains_key("--campaign") && !fields.contains_key("--target-root"),
            "unused target options",
        )?;
        (None, None)
    };
    let expected = if matches!(operation, 1 | 4) {
        let h = text("--expected-sha256")?.to_owned();
        require(hash(&h), "recovery exact hash")?;
        Some(h)
    } else {
        require(
            !fields.contains_key("--expected-sha256"),
            "unused recovery option",
        )?;
        None
    };
    Ok(Request {
        root: PathBuf::from(get("--root")?),
        approval: PathBuf::from(get("--approval")?),
        approval_hash,
        receipts,
        operation,
        profile,
        seed,
        reference_root: PathBuf::from(get("--reference-root")?),
        reference_id,
        target_root,
        campaign,
        expected,
    })
}
/// Enter the supervisor before loading context, reading stores or reconstructing inputs.
/// This source stage refuses at any missing retained input/store dependency.
pub fn run(args: Vec<OsString>) -> Result<()> {
    if args.first().is_some_and(|a| a == "--private-helper") {
        return crate::study_supervisor::helper(&args).map_err(|e| PreparationError(e.0));
    }
    let private = args.first().is_some_and(|a| a == "--private-worker");
    let r = parse(if private {
        args[1..].to_vec()
    } else {
        args.clone()
    })?;
    let work = r.work()?;
    if !private {
        return crate::study_supervisor::launch(
            &r.root,
            &r.approval,
            &r.approval_hash,
            &r.receipts,
            &args,
            work,
        )
        .map_err(|e| PreparationError(e.0));
    }
    let owner = crate::study_supervisor::worker(&args, &r.root, &r.approval_hash, work)
        .map_err(|e| PreparationError(e.0))?;
    let seeds = r.seeds();
    let context = VerifiedStudyContext::load_observed(
        ContextRequest::new(r.root, r.approval, r.approval_hash, r.receipts),
        &owner,
    )
    .map_err(|e| PreparationError(e.0))?;
    let registry = crate::study_profile::StudyRegistry::load(context.root())
        .map_err(|_| PreparationError("preparation registry unavailable"))?;
    require(
        registry.profile_ids().contains(&r.profile),
        "preparation profile unregistered",
    )?;
    owner
        .bind_preparation(r.operation, &r.profile, &seeds)
        .map_err(|e| PreparationError(e.0))?;
    // Input0/Reviewed2 must be actually retained under this same owner before
    // resolve/read/verification can reconstruct even a different file's profile.
    let scope = owner
        .preparation_scope()
        .map_err(|e| PreparationError(e.0))?;
    scope
        .before_read(&context, r.operation, &r.profile, r.seed)
        .map_err(|e| PreparationError(e.0))?;
    let references = ReferenceStore::open(&r.reference_root, &context)
        .map_err(|_| PreparationError("retained Reference store unavailable"))?;
    match r.operation {
        0 => {
            references
                .read_reference(&r.reference_id)
                .map_err(|_| PreparationError("reference read failed"))?;
        }
        1 => {
            references
                .recover_reference(
                    &r.reference_id,
                    r.expected.as_deref().expect("recovery hash"),
                )
                .map_err(|_| PreparationError("reference recovery failed"))?;
        }
        2 => {
            let snapshot = references
                .read_reference_snapshot(&r.reference_id)
                .map_err(|_| PreparationError("reference snapshot failed"))?;
            drop(snapshot);
        }
        operation => {
            let profile = registry
                .resolve(&r.profile, r.seed as u8)
                .map_err(|_| PreparationError("target selected profile unavailable"))?;
            let campaign = CampaignId::parse(r.campaign.as_deref().expect("target campaign"))
                .map_err(|_| PreparationError("target campaign invalid"))?;
            let mut targets = ProfileTargetStore::open(
                r.target_root.as_deref().expect("target root"),
                &campaign,
                &profile,
                &context,
            )
            .map_err(|_| PreparationError("retained Target store unavailable"))?;
            match operation {
                3 => {
                    let input = crate::study_inputs::materialize(&profile)
                        .map_err(|_| PreparationError("selected target input unavailable"))?;
                    targets
                        .read_target(&references, &input)
                        .map_err(|_| PreparationError("target read failed"))?;
                }
                4 => {
                    targets
                        .recover_lock(&references, r.expected.as_deref().expect("recovery hash"))
                        .map_err(|_| PreparationError("target recovery failed"))?;
                }
                5 => {
                    let snapshot = references
                        .read_reference_snapshot(&r.reference_id)
                        .map_err(|_| PreparationError("root5 snapshot unavailable"))?;
                    let guard = targets
                        .claim_new()
                        .map_err(|_| PreparationError("target short guard unavailable"))?;
                    lock_target(snapshot, guard)
                        .map_err(|_| PreparationError("target lock publication failed"))?;
                }
                _ => return Err(PreparationError("closed preparation operation")),
            }
        }
    }
    owner.check().map_err(|e| PreparationError(e.0))
}
#[cfg(test)]
#[path = "study_prepare_tests.rs"]
mod tests;
