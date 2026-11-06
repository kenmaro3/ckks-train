//! Expected registrations only. No input resolution, runtime, allocation or readiness authority.
use super::{APPROVED_REGISTRY_SHA256, StudyRegistry};
use crate::{data::sha256_hex, study_reference::DocumentError};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    str::FromStr,
};

/// A malformed reviewed projection or an unmatched request, never a readiness result.
#[derive(Debug)]
pub enum CaseError {
    /// The registry's declared axes conflict with the closed registration contract.
    InvalidRegistry(&'static str),
    /// A value is outside the closed vocabulary.
    UnknownValue,
    /// The request is not a concrete prospective full registration.
    Unregistered,
    /// Existing bounded canonical codec rejected the identity.
    Canonical(DocumentError),
}
impl fmt::Display for CaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CaseError {}
type Result<T> = std::result::Result<T, CaseError>;
fn require(ok: bool, why: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(CaseError::InvalidRegistry(why))
    }
}
macro_rules! vocabulary {
    ($name:ident { $($variant:ident = $tag:literal => $text:literal),+ $(,)? }) => {
        #[doc = "Closed registered-case vocabulary; values grant no execution authority."]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        #[repr(u8)]
        pub enum $name { $(#[doc = $text] $variant = $tag),+ }
        impl $name {
            /// Exact registration spelling.
            pub fn as_str(self) -> &'static str { match self { $(Self::$variant => $text),+ } }
        }
        impl FromStr for $name {
            type Err = CaseError;
            fn from_str(s: &str) -> Result<Self> { match s { $($text => Ok(Self::$variant)),+, _ => Err(CaseError::UnknownValue) } }
        }
    }
}
vocabulary!(CaseOrigin { Family=0=>"family", Supplement=1=>"supplement" });
vocabulary!(CaseScope { Full=0=>"full", Quick=1=>"quick", ArchivalFull=2=>"archival_full" });
vocabulary!(CaseRole { Main=0=>"main", BaselineA=1=>"baseline_a", BaselineB=2=>"baseline_b", BaselineC=3=>"baseline_c", MainAblation=4=>"main_ablation" });
vocabulary!(CaseBackend { Cpu=0=>"cpu", Cuda=1=>"cuda", ExternalCpu=2=>"external_cpu" });
vocabulary!(CaseMode { Interactive=0=>"interactive", Noninteractive=1=>"noninteractive" });
vocabulary!(CasePlacement { ClientRefresh=0=>"client_refresh", WhenNeeded=1=>"when_needed", AfterUpdate=2=>"after_update", AfterActivation=3=>"after_activation", UpstreamVerified=4=>"upstream_verified" });
vocabulary!(CasePacking { BatchedDiagonal=0=>"batched_diagonal", Row2020=1=>"row_2020", Diagonal2020=2=>"diagonal_2020", AlternatingRebootRowColumn=3=>"alternating_reboot_row_column", OfficialUpstream=4=>"official_upstream" });
vocabulary!(SecurityPolicy { N16Train=0=>"n16_train", N16Boot=1=>"n16_boot", Seal2020=2=>"seal_2020", UpstreamVerified=3=>"upstream_verified", ReducedLevelSupplement=4=>"reduced_level_supplement" });

/// Explicit untrusted request. Role, cryptography, batch and ablation flags cannot be supplied.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CaseSelector {
    /// Family or supplement namespace.
    pub origin: CaseOrigin,
    /// Exact registration ID, never a path.
    pub registration_id: String,
    /// Exact concrete profile ID.
    pub profile_id: String,
    /// Full statistical seed, restricted to 0..9 by selection.
    pub seed: u64,
    /// Quick and archival requests cannot select a RegisteredCase.
    pub scope: CaseScope,
    /// Declared backend, independent of availability.
    pub backend: CaseBackend,
    /// Declared training mode.
    pub mode: CaseMode,
    /// Declared refresh placement.
    pub placement: CasePlacement,
    /// Declared packing variant.
    pub packing: CasePacking,
}
/// Inspection of declared ablation flags; cannot construct a registration.
///
/// ```compile_fail
/// use ckks_train_bench::study_profile::{CaseSelector, CaseEvalConfig};
/// fn inject(request: &mut CaseSelector, flags: CaseEvalConfig) { request.eval_config = Some(flags); }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaseEvalConfig {
    /// Hoisting flag.
    pub hoist: bool,
    /// Baby-step/giant-step flag.
    pub bsgs: bool,
    /// Lazy relinearization flag.
    pub lazy_relin: bool,
}

/// Immutable registry-derived identity, with no execution/readiness capability.
/// No public constructor, deserializer, mutable selector or mutable canonical bytes exists.
///
/// ```compile_fail
/// use ckks_train_bench::study_profile::RegisteredCase;
/// let forged = RegisteredCase::default();
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_profile::RegisteredCase;
/// let forged: RegisteredCase = serde_json::from_str("{}").unwrap();
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_profile::RegisteredCase;
/// fn mutate(case: &mut RegisteredCase) { case.selector.seed = 2020; }
/// ```
#[derive(Debug)]
pub struct RegisteredCase {
    selector: CaseSelector,
    role: CaseRole,
    security: SecurityPolicy,
    batch: u64,
    flags: Option<CaseEvalConfig>,
    bytes: Vec<u8>,
    digest: String,
}
impl RegisteredCase {
    /// Exact full request matched by this identity; immutable and non-authorizing.
    pub fn selector(&self) -> &CaseSelector {
        &self.selector
    }
    /// Registry pin verified before projection.
    pub fn registry_sha256(&self) -> &str {
        APPROVED_REGISTRY_SHA256
    }
    /// Registered role, distinct even when later computation could be shared.
    pub fn role(&self) -> CaseRole {
        self.role
    }
    /// Registered security policy, not verified execution parameters.
    pub fn security_policy(&self) -> SecurityPolicy {
        self.security
    }
    /// Logical batch from the concrete profile.
    pub fn logical_batch(&self) -> u64 {
        self.batch
    }
    /// Registered ablation flags; absent for all current concrete families and exact Iris.
    pub fn eval_config(&self) -> Option<CaseEvalConfig> {
        self.flags
    }
    /// Existing document codec's exact registered-case domain and ordered payload.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// SHA256 of canonical bytes, never of JSON or Debug.
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

/// Descriptor category, not a readiness or admission status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupplementKind {
    /// Twenty prospective concrete identities plus a separate historical anchor.
    ExactIris,
    /// Preserved FULL-1 metadata only.
    ArchivalFull,
    /// Weighted B48 split32+16 with unresolved concrete profile binding.
    WeightedPartial,
    /// Separate top7/partial-top8 comparison with unresolved concrete binding.
    ReducedLevel,
    /// All eight flag tuples and three backend/level groups, unresolved binding.
    Rq3AllEight,
    /// Separate plaintext role.
    PlainReferences,
    /// Deferred activation study.
    DeferredActivation,
}
/// Complete immutable reviewed raw descriptor, with typed inspection only.
#[derive(Debug)]
pub struct SupplementDescriptor {
    id: String,
    kind: SupplementKind,
    raw: Value,
    flags: Vec<CaseEvalConfig>,
}
impl SupplementDescriptor {
    /// Exact registration ID.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Registration category, never execution availability.
    pub fn kind(&self) -> SupplementKind {
        self.kind
    }
    /// Every original reviewed field/value. Cannot mutate the registry.
    pub fn raw(&self) -> &Value {
        &self.raw
    }
    /// All eight RQ3 declarations; empty for descriptors with no ablation axis.
    pub fn ablation_flags(&self) -> &[CaseEvalConfig] {
        &self.flags
    }
}

/// Complete concrete sets plus all seven descriptors; not a whole-study denominator.
#[derive(Debug)]
pub struct RegisteredCases {
    families: Vec<RegisteredCase>,
    iris: Vec<RegisteredCase>,
    supplements: Vec<SupplementDescriptor>,
}
impl RegisteredCases {
    /// All 2,680 full family rows, including missing-input and unavailable implementations.
    pub fn family_cases(&self) -> &[RegisteredCase] {
        &self.families
    }
    /// All twenty prospective exact-Iris rows. Historical seed2020 is excluded.
    pub fn exact_iris_cases(&self) -> &[RegisteredCase] {
        &self.iris
    }
    /// All seven original supplement entries, preserving incomplete and deferred scope.
    pub fn supplements(&self) -> &[SupplementDescriptor] {
        &self.supplements
    }
    /// Seed0 family requests for quick inspection only; none can enter full selection.
    pub fn quick_family_selectors(&self) -> Vec<CaseSelector> {
        self.families
            .iter()
            .filter(|c| c.selector.seed == 0)
            .map(|c| {
                let mut s = c.selector.clone();
                s.scope = CaseScope::Quick;
                s
            })
            .collect()
    }
    /// Exact full membership. Quick, archival, unresolved and unmatched requests fail.
    pub fn select(&self, selector: &CaseSelector) -> Result<&RegisteredCase> {
        if selector.scope != CaseScope::Full || selector.seed > 9 {
            return Err(CaseError::Unregistered);
        }
        self.families
            .iter()
            .chain(&self.iris)
            .find(|c| &c.selector == selector)
            .ok_or(CaseError::Unregistered)
    }
}
impl StudyRegistry {
    /// Pure projection from this already hash-verified registry, without resolving inputs.
    pub fn registered_cases(&self) -> Result<RegisteredCases> {
        project(&self.value)
    }
}

fn text<'a>(value: &'a Value, key: &'static str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(CaseError::InvalidRegistry(key))
}
fn items<'a>(value: &'a Value, key: &'static str) -> Result<&'a [Value]> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or(CaseError::InvalidRegistry(key))
}
fn axis<'a>(value: &'a Value, key: &'static str) -> Result<Vec<&'a str>> {
    let mut seen = BTreeSet::new();
    let values = items(value, key)?
        .iter()
        .map(|x| {
            let s = x.as_str().ok_or(CaseError::InvalidRegistry(key))?;
            require(!s.is_empty() && seen.insert(s), "duplicate/empty axis")?;
            Ok(s)
        })
        .collect::<Result<Vec<_>>>()?;
    require(!values.is_empty(), "empty axis")?;
    Ok(values)
}
fn typed_axis<T: FromStr<Err = CaseError>>(v: &Value, key: &'static str) -> Result<Vec<T>> {
    axis(v, key)?.into_iter().map(str::parse).collect()
}
fn identifier(s: &str) -> Result<()> {
    require(!s.is_empty() && s.len() <= 128, "identifier bound")
}
fn packing(v: &Value) -> Result<Vec<CasePacking>> {
    match (v.get("packing"), v.get("packing_variants")) {
        (Some(_), None) => Ok(vec![text(v, "packing")?.parse()?]),
        (None, Some(_)) => typed_axis(v, "packing_variants"),
        _ => Err(CaseError::InvalidRegistry("ambiguous/missing packing")),
    }
}
fn known_family(id: &str) -> Result<(CaseRole, CaseMode)> {
    use CaseMode::{Interactive as I, Noninteractive as N};
    use CaseRole::{BaselineA, BaselineB, BaselineC, Main};
    match id {
        "main_i" | "main_i_mnist_b64" | "main_i_mnist_b128_candidate" => Ok((Main, I)),
        "main_ni" | "main_ni_mnist_b64" | "main_ni_mnist_b128_candidate" => Ok((Main, N)),
        "a_matched" | "a_matched_mnist_b64" => Ok((BaselineA, I)),
        "b_external" | "b_external_mnist_b64" | "b_external_mnist_b128_candidate" => {
            Ok((BaselineB, N))
        }
        "c_i" | "c_i_mnist_b64" | "c_i_mnist_b128_candidate" => Ok((BaselineC, I)),
        "c_ni" | "c_ni_mnist_b64" | "c_ni_mnist_b128_candidate" => Ok((BaselineC, N)),
        _ => Err(CaseError::InvalidRegistry("unknown family")),
    }
}
fn security(v: &Value, role: CaseRole, mode: CaseMode) -> Result<SecurityPolicy> {
    let (key, expected) = match (role, mode) {
        (CaseRole::Main, CaseMode::Interactive) => ("main_interactive", SecurityPolicy::N16Train),
        (CaseRole::Main, CaseMode::Noninteractive) => {
            ("main_noninteractive", SecurityPolicy::N16Boot)
        }
        (CaseRole::BaselineC, CaseMode::Interactive) => {
            ("packed_c_interactive", SecurityPolicy::N16Train)
        }
        (CaseRole::BaselineC, CaseMode::Noninteractive) => {
            ("packed_c_noninteractive", SecurityPolicy::N16Boot)
        }
        (CaseRole::BaselineA, CaseMode::Interactive) => {
            ("baseline_a_matched", SecurityPolicy::N16Train)
        }
        (CaseRole::BaselineB, CaseMode::Noninteractive) => {
            require(
                text(v, "official_b")? == "captured upstream context; no claim of equal security",
                "external crypto",
            )?;
            return Ok(SecurityPolicy::UpstreamVerified);
        }
        _ => return Err(CaseError::InvalidRegistry("role/mode")),
    };
    let actual = text(v, key)?.parse()?;
    require(actual == expected, "crypto policy")?;
    Ok(actual)
}
fn checked_axes(role: CaseRole, s: &CaseSelector, model: &str) -> Result<()> {
    let role_axes = match role {
        CaseRole::Main => {
            matches!(s.backend, CaseBackend::Cpu | CaseBackend::Cuda)
                && s.packing == CasePacking::BatchedDiagonal
        }
        CaseRole::BaselineA => {
            s.backend == CaseBackend::Cpu
                && s.mode == CaseMode::Interactive
                && model == "mlp2020"
                && matches!(s.packing, CasePacking::Row2020 | CasePacking::Diagonal2020)
        }
        CaseRole::BaselineB => {
            s.backend == CaseBackend::ExternalCpu
                && s.mode == CaseMode::Noninteractive
                && model == "emlp1"
                && s.packing == CasePacking::OfficialUpstream
        }
        CaseRole::BaselineC => {
            s.backend == CaseBackend::Cuda && s.packing == CasePacking::AlternatingRebootRowColumn
        }
        CaseRole::MainAblation => false,
    };
    require(role_axes, "role/backend/model/packing")?;
    require(
        match (role, s.mode) {
            (CaseRole::BaselineB, _) => s.placement == CasePlacement::UpstreamVerified,
            (_, CaseMode::Interactive) => s.placement == CasePlacement::ClientRefresh,
            (_, CaseMode::Noninteractive) => matches!(
                s.placement,
                CasePlacement::WhenNeeded
                    | CasePlacement::AfterUpdate
                    | CasePlacement::AfterActivation
            ),
        },
        "mode/placement",
    )
}
fn make_case(
    selector: CaseSelector,
    role: CaseRole,
    security: SecurityPolicy,
    batch: u64,
) -> Result<RegisteredCase> {
    let mut case = RegisteredCase {
        selector,
        role,
        security,
        batch,
        flags: None,
        bytes: Vec::new(),
        digest: String::new(),
    };
    case.bytes =
        crate::study_reference::registered_case_canonical(&case).map_err(CaseError::Canonical)?;
    case.digest = sha256_hex(&case.bytes);
    Ok(case)
}

fn project(v: &Value) -> Result<RegisteredCases> {
    require(
        v["schema_version"].as_u64() == Some(1)
            && v["artifact_type"].as_str() == Some("design_registration"),
        "schema/artifact",
    )?;
    require(
        v["scope"]["full_seeds"] == serde_json::json!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            && v["scope"]["quick_seeds"] == serde_json::json!([0]),
        "seed axes",
    )?;
    let mut profiles = BTreeMap::new();
    for p in items(v, "profiles")? {
        let id = text(p, "id")?;
        identifier(id)?;
        let batch = p["batch"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or(CaseError::InvalidRegistry("batch"))?;
        let model = text(p, "model")?;
        require(
            matches!(model, "mlp2020" | "emlp1" | "emlp2" | "emlp3"),
            "profile model",
        )?;
        require(
            profiles.insert(id, (batch, model)).is_none(),
            "duplicate profile",
        )?;
    }
    require(profiles.len() == 21, "profile count")?;
    let seeds = items(&v["scope"], "full_seeds")?;
    let mut families = Vec::new();
    let mut seen = BTreeSet::new();
    for family in items(v, "families")? {
        let allowed = [
            "id",
            "role",
            "profile_ids",
            "backends",
            "modes",
            "placements",
            "seeds",
            "packing",
            "packing_variants",
            "crypto_override",
            "claim",
            "requested_configuration",
            "verified_upstream_profile_by_dataset",
            "upstream_dataset_key_by_dataset",
            "unresolved_values_status",
            "study",
            "override_readiness",
        ];
        require(
            family
                .as_object()
                .is_some_and(|o| o.keys().all(|k| allowed.contains(&k.as_str()))),
            "unknown family key",
        )?;
        let id = text(family, "id")?;
        identifier(id)?;
        require(seen.insert(id), "duplicate family")?;
        let (expected_role, expected_mode) = known_family(id)?;
        let role: CaseRole = text(family, "role")?.parse()?;
        require(role == expected_role, "family role")?;
        require(
            text(family, "seeds")? == "full_seeds or quick_seeds",
            "family seeds",
        )?;
        let profiles_axis = axis(family, "profile_ids")?;
        let backends = typed_axis::<CaseBackend>(family, "backends")?;
        let modes = typed_axis::<CaseMode>(family, "modes")?;
        let placements = typed_axis::<CasePlacement>(family, "placements")?;
        let packings = packing(family)?;
        require(modes == [expected_mode], "family mode")?;
        let security = security(&v["crypto"], role, expected_mode)?;
        if role == CaseRole::BaselineA {
            require(
                text(family, "crypto_override")? == security.as_str(),
                "matched crypto override",
            )?;
        } else {
            require(
                family.get("crypto_override").is_none(),
                "unexpected crypto override",
            )?;
        }
        for profile in profiles_axis {
            let &(batch, model) = profiles
                .get(profile)
                .ok_or(CaseError::InvalidRegistry("unknown profile"))?;
            for &backend in &backends {
                for &mode in &modes {
                    for &placement in &placements {
                        for &packing in &packings {
                            for seed in seeds {
                                let selector = CaseSelector {
                                    origin: CaseOrigin::Family,
                                    registration_id: id.into(),
                                    profile_id: profile.into(),
                                    seed: seed
                                        .as_u64()
                                        .ok_or(CaseError::InvalidRegistry("seed"))?,
                                    scope: CaseScope::Full,
                                    backend,
                                    mode,
                                    placement,
                                    packing,
                                };
                                checked_axes(role, &selector, model)?;
                                families.push(make_case(selector, role, security, batch)?);
                            }
                        }
                    }
                }
            }
        }
    }
    require(seen.len() == 17, "family count")?;
    let mut supplements = Vec::new();
    let mut iris = Vec::new();
    let mut seen = BTreeSet::new();
    for raw in items(v, "supplements")? {
        let id = text(raw, "id")?;
        require(seen.insert(id), "duplicate supplement")?;
        let kind = match id {
            "iris_2020_exact_crypto" => SupplementKind::ExactIris,
            "mnist_b64_registered" => SupplementKind::ArchivalFull,
            "mnist_784_emlp1_b48_weighted_partial" => SupplementKind::WeightedPartial,
            "reduced_level_matched_training" => SupplementKind::ReducedLevel,
            "rq3_all8" => SupplementKind::Rq3AllEight,
            "task51_plain_references" => SupplementKind::PlainReferences,
            "task53_degree4_emlp1" => SupplementKind::DeferredActivation,
            _ => return Err(CaseError::InvalidRegistry("unknown supplement")),
        };
        if kind == SupplementKind::ExactIris {
            require(
                raw.as_object().is_some_and(|o| {
                    o.keys().all(|k| {
                        [
                            "id",
                            "profile",
                            "role",
                            "backends",
                            "modes",
                            "packing_variants",
                            "preset",
                            "seeds",
                            "historical_seed2020_anchor",
                            "comparison",
                        ]
                        .contains(&k.as_str())
                    })
                }),
                "unknown exact-Iris key",
            )?;
            require(
                text(raw, "profile")? == "iris__mlp2020"
                    && text(raw, "role")? == "baseline_a"
                    && text(raw, "preset")? == "seal_2020"
                    && text(&v["crypto"], "baseline_a_exact_iris")? == "seal_2020"
                    && text(raw, "seeds")? == "full_seeds or quick_seeds",
                "exact Iris policy",
            )?;
            require(
                axis(raw, "backends")? == ["cpu"] && axis(raw, "modes")? == ["interactive"],
                "exact Iris axes",
            )?;
            let packings = packing(raw)?;
            require(
                packings == [CasePacking::Row2020, CasePacking::Diagonal2020],
                "exact Iris packing",
            )?;
            let &(batch, _) = profiles
                .get("iris__mlp2020")
                .ok_or(CaseError::InvalidRegistry("Iris profile"))?;
            for packing in packings {
                for seed in seeds {
                    iris.push(make_case(
                        CaseSelector {
                            origin: CaseOrigin::Supplement,
                            registration_id: id.into(),
                            profile_id: "iris__mlp2020".into(),
                            seed: seed
                                .as_u64()
                                .ok_or(CaseError::InvalidRegistry("Iris seed"))?,
                            scope: CaseScope::Full,
                            backend: CaseBackend::Cpu,
                            mode: CaseMode::Interactive,
                            placement: CasePlacement::ClientRefresh,
                            packing,
                        },
                        CaseRole::BaselineA,
                        SecurityPolicy::Seal2020,
                        batch,
                    )?);
                }
            }
        }
        let mut flags = Vec::new();
        if kind == SupplementKind::Rq3AllEight {
            require(
                text(raw, "configs")?
                    == "all eight hoist/bsgs/lazy_relin tuples; four caveats unchanged",
                "rq3 configs",
            )?;
            for hoist in [false, true] {
                for bsgs in [false, true] {
                    for lazy_relin in [false, true] {
                        flags.push(CaseEvalConfig {
                            hoist,
                            bsgs,
                            lazy_relin,
                        });
                    }
                }
            }
        }
        supplements.push(SupplementDescriptor {
            id: id.into(),
            kind,
            raw: raw.clone(),
            flags,
        });
    }
    require(seen.len() == 7, "supplement count")?;
    Ok(RegisteredCases {
        families,
        iris,
        supplements,
    })
}

#[cfg(test)]
#[path = "registered_case_tests.rs"]
mod tests;
