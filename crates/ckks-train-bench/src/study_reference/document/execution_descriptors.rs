//! Four private inner representations. Inspection bytes and source projections are
//! never a prepared route, admission, live gate, start token or scientific result.
#![allow(dead_code)] // Pure transport validators remain non-authorizing inspection helpers.
use super::*;
use crate::study_admission::{
    Consumption, GeometryBody, GeometryWriter, MicroPart, encode_geometry_body,
};
use crate::study_client::PreparationDescriptorFacts;
use ckks_train_ckks::{
    CkksParams, EvalConfig,
    bootstrap::metadata::{SecretDeclaration, UnvalidatedRequirement},
};
use ckks_train_nn::dense::TransposeKeys;
const PARAMETERS: &str = "hekit-study-interactive-parameters-v1";
const LAYOUT: &str = "hekit-study-diagonal-layout-v1";
const KEYS: &str = "hekit-study-cpu-diagonal-keys-v1";
const SECURITY: &str = "hekit-study-main-security-policy-v1";
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Parameters,
    Layout,
    Keys,
    Security,
}
impl Kind {
    fn domain(self) -> &'static str {
        match self {
            Self::Parameters => PARAMETERS,
            Self::Layout => LAYOUT,
            Self::Keys => KEYS,
            Self::Security => SECURITY,
        }
    }
}
fn valid(ok: bool, path: &str) -> Result<()> {
    ensure(ok, DocumentErrorCode::CrossLinkMismatch, path)
}
pub(super) fn checked_add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| err(DocumentErrorCode::TooLarge, "execution.add"))
}
pub(super) fn checked_mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| err(DocumentErrorCode::TooLarge, "execution.multiply"))
}
pub(super) fn checked_usize(n: u64, bits: u32) -> Result<usize> {
    ensure(
        bits > 0 && bits <= usize::BITS && (bits == 64 || n < (1u64 << bits)),
        DocumentErrorCode::TooLarge,
        "execution.conversion",
    )?;
    usize::try_from(n).map_err(|_| err(DocumentErrorCode::TooLarge, "execution.conversion"))
}
pub(super) fn checked_size(n: u64) -> Result<usize> {
    ensure(
        n <= A_MAX as u64,
        DocumentErrorCode::TooLarge,
        "execution.size",
    )?;
    checked_usize(n, usize::BITS)
}
fn weight(n: Option<u64>, degree: u64) -> Result<()> {
    valid(n.is_none_or(|n| n > 0 && n <= degree), "execution.weight")
}
fn hash_valid(s: &str) -> Result<()> {
    ensure(
        s.len() == 64 && lower_hex(s),
        DocumentErrorCode::BadHex,
        "execution.hash",
    )
}
// One borrowed size pass, one reserved output pass. No geometry/partition clone,
// allocation or digest occurs during sizing. Encoder supplies the existing LE,
// Text/Hash/Bool/Option primitives; every append remains independently bounded.
pub(super) struct Writer {
    length: u64,
    output: Option<Encoder>,
}
impl Writer {
    pub(super) fn u64(&mut self, n: u64) -> Result<()> {
        self.raw(&n.to_le_bytes())
    }
    pub(super) fn text(&mut self, s: &str) -> Result<()> {
        self.u64(s.len() as u64)?;
        self.raw(s.as_bytes())
    }
    pub(super) fn hash(&mut self, s: &str) -> Result<()> {
        hash_valid(s)?;
        self.text(s)
    }
    pub(super) fn optional_hash(&mut self, s: Option<&str>) -> Result<()> {
        self.tag(u8::from(s.is_some()))?;
        if let Some(s) = s {
            self.hash(s)?;
        }
        Ok(())
    }
    pub(super) fn boolean(&mut self, b: bool) -> Result<()> {
        self.tag(u8::from(b))
    }
    fn secret(&mut self, n: Option<u64>) -> Result<()> {
        self.tag(u8::from(n.is_some()))?;
        if let Some(n) = n {
            self.u64(n)?;
        }
        Ok(())
    }
}
impl GeometryWriter for Writer {
    type Error = DocumentError;
    fn raw(&mut self, b: &[u8]) -> Result<()> {
        self.length = checked_add(self.length, b.len() as u64)?;
        checked_size(self.length)?;
        if let Some(e) = &mut self.output {
            e.append(b, false)?;
        }
        Ok(())
    }
    fn number(&mut self, n: usize) -> Result<()> {
        self.u64(u64::try_from(n).map_err(|_| err(DocumentErrorCode::TooLarge, "execution.index"))?)
    }
}
fn traverse(
    kind: Kind,
    output: bool,
    body: impl Fn(&mut Writer) -> Result<()>,
) -> Result<(usize, Vec<u8>)> {
    traverse_domain(Some(kind.domain()), output, body)
}
// Shared two-pass primitives; None is the exact preexisting raw outer body.
pub(super) fn traverse_domain(
    domain: Option<&str>,
    output: bool,
    body: impl Fn(&mut Writer) -> Result<()>,
) -> Result<(usize, Vec<u8>)> {
    let header = |w: &mut Writer| -> Result<()> {
        if let Some(domain) = domain {
            w.text(ENCODING_VERSION)?;
            w.text(domain)?;
        }
        Ok(())
    };
    let mut size = Writer {
        length: 0,
        output: None,
    };
    header(&mut size)?;
    body(&mut size)?;
    let n = checked_size(size.length)?;
    if !output {
        return Ok((n, Vec::new()));
    }
    let mut output = Encoder::new(DocumentLayout::Reference);
    output
        .bytes
        .try_reserve_exact(n)
        .map_err(|_| err(DocumentErrorCode::TooLarge, "execution.reserve"))?;
    let mut writer = Writer {
        length: 0,
        output: Some(output),
    };
    header(&mut writer)?;
    body(&mut writer)?;
    valid(writer.length == size.length, "execution.size_changed")?;
    Ok((n, writer.output.expect("output pass").bytes))
}
#[derive(Clone, Copy)]
pub(super) enum BitSizes<'a> {
    U32(&'a [u32]),
    #[cfg(test)]
    U64(&'a [u64]),
}
impl BitSizes<'_> {
    fn len(self) -> usize {
        match self {
            Self::U32(v) => v.len(),
            #[cfg(test)]
            Self::U64(v) => v.len(),
        }
    }
    fn get(self, i: usize) -> u64 {
        match self {
            Self::U32(v) => v[i].into(),
            #[cfg(test)]
            Self::U64(v) => v[i],
        }
    }
}
pub(super) struct ParameterFields<'a> {
    pub(super) q: &'a [u64],
    pub(super) p: &'a [u64],
    pub(super) q_bits: BitSizes<'a>,
    pub(super) p_bits: BitSizes<'a>,
    pub(super) log_n: u64,
    pub(super) scale_bits: u64,
    pub(super) dnum: u64,
    pub(super) main_weight: Option<u64>,
}
impl<'a> ParameterFields<'a> {
    fn from_params(p: &'a CkksParams) -> Self {
        Self {
            q: p.q(),
            p: p.p(),
            q_bits: BitSizes::U32(p.q_bits()),
            p_bits: BitSizes::U32(p.p_bits()),
            log_n: p.log_n().into(),
            scale_bits: p.scale().to_bits(),
            dnum: p.dnum() as u64,
            main_weight: p.hamming_weight().map(|n| n as u64),
        }
    }
    fn validate(&self) -> Result<()> {
        use ckks_train_ckks::params::{MAX_LOG_N, MIN_LOG_N};
        valid(
            (u64::from(MIN_LOG_N)..=u64::from(MAX_LOG_N)).contains(&self.log_n),
            "parameters.log_n",
        )?;
        valid(
            !self.q.is_empty()
                && !self.p.is_empty()
                && self.q.len() == self.q_bits.len()
                && self.p.len() == self.p_bits.len(),
            "parameters.chains",
        )?;
        for bits in [self.q_bits, self.p_bits] {
            for i in 0..bits.len() {
                valid((2..=60).contains(&bits.get(i)), "parameters.bits")?;
            }
        }
        valid(
            self.dnum > 0 && (self.q.len() as u64) % self.dnum == 0,
            "parameters.dnum",
        )?;
        scale_valid(self.scale_bits)?;
        weight(self.main_weight, 1u64 << self.log_n)
    }
    pub(super) fn encode(&self) -> Result<Vec<u8>> {
        Ok(self.traverse(true)?.1)
    }
    fn traverse(&self, output: bool) -> Result<(usize, Vec<u8>)> {
        self.validate()?;
        traverse(Kind::Parameters, output, |w| {
            for v in [self.q, self.p] {
                w.number(v.len())?;
                for n in v {
                    w.u64(*n)?;
                }
            }
            for bits in [self.q_bits, self.p_bits] {
                w.number(bits.len())?;
                for i in 0..bits.len() {
                    w.u64(bits.get(i))?;
                }
            }
            for n in [self.log_n, self.scale_bits, self.dnum] {
                w.u64(n)?;
            }
            w.secret(self.main_weight)
        })
    }
}
fn scale_valid(bits: u64) -> Result<()> {
    // Constructor takes an integer exponent1..60; the wire stores VALUE bits.
    valid(
        (1..=60).any(|n| (2f64.powi(n)).to_bits() == bits),
        "parameters.scale_bits",
    )
}
pub(super) struct StageFields<'a> {
    pub(super) dims: &'a [usize],
    pub(super) geometry: GeometryBody<'a>,
}
pub(super) trait GeometryStages {
    fn len(&self) -> usize;
    fn stage(&self, index: usize) -> StageFields<'_>;
}
impl GeometryStages for PreparationDescriptorFacts<'_> {
    fn len(&self) -> usize {
        self.stage_count()
    }
    fn stage(&self, index: usize) -> StageFields<'_> {
        let g = self.geometry(index);
        StageFields {
            dims: g.dims(),
            geometry: GeometryBody::from(g),
        }
    }
}
pub(super) struct LayoutFields<'a, S: GeometryStages> {
    pub(super) parameter_identity: &'a str,
    pub(super) consumption: u8,
    pub(super) mode: u8,
    pub(super) config: EvalConfig,
    pub(super) stages: &'a S,
    pub(super) parts: &'a [Vec<MicroPart>],
}
impl<S: GeometryStages> LayoutFields<'_, S> {
    pub(super) fn encode(&self) -> Result<Vec<u8>> {
        Ok(self.traverse(true)?.1)
    }
    fn traverse(&self, output: bool) -> Result<(usize, Vec<u8>)> {
        valid(
            self.consumption <= 1
                && self.mode <= 1
                && (1..=3).contains(&self.stages.len())
                && !self.parts.is_empty(),
            "layout.tags_shape",
        )?;
        let first = self.stages.stage(0).geometry.layout;
        for i in 0..self.stages.len() {
            let stage = self.stages.stage(i);
            let g = &stage.geometry;
            let l = g.layout;
            valid(
                l.slots == first.slots
                    && l.d == first.d
                    && l.batch == first.batch
                    && l.slots.is_power_of_two()
                    && l.d.is_power_of_two()
                    && l.batch.is_power_of_two(),
                "layout.common",
            )?;
            let occupied = checked_mul(l.d as u64, l.batch as u64)?;
            valid(
                occupied > 0 && (l.slots as u64) % occupied == 0,
                "layout.occupancy",
            )?;
            valid(
                stage.dims.len() >= 2
                    && g.shapes.len() == stage.dims.len() - 1
                    && g.masks.len() == g.shapes.len()
                    && g.transpose.len() == g.shapes.len(),
                "layout.cardinality",
            )?;
            for (shape, dims) in g.shapes.iter().zip(stage.dims.windows(2)) {
                valid(
                    dims[0] > 0 && dims[1] > 0 && shape.d_in == dims[0] && shape.d_out == dims[1],
                    "layout.dims",
                )?;
            }
        }
        for parts in self.parts {
            valid(!parts.is_empty(), "layout.parts")?;
            let mut end = 0u64;
            for p in parts {
                valid(
                    p.start as u64 == end && p.valid > 0 && p.valid <= first.batch,
                    "layout.part",
                )?;
                end = checked_add(end, p.valid as u64)?;
            }
        }
        traverse(Kind::Layout, output, |w| {
            w.hash(self.parameter_identity)?;
            w.tag(self.consumption)?;
            w.tag(self.mode)?;
            for b in [self.config.hoist, self.config.bsgs, self.config.lazy_relin] {
                w.boolean(b)?;
            }
            w.number(self.stages.len())?;
            for i in 0..self.stages.len() {
                let s = self.stages.stage(i);
                w.number(s.dims.len())?;
                for n in s.dims {
                    w.number(*n)?;
                }
                encode_geometry_body(w, s.geometry)?;
            }
            w.number(self.parts.len())?;
            for parts in self.parts {
                w.number(parts.len())?;
                for p in parts {
                    w.number(p.start)?;
                    w.number(p.valid)?;
                }
            }
            Ok(())
        })
    }
}
pub(super) struct KeyFields<'a> {
    pub(super) parameter_identity: &'a str,
    pub(super) bootstrap_descriptor: Option<&'a str>,
    pub(super) transpose: Option<&'a [i32]>,
    pub(super) training_level: u64,
    pub(super) training_rotations: &'a [i32],
    pub(super) training_relin: bool,
    pub(super) training_conjugation: bool,
    pub(super) training_sse: bool,
    pub(super) bootstrap_generated: &'a [i32],
    pub(super) bootstrap_execution: &'a [i32],
    pub(super) bootstrap_conjugation: bool,
    pub(super) bootstrap_sse: bool,
    pub(super) separate_bootstrap_bundle: bool,
    pub(super) owner_policy: u8,
}
impl KeyFields<'_> {
    pub(super) fn encode(&self) -> Result<Vec<u8>> {
        Ok(self.traverse(true)?.1)
    }
    fn traverse(&self, output: bool) -> Result<(usize, Vec<u8>)> {
        valid(self.owner_policy == 0, "keys.owner")?;
        traverse(Kind::Keys, output, |w| {
            w.hash(self.parameter_identity)?;
            w.optional_hash(self.bootstrap_descriptor)?;
            w.tag(u8::from(self.transpose.is_some()))?;
            if let Some(pool) = self.transpose {
                w.steps(pool)?;
            }
            w.u64(self.training_level)?;
            w.steps(self.training_rotations)?;
            for b in [
                self.training_relin,
                self.training_conjugation,
                self.training_sse,
            ] {
                w.boolean(b)?;
            }
            w.steps(self.bootstrap_generated)?;
            w.steps(self.bootstrap_execution)?;
            for b in [
                self.bootstrap_conjugation,
                self.bootstrap_sse,
                self.separate_bootstrap_bundle,
            ] {
                w.boolean(b)?;
            }
            w.tag(self.owner_policy)
        })
    }
}
#[derive(Clone, Copy)]
pub(super) enum Requirements<'a> {
    P1(&'a [UnvalidatedRequirement]),
    #[cfg(test)]
    Tags(&'a [u8]),
}
fn requirement_tag(r: UnvalidatedRequirement) -> u8 {
    match r {
        UnvalidatedRequirement::EncodingQOverTwoFeasibility => 0,
        UnvalidatedRequirement::EvalModInternalScaleUnvalidated => 1,
        UnvalidatedRequirement::EncryptedInputRange => 2,
        UnvalidatedRequirement::ApproximationPrecision => 3,
        UnvalidatedRequirement::ActualSecretContract => 4,
        UnvalidatedRequirement::KeyBinding => 5,
        UnvalidatedRequirement::SecurityEstimate => 6,
        UnvalidatedRequirement::RuntimeMemoryFit => 7,
        UnvalidatedRequirement::BackendSupport => 8,
    }
}
impl Requirements<'_> {
    fn len(self) -> usize {
        match self {
            Self::P1(v) => v.len(),
            #[cfg(test)]
            Self::Tags(v) => v.len(),
        }
    }
    fn get(self, i: usize) -> u8 {
        match self {
            Self::P1(v) => requirement_tag(v[i]),
            #[cfg(test)]
            Self::Tags(v) => v[i],
        }
    }
}
pub(super) struct SecurityFields<'a> {
    pub(super) registered_policy: u8,
    pub(super) parameter_identity: &'a str,
    pub(super) bootstrap_descriptor: Option<&'a str>,
    pub(super) main: Option<u64>,
    pub(super) ephemeral: Option<Option<u64>>,
    pub(super) assurance: u8,
    pub(super) bootstrap_scope: Option<u8>,
    pub(super) bootstrap_unvalidated: Requirements<'a>,
}
impl SecurityFields<'_> {
    pub(super) fn encode(&self) -> Result<Vec<u8>> {
        Ok(self.traverse(true)?.1)
    }
    fn traverse(&self, output: bool) -> Result<(usize, Vec<u8>)> {
        valid(
            self.registered_policy <= 1
                && self.assurance == 0
                && self.bootstrap_scope.is_none_or(|n| n == 0),
            "security.tags",
        )?;
        traverse(Kind::Security, output, |w| {
            w.tag(self.registered_policy)?;
            w.hash(self.parameter_identity)?;
            w.optional_hash(self.bootstrap_descriptor)?;
            w.secret(self.main)?;
            w.tag(u8::from(self.ephemeral.is_some()))?;
            if let Some(s) = self.ephemeral {
                w.secret(s)?;
            }
            w.tag(self.assurance)?;
            w.tag(u8::from(self.bootstrap_scope.is_some()))?;
            if let Some(s) = self.bootstrap_scope {
                w.tag(s)?;
            }
            w.number(self.bootstrap_unvalidated.len())?;
            for i in 0..self.bootstrap_unvalidated.len() {
                let n = self.bootstrap_unvalidated.get(i);
                valid(n <= 8, "security.requirement_tag")?;
                w.tag(n)?;
            }
            Ok(())
        })
    }
}
fn secret_weight(s: SecretDeclaration) -> Option<u64> {
    match s {
        SecretDeclaration::DenseTernary => None,
        SecretDeclaration::FixedWeight(n) => Some(n as u64),
    }
}
// These methods borrow genuine canonical checked preparations. No parser or field
// DTO can construct PreparationDescriptorFacts, a route, or any live capability.
impl PreparationDescriptorFacts<'_> {
    pub(crate) fn interactive_parameter_bytes(&self) -> Result<Vec<u8>> {
        valid(self.bootstrap().is_none(), "source.interactive_only")?;
        ParameterFields::from_params(self.parameters()).encode()
    }
    fn parameter_hash(&self) -> Result<String> {
        match self.bootstrap() {
            Some(b) => Ok(sha256_hex(b.parameter_identity_bytes())),
            None => Ok(sha256_hex(&self.interactive_parameter_bytes()?)),
        }
    }
    pub(crate) fn physical_layout_bytes(&self) -> Result<Vec<u8>> {
        let parameter = self.parameter_hash()?;
        Ok(self
            .layout_with(
                DescriptorDigests {
                    parameter: &parameter,
                    bootstrap: None,
                },
                true,
            )?
            .1)
    }
    fn layout_with(&self, hashes: DescriptorDigests<'_>, output: bool) -> Result<(usize, Vec<u8>)> {
        let p = self.parameters();
        valid(self.stage_count() > 0, "source.stages")?;
        for i in 0..self.stage_count() {
            valid(
                self.geometry(i).layout().slots == p.slots()
                    && self.geometry(i).eval_config() == self.geometry(0).eval_config(),
                "source.geometry",
            )?;
        }
        LayoutFields {
            parameter_identity: hashes.parameter,
            consumption: match self.plan().consumption() {
                Consumption::Full => 0,
                Consumption::Quick => 1,
            },
            mode: u8::from(self.bootstrap().is_some()),
            config: self.geometry(0).eval_config(),
            stages: self,
            parts: self.plan().partitions(),
        }
        .traverse(output)
    }
    pub(crate) fn cpu_key_policy_bytes(&self) -> Result<Vec<u8>> {
        let parameter = self.parameter_hash()?;
        let bootstrap = self
            .bootstrap()
            .map(|b| sha256_hex(b.descriptor_identity_bytes()));
        Ok(self
            .keys_with(
                DescriptorDigests {
                    parameter: &parameter,
                    bootstrap: bootstrap.as_deref(),
                },
                true,
            )?
            .1)
    }
    fn keys_with(&self, hashes: DescriptorDigests<'_>, output: bool) -> Result<(usize, Vec<u8>)> {
        let keys = self.plan().keys();
        let rotations = keys.training_rotations();
        let slots = self.parameters().slots();
        valid(
            rotations.iter().all(|n| *n > 0 && (*n as usize) < slots)
                && rotations.windows(2).all(|p| p[0] < p[1]),
            "source.training_union",
        )?;
        // Both directions: no extra key may be smuggled in under a subset check.
        for i in 0..self.stage_count() {
            for r in self.geometry(i).rotation_steps() {
                valid(rotations.binary_search(r).is_ok(), "source.geometry_key")?;
            }
        }
        for r in rotations {
            valid(
                (0..self.stage_count()).any(|i| self.geometry(i).rotation_steps().contains(r)),
                "source.extra_key",
            )?;
        }
        if let Some(b) = self.bootstrap() {
            valid(
                keys.bootstrap_generated_rotations() == b.generated_keys().rotation_steps()
                    && keys.bootstrap_execution_rotations() == b.execution_rotation_steps()
                    && keys.bootstrap_conjugation() == b.generated_keys().conjugation().is_some()
                    && keys.bootstrap_sse()
                        == (b.generated_keys().sse_to_sparse().is_some()
                            && b.generated_keys().sse_to_main().is_some())
                    && keys.separate_bootstrap_bundle(),
                "source.bootstrap_keys",
            )?;
        } else {
            valid(
                keys.bootstrap_generated_rotations().is_empty()
                    && keys.bootstrap_execution_rotations().is_empty()
                    && !keys.bootstrap_conjugation()
                    && !keys.bootstrap_sse()
                    && !keys.separate_bootstrap_bundle(),
                "source.interactive_keys",
            )?;
        }
        let pool = match self.transpose() {
            TransposeKeys::Dedicated => None,
            TransposeKeys::Reuse(v) => Some(v.as_slice()),
        };
        KeyFields {
            parameter_identity: hashes.parameter,
            bootstrap_descriptor: hashes.bootstrap,
            transpose: pool,
            training_level: self.parameters().max_level() as u64,
            training_rotations: rotations,
            training_relin: true,
            training_conjugation: false,
            training_sse: false,
            bootstrap_generated: keys.bootstrap_generated_rotations(),
            bootstrap_execution: keys.bootstrap_execution_rotations(),
            bootstrap_conjugation: keys.bootstrap_conjugation(),
            bootstrap_sse: keys.bootstrap_sse(),
            separate_bootstrap_bundle: keys.separate_bootstrap_bundle(),
            owner_policy: 0,
        }
        .traverse(output)
    }
    pub(crate) fn security_declaration_bytes(&self) -> Result<Vec<u8>> {
        let parameter = self.parameter_hash()?;
        let bootstrap = self
            .bootstrap()
            .map(|b| sha256_hex(b.descriptor_identity_bytes()));
        Ok(self
            .security_with(
                DescriptorDigests {
                    parameter: &parameter,
                    bootstrap: bootstrap.as_deref(),
                },
                true,
            )?
            .1)
    }
    fn security_with(
        &self,
        hashes: DescriptorDigests<'_>,
        output: bool,
    ) -> Result<(usize, Vec<u8>)> {
        let main = self.parameters().hamming_weight().map(|n| n as u64);
        if let Some(b) = self.bootstrap() {
            valid(
                secret_weight(b.declared_main_secret()) == main,
                "source.main_secret",
            )?;
        }
        let bytes=SecurityFields{registered_policy:u8::from(self.bootstrap().is_some()),parameter_identity:hashes.parameter,bootstrap_descriptor:hashes.bootstrap,main,ephemeral:self.bootstrap().and_then(|b|b.declared_ephemeral_secret()).map(secret_weight),assurance:0,bootstrap_scope:self.bootstrap().map(|b|match b.proof_scope(){ckks_train_ckks::bootstrap::metadata::ProofScope::StructuralCostAndBoundaryScales=>0}),bootstrap_unvalidated:Requirements::P1(self.bootstrap().map_or(&[],|b|b.unvalidated_requirements()))}.traverse(output)?;
        if output {
            WireDescriptor::parse(Kind::Security, &bytes.1)?
                .validate_registered_security_declarations(self.parameters().n() as u64)?;
        }
        Ok(bytes)
    }
}
// Hash cache is module-private and only derived in the fixed same-owner collector.
struct DescriptorDigests<'a> {
    parameter: &'a str,
    bootstrap: Option<&'a str>,
}
pub(crate) struct DescriptorSizes {
    pub(crate) interactive: Option<usize>,
    pub(crate) layout: usize,
    pub(crate) keys: usize,
    pub(crate) security: usize,
}
impl PreparationDescriptorFacts<'_> {
    pub(crate) fn descriptor_sizes(&self) -> Result<DescriptorSizes> {
        const HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
        let hashes = || DescriptorDigests {
            parameter: HASH,
            bootstrap: self.bootstrap().map(|_| HASH),
        };
        Ok(DescriptorSizes {
            interactive: if self.bootstrap().is_none() {
                Some(
                    ParameterFields::from_params(self.parameters())
                        .traverse(false)?
                        .0,
                )
            } else {
                None
            },
            layout: self.layout_with(hashes(), false)?.0,
            keys: self.keys_with(hashes(), false)?.0,
            security: self.security_with(hashes(), false)?.0,
        })
    }
}
fn collect_descriptors(
    facts: PreparationDescriptorFacts<'_>,
    accounting: &mut crate::study_execution::PreparationAccounting,
) -> crate::study_client::CpuPreparationResult<crate::study_execution::CanonicalDescriptors> {
    use crate::study_client::CpuPreparationError;
    use crate::study_execution::CanonicalDescriptors;
    let document = |e| CpuPreparationError::Descriptor(e);
    let sizes = facts.descriptor_sizes().map_err(document)?;
    // Every output and digest reservation precedes the corresponding allocation.
    let mut bytes = accounting.reserve(
        sizes
            .interactive
            .unwrap_or(0)
            .checked_add(sizes.layout)
            .and_then(|n| n.checked_add(sizes.keys))
            .and_then(|n| n.checked_add(sizes.security))
            .and_then(|n| {
                n.checked_add(if facts.bootstrap().is_some() {
                    320
                } else {
                    256
                })
            })
            .ok_or(crate::study_execution::AccountingError::Overflow)?,
    )?;
    let interactive_parameters = if sizes.interactive.is_some() {
        Some(facts.interactive_parameter_bytes().map_err(document)?)
    } else {
        None
    };
    let digest = |bytes: &[u8]| {
        crate::data::try_sha256_hex(bytes)
            .map_err(|_| crate::study_execution::AccountingError::Allocation)
    };
    let parameter_digest = digest(match &interactive_parameters {
        Some(p) => p,
        None => facts
            .bootstrap()
            .expect("NI metadata")
            .parameter_identity_bytes(),
    })?;
    let bootstrap_digest = facts
        .bootstrap()
        .map(|b| digest(b.descriptor_identity_bytes()))
        .transpose()?;
    let hashes = || DescriptorDigests {
        parameter: &parameter_digest,
        bootstrap: bootstrap_digest.as_deref(),
    };
    let layout = facts.layout_with(hashes(), true).map_err(document)?.1;
    let keys = facts.keys_with(hashes(), true).map_err(document)?.1;
    let security = facts.security_with(hashes(), true).map_err(document)?.1;
    let layout_digest = digest(&layout)?;
    let keys_digest = digest(&keys)?;
    let security_digest = digest(&security)?;
    bytes.retain_all()?;
    Ok(CanonicalDescriptors {
        interactive_parameters,
        layout,
        keys,
        security,
        parameter_digest,
        bootstrap_digest,
        layout_digest,
        keys_digest,
        security_digest,
        actual_binding: None,
    })
}
// Only these fixed collectors may use the private hash cache. The complete owner
// enters and leaves intact; no checked-by-value or arbitrary digest API is exposed.
pub(crate) fn collect_single<'i, 'p>(
    mut preparation: crate::study_execution::RetainedPreparation<
        'i,
        crate::study_client::CheckedPreparation<'p>,
    >,
) -> crate::study_client::CpuPreparationResult<
    crate::study_execution::PreparedCanonicalBundle<
        'i,
        crate::study_client::CheckedPreparation<'p>,
    >,
> {
    let (facts, accounting) = preparation.descriptor_parts();
    let descriptors = collect_descriptors(facts, accounting)?;
    Ok(crate::study_execution::PreparedCanonicalBundle::new(
        descriptors,
        preparation,
    ))
}
pub(crate) fn collect_blocks<'i, 'p>(
    mut preparation: crate::study_execution::RetainedPreparation<
        'i,
        crate::study_client::blocks::CheckedBlockPreparation<'p>,
    >,
) -> crate::study_client::CpuPreparationResult<
    crate::study_execution::PreparedCanonicalBundle<
        'i,
        crate::study_client::blocks::CheckedBlockPreparation<'p>,
    >,
> {
    let (facts, accounting) = preparation.descriptor_parts();
    let descriptors = collect_descriptors(facts, accounting)?;
    Ok(crate::study_execution::PreparedCanonicalBundle::new(
        descriptors,
        preparation,
    ))
}
// Allocation-free transport preflight. Lists borrow their encoded ranges; every
// count is checked against the remaining complete image before entering a loop.
// It intentionally does NOT grant source/production validity to wire fixtures.
#[derive(Clone, Copy)]
pub(super) struct WireDescriptor<'a> {
    kind: Kind,
    bytes: &'a [u8],
}
impl<'a> WireDescriptor<'a> {
    pub(super) fn parse(kind: Kind, bytes: &'a [u8]) -> Result<Self> {
        checked_size(bytes.len() as u64)?;
        let mut c = Cursor { bytes, offset: 0 };
        valid(
            c.text()? == ENCODING_VERSION && c.text()? == kind.domain(),
            "execution.domain",
        )?;
        match kind {
            Kind::Parameters => c.parameters()?,
            Kind::Layout => c.layout()?,
            Kind::Keys => c.keys()?,
            Kind::Security => {
                c.security()?;
            }
        }
        valid(c.offset == bytes.len(), "execution.trailing")?;
        Ok(Self { kind, bytes })
    }
    pub(super) fn canonical_bytes(&self) -> &'a [u8] {
        self.bytes
    }
    pub(super) fn matches_bytes(&self, expected: &[u8]) -> Result<()> {
        WireDescriptor::parse(self.kind, expected)?;
        valid(self.bytes == expected, "execution.source_mismatch")
    }
    pub(super) fn matches_source(&self, source: &PreparationDescriptorFacts<'_>) -> Result<()> {
        let expected = match self.kind {
            Kind::Parameters => source.interactive_parameter_bytes()?,
            Kind::Layout => source.physical_layout_bytes()?,
            Kind::Keys => source.cpu_key_policy_bytes()?,
            Kind::Security => source.security_declaration_bytes()?,
        };
        self.matches_bytes(&expected)
    }
    pub(super) fn validate_security_declarations(&self, n: u64) -> Result<()> {
        valid(self.kind == Kind::Security, "security.domain")?;
        let mut c = Cursor {
            bytes: self.bytes,
            offset: 0,
        };
        c.text()?;
        c.text()?;
        let s = c.security()?;
        weight(s.main, n)?;
        if let Some(e) = s.ephemeral {
            weight(e, n)?;
        }
        Ok(())
    }
    pub(super) fn validate_registered_security_declarations(&self, n: u64) -> Result<()> {
        self.validate_security_declarations(n)?;
        let mut c = Cursor {
            bytes: self.bytes,
            offset: 0,
        };
        c.text()?;
        c.text()?;
        let s = c.security()?;
        valid(n == 1 << 16 && s.main.is_none(), "security.registered_main")?;
        if s.policy == 0 {
            valid(
                !s.bootstrap
                    && s.ephemeral.is_none()
                    && s.scope.is_none()
                    && s.requirements.is_empty(),
                "security.registered_i",
            )
        } else {
            valid(
                s.bootstrap
                    && s.ephemeral
                        == Some(Some(
                            ckks_train_ckks::bootstrap::params::N16_EPHEMERAL_HW as u64,
                        ))
                    && s.scope == Some(0)
                    && s.requirements == [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "security.registered_ni",
            )
        }
    }
}
struct SecurityWire<'a> {
    policy: u8,
    bootstrap: bool,
    main: Option<u64>,
    ephemeral: Option<Option<u64>>,
    scope: Option<u8>,
    requirements: &'a [u8],
}
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(n)
            .filter(|n| *n <= self.bytes.len())
            .ok_or_else(|| err(DocumentErrorCode::OutOfRange, "execution.truncated"))?;
        let b = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(b)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn tag(&mut self, max: u8) -> Result<u8> {
        let b = self.byte()?;
        ensure(b <= max, DocumentErrorCode::UnsupportedTag, "execution.tag")?;
        Ok(b)
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        b.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(b))
    }
    fn count(&mut self, min: u64) -> Result<usize> {
        let n = self.u64()?;
        let need = checked_mul(n, min)?;
        valid(
            need <= self.bytes.len().saturating_sub(self.offset) as u64,
            "execution.count",
        )?;
        checked_usize(n, usize::BITS)
    }
    fn text(&mut self) -> Result<&'a str> {
        let n = self.count(1)?;
        std::str::from_utf8(self.take(n)?)
            .map_err(|_| err(DocumentErrorCode::OutOfRange, "execution.utf8"))
    }
    fn hash(&mut self) -> Result<()> {
        hash_valid(self.text()?)
    }
    fn option<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<Option<T>> {
        match self.tag(1)? {
            0 => Ok(None),
            _ => f(self).map(Some),
        }
    }
    fn secret(&mut self) -> Result<Option<u64>> {
        self.option(Self::u64)
    }
    fn hash_option(&mut self) -> Result<bool> {
        Ok(self.option(Self::hash)?.is_some())
    }
    fn fixed(&mut self, count: usize, width: usize) -> Result<&'a [u8]> {
        let n = count
            .checked_mul(width)
            .ok_or_else(|| err(DocumentErrorCode::TooLarge, "execution.multiply"))?;
        self.take(n)
    }
    fn list(&mut self, width: usize) -> Result<&'a [u8]> {
        let n = self.count(width as u64)?;
        self.fixed(n, width)
    }
    fn parameters(&mut self) -> Result<()> {
        let q = self.list(8)?;
        let p = self.list(8)?;
        let qb = self.list(8)?;
        let pb = self.list(8)?;
        let log_n = self.u64()?;
        let scale = self.u64()?;
        let dnum = self.u64()?;
        let main = self.secret()?;
        valid(
            !q.is_empty() && !p.is_empty() && q.len() == qb.len() && p.len() == pb.len(),
            "parameters.chains",
        )?;
        valid(
            (1..=17).contains(&log_n) && dnum > 0 && (q.len() as u64 / 8) % dnum == 0,
            "parameters.invariants",
        )?;
        for chunk in qb.chunks_exact(8).chain(pb.chunks_exact(8)) {
            let mut b = [0; 8];
            b.copy_from_slice(chunk);
            valid((2..=60).contains(&u64::from_le_bytes(b)), "parameters.bits")?;
        }
        scale_valid(scale)?;
        weight(main, 1 << log_n)
    }
    fn compact(&mut self) -> Result<()> {
        self.fixed(5, 8)?;
        Ok(())
    }
    fn geometry(&mut self) -> Result<()> {
        self.fixed(3, 8)?;
        self.list(64)?;
        let masks = self.count(49)?;
        for _ in 0..masks {
            self.fixed(4, 8)?;
            self.list(4)?;
            self.list(4)?;
            self.tag(1)?;
        }
        let recipes = self.count(8)?;
        for _ in 0..recipes {
            let steps = self.count(16)?;
            for _ in 0..steps {
                self.u64()?;
                self.list(4)?;
            }
        }
        self.list(4)?;
        self.option(|c| {
            let plans = c.count(40)?;
            for _ in 0..plans {
                c.compact()?;
            }
            let containers = c.count(49)?;
            for _ in 0..containers {
                c.option(Self::u64)?;
                c.u64()?;
                c.compact()?;
            }
            c.list(16)?;
            Ok(())
        })?;
        Ok(())
    }
    fn layout(&mut self) -> Result<()> {
        self.hash()?;
        self.tag(1)?;
        self.tag(1)?;
        for _ in 0..3 {
            self.tag(1)?;
        }
        let stages = self.count(8)?;
        valid((1..=3).contains(&stages), "layout.stages")?;
        for _ in 0..stages {
            self.list(8)?;
            self.geometry()?;
        }
        let updates = self.count(8)?;
        valid(updates > 0, "layout.updates")?;
        for _ in 0..updates {
            let parts = self.count(16)?;
            valid(parts > 0, "layout.parts")?;
            self.fixed(parts, 16)?;
        }
        Ok(())
    }
    fn keys(&mut self) -> Result<()> {
        self.hash()?;
        self.hash_option()?;
        if self.tag(1)? == 1 {
            self.list(4)?;
        }
        self.u64()?;
        self.list(4)?;
        for _ in 0..3 {
            self.tag(1)?;
        }
        self.list(4)?;
        self.list(4)?;
        for _ in 0..3 {
            self.tag(1)?;
        }
        self.tag(0)?;
        Ok(())
    }
    fn security(&mut self) -> Result<SecurityWire<'a>> {
        let policy = self.tag(1)?;
        self.hash()?;
        let bootstrap = self.hash_option()?;
        let main = self.secret()?;
        let ephemeral = self.option(Self::secret)?;
        self.tag(0)?;
        let scope = self.option(|c| c.tag(0))?;
        let requirements = self.list(1)?;
        valid(requirements.iter().all(|n| *n <= 8), "security.requirement")?;
        Ok(SecurityWire {
            policy,
            bootstrap,
            main,
            ephemeral,
            scope,
            requirements,
        })
    }
}
