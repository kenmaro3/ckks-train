//! Bounded untrusted document transport. No imported identity is a trusted identity.
//!
//! Before any VerifiedReference/target, a separately approved reader MUST materialize
//! this exact canonical profile/seed, call derive_identities, compare BOTH COMPLETE
//! byte arrays exactly (not only hashes), and verify every completion/checkpoint,
//! denominator, arithmetic and digest claim against the trusted full plan/data.
//! Source/build/session/prefix/state/publication policies remain separate gates.
mod producer;
mod reader;
use super::{CONTRACT_VERSION, ENCODING_VERSION, RECEIPT_VERSION};
use crate::data::sha256_hex;
pub(in crate::study_reference) use producer::{
    serialize_diagnostic, serialize_failure, serialize_local_quick, serialize_local_quick_f64,
    serialize_reference,
};
pub(in crate::study_reference) use reader::verify_reference;
pub use reader::{
    ReferenceCompletionView, ReferenceEndpointView, ReferenceTimeView, VerifiedReference,
};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;
use std::{fmt, marker::PhantomData};
const C_MAX: usize = 2_613_526;
const R_MAX: usize = 31_584_142;
const A_MAX: usize = 2_097_152;
const J_MAX: usize = 70_492_488;
const ENVELOPE_MAX: usize = 36_294_820;
const REFERENCE: &str = "hekit-study-reference-v1";
const DIAGNOSTIC: &str = "hekit-study-diagnostic-v1";
const FAILURE: &str = "hekit-study-reference-failure-v1";
/// Bounded parser diagnostics, separate from producer failure records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentErrorCode {
    /// TooLarge rejection under the fixed document schema.
    TooLarge,
    /// BadJson rejection under the fixed document schema.
    BadJson,
    /// DepthLimit rejection under the fixed document schema.
    DepthLimit,
    /// TokenLimit rejection under the fixed document schema.
    TokenLimit,
    /// DuplicateField rejection under the fixed document schema.
    DuplicateField,
    /// UnknownField rejection under the fixed document schema.
    UnknownField,
    /// MissingField rejection under the fixed document schema.
    MissingField,
    /// UnsupportedSchema rejection under the fixed document schema.
    UnsupportedSchema,
    /// UnsupportedTag rejection under the fixed document schema.
    UnsupportedTag,
    /// OutOfRange rejection under the fixed document schema.
    OutOfRange,
    /// NonFinite rejection under the fixed document schema.
    NonFinite,
    /// BadHex rejection under the fixed document schema.
    BadHex,
    /// DigestMismatch rejection under the fixed document schema.
    DigestMismatch,
    /// CrossLinkMismatch rejection under the fixed document schema.
    CrossLinkMismatch,
    /// OuterProgressMismatch rejection under the fixed document schema.
    OuterProgressMismatch,
    /// ObservationMismatch rejection under the fixed document schema.
    ObservationMismatch,
    /// TimingMismatch rejection under the fixed document schema.
    TimingMismatch,
}
/// Does not retain or echo untrusted values. Path length is bounded to 256 bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentError {
    code: DocumentErrorCode,
    path: String,
    offset: Option<usize>,
}
impl DocumentError {
    /// Stable diagnostic category.
    pub fn code(&self) -> DocumentErrorCode {
        self.code
    }
    /// Bounded logical field path, without supplied field values.
    pub fn path(&self) -> &str {
        &self.path
    }
    /// Byte offset when the lexical preflight identified one.
    pub fn offset(&self) -> Option<usize> {
        self.offset
    }
}
impl fmt::Display for DocumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} at {}", self.code, self.path)
    }
}
impl std::error::Error for DocumentError {}
type Result<T> = std::result::Result<T, DocumentError>;
fn err(code: DocumentErrorCode, path: &str) -> DocumentError {
    let mut path = path.to_owned();
    while path.len() > 256 {
        path.pop();
    }
    DocumentError {
        code,
        path,
        offset: None,
    }
}
fn ensure(ok: bool, code: DocumentErrorCode, path: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(err(code, path)) }
}
// Diagnostic display only: arbitrary keys can contain dots or stars. Never use
// this bounded, potentially ambiguous spelling to choose allocation authority.
fn path(parent: &str, key: &str) -> String {
    let mut display = if parent.is_empty() {
        key.to_owned()
    } else {
        format!("{parent}.{key}")
    };
    while display.len() > 256 {
        display.pop();
    }
    display
}
fn json_err(e: serde_json::Error, p: &str) -> DocumentError {
    let m = e.to_string();
    let code = if m.starts_with("DuplicateField") {
        DocumentErrorCode::DuplicateField
    } else if m.starts_with("UnknownField") {
        DocumentErrorCode::UnknownField
    } else if m.starts_with("OutOfRange") {
        DocumentErrorCode::OutOfRange
    } else {
        DocumentErrorCode::BadJson
    };
    err(code, p)
}
// The preflight caps scratch allocations BEFORE serde handles strings. It tracks
// exact decoded keys and array positions; only the two exact identity paths get
// giant unescaped hex tokens. serde remains the JSON/unicode syntax authority.
enum PathComponent {
    Key(String),
    ArrayElement,
}
#[derive(Clone, Copy)]
enum DocumentLayout {
    Reference,
    TargetLock,
}
impl DocumentLayout {
    fn file_cap(self) -> usize {
        match self {
            Self::Reference => J_MAX,
            Self::TargetLock => 7_324_204,
        }
    }
    fn canonical_cap(self) -> usize {
        match self {
            Self::Reference => ENVELOPE_MAX,
            Self::TargetLock => 4_710_678,
        }
    }
}
struct Scan<'a> {
    layout: DocumentLayout,
    s: &'a [u8],
    i: usize,
    excluded: usize,
    // At most the fixed depth limit, with each decoded key bounded to 64 bytes.
    // An object key "*" is distinct from an actual array element.
    structural_path: Vec<PathComponent>,
}
impl Scan<'_> {
    fn at_keys(&self, keys: &[&str]) -> bool {
        self.structural_path.len() == keys.len()
            && self.structural_path.iter().zip(keys).all(
                |(part, key)| matches!(part, PathComponent::Key(actual) if actual.as_str() == *key),
            )
    }
    fn at_argv_bytes(&self) -> bool {
        use PathComponent::{ArrayElement, Key};
        matches!(
            self.structural_path.as_slice(),
            [Key(provenance), Key(value), Key(invocation), Key(argv), ArrayElement, Key(bytes)]
                if provenance == "provenance" && value == "value"
                    && invocation == "invocation" && argv == "argv" && bytes == "bytes"
        )
    }
    fn ws(&mut self) {
        while self
            .s
            .get(self.i)
            .is_some_and(|c| matches!(c, b' ' | b'\r' | b'\n' | b'\t'))
        {
            self.i += 1;
        }
    }
    fn error(&self, c: DocumentErrorCode, p: &str) -> DocumentError {
        let mut e = err(c, p);
        e.offset = Some(self.i);
        e
    }
    fn string(&mut self, p: &str, key: bool) -> Result<Option<String>> {
        let start = self.i;
        self.i += 1;
        let mut escaped = false;
        let blob = if key {
            None
        } else if matches!(self.layout, DocumentLayout::Reference)
            && self.at_keys(&["identity", "contract", "canonical_bytes"])
        {
            Some(C_MAX)
        } else if matches!(self.layout, DocumentLayout::Reference)
            && self.at_keys(&["identity", "seed_receipt", "canonical_bytes"])
        {
            Some(R_MAX)
        } else if matches!(self.layout, DocumentLayout::TargetLock)
            && self.at_keys(&["contract", "canonical_bytes"])
        {
            Some(C_MAX)
        } else {
            None
        };
        let os = self.at_keys(&["provenance", "value", "source", "root", "bytes"])
            || self.at_keys(&["provenance", "value", "executable", "path", "bytes"])
            || self.at_keys(&["provenance", "value", "invocation", "cwd", "bytes"])
            || self.at_argv_bytes();
        let decoded = if key {
            64
        } else if self.at_keys(&["provenance", "value", "build", "compiler_vv"]) {
            16384
        } else if os {
            8192
        } else {
            4096
        };
        let cap = if let Some(n) = blob {
            n * 2 + 2
        } else {
            decoded * 6 + 2
        };
        loop {
            if self.i - start >= cap {
                return Err(self.error(DocumentErrorCode::TokenLimit, p));
            }
            let c = *self
                .s
                .get(self.i)
                .ok_or_else(|| self.error(DocumentErrorCode::BadJson, p))?;
            self.i += 1;
            if c == b'"' {
                break;
            }
            if c == b'\\' {
                escaped = true;
                if self.s.get(self.i).is_none() {
                    return Err(self.error(DocumentErrorCode::BadJson, p));
                }
                self.i += 1;
            }
        }
        let token = &self.s[start..self.i];
        if let Some(limit) = blob {
            ensure(!escaped, DocumentErrorCode::BadHex, p)?;
            hex_len(token.len() - 2, limit, p)?;
            self.excluded = self
                .excluded
                .checked_add(token.len() - 2)
                .ok_or_else(|| err(DocumentErrorCode::TooLarge, p))?;
            return Ok(None);
        }
        let text: String = serde_json::from_slice(token).map_err(|e| json_err(e, p))?;
        ensure(text.len() <= decoded, DocumentErrorCode::TokenLimit, p)?;
        Ok(key.then_some(text))
    }
    fn value(&mut self, p: &str, depth: usize) -> Result<()> {
        self.ws();
        let c = *self
            .s
            .get(self.i)
            .ok_or_else(|| self.error(DocumentErrorCode::BadJson, p))?;
        match c {
            b'{' | b'[' => {
                if depth >= 16 {
                    return Err(self.error(DocumentErrorCode::DepthLimit, p));
                }
                let object = c == b'{';
                let close = if object { b'}' } else { b']' };
                self.i += 1;
                self.ws();
                if self.s.get(self.i) == Some(&close) {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    let child = if object {
                        self.ws();
                        if self.s.get(self.i) != Some(&b'"') {
                            return Err(self.error(DocumentErrorCode::BadJson, p));
                        }
                        let k = self.string(p, true)?.unwrap();
                        self.ws();
                        if self.s.get(self.i) != Some(&b':') {
                            return Err(self.error(DocumentErrorCode::BadJson, p));
                        }
                        self.i += 1;
                        let display = path(p, &k);
                        self.structural_path.push(PathComponent::Key(k));
                        display
                    } else {
                        self.structural_path.push(PathComponent::ArrayElement);
                        path(p, "*")
                    };
                    let result = self.value(&child, depth + 1);
                    self.structural_path.pop();
                    result?;
                    self.ws();
                    match self.s.get(self.i) {
                        Some(x) if *x == close => {
                            self.i += 1;
                            break;
                        }
                        Some(b',') => self.i += 1,
                        _ => return Err(self.error(DocumentErrorCode::BadJson, p)),
                    }
                }
            }
            b'"' => {
                self.string(p, false)?;
            }
            _ => {
                let start = self.i;
                while self.s.get(self.i).is_some_and(|c| {
                    !matches!(c, b' ' | b'\r' | b'\n' | b'\t' | b',' | b']' | b'}')
                }) {
                    self.i += 1;
                    if self.i - start > 64 {
                        return Err(self.error(DocumentErrorCode::TokenLimit, p));
                    }
                }
                if self.i == start {
                    return Err(self.error(DocumentErrorCode::BadJson, p));
                }
            }
        }
        Ok(())
    }
}
fn preflight(s: &[u8]) -> Result<()> {
    preflight_layout(s, DocumentLayout::Reference)
}
fn preflight_layout(s: &[u8], layout: DocumentLayout) -> Result<()> {
    ensure(
        s.len() <= layout.file_cap(),
        DocumentErrorCode::TooLarge,
        "",
    )?;
    std::str::from_utf8(s).map_err(|_| err(DocumentErrorCode::BadJson, ""))?;
    let mut scan = Scan {
        layout,
        s,
        i: 0,
        excluded: 0,
        structural_path: Vec::new(),
    };
    scan.value("", 0)?;
    scan.ws();
    ensure(scan.i == s.len(), DocumentErrorCode::BadJson, "")?;
    ensure(
        s.len()
            .checked_sub(scan.excluded)
            .is_some_and(|n| n <= A_MAX),
        DocumentErrorCode::TooLarge,
        "",
    )
}
struct Fields<'a> {
    values: Vec<Option<&'a RawValue>>,
    names: &'static [&'static str],
}
struct FieldSeed(&'static [&'static str]);
impl<'de> DeserializeSeed<'de> for FieldSeed {
    type Value = Fields<'de>;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for FieldSeed {
    type Value = Fields<'de>;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("exact-field object")
    }
    fn visit_map<M: MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        let mut seen = 0u64;
        let mut values = vec![None; self.0.len()];
        while let Some(k) = map.next_key::<String>()? {
            let i = self
                .0
                .iter()
                .position(|x| *x == k)
                .ok_or_else(|| de::Error::custom("UnknownField"))?;
            if seen & (1 << i) != 0 {
                return Err(de::Error::custom("DuplicateField"));
            }
            seen |= 1 << i;
            values[i] = Some(map.next_value::<&'de RawValue>()?);
        }
        Ok(Fields {
            values,
            names: self.0,
        })
    }
}
impl<'a> Fields<'a> {
    fn new(raw: &'a RawValue, names: &'static [&'static str], p: &str) -> Result<Self> {
        let mut d = serde_json::Deserializer::from_str(raw.get());
        FieldSeed(names)
            .deserialize(&mut d)
            .map_err(|e| json_err(e, p))
    }
    fn get(&mut self, name: &str, p: &str) -> Result<&'a RawValue> {
        let i = self
            .names
            .iter()
            .position(|n| *n == name)
            .expect("static field");
        self.values[i]
            .take()
            .ok_or_else(|| err(DocumentErrorCode::MissingField, &path(p, name)))
    }
    fn field<T: Decode>(&mut self, name: &str, p: &str) -> Result<T> {
        T::decode(self.get(name, p)?, &path(p, name))
    }
}
trait Decode: Sized {
    fn decode(r: &RawValue, p: &str) -> Result<Self>;
}
fn bounded_add(current: usize, extra: usize, limit: usize, p: &str) -> Result<usize> {
    current
        .checked_add(extra)
        .filter(|n| *n <= limit)
        .ok_or_else(|| err(DocumentErrorCode::TooLarge, p))
}
struct Encoder {
    layout: DocumentLayout,
    bytes: Vec<u8>,
    outer: usize,
}
impl Encoder {
    fn new(layout: DocumentLayout) -> Self {
        Self {
            layout,
            bytes: Vec::new(),
            outer: 0,
        }
    }
    fn append(&mut self, b: &[u8], blob: bool) -> Result<()> {
        bounded_add(
            self.bytes.len(),
            b.len(),
            self.layout.canonical_cap(),
            "canonical",
        )?;
        if !blob {
            self.outer = bounded_add(self.outer, b.len(), A_MAX, "canonical")?;
        }
        self.bytes.extend_from_slice(b);
        Ok(())
    }
    fn byte(&mut self, x: u8) -> Result<()> {
        self.append(&[x], false)
    }
    fn u64(&mut self, x: u64) -> Result<()> {
        self.append(&x.to_le_bytes(), false)
    }
    fn text(&mut self, x: &str) -> Result<()> {
        self.u64(x.len() as u64)?;
        self.append(x.as_bytes(), false)
    }
}
trait Encode {
    fn encode(&self, e: &mut Encoder) -> Result<()>;
}
impl Decode for u64 {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let s = r.get();
        ensure(
            !s.is_empty()
                && s.bytes().all(|x| x.is_ascii_digit())
                && (s == "0" || !s.starts_with('0')),
            DocumentErrorCode::OutOfRange,
            p,
        )?;
        s.parse().map_err(|_| err(DocumentErrorCode::OutOfRange, p))
    }
}
impl Encode for u64 {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.u64(*self)
    }
}
impl Decode for bool {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        serde_json::from_str(r.get()).map_err(|x| json_err(x, p))
    }
}
impl Encode for bool {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.byte(u8::from(*self))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Text<const N: usize>(String);
impl<const N: usize> Decode for Text<N> {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let s: String = serde_json::from_str(r.get()).map_err(|e| json_err(e, p))?;
        ensure(s.len() <= N, DocumentErrorCode::OutOfRange, p)?;
        Ok(Self(s))
    }
}
impl<const N: usize> Encode for Text<N> {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.text(&self.0)
    }
}
impl<T: Decode> Decode for Option<T> {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        if r.get() == "null" {
            Ok(None)
        } else {
            T::decode(r, p).map(Some)
        }
    }
}
impl<T: Encode> Encode for Option<T> {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.byte(u8::from(self.is_some()))?;
        if let Some(x) = self {
            x.encode(e)?;
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct Hash(String);
fn lower_hex(s: &str) -> bool {
    s.bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl Decode for Hash {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let s = Text::<64>::decode(r, p)?.0;
        ensure(s.len() == 64 && lower_hex(&s), DocumentErrorCode::BadHex, p)?;
        Ok(Self(s))
    }
}
impl Encode for Hash {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.text(&self.0)
    }
}
struct ArraySeed<'a, T> {
    p: &'a str,
    limit: usize,
    marker: PhantomData<T>,
    error: &'a mut Option<DocumentError>,
}
impl<'de, T: Decode> Visitor<'de> for ArraySeed<'_, T> {
    type Value = Vec<T>;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("bounded array")
    }
    fn visit_seq<S: SeqAccess<'de>>(self, mut s: S) -> std::result::Result<Self::Value, S::Error> {
        let mut v = Vec::new();
        while let Some(raw) = s.next_element::<&RawValue>()? {
            if v.len() == self.limit {
                *self.error = Some(err(DocumentErrorCode::OutOfRange, self.p));
                return Err(de::Error::custom("bounded array rejection"));
            }
            match T::decode(raw, self.p) {
                Ok(x) => v.push(x),
                Err(e) => {
                    *self.error = Some(e);
                    return Err(de::Error::custom("typed element rejection"));
                }
            }
        }
        Ok(v)
    }
}
#[derive(Debug, Clone)]
struct List<T, const N: usize>(Vec<T>);
impl<T: Decode, const N: usize> Decode for List<T, N> {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut d = serde_json::Deserializer::from_str(r.get());
        let mut error = None;
        de::Deserializer::deserialize_seq(
            &mut d,
            ArraySeed {
                p,
                limit: N,
                marker: PhantomData,
                error: &mut error,
            },
        )
        .map(Self)
        .map_err(|e| error.unwrap_or_else(|| json_err(e, p)))
    }
}
impl<T: Encode, const N: usize> Encode for List<T, N> {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.u64(self.0.len() as u64)?;
        for x in &self.0 {
            x.encode(e)?;
        }
        Ok(())
    }
}
// These macros declare fixed private record fields in their normative encoding
// order. They never retain a generic JSON map or expose arbitrary-metric encoding.
macro_rules! record {
    ($name:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        #[derive(Debug, Clone)]
        struct $name { $($field: $ty),* }
        impl Decode for $name {
            fn decode(raw: &RawValue, p: &str) -> Result<Self> {
                let mut fields = Fields::new(raw, &[$(stringify!($field)),*], p)?;
                Ok(Self { $($field: fields.field(stringify!($field), p)?),* })
            }
        }
        impl Encode for $name {
            fn encode(&self, encoder: &mut Encoder) -> Result<()> {
                $(self.$field.encode(encoder)?;)*
                Ok(())
            }
        }
    };
}
macro_rules! tags {
    ($name:ident { $($variant:ident = $tag:literal => $wire:literal),* $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum $name { $($variant),* }
        impl Decode for $name {
            fn decode(raw: &RawValue, p: &str) -> Result<Self> {
                match Text::<128>::decode(raw, p)?.0.as_str() {
                    $($wire => Ok(Self::$variant),)*
                    _ => Err(err(DocumentErrorCode::UnsupportedTag, p)),
                }
            }
        }
        impl Encode for $name {
            fn encode(&self, encoder: &mut Encoder) -> Result<()> {
                encoder.byte(match self { $(Self::$variant => $tag),* })
            }
        }
    };
}
tags!(Kind {
    Reference=0=>"full_fp32_final",
    Quick=1=>"diagnostic_quick",
    Full=2=>"diagnostic_full",
    Failure=3=>"reference_failure",
});
tags!(Status {
    Complete=0=>"complete",
    Failed=1=>"failed",
});
tags!(Dtype {
    F32=32=>"f32",
    F64=64=>"f64",
});
tags!(Accumulation {
    F32=0=>"scalar_f32",
    F64=1=>"scalar_f64",
});
tags!(Cast {
    F32=0=>"f64_to_f32_once",
    F64=1=>"canonical_f64",
});
tags!(DiagnosticScope {
    Quick=0=>"quick_two_updates",
    Full=1=>"full_diagnostic",
});
tags!(RequestedScope {
    F32=0=>"full_fp32",
    Quick=1=>"quick",
    F64=2=>"full_f64_diagnostic",
});
tags!(Phase {
    Resolve=0=>"resolve",
    Materialize=1=>"materialize",
    Start=2=>"start",
    Update=3=>"update",
    Checkpoint=4=>"checkpoint",
    Finalize=5=>"finalize",
    Publish=6=>"publish",
});
tags!(WallSource {
    Unix=0=>"unix_system_time",
});
tags!(MonotonicSource {
    Instant=0=>"std_instant",
});
tags!(TimingScope {
    After=0=>"after_materialization",
    Includes=1=>"includes_materialization",
});
tags!(GitState {
    Clean=0=>"clean",
    Dirty=1=>"dirty",
    Unavailable=2=>"unavailable",
});
tags!(OsEncoding {
    Unix=0=>"unix_bytes",
});
#[derive(Debug, Clone)]
enum Observation<T> {
    Available(T),
    Unavailable(Text<256>),
}
impl<T: Decode> Decode for Observation<T> {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(r, &["status", "value", "reason"], p)?;
        let status: Text<128> = f.field("status", p)?;
        match status.0.as_str() {
            "available" => {
                ensure(f.values[2].is_none(), DocumentErrorCode::UnknownField, p)?;
                Ok(Self::Available(f.field("value", p)?))
            }
            "unavailable" => {
                ensure(f.values[1].is_none(), DocumentErrorCode::UnknownField, p)?;
                Ok(Self::Unavailable(f.field("reason", p)?))
            }
            _ => Err(err(DocumentErrorCode::UnsupportedTag, p)),
        }
    }
}
impl<T: Encode> Encode for Observation<T> {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        match self {
            Self::Available(x) => {
                e.byte(0)?;
                x.encode(e)
            }
            Self::Unavailable(x) => {
                e.byte(1)?;
                x.encode(e)
            }
        }
    }
}
impl<T> Observation<T> {
    fn available(&self) -> Option<&T> {
        if let Self::Available(x) = self {
            Some(x)
        } else {
            None
        }
    }
}
fn hex_len(n: usize, cap: usize, p: &str) -> Result<usize> {
    ensure(n % 2 == 0, DocumentErrorCode::BadHex, p)?;
    let count = n / 2;
    ensure(count <= cap, DocumentErrorCode::TooLarge, p)?;
    Ok(count)
}
fn bytes(r: &RawValue, cap: usize, p: &str, opaque: bool) -> Result<Vec<u8>> {
    let raw = r.get();
    ensure(
        raw.starts_with('"') && raw.ends_with('"'),
        DocumentErrorCode::BadJson,
        p,
    )?;
    let borrowed = &raw[1..raw.len() - 1];
    let decoded;
    let s = if opaque {
        borrowed
    } else {
        decoded = Text::<8192>::decode(r, p)?.0;
        &decoded
    };
    let len = hex_len(s.len(), cap, p)?;
    ensure(lower_hex(s), DocumentErrorCode::BadHex, p)?;
    fn nibble(b: u8) -> u8 {
        if b <= b'9' { b - b'0' } else { b - b'a' + 10 }
    }
    let mut out = Vec::with_capacity(len);
    for pair in s.as_bytes().chunks_exact(2) {
        out.push(nibble(pair[0]) * 16 + nibble(pair[1]));
    }
    Ok(out)
}
/// Opaque untrusted complete identity bytes. Matching SHA256 is NOT semantic validity.
#[derive(Debug, Clone)]
pub struct ClaimedIdentityBytes {
    schema: Text<128>,
    encoding: Text<128>,
    bytes: Vec<u8>,
    sha256: Hash,
}
impl ClaimedIdentityBytes {
    /// Full supplied byte string, never an imported trusted identity.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Byte-integrity digest, not attestation or approved-context verification.
    pub fn sha256(&self) -> &str {
        &self.sha256.0
    }
    fn parse(r: &RawValue, p: &str, schema: &str, cap: usize) -> Result<Self> {
        let mut f = Fields::new(r, &["schema", "encoding", "canonical_bytes", "sha256"], p)?;
        let result = Self {
            schema: f.field("schema", p)?,
            encoding: f.field("encoding", p)?,
            bytes: bytes(
                f.get("canonical_bytes", p)?,
                cap,
                &path(p, "canonical_bytes"),
                true,
            )?,
            sha256: f.field("sha256", p)?,
        };
        ensure(
            result.schema.0 == schema && result.encoding.0 == ENCODING_VERSION,
            DocumentErrorCode::UnsupportedSchema,
            p,
        )?;
        ensure(!result.bytes.is_empty(), DocumentErrorCode::OutOfRange, p)?;
        ensure(
            sha256_hex(&result.bytes) == result.sha256.0,
            DocumentErrorCode::DigestMismatch,
            p,
        )?;
        Ok(result)
    }
}
impl Encode for ClaimedIdentityBytes {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        self.schema.encode(e)?;
        self.encoding.encode(e)?;
        e.u64(self.bytes.len() as u64)?;
        e.append(&self.bytes, true)?;
        self.sha256.encode(e)
    }
}
/// Requested profile/seed plus untrusted transported identity claims. No public constructor.
#[derive(Debug, Clone)]
pub struct ClaimedIdentity {
    profile_id: Text<128>,
    seed: u64,
    materialized_sha256: Hash,
    contract: ClaimedIdentityBytes,
    seed_receipt: ClaimedIdentityBytes,
}
impl ClaimedIdentity {
    /// Identifier requested by this untrusted document; not registry approval.
    pub fn profile_id(&self) -> &str {
        &self.profile_id.0
    }
    /// Requested seed, not target eligibility.
    pub fn seed(&self) -> u64 {
        self.seed
    }
    /// Opaque materialization link requiring trusted-context comparison.
    pub fn materialized_sha256(&self) -> &str {
        &self.materialized_sha256.0
    }
    /// Complete shared identity transport claim.
    pub fn contract(&self) -> &ClaimedIdentityBytes {
        &self.contract
    }
    /// Complete selected identity transport claim.
    pub fn seed_receipt(&self) -> &ClaimedIdentityBytes {
        &self.seed_receipt
    }
}
impl Decode for ClaimedIdentity {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(
            r,
            &[
                "profile_id",
                "seed",
                "materialized_sha256",
                "contract",
                "seed_receipt",
            ],
            p,
        )?;
        let x = Self {
            profile_id: f.field("profile_id", p)?,
            seed: f.field("seed", p)?,
            materialized_sha256: f.field("materialized_sha256", p)?,
            contract: ClaimedIdentityBytes::parse(
                f.get("contract", p)?,
                &path(p, "contract"),
                CONTRACT_VERSION,
                C_MAX,
            )?,
            seed_receipt: ClaimedIdentityBytes::parse(
                f.get("seed_receipt", p)?,
                &path(p, "seed_receipt"),
                RECEIPT_VERSION,
                R_MAX,
            )?,
        };
        ensure(x.seed <= 9, DocumentErrorCode::OutOfRange, p)?;
        Ok(x)
    }
}
impl Encode for ClaimedIdentity {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        self.profile_id.encode(e)?;
        self.seed.encode(e)?;
        self.materialized_sha256.encode(e)?;
        self.contract.encode(e)?;
        self.seed_receipt.encode(e)
    }
}
/// Exact count/ratio self-consistency only; this is not measured accuracy evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedAccuracy {
    correct: u64,
    test_count: u64,
    bits: u64,
}
impl ClaimedAccuracy {
    /// Untrusted declared correct count.
    pub fn claimed_correct(&self) -> u64 {
        self.correct
    }
    /// Untrusted held-out denominator; requires trusted-data comparison.
    pub fn claimed_test_count(&self) -> u64 {
        self.test_count
    }
    /// Ratio derived exactly from the supplied counts, not proof of evaluation.
    pub fn claimed_value(&self) -> f64 {
        f64::from_bits(self.bits)
    }
}
impl Decode for ClaimedAccuracy {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(r, &["correct", "test_count", "value", "bits"], p)?;
        let correct: u64 = f.field("correct", p)?;
        let test_count: u64 = f.field("test_count", p)?;
        ensure(
            test_count > 0 && test_count <= 1 << 53 && correct <= test_count,
            DocumentErrorCode::OutOfRange,
            p,
        )?;
        let raw = f.get("value", p)?.get();
        ensure(raw.len() <= 64, DocumentErrorCode::TokenLimit, p)?;
        // RawValue preserves lexical spelling. serde validates number grammar without
        // converting through Value/Number; local std parsing preserves reporting bits.
        ensure(
            raw.as_bytes()
                .first()
                .is_some_and(|x| x.is_ascii_digit() || *x == b'-'),
            DocumentErrorCode::BadJson,
            p,
        )?;
        let value: f64 = raw
            .parse()
            .map_err(|_| err(DocumentErrorCode::BadJson, p))?;
        ensure(value.is_finite(), DocumentErrorCode::NonFinite, p)?;
        let spelling: Text<16> = f.field("bits", p)?;
        ensure(
            spelling.0.len() == 16 && lower_hex(&spelling.0),
            DocumentErrorCode::BadHex,
            p,
        )?;
        let bits =
            u64::from_str_radix(&spelling.0, 16).map_err(|_| err(DocumentErrorCode::BadHex, p))?;
        ensure(
            value.to_bits() == bits && ((correct as f64) / (test_count as f64)).to_bits() == bits,
            DocumentErrorCode::ObservationMismatch,
            p,
        )?;
        Ok(Self {
            correct,
            test_count,
            bits,
        })
    }
}
impl Encode for ClaimedAccuracy {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.u64(self.correct)?;
        e.u64(self.test_count)?;
        e.u64(self.bits)
    }
}
record!(Writer {
    api_version: Text<128>,
    algorithm_revision: Text<128>,
});
record!(Arithmetic {
    dtype: Dtype,
    accumulation: Accumulation,
    cast: Cast,
    reduction: Text<1024>,
    classification: Text<1024>,
});
record!(Endpoint {
    completed_updates: u64,
    processed_samples: u64,
    crossed_logical_boundaries: List<u64,400>,
    exact_logical_boundary: bool,
    interval_updates: u64,
    interval_samples: u64,
    observation: ClaimedAccuracy,
    update_elapsed_ns: Observation<u64>,
    evaluation_elapsed_ns: Observation<u64>,
});
record!(Completion {
    origin_completed_updates: u64,
    origin_processed_samples: u64,
    planned_updates: u64,
    actual_updates: u64,
    planned_samples: u64,
    actual_samples: u64,
    planned_dropped_samples: u64,
    actual_dropped_samples: u64,
    full_plan_sha256: Hash,
    consumed_plan_sha256: Hash,
    final_endpoint_index: u64,
    final_state_sha256: Hash
});
record!(DiagnosticProgress {
    origin_completed_updates: u64,
    origin_processed_samples: u64,
    planned_updates: u64,
    consumed_updates: u64,
    planned_samples: u64,
    consumed_samples: u64,
    planned_dropped_samples: u64,
    full_plan_sha256: Hash,
    consumed_plan_sha256: Hash
});
record!(FailureProgress {
    completed_updates: u64,
    processed_samples: u64
});
record!(FailureError {
    code: Text<128>,
    message: Text<4096>,
});
record!(Timing {
    wall_source: WallSource,
    start_unix_ns: u64,
    end_unix_ns: u64,
    monotonic_source: MonotonicSource,
    scope: TimingScope,
    end_to_end_ns: u64,
    setup_ns: Observation<u64>,
    update_ns: Observation<u64>,
    evaluation_ns: Observation<u64>,
    finalization_ns: Observation<u64>,
});
record!(Build {
    compiler_vv: Text<16384>,
    target: Text<4096>,
    host: Text<4096>,
    profile: Text<4096>,
    opt_level: Text<4096>,
    debug: bool,
    features: List<Text<4096>,64>,
    package_version: Text<4096>,
    cargo_lock_sha256: Hash,
    source_manifest_sha256: Hash,
    config_manifest_sha256: Hash,
    build_environment_sha256: Hash,
    recipe_version: Text<4096>,
    receipt_file_sha256: Hash,
});
#[derive(Debug, Clone)]
struct RawOs {
    encoding: OsEncoding,
    bytes: Vec<u8>,
}
impl Decode for RawOs {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(r, &["encoding", "bytes"], p)?;
        Ok(Self {
            encoding: f.field("encoding", p)?,
            bytes: bytes(f.get("bytes", p)?, 4096, &path(p, "bytes"), false)?,
        })
    }
}
impl Encode for RawOs {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        self.encoding.encode(e)?;
        e.u64(self.bytes.len() as u64)?;
        e.append(&self.bytes, false)
    }
}
#[derive(Debug, Clone)]
struct Revision(Text<40>);
impl Decode for Revision {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let x = Text::<40>::decode(r, p)?;
        ensure(
            x.0.len() == 40 && lower_hex(&x.0),
            DocumentErrorCode::BadHex,
            p,
        )?;
        Ok(Self(x))
    }
}
impl Encode for Revision {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        self.0.encode(e)
    }
}
record!(SourceFile {
    path: Text<4096>,
    sha256: Hash,
});
record!(Source {
    root: RawOs,
    commit: Observation<Revision>,
    tracked: GitState,
    untracked: GitState,
    manifest: List<SourceFile,512>,
});
record!(Executable {
    path: RawOs,
    start_sha256: Hash,
    end_sha256: Hash
});
record!(Invocation {
    argv: List<RawOs,256>,
    cwd: RawOs,
});
record!(Host {
    cpu_model: Observation<Text<4096>>,
    os: Observation<Text<4096>>,
    hostname: Observation<Text<4096>>,
    available_cpus: Observation<u64>,
    process_id: Observation<u64>,
    observed_threads: Observation<u64>,
    execution_threads: u64,
});
record!(ThreadEnvironment {
    key: Text<4096>,
    value: Observation<Text<4096>>,
});
record!(Recheck {
    source_manifest_sha256: Hash,
    build_receipt_file_sha256: Hash
});
record!(Provenance {
    schema: Text<128>,
    build: Build,
    source: Source,
    executable: Executable,
    invocation: Invocation,
    host: Host,
    thread_environment: List<ThreadEnvironment,32>,
    recheck: Recheck,
});
#[derive(Debug, Clone)]
enum ProvenanceClaim {
    Captured(Box<Provenance>),
    Unavailable(Text<256>),
}
impl Decode for ProvenanceClaim {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(r, &["status", "value", "reason"], p)?;
        let status: Text<128> = f.field("status", p)?;
        match status.0.as_str() {
            "captured" => {
                ensure(f.values[2].is_none(), DocumentErrorCode::UnknownField, p)?;
                Ok(Self::Captured(Box::new(f.field("value", p)?)))
            }
            "unavailable" => {
                ensure(f.values[1].is_none(), DocumentErrorCode::UnknownField, p)?;
                Ok(Self::Unavailable(f.field("reason", p)?))
            }
            _ => Err(err(DocumentErrorCode::UnsupportedTag, p)),
        }
    }
}
impl Encode for ProvenanceClaim {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        match self {
            Self::Captured(x) => {
                e.byte(0)?;
                x.encode(e)
            }
            Self::Unavailable(x) => {
                e.byte(1)?;
                x.encode(e)
            }
        }
    }
}
record!(Integrity {
    encoding: Text<128>,
    hash: Text<128>,
    payload_sha256: Hash,
});
#[derive(Debug, Clone)]
struct Common {
    schema: Text<128>,
    kind: Kind,
    status: Status,
    run_id: Text<128>,
    writer: Writer,
    synthetic: bool,
    arithmetic: Arithmetic,
    provenance: ProvenanceClaim,
    integrity: Integrity,
    file_sha256: String,
}
#[derive(Debug, Clone)]
struct ReferenceBody {
    identity: ClaimedIdentity,
    completion: Completion,
    endpoints: List<Endpoint, 400>,
    final_observation: ClaimedAccuracy,
    timing: Timing,
}
#[derive(Debug, Clone)]
struct DiagnosticBody {
    identity: ClaimedIdentity,
    target_eligible: bool,
    scope: DiagnosticScope,
    progress: DiagnosticProgress,
    endpoints: List<Endpoint, 400>,
    last_observation: ClaimedAccuracy,
    last_state_sha256: Hash,
    timing: Timing,
}
#[derive(Debug, Clone)]
struct FailureBody {
    identity: Option<ClaimedIdentity>,
    target_eligible: bool,
    requested_scope: RequestedScope,
    phase: Phase,
    error: FailureError,
    last_successful_endpoint: Option<Endpoint>,
    progress: Option<FailureProgress>,
    timing: Option<Timing>,
}
/// Untrusted full-reference declarations. Parsing proves neither training nor capture.
///
/// A future approved-context verifier must reconstruct this exact canonical seed,
/// compare both full identity byte arrays, and validate the full plan, checkpoints,
/// denominators and approved source/build/session policy before any target eligibility.
/// No parser conversion into a trusted identity or completed result exists.
/// ```compile_fail
/// use ckks_train_bench::study_reference::ParsedReferenceDocument;
/// let forged = ParsedReferenceDocument { common: todo!(), body: todo!() };
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_reference::{parse_reference_document, StudyProfileContractV1};
/// let parsed = parse_reference_document(b"{}").unwrap();
/// let trusted: StudyProfileContractV1 = parsed.claimed_identity().contract().into();
/// ```
#[derive(Debug)]
pub struct ParsedReferenceDocument {
    common: Common,
    body: ReferenceBody,
}
/// Permanently target-ineligible untrusted diagnostic declarations.
#[derive(Debug)]
pub struct ParsedDiagnosticDocument {
    common: Common,
    body: DiagnosticBody,
}
/// Untrusted failure declarations, never a successful final observation.
#[derive(Debug)]
pub struct ParsedFailureDocument {
    common: Common,
    body: FailureBody,
}
macro_rules! common_accessors {
    ($name:ident) => {
        impl $name {
            /// SHA256 of all exact supplied file bytes, including whitespace.
            pub fn file_sha256(&self) -> &str {
                &self.common.file_sha256
            }
            /// Integrity of the typed outer envelope; not scientific verification.
            pub fn payload_sha256(&self) -> &str {
                &self.common.integrity.payload_sha256.0
            }
            /// Untrusted run identifier, without filesystem authority.
            pub fn claimed_run_id(&self) -> &str {
                &self.common.run_id.0
            }
        }
    };
}
common_accessors!(ParsedReferenceDocument);
common_accessors!(ParsedDiagnosticDocument);
common_accessors!(ParsedFailureDocument);
impl ParsedReferenceDocument {
    /// Opaque identity declarations requiring exact approved-context byte comparison.
    pub fn claimed_identity(&self) -> &ClaimedIdentity {
        &self.body.identity
    }
    /// Claimed final ratio; not proof of execution or evaluation.
    pub fn claimed_final_observation(&self) -> &ClaimedAccuracy {
        &self.body.final_observation
    }
}
impl ParsedDiagnosticDocument {
    /// Opaque identity declarations; never target-ready.
    pub fn claimed_identity(&self) -> &ClaimedIdentity {
        &self.body.identity
    }
    /// Terminal evaluation claim, possibly after the last checkpoint.
    pub fn claimed_last_observation(&self) -> &ClaimedAccuracy {
        &self.body.last_observation
    }
}
impl ParsedFailureDocument {
    /// Absence remains absent, including late phases; no zero-filled identity.
    pub fn claimed_identity(&self) -> Option<&ClaimedIdentity> {
        self.body.identity.as_ref()
    }
}
fn nonempty<const N: usize>(t: &Text<N>, p: &str) -> Result<()> {
    ensure(!t.0.is_empty(), DocumentErrorCode::OutOfRange, p)
}
fn common(f: &mut Fields<'_>, file_sha256: String, expected: &str) -> Result<Common> {
    let x = Common {
        schema: f.field("schema", "")?,
        kind: f.field("kind", "")?,
        status: f.field("status", "")?,
        run_id: f.field("run_id", "")?,
        writer: f.field("writer", "")?,
        synthetic: f.field("synthetic", "")?,
        arithmetic: f.field("arithmetic", "")?,
        provenance: f.field("provenance", "")?,
        integrity: f.field("integrity", "")?,
        file_sha256,
    };
    ensure(
        x.schema.0 == expected,
        DocumentErrorCode::UnsupportedSchema,
        "schema",
    )?;
    ensure(!x.synthetic, DocumentErrorCode::UnsupportedTag, "synthetic")?;
    ensure(
        !x.run_id.0.is_empty()
            && x.run_id
                .0
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
        DocumentErrorCode::OutOfRange,
        "run_id",
    )?;
    nonempty(&x.writer.api_version, "writer.api_version")?;
    nonempty(&x.writer.algorithm_revision, "writer.algorithm_revision")?;
    nonempty(&x.arithmetic.reduction, "arithmetic.reduction")?;
    nonempty(&x.arithmetic.classification, "arithmetic.classification")?;
    let a = &x.arithmetic;
    ensure(
        matches!(
            (a.dtype, a.accumulation, a.cast),
            (Dtype::F32, Accumulation::F32, Cast::F32) | (Dtype::F64, Accumulation::F64, Cast::F64)
        ),
        DocumentErrorCode::UnsupportedTag,
        "arithmetic",
    )?;
    ensure(
        x.integrity.encoding.0 == ENCODING_VERSION && x.integrity.hash.0 == "sha256",
        DocumentErrorCode::UnsupportedSchema,
        "integrity",
    )?;
    Ok(x)
}
fn prepare(bytes: &[u8]) -> Result<(String, &RawValue)> {
    ensure(bytes.len() <= J_MAX, DocumentErrorCode::TooLarge, "")?;
    // SHA scratch is dropped before parsing, hex decoding or canonical encoding.
    let digest = sha256_hex(bytes);
    preflight(bytes)?;
    let raw = serde_json::from_slice(bytes).map_err(|e| json_err(e, ""))?;
    Ok((digest, raw))
}
fn start_encoding(c: &Common) -> Result<Encoder> {
    let mut e = Encoder {
        layout: DocumentLayout::Reference,
        bytes: Vec::new(),
        outer: 0,
    };
    e.text(ENCODING_VERSION)?;
    e.text(&c.schema.0)?;
    c.schema.encode(&mut e)?;
    c.kind.encode(&mut e)?;
    c.status.encode(&mut e)?;
    c.run_id.encode(&mut e)?;
    c.writer.encode(&mut e)?;
    c.synthetic.encode(&mut e)?;
    Ok(e)
}
fn finish_encoding(c: &Common, e: &mut Encoder) -> Result<()> {
    c.provenance.encode(e)?;
    c.integrity.encoding.encode(e)?;
    c.integrity.hash.encode(e)
}
fn reference_encoding(c: &Common, b: &ReferenceBody) -> Result<Encoder> {
    let mut e = start_encoding(c)?;
    b.identity.encode(&mut e)?;
    c.arithmetic.encode(&mut e)?;
    b.completion.encode(&mut e)?;
    b.endpoints.encode(&mut e)?;
    b.final_observation.encode(&mut e)?;
    b.timing.encode(&mut e)?;
    finish_encoding(c, &mut e)?;
    Ok(e)
}
fn diagnostic_encoding(c: &Common, b: &DiagnosticBody) -> Result<Encoder> {
    let mut e = start_encoding(c)?;
    b.identity.encode(&mut e)?;
    c.arithmetic.encode(&mut e)?;
    b.target_eligible.encode(&mut e)?;
    b.scope.encode(&mut e)?;
    b.progress.encode(&mut e)?;
    b.endpoints.encode(&mut e)?;
    b.last_observation.encode(&mut e)?;
    b.last_state_sha256.encode(&mut e)?;
    b.timing.encode(&mut e)?;
    finish_encoding(c, &mut e)?;
    Ok(e)
}
fn failure_encoding(c: &Common, b: &FailureBody) -> Result<Encoder> {
    let mut e = start_encoding(c)?;
    b.identity.encode(&mut e)?;
    c.arithmetic.encode(&mut e)?;
    b.target_eligible.encode(&mut e)?;
    b.requested_scope.encode(&mut e)?;
    b.phase.encode(&mut e)?;
    b.error.encode(&mut e)?;
    b.last_successful_endpoint.encode(&mut e)?;
    b.progress.encode(&mut e)?;
    b.timing.encode(&mut e)?;
    finish_encoding(c, &mut e)?;
    Ok(e)
}
fn verify_digest(c: &Common, e: Encoder) -> Result<()> {
    ensure(
        sha256_hex(&e.bytes) == c.integrity.payload_sha256.0,
        DocumentErrorCode::DigestMismatch,
        "integrity.payload_sha256",
    )
}
fn timing(t: &Timing) -> Result<()> {
    ensure(
        t.start_unix_ns <= t.end_unix_ns,
        DocumentErrorCode::TimingMismatch,
        "timing",
    )?;
    let mut sum = 0u64;
    for value in [
        &t.setup_ns,
        &t.update_ns,
        &t.evaluation_ns,
        &t.finalization_ns,
    ] {
        if let Some(n) = value.available() {
            sum = sum
                .checked_add(*n)
                .ok_or_else(|| err(DocumentErrorCode::TimingMismatch, "timing"))?;
        }
    }
    ensure(
        sum <= t.end_to_end_ns,
        DocumentErrorCode::TimingMismatch,
        "timing",
    )
}
fn endpoint_timing(e: &Endpoint, t: Option<&Timing>) -> Result<()> {
    if let Some(t) = t {
        for (value, total) in [
            (&e.update_elapsed_ns, &t.update_ns),
            (&e.evaluation_elapsed_ns, &t.evaluation_ns),
        ] {
            if let Some(n) = value.available() {
                ensure(
                    *n <= t.end_to_end_ns && total.available().is_none_or(|total| n <= total),
                    DocumentErrorCode::TimingMismatch,
                    "endpoints.timing",
                )?;
            }
        }
    }
    Ok(())
}
fn endpoints(v: &[Endpoint], t: &Timing) -> Result<()> {
    let (mut updates, mut samples, mut crossed) = (0u64, 0u64, 0usize);
    let (mut update_ns, mut evaluation_ns) = (None, None);
    for x in v {
        ensure(
            x.completed_updates > updates
                && x.processed_samples > samples
                && x.interval_updates == x.completed_updates - updates
                && x.interval_samples == x.processed_samples - samples,
            DocumentErrorCode::OuterProgressMismatch,
            "endpoints",
        )?;
        updates = x.completed_updates;
        samples = x.processed_samples;
        crossed = crossed
            .checked_add(x.crossed_logical_boundaries.0.len())
            .ok_or_else(|| {
                err(
                    DocumentErrorCode::OutOfRange,
                    "endpoints.crossed_logical_boundaries",
                )
            })?;
        ensure(
            crossed <= 400,
            DocumentErrorCode::OutOfRange,
            "endpoints.crossed_logical_boundaries",
        )?;
        for (observation, previous) in [
            (&x.update_elapsed_ns, &mut update_ns),
            (&x.evaluation_elapsed_ns, &mut evaluation_ns),
        ] {
            if let Some(n) = observation.available() {
                ensure(
                    previous.is_none_or(|prev| *n >= prev),
                    DocumentErrorCode::TimingMismatch,
                    "endpoints.timing",
                )?;
                *previous = Some(*n);
            }
        }
        endpoint_timing(x, Some(t))?;
    }
    Ok(())
}
fn unique_by<T, F: Fn(&T) -> &str>(v: &mut [T], key: F, p: &str) -> Result<()> {
    v.sort_by(|a, b| key(a).cmp(key(b)));
    ensure(
        v.windows(2).all(|x| key(&x[0]) != key(&x[1])),
        DocumentErrorCode::DuplicateField,
        p,
    )
}
fn provenance(c: &mut Common, complete: bool) -> Result<()> {
    validate_provenance(&mut c.provenance, complete)
}
fn validate_provenance(claim: &mut ProvenanceClaim, complete: bool) -> Result<()> {
    let ProvenanceClaim::Captured(p) = claim else {
        return Ok(());
    };
    ensure(
        p.schema.0 == "hekit-study-provenance-v1",
        DocumentErrorCode::UnsupportedSchema,
        "provenance.schema",
    )?;
    // Root clarification: lexical absolute claims only; argv is unrestricted raw bytes.
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
        |t| &t.0,
        "provenance.build.features",
    )?;
    unique_by(
        &mut p.source.manifest.0,
        |s| &s.path.0,
        "provenance.source.manifest",
    )?;
    for file in &p.source.manifest.0 {
        let s = &file.path.0;
        ensure(
            !s.contains(['\\', '\0'])
                && s.split('/').all(|c| !c.is_empty() && c != "." && c != ".."),
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
        p.host.execution_threads == 1,
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
    if complete {
        ensure(
            p.executable.start_sha256 == p.executable.end_sha256
                && p.build.source_manifest_sha256 == p.recheck.source_manifest_sha256
                && p.build.receipt_file_sha256 == p.recheck.build_receipt_file_sha256,
            DocumentErrorCode::CrossLinkMismatch,
            "provenance.recheck",
        )?;
    }
    Ok(())
}
/// Parse bounded untrusted full F32 claims. No I/O/training, semantic identity
/// verification, completion capability or target eligibility is performed.
pub fn parse_reference_document(bytes: &[u8]) -> Result<ParsedReferenceDocument> {
    let (file, raw) = prepare(bytes)?;
    let mut f = Fields::new(
        raw,
        &[
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
        "",
    )?;
    let mut c = common(&mut f, file, REFERENCE)?;
    ensure(
        c.kind == Kind::Reference
            && c.status == Status::Complete
            && c.arithmetic.dtype == Dtype::F32,
        DocumentErrorCode::UnsupportedTag,
        "kind",
    )?;
    let b = ReferenceBody {
        identity: f.field("identity", "")?,
        completion: f.field("completion", "")?,
        endpoints: f.field("endpoints", "")?,
        final_observation: f.field("final_observation", "")?,
        timing: f.field("timing", "")?,
    };
    let p = &b.completion;
    ensure(
        p.origin_completed_updates == 0
            && p.origin_processed_samples == 0
            && p.actual_updates == p.planned_updates
            && p.actual_samples == p.planned_samples
            && p.actual_dropped_samples == p.planned_dropped_samples,
        DocumentErrorCode::OuterProgressMismatch,
        "completion",
    )?;
    ensure(
        p.full_plan_sha256 == p.consumed_plan_sha256,
        DocumentErrorCode::CrossLinkMismatch,
        "completion.consumed_plan_sha256",
    )?;
    timing(&b.timing)?;
    endpoints(&b.endpoints.0, &b.timing)?;
    let last = b
        .endpoints
        .0
        .last()
        .ok_or_else(|| err(DocumentErrorCode::OuterProgressMismatch, "endpoints"))?;
    ensure(
        p.final_endpoint_index == b.endpoints.0.len() as u64 - 1
            && last.completed_updates == p.actual_updates
            && last.processed_samples == p.actual_samples,
        DocumentErrorCode::OuterProgressMismatch,
        "completion.final_endpoint_index",
    )?;
    ensure(
        last.observation == b.final_observation,
        DocumentErrorCode::ObservationMismatch,
        "final_observation",
    )?;
    provenance(&mut c, true)?;
    verify_digest(&c, reference_encoding(&c, &b)?)?;
    Ok(ParsedReferenceDocument { common: c, body: b })
}
/// Parse permanently target-ineligible quick/full diagnostic declarations.
pub fn parse_diagnostic_document(bytes: &[u8]) -> Result<ParsedDiagnosticDocument> {
    let (file, raw) = prepare(bytes)?;
    let mut f = Fields::new(
        raw,
        &[
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
        "",
    )?;
    let mut c = common(&mut f, file, DIAGNOSTIC)?;
    ensure(
        matches!(c.kind, Kind::Quick | Kind::Full) && c.status == Status::Complete,
        DocumentErrorCode::UnsupportedTag,
        "kind",
    )?;
    let b = DiagnosticBody {
        identity: f.field("identity", "")?,
        target_eligible: f.field("target_eligible", "")?,
        scope: f.field("scope", "")?,
        progress: f.field("progress", "")?,
        endpoints: f.field("endpoints", "")?,
        last_observation: f.field("last_observation", "")?,
        last_state_sha256: f.field("last_state_sha256", "")?,
        timing: f.field("timing", "")?,
    };
    ensure(
        !b.target_eligible
            && matches!(
                (c.kind, b.scope),
                (Kind::Quick, DiagnosticScope::Quick) | (Kind::Full, DiagnosticScope::Full)
            ),
        DocumentErrorCode::UnsupportedTag,
        "scope",
    )?;
    let p = &b.progress;
    ensure(
        p.origin_completed_updates == 0
            && p.origin_processed_samples == 0
            && p.consumed_updates <= p.planned_updates
            && p.consumed_samples <= p.planned_samples,
        DocumentErrorCode::OuterProgressMismatch,
        "progress",
    )?;
    timing(&b.timing)?;
    endpoints(&b.endpoints.0, &b.timing)?;
    for x in &b.endpoints.0 {
        ensure(
            x.completed_updates <= p.consumed_updates && x.processed_samples <= p.consumed_samples,
            DocumentErrorCode::OuterProgressMismatch,
            "endpoints",
        )?;
    }
    let last = b.endpoints.0.last();
    let terminal = last.is_some_and(|x| {
        x.completed_updates == p.consumed_updates && x.processed_samples == p.consumed_samples
    });
    if terminal {
        ensure(
            last.unwrap().observation == b.last_observation,
            DocumentErrorCode::ObservationMismatch,
            "last_observation",
        )?;
    }
    if c.kind == Kind::Quick {
        ensure(
            p.consumed_updates == 2,
            DocumentErrorCode::OuterProgressMismatch,
            "progress.consumed_updates",
        )?;
    } else {
        ensure(
            p.consumed_updates == p.planned_updates
                && p.consumed_samples == p.planned_samples
                && terminal,
            DocumentErrorCode::OuterProgressMismatch,
            "progress",
        )?;
        ensure(
            p.full_plan_sha256 == p.consumed_plan_sha256,
            DocumentErrorCode::CrossLinkMismatch,
            "progress.consumed_plan_sha256",
        )?;
    }
    provenance(&mut c, true)?;
    verify_digest(&c, diagnostic_encoding(&c, &b)?)?;
    Ok(ParsedDiagnosticDocument { common: c, body: b })
}
/// Parse untrusted failure claims. All phases retain independent Option identity,
/// progress and timing; null is never converted into a zero-valued observation.
pub fn parse_failure_document(bytes: &[u8]) -> Result<ParsedFailureDocument> {
    let (file, raw) = prepare(bytes)?;
    let mut f = Fields::new(
        raw,
        &[
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
        "",
    )?;
    let mut c = common(&mut f, file, FAILURE)?;
    ensure(
        c.kind == Kind::Failure && c.status == Status::Failed,
        DocumentErrorCode::UnsupportedTag,
        "kind",
    )?;
    let b = FailureBody {
        identity: f.field("identity", "")?,
        target_eligible: f.field("target_eligible", "")?,
        requested_scope: f.field("requested_scope", "")?,
        phase: f.field("phase", "")?,
        error: f.field("error", "")?,
        last_successful_endpoint: f.field("last_successful_endpoint", "")?,
        progress: f.field("progress", "")?,
        timing: f.field("timing", "")?,
    };
    ensure(
        !b.target_eligible
            && matches!(
                (b.requested_scope, c.arithmetic.dtype),
                (RequestedScope::Quick, _)
                    | (RequestedScope::F32, Dtype::F32)
                    | (RequestedScope::F64, Dtype::F64)
            ),
        DocumentErrorCode::UnsupportedTag,
        "requested_scope",
    )?;
    if let Some(t) = &b.timing {
        timing(t)?;
    }
    if let Some(x) = &b.last_successful_endpoint {
        let p = b
            .progress
            .as_ref()
            .ok_or_else(|| err(DocumentErrorCode::OuterProgressMismatch, "progress"))?;
        ensure(
            b.identity.is_some()
                && x.completed_updates <= p.completed_updates
                && x.processed_samples <= p.processed_samples
                && x.interval_updates > 0
                && x.interval_updates <= x.completed_updates
                && x.interval_samples > 0
                && x.interval_samples <= x.processed_samples,
            DocumentErrorCode::OuterProgressMismatch,
            "last_successful_endpoint",
        )?;
        endpoint_timing(x, b.timing.as_ref())?;
    }
    provenance(&mut c, false)?;
    verify_digest(&c, failure_encoding(&c, &b)?)?;
    Ok(ParsedFailureDocument { common: c, body: b })
}
#[cfg(test)]
#[path = "document_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "document/target_tests.rs"]
mod target_tests;

pub(super) mod target;
