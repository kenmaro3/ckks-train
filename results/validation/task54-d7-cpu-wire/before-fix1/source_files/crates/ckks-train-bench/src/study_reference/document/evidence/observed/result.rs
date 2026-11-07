//! Scientific-result representation only. Metrics remain supplied untrusted bits.
use super::*;
use serde_json::{Value, json};
impl<'a> Wire<'a> for bool {
    const MIN: usize = 1;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(c.tag(1)? == 1)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        e.byte(u8::from(*self))
    }
}
trait JsonWire<'a>: Wire<'a> {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()>;
    fn json(&self) -> Value;
    fn write_json(&self, out: &mut Vec<u8>) -> Result<()> {
        write_value(out, &self.json())
    }
}
impl JsonWire<'_> for u64 {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        u64::decode(r, p)?.write(e)
    }
    fn json(&self) -> Value {
        json!(self)
    }
}
impl JsonWire<'_> for bool {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        bool::decode(r, p)?.write(e)
    }
    fn json(&self) -> Value {
        json!(self)
    }
}
impl<const M: u8> JsonWire<'_> for Tag<M> {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        let v = u64::decode(r, p)?;
        ensure(v <= u64::from(M), DocumentErrorCode::UnsupportedTag, p)?;
        e.byte(v as u8)
    }
    fn json(&self) -> Value {
        json!(self.0)
    }
}
impl<'a, const N: usize> JsonWire<'a> for Str<'a, N> {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        let s = Text::<N>::decode(r, p)?;
        e.text(&s.0)
    }
    fn json(&self) -> Value {
        json!(self.0)
    }
}
impl<'a> JsonWire<'a> for Id<'a> {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        let s = Text::<128>::decode(r, p)?;
        target::valid_id(&s.0, p)?;
        e.text(&s.0)
    }
    fn json(&self) -> Value {
        json!(self.0)
    }
}
impl<'a> JsonWire<'a> for H<'a> {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        e.text(&Hash::decode(r, p)?.0)
    }
    fn json(&self) -> Value {
        json!(self.0)
    }
}
impl<'a> JsonWire<'a> for Git<'a> {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        let s = Text::<40>::decode(r, p)?;
        same(s.0.len() == 40 && lower_hex(&s.0), "json.revision")?;
        e.text(&s.0)
    }
    fn json(&self) -> Value {
        json!(self.0)
    }
}
impl<'a> JsonWire<'a> for Bytes<'a> {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        let b = bytes(r, A_MAX, p, true)?;
        Bytes(&b).write(e)
    }
    fn json(&self) -> Value {
        let mut s = String::with_capacity(self.0.len() * 2);
        const HEX: &[u8] = b"0123456789abcdef";
        for b in self.0 {
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 15) as usize] as char);
        }
        json!(s)
    }
}
impl<'a, T: JsonWire<'a>> JsonWire<'a> for Option<T> {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        if r.get() == "null" {
            e.byte(0)
        } else {
            e.byte(1)?;
            T::encode_json(r, p, e)
        }
    }
    fn json(&self) -> Value {
        self.as_ref().map_or(Value::Null, JsonWire::json)
    }
    fn write_json(&self, out: &mut Vec<u8>) -> Result<()> {
        match self {
            Some(v) => v.write_json(out),
            None => append_json(out, b"null"),
        }
    }
}
struct JsonRaw(Box<RawValue>);
impl Decode for JsonRaw {
    fn decode(r: &RawValue, _: &str) -> Result<Self> {
        Ok(Self(r.to_owned()))
    }
}
impl<'a, T: JsonWire<'a>> JsonWire<'a> for Vec<T> {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        let v = List::<JsonRaw, 400>::decode(r, p)?;
        e.u64(v.0.len() as u64)?;
        for x in v.0 {
            T::encode_json(&x.0, p, e)?;
        }
        Ok(())
    }
    fn json(&self) -> Value {
        Value::Array(self.iter().map(JsonWire::json).collect())
    }
    fn write_json(&self, out: &mut Vec<u8>) -> Result<()> {
        write_array(out, self.iter())
    }
}
impl<'a, T: Wire<'a>, const N: usize> Wire<'a> for [T; N] {
    const MIN: usize = T::MIN * N;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        ensure(
            c.remaining() >= Self::MIN,
            DocumentErrorCode::OutOfRange,
            "array.truncated",
        )?;
        let mut v = Vec::new();
        v.try_reserve_exact(N)
            .map_err(|_| err(DocumentErrorCode::TooLarge, "array.allocation"))?;
        for _ in 0..N {
            v.push(T::read(c)?);
        }
        v.try_into()
            .map_err(|_| err(DocumentErrorCode::OutOfRange, "array.length"))
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        for x in self {
            x.write(e)?;
        }
        Ok(())
    }
}
impl<'a, T: JsonWire<'a>, const N: usize> JsonWire<'a> for [T; N] {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        let v = List::<JsonRaw, N>::decode(r, p)?;
        same(v.0.len() == N, "array.length")?;
        for x in v.0 {
            T::encode_json(&x.0, p, e)?;
        }
        Ok(())
    }
    fn json(&self) -> Value {
        Value::Array(self.iter().map(JsonWire::json).collect())
    }
    fn write_json(&self, out: &mut Vec<u8>) -> Result<()> {
        write_array(out, self.iter())
    }
}
macro_rules! json_wire {($n:ty {$($f:ident:$t:ty => $key:literal),*$(,)?}) => {
impl<'a> JsonWire<'a> for $n {
 fn encode_json(r:&RawValue,p:&str,e:&mut Encoder)->Result<()>{let mut f=Fields::new(r,&[$($key),*],p)?;$(<$t as JsonWire<'a>>::encode_json(f.get($key,p)?,&path(p,$key),e)?;)*Ok(())}
 fn json(&self)->Value{let mut map=serde_json::Map::new();$(map.insert($key.into(),self.$f.json());)*Value::Object(map)}
 fn write_json(&self,out:&mut Vec<u8>)->Result<()> {append_json(out,b"{")?;let mut first=true;$(json_pair(out,$key,&self.$f,&mut first)?;)*append_json(out,b"}")}
}
};}
// Explicit field-name mapping below preserves the declaration order on the wire.
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct Counter {
    pub(in crate::study_reference::document::evidence) values: [u64; 7],
}
impl<'a> Wire<'a> for Counter {
    const MIN: usize = 56;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            values: <[u64; 7]>::read(c)?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.values.write(e)
    }
}
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct Times {
    pub(in crate::study_reference::document::evidence) values: [u64; 9],
}
impl<'a> Wire<'a> for Times {
    const MIN: usize = 72;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            values: <[u64; 9]>::read(c)?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.values.write(e)
    }
}
const COUNTER_KEYS: [&str; 7] = [
    "rotations",
    "relins",
    "ct_ct_tensor",
    "ct_pt_mults",
    "rescales",
    "sse_switches",
    "bootstraps",
];
const TIME_KEYS: [&str; 9] = [
    "setup",
    "server",
    "bootstrap",
    "refresh",
    "input_encryption",
    "diagnostic",
    "fp32",
    "end_to_end",
    "publish",
];
fn named_numbers(
    r: &RawValue,
    p: &str,
    e: &mut Encoder,
    keys: &'static [&'static str],
) -> Result<()> {
    let mut f = Fields::new(r, keys, p)?;
    for key in keys {
        f.field::<u64>(key, p)?.write(e)?;
    }
    Ok(())
}
fn numbers_json(keys: &[&str], values: &[u64]) -> Value {
    Value::Object(
        keys.iter()
            .zip(values)
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect(),
    )
}
impl JsonWire<'_> for Counter {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        named_numbers(r, p, e, &COUNTER_KEYS)
    }
    fn json(&self) -> Value {
        numbers_json(&COUNTER_KEYS, &self.values)
    }
    fn write_json(&self, out: &mut Vec<u8>) -> Result<()> {
        write_numbers(out, &COUNTER_KEYS, &self.values)
    }
}
impl JsonWire<'_> for Times {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        named_numbers(r, p, e, &TIME_KEYS)
    }
    fn json(&self) -> Value {
        numbers_json(&TIME_KEYS, &self.values)
    }
    fn write_json(&self, out: &mut Vec<u8>) -> Result<()> {
        write_numbers(out, &TIME_KEYS, &self.values)
    }
}
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct Endpoint {
    pub(in crate::study_reference::document::evidence) updates: u64,
    pub(in crate::study_reference::document::evidence) samples: u64,
    pub(in crate::study_reference::document::evidence) crossed: Vec<u64>,
    pub(in crate::study_reference::document::evidence) counter: Counter,
    pub(in crate::study_reference::document::evidence) times: Times,
    pub(in crate::study_reference::document::evidence) he_correct: u64,
    pub(in crate::study_reference::document::evidence) fp32_correct: u64,
    pub(in crate::study_reference::document::evidence) test_count: u64,
    pub(in crate::study_reference::document::evidence) weight_max_abs_diff_bits: u64,
    pub(in crate::study_reference::document::evidence) prediction_max_abs_diff_bits: u64,
}
impl<'a> Wire<'a> for Endpoint {
    const MIN: usize = 192;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            updates: c.u64()?,
            samples: c.u64()?,
            crossed: wire::read_vec(c, 400)?,
            counter: Counter::read(c)?,
            times: Times::read(c)?,
            he_correct: c.u64()?,
            fp32_correct: c.u64()?,
            test_count: c.u64()?,
            weight_max_abs_diff_bits: c.u64()?,
            prediction_max_abs_diff_bits: c.u64()?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.updates.write(e)?;
        self.samples.write(e)?;
        self.crossed.write(e)?;
        self.counter.write(e)?;
        self.times.write(e)?;
        self.he_correct.write(e)?;
        self.fp32_correct.write(e)?;
        self.test_count.write(e)?;
        self.weight_max_abs_diff_bits.write(e)?;
        self.prediction_max_abs_diff_bits.write(e)
    }
}
json_wire!(Endpoint{updates:u64 => "updates",samples:u64 => "samples",crossed:Vec<u64> => "crossed",counter:Counter => "counter",times:Times => "times",he_correct:u64 => "he_correct",fp32_correct:u64 => "fp32_correct",test_count:u64 => "test_count",weight_max_abs_diff_bits:u64 => "weight_max_abs_diff_bits",prediction_max_abs_diff_bits:u64 => "prediction_max_abs_diff_bits"});
wire_record!(Failure {update:u64,phase:Id<'a>,counter:Counter});
json_wire!(Failure<'a>{update:u64 => "update",phase:Id<'a> => "phase",counter:Counter => "counter"});
wire_record!(TargetOutcome {lock:FileRef<'a>,threshold_bits:u64,reached:bool,first_checkpoint:Option<u64>,time_basis:Id<'a>,time_ns:Option<u64>,censor_ns:u64});
json_wire!(TargetOutcome<'a>{lock:FileRef<'a> => "lock",threshold_bits:u64 => "threshold_bits",reached:bool => "reached",first_checkpoint:Option<u64> => "first_checkpoint",time_basis:Id<'a> => "time_basis",time_ns:Option<u64> => "time_ns",censor_ns:u64 => "censor_ns"});
json_wire!(FileRef<'a>{root:Tag<5> => "root",path:Str<'a,4096> => "path",length:u64 => "length",sha256:H<'a> => "sha256"});
impl JsonWire<'_> for Clock {
    fn encode_json(r: &RawValue, p: &str, e: &mut Encoder) -> Result<()> {
        let mut f = Fields::new(
            r,
            &[
                "wall_source",
                "unix_ns",
                "monotonic_source",
                "elapsed_since_operation_start_ns",
            ],
            p,
        )?;
        same(f.field::<u64>("wall_source", p)? == 0, "clock.wall_source")?;
        e.byte(0)?;
        f.field::<u64>("unix_ns", p)?.write(e)?;
        same(
            f.field::<u64>("monotonic_source", p)? == 0,
            "clock.monotonic_source",
        )?;
        e.byte(0)?;
        f.field::<u64>("elapsed_since_operation_start_ns", p)?
            .write(e)
    }
    fn json(&self) -> Value {
        json!({"wall_source":0,"unix_ns":self.unix_ns,"monotonic_source":0,"elapsed_since_operation_start_ns":self.elapsed_since_operation_start_ns})
    }
    fn write_json(&self, out: &mut Vec<u8>) -> Result<()> {
        write_numbers(
            out,
            &[
                "wall_source",
                "unix_ns",
                "monotonic_source",
                "elapsed_since_operation_start_ns",
            ],
            &[0, self.unix_ns, 0, self.elapsed_since_operation_start_ns],
        )
    }
}
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct CpuResult<'a> {
    pub(in crate::study_reference::document::evidence) scope: Tag<1>,
    pub(in crate::study_reference::document::evidence) case: Bytes<'a>,
    pub(in crate::study_reference::document::evidence) execution: Bytes<'a>,
    pub(in crate::study_reference::document::evidence) attempt: Id<'a>,
    pub(in crate::study_reference::document::evidence) admission_or_preflight: FileRef<'a>,
    pub(in crate::study_reference::document::evidence) start: FileRef<'a>,
    pub(in crate::study_reference::document::evidence) evidence: [FileRef<'a>; 5],
    pub(in crate::study_reference::document::evidence) policy: Id<'a>,
    pub(in crate::study_reference::document::evidence) source_manifest: H<'a>,
    pub(in crate::study_reference::document::evidence) source_revision: Git<'a>,
    pub(in crate::study_reference::document::evidence) executable: H<'a>,
    pub(in crate::study_reference::document::evidence) contract: H<'a>,
    pub(in crate::study_reference::document::evidence) receipt: H<'a>,
    pub(in crate::study_reference::document::evidence) materialized: H<'a>,
    pub(in crate::study_reference::document::evidence) consumed_plan: H<'a>,
    pub(in crate::study_reference::document::evidence) status: Tag<2>,
    pub(in crate::study_reference::document::evidence) completed_updates: u64,
    pub(in crate::study_reference::document::evidence) processed_samples: u64,
    pub(in crate::study_reference::document::evidence) poisoned: bool,
    pub(in crate::study_reference::document::evidence) failed_prefix: Option<Failure<'a>>,
    pub(in crate::study_reference::document::evidence) checkpoints: Vec<Endpoint>,
    pub(in crate::study_reference::document::evidence) final_endpoint: Option<Endpoint>,
    pub(in crate::study_reference::document::evidence) model_digest: Option<H<'a>>,
    pub(in crate::study_reference::document::evidence) target: Option<TargetOutcome<'a>>,
    pub(in crate::study_reference::document::evidence) ended: Clock,
}
macro_rules! result_fields {($m:ident) => {$m!(scope:Tag<1> => "scope",case:Bytes<'a> => "case",execution:Bytes<'a> => "execution",attempt:Id<'a> => "attempt",admission_or_preflight:FileRef<'a> => "admission_or_preflight",start:FileRef<'a> => "start",evidence:[FileRef<'a>;5] => "evidence",policy:Id<'a> => "policy",source_manifest:H<'a> => "source_manifest",source_revision:Git<'a> => "source_revision",executable:H<'a> => "executable",contract:H<'a> => "contract",receipt:H<'a> => "receipt",materialized:H<'a> => "materialized",consumed_plan:H<'a> => "consumed_plan",status:Tag<2> => "status",completed_updates:u64 => "completed_updates",processed_samples:u64 => "processed_samples",poisoned:bool => "poisoned",failed_prefix:Option<Failure<'a>> => "failed_prefix",checkpoints:Vec<Endpoint> => "checkpoints",final_endpoint:Option<Endpoint> => "final",model_digest:Option<H<'a>> => "model_digest",target:Option<TargetOutcome<'a>> => "target",ended:Clock => "ended");};}
macro_rules! result_read {
    ($c:ident,checkpoints,$t:ty) => {
        wire::read_vec($c, 400)
    };
    ($c:ident,$f:ident,$t:ty) => {
        <$t as Wire<'a>>::read($c)
    };
}
macro_rules! cpu_result_wire {($($f:ident:$t:ty => $k:literal),*) => {
impl<'a> Wire<'a> for CpuResult<'a>{const MIN:usize=0$(+<$t as Wire<'a>>::MIN)*;fn read(c:&mut Cursor<'a>)->Result<Self>{Ok(Self{$($f:result_read!(c,$f,$t)?),*})}fn write(&self,e:&mut Encoder)->Result<()>{$ (self.$f.write(e)?;)*Ok(())}}
json_wire!(CpuResult<'a>{$($f:$t => $k),*});
};}
result_fields!(cpu_result_wire);
fn same_endpoint(a: &Endpoint, b: &Endpoint) -> Result<()> {
    let mut x = Encoder::new(DocumentLayout::Start);
    a.write(&mut x)?;
    let mut y = Encoder::new(DocumentLayout::Start);
    b.write(&mut y)?;
    same(x.bytes == y.bytes, "result.final.last")
}
impl CpuResult<'_> {
    pub(in crate::study_reference::document::evidence) fn validate(
        &self,
        _mode: Mode,
    ) -> Result<()> {
        same(self.policy.0 == POLICY, "result.policy")?;
        self.start.validate()?;
        self.admission_or_preflight.validate()?;
        for f in &self.evidence {
            f.validate()?;
        }
        let limit = if self.scope.0 == 0 { 2400 } else { 2 };
        same(
            self.completed_updates <= limit
                && self.processed_samples
                    == self
                        .completed_updates
                        .checked_mul(20)
                        .ok_or_else(|| err(DocumentErrorCode::TooLarge, "result.samples"))?,
            "result.counts",
        )?;
        same(self.checkpoints.len() <= 400, "result.checkpoints.count")?;
        let mut previous = None;
        for endpoint in &self.checkpoints {
            same(
                endpoint.updates > 0
                    && endpoint.updates <= self.completed_updates
                    && endpoint.samples == endpoint.updates * 20
                    && endpoint.test_count == 30
                    && endpoint.he_correct <= 30
                    && endpoint.fp32_correct <= 30,
                "result.endpoint.counts",
            )?;
            same(
                endpoint.crossed.windows(2).all(|w| w[0] < w[1]),
                "result.endpoint.crossed",
            )?;
            for bits in [
                endpoint.weight_max_abs_diff_bits,
                endpoint.prediction_max_abs_diff_bits,
            ] {
                let n = f64::from_bits(bits);
                same(
                    n.is_finite() && !n.is_sign_negative(),
                    "result.endpoint.finite",
                )?;
            }
            endpoint.counter.values[0]
                .checked_add(endpoint.counter.values[1])
                .and_then(|v| v.checked_add(endpoint.counter.values[5]))
                .ok_or_else(|| err(DocumentErrorCode::TooLarge, "result.counter.key_switches"))?;
            let t = endpoint.times.values;
            same(
                t[2] <= t[1] && t[3] <= t[1] && t[0] <= t[7] && t[1] <= t[7],
                "result.endpoint.times",
            )?;
            if let Some(p) = previous {
                let p: &Endpoint = p;
                same(
                    p.updates < endpoint.updates
                        && p.times.values.iter().zip(t).all(|(a, b)| *a <= b)
                        && p.counter
                            .values
                            .iter()
                            .zip(endpoint.counter.values)
                            .all(|(a, b)| *a <= b),
                    "result.endpoint.monotonic",
                )?;
            }
            previous = Some(endpoint);
        }
        if self.status.0 == 0 {
            same(
                self.completed_updates == limit
                    && !self.poisoned
                    && self.failed_prefix.is_none()
                    && self.model_digest.is_some(),
                "result.complete",
            )?;
            let last = self
                .checkpoints
                .last()
                .ok_or_else(|| err(DocumentErrorCode::CrossLinkMismatch, "result.last"))?;
            let final_endpoint = self
                .final_endpoint
                .as_ref()
                .ok_or_else(|| err(DocumentErrorCode::CrossLinkMismatch, "result.final"))?;
            same_endpoint(last, final_endpoint)?;
            same(last.updates == limit, "result.final.count")?;
            same(
                if self.scope.0 == 1 {
                    self.checkpoints.len() == 2
                        && self.checkpoints[0].updates == 1
                        && self.target.is_none()
                } else {
                    self.checkpoints.len() == 400 && self.target.is_some()
                },
                "result.scope.endpoints",
            )?;
        } else {
            same(
                self.final_endpoint.is_none() && self.model_digest.is_none(),
                "result.failed.no_final",
            )?;
        }
        if self.scope.0 == 1 {
            same(self.target.is_none(), "result.quick.target")?;
        }
        if let Some(f) = &self.failed_prefix {
            same(
                f.update >= self.completed_updates && f.update < limit,
                "result.failure.prefix",
            )?;
        }
        if let Some(target) = &self.target {
            target.lock.validate()?;
            let threshold = f64::from_bits(target.threshold_bits);
            same(
                threshold.is_finite()
                    && (0.0..=1.0).contains(&threshold)
                    && !threshold.is_sign_negative()
                    && target.time_basis.0 == "cumulative_update_ns",
                "result.target.threshold",
            )?;
            let first = self
                .checkpoints
                .iter()
                .find(|e| e.he_correct as f64 / e.test_count as f64 >= threshold);
            same(
                target.reached == first.is_some()
                    && target.first_checkpoint == first.map(|e| e.updates)
                    && target.time_ns == first.map(|e| e.times.values[1])
                    && target.censor_ns == self.checkpoints.last().map_or(0, |e| e.times.values[1]),
                "result.target.coordinate",
            )?;
        }
        Ok(())
    }
}
pub(in crate::study_reference::document::evidence) fn export_id(
    scope: u8,
    start_whole_file_sha256: &str,
    mode: Mode,
) -> Result<String> {
    same(scope <= 1, "export.scope")?;
    let mut e = Encoder::new(DocumentLayout::Start);
    e.text(ENCODING_VERSION)?;
    e.text(&domain(mode, "result-export-id", 1))?;
    e.byte(scope)?;
    H(start_whole_file_sha256).write(&mut e)?;
    Ok(sha256_hex(&e.bytes))
}
pub(in crate::study_reference::document::evidence) fn result_json(
    record: &ObservedRecord<'_>,
    canonical_result: &FileRef<'_>,
) -> Result<Vec<u8>> {
    let RecordBody::Result(r) = &record.body else {
        return Err(err(DocumentErrorCode::CrossLinkMismatch, "result.kind"));
    };
    let b = record.canonical()?;
    canonical_result.check_bytes(&b)?;
    let mut out = Vec::new();
    append_json(&mut out, b"{")?;
    let mut first = true;
    json_pair(
        &mut out,
        "schema",
        &Str::<128>(&domain(record.mode, "cpu-result-json", 1)),
        &mut first,
    )?;
    json_pair(&mut out, "synthetic", &record.mode.synthetic(), &mut first)?;
    json_pair(
        &mut out,
        "export_id",
        &H(&export_id(r.scope.0, r.start.sha256.0, record.mode)?),
        &mut first,
    )?;
    json_pair(&mut out, "canonical_result", canonical_result, &mut first)?;
    json_pair(&mut out, "payload", r, &mut first)?;
    append_json(&mut out, b"}")?;
    Ok(out)
}
pub(in crate::study_reference::document::evidence) fn check_result_json(
    raw: &[u8],
    binary: &[u8],
    mode: Mode,
) -> Result<()> {
    preflight_layout(raw, DocumentLayout::Start)?;
    let r: &RawValue = serde_json::from_slice(raw).map_err(|e| json_err(e, "result.json"))?;
    let mut f = Fields::new(
        r,
        &[
            "schema",
            "synthetic",
            "export_id",
            "canonical_result",
            "payload",
        ],
        "result",
    )?;
    same(
        f.field::<Text<128>>("schema", "result")?.0 == domain(mode, "cpu-result-json", 1)
            && f.field::<bool>("synthetic", "result")? == mode.synthetic(),
        "result.json.schema",
    )?;
    let mut link = Encoder::new(DocumentLayout::Start);
    FileRef::encode_json(
        f.get("canonical_result", "result")?,
        "canonical_result",
        &mut link,
    )?;
    let mut c = Cursor::new(&link.bytes);
    FileRef::read(&mut c)?.check_bytes(binary)?;
    let mut e = encoder("cpu-result", 1, mode)?;
    CpuResult::encode_json(f.get("payload", "result")?, "payload", &mut e)?;
    same(e.bytes == binary, "result.json.canonical_equality")?;
    let record = ObservedRecord::decode(RecordKind::Result, binary, mode)?;
    let RecordBody::Result(r) = record.body else {
        unreachable!()
    };
    same(
        f.field::<Hash>("export_id", "result")?.0 == export_id(r.scope.0, r.start.sha256.0, mode)?,
        "result.json.export_id",
    )
}
pub(in crate::study_reference::document::evidence) fn check_three_outputs(
    binary: &[u8],
    attempt_json: &[u8],
    export_json: &[u8],
    mode: Mode,
) -> Result<()> {
    same(attempt_json == export_json, "result.export.whole_file")?;
    check_result_json(attempt_json, binary, mode)
}

fn append_json(out: &mut Vec<u8>, b: &[u8]) -> Result<()> {
    bounded_add(out.len(), b.len(), A_MAX, "result.json")?;
    out.try_reserve(b.len())
        .map_err(|_| err(DocumentErrorCode::TooLarge, "result.json"))?;
    out.extend_from_slice(b);
    Ok(())
}
struct LimitedJson<'a>(&'a mut Vec<u8>);
impl std::io::Write for LimitedJson<'_> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        append_json(self.0, b).map_err(|_| std::io::Error::other("result JSON cap"))?;
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn write_value(out: &mut Vec<u8>, v: &Value) -> Result<()> {
    serde_json::to_writer(LimitedJson(out), v)
        .map_err(|_| err(DocumentErrorCode::TooLarge, "result.json"))
}
fn json_pair<'a, T: JsonWire<'a>>(
    out: &mut Vec<u8>,
    key: &str,
    value: &T,
    first: &mut bool,
) -> Result<()> {
    if !*first {
        append_json(out, b",")?;
    }
    *first = false;
    write_value(out, &json!(key))?;
    append_json(out, b":")?;
    value.write_json(out)
}
fn write_array<'a, 'b, T: JsonWire<'a> + 'b>(
    out: &mut Vec<u8>,
    values: impl Iterator<Item = &'b T>,
) -> Result<()> {
    append_json(out, b"[")?;
    let mut first = true;
    for v in values {
        if !first {
            append_json(out, b",")?;
        }
        first = false;
        v.write_json(out)?;
    }
    append_json(out, b"]")
}
fn write_numbers(out: &mut Vec<u8>, keys: &[&str], values: &[u64]) -> Result<()> {
    append_json(out, b"{")?;
    let mut first = true;
    for (k, v) in keys.iter().zip(values) {
        json_pair(out, k, v, &mut first)?;
    }
    append_json(out, b"}")
}
