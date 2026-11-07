//! Separately versioned inert observed-resource and CPU-result claims.
//! No decoder, fixed request or supplied-record check can construct live authority.
#![allow(dead_code)] // Consumed by later genuine owners, never converted to capabilities.
use super::*;
mod requested;
pub(super) use requested::*;
mod records;
pub(super) use records::*;
mod result;
pub(super) use result::*;
mod bundle;
pub(super) use bundle::*;
mod quick_v2;
pub(super) use quick_v2::*;
const RSS_LIMIT: u64 = 94_371_840;
const SAMPLE_INTERVAL: u64 = 10_000_000_000;
const TERM_GRACE: u64 = 5_000_000_000;
const SIZING: &str = "d7-cpu-source-sizing-v1";
const POLICY: &str = "d7-resource-observed-rss-v1";
fn domain(mode: Mode, name: &str, version: u8) -> String {
    format!(
        "hekit-{}-{name}-v{version}",
        if mode.synthetic() { "test" } else { "study" }
    )
}
fn cursor<'a>(b: &'a [u8], name: &str, version: u8, mode: Mode, cap: usize) -> Result<Cursor<'a>> {
    ensure(
        b.len() <= cap,
        DocumentErrorCode::TooLarge,
        "observed.length",
    )?;
    let mut c = Cursor::new(b);
    same(
        c.bounded_text(128)? == ENCODING_VERSION
            && c.bounded_text(128)?
                .strip_prefix(if mode.synthetic() {
                    "hekit-test-"
                } else {
                    "hekit-study-"
                })
                .and_then(|s| s.strip_suffix(if version == 1 { "-v1" } else { "-v2" }))
                == Some(name),
        "observed.domain",
    )?;
    same(
        c.tag(1)? == u8::from(mode.synthetic()),
        "observed.synthetic",
    )?;
    Ok(c)
}
fn encoder(name: &str, version: u8, mode: Mode) -> Result<Encoder> {
    let mut e = Encoder::new(DocumentLayout::Start);
    e.text(ENCODING_VERSION)?;
    e.text(&domain(mode, name, version))?;
    e.byte(u8::from(mode.synthetic()))?;
    Ok(e)
}
fn uuid(s: &str) -> Result<()> {
    same(
        s.len() == 36
            && s.bytes().enumerate().all(|(i, b)| {
                if [8, 13, 18, 23].contains(&i) {
                    b == b'-'
                } else {
                    b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
                }
            }),
        "observed.uuid",
    )
}
pub(super) mod policy;
#[cfg(test)]
pub(super) use policy::*;
pub(super) mod terminal;
#[cfg(test)]
pub(super) use terminal::*;
