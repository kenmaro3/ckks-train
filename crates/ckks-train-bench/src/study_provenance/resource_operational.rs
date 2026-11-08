//! Monitored operational ownership; parsed records alone confer no authority.
#![forbid(unsafe_code)]
pub(crate) mod approval;
pub(crate) mod host;
pub(crate) mod package;
pub(crate) mod probe;
#[cfg(target_os = "linux")]
pub(crate) mod retained;
