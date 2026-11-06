//! Error type shared by all hekit crates.

use std::fmt;

/// Errors returned by hekit operations. Library code returns these instead of panicking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HekitError {
    /// Invalid parameters (modulus out of range, bad sizes, not enough primes, ...).
    InvalidParams(String),
    /// Two ciphertexts/plaintexts have different scales.
    ScaleMismatch,
    /// Two operands are at different levels.
    LevelMismatch,
    /// The ciphertext is at level 0; no further rescale or multiplication is possible.
    NoLevelsLeft,
    /// Decoding failed.
    Decode(String),
    /// A GPU / device backend failed (no driver or device, bad ordinal, out of device memory,
    /// kernel compilation or launch failure, ...). Backends return this instead of panicking.
    Device(String),
}

impl fmt::Display for HekitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HekitError::InvalidParams(m) => write!(f, "invalid parameters: {m}"),
            HekitError::ScaleMismatch => write!(f, "scale mismatch"),
            HekitError::LevelMismatch => write!(f, "level mismatch"),
            HekitError::NoLevelsLeft => write!(f, "no levels left"),
            HekitError::Decode(m) => write!(f, "decode error: {m}"),
            HekitError::Device(m) => write!(f, "device error: {m}"),
        }
    }
}

impl std::error::Error for HekitError {}

/// Result alias using [`HekitError`].
pub type Result<T> = std::result::Result<T, HekitError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_error_display() {
        let e = HekitError::Device("out of memory".into());
        assert_eq!(e.to_string(), "device error: out of memory");
    }
}
