//! Restoring the levels of the trained values between training steps (Task 30).
//!
//! A training step consumes the whole modulus chain (`docs/layout.md` §4: on seal_2020 the
//! Task 30 step ends with `W1'` on level 0). Before the next step every weight, bias and
//! optimizer state must be back on the top level. [`Refresher`] is that operation; the
//! [`crate::trainer::Trainer`] calls it on every ciphertext the next step consumes.
//!
//! [`ClientRefresher`] is the **interactive mode** of M2 (spec §4.3.1, the development stage;
//! not used for the paper's claims): the server sends the ciphertexts to the key owner, who
//! decrypts, decodes, re-encodes and re-encrypts them on the top level. M3 replaces it by a
//! bootstrapping refresher behind the same trait.
//!
//! # Scale contract
//!
//! [`Refresher::refresh`] keeps the **exact** scale of its input. That is the scale the
//! consumer needs: the training step keeps every trained value on a fixed scale (the weights
//! and the NAG velocity on `Delta`, each bias on the exact product scale its forward adds it
//! to, see [`crate::dense::Dense`]), so a refreshed value is bit-for-bit what the client
//! would have encrypted at the start. The slot layout is kept as is (all `N/2` slots are
//! decoded and re-encoded), so the replicated, pre-rotated weight diagonals need no
//! repacking.

use std::sync::Arc;

use hekit_ckks::{Ciphertext, Decryptor, Encoder, Encryptor};
use hekit_core::error::{HekitError, Result};
use hekit_core::sampler::Rng;

/// Brings a ciphertext back to the top level (`max_level`) at its own exact scale, with the
/// same message in every slot (up to the refresh's error).
pub trait Refresher {
    /// A fresh ciphertext on the top level with exactly `ct.scale()` and `ct`'s message.
    ///
    /// Errors: implementation specific (e.g. `InvalidParams` for a ciphertext of other
    /// parameters).
    fn refresh(&mut self, ct: &Ciphertext) -> Result<Ciphertext>;
}

/// The interactive refresh: decrypt → decode → re-encode → encrypt on `max_level` (see the
/// [module docs](self)). Holds the secret key: it stands for the client.
pub struct ClientRefresher {
    dec: Decryptor,
    enc: Encryptor,
    encoder: Encoder,
    rng: Rng,
    refreshed: u64,
}

impl std::fmt::Debug for ClientRefresher {
    /// A summary without the keys.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientRefresher")
            .field("n", &self.encoder.params().n())
            .field("max_level", &self.encoder.params().max_level())
            .field("refreshed", &self.refreshed)
            .finish()
    }
}

impl ClientRefresher {
    /// A client with the decryptor (secret key), an encryptor and an encoder of one parameter
    /// set (share one `Arc<CkksParams>`), and its own randomness (`rng`, ChaCha20).
    pub fn new(dec: Decryptor, enc: Encryptor, encoder: Encoder, rng: Rng) -> ClientRefresher {
        ClientRefresher {
            dec,
            enc,
            encoder,
            rng,
            refreshed: 0,
        }
    }

    /// The number of ciphertexts refreshed so far (the interactive mode's round-trip volume).
    pub fn refreshed(&self) -> u64 {
        self.refreshed
    }

    /// The encoder.
    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// [`Refresher::refresh`] at a chosen `scale` instead of `ct.scale()` (the message is
    /// kept; only the encoding scale changes).
    ///
    /// Errors: `InvalidParams` if `ct` is not over this client's parameters, and the
    /// [`Encoder::encode`] errors (a scale that is not positive and finite, or too large for
    /// the top level's modulus).
    pub fn refresh_to_scale(&mut self, ct: &Ciphertext, scale: f64) -> Result<Ciphertext> {
        let params = self.encoder.params();
        if !ct
            .polys()
            .iter()
            .all(|p| Arc::ptr_eq(p.basis(), params.basis_q()))
        {
            return Err(HekitError::InvalidParams(
                "refresh: ciphertext is not over the client's parameters".into(),
            ));
        }
        let values = self.encoder.decode(&self.dec.decrypt(ct));
        let pt = self.encoder.encode(&values, scale, params.max_level())?;
        let fresh = self.enc.encrypt(&pt, &mut self.rng);
        self.refreshed += 1;
        Ok(fresh)
    }
}

impl Refresher for ClientRefresher {
    /// Decrypts, decodes all `N/2` slots, re-encodes them at exactly `ct.scale()` on the top
    /// level and encrypts (public-key encryption with the client's ChaCha20 stream).
    ///
    /// Errors as [`ClientRefresher::refresh_to_scale`].
    fn refresh(&mut self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.refresh_to_scale(ct, ct.scale())
    }
}
