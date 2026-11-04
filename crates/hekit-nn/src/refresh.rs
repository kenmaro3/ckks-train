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
//!
//! # Replicated blocks ([`Refresher::bind_layout`])
//!
//! When `d * batch < slots` every packed vector is the block of `d * batch` slots repeated
//! `slots / (d * batch)` times. Each copy carries its own noise, every training step computes
//! every copy separately, and rotations that wrap across a block boundary mix neighbouring
//! copies. Over a long run the copies drift apart and the drift grows **exponentially** (Task
//! 31, Iris 4-10-3 on seal_2020 with 32 blocks: the block spread grew from 2^-19 to 2^-12 in 13
//! epochs, the weights left the plaintext trajectory, and the run overflowed at epoch 30). A
//! refresher bound to the layout re-lays the canonical form: [`ClientRefresher`] replaces every
//! block by the mean of the blocks before re-encoding. [`crate::trainer::Trainer::new`] binds
//! its refresher to its layout, so this is the default in training; a bare
//! [`ClientRefresher::new`] used outside a trainer re-encodes the slots as decoded until a
//! layout is bound ([`ClientRefresher::with_layout`]). That is exact for every packed vector (all are
//! block-periodic) and also averages the noise of the copies. A single-block layout
//! (`d * batch == slots`, e.g. MNIST in Task 32) has nothing to average.

use std::sync::Arc;

use hekit_ckks::{Ciphertext, Decryptor, Encoder, Encryptor};
use hekit_core::error::{HekitError, Result};
use hekit_core::sampler::Rng;

use crate::layout::Layout;

/// Brings a ciphertext back to the top level (`max_level`) at its own exact scale, with the
/// same message in every slot (up to the refresh's error).
///
/// # Layout binding
///
/// [`Refresher::bind_layout`] is required (no default): a refresher must decide what it does
/// with a replicated layout (`layout.block() < slots`), whose block copies drift apart
/// exponentially over a long run unless every refresh re-lays them (module docs, "Replicated
/// blocks"). [`crate::trainer::Trainer::new`] binds its refresher to its layout.
/// [`ClientRefresher`] averages the blocks on the client. A future bootstrapping refresher
/// (M3) has no client to re-lay: it must either average the blocks on the server
/// (`log2(slots / block)` rotations and additions, the constant `block / slots` folded into a
/// plaintext) or return `InvalidParams` for a replicated layout.
pub trait Refresher {
    /// A fresh ciphertext on the top level with exactly `ct.scale()` and `ct`'s message.
    ///
    /// Errors: implementation specific (e.g. `InvalidParams` for a ciphertext of other
    /// parameters).
    fn refresh(&mut self, ct: &Ciphertext) -> Result<Ciphertext>;

    /// Binds the refresher to the slot layout of the values it will refresh: from now on
    /// every refresh returns the canonical (block-periodic) form of `layout`, its replicated
    /// blocks re-laid. Called by [`crate::trainer::Trainer::new`].
    ///
    /// Errors: `InvalidParams` if `layout` does not fit the refresher's parameters, or if the
    /// refresher cannot re-lay a replicated layout (see the trait docs).
    fn bind_layout(&mut self, layout: &Layout) -> Result<()>;
}

/// The interactive refresh: decrypt → decode → re-encode → encrypt on `max_level` (see the
/// [module docs](self)). Holds the secret key: it stands for the client.
pub struct ClientRefresher {
    dec: Decryptor,
    enc: Encryptor,
    encoder: Encoder,
    rng: Rng,
    refreshed: u64,
    block: Option<usize>,
}

impl std::fmt::Debug for ClientRefresher {
    /// A summary without the keys.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientRefresher")
            .field("n", &self.encoder.params().n())
            .field("max_level", &self.encoder.params().max_level())
            .field("refreshed", &self.refreshed)
            .field("block", &self.block)
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
            block: None,
        }
    }

    /// [`Refresher::bind_layout`] as a builder: re-lays the canonical form of `layout` on
    /// every refresh, each of the `slots / layout.block()` replicated blocks replaced by the
    /// mean of the blocks (see the [module docs](self)). A [`crate::trainer::Trainer`] binds
    /// its layout itself; use this for refreshes outside a trainer.
    ///
    /// Errors: `InvalidParams` if `layout` does not have the parameters' slot count.
    pub fn with_layout(mut self, layout: Layout) -> Result<ClientRefresher> {
        self.bind_layout(&layout)?;
        Ok(self)
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
    /// kept; only the encoding scale changes). With a bound layout
    /// ([`Refresher::bind_layout`], [`ClientRefresher::with_layout`]) the replicated blocks
    /// are averaged here too; without one, all slots are re-encoded as decoded.
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
        let mut values = self.encoder.decode(&self.dec.decrypt(ct));
        if let Some(bk) = self.block {
            average_blocks(&mut values, bk);
        }
        let pt = self.encoder.encode(&values, scale, params.max_level())?;
        let fresh = self.enc.encrypt(&pt, &mut self.rng);
        self.refreshed += 1;
        Ok(fresh)
    }
}

/// Replaces every block of `bk` values by the mean of the blocks.
fn average_blocks(values: &mut [f64], bk: usize) {
    let copies = values.len() / bk;
    if copies <= 1 {
        return;
    }
    let mut mean = vec![0.0; bk];
    for chunk in values.chunks_exact(bk) {
        mean.iter_mut().zip(chunk).for_each(|(m, v)| *m += v);
    }
    mean.iter_mut().for_each(|m| *m /= copies as f64);
    for chunk in values.chunks_exact_mut(bk) {
        chunk.copy_from_slice(&mean);
    }
}

impl Refresher for ClientRefresher {
    /// Decrypts, decodes all `N/2` slots (averaging the replicated blocks if
    /// [`ClientRefresher::with_layout`] was set), re-encodes them at exactly `ct.scale()` on
    /// the top level and encrypts (public-key encryption with the client's ChaCha20 stream).
    ///
    /// Errors as [`ClientRefresher::refresh_to_scale`].
    fn refresh(&mut self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.refresh_to_scale(ct, ct.scale())
    }

    /// Averages the replicated blocks of `layout` on every later refresh (a no-op for a
    /// single-block layout). Rebinding replaces the previous layout.
    ///
    /// Errors: `InvalidParams` if `layout` does not have the parameters' slot count (the
    /// binding is then unchanged).
    fn bind_layout(&mut self, layout: &Layout) -> Result<()> {
        if layout.slots != self.encoder.params().slots() {
            return Err(HekitError::InvalidParams(format!(
                "layout of {} slots for parameters with {} slots",
                layout.slots,
                self.encoder.params().slots()
            )));
        }
        self.block = Some(layout.block());
        Ok(())
    }
}
