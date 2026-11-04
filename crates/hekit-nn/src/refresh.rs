//! Restoring the levels of the trained values between training steps (Task 30).
//!
//! A training step consumes the whole modulus chain (`docs/layout.md` §4: on seal_2020 the
//! Task 30 step ends with `W1'` on level 0). Before the next step every weight, bias and
//! optimizer state must be back on a high level ([`Refresher::output_level`]: the top level
//! for the client, the bootstrap output level for bootstrapping). [`Refresher`] is that
//! operation; the [`crate::trainer::Trainer`] calls it on every ciphertext the next step
//! consumes.
//!
//! [`ClientRefresher`] is the **interactive mode** of M2 (spec §4.3.1, the development stage;
//! not used for the paper's claims): the server sends the ciphertexts to the key owner, who
//! decrypts, decodes, re-encodes and re-encrypts them on the top level. M3 adds
//! [`BootstrapRefresher`], the non-interactive refresh by CKKS bootstrapping behind the same
//! trait (Task 38).
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

use hekit_ckks::bootstrap::Bootstrapper;
use hekit_ckks::{Ciphertext, Decryptor, Encoder, Encryptor, Evaluator};
use hekit_core::error::{HekitError, Result};
use hekit_core::sampler::Rng;

use crate::layout::Layout;

/// Brings a ciphertext back to the refresher's output level at its own exact scale, with the
/// same message in every slot (up to the refresh's error). The output level is the top level
/// (`max_level`) for [`ClientRefresher`] and the level after bootstrapping
/// ([`Bootstrapper::output_level`], 16 on n16_boot) for [`BootstrapRefresher`].
///
/// # Layout binding
///
/// [`Refresher::bind_layout`] is required (no default): a refresher must decide what it does
/// with a replicated layout (`layout.block() < slots`), whose block copies drift apart
/// exponentially over a long run unless every refresh re-lays them (module docs, "Replicated
/// blocks"). [`crate::trainer::Trainer::new`] binds its refresher to its layout.
/// [`ClientRefresher`] averages the blocks on the client. [`BootstrapRefresher`] has no
/// client to re-lay and returns `InvalidParams` for a replicated layout (ruling R-M3-3).
pub trait Refresher {
    /// A fresh ciphertext on the refresher's output level with exactly `ct.scale()` and
    /// `ct`'s message.
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

    /// The level of every [`Refresher::refresh`] output: `max_level` for
    /// [`ClientRefresher`], [`Bootstrapper::output_level`] (`levels_after_boot`, 16 on
    /// n16_boot) for [`BootstrapRefresher`]. Code that consumes refreshed values must start
    /// its level chain here, not at `max_level`.
    fn output_level(&self) -> usize;
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

    /// The top level, `max_level`: the client encrypts afresh.
    fn output_level(&self) -> usize {
        self.encoder.params().max_level()
    }
}

/// The non-interactive refresh (M3, Task 38): every [`Refresher::refresh`] drops the
/// ciphertext to level 0 (free) and bootstraps it ([`Bootstrapper::bootstrap`]) to
/// [`Bootstrapper::output_level`]. The evaluator supplies the relinearization key and the
/// counter: each refresh adds one bootstrap and all its key switches to it.
///
/// # Scale
///
/// The bootstrap takes and returns exactly `Delta` (the preset scale). A ciphertext at another
/// scale `s` (a bias on its exact product scale, see the [module docs](self)) is relabelled to
/// `Delta` before the bootstrap and back to `s` after it. Relabelling is exact: it multiplies
/// the message by `s / Delta` during the bootstrap and divides it back, so the result decrypts
/// to `ct`'s message at exactly `s`, with the bootstrap error scaled by `Delta / s`. The
/// bootstrap's precondition `|m| <= 1` then applies to `(s / Delta) m`; scales outside
/// `[Delta / 2, 2 Delta]` are refused (`ScaleMismatch`).
///
/// # Replicated layouts are refused (ruling R-M3-3)
///
/// Bootstrapping refreshes every slot independently: it cannot replace the drifting block
/// copies of a replicated layout (`layout.block() < slots`) by their mean, as
/// [`ClientRefresher`] does (module docs, "Replicated blocks"). So
/// [`Refresher::bind_layout`] returns `InvalidParams` for a layout with more than one block
/// copy, and a [`crate::trainer::Trainer`] cannot be built with it on such a layout. A
/// single-block layout (`d * batch == slots`, e.g. MNIST) is accepted. Replicated layouts
/// take the compact path of Tasks 40/41 instead: pack one lane per diagonal, bootstrap the
/// compact ciphertext, and unpack (which re-replicates the blocks by construction).
#[derive(Debug)]
pub struct BootstrapRefresher<'a> {
    bs: &'a Bootstrapper,
    ev: &'a Evaluator,
}

impl<'a> BootstrapRefresher<'a> {
    /// A refresher bootstrapping with `bs` on `ev` (which must hold the relinearization key
    /// of `bs`'s parameters; checked at each refresh).
    pub fn new(bs: &'a Bootstrapper, ev: &'a Evaluator) -> BootstrapRefresher<'a> {
        BootstrapRefresher { bs, ev }
    }
}

impl Refresher for BootstrapRefresher<'_> {
    /// Drops `ct` to level 0, bootstraps it (relabelled to `Delta` and back if its scale
    /// differs, see the type docs) and returns it at [`Bootstrapper::output_level`] with
    /// exactly `ct.scale()`.
    ///
    /// Errors: `ScaleMismatch` for a scale outside `[Delta / 2, 2 Delta]`; the errors of
    /// [`Evaluator::mod_switch_to_level`] and [`Bootstrapper::bootstrap`] (`InvalidParams`
    /// for a ciphertext or evaluator of other parameters, a 3-component ciphertext, or no
    /// relinearization key).
    fn refresh(&mut self, ct: &Ciphertext) -> Result<Ciphertext> {
        let delta = self.ev.params().scale();
        let s = ct.scale();
        if !(s >= delta / 2.0 && s <= 2.0 * delta) {
            return Err(HekitError::ScaleMismatch);
        }
        let low = self.ev.mod_switch_to_level(ct, 0)?;
        let low = Ciphertext::new(low.into_polys(), delta)?;
        let out = self.bs.bootstrap(self.ev, &low)?;
        Ciphertext::new(out.into_polys(), s)
    }

    /// Accepts a single-block layout of the parameters' slot count.
    ///
    /// Errors (`InvalidParams`): `layout` has another slot count, or it is replicated
    /// (`layout.block() < slots`): bootstrapping cannot re-lay the block copies (type docs,
    /// ruling R-M3-3).
    fn bind_layout(&mut self, layout: &Layout) -> Result<()> {
        let slots = self.ev.params().slots();
        if layout.slots != slots {
            return Err(HekitError::InvalidParams(format!(
                "layout of {} slots for parameters with {slots} slots",
                layout.slots
            )));
        }
        if layout.block() < slots {
            return Err(HekitError::InvalidParams(format!(
                "BootstrapRefresher cannot re-lay a replicated layout ({} copies of a \
                 {}-slot block); use the compact pack / bootstrap / unpack path (Tasks 40/41)",
                slots / layout.block(),
                layout.block()
            )));
        }
        Ok(())
    }

    /// [`Bootstrapper::output_level`] (`levels_after_boot`, 16 on n16_boot).
    fn output_level(&self) -> usize {
        self.bs.output_level()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use hekit_ckks::bootstrap::BootParams;
    use hekit_ckks::precision::precision_bits;
    use hekit_ckks::{CkksParams, KeyGenerator};

    use super::*;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    /// The n16_boot bands in miniature at N = 2^10 (insecure, tests only).
    fn mini() -> CkksParams {
        let mut q = vec![54, 42, 42];
        q.extend([34; 4]);
        q.extend([58; 7]);
        q.extend([56; 4]);
        CkksParams::new(10, &q, &[60, 60], 42, 9).unwrap()
    }

    /// R-M3-3, the scale contract and the counter, on the miniature.
    #[test]
    fn bootstrap_refresher_levels_scales_and_layouts() {
        let params = Arc::new(mini());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(1)).unwrap();
        let encoder = Encoder::new(Arc::clone(&params));
        let bp = BootParams {
            log_slots: 9,
            ..BootParams::default_n16()
        };
        let bs = Bootstrapper::new(Arc::clone(&params), bp, &encoder, &mut kg).unwrap();
        let ev = Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), None);
        let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
        let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
        let mut r = rng(2);
        let slots = params.slots();
        let m: Vec<f64> = (0..slots)
            .map(|_| 2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0)
            .collect();
        let mut rf = BootstrapRefresher::new(&bs, &ev);

        // Layouts: one block is fine, copies are refused, other slot counts too.
        assert!(rf.bind_layout(&Layout::new(slots, 32, 16).unwrap()).is_ok());
        let inv = |e: Result<()>| matches!(e, Err(HekitError::InvalidParams(_)));
        assert!(inv(rf.bind_layout(&Layout::new(slots, 16, 16).unwrap())));
        assert!(inv(rf.bind_layout(&Layout::new(2 * slots, 32, 32).unwrap())));

        // From any level, at Delta and at a nearby product scale.
        let delta = params.scale();
        for (level, scale) in [(5, delta), (0, delta * (1.0 + 2f64.powi(-11)))] {
            let pt = encoder.encode(&m, scale, level).unwrap();
            let ct = enc.encrypt(&pt, &mut r);
            let before = ev.counter().bootstraps;
            let out = rf.refresh(&ct).unwrap();
            assert_eq!(rf.output_level(), bs.output_level());
            assert_eq!((out.level(), out.scale()), (rf.output_level(), scale));
            assert_eq!(ev.counter().bootstraps, before + 1);
            let bits = precision_bits(&m, &encoder.decode(&dec.decrypt(&out)));
            assert!(bits >= 16.0, "level {level}: {bits}");
        }
        // A scale far from Delta is refused before any work.
        let pt = encoder.encode(&m, 4.0 * delta, 0).unwrap();
        let before = ev.counter();
        assert_eq!(
            rf.refresh(&enc.encrypt(&pt, &mut r)).unwrap_err(),
            HekitError::ScaleMismatch
        );
        assert_eq!(ev.counter(), before);
    }
}
