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

use ckks_train_ckks::bootstrap::Bootstrapper;
use ckks_train_ckks::{
    BootstrapOps, Ciphertext, Decryptor, DiagonalEntry, Encoder, Encryptor, Evaluator, HeOps,
};
use ckks_train_core::error::{HekitError, Result};
use ckks_train_core::sampler::Rng;

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
///
/// # Backends (Task 43)
///
/// `E` is the evaluator whose ciphertexts are refreshed ([`Evaluator`] by default, so
/// `dyn Refresher` is the CPU refresher). [`ClientRefresher`] is host-side
/// (`E = Evaluator`); [`BootstrapRefresher`] works for any `E` with a [`BootstrapOps<E>`]
/// bootstrapper; [`HostRefresher`] runs a CPU refresher for the ciphertexts of another backend
/// (download, refresh on the host, upload).
pub trait Refresher<E: HeOps = Evaluator> {
    /// A fresh ciphertext on the refresher's output level with exactly `ct.scale()` and
    /// `ct`'s message.
    ///
    /// Errors: implementation specific (e.g. `InvalidParams` for a ciphertext of other
    /// parameters).
    fn refresh(&mut self, ct: &E::Ct) -> Result<E::Ct>;

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
    /// The level of every fresh ciphertext (`max_level` unless
    /// [`ClientRefresher::with_output_level`] lowered it).
    level: usize,
}

impl std::fmt::Debug for ClientRefresher {
    /// A summary without the keys.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientRefresher")
            .field("n", &self.encoder.params().n())
            .field("max_level", &self.encoder.params().max_level())
            .field("refreshed", &self.refreshed)
            .field("block", &self.block)
            .field("level", &self.level)
            .finish()
    }
}

impl ClientRefresher {
    /// A client with the decryptor (secret key), an encryptor and an encoder of one parameter
    /// set (share one `Arc<CkksParams>`), and its own randomness (`rng`, ChaCha20).
    pub fn new(dec: Decryptor, enc: Encryptor, encoder: Encoder, rng: Rng) -> ClientRefresher {
        let level = encoder.params().max_level();
        ClientRefresher {
            dec,
            enc,
            encoder,
            rng,
            refreshed: 0,
            block: None,
            level,
        }
    }

    /// Encrypts every refreshed ciphertext on `level` instead of the top level (Task 48): a
    /// level-capped interactive run, whose network starts on `level`
    /// ([`crate::trainer::Mlp::encrypt_at`], [`crate::trainer::Mlp::batch_spec_at`]) so that
    /// a device backend needs its keys only for the levels `<= level` (much smaller device
    /// keys). The client encrypts with the same public key and parameters; a ciphertext on a
    /// lower level is what the top-level one becomes after dropping limbs, so this changes no
    /// security property, only how many levels a step may use ([`Refresher::output_level`]).
    ///
    /// Errors: `InvalidParams` for `level > max_level`.
    pub fn with_output_level(mut self, level: usize) -> Result<ClientRefresher> {
        let max = self.encoder.params().max_level();
        if level > max {
            return Err(HekitError::InvalidParams(format!(
                "refresh level {level} above max_level {max}"
            )));
        }
        self.level = level;
        Ok(self)
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
        let pt = self.encoder.encode(&values, scale, self.level)?;
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

    /// The top level, `max_level`: the client encrypts afresh (or the level set with
    /// [`ClientRefresher::with_output_level`]).
    fn output_level(&self) -> usize {
        self.level
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
pub struct BootstrapRefresher<'a, E: HeOps = Evaluator, B: BootstrapOps<E> = Bootstrapper> {
    bs: &'a B,
    ev: &'a E,
}

impl<E: HeOps, B: BootstrapOps<E>> std::fmt::Debug for BootstrapRefresher<'_, E, B> {
    /// A summary (the output level), not the keys.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BootstrapRefresher")
            .field("output_level", &self.bs.output_level())
            .finish()
    }
}

impl<'a, E: HeOps, B: BootstrapOps<E>> BootstrapRefresher<'a, E, B> {
    /// A refresher bootstrapping with `bs` on `ev` (which must hold the relinearization key
    /// of `bs`'s parameters; checked at each refresh). `bs` is the CPU [`Bootstrapper`] for
    /// `E = Evaluator`, or any [`BootstrapOps<E>`].
    pub fn new(bs: &'a B, ev: &'a E) -> BootstrapRefresher<'a, E, B> {
        BootstrapRefresher { bs, ev }
    }
}

impl<E: HeOps, B: BootstrapOps<E>> Refresher<E> for BootstrapRefresher<'_, E, B> {
    /// Drops `ct` to level 0, bootstraps it (relabelled to `Delta` and back if its scale
    /// differs, see the type docs) and returns it at [`Bootstrapper::output_level`] with
    /// exactly `ct.scale()`.
    ///
    /// Errors: `ScaleMismatch` for a scale outside `[Delta / 2, 2 Delta]`; the errors of
    /// [`Evaluator::mod_switch_to_level`] and [`Bootstrapper::bootstrap`] (`InvalidParams`
    /// for a ciphertext or evaluator of other parameters, a 3-component ciphertext, or no
    /// relinearization key).
    fn refresh(&mut self, ct: &E::Ct) -> Result<E::Ct> {
        let delta = self.ev.params().scale();
        let s = ct.scale();
        if !(s >= delta / 2.0 && s <= 2.0 * delta) {
            return Err(HekitError::ScaleMismatch);
        }
        let low = self.ev.mod_switch_to_level(ct, 0)?;
        let low = self.ev.relabel(low, delta)?;
        let out = self.bs.bootstrap(self.ev, &low)?;
        self.ev.relabel(out, s)
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

/// A CPU refresher (`R: Refresher<Evaluator>`, e.g. [`ClientRefresher`] or a
/// [`BootstrapRefresher`] on the CPU) serving the ciphertexts of another backend `E`: every
/// refresh downloads the ciphertext ([`HeOps::download_ct`]), refreshes it on the host and
/// uploads the result ([`HeOps::upload_ct_owned`]). This is the interactive mode on a device
/// backend, and "level operations on the device, bootstrapping on the CPU" for the
/// bootstrapping refresher (research plan §5, M4 split).
///
/// # Counts
///
/// The host refresher's operations run on its own CPU evaluator. Built with
/// [`HostRefresher::counting`] (pass that host evaluator), every refresh merges the host
/// evaluator's counts of the refresh into `E`'s counter ([`HeOps::add_counts`]), so a
/// trainer's [`crate::trainer::EpochLog`] and `key_switches()` include them: the totals are
/// those of the native refresher on `E`. [`HostRefresher::new`] merges nothing (right for a
/// [`ClientRefresher`], which performs no evaluator operation). The host evaluator must not
/// be `E` itself (its counts would be merged twice; for a CPU `E`, use a [`HeOps::fork`]):
/// a refresh with `E` as its host is `InvalidParams` before any work, checked by address, as
/// in [`ckks_train_ckks::HostBootstrapper`]. It must not be used by another thread during a
/// refresh (the merged counts are the difference of its counter around the refresh). The
/// counts are merged only after the upload of the refreshed ciphertext succeeded: a failed
/// refresh or upload counts nothing on `E` (Task 48, Task 43 review N1/N2).
pub struct HostRefresher<'e, E: HeOps, R: Refresher<Evaluator>> {
    ev: &'e E,
    inner: R,
    host: Option<&'e Evaluator>,
}

impl<E: HeOps, R: Refresher<Evaluator> + std::fmt::Debug> std::fmt::Debug
    for HostRefresher<'_, E, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostRefresher")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<'e, E: HeOps, R: Refresher<Evaluator>> HostRefresher<'e, E, R> {
    /// Wraps the host refresher `inner` for the ciphertexts of `ev`, merging no counts
    /// (for a refresher that performs no evaluator operation, e.g. [`ClientRefresher`]).
    pub fn new(ev: &'e E, inner: R) -> HostRefresher<'e, E, R> {
        HostRefresher {
            ev,
            inner,
            host: None,
        }
    }

    /// Wraps the host refresher `inner`, which runs on the host evaluator `host` (e.g. a
    /// [`BootstrapRefresher`] built on `host`), and merges `host`'s counts of every refresh
    /// into `ev`'s counter (see the type docs).
    pub fn counting(ev: &'e E, inner: R, host: &'e Evaluator) -> HostRefresher<'e, E, R> {
        HostRefresher {
            ev,
            inner,
            host: Some(host),
        }
    }

    /// The host refresher.
    pub fn inner(&self) -> &R {
        &self.inner
    }

    /// Unwraps the host refresher.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<E: HeOps, R: Refresher<Evaluator>> Refresher<E> for HostRefresher<'_, E, R> {
    /// `upload(inner.refresh(download(ct)))`, merging the host evaluator's counts (if any)
    /// after the upload succeeded.
    ///
    /// Errors: `InvalidParams` when the host evaluator of [`HostRefresher::counting`] is `E`
    /// itself (its counts would be merged twice; nothing is done); those of the transfers and
    /// of the host refresher (nothing is counted on `E`).
    fn refresh(&mut self, ct: &E::Ct) -> Result<E::Ct> {
        let is_self = |h: &Evaluator| {
            std::ptr::eq(
                self.ev as *const E as *const u8,
                h as *const Evaluator as *const u8,
            )
        };
        if self.host.is_some_and(is_self) {
            return Err(HekitError::InvalidParams(
                "HostRefresher: the host evaluator must not be the backend evaluator itself \
                 (its counts would be merged twice); pass a fork"
                    .into(),
            ));
        }
        let host_ct = self.ev.download_ct(ct)?;
        let before = self.host.map(Evaluator::counter);
        let fresh = self.inner.refresh(&host_ct)?;
        let delta = match (self.host, before) {
            (Some(h), Some(b)) => Some(&h.counter() - &b),
            _ => None,
        };
        let up = self.ev.upload_ct_owned(fresh)?;
        if let Some(d) = delta {
            self.ev.add_counts(&d);
        }
        Ok(up)
    }

    /// The host refresher's [`Refresher::bind_layout`].
    fn bind_layout(&mut self, layout: &Layout) -> Result<()> {
        self.inner.bind_layout(layout)
    }

    /// The host refresher's [`Refresher::output_level`].
    fn output_level(&self) -> usize {
        self.inner.output_level()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ckks_train_ckks::bootstrap::BootParams;
    use ckks_train_ckks::precision::precision_bits;
    use ckks_train_ckks::{CkksParams, KeyGenerator};

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
        // Review m4: the window edges Delta / 2 and 2 Delta are accepted (the bootstrap then
        // sees m / 2 and 2 m), scales just outside and far outside are refused before any work.
        for edge in [delta / 2.0, 2.0 * delta] {
            let pt = encoder.encode(&m, edge, 0).unwrap();
            let out = rf.refresh(&enc.encrypt(&pt, &mut r)).unwrap();
            assert_eq!((out.level(), out.scale()), (bs.output_level(), edge));
            let bits = precision_bits(&m, &encoder.decode(&dec.decrypt(&out)));
            assert!(bits >= 16.0, "edge 2^{}: {bits}", edge.log2());
        }
        let before = ev.counter();
        let outside = [
            delta / 2.0 * (1.0 - 2f64.powi(-40)),
            2.0 * delta * (1.0 + 2f64.powi(-40)),
            4.0 * delta,
        ];
        for s in outside {
            let pt = encoder.encode(&m, s, 0).unwrap();
            assert_eq!(
                rf.refresh(&enc.encrypt(&pt, &mut r)).unwrap_err(),
                HekitError::ScaleMismatch,
                "scale {s:e}"
            );
        }
        assert_eq!(ev.counter(), before);
    }

    /// A CPU backend whose uploads fail on demand (Task 48, Task 43 review N2): every other
    /// operation is the wrapped [`Evaluator`]'s.
    struct FlakyUpload {
        inner: Evaluator,
        fail: std::cell::Cell<bool>,
    }

    impl HeOps for FlakyUpload {
        type Ct = Ciphertext;
        type Pt = ckks_train_ckks::Plaintext;
        fn params(&self) -> &Arc<CkksParams> {
            HeOps::params(&self.inner)
        }
        fn config(&self) -> &ckks_train_ckks::EvalConfig {
            HeOps::config(&self.inner)
        }
        fn counter(&self) -> ckks_train_ckks::OpCounter {
            HeOps::counter(&self.inner)
        }
        fn reset_counter(&self) {
            HeOps::reset_counter(&self.inner)
        }
        fn add_counts(&self, counts: &ckks_train_ckks::OpCounter) {
            self.inner.add_counts(counts)
        }
        fn fork(&self) -> FlakyUpload {
            FlakyUpload {
                inner: self.inner.fork(),
                fail: std::cell::Cell::new(self.fail.get()),
            }
        }
        fn has_relin_key(&self) -> bool {
            HeOps::has_relin_key(&self.inner)
        }
        fn has_galois_key(&self, step: i32) -> bool {
            HeOps::has_galois_key(&self.inner, step)
        }
        fn owns(&self, ct: &Ciphertext) -> bool {
            HeOps::owns(&self.inner, ct)
        }
        fn upload_ct(&self, ct: &Ciphertext) -> Result<Ciphertext> {
            if self.fail.get() {
                return Err(HekitError::Device("injected upload failure".into()));
            }
            self.inner.upload_ct(ct)
        }
        fn download_ct(&self, ct: &Ciphertext) -> Result<Ciphertext> {
            self.inner.download_ct(ct)
        }
        fn upload_pt(&self, pt: &ckks_train_ckks::Plaintext) -> Result<ckks_train_ckks::Plaintext> {
            self.inner.upload_pt(pt)
        }
        fn add(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
            HeOps::add(&self.inner, a, b)
        }
        fn sub(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
            HeOps::sub(&self.inner, a, b)
        }
        fn negate(&self, a: &Ciphertext) -> Result<Ciphertext> {
            HeOps::negate(&self.inner, a)
        }
        fn add_plain(&self, a: &Ciphertext, p: &ckks_train_ckks::Plaintext) -> Result<Ciphertext> {
            HeOps::add_plain(&self.inner, a, p)
        }
        fn mul_plain(&self, a: &Ciphertext, p: &ckks_train_ckks::Plaintext) -> Result<Ciphertext> {
            HeOps::mul_plain(&self.inner, a, p)
        }
        fn mul_scalar(&self, ct: &Ciphertext, v: f64, t: f64) -> Result<Ciphertext> {
            HeOps::mul_scalar(&self.inner, ct, v, t)
        }
        fn mul_plain_to_scale(
            &self,
            a: &Ciphertext,
            p: &ckks_train_ckks::Plaintext,
            t: f64,
        ) -> Result<Ciphertext> {
            HeOps::mul_plain_to_scale(&self.inner, a, p, t)
        }
        fn mul_no_relin(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
            HeOps::mul_no_relin(&self.inner, a, b)
        }
        fn relinearize(&self, ct: &Ciphertext) -> Result<Ciphertext> {
            HeOps::relinearize(&self.inner, ct)
        }
        fn mul(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
            HeOps::mul(&self.inner, a, b)
        }
        fn rescale(&self, ct: &Ciphertext) -> Result<Ciphertext> {
            HeOps::rescale(&self.inner, ct)
        }
        fn mod_switch_to_level(&self, ct: &Ciphertext, level: usize) -> Result<Ciphertext> {
            HeOps::mod_switch_to_level(&self.inner, ct, level)
        }
        fn relabel(&self, ct: Ciphertext, scale: f64) -> Result<Ciphertext> {
            HeOps::relabel(&self.inner, ct, scale)
        }
        fn rotate(&self, ct: &Ciphertext, steps: i32) -> Result<Ciphertext> {
            HeOps::rotate(&self.inner, ct, steps)
        }
        fn conjugate(&self, ct: &Ciphertext) -> Result<Ciphertext> {
            HeOps::conjugate(&self.inner, ct)
        }
        fn hoisted_rotate(&self, ct: &Ciphertext, steps: &[i32]) -> Result<Vec<Ciphertext>> {
            HeOps::hoisted_rotate(&self.inner, ct, steps)
        }
        fn matvec_plain(
            &self,
            diag: &ckks_train_ckks::Diagonals<ckks_train_ckks::Plaintext>,
            x: &Ciphertext,
        ) -> Result<Ciphertext> {
            HeOps::matvec_plain(&self.inner, diag, x)
        }
        fn matvec_enc(
            &self,
            diag: &ckks_train_ckks::Diagonals<Ciphertext>,
            x: &Ciphertext,
        ) -> Result<Ciphertext> {
            HeOps::matvec_enc(&self.inner, diag, x)
        }
    }

    /// The miniature bootstrapping fixture: params, keys, a bootstrapper, an evaluator with
    /// the relinearization key, and a level-3 ciphertext at `Delta`.
    struct Boot {
        params: Arc<CkksParams>,
        bs: Bootstrapper,
        ev: Evaluator,
        ct: Ciphertext,
    }

    fn boot_fixture(seed: u8) -> Boot {
        let params = Arc::new(mini());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let encoder = Encoder::new(Arc::clone(&params));
        let bp = BootParams {
            log_slots: 9,
            ..BootParams::default_n16()
        };
        let bs = Bootstrapper::new(Arc::clone(&params), bp, &encoder, &mut kg).unwrap();
        let ev = Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), None);
        let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
        let mut r = rng(seed.wrapping_add(1));
        let m: Vec<f64> = (0..params.slots())
            .map(|_| ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 0.5)
            .collect();
        let ct = enc.encrypt(&encoder.encode(&m, params.scale(), 3).unwrap(), &mut r);
        Boot { params, bs, ev, ct }
    }

    /// Task 48 (Task 43 review N1): `HostRefresher::counting` refuses the backend evaluator
    /// itself as its host (its counts would be merged twice): `InvalidParams` before any work,
    /// nothing counted. A fork as host works and merges exactly the host's delta.
    #[test]
    fn host_refresher_refuses_the_backend_as_host() {
        let f = boot_fixture(31);
        let ev = f.ev.fork();
        let mut same = HostRefresher::counting(&ev, BootstrapRefresher::new(&f.bs, &ev), &ev);
        let r = Refresher::<Evaluator>::refresh(&mut same, &f.ct);
        assert!(matches!(r, Err(HekitError::InvalidParams(_))), "{r:?}");
        assert_eq!(ev.counter(), ckks_train_ckks::OpCounter::default());

        let host = ev.fork();
        let mut ok = HostRefresher::counting(&ev, BootstrapRefresher::new(&f.bs, &host), &host);
        let out = Refresher::<Evaluator>::refresh(&mut ok, &f.ct).unwrap();
        assert_eq!(out.level(), f.bs.output_level());
        assert_eq!(ev.counter(), host.counter());
        assert_eq!(ev.counter().bootstraps, 1);
    }

    /// Task 48 (Task 43 review N2): the host's counts are merged only after the upload of the
    /// refreshed ciphertext succeeded; a failed upload counts nothing on the backend.
    #[test]
    fn host_refresher_merges_counts_only_after_the_upload() {
        let f = boot_fixture(41);
        let w = FlakyUpload {
            inner: f.ev.fork(),
            fail: std::cell::Cell::new(true),
        };
        let host = f.ev.fork();
        let mut rf = HostRefresher::counting(&w, BootstrapRefresher::new(&f.bs, &host), &host);
        let r = rf.refresh(&f.ct);
        assert!(matches!(r, Err(HekitError::Device(_))), "{r:?}");
        assert_eq!(w.counter(), ckks_train_ckks::OpCounter::default());
        assert_eq!(host.counter().bootstraps, 1, "the host did the work");
        w.fail.set(false);
        let before = host.counter();
        let out = rf.refresh(&f.ct).unwrap();
        assert_eq!(w.counter(), &host.counter() - &before);
        assert_eq!(w.counter().bootstraps, 1);
        assert_eq!(
            (out.level(), out.scale()),
            (f.bs.output_level(), f.ct.scale())
        );
        assert_eq!(f.params.slots(), 512);
    }

    /// Task 48: a client refresh on a chosen output level (level-capped interactive runs):
    /// the fresh ciphertext lands on that level at the exact input scale, with the message;
    /// a level above `max_level` is refused.
    #[test]
    fn client_refresher_on_a_lower_output_level() {
        let params = Arc::new(mini());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(51)).unwrap();
        let encoder = Encoder::new(Arc::clone(&params));
        let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
        let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
        let client = || {
            ClientRefresher::new(
                dec.clone(),
                enc.clone(),
                Encoder::new(Arc::clone(&params)),
                rng(52),
            )
        };
        assert!(matches!(
            client().with_output_level(params.max_level() + 1),
            Err(HekitError::InvalidParams(_))
        ));
        let mut rf = client().with_output_level(5).unwrap();
        assert_eq!(rf.output_level(), 5);
        assert_eq!(client().output_level(), params.max_level());
        let mut r = rng(53);
        let m: Vec<f64> = (0..params.slots()).map(|i| (i % 7) as f64 / 8.0).collect();
        let scale = params.scale() * (1.0 + 2f64.powi(-12));
        let ct = enc.encrypt(&encoder.encode(&m, scale, 1).unwrap(), &mut r);
        let out = rf.refresh(&ct).unwrap();
        assert_eq!((out.level(), out.scale()), (5, scale));
        let bits = precision_bits(&m, &encoder.decode(&dec.decrypt(&out)));
        assert!(bits >= 20.0, "{bits}");
    }
}
