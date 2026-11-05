//! Device context: CUDA context and streams, NVRTC-compiled kernels, device twiddle tables,
//! the per-polynomial limb-to-prime maps and conversion tables (cached), and the batched NTT /
//! inverse NTT over the limbs of a [`DevicePoly`].

use std::cell::Cell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
    result, sys,
};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use hekit_ckks::{CkksParams, Plaintext, QpPoly};
use hekit_core::{HekitError, Result, RnsPoly};

use crate::basis::PrimeTable;
use crate::plan::NttPlan;
use crate::poly::{DevicePlaintext, DevicePoly};
use crate::tables::HostNttTables;
use crate::{BASIS_KERNEL_SRC, KEYSWITCH_KERNEL_SRC, NTT_KERNEL_SRC, POINTWISE_KERNEL_SRC};

/// `HekitError::Device` from any displayable error, with context.
pub(crate) fn dev_err(what: &str, e: impl std::fmt::Display) -> HekitError {
    HekitError::Device(format!("{what}: {e}"))
}

/// Runs `f`, turning a panic inside `cudarc` (e.g. a driver entry point missing from an older
/// driver) into `HekitError::Device` instead of unwinding through the caller.
pub(crate) fn guarded<T>(what: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".into());
            Err(HekitError::Device(format!(
                "{what}: panic in CUDA bindings: {msg}"
            )))
        }
    }
}

/// Number of CUDA devices, or `Device` explaining why CUDA is unusable (no driver library, no
/// NVRTC library, driver initialisation failure, no device). Never panics.
pub fn probe() -> Result<usize> {
    // SAFETY: only tries to dlopen the driver / NVRTC libraries (no symbol is called).
    if !unsafe { sys::is_culib_present() } {
        return Err(HekitError::Device(
            "CUDA driver library (libcuda) not found: no NVIDIA driver on this machine".into(),
        ));
    }
    // SAFETY: as above.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(HekitError::Device(
            "NVRTC library (libnvrtc) not found: add the CUDA toolkit's lib64 (e.g. \
             /usr/local/cuda-13.1/lib64) to LD_LIBRARY_PATH"
                .into(),
        ));
    }
    guarded("probe", || {
        result::init().map_err(|e| dev_err("cuInit", e))?;
        let n = result::device::get_count().map_err(|e| dev_err("cuDeviceGetCount", e))?;
        if n <= 0 {
            return Err(HekitError::Device(
                "CUDA driver loaded but no CUDA device is present".into(),
            ));
        }
        Ok(n as usize)
    })
}

/// The NTT tables of all primes (Q then P) on the device, in the flat layout of
/// [`HostNttTables`] (`consts` also holds the Barrett constants of the pointwise product).
#[derive(Debug)]
pub(crate) struct DeviceNttTables {
    pub(crate) roots: CudaSlice<u64>,
    pub(crate) roots_shoup: CudaSlice<u64>,
    pub(crate) inv_roots: CudaSlice<u64>,
    pub(crate) inv_roots_shoup: CudaSlice<u64>,
    pub(crate) consts: CudaSlice<u64>,
}

/// The compiled kernels (four NVRTC modules: `ntt.cu`, `pointwise.cu`, `basis.cu`,
/// `keyswitch.cu`).
#[derive(Debug)]
pub(crate) struct Kernels {
    fwd_global: CudaFunction,
    fwd_shared: CudaFunction,
    inv_global: CudaFunction,
    inv_shared: CudaFunction,
    pub(crate) pw_add: CudaFunction,
    pub(crate) pw_sub: CudaFunction,
    pub(crate) pw_mul: CudaFunction,
    pub(crate) pw_neg: CudaFunction,
    pub(crate) pw_mul_shoup: CudaFunction,
    pub(crate) automorphism_ntt: CudaFunction,
    pub(crate) automorphism_coeff: CudaFunction,
    pub(crate) basis_convert: CudaFunction,
    pub(crate) basis_div_finish: CudaFunction,
    pub(crate) ks_inner_product: CudaFunction,
}

/// A [`crate::basis::ConvTable`] on the device.
#[derive(Debug)]
pub(crate) struct DeviceConvTable {
    pub(crate) k: u32,
    pub(crate) n_targets: u32,
    pub(crate) use_f64: bool,
    pub(crate) data: CudaSlice<u64>,
}

/// Small device buffers built on first use and kept for the device's lifetime: limb-to-prime
/// maps (one per distinct prime list, at most a few hundred u32 lists for a parameter set)
/// and conversion tables (one per (sources, targets) pair: per level for rescale and
/// mod_down, per level and digit for mod_up; about 1 MiB in all for n16_train).
#[derive(Debug, Default)]
struct Caches {
    limb_maps: HashMap<Vec<u32>, Arc<CudaSlice<u32>>>,
    conv: HashMap<(Vec<u32>, Vec<u32>), Arc<DeviceConvTable>>,
}

thread_local! {
    /// The calling thread's current stream selection ([`CudaDevice::use_stream`]): the
    /// address of the device it was made for and the stream index (`(0, 0)`: none). A
    /// selection applies to that device only; every other device uses its stream 0.
    static CURRENT_STREAM: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

/// Slots per staging ring (per stream).
const STAGING_SLOTS: usize = 32;

/// Words per staging slot for `params`: the digit addresses of an inner product (`dnum`) or
/// the Shoup constants of a scalar product (2 per prime).
fn staging_words(params: &CkksParams) -> usize {
    params.dnum().max(2 * (params.q().len() + params.p().len()))
}

/// A ring of pinned host buffers and device buffers for small per-launch uploads
/// ([`CudaDevice::with_staged`]).
struct Staging {
    slots: Vec<(PinnedHostSlice<u64>, CudaSlice<u64>)>,
    next: usize,
}

impl std::fmt::Debug for Staging {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Staging")
            .field("slots", &self.slots.len())
            .field("next", &self.next)
            .finish()
    }
}

impl Staging {
    fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, words: usize) -> Result<Staging> {
        guarded("staging ring", || {
            let slots = (0..STAGING_SLOTS)
                .map(|_| {
                    // SAFETY: the pinned buffer is only written (never read) by the host
                    // before its first copy; unset words are never read by a kernel (the
                    // launches read only the words just written).
                    let mut host = unsafe { ctx.alloc_pinned::<u64>(words) }
                        .map_err(|e| dev_err("pinned host allocation", e))?;
                    host.as_mut_slice()
                        .map_err(|e| dev_err("pinned staging buffer", e))?
                        .fill(0);
                    let dev = stream
                        .alloc_zeros::<u64>(words)
                        .map_err(|e| dev_err("staging device buffer", e))?;
                    Ok((host, dev))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Staging { slots, next: 0 })
        })
    }
}

/// Selects the calling thread's stream on one device until dropped
/// ([`CudaDevice::use_stream`]). It borrows the device (so it cannot outlive it) and is not
/// `Send` (the selection belongs to the thread that made it). Dropping restores the previous
/// selection, so nested guards must be dropped in reverse order (as scopes do).
#[derive(Debug)]
#[must_use = "the stream selection ends when the guard is dropped"]
pub struct StreamGuard<'d> {
    prev: (usize, usize),
    _dev: PhantomData<&'d CudaDevice>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for StreamGuard<'_> {
    fn drop(&mut self) {
        CURRENT_STREAM.with(|c| c.set(self.prev));
    }
}

/// The default memory pool of `ctx`'s device (the pool of the stream-ordered allocations).
fn default_mem_pool(ctx: &Arc<CudaContext>) -> Result<sys::CUmemoryPool> {
    ctx.bind_to_thread()
        .map_err(|e| dev_err("bind context", e))?;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: `pool` is a valid out-pointer; the device handle comes from the live context.
    unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) }
        .result()
        .map_err(|e| dev_err("cuDeviceGetDefaultMemPool", e))?;
    Ok(pool)
}

/// A CUDA device prepared for one CKKS parameter set: context, streams, compiled kernels and
/// the NTT tables of every Q and P prime.
///
/// Prime indices: the device tables hold the `Q` primes (indices `0..=L`) then the `P`
/// primes (`L+1..`); every [`DevicePoly`] carries one prime index per limb, so any list of
/// distinct primes (e.g. `Q_l * P`, `[0..=l, L+1..]`) is one polynomial and one launch
/// sequence.
///
/// # Streams
///
/// Every operation is issued, in order, on **the calling thread's current stream of this
/// device**: stream 0 unless the thread holds a [`StreamGuard`] of this device from
/// [`CudaDevice::use_stream`] (a selection made on another device does not apply; every launch,
/// allocation and copy goes through one internal stream accessor). Work on different streams
/// may overlap. Cross-stream data dependencies are ordered by cudarc's event tracking (a
/// buffer written on one stream and read on another waits for the write); a buffer read on
/// several streams at once records only its last reader, so a caller that shares operands
/// across streams must [`CudaDevice::sync`] (or synchronize those streams) before the
/// operands can be dropped. `CudaEvaluator::matvec_enc_batch` does this.
///
/// Small per-launch tables (the digit addresses of the key-switching inner product, the
/// constants of `mul_scalar`) are staged through a per-stream ring of pinned host buffers
/// and device buffers, so their upload is asynchronous and does not synchronize the stream
/// (Task 46 review M4).
///
/// Device memory held by the device itself is [`CudaDevice::table_bytes`], plus the small
/// cached limb maps and conversion tables and staging rings, plus what the CUDA context
/// reserves.
#[derive(Debug)]
pub struct CudaDevice {
    ctx: Arc<CudaContext>,
    streams: Vec<Arc<CudaStream>>,
    pub(crate) ntt_tables: DeviceNttTables,
    pub(crate) kernels: Kernels,
    params: CkksParams,
    primes: PrimeTable,
    caches: Mutex<Caches>,
    /// One staging ring per stream (built on first use).
    staging: Vec<Mutex<Option<Staging>>>,
    /// Words per staging slot.
    staging_words: usize,
    n: usize,
    num_primes: usize,
    table_bytes: usize,
    name: String,
    compute_capability: (i32, i32),
}

impl CudaDevice {
    /// Opens device `ordinal`, compiles the kernels for its compute capability with NVRTC,
    /// creates `n_streams >= 1` streams and uploads the NTT tables of `params` (Q primes then
    /// P primes).
    ///
    /// Errors: `Device` when there is no driver / NVRTC / device, `ordinal` does not exist,
    /// compilation fails or the tables do not fit in device memory; `InvalidParams` when
    /// `n_streams == 0`.
    pub fn new(ordinal: usize, params: &CkksParams, n_streams: usize) -> Result<CudaDevice> {
        if n_streams == 0 {
            return Err(HekitError::InvalidParams(
                "CudaDevice needs at least one stream".into(),
            ));
        }
        let count = probe()?;
        if ordinal >= count {
            return Err(HekitError::Device(format!(
                "CUDA device ordinal {ordinal} does not exist ({count} device(s) present)"
            )));
        }
        let host = HostNttTables::from_params(params)?;
        guarded("CudaDevice::new", || {
            let ctx = CudaContext::new(ordinal).map_err(|e| dev_err("CUDA context", e))?;
            let name = ctx.name().map_err(|e| dev_err("device name", e))?;
            let (major, minor) = ctx
                .compute_capability()
                .map_err(|e| dev_err("compute capability", e))?;
            // `--fmad=false`: defence in depth for the one floating-point step, the f64
            // overflow estimate in kernels/basis.cu. It must round exactly like the CPU's
            // separate multiply and add; the `__dmul_rn` / `__dadd_rn` intrinsics already
            // forbid contraction, and this keeps a future plain `f += y * w` from being fused
            // into an FMA (which differs from the CPU near rounding ties; the near-tie tests
            // catch it). The other kernels have no floating point, so it costs nothing.
            let opts = CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                fmad: Some(false),
                ..Default::default()
            };
            let module = |src: &str, file: &str| {
                let ptx = compile_ptx_with_opts(src, opts.clone())
                    .map_err(|e| dev_err(&format!("NVRTC compilation of kernels/{file}"), e))?;
                ctx.load_module(ptx)
                    .map_err(|e| dev_err(&format!("loading the module of kernels/{file}"), e))
            };
            let (ntt, pw, basis, ks) = (
                module(NTT_KERNEL_SRC, "ntt.cu")?,
                module(POINTWISE_KERNEL_SRC, "pointwise.cu")?,
                module(BASIS_KERNEL_SRC, "basis.cu")?,
                module(KEYSWITCH_KERNEL_SRC, "keyswitch.cu")?,
            );
            let load = |m: &Arc<cudarc::driver::CudaModule>, f: &str| {
                m.load_function(f)
                    .map_err(|e| dev_err(&format!("kernel {f}"), e))
            };
            let kernels = Kernels {
                fwd_global: load(&ntt, "ntt_fwd_global_stage")?,
                fwd_shared: load(&ntt, "ntt_fwd_shared")?,
                inv_global: load(&ntt, "ntt_inv_global_stage")?,
                inv_shared: load(&ntt, "ntt_inv_shared")?,
                pw_add: load(&pw, "pw_add")?,
                pw_sub: load(&pw, "pw_sub")?,
                pw_mul: load(&pw, "pw_mul")?,
                pw_neg: load(&pw, "pw_neg")?,
                pw_mul_shoup: load(&pw, "pw_mul_shoup")?,
                automorphism_ntt: load(&pw, "automorphism_ntt")?,
                automorphism_coeff: load(&pw, "automorphism_coeff")?,
                basis_convert: load(&basis, "basis_convert")?,
                basis_div_finish: load(&basis, "basis_div_finish")?,
                ks_inner_product: load(&ks, "ks_inner_product")?,
            };
            let streams = (0..n_streams)
                .map(|_| ctx.new_stream().map_err(|e| dev_err("stream creation", e)))
                .collect::<Result<Vec<_>>>()?;
            let s = &streams[0];
            let up = |v: &[u64], what: &str| {
                s.clone_htod(v)
                    .map_err(|e| dev_err(&format!("uploading {what}"), e))
            };
            let ntt_tables = DeviceNttTables {
                roots: up(&host.roots, "NTT roots")?,
                roots_shoup: up(&host.roots_shoup, "NTT roots (Shoup)")?,
                inv_roots: up(&host.inv_roots, "inverse NTT roots")?,
                inv_roots_shoup: up(&host.inv_roots_shoup, "inverse NTT roots (Shoup)")?,
                consts: up(&host.consts, "prime constants")?,
            };
            s.synchronize().map_err(|e| dev_err("synchronize", e))?;
            // Keep freed memory in the device's default pool instead of returning it to the
            // driver at every synchronization (the default threshold is 0): repeated
            // operations then reuse their temporaries without re-mapping physical memory
            // (Task 47 review I2). `CudaDevice::trim_pool` gives it back.
            let pool = default_mem_pool(&ctx)?;
            let mut threshold: u64 = u64::MAX;
            // SAFETY: `pool` is the device's default pool; the release threshold is a
            // `cuuint64_t` attribute, `threshold` lives across the call.
            unsafe {
                sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut threshold as *mut u64 as *mut std::ffi::c_void,
                )
            }
            .result()
            .map_err(|e| dev_err("memory pool release threshold", e))?;
            Ok(CudaDevice {
                ctx,
                streams,
                ntt_tables,
                kernels,
                params: params.clone(),
                primes: PrimeTable::from_params(params),
                caches: Mutex::new(Caches::default()),
                staging: (0..n_streams).map(|_| Mutex::new(None)).collect(),
                staging_words: staging_words(params),
                n: host.n,
                num_primes: host.num_primes(),
                table_bytes: host.device_bytes(),
                name,
                compute_capability: (major, minor),
            })
        })
    }

    /// Device memory taken by this device's NTT and prime tables
    /// ([`crate::tables::device_bytes_for`]; about 72 MiB for `n16_train`).
    pub fn table_bytes(&self) -> usize {
        self.table_bytes
    }

    /// `"<name> (sm_XY)"`.
    pub fn describe(&self) -> String {
        let (major, minor) = self.compute_capability;
        format!("{} (sm_{major}{minor})", self.name)
    }

    /// The device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ctx.ordinal()
    }

    /// The parameter set the tables were built for.
    pub fn params(&self) -> &CkksParams {
        &self.params
    }

    /// Number of streams.
    pub fn n_streams(&self) -> usize {
        self.streams.len()
    }

    /// `(free, total)` device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        guarded("mem_info", || {
            self.ctx
                .bind_to_thread()
                .map_err(|e| dev_err("bind context", e))?;
            result::mem_get_info().map_err(|e| dev_err("cuMemGetInfo", e))
        })
    }

    /// Device memory available for new allocations, in bytes: the driver's free memory plus
    /// what this process's default memory pool holds but does not use (freed memory kept for
    /// reuse, see [`CudaDevice::trim_pool`]).
    pub fn available_memory(&self) -> Result<usize> {
        let (free, _) = self.mem_info()?;
        let pool = guarded("memory pool", || default_mem_pool(&self.ctx))?;
        let attr = |a: sys::CUmemPool_attribute| -> Result<u64> {
            let mut v: u64 = 0;
            // SAFETY: the reserved / used attributes are `cuuint64_t`; `v` lives across the
            // call.
            unsafe {
                sys::cuMemPoolGetAttribute(pool, a, &mut v as *mut u64 as *mut std::ffi::c_void)
            }
            .result()
            .map_err(|e| dev_err("memory pool attribute", e))?;
            Ok(v)
        };
        let reserved = attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?;
        let used = attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?;
        Ok(free + reserved.saturating_sub(used) as usize)
    }

    /// Returns the memory the default pool holds but does not use to the driver, keeping at
    /// most `keep_bytes` (after a peak, to leave room for other processes on a shared GPU).
    /// Waits for every stream first, so frees already queued are included.
    pub fn trim_pool(&self, keep_bytes: usize) -> Result<()> {
        self.sync()?;
        let pool = guarded("memory pool", || default_mem_pool(&self.ctx))?;
        // SAFETY: `pool` is the device's default pool.
        unsafe { sys::cuMemPoolTrimTo(pool, keep_bytes) }
            .result()
            .map_err(|e| dev_err("cuMemPoolTrimTo", e))
    }

    /// Waits until all work issued on every stream of this device has finished.
    pub fn sync(&self) -> Result<()> {
        guarded("sync", || {
            for s in &self.streams {
                s.synchronize()
                    .map_err(|e| dev_err("stream synchronize", e))?;
            }
            Ok(())
        })
    }

    /// The stream every operation of the calling thread is issued on: stream 0, or the one
    /// selected by a live [`StreamGuard`] of this thread (see the type docs, "Streams").
    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.streams[self.stream_index()]
    }

    /// The index of the calling thread's current stream on this device (0 without a
    /// [`StreamGuard`] of this device).
    pub fn stream_index(&self) -> usize {
        let (dev, idx) = CURRENT_STREAM.with(Cell::get);
        if dev == self as *const CudaDevice as usize && idx < self.streams.len() {
            idx
        } else {
            0
        }
    }

    /// Issues the calling thread's work on stream `index` until the returned guard is dropped
    /// (then the previous stream is restored). The selection is per thread, so worker threads
    /// can drive different streams of one device concurrently.
    ///
    /// `InvalidParams` when `index >= n_streams()`.
    pub fn use_stream(&self, index: usize) -> Result<StreamGuard<'_>> {
        if index >= self.streams.len() {
            return Err(HekitError::InvalidParams(format!(
                "stream {index} does not exist ({} stream(s))",
                self.streams.len()
            )));
        }
        let prev = CURRENT_STREAM.with(|c| c.replace((self as *const CudaDevice as usize, index)));
        Ok(StreamGuard {
            prev,
            _dev: PhantomData,
            _not_send: PhantomData,
        })
    }

    /// Waits until all work issued on stream `index` has finished. `InvalidParams` when
    /// `index >= n_streams()`.
    pub fn sync_stream(&self, index: usize) -> Result<()> {
        let s = self
            .streams
            .get(index)
            .ok_or_else(|| HekitError::InvalidParams(format!("stream {index} does not exist")))?;
        guarded("sync_stream", || {
            s.synchronize()
                .map_err(|e| dev_err("stream synchronize", e))
        })
    }

    /// Runs `f` with a device copy of `data` (at most the staging slot size, else uploaded
    /// directly) on the current stream: the words are written to the next pinned host slot of
    /// this stream's ring and copied asynchronously to that slot's device buffer, which `f`
    /// passes to a launch. The ring lock is held during `f`, so the slot is not reused before
    /// the launch is queued; stream order then keeps the copy for the next use of the slot
    /// behind this launch, and writing a pinned slot first waits for its previous copy (a host
    /// wait only when the stream lags `STAGING_SLOTS` uploads behind).
    pub(crate) fn with_staged<R>(
        &self,
        data: &[u64],
        f: impl FnOnce(&CudaSlice<u64>) -> Result<R>,
    ) -> Result<R> {
        let stream = self.stream();
        if data.len() > self.staging_words {
            let buf = guarded("small upload", || {
                stream
                    .clone_htod(data)
                    .map_err(|e| dev_err("uploading a launch table", e))
            })?;
            return f(&buf);
        }
        let idx = self.stream_index();
        let mut ring = self.staging[idx].lock().unwrap_or_else(|e| e.into_inner());
        if ring.is_none() {
            *ring = Some(Staging::new(&self.ctx, stream, self.staging_words)?);
        }
        let ring = ring.as_mut().expect("initialised above");
        let slot = ring.next;
        ring.next = (ring.next + 1) % STAGING_SLOTS;
        let (host, dev) = &mut ring.slots[slot];
        guarded("staged upload", || {
            let h = host
                .as_mut_slice()
                .map_err(|e| dev_err("pinned staging buffer", e))?;
            h[..data.len()].copy_from_slice(data);
            stream
                .memcpy_htod(&*host, dev)
                .map_err(|e| dev_err("staged host-to-device copy", e))
        })?;
        f(dev)
    }

    /// The device's prime indices (Q then P).
    pub fn prime_table(&self) -> &PrimeTable {
        &self.primes
    }

    /// Ring degree `N`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Copies `poly` (any form; its first `level + 1` primes must be a prefix of the device's
    /// Q or P primes, compared by value) to the device. The returned polynomial's limbs have
    /// prime indices `0..=level` (Q) or `L+1..L+1+level+1` (P). Synchronous with respect to
    /// `poly`.
    pub fn upload<'d>(&'d self, poly: &RnsPoly) -> Result<DevicePoly<'d>> {
        let primes = self.primes.primes_of(poly)?;
        self.upload_flat(&poly.limbs().concat(), primes, poly.is_ntt_form())
    }

    /// Copies a `Q_l * P` pair (`Q` part over the Q primes `0..=l`, `P` part over **all** P
    /// primes, same form; e.g. a key-switching digit) to the device as **one** polynomial with
    /// limbs `[0..=l, L+1..L+1+k]` (see [`CudaDevice::mod_down`], [`DevicePoly::download_qp`]).
    pub fn upload_qp<'d>(&'d self, x: &QpPoly) -> Result<DevicePoly<'d>> {
        let (primes, flat) = self.primes.qp_of(x)?;
        self.upload_flat(&flat, primes, x.q().is_ntt_form())
    }

    fn upload_flat<'d>(
        &'d self,
        flat: &[u64],
        primes: Vec<u32>,
        ntt_form: bool,
    ) -> Result<DevicePoly<'d>> {
        let limbs = guarded("upload", || {
            self.stream()
                .clone_htod(flat)
                .map_err(|e| dev_err("upload (allocation or host-to-device copy)", e))
        })?;
        Ok(DevicePoly::from_parts(
            limbs,
            self.n,
            self,
            primes.into(),
            ntt_form,
        ))
    }

    /// Copies a plaintext (NTT-form polynomial, scale, level) to the device.
    pub fn upload_pt<'d>(&'d self, pt: &Plaintext) -> Result<DevicePlaintext<'d>> {
        let poly = self.upload(pt.poly())?;
        Ok(DevicePlaintext::from_parts(poly, pt.scale(), pt.level()))
    }

    pub(crate) fn check_owner(&self, poly: &DevicePoly<'_>) -> Result<()> {
        if std::ptr::eq(poly.device(), self) {
            Ok(())
        } else {
            Err(HekitError::InvalidParams(
                "DevicePoly belongs to another CudaDevice".into(),
            ))
        }
    }

    /// The device copy of the limb-to-prime map `primes` (cached).
    pub(crate) fn limb_map(&self, primes: &[u32]) -> Result<Arc<CudaSlice<u32>>> {
        if primes.is_empty() || primes.iter().any(|&p| p as usize >= self.num_primes) {
            return Err(HekitError::InvalidParams(format!(
                "limb map {primes:?}: prime indices must be < {}",
                self.num_primes
            )));
        }
        if let Some(m) = self.caches().limb_maps.get(primes) {
            return Ok(Arc::clone(m));
        }
        // Upload outside the lock; a racing thread's identical entry wins and ours is dropped.
        let m = Arc::new(guarded("limb map upload", || {
            self.stream()
                .clone_htod(primes)
                .map_err(|e| dev_err("uploading a limb map", e))
        })?);
        let mut c = self.caches();
        Ok(Arc::clone(c.limb_maps.entry(primes.to_vec()).or_insert(m)))
    }

    fn caches(&self) -> std::sync::MutexGuard<'_, Caches> {
        self.caches.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The device copy of the conversion table `src -> targets` (cached).
    pub(crate) fn conv_table(&self, src: &[u32], targets: &[u32]) -> Result<Arc<DeviceConvTable>> {
        let key = (src.to_vec(), targets.to_vec());
        if let Some(t) = self.caches().conv.get(&key) {
            return Ok(Arc::clone(t));
        }
        // Built and uploaded outside the lock, as in `limb_map`.
        let host = self.primes.conv_table(src, targets)?;
        let data = guarded("conversion table upload", || {
            self.stream()
                .clone_htod(&host.data)
                .map_err(|e| dev_err("uploading a conversion table", e))
        })?;
        let t = Arc::new(DeviceConvTable {
            k: host.k() as u32,
            n_targets: host.targets.len() as u32,
            use_f64: host.use_f64,
            data,
        });
        let mut c = self.caches();
        Ok(Arc::clone(c.conv.entry(key).or_insert(t)))
    }

    /// In-place forward NTT of every limb (no-op when already in NTT form): one launch per
    /// global stage plus one shared-memory launch, each covering all limbs (any prime list,
    /// e.g. `Q_l * P`). Asynchronous on the current stream (later downloads / [`CudaDevice::sync`] wait
    /// for it).
    ///
    /// Accepts residues in `[0, 4q)` (every polynomial produced by this crate is fully
    /// reduced). After an `Err` the polynomial's contents are unspecified (some stages may
    /// have run; the form flag is unchanged).
    ///
    /// Copy on write: when `poly`'s storage is shared with a clone, the limbs are first copied
    /// to a private allocation, so the clones are unchanged (`Device` if that copy does not
    /// fit in device memory).
    pub fn ntt(&self, poly: &mut DevicePoly<'_>) -> Result<()> {
        self.check_owner(poly)?;
        if poly.is_ntt_form() {
            return Ok(());
        }
        let primes = Arc::clone(poly.primes_arc());
        self.launch_ntt(poly.limbs_mut()?, &primes, false)?;
        poly.set_ntt_form(true);
        Ok(())
    }

    /// In-place inverse NTT of every limb (no-op when already in coefficient form), batched
    /// as in [`CudaDevice::ntt`].
    ///
    /// **Precondition: every residue is `< 2q`** (the Gentleman-Sande butterflies compute
    /// `x + 2q - y`, which wraps for `y >= x + 2q`). Uploads and every kernel of this crate
    /// produce fully reduced residues (`< q`), so this holds for every [`DevicePoly`]. After an
    /// `Err` the contents are unspecified, as for [`CudaDevice::ntt`]. Copy on write, as
    /// [`CudaDevice::ntt`].
    pub fn intt(&self, poly: &mut DevicePoly<'_>) -> Result<()> {
        self.check_owner(poly)?;
        if !poly.is_ntt_form() {
            return Ok(());
        }
        let primes = Arc::clone(poly.primes_arc());
        self.launch_ntt(poly.limbs_mut()?, &primes, true)?;
        poly.set_ntt_form(false);
        Ok(())
    }

    /// Batched NTT of `a` (`primes.len()` limbs, limb `i` with prime `primes[i]`). The inverse
    /// needs residues `< 2q` (see [`CudaDevice::intt`]).
    pub(crate) fn launch_ntt(
        &self,
        a: &mut CudaSlice<u64>,
        primes: &[u32],
        inverse: bool,
    ) -> Result<()> {
        let num_limbs = primes.len();
        if a.len() != num_limbs * self.n {
            return Err(HekitError::InvalidParams(
                "NTT buffer does not match its limb map".into(),
            ));
        }
        let plan = NttPlan::new(self.n, num_limbs)?;
        let map = self.limb_map(primes)?;
        let t = &self.ntt_tables;
        let (roots, shoup) = if inverse {
            (&t.inv_roots, &t.inv_roots_shoup)
        } else {
            (&t.roots, &t.roots_shoup)
        };
        let k = &self.kernels;
        let (f_global, f_shared) = if inverse {
            (&k.inv_global, &k.inv_shared)
        } else {
            (&k.fwd_global, &k.fwd_shared)
        };
        let log_n = plan.log_n;
        let stream = self.stream();
        guarded("NTT launch", || {
            let limb_prime: &CudaSlice<u32> = &map;
            let shared_cfg = LaunchConfig {
                grid_dim: plan.shared.grid,
                block_dim: plan.shared.block,
                shared_mem_bytes: plan.shared.shared_mem_bytes,
            };
            let log_chunk = plan.shared.log_chunk;
            let launch_shared = |a: &mut CudaSlice<u64>| -> Result<()> {
                let mut b = stream.launch_builder(f_shared);
                b.arg(a)
                    .arg(roots)
                    .arg(shoup)
                    .arg(&t.consts)
                    .arg(limb_prime)
                    .arg(&log_n)
                    .arg(&log_chunk);
                // SAFETY: the arguments match the kernel signature in kernels/ntt.cu
                // (u64*, const u64*, const u64*, const u64*, const u32*, u32, u32); `a` holds
                // num_limbs * N entries, `limb_prime` num_limbs prime indices < num_primes
                // (checked by `limb_map`), the tables num_primes * N entries;
                // shared_mem_bytes = chunk * 8 as required.
                unsafe { b.launch(shared_cfg) }.map_err(|e| dev_err("NTT shared kernel", e))?;
                Ok(())
            };
            let launch_global = |a: &mut CudaSlice<u64>, g: &crate::plan::GlobalStage| {
                let cfg = LaunchConfig {
                    grid_dim: g.grid,
                    block_dim: g.block,
                    shared_mem_bytes: 0,
                };
                let mut b = stream.launch_builder(f_global);
                b.arg(a)
                    .arg(roots)
                    .arg(shoup)
                    .arg(&t.consts)
                    .arg(limb_prime)
                    .arg(&log_n)
                    .arg(&g.log_t);
                // SAFETY: as for the shared kernel; one thread per butterfly, indices stay in
                // [0, N) of each limb (kernel returns for k >= N/2).
                unsafe { b.launch(cfg) }.map_err(|e| dev_err("NTT global-stage kernel", e))?;
                Ok::<(), HekitError>(())
            };
            if inverse {
                launch_shared(a)?;
                for g in plan.global_stages.iter().rev() {
                    launch_global(a, g)?;
                }
            } else {
                for g in &plan.global_stages {
                    launch_global(a, g)?;
                }
                launch_shared(a)?;
            }
            Ok(())
        })
    }
}
