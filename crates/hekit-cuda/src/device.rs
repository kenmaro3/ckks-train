//! Device context: CUDA context and streams, NVRTC-compiled NTT kernels, device twiddle tables,
//! and the batched NTT / inverse NTT over the limbs of a [`DevicePoly`].

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, result, sys,
};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use hekit_ckks::{CkksParams, Plaintext};
use hekit_core::{HekitError, Result, RnsPoly};

use crate::NTT_KERNEL_SRC;
use crate::plan::NttPlan;
use crate::poly::{DevicePlaintext, DevicePoly};
use crate::tables::HostNttTables;

const FWD_GLOBAL: &str = "ntt_fwd_global_stage";
const FWD_SHARED: &str = "ntt_fwd_shared";
const INV_GLOBAL: &str = "ntt_inv_global_stage";
const INV_SHARED: &str = "ntt_inv_shared";

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
/// [`HostNttTables`].
#[derive(Debug)]
pub(crate) struct DeviceNttTables {
    pub(crate) roots: CudaSlice<u64>,
    pub(crate) roots_shoup: CudaSlice<u64>,
    pub(crate) inv_roots: CudaSlice<u64>,
    pub(crate) inv_roots_shoup: CudaSlice<u64>,
    pub(crate) consts: CudaSlice<u64>,
    /// Identity limb-to-prime map `[0, 1, ..., num_primes - 1]`: a polynomial whose limbs use
    /// the consecutive primes `off..off + limbs` passes the view `[off, off + limbs)`.
    pub(crate) identity: CudaSlice<u32>,
}

#[derive(Debug)]
struct Kernels {
    fwd_global: CudaFunction,
    fwd_shared: CudaFunction,
    inv_global: CudaFunction,
    inv_shared: CudaFunction,
}

/// A CUDA device prepared for one CKKS parameter set: context, streams, compiled kernels and
/// the NTT tables of every Q and P prime.
///
/// All work of this task's operations is issued on `streams[0]`, in order; the extra streams
/// are for later multi-stream batching. Device memory held by the context itself is
/// [`CudaDevice::table_bytes`] plus what the CUDA context reserves.
#[derive(Debug)]
pub struct CudaDevice {
    ctx: Arc<CudaContext>,
    streams: Vec<Arc<CudaStream>>,
    ntt_tables: DeviceNttTables,
    kernels: Kernels,
    params: CkksParams,
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
            let opts = CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(NTT_KERNEL_SRC, opts)
                .map_err(|e| dev_err("NVRTC compilation of kernels/ntt.cu", e))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| dev_err("loading the NTT module", e))?;
            let load = |f: &str| {
                module
                    .load_function(f)
                    .map_err(|e| dev_err(&format!("kernel {f}"), e))
            };
            let kernels = Kernels {
                fwd_global: load(FWD_GLOBAL)?,
                fwd_shared: load(FWD_SHARED)?,
                inv_global: load(INV_GLOBAL)?,
                inv_shared: load(INV_SHARED)?,
            };
            let streams = (0..n_streams)
                .map(|_| ctx.new_stream().map_err(|e| dev_err("stream creation", e)))
                .collect::<Result<Vec<_>>>()?;
            let s = &streams[0];
            let up = |v: &[u64], what: &str| {
                s.clone_htod(v)
                    .map_err(|e| dev_err(&format!("uploading {what}"), e))
            };
            let identity: Vec<u32> = (0..host.num_primes() as u32).collect();
            let ntt_tables = DeviceNttTables {
                roots: up(&host.roots, "NTT roots")?,
                roots_shoup: up(&host.roots_shoup, "NTT roots (Shoup)")?,
                inv_roots: up(&host.inv_roots, "inverse NTT roots")?,
                inv_roots_shoup: up(&host.inv_roots_shoup, "inverse NTT roots (Shoup)")?,
                consts: up(&host.consts, "NTT constants")?,
                identity: s
                    .clone_htod(&identity)
                    .map_err(|e| dev_err("uploading the limb map", e))?,
            };
            s.synchronize().map_err(|e| dev_err("synchronize", e))?;
            Ok(CudaDevice {
                ctx,
                streams,
                ntt_tables,
                kernels,
                params: params.clone(),
                n: host.n,
                num_primes: host.num_primes(),
                table_bytes: host.device_bytes(),
                name,
                compute_capability: (major, minor),
            })
        })
    }

    /// Device memory taken by this device's NTT tables
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

    pub(crate) fn stream(&self) -> &Arc<CudaStream> {
        &self.streams[0]
    }

    /// Prime-table offset of a polynomial over `poly.basis()`: 0 if its limbs use the first Q
    /// primes, `L + 1` if they use the first P primes; `InvalidParams` otherwise.
    fn prime_offset(&self, poly: &RnsPoly) -> Result<usize> {
        let limbs = poly.level() + 1;
        let primes: Vec<u64> = poly.basis().moduli()[..limbs]
            .iter()
            .map(|m| m.value())
            .collect();
        let nq = self.params.q().len();
        if poly.basis().n() == self.n {
            if self.params.q().get(..limbs) == Some(&primes[..]) {
                return Ok(0);
            }
            if self.params.p().get(..limbs) == Some(&primes[..]) {
                return Ok(nq);
            }
        }
        Err(HekitError::InvalidParams(
            "polynomial's primes are not a prefix of this device's Q or P basis".into(),
        ))
    }

    /// Copies `poly` (any form; its first `level + 1` primes must be a prefix of the device's
    /// Q or P primes) to the device. Synchronous.
    pub fn upload<'d>(&'d self, poly: &RnsPoly) -> Result<DevicePoly<'d>> {
        let prime_offset = self.prime_offset(poly)?;
        let num_limbs = poly.level() + 1;
        let flat: Vec<u64> = poly.limbs().concat();
        let limbs = guarded("upload", || {
            self.stream()
                .clone_htod(&flat)
                .map_err(|e| dev_err("upload (allocation or host-to-device copy)", e))
        })?;
        Ok(DevicePoly::from_parts(
            limbs,
            self.n,
            num_limbs,
            self,
            Arc::clone(poly.basis()),
            prime_offset,
            poly.is_ntt_form(),
        ))
    }

    /// Copies a plaintext (NTT-form polynomial, scale, level) to the device.
    pub fn upload_pt<'d>(&'d self, pt: &Plaintext) -> Result<DevicePlaintext<'d>> {
        let poly = self.upload(pt.poly())?;
        Ok(DevicePlaintext::from_parts(poly, pt.scale(), pt.level()))
    }

    fn check_owner(&self, poly: &DevicePoly<'_>) -> Result<()> {
        if std::ptr::eq(poly.device(), self) {
            Ok(())
        } else {
            Err(HekitError::InvalidParams(
                "DevicePoly belongs to another CudaDevice".into(),
            ))
        }
    }

    /// In-place forward NTT of every limb (no-op when already in NTT form): one launch per
    /// global stage plus one shared-memory launch, each covering all limbs. Asynchronous on
    /// stream 0 (later downloads / [`CudaDevice::sync`] wait for it).
    pub fn ntt(&self, poly: &mut DevicePoly<'_>) -> Result<()> {
        self.check_owner(poly)?;
        if poly.is_ntt_form() {
            return Ok(());
        }
        self.launch_ntt(poly, false)?;
        poly.set_ntt_form(true);
        Ok(())
    }

    /// In-place inverse NTT of every limb (no-op when already in coefficient form), batched
    /// as in [`CudaDevice::ntt`].
    pub fn intt(&self, poly: &mut DevicePoly<'_>) -> Result<()> {
        self.check_owner(poly)?;
        if !poly.is_ntt_form() {
            return Ok(());
        }
        self.launch_ntt(poly, true)?;
        poly.set_ntt_form(false);
        Ok(())
    }

    fn launch_ntt(&self, poly: &mut DevicePoly<'_>, inverse: bool) -> Result<()> {
        let num_limbs = poly.num_limbs();
        let off = poly.prime_offset();
        debug_assert!(off + num_limbs <= self.num_primes);
        let plan = NttPlan::new(self.n, num_limbs)?;
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
            let limb_prime = t.identity.slice(off..off + num_limbs);
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
                    .arg(&limb_prime)
                    .arg(&log_n)
                    .arg(&log_chunk);
                // SAFETY: the arguments match the kernel signature in kernels/ntt.cu
                // (u64*, const u64*, const u64*, const u64*, const u32*, u32, u32); `a` holds
                // num_limbs * N entries, `limb_prime` num_limbs prime indices < num_primes, the
                // tables num_primes * N entries; shared_mem_bytes = chunk * 8 as required.
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
                    .arg(&limb_prime)
                    .arg(&log_n)
                    .arg(&g.log_t);
                // SAFETY: as for the shared kernel; one thread per butterfly, indices stay in
                // [0, N) of each limb (kernel returns for k >= N/2).
                unsafe { b.launch(cfg) }.map_err(|e| dev_err("NTT global-stage kernel", e))?;
                Ok::<(), HekitError>(())
            };
            let a = poly.limbs_mut();
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
