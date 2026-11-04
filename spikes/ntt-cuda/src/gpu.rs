//! Host side: device probe, NVRTC compilation of `kernels/ntt.cu`, kernel launches following
//! [`LaunchPlan`], and a CUDA-event timer.
//!
//! `cudarc` (built with the explicit `cuda-12040` bindings and its default dynamic loading)
//! opens `libcuda` / `libnvrtc` only at first use, so this module compiles on machines without
//! CUDA. A missing library makes `cudarc` panic; [`probe_device`] and [`GpuNtt::new`] turn that
//! into an `Err` so tests and the bench can skip cleanly.

use std::error::Error;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig, result, sys,
};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use crate::{KERNEL_SRC, LaunchPlan, NttTables};

/// Boxed error used by the host code.
pub type BoxError = Box<dyn Error + Send + Sync>;

const MODULE: &str = "ntt";
const GLOBAL_STAGE: &str = "ntt_fwd_global_stage";
const SHARED: &str = "ntt_fwd_shared";

/// Runs `f`, converting a panic (e.g. cudarc failing to `dlopen` a CUDA library) into `Err`.
/// The default panic hook is silenced meanwhile so the skip message stays readable (the hook
/// swap is serialised so concurrent test threads restore the right hook).
fn no_panic<T>(f: impl FnOnce() -> Result<T, BoxError>) -> Result<T, BoxError> {
    static HOOK_LOCK: Mutex<()> = Mutex::new(());
    let _guard = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(prev);
    match r {
        Ok(v) => v,
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".into());
            Err(msg.into())
        }
    }
}

/// Number of CUDA devices, or why CUDA is unusable (no driver library, no device, ...).
pub fn probe_device() -> Result<usize, BoxError> {
    no_panic(|| {
        result::init()?;
        let n = result::device::get_count()?;
        if n <= 0 {
            return Err("CUDA driver loaded but no CUDA device is present".into());
        }
        Ok(n as usize)
    })
}

/// Batched forward NTT on one device, for one fixed `N` and `q`.
pub struct GpuNtt {
    dev: Arc<CudaDevice>,
    q: u64,
    n: usize,
    d_roots: CudaSlice<u64>,
    d_roots_shoup: CudaSlice<u64>,
    f_global: CudaFunction,
    f_shared: CudaFunction,
}

impl GpuNtt {
    /// Opens device `ordinal`, compiles the kernels for its compute capability with NVRTC and
    /// uploads the twiddle tables of `t`.
    pub fn new(ordinal: usize, t: &NttTables) -> Result<Self, BoxError> {
        no_panic(|| {
            let dev = CudaDevice::new(ordinal)?;
            use sys::CUdevice_attribute::*;
            let major = dev.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
            let minor = dev.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
            let opts = CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)?;
            dev.load_ptx(ptx, MODULE, &[GLOBAL_STAGE, SHARED])?;
            let f_global = dev.get_func(MODULE, GLOBAL_STAGE).ok_or("missing kernel")?;
            let f_shared = dev.get_func(MODULE, SHARED).ok_or("missing kernel")?;
            let d_roots = dev.htod_sync_copy(&t.roots)?;
            let d_roots_shoup = dev.htod_sync_copy(&t.roots_shoup)?;
            Ok(Self {
                dev,
                q: t.q,
                n: t.n,
                d_roots,
                d_roots_shoup,
                f_global,
                f_shared,
            })
        })
    }

    /// The device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.dev
    }

    /// `"<name> (sm_XY)"`.
    pub fn describe(&self) -> Result<String, BoxError> {
        use sys::CUdevice_attribute::*;
        let major = self
            .dev
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self
            .dev
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok(format!("{} (sm_{major}{minor})", self.dev.name()?))
    }

    /// Enqueues the forward NTT of `batch` contiguous polynomials held in `d_a` (in place,
    /// asynchronous on the device's stream). `d_a.len()` must be `N * batch`.
    pub fn forward_batch_device(
        &self,
        d_a: &mut CudaSlice<u64>,
        batch: usize,
    ) -> Result<(), BoxError> {
        assert_eq!(d_a.len(), self.n * batch);
        let plan = LaunchPlan::new(self.n, batch);
        for g in &plan.global_stages {
            let cfg = LaunchConfig {
                grid_dim: g.grid,
                block_dim: g.block,
                shared_mem_bytes: g.shared_mem_bytes,
            };
            let params = (
                &mut *d_a,
                &self.d_roots,
                &self.d_roots_shoup,
                self.q,
                plan.log_n,
                g.m,
                g.log_t,
            );
            // SAFETY: argument list matches `ntt_fwd_global_stage` in kernels/ntt.cu
            // (u64*, const u64*, const u64*, u64, u32, u32, u32); the plan keeps all indices
            // inside `d_a` (N * batch) and the tables (N).
            unsafe { self.f_global.clone().launch(cfg, params) }?;
        }
        let s = &plan.shared;
        let cfg = LaunchConfig {
            grid_dim: s.grid,
            block_dim: s.block,
            shared_mem_bytes: s.shared_mem_bytes,
        };
        let params = (
            &mut *d_a,
            &self.d_roots,
            &self.d_roots_shoup,
            self.q,
            plan.log_n,
            s.m,
            s.log_chunk,
        );
        // SAFETY: matches `ntt_fwd_shared` (u64*, const u64*, const u64*, u64, u32, u32, u32);
        // shared_mem_bytes = chunk * 8 as the kernel expects.
        unsafe { self.f_shared.clone().launch(cfg, params) }?;
        Ok(())
    }

    /// Host-to-host convenience: uploads `a`, runs [`Self::forward_batch_device`], downloads.
    pub fn forward_batch(&self, a: &mut [u64], batch: usize) -> Result<(), BoxError> {
        let mut d_a = self.dev.htod_sync_copy(a)?;
        self.forward_batch_device(&mut d_a, batch)?;
        self.dev.dtoh_sync_copy_into(&d_a, a)?;
        Ok(())
    }
}

/// A pair of CUDA events on the device's (default) stream.
pub struct EventTimer {
    start: sys::CUevent,
    stop: sys::CUevent,
    stream: sys::CUstream,
}

impl EventTimer {
    /// Creates the two events (`dev` must be bound to the calling thread, which
    /// `CudaDevice::new` does).
    pub fn new(dev: &Arc<CudaDevice>) -> Result<Self, BoxError> {
        dev.bind_to_thread()?;
        let flags = sys::CUevent_flags::CU_EVENT_DEFAULT;
        Ok(Self {
            start: result::event::create(flags)?,
            stop: result::event::create(flags)?,
            stream: *dev.cu_stream(),
        })
    }

    /// Records the start event.
    pub fn start(&self) -> Result<(), BoxError> {
        // SAFETY: events created in `new`, not yet destroyed; stream belongs to the device.
        unsafe { result::event::record(self.start, self.stream) }?;
        Ok(())
    }

    /// Records the stop event, synchronises the stream, and returns the elapsed milliseconds.
    pub fn stop_ms(&self) -> Result<f32, BoxError> {
        // SAFETY: as in `start`; both events recorded on the same stream.
        unsafe {
            result::event::record(self.stop, self.stream)?;
            result::stream::synchronize(self.stream)?;
            Ok(result::event::elapsed(self.start, self.stop)?)
        }
    }
}

impl Drop for EventTimer {
    fn drop(&mut self) {
        // SAFETY: created in `new`, destroyed exactly once here.
        unsafe {
            let _ = result::event::destroy(self.start);
            let _ = result::event::destroy(self.stop);
        }
    }
}
