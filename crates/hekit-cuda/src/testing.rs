//! Helpers for GPU tests: skip cleanly (with a message, without failing) when there is no
//! usable GPU or not enough free device memory.
//!
//! The GPU server's card is shared with other processes, so every GPU test declares how much
//! device memory it needs and calls [`require_gpu_mem`] first.

use cudarc::driver::{CudaContext, result};
use hekit_ckks::CkksParams;

use crate::device::{dev_err, guarded, probe};
use crate::ks::KskShape;
use crate::{CudaDevice, tables};

/// Free device memory (bytes) on device `ordinal`, or why it cannot be measured (no driver,
/// no device, ...). Creates (retains) the device's primary context, which is what
/// [`CudaDevice::new`] uses, so the context's own reservation is already subtracted.
pub fn free_device_memory(ordinal: usize) -> hekit_core::Result<usize> {
    let count = probe()?;
    if ordinal >= count {
        return Err(hekit_core::HekitError::Device(format!(
            "CUDA device ordinal {ordinal} does not exist ({count} device(s) present)"
        )));
    }
    guarded("free_device_memory", || {
        let ctx = CudaContext::new(ordinal).map_err(|e| dev_err("CUDA context", e))?;
        ctx.bind_to_thread()
            .map_err(|e| dev_err("bind context", e))?;
        let (free, _total) = result::mem_get_info().map_err(|e| dev_err("cuMemGetInfo", e))?;
        Ok(free)
    })
}

/// Device memory of the NTT tables for `params` (what [`CudaDevice::new`] allocates).
pub fn table_bytes(params: &CkksParams) -> usize {
    tables::device_bytes_for(params.n(), params.q().len() + params.p().len())
}

/// Device memory of a key-switching key for `params` uploaded for levels `<= max_level`
/// ([`CudaDevice::upload_ksk_with`]; `max_level = L` for a full key: n16_train 288 MiB,
/// seal_2020 55 MiB). `InvalidParams` above `L`.
pub fn ksk_bytes(params: &CkksParams, max_level: usize) -> hekit_core::Result<usize> {
    Ok(KskShape::new(params, max_level)?.device_bytes(params.n()))
}

/// Allocates (and frees) `limbs` limbs of device memory through the allocation path every
/// device operation and [`CudaDevice::upload_ksk`] use, to observe that out-of-memory is a
/// `Device` error and never an abort.
pub fn try_alloc_limbs(dev: &CudaDevice, limbs: usize) -> hekit_core::Result<()> {
    dev.alloc_limbs(limbs).map(drop)
}

/// Opens device 0 for `params` (one stream) if the test can run there, else prints
/// `SKIP <test>: ...` and returns `None`.
///
/// `need_bytes` is the test's total device-memory need **including** the NTT tables
/// ([`table_bytes`]). The test is skipped (not failed) when there is no driver, NVRTC or GPU,
/// or when free device memory is below `need_bytes`. If the GPU is usable and has room but
/// [`CudaDevice::new`] still fails (e.g. NVRTC compilation), this panics: that is a real
/// failure, not a skip.
pub fn require_gpu_mem(test: &str, need_bytes: usize, params: &CkksParams) -> Option<CudaDevice> {
    const MIB: f64 = (1u64 << 20) as f64;
    let free = match free_device_memory(0) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("SKIP {test}: no usable CUDA device: {e}");
            return None;
        }
    };
    if free < need_bytes {
        eprintln!(
            "SKIP {test}: needs {:.1} MiB of GPU memory, only {:.1} MiB free",
            need_bytes as f64 / MIB,
            free as f64 / MIB
        );
        return None;
    }
    match CudaDevice::new(0, params, 1) {
        Ok(d) => {
            eprintln!(
                "{test}: {} (need {:.1} MiB, free {:.1} MiB)",
                d.describe(),
                need_bytes as f64 / MIB,
                free as f64 / MIB
            );
            Some(d)
        }
        Err(e) => panic!("{test}: GPU present with enough memory but CudaDevice::new failed: {e}"),
    }
}
