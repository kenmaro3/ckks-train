//! Task 48: the evaluation backend of the training binaries (`mnist`, `iris`, `ablation`).
//!
//! * `--backend cpu|cuda`: the CPU [`hekit_ckks::Evaluator`] or the CUDA evaluator
//!   (`hekit_cuda::CudaEvaluator`, feature `cuda`). Encryption, decryption, encoding, the
//!   client refresh and bootstrapping stay on the host; on the GPU the client refresh runs
//!   behind `hekit_nn::refresh::HostRefresher` and bootstrapping behind
//!   [`hekit_ckks::HostBootstrapper`] ("level operations on the GPU, bootstrapping on the
//!   CPU"; GPU bootstrapping is Task 49).
//! * `--no-hoist --no-bsgs --no-lazy-relin`: the RQ3 ablation flags ([`EvalConfig`]; see
//!   `hekit_ckks::ops`, "What the RQ3 table must state").
//! * `--matvec sequential|batch`: the matrix-vector products of a training step run one after
//!   the other with `matvec_enc` (the default and the only mode wired into training). `batch`
//!   (`CudaEvaluator::matvec_enc_batch`, several streams) is refused: R-47-2 measured the
//!   4-stream batch 19% **slower** than sequential at n16_train, the re-measurement R-47-3 is
//!   pending, and a training step has no independent products to batch (each matvec consumes
//!   the previous one's output); see [`BackendOpts::validate`].
//! * `--top-level L` (interactive mode): the network, the batches and every client refresh live
//!   on level `L` instead of `max_level`, and the GPU keys are uploaded for levels `<= L` only
//!   (`CudaEvaluator::with_keys`: an n16_train key is 288 MiB in full, 24 MiB at `L = 7`). The
//!   eMLP-1 interactive step uses 7 levels on a full batch (`results/mnist_emlp1.json`, `W1'`
//!   on level 24 from 31), so `L = 7` is the smallest cap that trains it; a partial batch
//!   (fewer valid lanes than `B`) takes one more level at the linear output and needs `L = 8`.
//! * `--gpu-pool default|after-keys`: the device memory pool's release threshold. `default`
//!   keeps `CudaDevice::DEFAULT_POOL_RELEASE_THRESHOLD` (256 MiB, which at n16_train acts
//!   almost like threshold 0); `after-keys` raises it to `pool_used_memory() + 256 MiB` once
//!   the keys and the network are uploaded (the documented Task 47 option). In both cases the
//!   binaries call `CudaDevice::trim_pool(0)` at every epoch boundary and after one-off peaks
//!   (key upload, network upload), through [`DeviceHooks::trim`].
//!
//! The GPU is shared (another tenant holds about 20.7 GB of the 24 GB card), so a GPU run first
//! estimates its device-memory need ([`GpuNeed`]) and stops with [`SKIP_EXIT_CODE`] and a
//! `SKIP` line when the free memory is below it, before generating any key.

use std::cell::Cell;
use std::fmt::Write as _;
use std::rc::Rc;
use std::time::Instant;

use hekit_ckks::{CkksParams, EvalConfig, HeOps};
use hekit_core::error::HekitError;
use hekit_nn::layout::Layout;
use hekit_nn::refresh::Refresher;

/// The process exit code of a binary that skipped its GPU run for lack of device memory.
pub const SKIP_EXIT_CODE: i32 = 3;

/// The prefix of an error that is a skip (insufficient GPU memory), not a failure.
pub const SKIP_PREFIX: &str = "SKIP";

/// Bytes in a MiB.
pub const MIB: f64 = (1u64 << 20) as f64;

/// The evaluator backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The CPU `hekit_ckks::Evaluator`.
    Cpu,
    /// `hekit_cuda::CudaEvaluator` on device 0 (feature `cuda`).
    Cuda,
}

impl Backend {
    /// `"cpu"` or `"cuda"`.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
        }
    }
}

/// How the matrix-vector products of a step are issued (see the [module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatvecMode {
    /// One `matvec_enc` after the other (default).
    Sequential,
    /// `matvec_enc_batch` on several streams (refused, see [`BackendOpts::validate`]).
    Batch,
}

/// The GPU memory pool's release threshold (see the [module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolPolicy {
    /// `CudaDevice::DEFAULT_POOL_RELEASE_THRESHOLD` (256 MiB).
    Default,
    /// `pool_used_memory() + 256 MiB` after the keys and the network are uploaded.
    AfterKeys,
}

/// The backend options shared by the training binaries.
#[derive(Clone, Debug, PartialEq)]
pub struct BackendOpts {
    /// `--backend`.
    pub backend: Backend,
    /// The ablation flags (`--no-hoist`, `--no-bsgs`, `--no-lazy-relin`).
    pub config: EvalConfig,
    /// `--matvec`.
    pub matvec: MatvecMode,
    /// `--gpu-pool`.
    pub pool: PoolPolicy,
    /// `--top-level`: the level of the network and of every refresh in interactive mode
    /// (`None`: `max_level`), and the GPU key level.
    pub top_level: Option<usize>,
}

impl Default for BackendOpts {
    fn default() -> BackendOpts {
        BackendOpts {
            backend: Backend::Cpu,
            config: EvalConfig::default(),
            matvec: MatvecMode::Sequential,
            pool: PoolPolicy::Default,
            top_level: None,
        }
    }
}

/// The usage line of the backend flags.
pub const USAGE: &str = "[--backend cpu|cuda] [--no-hoist] [--no-bsgs] [--no-lazy-relin] \
                         [--matvec sequential|batch] [--gpu-pool default|after-keys] \
                         [--top-level L]";

impl BackendOpts {
    /// Handles `flag` if it is a backend flag (taking its value from `next`): `Ok(true)` if it
    /// was one, `Ok(false)` if not (the caller handles it).
    ///
    /// Errors: a missing or unknown value.
    pub fn take(
        &mut self,
        flag: &str,
        next: &mut dyn FnMut() -> Option<String>,
    ) -> Result<bool, String> {
        let mut value = || next().ok_or_else(|| format!("{flag} needs a value"));
        match flag {
            "--no-hoist" => self.config.hoist = false,
            "--no-bsgs" => self.config.bsgs = false,
            "--no-lazy-relin" => self.config.lazy_relin = false,
            "--backend" => {
                self.backend = match value()?.as_str() {
                    "cpu" => Backend::Cpu,
                    "cuda" => Backend::Cuda,
                    v => return Err(format!("unknown --backend {v} (cpu | cuda)")),
                }
            }
            "--matvec" => {
                self.matvec = match value()?.as_str() {
                    "sequential" => MatvecMode::Sequential,
                    "batch" => MatvecMode::Batch,
                    v => return Err(format!("unknown --matvec {v} (sequential | batch)")),
                }
            }
            "--gpu-pool" => {
                self.pool = match value()?.as_str() {
                    "default" => PoolPolicy::Default,
                    "after-keys" => PoolPolicy::AfterKeys,
                    v => return Err(format!("unknown --gpu-pool {v} (default | after-keys)")),
                }
            }
            "--top-level" => {
                let v = value()?;
                self.top_level = Some(v.parse().map_err(|e| format!("--top-level {v}: {e}"))?);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Checks the combination: `--matvec batch` is refused (module docs), `--backend cuda`
    /// needs the `cuda` feature.
    ///
    /// Errors: a message naming the flag.
    pub fn validate(&self) -> Result<(), String> {
        if self.matvec == MatvecMode::Batch {
            return Err(
                "--matvec batch is not wired into training: R-47-2 measured the \
                        4-stream matvec_enc_batch 19% slower than sequential matvec_enc at \
                        n16_train (R-47-3 re-measurement pending), and the matvecs of one \
                        training step depend on each other; use --matvec sequential"
                    .into(),
            );
        }
        if self.backend == Backend::Cuda && !cfg!(feature = "cuda") {
            return Err("--backend cuda needs a build with the `cuda` feature \
                        (cargo build -p hekit-bench --release --features cuda)"
                .into());
        }
        Ok(())
    }

    /// The level the network and every interactive refresh live on: `--top-level`, or
    /// `max_level`.
    ///
    /// Errors: `--top-level` above `max_level`.
    pub fn start_level(&self, params: &CkksParams) -> Result<usize, String> {
        match self.top_level {
            None => Ok(params.max_level()),
            Some(l) if l <= params.max_level() => Ok(l),
            Some(l) => Err(format!(
                "--top-level {l} above max_level {}",
                params.max_level()
            )),
        }
    }

    /// The options as a compact JSON object.
    pub fn json(&self) -> String {
        format!(
            "{{\"backend\": \"{}\", \"eval_config\": {}, \"matvec\": \"{}\", \"gpu_pool\": \"{}\", \
             \"top_level\": {}}}",
            self.backend.name(),
            config_json(&self.config),
            match self.matvec {
                MatvecMode::Sequential => "sequential",
                MatvecMode::Batch => "batch",
            },
            match self.pool {
                PoolPolicy::Default => "default (256 MiB release threshold)",
                PoolPolicy::AfterKeys => "after-keys (pool used after upload + 256 MiB)",
            },
            self.top_level
                .map_or_else(|| "null".to_string(), |l| l.to_string())
        )
    }
}

/// `"hoist=1 bsgs=1 lazy_relin=1"`-style label of a flag combination.
pub fn config_label(c: &EvalConfig) -> String {
    format!(
        "hoist={} bsgs={} lazy_relin={}",
        u8::from(c.hoist),
        u8::from(c.bsgs),
        u8::from(c.lazy_relin)
    )
}

/// A flag combination as a compact JSON object.
pub fn config_json(c: &EvalConfig) -> String {
    format!(
        "{{\"hoist\": {}, \"bsgs\": {}, \"lazy_relin\": {}, \"default\": {}}}",
        c.hoist,
        c.bsgs,
        c.lazy_relin,
        *c == EvalConfig::default()
    )
}

// ---------------------------------------------------------------------------------------------
// Device-memory estimates (no GPU needed).
// ---------------------------------------------------------------------------------------------

/// Device bytes of one key-switching key uploaded for levels `<= level`
/// (`hekit_cuda::ks::KskShape::device_bytes`: `8 N parts 2 (level + 1 + k)` with `parts` the
/// digits of `level`): n16_train 288 MiB in full, 42 MiB at level 9, 24 MiB at level 7.
pub fn ksk_bytes(params: &CkksParams, level: usize) -> usize {
    let parts = hekit_ckks::keyswitch::digit_ranges(params, level).len();
    8 * params.n() * parts * 2 * (level + 1 + params.p().len())
}

/// Device bytes of a 2-component ciphertext on `level`.
pub fn ct_bytes(params: &CkksParams, level: usize) -> usize {
    2 * (level + 1) * params.n() * 8
}

/// Device bytes of the NTT and prime tables (`hekit_cuda::tables::device_bytes_for`).
pub fn table_bytes(params: &CkksParams) -> usize {
    let np = params.q().len() + params.p().len();
    8 * (4 * np * params.n() + 7 * np)
}

/// The device-memory need of a GPU training run (an estimate, declared before the run).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuNeed {
    /// The relinearization key and the Galois keys at the key level.
    pub keys: usize,
    /// The NTT tables, plus 64 MiB for the device's limb maps, conversion tables and staging.
    pub tables: usize,
    /// The resident state at the start level (weights, biases, optimizer state).
    pub state: usize,
    /// The step's transient ciphertexts at the start level (gradients, new weights,
    /// rotations), an upper estimate.
    pub work: usize,
}

impl GpuNeed {
    /// The total in bytes.
    pub fn total(&self) -> usize {
        self.keys + self.tables + self.state + self.work
    }

    /// The need in MiB as a compact JSON object.
    pub fn json(&self) -> String {
        let m = |b: usize| b as f64 / MIB;
        format!(
            "{{\"total_mib\": {:.1}, \"keys_mib\": {:.1}, \"tables_mib\": {:.1}, \
             \"state_mib\": {:.1}, \"work_mib\": {:.1}}}",
            m(self.total()),
            m(self.keys),
            m(self.tables),
            m(self.state),
            m(self.work)
        )
    }
}

/// The device-memory estimate of an MLP training run on the GPU whose layers have `m[l]`
/// BSGS diagonals each, at `level` with keys for levels `<= level`: `galois + 1` keys; the
/// state is every weight diagonal, its optimizer copy if `velocity` (NAG), and one bias per
/// layer; the work is all gradient diagonals (the trainer computes every gradient before it
/// writes), the largest layer once more (a transposed copy, or the baby rotations), and 16
/// ciphertexts for the batch, the activations, the deltas and the key-switch workspace. Every
/// transient is counted at `level`, which overstates it (the gradients live 4 to 6 levels
/// lower); the optimizer's new weights land on the lowest levels (1 or 2 limbs) and are inside
/// that margin.
pub fn mlp_gpu_need(
    params: &CkksParams,
    galois: usize,
    level: usize,
    m: &[usize],
    velocity: bool,
) -> GpuNeed {
    let ct = ct_bytes(params, level);
    let sum: usize = m.iter().sum();
    let max = m.iter().copied().max().unwrap_or(0);
    let state = (sum * (1 + usize::from(velocity)) + m.len()) * ct;
    let work = (sum + max + 16) * ct;
    GpuNeed {
        keys: (galois + 1) * ksk_bytes(params, level),
        tables: table_bytes(params) + (64 << 20),
        state,
        work,
    }
}

// ---------------------------------------------------------------------------------------------
// Device hooks (pool trimming and the memory log).
// ---------------------------------------------------------------------------------------------

/// The device memory after a trim: `(label, available bytes, pool bytes in use)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemRecord {
    /// Where in the run (e.g. `"epoch 0 end"`).
    pub at: String,
    /// `CudaDevice::available_memory` after the trim.
    pub available: usize,
    /// `CudaDevice::pool_used_memory` (hekit's live device memory).
    pub pool_used: usize,
}

/// What a training loop calls on its backend's device: nothing on the CPU, `trim_pool(0)` and
/// a memory record on the GPU.
pub trait DeviceHooks {
    /// Called at every epoch boundary and after one-off peaks (key and network uploads).
    ///
    /// Errors: the device's.
    fn trim(&self, at: &str) -> Result<(), String>;

    /// Raises the pool's release threshold per [`PoolPolicy::AfterKeys`] (no-op otherwise or
    /// on the CPU). Call it once the keys and the network are on the device.
    ///
    /// Errors: the device's.
    fn apply_pool_policy(&self, policy: PoolPolicy) -> Result<(), String>;

    /// Waits until the device has finished all queued work (no-op on the CPU).
    ///
    /// Errors: the device's.
    fn sync(&self) -> Result<(), String>;

    /// The memory records so far.
    fn memory_log(&self) -> Vec<MemRecord>;

    /// A description of the device (`"cpu"` on the CPU).
    fn describe(&self) -> String;

    /// The memory log as a compact JSON array (MiB).
    fn memory_json(&self) -> String {
        let mut s = String::from("[");
        for (i, r) in self.memory_log().iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            write!(
                s,
                "{{\"at\": \"{}\", \"available_mib\": {:.1}, \"pool_used_mib\": {:.1}}}",
                r.at,
                r.available as f64 / MIB,
                r.pool_used as f64 / MIB
            )
            .expect("write to string");
        }
        s.push(']');
        s
    }
}

/// The CPU: no device.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuHooks;

impl DeviceHooks for CpuHooks {
    fn trim(&self, _at: &str) -> Result<(), String> {
        Ok(())
    }
    fn apply_pool_policy(&self, _policy: PoolPolicy) -> Result<(), String> {
        Ok(())
    }
    fn sync(&self) -> Result<(), String> {
        Ok(())
    }
    fn memory_log(&self) -> Vec<MemRecord> {
        Vec::new()
    }
    fn describe(&self) -> String {
        "cpu".into()
    }
}

/// A refresher that first waits for the device ([`DeviceHooks::sync`]) and records the wait,
/// so that the device time of a step, queued asynchronously, is not attributed to the refresh:
/// the trainer times its refreshes (`EpochLog::refresh_seconds`), and the first download of a
/// refresh would otherwise wait for the whole step. The step's device time is then
/// `seconds - refresh_seconds + wait`, the host refresh `refresh_seconds - wait`
/// ([`SyncedRefresher::wait_seconds`]). On the CPU the wait is zero.
///
/// It syncs before **every** refresh, not only the first of a step: from the second refresh
/// on, the wait also covers the completion of the previous refresh's (asynchronous) upload,
/// so a little upload time moves from the refresh into the step (review M2). Callers that do
/// not sync after uploading the batch (the `mnist` interactive loop) also count that upload's
/// device time in the step; the `ablation` runner syncs after it.
pub struct SyncedRefresher<'h, R> {
    inner: R,
    hooks: &'h dyn DeviceHooks,
    wait: Rc<Cell<f64>>,
}

impl<'h, R> SyncedRefresher<'h, R> {
    /// Wraps `inner`; read the accumulated wait through the returned handle.
    pub fn new(inner: R, hooks: &'h dyn DeviceHooks) -> (SyncedRefresher<'h, R>, Rc<Cell<f64>>) {
        let wait = Rc::new(Cell::new(0.0));
        (
            SyncedRefresher {
                inner,
                hooks,
                wait: Rc::clone(&wait),
            },
            wait,
        )
    }

    /// Seconds spent waiting for the device so far.
    pub fn wait_seconds(&self) -> f64 {
        self.wait.get()
    }
}

impl<E: HeOps, R: Refresher<E>> Refresher<E> for SyncedRefresher<'_, R> {
    fn refresh(&mut self, ct: &E::Ct) -> hekit_core::error::Result<E::Ct> {
        let t = Instant::now();
        self.hooks.sync().map_err(HekitError::Device)?;
        self.wait.set(self.wait.get() + t.elapsed().as_secs_f64());
        self.inner.refresh(ct)
    }

    fn bind_layout(&mut self, layout: &Layout) -> hekit_core::error::Result<()> {
        self.inner.bind_layout(layout)
    }

    fn output_level(&self) -> usize {
        self.inner.output_level()
    }
}

#[cfg(feature = "cuda")]
pub use gpu::{GpuHooks, open_gpu};

#[cfg(feature = "cuda")]
mod gpu {
    use std::cell::RefCell;

    use hekit_ckks::CkksParams;
    use hekit_cuda::CudaDevice;

    use super::{DeviceHooks, GpuNeed, MIB, MemRecord, PoolPolicy, SKIP_PREFIX};

    /// Opens device 0 for `params` (one stream) if `need` fits in its free memory; otherwise
    /// an error starting with [`SKIP_PREFIX`] that states the need and the free memory (the
    /// binaries exit with [`super::SKIP_EXIT_CODE`]). No usable device is an error too.
    ///
    /// Errors: as above, and `CudaDevice::new`'s.
    pub fn open_gpu(what: &str, params: &CkksParams, need: &GpuNeed) -> Result<CudaDevice, String> {
        let free = hekit_cuda::testing::free_device_memory(0)
            .map_err(|e| format!("{what}: no usable CUDA device: {e}"))?;
        if free < need.total() {
            return Err(format!(
                "{SKIP_PREFIX} {what}: needs {:.1} MiB of GPU memory ({}), only {:.1} MiB free; \
                 GPU run unmeasured",
                need.total() as f64 / MIB,
                need.json(),
                free as f64 / MIB
            ));
        }
        let dev = CudaDevice::new(0, params, 1).map_err(|e| format!("{what}: {e}"))?;
        eprintln!(
            "{what}: {} (need {:.1} MiB, free {:.1} MiB)",
            dev.describe(),
            need.total() as f64 / MIB,
            free as f64 / MIB
        );
        Ok(dev)
    }

    /// The GPU hooks: `trim_pool(0)` and a memory record per call.
    #[derive(Debug)]
    pub struct GpuHooks<'d> {
        dev: &'d CudaDevice,
        log: RefCell<Vec<MemRecord>>,
    }

    impl<'d> GpuHooks<'d> {
        /// Hooks on `dev`.
        pub fn new(dev: &'d CudaDevice) -> GpuHooks<'d> {
            GpuHooks {
                dev,
                log: RefCell::new(Vec::new()),
            }
        }
    }

    impl DeviceHooks for GpuHooks<'_> {
        fn trim(&self, at: &str) -> Result<(), String> {
            let e = |e: hekit_core::HekitError| format!("trim_pool at {at}: {e}");
            self.dev.trim_pool(0).map_err(e)?;
            let available = self.dev.available_memory().map_err(e)?;
            let pool_used = self.dev.pool_used_memory().map_err(e)?;
            self.log.borrow_mut().push(MemRecord {
                at: at.to_string(),
                available,
                pool_used,
            });
            Ok(())
        }

        fn apply_pool_policy(&self, policy: PoolPolicy) -> Result<(), String> {
            if policy == PoolPolicy::AfterKeys {
                let used = self.dev.pool_used_memory().map_err(|e| e.to_string())?;
                let t = used as u64 + CudaDevice::DEFAULT_POOL_RELEASE_THRESHOLD;
                self.dev
                    .set_pool_release_threshold(t)
                    .map_err(|e| e.to_string())?;
                eprintln!(
                    "gpu: pool release threshold {:.1} MiB (in use + 256 MiB)",
                    t as f64 / MIB
                );
            }
            Ok(())
        }

        fn sync(&self) -> Result<(), String> {
            self.dev.sync().map_err(|e| e.to_string())
        }

        fn memory_log(&self) -> Vec<MemRecord> {
            self.log.borrow().clone()
        }

        fn describe(&self) -> String {
            self.dev.describe()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<BackendOpts, String> {
        let mut o = BackendOpts::default();
        let mut it = args.iter().map(|s| s.to_string());
        while let Some(f) = it.next() {
            if !o.take(&f, &mut || it.next())? {
                return Err(format!("not a backend flag: {f}"));
            }
        }
        Ok(o)
    }

    #[test]
    fn flags_parse_and_validate() {
        let o = parse(&[]).unwrap();
        assert_eq!(o, BackendOpts::default());
        assert!(o.validate().is_ok());
        let o = parse(&["--no-bsgs", "--no-lazy-relin", "--top-level", "7"]).unwrap();
        assert_eq!(
            o.config,
            EvalConfig {
                hoist: true,
                bsgs: false,
                lazy_relin: false
            }
        );
        assert_eq!(o.top_level, Some(7));
        let o = parse(&["--no-hoist", "--no-bsgs", "--no-lazy-relin"]).unwrap();
        assert_eq!(o.config, EvalConfig::NAIVE);
        assert_eq!(config_label(&o.config), "hoist=0 bsgs=0 lazy_relin=0");
        assert!(parse(&["--backend", "gpu"]).is_err());
        assert!(parse(&["--backend"]).is_err());
        let b = parse(&["--matvec", "batch"]).unwrap();
        assert!(b.validate().unwrap_err().contains("R-47-2"));
        let c = parse(&["--backend", "cuda"]).unwrap();
        assert_eq!(c.validate().is_ok(), cfg!(feature = "cuda"));
        let p = CkksParams::seal_2020();
        assert_eq!(parse(&[]).unwrap().start_level(&p), Ok(9));
        assert!(
            parse(&["--top-level", "10"])
                .unwrap()
                .start_level(&p)
                .is_err()
        );
        assert!(o.json().contains("\"bsgs\": false"));
    }

    /// The key sizes quoted in the Task 47 docs (n16_train: 288 MiB full, 42 MiB at level 9;
    /// seal_2020 55 MiB full) and the table bytes (n16_train about 72 MiB).
    #[test]
    fn memory_estimates_match_the_documented_sizes() {
        let p = CkksParams::n16_train();
        assert_eq!(ksk_bytes(&p, p.max_level()), 288 << 20);
        assert_eq!(ksk_bytes(&p, 9), 42 << 20);
        assert_eq!(ksk_bytes(&p, 7), 24 << 20);
        let s = CkksParams::seal_2020();
        assert_eq!(ksk_bytes(&s, s.max_level()) / (1 << 20), 55);
        assert_eq!(table_bytes(&p) >> 20, 72);
        assert_eq!(ct_bytes(&p, 7), 8 << 20);
        let need = mlp_gpu_need(&p, 31, 7, &[32, 16], true);
        assert_eq!(need.keys, 32 * (24 << 20));
        assert_eq!(need.state, (48 * 2 + 2) * (8 << 20));
        assert_eq!(need.work, (48 + 32 + 16) * (8 << 20));
        assert!(need.json().contains("\"total_mib\""));
    }
}
