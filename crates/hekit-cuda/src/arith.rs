//! Device arithmetic: pointwise operations, automorphisms, rescaling and the key-switching
//! base conversions (`mod_down`, `mod_up`), each bit-identical to its CPU counterpart
//! (kernels in `kernels/pointwise.cu` and `kernels/basis.cu`; the multi-step algorithms are
//! the shared ones of [`crate::basis`]).
//!
//! Every operation returns a new [`DevicePoly`] and leaves its inputs unchanged. Work is
//! asynchronous on stream 0; [`DevicePoly::download`] waits for it. Operands must belong to
//! this device; binary operands must have the same limbs (prime list) and form. Violations
//! are `InvalidParams`; launch failures are `Device`.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use hekit_core::{HekitError, Result};

use crate::basis::{
    Backend, ModUpPlan, check_galois_elt, divide_round_tail, mod_down_keep, mod_up_digit,
    rescale_keep,
};
use crate::device::{CudaDevice, DeviceConvTable, dev_err, guarded};
use crate::ks::PwOp;
use crate::plan::ElemLaunch;
use crate::poly::DevicePoly;

fn invalid(msg: &str) -> HekitError {
    HekitError::InvalidParams(msg.into())
}

fn elem_cfg(n: usize, rows: usize) -> Result<LaunchConfig> {
    let l = ElemLaunch::new(n, rows)?;
    Ok(LaunchConfig {
        grid_dim: l.grid,
        block_dim: l.block,
        shared_mem_bytes: 0,
    })
}

impl CudaDevice {
    pub(crate) fn alloc_limbs(&self, limbs: usize) -> Result<CudaSlice<u64>> {
        guarded("alloc", || {
            // SAFETY: device memory, not read by the host before every element is written:
            // each caller fills the whole buffer (kernel over all limbs and coefficients, or
            // copies covering every limb) before it is read.
            unsafe { self.stream().alloc::<u64>(limbs * self.n()) }
                .map_err(|e| dev_err("device allocation", e))
        })
    }

    /// A private device copy of `src` (copy on write of [`DevicePoly`]; `Device` on
    /// out-of-memory).
    pub(crate) fn clone_limbs(&self, src: &CudaSlice<u64>) -> Result<CudaSlice<u64>> {
        let mut out = self.alloc_limbs(src.len() / self.n())?;
        guarded("limb copy", || {
            self.stream()
                .memcpy_dtod(src, &mut out)
                .map_err(|e| dev_err("device-to-device copy", e))
        })?;
        Ok(out)
    }

    pub(crate) fn new_poly(
        &self,
        limbs: CudaSlice<u64>,
        primes: Arc<[u32]>,
        ntt_form: bool,
    ) -> DevicePoly<'_> {
        DevicePoly::from_parts(limbs, self.n(), self, primes, ntt_form)
    }

    fn check_pair(&self, a: &DevicePoly<'_>, b: &DevicePoly<'_>) -> Result<()> {
        self.check_owner(a)?;
        self.check_owner(b)?;
        if a.primes() != b.primes() {
            return Err(invalid("operands have different limbs (levels or primes)"));
        }
        if a.is_ntt_form() != b.is_ntt_form() {
            return Err(invalid("operands are in different forms"));
        }
        Ok(())
    }

    /// Launches `pw_add` / `pw_sub` / `pw_mul` into `out` (all `primes.len()` limbs).
    pub(crate) fn launch_binary(
        &self,
        op: PwOp,
        out: &mut CudaSlice<u64>,
        a: &CudaSlice<u64>,
        b: &CudaSlice<u64>,
        primes: &[u32],
    ) -> Result<()> {
        let len = primes.len() * self.n();
        if out.len() != len || a.len() != len || b.len() != len {
            return Err(invalid("pointwise buffers do not match their limbs"));
        }
        let k = &self.kernels;
        let (f, name): (&CudaFunction, _) = match op {
            PwOp::Add => (&k.pw_add, "pw_add"),
            PwOp::Sub => (&k.pw_sub, "pw_sub"),
            PwOp::Mul => (&k.pw_mul, "pw_mul"),
        };
        let map = self.limb_map(primes)?;
        let cfg = elem_cfg(self.n(), primes.len())?;
        let log_n = self.n().trailing_zeros();
        guarded(name, || {
            let mut l = self.stream().launch_builder(f);
            l.arg(out)
                .arg(a)
                .arg(b)
                .arg(&self.ntt_tables.consts)
                .arg(&*map)
                .arg(&log_n);
            // SAFETY: matches `(u64*, const u64*, const u64*, const u64*, const u32*, u32)` in
            // kernels/pointwise.cu; out, a, b hold num_limbs * N entries (checked above), the
            // map num_limbs valid prime indices; grid (ceil(N/256), num_limbs), threads c >= N
            // return.
            unsafe { l.launch(cfg) }.map_err(|e| dev_err(name, e))?;
            Ok(())
        })
    }

    fn binary<'d>(
        &'d self,
        op: PwOp,
        a: &DevicePoly<'d>,
        b: &DevicePoly<'d>,
    ) -> Result<DevicePoly<'d>> {
        self.check_pair(a, b)?;
        let mut out = self.alloc_limbs(a.num_limbs())?;
        self.launch_binary(op, &mut out, a.limbs(), b.limbs(), a.primes())?;
        Ok(self.new_poly(out, Arc::clone(a.primes_arc()), a.is_ntt_form()))
    }

    /// `a + b` (either form, same limbs and form): `RnsPoly::add_assign`.
    pub fn add<'d>(&'d self, a: &DevicePoly<'d>, b: &DevicePoly<'d>) -> Result<DevicePoly<'d>> {
        self.binary(PwOp::Add, a, b)
    }

    /// `a - b` (either form, same limbs and form): `RnsPoly::sub_assign`.
    pub fn sub<'d>(&'d self, a: &DevicePoly<'d>, b: &DevicePoly<'d>) -> Result<DevicePoly<'d>> {
        self.binary(PwOp::Sub, a, b)
    }

    /// The ring product `a * b` of two NTT-form polynomials (pointwise, same limbs):
    /// `RnsPoly::mul_assign`. `InvalidParams` for coefficient-form operands.
    pub fn mul_pointwise<'d>(
        &'d self,
        a: &DevicePoly<'d>,
        b: &DevicePoly<'d>,
    ) -> Result<DevicePoly<'d>> {
        if !a.is_ntt_form() || !b.is_ntt_form() {
            return Err(invalid("mul_pointwise needs both operands in NTT form"));
        }
        self.binary(PwOp::Mul, a, b)
    }

    /// `-a` (either form): `RnsPoly::neg`.
    pub fn neg<'d>(&'d self, a: &DevicePoly<'d>) -> Result<DevicePoly<'d>> {
        self.check_owner(a)?;
        let map = self.limb_map(a.primes())?;
        let mut out = self.alloc_limbs(a.num_limbs())?;
        let cfg = elem_cfg(self.n(), a.num_limbs())?;
        let log_n = self.n().trailing_zeros();
        guarded("pw_neg", || {
            let mut l = self.stream().launch_builder(&self.kernels.pw_neg);
            l.arg(&mut out)
                .arg(a.limbs())
                .arg(&self.ntt_tables.consts)
                .arg(&*map)
                .arg(&log_n);
            // SAFETY: matches `(u64*, const u64*, const u64*, const u32*, u32)`; sizes as in
            // `binary`.
            unsafe { l.launch(cfg) }.map_err(|e| dev_err("pw_neg", e))?;
            Ok(())
        })?;
        Ok(self.new_poly(out, Arc::clone(a.primes_arc()), a.is_ntt_form()))
    }

    /// `a * scalar` (either form; any `u64`, reduced modulo each prime):
    /// `RnsPoly::mul_scalar_assign`. Uploads `2 * num_limbs` words of Shoup constants.
    pub fn mul_scalar<'d>(&'d self, a: &DevicePoly<'d>, scalar: u64) -> Result<DevicePoly<'d>> {
        self.check_owner(a)?;
        let w = self.prime_table().scalar_shoup(a.primes(), scalar);
        let map = self.limb_map(a.primes())?;
        let mut out = self.alloc_limbs(a.num_limbs())?;
        let cfg = elem_cfg(self.n(), a.num_limbs())?;
        let log_n = self.n().trailing_zeros();
        guarded("pw_mul_shoup", || {
            let w = self
                .stream()
                .clone_htod(&w)
                .map_err(|e| dev_err("uploading scalar constants", e))?;
            let mut l = self.stream().launch_builder(&self.kernels.pw_mul_shoup);
            l.arg(&mut out)
                .arg(a.limbs())
                .arg(&w)
                .arg(&self.ntt_tables.consts)
                .arg(&*map)
                .arg(&log_n);
            // SAFETY: matches `(u64*, const u64*, const u64*, const u64*, const u32*, u32)`;
            // w holds 2 * num_limbs words; sizes as in `binary`. `w` is freed stream-ordered
            // after the launch.
            unsafe { l.launch(cfg) }.map_err(|e| dev_err("pw_mul_shoup", e))?;
            Ok(())
        })?;
        Ok(self.new_poly(out, Arc::clone(a.primes_arc()), a.is_ntt_form()))
    }

    /// The Galois automorphism `a(X) -> a(X^g)` in the current form: `RnsPoly::automorphism`
    /// (NTT form: SEAL's index permutation, computed in the kernel; coefficient form: the
    /// signed permutation). `InvalidParams` unless `galois_elt` is odd and `< 2N`.
    pub fn automorphism<'d>(
        &'d self,
        a: &DevicePoly<'d>,
        galois_elt: u64,
    ) -> Result<DevicePoly<'d>> {
        self.check_owner(a)?;
        check_galois_elt(self.n(), galois_elt)?;
        let mut out = self.alloc_limbs(a.num_limbs())?;
        self.launch_automorphism(&mut out, a.limbs(), a.primes(), a.is_ntt_form(), galois_elt)?;
        Ok(self.new_poly(out, Arc::clone(a.primes_arc()), a.is_ntt_form()))
    }

    /// Launches `automorphism_ntt` (NTT form) or `automorphism_coeff` into `out` (a distinct
    /// allocation of `primes.len()` limbs). `galois_elt` must be odd and `< 2N`.
    pub(crate) fn launch_automorphism(
        &self,
        out: &mut CudaSlice<u64>,
        a: &CudaSlice<u64>,
        primes: &[u32],
        ntt_form: bool,
        galois_elt: u64,
    ) -> Result<()> {
        check_galois_elt(self.n(), galois_elt)?;
        let len = primes.len() * self.n();
        if out.len() != len || a.len() != len {
            return Err(invalid("automorphism buffers do not match their limbs"));
        }
        let cfg = elem_cfg(self.n(), primes.len())?;
        let log_n = self.n().trailing_zeros();
        if ntt_form {
            guarded("automorphism_ntt", || {
                let mut l = self.stream().launch_builder(&self.kernels.automorphism_ntt);
                l.arg(out).arg(a).arg(&log_n).arg(&galois_elt);
                // SAFETY: matches `(u64*, const u64*, u32, u64)`; out and a are distinct
                // allocations (`&mut` and `&` borrows) of num_limbs * N entries (checked
                // above); source indices are < N.
                unsafe { l.launch(cfg) }.map_err(|e| dev_err("automorphism_ntt", e))?;
                Ok(())
            })
        } else {
            let map = self.limb_map(primes)?;
            guarded("automorphism_coeff", || {
                let mut l = self
                    .stream()
                    .launch_builder(&self.kernels.automorphism_coeff);
                l.arg(out)
                    .arg(a)
                    .arg(&self.ntt_tables.consts)
                    .arg(&*map)
                    .arg(&log_n)
                    .arg(&galois_elt);
                // SAFETY: matches `(u64*, const u64*, const u64*, const u32*, u32, u64)`;
                // distinct allocations; c -> c g mod N is a bijection (g odd), so every
                // output element is written exactly once.
                unsafe { l.launch(cfg) }.map_err(|e| dev_err("automorphism_coeff", e))?;
                Ok(())
            })
        }
    }

    /// Divides by the last limb's prime with rounding and drops that limb (either form; at
    /// least two limbs over a prefix of the Q or the P primes): `RnsPoly::rescale_by_last`
    /// (SEAL's `divide_and_round_q_last`).
    pub fn rescale_poly<'d>(&'d self, a: &DevicePoly<'d>) -> Result<DevicePoly<'d>> {
        self.check_owner(a)?;
        let keep = rescale_keep(self.prime_table(), a.primes())?;
        let out = divide_round_tail(&Dev(self), a.limbs(), a.primes(), keep, a.is_ntt_form())?;
        Ok(self.new_poly(out, a.primes()[..keep].into(), a.is_ntt_form()))
    }

    /// Divides a `Q_l * P` polynomial (limbs `[0..=l]` then every P prime, e.g.
    /// [`CudaDevice::upload_qp`]; either form) by the product of the P primes with
    /// rounding: the level-`l` result over the Q primes, `RnsPoly::mod_down` of the pair
    /// (exact SEAL rounding for one P prime, the f64 overflow estimate of
    /// `BaseConverter::convert_exact` for several, evaluated with the CPU's exact
    /// floating-point operation sequence).
    pub fn mod_down<'d>(&'d self, x: &DevicePoly<'d>) -> Result<DevicePoly<'d>> {
        self.check_owner(x)?;
        let keep = mod_down_keep(self.prime_table(), x.primes())?;
        let out = divide_round_tail(&Dev(self), x.limbs(), x.primes(), keep, x.is_ntt_form())?;
        Ok(self.new_poly(out, x.primes()[..keep].into(), x.is_ntt_form()))
    }

    /// The exact centered lift of digit `digit_index` of a level-`l` polynomial (limbs
    /// `[0..=l]`, NTT form) to `Q_l * P`: limbs `[0..=l, L+1..L+1+k]`, NTT form, equal to
    /// `hekit_ckks::keyswitch::decompose(params, x)[digit_index]` (digit
    /// `j*alpha .. min((j+1)*alpha, l+1)`, see `digit_ranges`). Download it with
    /// [`DevicePoly::download_qp`].
    ///
    /// `InvalidParams` when `x` is not in NTT form or not over a Q prefix, or the digit does
    /// not exist at this level.
    pub fn mod_up<'d>(&'d self, x: &DevicePoly<'d>, digit_index: usize) -> Result<DevicePoly<'d>> {
        self.check_owner(x)?;
        if !x.is_ntt_form() {
            return Err(invalid("mod_up needs an NTT-form polynomial"));
        }
        let plan = ModUpPlan::new(self.params(), self.prime_table(), x.primes(), digit_index)?;
        let out = mod_up_digit(&Dev(self), x.limbs(), &plan)?;
        Ok(self.new_poly(out, plan.out_primes.into(), true))
    }
}

/// [`Backend`] and [`KsBackend`] on the device: the kernels of `kernels/basis.cu`,
/// `pointwise.cu` and `keyswitch.cu`, the batched NTT and device copies, all on stream 0.
pub(crate) struct Dev<'a>(pub(crate) &'a CudaDevice);

impl Backend for Dev<'_> {
    type Buf = CudaSlice<u64>;
    type Table = Arc<DeviceConvTable>;

    fn alloc(&self, limbs: usize) -> Result<CudaSlice<u64>> {
        self.0.alloc_limbs(limbs)
    }

    fn copy_limbs(
        &self,
        src: &CudaSlice<u64>,
        src_at: usize,
        dst: &mut CudaSlice<u64>,
        dst_at: usize,
        count: usize,
    ) -> Result<()> {
        let n = self.0.n();
        let (s, d) = (
            src_at * n..(src_at + count) * n,
            dst_at * n..(dst_at + count) * n,
        );
        if count == 0 || s.end > src.len() || d.end > dst.len() {
            return Err(invalid("limb copy out of range"));
        }
        guarded("limb copy", || {
            self.0
                .stream()
                .memcpy_dtod(&src.slice(s), &mut dst.slice_mut(d))
                .map_err(|e| dev_err("device-to-device copy", e))
        })
    }

    fn ntt(&self, buf: &mut CudaSlice<u64>, primes: &[u32], inverse: bool) -> Result<()> {
        self.0.launch_ntt(buf, primes, inverse)
    }

    fn conv_table(&self, src: &[u32], targets: &[u32]) -> Result<Arc<DeviceConvTable>> {
        self.0.conv_table(src, targets)
    }

    fn convert(
        &self,
        out: &mut CudaSlice<u64>,
        src: &CudaSlice<u64>,
        t: &Arc<DeviceConvTable>,
    ) -> Result<()> {
        let n = self.0.n();
        if src.len() != t.k as usize * n || out.len() != t.n_targets as usize * n {
            return Err(invalid("conversion buffers do not match the table"));
        }
        let cfg = elem_cfg(n, 1)?;
        let (k, nt, log_n) = (t.k, t.n_targets, n.trailing_zeros());
        let use_f64 = u32::from(t.use_f64);
        guarded("basis_convert", || {
            let mut l = self
                .0
                .stream()
                .launch_builder(&self.0.kernels.basis_convert);
            l.arg(out)
                .arg(src)
                .arg(&t.data)
                .arg(&k)
                .arg(&nt)
                .arg(&log_n)
                .arg(&use_f64);
            // SAFETY: matches `(u64*, const u64*, const u64*, u32, u32, u32, u32)` in
            // kernels/basis.cu; src holds k limbs, out n_targets limbs (checked above), the
            // table 4k + n_targets (6 + 2k) words, k <= MAX_SRC (checked by ConvTable::new).
            unsafe { l.launch(cfg) }.map_err(|e| dev_err("basis_convert", e))?;
            Ok(())
        })
    }

    fn div_finish(
        &self,
        t_out: &mut CudaSlice<u64>,
        a: &CudaSlice<u64>,
        t: &Arc<DeviceConvTable>,
    ) -> Result<()> {
        let n = self.0.n();
        let rows = t.n_targets as usize;
        if t_out.len() != rows * n || a.len() < rows * n {
            return Err(invalid("division buffers do not match the table"));
        }
        let cfg = elem_cfg(n, rows)?;
        let (k, log_n) = (t.k, n.trailing_zeros());
        guarded("basis_div_finish", || {
            let mut l = self
                .0
                .stream()
                .launch_builder(&self.0.kernels.basis_div_finish);
            l.arg(t_out).arg(a).arg(&t.data).arg(&k).arg(&log_n);
            // SAFETY: matches `(u64*, const u64*, const u64*, u32, u32)`; t_out holds
            // n_targets limbs, a at least that many (checked above).
            unsafe { l.launch(cfg) }.map_err(|e| dev_err("basis_div_finish", e))?;
            Ok(())
        })
    }
}
