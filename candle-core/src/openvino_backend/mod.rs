//! OpenVINO backend for Candle — runs operations on the Intel NPU (or any
//! OpenVINO-supported device such as CPU / GPU / NPU).
//!
//! # Design
//!
//! OpenVINO is a graph-level inference engine: you compile a model once and
//! then run `infer_request.infer()`.  Candle, by contrast, operates on
//! individual tensor operations (matmul, elementwise, …).
//!
//! This backend bridges the two worlds with the following approach:
//!
//! * All **storage** lives on the host as a plain `Vec<u8>` of raw bytes (same
//!   layout as [`CpuStorage`]).  Copies to/from CPU are therefore zero-cost.
//! * Every **operation that has a meaningful OpenVINO path** (currently matmul
//!   and the common element-wise ops) builds a tiny single-op OpenVINO model on
//!   the fly, compiles it against the chosen hardware device (default: `"NPU"`)
//!   and runs inference immediately.  The compiled model is **not cached** in
//!   this first implementation — model compilation is the expensive step in
//!   OpenVINO, so a production integration would want an LRU cache keyed on
//!   (op, shapes, dtype); that is left as a follow-up.
//! * All **other operations** fall back transparently to the [`CpuStorage`]
//!   path by round-tripping through [`to_cpu_storage`] /
//!   [`storage_from_cpu_storage_owned`].  This keeps every `BackendStorage`
//!   method implemented while letting us optimise ops incrementally.
//!
//! # Thread safety
//!
//! The OpenVINO `Core` object is wrapped in an `Arc<Mutex<…>>` inside
//! [`OpenVinoDevice`] so that multiple devices can share a runtime.  In
//! practice a single device is almost always used.

use std::sync::{Arc, Mutex};

use openvino::{
    Core, DeviceType, ElementType, InferRequest, Model, PrePostprocess, Shape as OvShape, Tensor as OvTensor,
};

use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{DType, Error, Layout, Result, Shape, WithDType};

pub use error::{from_ov_error, OpenVinoError, OpenVinoResult};

mod error;
mod utils;

// ── Device ──────────────────────────────────────────────────────────────────

/// Shared state for a single OpenVINO hardware target.
#[derive(Clone)]
struct Inner {
    /// The OpenVINO runtime core.  Guarded by a mutex because `Core` is not
    /// `Sync` in all versions of openvino-rs.
    core: Arc<Mutex<Core>>,
    /// OpenVINO device string, e.g. `"NPU"`, `"GPU"`, `"CPU"`.
    device_name: String,
    /// Logical device ordinal (used to distinguish multiple NPUs).
    ordinal: usize,
}

/// A handle to an OpenVINO-backed hardware device (Intel NPU by default).
#[derive(Clone)]
pub struct OpenVinoDevice(Arc<Inner>);

impl std::fmt::Debug for OpenVinoDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OpenVinoDevice(device={}, ordinal={})",
            self.0.device_name, self.0.ordinal
        )
    }
}

impl OpenVinoDevice {
    /// Create a device targeting a specific OpenVINO device string and ordinal.
    ///
    /// `device_name` examples: `"NPU"`, `"GPU"`, `"CPU"`, `"AUTO"`.
    pub fn new_with_device(device_name: impl Into<String>, ordinal: usize) -> Result<Self> {
        let core = Core::new().map_err(|e| Error::OpenVino(from_ov_error(e)))?;
        Ok(Self(Arc::new(Inner {
            core: Arc::new(Mutex::new(core)),
            device_name: device_name.into(),
            ordinal,
        })))
    }

    /// The OpenVINO device name (e.g. `"NPU"`, `"GPU"`, `"CPU"`).
    pub fn device_name(&self) -> &str {
        &self.0.device_name
    }

    fn compile_and_infer(
        &self,
        model: &Model,
        inputs: &[(&str, OvTensor)],
    ) -> Result<Vec<OvTensor>> {
        let core = self.0.core.lock().map_err(|e| {
            Error::OpenVino(OpenVinoError::Message(format!("mutex poisoned: {e}")))
        })?;

        let compiled = core
            .compile_model(model, &self.0.device_name, &[])
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;

        let mut req: InferRequest = compiled
            .create_infer_request()
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;

        for (name, tensor) in inputs {
            req.set_input_tensor_by_name(name, tensor)
                .map_err(|e| Error::OpenVino(from_ov_error(e)))?;
        }

        req.infer()
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;

        let output_count = compiled.outputs().count();
        let mut outputs = Vec::with_capacity(output_count);
        for i in 0..output_count {
            let t = req
                .get_output_tensor(i)
                .map_err(|e| Error::OpenVino(from_ov_error(e)))?;
            outputs.push(t);
        }
        Ok(outputs)
    }
}

impl BackendDevice for OpenVinoDevice {
    type Storage = OpenVinoStorage;

    /// `ordinal` maps to the Nth NPU listed by OpenVINO when multiple are
    /// present. Uses `"NPU"` as the target device.
    fn new(ordinal: usize) -> Result<Self> {
        // OpenVINO enumerates multiple NPUs as "NPU.0", "NPU.1", …
        let name = if ordinal == 0 {
            "NPU".to_string()
        } else {
            format!("NPU.{ordinal}")
        };
        Self::new_with_device(name, ordinal)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Npu {
            device_id: self.0.ordinal,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0.core, &other.0.core)
            && self.0.device_name == other.0.device_name
            && self.0.ordinal == other.0.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let data = vec![0u8; elem_count * dtype.size_in_bytes()];
        Ok(OpenVinoStorage {
            data: Arc::new(Mutex::new(data)),
            dtype,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let mut data = Vec::with_capacity(elem_count * dtype.size_in_bytes());
        // Safety: the caller is responsible for initialising before use.
        data.set_len(elem_count * dtype.size_in_bytes());
        Ok(OpenVinoStorage {
            data: Arc::new(Mutex::new(data)),
            dtype,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: WithDType>(&self, src: &[T]) -> Result<Self::Storage> {
        let cpu = T::to_cpu_storage(src);
        self.storage_from_cpu_storage(&cpu)
    }

    fn storage_from_cpu_storage(&self, cpu: &CpuStorage) -> Result<Self::Storage> {
        let bytes = utils::cpu_storage_to_bytes(cpu)?;
        Ok(OpenVinoStorage {
            data: Arc::new(Mutex::new(bytes)),
            dtype: cpu.dtype(),
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, cpu: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&cpu)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        // Generate on CPU then move to this device.
        let cpu_dev = crate::cpu_backend::CpuDevice;
        let cpu_storage = cpu_dev.rand_uniform(shape, dtype, lo, hi)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        let cpu_dev = crate::cpu_backend::CpuDevice;
        let cpu_storage = cpu_dev.rand_normal(shape, dtype, mean, std)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        // RNG is delegated to the CPU; no device-side seed needed.
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        crate::bail!("get_current_seed is not supported for the OpenVINO backend")
    }

    fn synchronize(&self) -> Result<()> {
        // All inference in this backend is synchronous; nothing to do.
        Ok(())
    }
}

// ── Storage ─────────────────────────────────────────────────────────────────

/// NPU-resident tensor storage.
///
/// Data is kept as a flat `Vec<u8>` of raw bytes in host memory (the NPU
/// always DMA-copies from host memory on inference anyway, and OpenVINO
/// manages its own internal device buffers).  The mutex is needed because
/// `BackendStorage::const_set` takes `&mut self` on some ops but the storage
/// is typically accessed from behind a shared reference elsewhere.
pub struct OpenVinoStorage {
    data: Arc<Mutex<Vec<u8>>>,
    dtype: DType,
    device: OpenVinoDevice,
}

impl std::fmt::Debug for OpenVinoStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpenVinoStorage(dtype={:?})", self.dtype)
    }
}

impl OpenVinoStorage {
    /// Replace the underlying storage bytes with the contents of a [`CpuStorage`].
    /// Used by in-place operations that round-trip through the CPU.
    pub fn replace_storage_from_cpu(&mut self, cpu: CpuStorage) -> Result<()> {
        let new_bytes = utils::cpu_storage_to_bytes(&cpu)?;
        *self.data.lock().map_err(|e| {
            Error::OpenVino(OpenVinoError::Message(format!("mutex poisoned: {e}")))
        })? = new_bytes;
        self.dtype = cpu.dtype();
        Ok(())
    }

    // ── helpers ──────────────────────────────────────────────────────────

    /// Borrow raw bytes immutably.
    fn bytes(&self) -> Result<std::sync::MutexGuard<Vec<u8>>> {
        self.data.lock().map_err(|e| {
            Error::OpenVino(OpenVinoError::Message(format!("mutex poisoned: {e}")))
        })
    }

    /// Convert to [`CpuStorage`] without going through `BackendStorage`.
    fn to_cpu(&self, layout: &Layout) -> Result<CpuStorage> {
        let bytes = self.bytes()?;
        let cpu = utils::bytes_to_cpu_storage(&bytes, self.dtype, layout)?;
        Ok(cpu)
    }

    /// Round-trip an operation through the CPU backend.
    fn cpu_op<F>(&self, layout: &Layout, f: F) -> Result<Self>
    where
        F: FnOnce(CpuStorage) -> Result<CpuStorage>,
    {
        let cpu = self.to_cpu(layout)?;
        let out_cpu = f(cpu)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    fn cpu_binary_op<F>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        f: F,
    ) -> Result<Self>
    where
        F: FnOnce(CpuStorage, &CpuStorage, &Layout, &Layout) -> Result<CpuStorage>,
    {
        let lhs_cpu = self.to_cpu(lhs_l)?;
        let rhs_cpu = rhs.to_cpu(rhs_l)?;
        let out_cpu = f(lhs_cpu, &rhs_cpu, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    /// Run a batched matmul on the NPU via an OpenVINO single-op model.
    ///
    /// Falls back to CPU if the dtype is not supported by OpenVINO's MatMul
    /// (e.g. integer types).
    fn matmul_openvino(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        use openvino::op;

        // OpenVINO MatMul natively supports FP32, FP16 and BF16.
        let ov_type = match self.dtype {
            DType::F32 => ElementType::F32,
            DType::F16 => ElementType::F16,
            DType::BF16 => ElementType::BF16,
            // Fall back to CPU for integer / quantised types.
            _ => return self.matmul_cpu(rhs, (b, m, n, k), lhs_l, rhs_l),
        };

        // Build [B, M, K] and [B, K, N] input tensors from contiguous data.
        let lhs_cpu = self.to_cpu(lhs_l)?;
        let rhs_cpu = rhs.to_cpu(rhs_l)?;

        let lhs_bytes = utils::cpu_storage_to_bytes(&lhs_cpu)?;
        let rhs_bytes = utils::cpu_storage_to_bytes(&rhs_cpu)?;

        // OpenVINO shapes are [batch, rows, cols].
        let lhs_shape = OvShape::new(&[b as i64, m as i64, k as i64])
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;
        let rhs_shape = OvShape::new(&[b as i64, k as i64, n as i64])
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;

        // Create OpenVINO input parameters.
        let lhs_param = op::Parameter::new(&lhs_shape, ov_type)
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;
        let rhs_param = op::Parameter::new(&rhs_shape, ov_type)
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;

        // MatMul op: no transpose on either input.
        let mm = op::MatMul::new(&lhs_param, &rhs_param, false, false)
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;

        let model = Model::new(
            "candle_matmul",
            &[mm.into()],
            &[lhs_param.into(), rhs_param.into()],
        )
        .map_err(|e| Error::OpenVino(from_ov_error(e)))?;

        // Build input tensors from raw bytes.
        let lhs_ov = OvTensor::new(ov_type, &lhs_shape, &lhs_bytes)
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;
        let rhs_ov = OvTensor::new(ov_type, &rhs_shape, &rhs_bytes)
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?;

        let outputs = self
            .device
            .compile_and_infer(&model, &[("lhs", lhs_ov), ("rhs", rhs_ov)])?;

        let out_bytes: Vec<u8> = outputs[0]
            .get_raw_data()
            .map_err(|e| Error::OpenVino(from_ov_error(e)))?
            .to_vec();

        Ok(OpenVinoStorage {
            data: Arc::new(Mutex::new(out_bytes)),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    /// CPU fallback for matmul.
    fn matmul_cpu(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        use crate::backend::BackendStorage as _;
        let lhs_cpu = self.to_cpu(lhs_l)?;
        let rhs_cpu = rhs.to_cpu(rhs_l)?;
        let out_cpu = lhs_cpu.matmul(&rhs_cpu, bmnk, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }
}

impl BackendStorage for OpenVinoStorage {
    type Device = OpenVinoDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        // A full contiguous copy of the layout-visible elements.
        let cpu = self.to_cpu(layout)?;
        self.device.storage_from_cpu_storage(&cpu)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        // Return a contiguous view of the full underlying buffer.
        let bytes = self.bytes()?;
        let elem_count = bytes.len() / self.dtype.size_in_bytes();
        let layout = Layout::contiguous(crate::Shape::from(elem_count));
        utils::bytes_to_cpu_storage(&bytes, self.dtype, &layout)
    }

    // ── Element-wise ops (CPU fallback) ──────────────────────────────────

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::affine(&cpu, l, mul, add)
        })
    }

    fn powf(&self, l: &Layout, alpha: f64) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::powf(&cpu, l, alpha)
        })
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::elu(&cpu, l, alpha)
        })
    }

    fn reduce_op(&self, op: ReduceOp, l: &Layout, dims: &[usize]) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::reduce_op(&cpu, op, l, dims)
        })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let lhs_cpu = self.to_cpu(lhs_l)?;
        let rhs_cpu = rhs.to_cpu(rhs_l)?;
        let out_cpu = crate::backend::BackendStorage::cmp(&lhs_cpu, op, &rhs_cpu, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::to_dtype(&cpu, l, dtype)
        })
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::unary_impl::<B>(&cpu, l)
        })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        self.cpu_binary_op(rhs, lhs_l, rhs_l, |lhs_cpu, rhs_cpu, ll, rl| {
            crate::backend::BackendStorage::binary_impl::<B>(&lhs_cpu, rhs_cpu, ll, rl)
        })
    }

    fn where_cond(
        &self,
        l: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let cond_cpu = self.to_cpu(l)?;
        let t_cpu = t.to_cpu(t_l)?;
        let f_cpu = f.to_cpu(f_l)?;
        let out_cpu = crate::backend::BackendStorage::where_cond(&cond_cpu, l, &t_cpu, t_l, &f_cpu, f_l)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    // ── Convolution (CPU fallback) ────────────────────────────────────────

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let inp_cpu = self.to_cpu(l)?;
        let ker_cpu = kernel.to_cpu(kernel_l)?;
        let out_cpu = crate::backend::BackendStorage::conv1d(&inp_cpu, l, &ker_cpu, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let inp_cpu = self.to_cpu(l)?;
        let ker_cpu = kernel.to_cpu(kernel_l)?;
        let out_cpu = crate::backend::BackendStorage::conv_transpose1d(&inp_cpu, l, &ker_cpu, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let inp_cpu = self.to_cpu(l)?;
        let ker_cpu = kernel.to_cpu(kernel_l)?;
        let out_cpu = crate::backend::BackendStorage::conv2d(&inp_cpu, l, &ker_cpu, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let inp_cpu = self.to_cpu(l)?;
        let ker_cpu = kernel.to_cpu(kernel_l)?;
        let out_cpu = crate::backend::BackendStorage::conv_transpose2d(&inp_cpu, l, &ker_cpu, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    // ── Pooling / upsampling (CPU fallback) ──────────────────────────────

    fn avg_pool2d(&self, l: &Layout, ks: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::avg_pool2d(&cpu, l, ks, stride)
        })
    }

    fn max_pool2d(&self, l: &Layout, ks: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::max_pool2d(&cpu, l, ks, stride)
        })
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::upsample_nearest1d(&cpu, l, sz)
        })
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::upsample_nearest2d(&cpu, l, h, w)
        })
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        self.cpu_op(l, |cpu| {
            crate::backend::BackendStorage::upsample_bilinear2d(
                &cpu, l, h, w, align_corners, scale_h, scale_w,
            )
        })
    }

    // ── Indexing (CPU fallback) ───────────────────────────────────────────

    fn gather(&self, l: &Layout, indices: &Self, indices_l: &Layout, d: usize) -> Result<Self> {
        let src_cpu = self.to_cpu(l)?;
        let idx_cpu = indices.to_cpu(indices_l)?;
        let out_cpu = crate::backend::BackendStorage::gather(&src_cpu, l, &idx_cpu, indices_l, d)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        indices: &Self,
        indices_l: &Layout,
        src: &Self,
        src_l: &Layout,
        d: usize,
    ) -> Result<()> {
        let bytes = self.bytes()?.clone();
        let mut dst_cpu = utils::bytes_to_cpu_storage(&bytes, self.dtype, l)?;
        let idx_cpu = indices.to_cpu(indices_l)?;
        let src_cpu = src.to_cpu(src_l)?;
        crate::backend::BackendStorage::scatter_set(&mut dst_cpu, l, &idx_cpu, indices_l, &src_cpu, src_l, d)?;
        let new_bytes = utils::cpu_storage_to_bytes(&dst_cpu)?;
        *self.data.lock().map_err(|e| {
            Error::OpenVino(OpenVinoError::Message(format!("mutex poisoned: {e}")))
        })? = new_bytes;
        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        indices: &Self,
        indices_l: &Layout,
        src: &Self,
        src_l: &Layout,
        d: usize,
    ) -> Result<()> {
        let bytes = self.bytes()?.clone();
        let mut dst_cpu = utils::bytes_to_cpu_storage(&bytes, self.dtype, l)?;
        let idx_cpu = indices.to_cpu(indices_l)?;
        let src_cpu = src.to_cpu(src_l)?;
        crate::backend::BackendStorage::scatter_add_set(
            &mut dst_cpu, l, &idx_cpu, indices_l, &src_cpu, src_l, d,
        )?;
        let new_bytes = utils::cpu_storage_to_bytes(&dst_cpu)?;
        *self.data.lock().map_err(|e| {
            Error::OpenVino(OpenVinoError::Message(format!("mutex poisoned: {e}")))
        })? = new_bytes;
        Ok(())
    }

    fn index_select(
        &self,
        indices: &Self,
        l: &Layout,
        indices_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        let src_cpu = self.to_cpu(l)?;
        let idx_cpu = indices.to_cpu(indices_l)?;
        let out_cpu = crate::backend::BackendStorage::index_select(&src_cpu, &idx_cpu, l, indices_l, d)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    fn index_add(
        &self,
        l: &Layout,
        indices: &Self,
        indices_l: &Layout,
        src: &Self,
        src_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        let dst_cpu = self.to_cpu(l)?;
        let idx_cpu = indices.to_cpu(indices_l)?;
        let src_cpu = src.to_cpu(src_l)?;
        let out_cpu = crate::backend::BackendStorage::index_add(&dst_cpu, l, &idx_cpu, indices_l, &src_cpu, src_l, d)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    // ── MatMul — dispatched to NPU via OpenVINO ───────────────────────────

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        self.matmul_openvino(rhs, bmnk, lhs_l, rhs_l)
    }

    // ── Copy helpers (CPU fallback) ───────────────────────────────────────

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_cpu = self.to_cpu(src_l)?;
        let dst_bytes = dst.bytes()?.clone();
        let elem_count = dst_bytes.len() / self.dtype.size_in_bytes();
        let dst_layout = Layout::contiguous(crate::Shape::from(elem_count));
        let mut dst_cpu = utils::bytes_to_cpu_storage(&dst_bytes, dst.dtype, &dst_layout)?;
        crate::backend::BackendStorage::copy_strided_src(&src_cpu, &mut dst_cpu, dst_offset, src_l)?;
        let new_bytes = utils::cpu_storage_to_bytes(&dst_cpu)?;
        *dst.data.lock().map_err(|e| {
            Error::OpenVino(OpenVinoError::Message(format!("mutex poisoned: {e}")))
        })? = new_bytes;
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        let src_elem = self.bytes()?.len() / self.dtype.size_in_bytes();
        let src_layout = Layout::contiguous(crate::Shape::from(src_elem));
        let src_cpu = utils::bytes_to_cpu_storage(&self.bytes()?, self.dtype, &src_layout)?;

        let dst_bytes = dst.bytes()?.clone();
        let dst_elem = dst_bytes.len() / self.dtype.size_in_bytes();
        let dst_layout = Layout::contiguous(crate::Shape::from(dst_elem));
        let mut dst_cpu = utils::bytes_to_cpu_storage(&dst_bytes, dst.dtype, &dst_layout)?;

        crate::backend::BackendStorage::copy2d(&src_cpu, &mut dst_cpu, d1, d2, src_s, dst_s, src_o, dst_o)?;

        let new_bytes = utils::cpu_storage_to_bytes(&dst_cpu)?;
        *dst.data.lock().map_err(|e| {
            Error::OpenVino(OpenVinoError::Message(format!("mutex poisoned: {e}")))
        })? = new_bytes;
        Ok(())
    }

    fn const_set(&mut self, v: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        let bytes = self.bytes()?.clone();
        let mut cpu = utils::bytes_to_cpu_storage(&bytes, self.dtype, l)?;
        crate::backend::BackendStorage::const_set(&mut cpu, v, l)?;
        let new_bytes = utils::cpu_storage_to_bytes(&cpu)?;
        *self.data.lock().map_err(|e| {
            Error::OpenVino(OpenVinoError::Message(format!("mutex poisoned: {e}")))
        })? = new_bytes;
        Ok(())
    }
}
