//! OpenVINO backend for Candle — runs operations on the Intel NPU (or any
//! OpenVINO-supported device such as CPU / GPU / NPU).
//!
//! # Design
//!
//! OpenVINO is a graph-level inference engine: you compile a model once and
//! then run `infer_request.infer()`.  Candle operates on individual tensor
//! operations (matmul, elementwise, …).
//!
//! This backend bridges the two worlds with the following approach:
//!
//! * All **storage** lives on the host as a plain `Vec<u8>` of raw bytes (same
//!   layout as [`CpuStorage`]).  Copies to/from CPU are therefore zero-cost.
//! * **matmul** (the primary NPU-accelerated op) builds a tiny OpenVINO IR XML
//!   model on the fly, loads it via `Core::read_model_from_buffer`, compiles it
//!   to the target device (NPU/GPU/CPU), and executes inference.
//! * All **other operations** fall back transparently to the [`CpuStorage`]
//!   path by round-tripping through [`to_cpu_storage`] /
//!   [`storage_from_cpu_storage_owned`].

use std::sync::{Arc, Mutex};

use openvino::{
    Core, DeviceType, ElementType, InferRequest, Shape as OvShape, Tensor as OvTensor,
};

use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{DType, Error, Layout, Result, Shape, WithDType};

// Re-use the always-available OpenVinoError from the dummy backend so that
// Error::OpenVino(e) always refers to one single type regardless of feature flags.
use crate::dummy_openvino_backend::OpenVinoError;

pub mod utils;

// ── Error helpers ────────────────────────────────────────────────────────────

/// Convert any `Display` OpenVINO error into our wrapper type.
fn ov_err(e: impl std::fmt::Display) -> Error {
    Error::OpenVino(OpenVinoError::from(format!("{e}")))
}

fn mutex_err(e: impl std::fmt::Display) -> Error {
    Error::OpenVino(OpenVinoError::Message(format!("mutex poisoned: {e}")))
}

// ── Device ───────────────────────────────────────────────────────────────────

/// Shared state for a single OpenVINO hardware target.
struct Inner {
    /// The OpenVINO runtime core. `Core` is not `Sync`, so wrap in a `Mutex`.
    core: Mutex<Core>,
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
        let core = Core::new().map_err(|e| ov_err(e))?;
        Ok(Self(Arc::new(Inner {
            core: Mutex::new(core),
            device_name: device_name.into(),
            ordinal,
        })))
    }

    /// The OpenVINO device name (e.g. `"NPU"`, `"GPU"`, `"CPU"`).
    pub fn device_name(&self) -> &str {
        &self.0.device_name
    }

    /// Build an IR XML string for a batched MatMul of shape [B,M,K] × [B,K,N].
    fn matmul_ir_xml(b: usize, m: usize, k: usize, n: usize, precision: &str, elem_type: &str) -> String {
        format!(
            r#"<?xml version="1.0"?>
<net name="candle_matmul" version="11">
    <layers>
        <layer id="0" name="lhs" type="Parameter" version="opset1">
            <data element_type="{elem_type}" shape="{b},{m},{k}"/>
            <output>
                <port id="0" precision="{precision}">
                    <dim>{b}</dim><dim>{m}</dim><dim>{k}</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="rhs" type="Parameter" version="opset1">
            <data element_type="{elem_type}" shape="{b},{k},{n}"/>
            <output>
                <port id="0" precision="{precision}">
                    <dim>{b}</dim><dim>{k}</dim><dim>{n}</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="matmul_op" type="MatMul" version="opset1">
            <data transpose_a="false" transpose_b="false"/>
            <input>
                <port id="0"><dim>{b}</dim><dim>{m}</dim><dim>{k}</dim></port>
                <port id="1"><dim>{b}</dim><dim>{k}</dim><dim>{n}</dim></port>
            </input>
            <output>
                <port id="2" precision="{precision}">
                    <dim>{b}</dim><dim>{m}</dim><dim>{n}</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="result" type="Result" version="opset1">
            <input>
                <port id="0"><dim>{b}</dim><dim>{m}</dim><dim>{n}</dim></port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>"#
        )
    }

    fn compile_and_run_matmul(
        &self,
        b: usize, m: usize, k: usize, n: usize,
        ov_type: ElementType,
        precision: &str,
        elem_type: &str,
        lhs_bytes: &[u8],
        rhs_bytes: &[u8],
    ) -> Result<Vec<u8>> {
        let xml = Self::matmul_ir_xml(b, m, k, n, precision, elem_type);

        let mut core = self.0.core.lock().map_err(|e| mutex_err(e))?;

        // Load model from the in-memory IR XML (no weights blob needed).
        let mut model = core
            .read_model_from_buffer(xml.as_bytes(), None)
            .map_err(|e| ov_err(e))?;

        // Compile to the target device.
        let device = DeviceType::from(self.0.device_name.as_str());
        let mut compiled = core
            .compile_model(&mut model, device)
            .map_err(|e| ov_err(e))?;

        let mut req: InferRequest = compiled
            .create_infer_request()
            .map_err(|e| ov_err(e))?;

        // Build input tensors and copy data in.
        let lhs_shape = OvShape::new(&[b as i64, m as i64, k as i64])
            .map_err(|e| ov_err(e))?;
        let rhs_shape = OvShape::new(&[b as i64, k as i64, n as i64])
            .map_err(|e| ov_err(e))?;

        let mut lhs_ov = OvTensor::new(ov_type, &lhs_shape).map_err(|e| ov_err(e))?;
        lhs_ov
            .get_raw_data_mut()
            .map_err(|e| ov_err(e))?
            .copy_from_slice(lhs_bytes);

        let mut rhs_ov = OvTensor::new(ov_type, &rhs_shape).map_err(|e| ov_err(e))?;
        rhs_ov
            .get_raw_data_mut()
            .map_err(|e| ov_err(e))?
            .copy_from_slice(rhs_bytes);

        req.set_input_tensor_by_index(0, &lhs_ov)
            .map_err(|e| ov_err(e))?;
        req.set_input_tensor_by_index(1, &rhs_ov)
            .map_err(|e| ov_err(e))?;

        req.infer().map_err(|e| ov_err(e))?;

        let out = req.get_output_tensor().map_err(|e| ov_err(e))?;
        Ok(out.get_raw_data().map_err(|e| ov_err(e))?.to_vec())
    }
}

impl BackendDevice for OpenVinoDevice {
    type Storage = OpenVinoStorage;

    /// `ordinal` maps to the Nth NPU. Uses `"NPU"` as the target device.
    fn new(ordinal: usize) -> Result<Self> {
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
        Arc::ptr_eq(&self.0, &other.0)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let data = vec![0u8; shape.elem_count() * dtype.size_in_bytes()];
        Ok(OpenVinoStorage {
            data: Arc::new(Mutex::new(data)),
            dtype,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let mut data = Vec::with_capacity(shape.elem_count() * dtype.size_in_bytes());
        data.set_len(shape.elem_count() * dtype.size_in_bytes());
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
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        crate::bail!("get_current_seed is not supported for the OpenVINO backend")
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

// ── Storage ──────────────────────────────────────────────────────────────────

pub struct OpenVinoStorage {
    pub(crate) data: Arc<Mutex<Vec<u8>>>,
    pub(crate) dtype: DType,
    pub(crate) device: OpenVinoDevice,
}

impl std::fmt::Debug for OpenVinoStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpenVinoStorage(dtype={:?})", self.dtype)
    }
}

impl OpenVinoStorage {
    fn lock_bytes(&self) -> Result<std::sync::MutexGuard<'_, Vec<u8>>> {
        self.data.lock().map_err(|e| mutex_err(e))
    }

    fn to_cpu(&self, layout: &Layout) -> Result<CpuStorage> {
        let bytes = self.lock_bytes()?;
        utils::bytes_to_cpu_storage(&bytes, self.dtype, layout)
    }

    /// Replace the internal byte buffer from a [`CpuStorage`].
    /// Used by in-place operations that round-trip through CPU.
    pub fn replace_storage_from_cpu(&mut self, cpu: CpuStorage) -> Result<()> {
        let new_bytes = utils::cpu_storage_to_bytes(&cpu)?;
        *self.lock_bytes()? = new_bytes;
        self.dtype = cpu.dtype();
        Ok(())
    }

    // ── CPU round-trip helpers ────────────────────────────────────────────

    fn cpu_op<F>(&self, layout: &Layout, f: F) -> Result<Self>
    where
        F: FnOnce(CpuStorage) -> Result<CpuStorage>,
    {
        let out_cpu = f(self.to_cpu(layout)?)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }

    // ── NPU matmul ────────────────────────────────────────────────────────

    fn matmul_openvino(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        // Map dtype to OpenVINO ElementType and string representations.
        let (ov_type, precision, elem_type_str) = match self.dtype {
            DType::F32  => (ElementType::F32,  "FP32", "f32"),
            DType::F16  => (ElementType::F16,  "FP16", "f16"),
            DType::BF16 => (ElementType::Bf16, "BF16", "bf16"),
            // Integer / quantised types: fall back to CPU.
            _ => return self.matmul_cpu(rhs, (b, m, n, k), lhs_l, rhs_l),
        };

        let lhs_cpu = self.to_cpu(lhs_l)?;
        let rhs_cpu = rhs.to_cpu(rhs_l)?;
        let lhs_bytes = utils::cpu_storage_to_bytes(&lhs_cpu)?;
        let rhs_bytes = utils::cpu_storage_to_bytes(&rhs_cpu)?;

        let out_bytes = self.device.compile_and_run_matmul(
            b, m, k, n,
            ov_type, precision, elem_type_str,
            &lhs_bytes, &rhs_bytes,
        )?;

        Ok(OpenVinoStorage {
            data: Arc::new(Mutex::new(out_bytes)),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn matmul_cpu(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let lhs_cpu = self.to_cpu(lhs_l)?;
        let rhs_cpu = rhs.to_cpu(rhs_l)?;
        let out_cpu = BackendStorage::matmul(&lhs_cpu, &rhs_cpu, bmnk, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }
}

impl BackendStorage for OpenVinoStorage {
    type Device = OpenVinoDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
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
        let bytes = self.lock_bytes()?;
        let elem_count = bytes.len() / self.dtype.size_in_bytes();
        let layout = Layout::contiguous(crate::Shape::from(elem_count));
        utils::bytes_to_cpu_storage(&bytes, self.dtype, &layout)
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::affine(&cpu, l, mul, add))
    }

    fn powf(&self, l: &Layout, alpha: f64) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::powf(&cpu, l, alpha))
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::elu(&cpu, l, alpha))
    }

    fn reduce_op(&self, op: ReduceOp, l: &Layout, dims: &[usize]) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::reduce_op(&cpu, op, l, dims))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let lhs_cpu = self.to_cpu(lhs_l)?;
        let rhs_cpu = rhs.to_cpu(rhs_l)?;
        let out = BackendStorage::cmp(&lhs_cpu, op, &rhs_cpu, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::to_dtype(&cpu, l, dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::unary_impl::<B>(&cpu, l))
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let lhs_cpu = self.to_cpu(lhs_l)?;
        let rhs_cpu = rhs.to_cpu(rhs_l)?;
        let out = BackendStorage::binary_impl::<B>(&lhs_cpu, &rhs_cpu, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn where_cond(&self, l: &Layout, t: &Self, t_l: &Layout, f: &Self, f_l: &Layout) -> Result<Self> {
        let cond = self.to_cpu(l)?;
        let t_cpu = t.to_cpu(t_l)?;
        let f_cpu = f.to_cpu(f_l)?;
        let out = BackendStorage::where_cond(&cond, l, &t_cpu, t_l, &f_cpu, f_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv1d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConv1D) -> Result<Self> {
        let inp = self.to_cpu(l)?;
        let ker = kernel.to_cpu(kernel_l)?;
        let out = BackendStorage::conv1d(&inp, l, &ker, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose1d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConvTranspose1D) -> Result<Self> {
        let inp = self.to_cpu(l)?;
        let ker = kernel.to_cpu(kernel_l)?;
        let out = BackendStorage::conv_transpose1d(&inp, l, &ker, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv2d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConv2D) -> Result<Self> {
        let inp = self.to_cpu(l)?;
        let ker = kernel.to_cpu(kernel_l)?;
        let out = BackendStorage::conv2d(&inp, l, &ker, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose2d(&self, l: &Layout, kernel: &Self, kernel_l: &Layout, params: &crate::conv::ParamsConvTranspose2D) -> Result<Self> {
        let inp = self.to_cpu(l)?;
        let ker = kernel.to_cpu(kernel_l)?;
        let out = BackendStorage::conv_transpose2d(&inp, l, &ker, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn avg_pool2d(&self, l: &Layout, ks: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::avg_pool2d(&cpu, l, ks, stride))
    }

    fn max_pool2d(&self, l: &Layout, ks: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::max_pool2d(&cpu, l, ks, stride))
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::upsample_nearest1d(&cpu, l, sz))
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::upsample_nearest2d(&cpu, l, h, w))
    }

    fn upsample_bilinear2d(&self, l: &Layout, h: usize, w: usize, align_corners: bool, scale_h: Option<f64>, scale_w: Option<f64>) -> Result<Self> {
        self.cpu_op(l, |cpu| BackendStorage::upsample_bilinear2d(&cpu, l, h, w, align_corners, scale_h, scale_w))
    }

    fn gather(&self, l: &Layout, indices: &Self, indices_l: &Layout, d: usize) -> Result<Self> {
        let src = self.to_cpu(l)?;
        let idx = indices.to_cpu(indices_l)?;
        let out = BackendStorage::gather(&src, l, &idx, indices_l, d)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn scatter_set(&mut self, l: &Layout, indices: &Self, indices_l: &Layout, src: &Self, src_l: &Layout, d: usize) -> Result<()> {
        let bytes = self.lock_bytes()?.clone();
        let mut dst_cpu = utils::bytes_to_cpu_storage(&bytes, self.dtype, l)?;
        let idx_cpu = indices.to_cpu(indices_l)?;
        let src_cpu = src.to_cpu(src_l)?;
        BackendStorage::scatter_set(&mut dst_cpu, l, &idx_cpu, indices_l, &src_cpu, src_l, d)?;
        *self.lock_bytes()? = utils::cpu_storage_to_bytes(&dst_cpu)?;
        Ok(())
    }

    fn scatter_add_set(&mut self, l: &Layout, indices: &Self, indices_l: &Layout, src: &Self, src_l: &Layout, d: usize) -> Result<()> {
        let bytes = self.lock_bytes()?.clone();
        let mut dst_cpu = utils::bytes_to_cpu_storage(&bytes, self.dtype, l)?;
        let idx_cpu = indices.to_cpu(indices_l)?;
        let src_cpu = src.to_cpu(src_l)?;
        BackendStorage::scatter_add_set(&mut dst_cpu, l, &idx_cpu, indices_l, &src_cpu, src_l, d)?;
        *self.lock_bytes()? = utils::cpu_storage_to_bytes(&dst_cpu)?;
        Ok(())
    }

    fn index_select(&self, indices: &Self, l: &Layout, indices_l: &Layout, d: usize) -> Result<Self> {
        let src = self.to_cpu(l)?;
        let idx = indices.to_cpu(indices_l)?;
        let out = BackendStorage::index_select(&src, &idx, l, indices_l, d)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn index_add(&self, l: &Layout, indices: &Self, indices_l: &Layout, src: &Self, src_l: &Layout, d: usize) -> Result<Self> {
        let dst = self.to_cpu(l)?;
        let idx = indices.to_cpu(indices_l)?;
        let src_cpu = src.to_cpu(src_l)?;
        let out = BackendStorage::index_add(&dst, l, &idx, indices_l, &src_cpu, src_l, d)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn matmul(&self, rhs: &Self, bmnk: (usize, usize, usize, usize), lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        self.matmul_openvino(rhs, bmnk, lhs_l, rhs_l)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_cpu = self.to_cpu(src_l)?;
        let dst_bytes = dst.lock_bytes()?.clone();
        let dst_elem = dst_bytes.len() / dst.dtype.size_in_bytes();
        let dst_layout = Layout::contiguous(crate::Shape::from(dst_elem));
        let mut dst_cpu = utils::bytes_to_cpu_storage(&dst_bytes, dst.dtype, &dst_layout)?;
        BackendStorage::copy_strided_src(&src_cpu, &mut dst_cpu, dst_offset, src_l)?;
        *dst.lock_bytes()? = utils::cpu_storage_to_bytes(&dst_cpu)?;
        Ok(())
    }

    fn copy2d(&self, dst: &mut Self, d1: usize, d2: usize, src_s: usize, dst_s: usize, src_o: usize, dst_o: usize) -> Result<()> {
        let src_bytes = self.lock_bytes()?.clone();
        let src_elem = src_bytes.len() / self.dtype.size_in_bytes();
        let src_cpu = utils::bytes_to_cpu_storage(&src_bytes, self.dtype, &Layout::contiguous(crate::Shape::from(src_elem)))?;

        let dst_bytes = dst.lock_bytes()?.clone();
        let dst_elem = dst_bytes.len() / dst.dtype.size_in_bytes();
        let mut dst_cpu = utils::bytes_to_cpu_storage(&dst_bytes, dst.dtype, &Layout::contiguous(crate::Shape::from(dst_elem)))?;

        BackendStorage::copy2d(&src_cpu, &mut dst_cpu, d1, d2, src_s, dst_s, src_o, dst_o)?;
        *dst.lock_bytes()? = utils::cpu_storage_to_bytes(&dst_cpu)?;
        Ok(())
    }

    fn const_set(&mut self, v: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        let bytes = self.lock_bytes()?.clone();
        let mut cpu = utils::bytes_to_cpu_storage(&bytes, self.dtype, l)?;
        BackendStorage::const_set(&mut cpu, v, l)?;
        *self.lock_bytes()? = utils::cpu_storage_to_bytes(&cpu)?;
        self.dtype = cpu.dtype();
        Ok(())
    }
}
