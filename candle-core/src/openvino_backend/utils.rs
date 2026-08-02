//! Helpers for converting between raw byte buffers and [`CpuStorage`].
//!
//! The OpenVINO backend keeps tensor data as `Vec<u8>` (host-mapped bytes) so
//! it can hand raw pointers to the OpenVINO inference engine. These helpers
//! handle the conversion to/from candle's typed [`CpuStorage`].

use crate::cpu_backend::CpuStorage;
use crate::{DType, Error, Layout, Result};
use float8::F8E4M3;
use half::{bf16, f16};

use super::error::OpenVinoError;

// ── CpuStorage → raw bytes ───────────────────────────────────────────────────

/// Copy the contents of a [`CpuStorage`] into a flat `Vec<u8>`.
pub fn cpu_storage_to_bytes(s: &CpuStorage) -> Result<Vec<u8>> {
    match s {
        CpuStorage::U8(v) => Ok(v.iter().copied().collect()),
        CpuStorage::U32(v) => Ok(bytecast(v)),
        CpuStorage::I16(v) => Ok(bytecast(v)),
        CpuStorage::I32(v) => Ok(bytecast(v)),
        CpuStorage::I64(v) => Ok(bytecast(v)),
        CpuStorage::BF16(v) => Ok(bytecast(v)),
        CpuStorage::F16(v) => Ok(bytecast(v)),
        CpuStorage::F32(v) => Ok(bytecast(v)),
        CpuStorage::F64(v) => Ok(bytecast(v)),
        CpuStorage::F8E4M3(v) => {
            // F8E4M3 stores one byte per element.
            Ok(v.iter().map(|x| x.to_bits()).collect())
        }
        // Dummy / raw-byte types — copy as-is.
        CpuStorage::F6E2M3(v)
        | CpuStorage::F6E3M2(v)
        | CpuStorage::F4(v)
        | CpuStorage::F8E8M0(v) => Ok(v.clone()),
    }
}

// ── Raw bytes → CpuStorage ───────────────────────────────────────────────────

/// Reconstruct a [`CpuStorage`] from raw bytes, dtype, and a (possibly
/// non-contiguous) layout.  The bytes are interpreted as a *contiguous* buffer
/// in memory layout order; `layout` is used only to determine element count.
pub fn bytes_to_cpu_storage(bytes: &[u8], dtype: DType, layout: &Layout) -> Result<CpuStorage> {
    let elem_count = layout.shape().elem_count();
    match dtype {
        DType::U8 => {
            check_len(bytes, elem_count, 1)?;
            Ok(CpuStorage::U8(bytes[..elem_count].to_vec()))
        }
        DType::U32 => {
            check_len(bytes, elem_count, 4)?;
            Ok(CpuStorage::U32(cast_slice::<u32>(bytes, elem_count)))
        }
        DType::I16 => {
            check_len(bytes, elem_count, 2)?;
            Ok(CpuStorage::I16(cast_slice::<i16>(bytes, elem_count)))
        }
        DType::I32 => {
            check_len(bytes, elem_count, 4)?;
            Ok(CpuStorage::I32(cast_slice::<i32>(bytes, elem_count)))
        }
        DType::I64 => {
            check_len(bytes, elem_count, 8)?;
            Ok(CpuStorage::I64(cast_slice::<i64>(bytes, elem_count)))
        }
        DType::BF16 => {
            check_len(bytes, elem_count, 2)?;
            Ok(CpuStorage::BF16(cast_slice::<bf16>(bytes, elem_count)))
        }
        DType::F16 => {
            check_len(bytes, elem_count, 2)?;
            Ok(CpuStorage::F16(cast_slice::<f16>(bytes, elem_count)))
        }
        DType::F32 => {
            check_len(bytes, elem_count, 4)?;
            Ok(CpuStorage::F32(cast_slice::<f32>(bytes, elem_count)))
        }
        DType::F64 => {
            check_len(bytes, elem_count, 8)?;
            Ok(CpuStorage::F64(cast_slice::<f64>(bytes, elem_count)))
        }
        DType::F8E4M3 => {
            check_len(bytes, elem_count, 1)?;
            Ok(CpuStorage::F8E4M3(
                bytes[..elem_count]
                    .iter()
                    .map(|&b| F8E4M3::from_bits(b))
                    .collect(),
            ))
        }
        DType::F6E2M3 => {
            Ok(CpuStorage::F6E2M3(bytes[..elem_count].to_vec()))
        }
        DType::F6E3M2 => {
            Ok(CpuStorage::F6E3M2(bytes[..elem_count].to_vec()))
        }
        DType::F4 => {
            Ok(CpuStorage::F4(bytes[..elem_count].to_vec()))
        }
        DType::F8E8M0 => {
            Ok(CpuStorage::F8E8M0(bytes[..elem_count].to_vec()))
        }
    }
}

// ── Private helpers ──────────────────────────────────────────────────────────

fn check_len(bytes: &[u8], elem_count: usize, elem_size: usize) -> Result<()> {
    if bytes.len() < elem_count * elem_size {
        return Err(Error::OpenVino(OpenVinoError::Message(format!(
            "buffer too small: need {} bytes for {} elements of size {}, got {}",
            elem_count * elem_size,
            elem_count,
            elem_size,
            bytes.len(),
        ))));
    }
    Ok(())
}

/// Reinterpret a typed slice as bytes.
fn bytecast<T: Copy>(v: &[T]) -> Vec<u8> {
    let len = v.len() * std::mem::size_of::<T>();
    let mut out = vec![0u8; len];
    // Safety: we own `out` and write the exact number of bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            v.as_ptr() as *const u8,
            out.as_mut_ptr(),
            len,
        );
    }
    out
}

/// Reinterpret raw bytes as a `Vec<T>` of `count` elements.
fn cast_slice<T: Copy>(bytes: &[u8], count: usize) -> Vec<T> {
    let mut out: Vec<T> = Vec::with_capacity(count);
    // Safety: we checked length above; T has no padding / alignment issues for
    // the numeric primitives used here.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr() as *const T,
            out.as_mut_ptr(),
            count,
        );
        out.set_len(count);
    }
    out
}
