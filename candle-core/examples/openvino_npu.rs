//! OpenVINO / Intel NPU backend smoke-test.
//!
//! Run with:
//!   cargo run -p candle-core --example openvino_npu --features openvino
//!
//! The example:
//!   1. Lists all devices visible to the OpenVINO runtime.
//!   2. Selects the NPU if present, otherwise falls back to CPU
//!      (useful on WSL2 / CI where the NPU driver is not loaded).
//!   3. Creates two small tensors, runs a batched matmul through the
//!      selected OpenVINO device, and compares the result element-by-
//!      element against the candle CPU reference.
//!   4. Runs a handful of extra element-wise and unary ops to exercise
//!      the CPU-fallback paths.

use candle_core::{DType, Device, Tensor};

// Absolute tolerance for f32 element comparisons.
const ATOL: f32 = 1e-4;

fn main() -> candle_core::Result<()> {
    // ── 1. Enumerate available OpenVINO devices ──────────────────────────
    let ov_core = openvino::Core::new()
        .map_err(|e| candle_core::Error::msg(format!("OpenVINO Core init failed: {e}")))?;

    let available = ov_core
        .available_devices()
        .map_err(|e| candle_core::Error::msg(format!("available_devices() failed: {e}")))?;

    println!("Available OpenVINO devices: {available:?}");

    // ── 2. Pick best device: NPU > GPU > CPU ─────────────────────────────
    let (ov_device_name, ov_ordinal) = {
        use openvino::DeviceType;
        let preferred: &[DeviceType] = &[DeviceType::NPU, DeviceType::GPU, DeviceType::CPU];
        let chosen = preferred
            .iter()
            .find(|&p| available.iter().any(|d| d == p));
        match chosen {
            Some(dev) => {
                let name = dev.as_ref().to_string();
                println!("Selected OpenVINO device: {name}");
                (name, 0usize)
            }
            None => {
                eprintln!("No usable OpenVINO device found in {available:?}");
                eprintln!("Falling back — nothing to test.");
                return Ok(());
            }
        }
    };

    // Drop the probe core; the candle device creates its own internal Core.
    drop(ov_core);

    // ── 3. Build the candle OpenVINO device ─────────────────────────────
    let ov_device = Device::new_openvino_with_device(&ov_device_name, ov_ordinal)?;
    println!("Candle device location: {:?}", ov_device.location());

    // ── 4. Matmul: [2, 3] × [3, 4]  (batched: b=1) ─────────────────────
    //
    // LHS data  (row-major):
    //   [[1, 2, 3],
    //    [4, 5, 6]]
    //
    // RHS data (row-major):
    //   [[7,  8,  9,  10],
    //    [11, 12, 13, 14],
    //    [15, 16, 17, 18]]
    //
    // Expected result:
    //   row 0: 1*7+2*11+3*15=74,  1*8+2*12+3*16=80,  1*9+2*13+3*17=86,  1*10+2*14+3*18=92
    //   row 1: 4*7+5*11+6*15=173, 4*8+5*12+6*16=188, 4*9+5*13+6*17=203, 4*10+5*14+6*18=218

    let lhs_data: Vec<f32> = vec![1., 2., 3., 4., 5., 6.];
    let rhs_data: Vec<f32> = vec![7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.];

    // CPU reference.
    let cpu = Device::Cpu;
    let lhs_cpu = Tensor::from_vec(lhs_data.clone(), (2, 3), &cpu)?;
    let rhs_cpu = Tensor::from_vec(rhs_data.clone(), (3, 4), &cpu)?;
    let out_cpu = lhs_cpu.matmul(&rhs_cpu)?;

    // OpenVINO device.
    let lhs_ov = Tensor::from_vec(lhs_data.clone(), (2, 3), &ov_device)?;
    let rhs_ov = Tensor::from_vec(rhs_data.clone(), (3, 4), &ov_device)?;
    let out_ov = lhs_ov.matmul(&rhs_ov)?;

    // Pull result back to CPU for comparison.
    let out_ov_cpu = out_ov.to_device(&cpu)?;

    let cpu_vals = out_cpu.to_vec2::<f32>()?;
    let ov_vals  = out_ov_cpu.to_vec2::<f32>()?;

    println!("\nMatmul [2,3] x [3,4]:");
    println!("  CPU result  : {cpu_vals:?}");
    println!("  OV  result  : {ov_vals:?}");

    assert_eq!(cpu_vals.len(), ov_vals.len(), "row count mismatch");
    let expected = [[74., 80., 86., 92.], [173., 188., 203., 218.]];
    for (r, (cpu_row, ov_row)) in cpu_vals.iter().zip(ov_vals.iter()).enumerate() {
        for (c, (cv, ov)) in cpu_row.iter().zip(ov_row.iter()).enumerate() {
            let diff = (cv - ov).abs();
            assert!(
                diff <= ATOL,
                "matmul mismatch at [{r},{c}]: cpu={cv}, ov={ov}, diff={diff}"
            );
            let exp = expected[r][c];
            let diff_exp = (cv - exp).abs();
            assert!(
                diff_exp <= ATOL,
                "matmul wrong value at [{r},{c}]: got={cv}, expected={exp}"
            );
        }
    }
    println!("  ✓ matmul result matches CPU reference");

    // ── 5. Batched matmul [2, 2, 3] × [2, 3, 4] ────────────────────────
    let lhs_b_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let rhs_b_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();

    let lhs_b_cpu = Tensor::from_vec(lhs_b_data.clone(), (2, 2, 3), &cpu)?;
    let rhs_b_cpu = Tensor::from_vec(rhs_b_data.clone(), (2, 3, 4), &cpu)?;
    let out_b_cpu = lhs_b_cpu.matmul(&rhs_b_cpu)?;

    let lhs_b_ov = Tensor::from_vec(lhs_b_data, (2, 2, 3), &ov_device)?;
    let rhs_b_ov = Tensor::from_vec(rhs_b_data, (2, 3, 4), &ov_device)?;
    let out_b_ov = lhs_b_ov.matmul(&rhs_b_ov)?;

    let out_b_ov_cpu = out_b_ov.to_device(&cpu)?;
    let b_cpu_vals = out_b_cpu.flatten_all()?.to_vec1::<f32>()?;
    let b_ov_vals  = out_b_ov_cpu.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(b_cpu_vals.len(), b_ov_vals.len());
    for (i, (cv, ov)) in b_cpu_vals.iter().zip(b_ov_vals.iter()).enumerate() {
        let diff = (cv - ov).abs();
        assert!(
            diff <= ATOL,
            "batched matmul mismatch at [{i}]: cpu={cv}, ov={ov}, diff={diff}"
        );
    }
    println!("  ✓ batched matmul [2,2,3] × [2,3,4] matches CPU reference");

    // ── 6. F16 matmul ───────────────────────────────────────────────────
    let lhs_f16 = lhs_cpu.to_dtype(DType::F16)?.to_device(&ov_device)?;
    let rhs_f16 = rhs_cpu.to_dtype(DType::F16)?.to_device(&ov_device)?;
    let out_f16_ov = lhs_f16.matmul(&rhs_f16)?;
    let out_f16_cpu = out_f16_ov.to_device(&cpu)?.to_dtype(DType::F32)?;
    let f16_vals = out_f16_cpu.to_vec2::<f32>()?;
    for (r, row) in f16_vals.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            let exp = expected[r][c];
            let diff = (v - exp).abs();
            // f16 has lower precision; use a wider tolerance.
            assert!(
                diff <= 1.0,
                "f16 matmul mismatch at [{r},{c}]: got={v}, expected={exp}, diff={diff}"
            );
        }
    }
    println!("  ✓ F16 matmul matches CPU reference (within f16 tolerance)");

    // ── 7. Element-wise ops (CPU-fallback paths) ─────────────────────────
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &ov_device)?;

    // Add
    let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2), &ov_device)?;
    let add_ov  = (&a + &b)?.to_device(&cpu)?.to_vec2::<f32>()?;
    let add_ref = [[11., 22.], [33., 44.]];
    println!("\nElement-wise ops:");
    for (r, row) in add_ov.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            let exp = add_ref[r][c];
            assert!((v - exp).abs() <= ATOL, "add mismatch [{r},{c}]: {v} != {exp}");
        }
    }
    println!("  ✓ add");

    // Mul
    let mul_ov  = (&a * &b)?.to_device(&cpu)?.to_vec2::<f32>()?;
    let mul_ref = [[10., 40.], [90., 160.]];
    for (r, row) in mul_ov.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            let exp = mul_ref[r][c];
            assert!((v - exp).abs() <= ATOL, "mul mismatch [{r},{c}]: {v} != {exp}");
        }
    }
    println!("  ✓ mul");

    // ReLU (unary)
    let neg = Tensor::from_vec(vec![-1.0f32, 2.0, -3.0, 4.0], (2, 2), &ov_device)?;
    let relu_ov  = neg.relu()?.to_device(&cpu)?.to_vec2::<f32>()?;
    let relu_ref = [[0., 2.], [0., 4.]];
    for (r, row) in relu_ov.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            let exp = relu_ref[r][c];
            assert!((v - exp).abs() <= ATOL, "relu mismatch [{r},{c}]: {v} != {exp}");
        }
    }
    println!("  ✓ relu");

    // Sum reduce
    let sum_ov = a.sum(1)?.to_device(&cpu)?.to_vec1::<f32>()?;
    let sum_ref = [3.0f32, 7.0];
    for (i, (v, &exp)) in sum_ov.iter().zip(sum_ref.iter()).enumerate() {
        assert!((v - exp).abs() <= ATOL, "sum mismatch [{i}]: {v} != {exp}");
    }
    println!("  ✓ sum (reduce)");

    // ── 8. Round-trip: CPU → OpenVINO → CPU ─────────────────────────────
    let cpu_tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &cpu)?;
    let ov_tensor  = cpu_tensor.to_device(&ov_device)?;
    let back       = ov_tensor.to_device(&cpu)?;
    let back_vals  = back.to_vec1::<f32>()?;
    assert_eq!(back_vals, vec![1.0f32, 2.0, 3.0], "round-trip mismatch");
    println!("\n  ✓ round-trip CPU → {ov_device_name} → CPU");

    // ── Summary ──────────────────────────────────────────────────────────
    println!("\n✓ All OpenVINO/{ov_device_name} tests passed.");
    if ov_device_name != "NPU" {
        println!(
            "\nNote: NPU device was not found on this system (running WSL2 or no driver loaded).\n\
             Tests ran on '{ov_device_name}' instead. To enable NPU support:\n\
             - On bare-metal Linux: install the intel-npu-driver package and load the\n\
               intel_npu kernel module.\n\
             - On Windows: ensure the Intel NPU driver is installed from\n\
               https://www.intel.com/content/www/us/en/download/794734/"
        );
    }

    Ok(())
}
