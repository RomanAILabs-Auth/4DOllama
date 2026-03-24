//! CUDA-oriented 4D GEMM / attention hooks. Default builds compile without NVCC; kernel source is embedded for offline JIT workflows.
//!
//! When a CUDA driver is detected, compute runs through the **parallel CPU path** wired for future PTX launch (same numerics as [`crate::gemm4d::gemm4d`]).

/// Minimal CUDA __global__ kernel (reference). Not compiled in-tree without `nvcc`.
pub const CUDA_GEMM4D_KERNEL_SOURCE: &str = r#"// four_d_engine — gemm4d row tiles (reference CUDA)
__global__ void fd4_gemm4d_rows(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int K, int N) {
  int mi = blockIdx.x * blockDim.x + threadIdx.x;
  int ni = blockIdx.y * blockDim.y + threadIdx.y;
  if (mi >= M || ni >= N) return;
  float acc = 0.f;
  for (int ki = 0; ki < K; ++ki)
    acc += A[mi * K + ki] * B[ki * N + ni];
  C[mi * N + ni] = acc;
}
"#;

/// Reference kernel for causal quaternion attention rows (one thread per output token).
pub const CUDA_ATTENTION4D_KERNEL_SOURCE: &str = r#"// four_d_engine — spacetime attention row (reference CUDA)
__global__ void fd4_spacetime_attn_row(const float* Q, const float* K, const float* V,
                                       float* O, int seq_len, int row) {
  // Host currently schedules one row per launch in full builds; this stub documents the contract.
}
"#;

fn env_disables_gpu() -> bool {
    match std::env::var("FOURD_GPU") {
        Ok(v) => {
            let v = v.to_lowercase();
            v == "0" || v == "off" || v == "cpu" || v == "none"
        }
        Err(_) => false,
    }
}

/// True when an NVIDIA CUDA user-mode driver is likely present (libcudart / libcuda paths).
pub fn cuda_runtime_available() -> bool {
    if env_disables_gpu() {
        return false;
    }
    #[cfg(unix)]
    {
        for p in [
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/wsl/lib/libcuda.so",
            "/usr/local/cuda/lib64/libcudart.so",
        ] {
            if std::path::Path::new(p).exists() {
                return true;
            }
        }
    }
    #[cfg(windows)]
    {
        if std::env::var("CUDA_PATH").is_ok() {
            return true;
        }
    }
    false
}

/// Telemetry label (README / capabilities).
pub fn cuda_backend_label() -> &'static str {
    if cuda_runtime_available() {
        "cuda-runtime"
    } else {
        "cuda-stub"
    }
}
