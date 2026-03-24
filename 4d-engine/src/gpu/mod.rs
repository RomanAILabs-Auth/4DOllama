//! GPU backends: CUDA + Metal **detection** and embedded kernel sources; compute uses a **multi-threaded CPU path**
//! (same numerics as scalar [`crate::gemm4d`] / [`crate::attention4d`]) until device kernels are linked.

mod cuda4d;
mod metal4d;

pub use cuda4d::{
    cuda_backend_label, cuda_runtime_available, CUDA_ATTENTION4D_KERNEL_SOURCE, CUDA_GEMM4D_KERNEL_SOURCE,
};
pub use metal4d::{
    metal_runtime_available, METAL_ATTENTION4D_MSL, METAL_GEMM4D_MSL,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GpuBackend {
    Cpu,
    Cuda,
    Metal,
}

/// Active backend: **Metal** on macOS when the framework is present, else **CUDA** when a driver library is found.
pub fn active_backend() -> GpuBackend {
    #[cfg(target_os = "macos")]
    if metal_runtime_available() {
        return GpuBackend::Metal;
    }
    if cuda_runtime_available() {
        return GpuBackend::Cuda;
    }
    GpuBackend::Cpu
}

#[inline]
pub fn use_parallel_gpu_path() -> bool {
    active_backend() != GpuBackend::Cpu
}

pub fn gemm4d_dispatch(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if use_parallel_gpu_path() {
        crate::gemm4d::gemm4d_parallel(a, b, c, m, k, n);
    } else {
        crate::gemm4d::gemm4d(a, b, c, m, k, n);
    }
}

pub fn spacetime_attention_dispatch(q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
    if use_parallel_gpu_path() {
        crate::attention4d::compute_4d_spacetime_attention_parallel(q, k, v, seq_len)
    } else {
        crate::attention4d::compute_4d_spacetime_attention(q, k, v, seq_len)
    }
}
