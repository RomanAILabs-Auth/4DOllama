//! Metal Shading Language references for 4D GEMM and attention; execution uses the shared parallel CPU path until MSL pipelines are linked at runtime.

pub const METAL_GEMM4D_MSL: &str = r#"// four_d_engine — gemm4d (reference Metal)
#include <metal_stdlib>
using namespace metal;

kernel void fd4_gemm4d_rows(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint mi = gid.x;
    uint ni = gid.y;
    if (mi >= (uint)M || ni >= (uint)N) return;
    float acc = 0.f;
    for (int ki = 0; ki < K; ++ki)
        acc += A[mi * K + ki] * B[ki * N + ni];
    C[mi * N + ni] = acc;
}
"#;

pub const METAL_ATTENTION4D_MSL: &str = r#"// four_d_engine — one causal attention row (reference Metal)
#include <metal_stdlib>
using namespace metal;
// Full quaternion softmax row scheduled per thread group in production builds.
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

/// Apple Metal stack is present (framework on disk).
pub fn metal_runtime_available() -> bool {
    if env_disables_gpu() {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}
