//! 4D computational engine for 4DOllama.
//!
//! // 4D ENGINE NOTE: The `converter` scans GGUF; `tensor4d` + `ops` execute hyper-ops; `ffi` exposes a stable C ABI.

pub mod converter;
mod ffi;
pub mod gpu;
pub mod ops;
pub mod tensor4d;
pub mod quaternion;
pub mod demo;
pub mod rope4d;
pub mod attention4d;
pub mod sampling4d;
pub mod gemm4d;
pub mod logits_project;

pub use attention4d::compute_4d_spacetime_attention;
pub use gemm4d::gemm4d;
pub use sampling4d::{sample_next_token_4d, sample_next_token_flat};
pub use logits_project::project_logits_stub;
pub use demo::compute_4d_demo;
pub use rope4d::{apply_quaternion_rope, apply_quaternion_rope_sequence};
pub use tensor4d::{gemm4d_w_contract, Quaternion, Tensor4D};
