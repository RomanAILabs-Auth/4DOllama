//! 4D ops: matmul4d, attention4d, rope4d, rmsnorm4d.

mod rmsnorm4d;

// 4D ENGINE NOTE: re-export tensor4d GEMM as the first real op.
pub use crate::tensor4d::gemm4d_w_contract;
pub use rmsnorm4d::rmsnorm_last_axis;
