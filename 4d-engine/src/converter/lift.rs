//! Logical “lift” of flat GGUF tensors into 4D storage layout (planning + byte accounting).
//!
//! // 4D ENGINE NOTE: Native `.4dgguf` will persist the w-axis; here we only estimate memory & coherence width.

use super::gguf::{GgufSummary, TensorInfo};
use crate::tensor4d::Quaternion;
use serde::Serialize;

/// Default **w** extent for lifted tensors (quaternion-aligned hyper-slices).
pub const DEFAULT_W_EXTENT: u32 = 4;

#[derive(Clone, Debug, Serialize)]
pub struct LiftPreview {
    pub w_extent: u32,
    /// Estimated source bytes for all tensors (GGUF payload, not metadata).
    pub estimated_source_bytes: u64,
    /// Estimated f32-equivalent elements if each element were promoted (upper bound heuristic).
    pub estimated_fp32_elements: u64,
    pub tensor_count: usize,
    pub note: &'static str,
}

/// Returns element byte size for common GGML tensor types (GGUF wire `ggml_type`).
fn ggml_element_size(ty: u32) -> Option<u64> {
    match ty {
        0 => Some(4),   // F32
        1 => Some(2),   // F16
        2 => Some(4),   // Q4_0 block — not per-element; treat block loosely (handled below)
        3 => Some(4),   // Q4_1
        _ => None,
    }
}

fn tensor_raw_bytes(t: &TensorInfo) -> u64 {
    let n: u64 = t.shape.iter().product();
    if let Some(sz) = ggml_element_size(t.ggml_type) {
        // Q4_0 / Q4_1 are block-quantized; use rough scale for preview only.
        if t.ggml_type == 2 {
            // Q4_0: 32 elements per 18-byte block in ggml — approximate.
            let blocks = (n + 31) / 32;
            return blocks * 18;
        }
        if t.ggml_type == 3 {
            let blocks = (n + 31) / 32;
            return blocks * 20;
        }
        n.saturating_mul(sz)
    } else {
        // Unknown quant: assume 1 byte/elem lower bound for logging.
        n
    }
}

/// Build a lift preview from a GGUF summary (no tensor bytes read).
pub fn lift_preview_from_summary(sum: &GgufSummary) -> LiftPreview {
    let mut est_bytes: u64 = 0;
    let mut est_fp32: u64 = 0;
    for t in &sum.tensors {
        let b = tensor_raw_bytes(t);
        est_bytes = est_bytes.saturating_add(b);
        let n: u64 = t.shape.iter().product();
        est_fp32 = est_fp32.saturating_add(n);
    }
    LiftPreview {
        w_extent: DEFAULT_W_EXTENT,
        estimated_source_bytes: est_bytes,
        estimated_fp32_elements: est_fp32,
        tensor_count: sum.tensors.len(),
        note: "4D ENGINE NOTE: lift expands each logical tensor with a w-axis for hyper-contractions; full weight migration runs in the runner.",
    }
}

/// Pad to a multiple of 4, treat each quad as quaternion components, apply a small seeded rotation on the left.
pub fn lift_to_4d(mut weights: Vec<f32>) -> Vec<f32> {
    if weights.is_empty() {
        return weights;
    }
    while weights.len() % 4 != 0 {
        weights.push(0.0);
    }
    let mut h: u64 = 0x811c9dc5;
    for w in &weights {
        h ^= (w.to_bits() as u64).rotate_left(13);
        h = h.wrapping_mul(0x100000001b3);
    }
    let t = (h as f32) * 1e-9;
    let angle = t.sin() * 0.08;
    let ax = (h as f32 * 1e-6).sin();
    let ay = (h as f32 * 1e-7).cos();
    let az = (h as f32 * 1e-8).sin();
    let an = (ax * ax + ay * ay + az * az).sqrt().max(1e-6);
    let half = angle * 0.5;
    let r = Quaternion::new(
        half.cos(),
        ax / an * half.sin(),
        ay / an * half.sin(),
        az / an * half.sin(),
    )
    .normalize();
    let mut out = Vec::with_capacity(weights.len());
    for q in weights.chunks_exact(4) {
        let qv = Quaternion::new(q[0], q[1], q[2], q[3]).normalize();
        let m = r.mul(qv).normalize();
        out.extend_from_slice(&[m.w, m.i, m.j, m.k]);
    }
    out
}
