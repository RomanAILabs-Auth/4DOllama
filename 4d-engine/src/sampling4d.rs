//! 4D token sampling: logits grouped as quaternions, refined with spacetime attention, then softmax in projected space.
//! Flat sampler (`sample_next_token_flat`) is used for full-vocab stub LM-head output (temperature 0.7, top_k 40).

use crate::attention4d::compute_4d_spacetime_attention;
use crate::tensor4d::Quaternion;

/// Default stub autoregressive sampling (first run / standard path).
pub const STUB_SAMPLER_TEMP: f32 = 0.7;
pub const STUB_SAMPLER_TOP_K: usize = 40;

#[inline]
fn quat_dot(a: Quaternion, b: Quaternion) -> f32 {
    a.w * b.w + a.i * b.i + a.j * b.j + a.k * b.k
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s <= 1e-12 {
        let u = 1.0 / scores.len() as f32;
        return vec![u; scores.len()];
    }
    exps.iter().map(|e| e / s).collect()
}

/// Fixed readout direction in **ℝ⁴** (matches quaternion scalar + vector mix used in attention).
fn readout_direction() -> Quaternion {
    Quaternion::new(0.72, 0.45, -0.38, 0.29).normalize()
}

/// Mix logits as quaternion rows, run one causal **4D spacetime attention** when **≥2** rows,
/// project each row to a scalar, apply **top-k** and **temperature**, softmax, then deterministic sample.
/// Returns the winning **row index** (0 ..= ⌊len/4⌋ − 1); caller maps rows to token indices.
pub fn sample_next_token_4d(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    let temp = temperature.max(1e-5);
    let mut v: Vec<f32> = logits.to_vec();
    while v.len() % 4 != 0 {
        v.push(0.0);
    }
    let seq_len = v.len() / 4;
    if seq_len == 0 {
        return 0;
    }
    if seq_len >= 2 {
        v = compute_4d_spacetime_attention(&v, &v, &v, seq_len);
    }
    let readout = readout_direction();
    let mut proj: Vec<f32> = Vec::with_capacity(seq_len);
    for chunk in v.chunks_exact(4) {
        let q = Quaternion::new(chunk[0], chunk[1], chunk[2], chunk[3]).normalize();
        let s = quat_dot(q, readout) * 2.0 + 0.25 * (chunk[0] + chunk[1] + chunk[2] + chunk[3]);
        proj.push(s / temp);
    }
    if top_k > 0 && top_k < proj.len() {
        let mut order: Vec<usize> = (0..proj.len()).collect();
        order.sort_by(|&a, &b| proj[b].partial_cmp(&proj[a]).unwrap_or(std::cmp::Ordering::Equal));
        let thr = proj[order[top_k - 1]];
        for p in proj.iter_mut() {
            if *p < thr {
                *p = f32::NEG_INFINITY;
            }
        }
    }
    let probs = softmax(&proj);
    let idx = det_sample_index(&probs, logits, temperature, top_k) as u32;
    idx.min((seq_len - 1) as u32)
}

fn det_sample_index(probs: &[f32], logits: &[f32], temperature: f32, top_k: usize) -> usize {
    let mut h: u64 = 1469598103934665603;
    for &x in logits {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h ^= (top_k as u64).rotate_left(17);
    h ^= (temperature.to_bits() as u64).rotate_left(31);
    let r = ((h % 1_000_000) as f32) / 1_000_000.0;
    let mut c = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        c += p;
        if r < c {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Standard top-k softmax sample over **per-token** logits (full vocabulary).
pub fn sample_next_token_flat(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    let temp = temperature.max(1e-5);
    if logits.is_empty() {
        return 0;
    }
    let mut scaled: Vec<f32> = logits.iter().map(|x| x / temp).collect();
    if top_k > 0 && top_k < scaled.len() {
        let mut order: Vec<usize> = (0..scaled.len()).collect();
        order.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal));
        let thr = scaled[order[top_k - 1]];
        for s in scaled.iter_mut() {
            if *s < thr {
                *s = f32::NEG_INFINITY;
            }
        }
    }
    let probs = softmax(&scaled);
    let idx = det_sample_index(&probs, logits, temperature, top_k);
    (idx.min(logits.len().saturating_sub(1))) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_returns_valid_row() {
        let logits: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let id = sample_next_token_4d(&logits, 0.8, 4);
        assert!(id < 4);
    }

    #[test]
    fn flat_sample_in_range() {
        let logits: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        let id = sample_next_token_flat(&logits, STUB_SAMPLER_TEMP, STUB_SAMPLER_TOP_K);
        assert!(id < 100);
    }
}
