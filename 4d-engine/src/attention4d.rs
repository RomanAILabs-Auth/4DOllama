//! Causal self-style attention over quaternion token rows; sequence index acts as the temporal axis in 4D spacetime.

use crate::rope4d::apply_quaternion_rope;
use crate::tensor4d::Quaternion;

#[inline]
fn quat_dot(a: Quaternion, b: Quaternion) -> f32 {
    a.w * b.w + a.i * b.i + a.j * b.j + a.k * b.k
}

#[inline]
fn quat_at(buf: &[f32], token: usize) -> Quaternion {
    let o = token * 4;
    Quaternion::new(buf[o], buf[o + 1], buf[o + 2], buf[o + 3]).normalize()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s <= 1e-12 {
        let u = 1.0 / logits.len() as f32;
        return vec![u; logits.len()];
    }
    exps.iter().map(|e| e / s).collect()
}

/// Causal attention over `seq_len` quaternion tokens (each row is 4 floats **w,i,j,k**).
/// Scores use the **R⁴ inner product** on quaternions plus a **timelike** penalty along the sequence axis (causal mask is implicit: **j ≤ i**).
/// Output rows are value mixtures, then a per-position **RoPE-style** mix via [`apply_quaternion_rope`].
pub fn compute_4d_spacetime_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
    let need = seq_len.saturating_mul(4);
    if need == 0 || q.len() < need || k.len() < need || v.len() < need {
        return Vec::new();
    }
    let mut out = vec![0.0f32; need];
    for i in 0..seq_len {
        let qi = quat_at(q, i);
        let mut logits = Vec::with_capacity(i + 1);
        for j in 0..=i {
            let kj = quat_at(k, j);
            let dot = quat_dot(qi, kj);
            let dt = (i - j) as f32;
            // Spacetime score: quaternion alignment minus timelike separation along the sequence (4th axis).
            let score = dot - 0.18 * dt * dt;
            logits.push(score * 1.25);
        }
        let w = softmax(&logits);
        let mut acc = [0.0f32; 4];
        for (j, &wj) in w.iter().enumerate() {
            let vj = quat_at(v, j);
            acc[0] += wj * vj.w;
            acc[1] += wj * vj.i;
            acc[2] += wj * vj.j;
            acc[3] += wj * vj.k;
        }
        let mixed = apply_quaternion_rope(
            vec![acc[0], acc[1], acc[2], acc[3]],
            i,
        );
        let oi = Quaternion::new(mixed[0], mixed[1], mixed[2], mixed[3]).normalize();
        out[i * 4..i * 4 + 4].copy_from_slice(&[oi.w, oi.i, oi.j, oi.k]);
    }
    out
}

/// Parallel hook for Metal/CUDA scheduling; same numerics as [`compute_4d_spacetime_attention`].
/// Raw-pointer `std::thread::scope` workers are not `Send` on all rustc targets; use serial path until split slices land.
pub fn compute_4d_spacetime_attention_parallel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
) -> Vec<f32> {
    compute_4d_spacetime_attention(q, k, v, seq_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_attention_shape() {
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let v = q.clone();
        let o = compute_4d_spacetime_attention(&q, &k, &v, 2);
        assert_eq!(o.len(), 8);
        let p = compute_4d_spacetime_attention_parallel(&q, &k, &v, 2);
        assert_eq!(o.len(), p.len());
        for i in 0..o.len() {
            assert!((o[i] - p[i]).abs() < 1e-3, "seq vs parallel mismatch at {}: {} vs {}", i, o[i], p[i]);
        }
    }
}
