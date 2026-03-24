//! Quaternion-style rotary mixing on embedding quads (4D token features).

use crate::tensor4d::Quaternion;

/// Apply a position-dependent unit rotation in quaternion space: **r q r\*** (conjugation), one quad at a time.
/// `embeddings` is padded to a multiple of 4; each quad is a quaternion **q**; axis for **r** derives from `position`.
pub fn apply_quaternion_rope(embeddings: Vec<f32>, position: usize) -> Vec<f32> {
    let mut v = embeddings;
    while v.len() % 4 != 0 {
        v.push(0.0);
    }
    if v.is_empty() {
        return v;
    }
    let pos = position as f32;
    let theta = pos * 0.07;
    let axis = [
        (pos * 0.31 + 0.1).sin(),
        (pos * 0.27 + 0.2).cos(),
        (pos * 0.19 + 0.3).sin(),
    ];
    let an = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt().max(1e-6);
    let h = theta * 0.5;
    let r = Quaternion::new(
        h.cos(),
        axis[0] / an * h.sin(),
        axis[1] / an * h.sin(),
        axis[2] / an * h.sin(),
    )
    .normalize();
    let mut out = Vec::with_capacity(v.len());
    for q in v.chunks_exact(4) {
        let qv = Quaternion::new(q[0], q[1], q[2], q[3]).normalize();
        let rot = r.mul(qv).mul(r.conj()).normalize();
        out.extend_from_slice(&[rot.w, rot.i, rot.j, rot.k]);
    }
    out
}

/// Per-token positions `0..N` for consecutive embedding quaternions.
pub fn apply_quaternion_rope_sequence(embeddings: Vec<f32>) -> Vec<f32> {
    let mut v = embeddings;
    while v.len() % 4 != 0 {
        v.push(0.0);
    }
    if v.is_empty() {
        return v;
    }
    let mut out = Vec::with_capacity(v.len());
    for (i, q) in v.chunks_exact(4).enumerate() {
        let rot = apply_quaternion_rope(q.to_vec(), i);
        out.extend_from_slice(&rot);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_preserves_quad_count() {
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        let o = apply_quaternion_rope(v.clone(), 0);
        assert_eq!(o.len(), 4);
        let n = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2] + o[3] * o[3]).sqrt();
        assert!((n - 1.0).abs() < 1e-3);
    }
}
