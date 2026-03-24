//! Deterministic quaternion rotation demo over flattened 3-vectors (4D pipeline hook).

use crate::tensor4d::Quaternion;

/// Applies a **unit quaternion rotation** to each consecutive triple in `input` (zero-padded to a multiple of 3).
/// The rotation axis is normalized from the first three samples; angle is derived from the sum of magnitudes.
pub fn compute_4d_demo(input: Vec<f32>) -> Vec<f32> {
    if input.is_empty() {
        return input;
    }
    let seed: f32 = input.iter().copied().fold(0.0_f32, |a, b| a + b.abs());
    let angle = (seed * 0.1).sin() * std::f32::consts::PI;
    let half = angle * 0.5;
    let mut ax = *input.first().unwrap_or(&1.0);
    let mut ay = *input.get(1).unwrap_or(&0.0);
    let mut az = *input.get(2).unwrap_or(&0.0);
    let n = (ax * ax + ay * ay + az * az).sqrt();
    if n <= 1e-6 {
        ax = 1.0;
        ay = 0.0;
        az = 0.0;
    } else {
        ax /= n;
        ay /= n;
        az /= n;
    }
    let q = Quaternion::new(
        half.cos(),
        ax * half.sin(),
        ay * half.sin(),
        az * half.sin(),
    )
    .normalize();

    let mut v = input;
    while v.len() % 3 != 0 {
        v.push(0.0);
    }
    let mut out = Vec::with_capacity(v.len());
    for chunk in v.chunks_exact(3) {
        let r = q.rotate_vec3([chunk[0], chunk[1], chunk[2]]);
        out.extend_from_slice(&r);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_nonempty() {
        let o = compute_4d_demo(vec![1.0, 0.0, 0.0]);
        assert_eq!(o.len(), 3);
        let len = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-4);
    }
}
