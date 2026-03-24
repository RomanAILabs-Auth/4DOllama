//! RMS normalization over the last axis of a 4D tensor (prototype).
//!
//! // 4D ENGINE NOTE: Full 4D RMSNorm couples w-axis curvature; this is a numerically stable baseline.

use ndarray::Array4;

pub fn rmsnorm_last_axis(x: &Array4<f32>, eps: f32) -> Array4<f32> {
    let mut y = x.clone();
    let sh = y.shape();
    let d = sh[3];
    if d == 0 {
        return y;
    }
    for i in 0..sh[0] {
        for j in 0..sh[1] {
            for k in 0..sh[2] {
                let mut sum = 0.0f32;
                for l in 0..d {
                    let v = y[[i, j, k, l]];
                    sum += v * v;
                }
                let rms = (sum / d as f32 + eps).sqrt();
                let scale = 1.0 / rms;
                for l in 0..d {
                    y[[i, j, k, l]] *= scale;
                }
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr4;

    #[test]
    fn unit_vector_rms_one() {
        let x = arr4(&[[[[1.0f32, 0.0, 0.0, 0.0]]]]);
        let y = rmsnorm_last_axis(&x, 1e-5f32);
        let mut sum = 0.0f32;
        for v in y.iter() {
            sum += v * v;
        }
        let rms = (sum / 4.0f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-4);
    }
}
