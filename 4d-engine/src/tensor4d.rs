//! Four-dimensional tensors and quaternion algebra for the 4D engine.
//!
//! // 4D ENGINE NOTE: RoPE-style mixing will call `Quaternion` rotation on 3-vectors instead of complex phase alone.

use ndarray::{Array3, Array4};

/// Hamilton quaternion (scalar **w**, bivectors **i**, **j**, **k**).
/// // 4D ENGINE NOTE: We store (w, i, j, k) to match quaternion literature; the fourth tensor axis is separate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    pub w: f32,
    pub i: f32,
    pub j: f32,
    pub k: f32,
}

impl Quaternion {
    pub fn new(w: f32, i: f32, j: f32, k: f32) -> Self {
        Self { w, i, j, k }
    }

    /// Identity rotation.
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            i: 0.0,
            j: 0.0,
            k: 0.0,
        }
    }

    pub fn norm_sq(self) -> f32 {
        self.w * self.w + self.i * self.i + self.j * self.j + self.k * self.k
    }

    pub fn normalize(self) -> Self {
        let n = self.norm_sq().sqrt();
        if n <= f32::EPSILON {
            return Self::identity();
        }
        Self {
            w: self.w / n,
            i: self.i / n,
            j: self.j / n,
            k: self.k / n,
        }
    }

    /// Quaternion product (Hamilton).
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.i * rhs.i - self.j * rhs.j - self.k * rhs.k,
            i: self.w * rhs.i + self.i * rhs.w + self.j * rhs.k - self.k * rhs.j,
            j: self.w * rhs.j - self.i * rhs.k + self.j * rhs.w + self.k * rhs.i,
            k: self.w * rhs.k + self.i * rhs.j - self.j * rhs.i + self.k * rhs.w,
        }
    }

    pub fn conj(self) -> Self {
        Self {
            w: self.w,
            i: -self.i,
            j: -self.j,
            k: -self.k,
        }
    }

    /// Inverse for unit quaternion; falls back to conjugate / norm².
    pub fn inv(self) -> Self {
        let n2 = self.norm_sq();
        if n2 <= f32::EPSILON {
            return Self::identity();
        }
        let c = self.conj();
        Self {
            w: c.w / n2,
            i: c.i / n2,
            j: c.j / n2,
            k: c.k / n2,
        }
    }

    /// Rotate a 3-vector **v** by this (unit) quaternion: q * (0,v) * q⁻¹, result as R³.
    /// // 4D ENGINE NOTE: This is the RoPE substitute path — spatial rotations instead of complex planes.
    pub fn rotate_vec3(self, v: [f32; 3]) -> [f32; 3] {
        let q = self.normalize();
        let p = Quaternion {
            w: 0.0,
            i: v[0],
            j: v[1],
            k: v[2],
        };
        let r = q.mul(p).mul(q.inv());
        [r.i, r.j, r.k]
    }
}

/// Owned 4D tensor in row-friendly layout: shape **(d0, d1, d2, d3)**.
/// // 4D ENGINE NOTE: **d3** is the **w-axis** used for hyper-contractions in `gemm4d_w_contract`.
#[derive(Clone, Debug)]
pub struct Tensor4D {
    pub data: Array4<f32>,
}

impl Tensor4D {
    pub fn zeros(shape: (usize, usize, usize, usize)) -> Self {
        Self {
            data: Array4::<f32>::zeros(shape),
        }
    }

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        let s = self.data.shape();
        (s[0], s[1], s[2], s[3])
    }
}

/// Batched “4D GEMM”: for each batch **b**, compute
/// `out[b, m, n] = Σ_k Σ_w A[b, m, k, w] * B[b, k, n, w]`.
///
/// // 4D ENGINE NOTE: The inner sum over **w** is the extra contraction that distinguishes this from flat matmul.
pub fn gemm4d_w_contract(a: &Array4<f32>, b: &Array4<f32>) -> Array3<f32> {
    let (b0, m, k, wa) = (
        a.shape()[0],
        a.shape()[1],
        a.shape()[2],
        a.shape()[3],
    );
    let (bb, kb, n, wb) = (
        b.shape()[0],
        b.shape()[1],
        b.shape()[2],
        b.shape()[3],
    );
    assert_eq!(b0, bb, "batch mismatch");
    assert_eq!(k, kb, "inner dim k mismatch");
    assert_eq!(
        wa, wb,
        "w-axis mismatch: 4D contraction requires identical w extent"
    );

    // Sequential contraction (correct ordering); parallel batch slices can layer on later.
    let mut out = Array3::<f32>::zeros((b0, m, n));
    for bi in 0..b0 {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    for wi in 0..wa {
                        acc += a[[bi, mi, ki, wi]] * b[[bi, ki, ni, wi]];
                    }
                }
                out[[bi, mi, ni]] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr4;

    #[test]
    fn quaternion_rotate_preserves_length_unit() {
        let angle = std::f32::consts::FRAC_PI_2;
        let half = angle * 0.5;
        // Unit quaternion: rotation by `angle` about +Y (i, j, k) = (0, 1, 0).
        let q = Quaternion::new(half.cos(), 0.0, half.sin(), 0.0).normalize();
        let v = [1.0f32, 0.0, 0.0];
        let r = q.rotate_vec3(v);
        let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "length not preserved");
    }

    #[test]
    fn gemm4d_w_contract_matches_manual() {
        // batch=1, m=2, k=2, w=2 — expectations from full double sum.
        let a = arr4(&[
            [[[1.0f32, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
        ]);
        let b = arr4(&[
            [[[1.0f32, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]],
        ]);
        let c = gemm4d_w_contract(&a, &b);
        assert!((c[[0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((c[[0, 0, 1]] - 1.0).abs() < 1e-5);
        assert!((c[[0, 1, 0]] - 1.0).abs() < 1e-5);
        assert!((c[[0, 1, 1]] - 1.0).abs() < 1e-5);
    }
}
