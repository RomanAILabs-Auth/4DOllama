//! Row-major **C = A × B** with **A** [m×k], **B** [k×n], **C** [m×n].
//! Inner dimension is accumulated in **4-wide quaternion blocks** (same ℝ⁴ inner product as [`crate::attention4d`]).

use crate::tensor4d::Quaternion;

#[inline]
fn quat_dot(a: Quaternion, b: Quaternion) -> f32 {
    a.w * b.w + a.i * b.i + a.j * b.j + a.k * b.k
}

#[inline]
pub(crate) fn gemm4d_accumulate(a: &[f32], b: &[f32], mi: usize, ni: usize, k: usize, n: usize) -> f32 {
    let k4 = (k / 4) * 4;
    let mut acc = 0.0f32;
    let mut ki = 0usize;
    while ki < k4 {
        let qa = Quaternion::new(
            a[mi * k + ki],
            a[mi * k + ki + 1],
            a[mi * k + ki + 2],
            a[mi * k + ki + 3],
        )
        .normalize();
        let qb = Quaternion::new(
            b[ki * n + ni],
            b[(ki + 1) * n + ni],
            b[(ki + 2) * n + ni],
            b[(ki + 3) * n + ni],
        )
        .normalize();
        acc += quat_dot(qa, qb);
        ki += 4;
    }
    while ki < k {
        acc += a[mi * k + ki] * b[ki * n + ni];
        ki += 1;
    }
    acc
}

/// Native 4D-flavored GEMM: along **k**, groups of 4 use normalized quaternion dot; remainder uses scalar multiply.
pub fn gemm4d(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let need_a = m.saturating_mul(k);
    let need_b = k.saturating_mul(n);
    let need_c = m.saturating_mul(n);
    if a.len() < need_a || b.len() < need_b || c.len() < need_c {
        return;
    }
    for mi in 0..m {
        for ni in 0..n {
            c[mi * n + ni] = gemm4d_accumulate(a, b, mi, ni, k, n);
        }
    }
}

/// Multi-threaded GEMM (used when Metal/CUDA runtime is detected — same numerics as [`gemm4d`]).
pub fn gemm4d_parallel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let need_a = m.saturating_mul(k);
    let need_b = k.saturating_mul(n);
    let need_c = m.saturating_mul(n);
    if a.len() < need_a || b.len() < need_b || c.len() < need_c {
        return;
    }
    let workers = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
        .min(16)
        .max(1);
    if workers <= 1 || m < 4 {
        gemm4d(a, b, c, m, k, n);
        return;
    }
    let chunk = (m + workers - 1) / workers;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();
    std::thread::scope(|s| {
        for t in 0..workers {
            let r0 = t * chunk;
            let r1 = (r0 + chunk).min(m);
            if r0 >= r1 {
                continue;
            }
            unsafe {
                let a = std::slice::from_raw_parts(ap, need_a);
                let b = std::slice::from_raw_parts(bp, need_b);
                s.spawn(move || {
                    for mi in r0..r1 {
                        for ni in 0..n {
                            let v = gemm4d_accumulate(a, b, mi, ni, k, n);
                            *cp.add(mi * n + ni) = v;
                        }
                    }
                });
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm4d_scalar_k1() {
        let a = vec![2.0f32];
        let b = vec![3.0f32];
        let mut c = vec![0.0f32];
        gemm4d(&a, &b, &mut c, 1, 1, 1);
        assert!((c[0] - 6.0).abs() < 1e-5);
    }
}
