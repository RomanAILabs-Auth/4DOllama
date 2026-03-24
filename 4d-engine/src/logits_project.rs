//! Stub LM-head: maps the final 4D state vector to `vocab_size` logits before flat softmax sampling.

use std::io::Write;

/// After attention, project `last4` and optional `lifted` weights into `out` (length ≥ vocab_size).
/// When `log_first`, prints a one-line debug trace to stderr on the first token (step 0).
pub fn project_logits_stub(
    last4: &[f32],
    lifted: &[f32],
    vocab_size: usize,
    step: u32,
    log_first: bool,
    out: &mut [f32],
) -> usize {
    let n = vocab_size.min(out.len());
    let l0 = last4.first().copied().unwrap_or(0.0);
    let l1 = last4.get(1).copied().unwrap_or(0.0);
    let l2 = last4.get(2).copied().unwrap_or(0.0);
    let l3 = last4.get(3).copied().unwrap_or(0.0);
    let s = step as f32;
    for i in 0..n {
        let fi = i as f32;
        let wi0 = (fi * 0.013 + s * 0.017).sin();
        let wi1 = (fi * 0.019 - s * 0.011).cos();
        let wi2 = (fi * 0.023 + s * 0.029).sin();
        let wi3 = (fi * 0.031 - s * 0.007).cos();
        let mut z = l0 * wi0 + l1 * wi1 + l2 * wi2 + l3 * wi3;
        if !lifted.is_empty() {
            z += 0.015 * lifted[i % lifted.len()];
        }
        z += 0.001 * (fi % 97.0);
        out[i] = z.tanh() * 2.0;
    }
    if log_first && n >= 5 && step == 0 {
        let msg = format!(
            "🔧 Logits shape: {} | First 5 logits: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            n, out[0], out[1], out[2], out[3], out[4]
        );
        let _ = writeln!(std::io::stderr(), "{}", msg);
    }
    n
}
