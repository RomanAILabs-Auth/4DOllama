//! GGUF scan + 4D lift planning.

mod gguf;
mod lift;

pub use crate::gemm4d::gemm4d;
pub use crate::rope4d::{apply_quaternion_rope, apply_quaternion_rope_sequence};
pub use gguf::{scan_gguf, GgufError, GgufSummary};
pub use lift::{lift_preview_from_summary, lift_to_4d, LiftPreview, DEFAULT_W_EXTENT};

use serde::Serialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Serialize)]
pub struct InspectReport {
    pub engine: &'static str,
    pub path: String,
    pub gguf: GgufSummary,
    pub lift_preview: LiftPreview,
}

/// Scan a GGUF file from disk and produce JSON-ready inspection report.
pub fn inspect_gguf_path(path: &Path) -> Result<InspectReport, GgufError> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let (gguf, _) = scan_gguf_with_layout(&mut r)?;
    let lift_preview = lift_preview_from_summary(&gguf);
    Ok(InspectReport {
        engine: "four_d_engine",
        path: path.display().to_string(),
        gguf,
        lift_preview,
    })
}
