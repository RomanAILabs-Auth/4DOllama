//! Minimal GGUF v2/v3 metadata scanner (no full weight decode).
//!
//! // 4D ENGINE NOTE: We only need tensor inventory + architecture KV before lifting weights in later phases.

use byteorder::{LittleEndian, ReadBytesExt};
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid magic (expected GGUF)")]
    BadMagic,
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("invalid utf-8 in GGUF string")]
    BadString,
    #[error("unknown GGUF value type {0}")]
    UnknownType(u32),
    #[error("no F32 tensor found for weight sample")]
    NoF32Tensor,
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct GgufSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub architecture: Option<String>,
    pub tensors: Vec<TensorInfo>,
}

fn read_string<R: Read>(r: &mut R) -> Result<String, GgufError> {
    let n = r.read_u64::<LittleEndian>()?;
    let mut buf = vec![0u8; n as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::BadString)
}

fn skip_value<R: Read>(r: &mut R, ty: u32) -> Result<(), GgufError> {
    // GGUF_TYPE_* order matches llama.cpp / GGUF spec.
    match ty {
        0 => {
            r.read_u8()?;
        }
        1 => {
            r.read_i8()?;
        }
        2 => {
            r.read_u16::<LittleEndian>()?;
        }
        3 => {
            r.read_i16::<LittleEndian>()?;
        }
        4 => {
            r.read_u32::<LittleEndian>()?;
        }
        5 => {
            r.read_i32::<LittleEndian>()?;
        }
        6 => {
            r.read_f32::<LittleEndian>()?;
        }
        7 => {
            r.read_u8()?;
        } // BOOL
        8 => {
            let _ = read_string(r)?;
        }
        9 => {
            let et = r.read_u32::<LittleEndian>()?;
            let ne = r.read_u64::<LittleEndian>()?;
            for _ in 0..ne {
                skip_value(r, et)?;
            }
        }
        10 => {
            r.read_u64::<LittleEndian>()?;
        }
        11 => {
            r.read_i64::<LittleEndian>()?;
        }
        12 => {
            r.read_f64::<LittleEndian>()?;
        }
        _ => return Err(GgufError::UnknownType(ty)),
    }
    Ok(())
}

/// Total parameter elements (sum of tensor element counts).
pub fn estimate_param_elements(summary: &GgufSummary) -> u64 {
    summary
        .tensors
        .iter()
        .map(|t| t.shape.iter().product::<u64>())
        .sum()
}

/// Scan GGUF and return tensor metadata plus byte offset where tensor payload begins (32-byte aligned).
pub fn scan_gguf_with_layout<R: Read + Seek>(r: &mut R) -> Result<(GgufSummary, u64), GgufError> {
    r.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }
    let version = r.read_u32::<LittleEndian>()?;
    if !(2..=4).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = r.read_u64::<LittleEndian>()?;
    let kv_count = r.read_u64::<LittleEndian>()?;

    let mut arch: Option<String> = None;

    for _ in 0..kv_count {
        let key = read_string(r)?;
        let ty = r.read_u32::<LittleEndian>()?;
        if key == "general.architecture" && ty == 8 {
            let s = read_string(r)?;
            arch = Some(s);
        } else {
            skip_value(r, ty)?;
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(4096) as usize);
    for _ in 0..tensor_count {
        let name = read_string(r)?;
        let n_dims = r.read_u32::<LittleEndian>()?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(r.read_u64::<LittleEndian>()?);
        }
        let ggml_type = r.read_u32::<LittleEndian>()?;
        let offset = r.read_u64::<LittleEndian>()?;
        tensors.push(TensorInfo {
            name,
            n_dims,
            shape,
            ggml_type,
            offset,
        });
    }

    let pos = r.stream_position()?;
    let align = 32u64;
    let data_base = pos
        .saturating_add(align - 1)
        .saturating_sub((pos.saturating_add(align - 1)) % align);

    Ok((
        GgufSummary {
            version,
            tensor_count,
            kv_count,
            architecture: arch,
            tensors,
        },
        data_base,
    ))
}

/// Scan GGUF metadata and tensor index from any seekable reader.
pub fn scan_gguf<R: Read + Seek>(r: &mut R) -> Result<GgufSummary, GgufError> {
    Ok(scan_gguf_with_layout(r)?.0)
}

/// Read up to `max_elems` f32 values from the first F32 tensor in the file.
pub fn sample_f32_weights_from_path(path: &Path, max_elems: usize) -> Result<(Vec<f32>, u64), GgufError> {
    let mut file = File::open(path)?;
    let mut br = BufReader::new(&mut file);
    let (summary, data_base) = scan_gguf_with_layout(&mut br)?;
    let param_count = estimate_param_elements(&summary);
    drop(br);
    for t in &summary.tensors {
        if t.ggml_type != 0 {
            continue;
        } // GGML_TYPE_F32
        let ne: u64 = t.shape.iter().product();
        if ne == 0 {
            continue;
        }
        let n = (ne as usize).min(max_elems);
        let start = data_base.saturating_add(t.offset);
        file.seek(SeekFrom::Start(start))?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(file.read_f32::<LittleEndian>()?);
        }
        return Ok((out, param_count));
    }
    // Quantized models: sample raw bytes from first tensor as [0,1] floats (stub).
    let t = match summary.tensors.first() {
        Some(t) => t,
        None => return Err(GgufError::NoF32Tensor),
    };
    let start = data_base.saturating_add(t.offset);
    file.seek(SeekFrom::Start(start))?;
    let mut raw = vec![0u8; max_elems.min(65536)];
    let n = file.read(&mut raw)?;
    raw.truncate(n);
    let out: Vec<f32> = raw.iter().map(|&b| b as f32 / 255.0).collect();
    if out.is_empty() {
        return Err(GgufError::NoF32Tensor);
    }
    Ok((out, param_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        let b = s.as_bytes();
        buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
        buf.extend_from_slice(b);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut c = Cursor::new(b"NOPE".to_vec());
        assert!(matches!(scan_gguf(&mut c), Err(GgufError::BadMagic)));
    }

    #[test]
    fn parses_empty_v3() {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes());
        let mut c = Cursor::new(v);
        let s = scan_gguf(&mut c).unwrap();
        assert_eq!(s.version, 3);
        assert_eq!(s.tensor_count, 0);
        assert!(s.tensors.is_empty());
    }

    #[test]
    fn parses_architecture_kv() {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes()); // tensors
        v.extend_from_slice(&1u64.to_le_bytes()); // kvs
        write_string(&mut v, "general.architecture");
        v.extend_from_slice(&8u32.to_le_bytes()); // STRING
        write_string(&mut v, "llama");
        let mut c = Cursor::new(v);
        let s = scan_gguf(&mut c).unwrap();
        assert_eq!(s.architecture.as_deref(), Some("llama"));
    }
}
