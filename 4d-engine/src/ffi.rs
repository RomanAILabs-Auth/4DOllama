//! C ABI for Go / other hosts.
//!
//! // 4D ENGINE NOTE: Strings from `fd4_gguf_inspect_json` are Rust `CString`; free only via `fd4_free_string`.

use crate::converter::gguf::{estimate_param_elements, sample_f32_weights_from_path, scan_gguf_with_layout};
use crate::converter::inspect_gguf_path;
use crate::converter::lift::lift_to_4d;
use crate::demo::compute_4d_demo;
use crate::gpu::{self, GpuBackend};
use crate::logits_project::project_logits_stub;
use crate::rope4d::apply_quaternion_rope_sequence;
use crate::sampling4d::{sample_next_token_4d, sample_next_token_flat};
use serde_json::json;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::io::BufReader;
use std::path::Path;
use std::ptr::null_mut;

thread_local! {
    static LAST_ERR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_last_error(msg: &str) {
    LAST_ERR.with(|e| {
        *e.borrow_mut() = CString::new(msg).unwrap_or_else(|_| CString::new("fd4: error").unwrap());
    });
}

fn clear_last_error() {
    LAST_ERR.with(|e| {
        *e.borrow_mut() = CString::new("").unwrap();
    });
}

#[no_mangle]
pub extern "C" fn fd4_version() -> *const libc::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

#[no_mangle]
pub extern "C" fn fd4_last_error() -> *const libc::c_char {
    LAST_ERR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn fd4_capabilities_json() -> *mut libc::c_char {
    clear_last_error();
    let gpu_backend = match gpu::active_backend() {
        GpuBackend::Cpu => "cpu",
        GpuBackend::Cuda => "cuda",
        GpuBackend::Metal => "metal",
    };
    let v = json!({
        "name": "four_d_engine",
        "version": env!("CARGO_PKG_VERSION"),
        "abi_version": 1,
        "computational_model": {
            "tensor_rank": 4,
            "quaternion_algebra": "hamilton",
            "description": "Hamilton quaternions + 4D tensor ops (w-axis contraction); GGUF scan and lift preview in-tree."
        },
        "ffi": {
            "gguf_inspect_json": true,
            "capabilities_json": true,
            "compute_demo": true,
            "gguf_param_count": true,
            "gguf_sample_lift": true,
            "rope4d_sequence": true,
            "spacetime_attention4d": true,
            "sample_next_token_4d": true,
            "gemm4d": true,
            "gpu_backend": gpu_backend,
            "autoregressive_inference": true
        },
        "prior_art": "Quaternion and hypercomplex neural networks are an active research area (vision, signals, some NLP). No major vendor ships this exact stack as a drop-in Ollama replacement; this project combines a 4D tensor core with Ollama-compatible APIs."
    });
    match CString::new(v.to_string()) {
        Ok(c) => c.into_raw(),
        Err(e) => {
            set_last_error(&format!("cstring: {e}"));
            null_mut()
        }
    }
}

/// Quaternion demo: copies `n_in` floats, pads to triples internally, writes rotated values.
/// Returns 0 on success; `*n_out` is written length (≤ out_cap). Returns -1 on error.
#[no_mangle]
pub unsafe extern "C" fn fd4_compute_demo(
    in_ptr: *const libc::c_float,
    in_len: libc::size_t,
    out_ptr: *mut libc::c_float,
    out_cap: libc::size_t,
    n_out: *mut libc::size_t,
) -> libc::c_int {
    clear_last_error();
    if in_ptr.is_null() || out_ptr.is_null() || n_out.is_null() {
        set_last_error("fd4_compute_demo: null pointer");
        return -1;
    }
    let in_len = in_len as usize;
    let out_cap = out_cap as usize;
    let slice = std::slice::from_raw_parts(in_ptr, in_len);
    let output = compute_4d_demo(slice.to_vec());
    if output.len() > out_cap {
        set_last_error("fd4_compute_demo: out_cap too small");
        return -1;
    }
    std::ptr::copy_nonoverlapping(output.as_ptr(), out_ptr, output.len());
    *n_out = output.len() as libc::size_t;
    0
}

#[no_mangle]
pub unsafe extern "C" fn fd4_gguf_param_count(
    path_utf8: *const libc::c_char,
    out_params: *mut u64,
) -> libc::c_int {
    clear_last_error();
    if path_utf8.is_null() || out_params.is_null() {
        set_last_error("fd4_gguf_param_count: null pointer");
        return -1;
    }
    let path_str = match CStr::from_ptr(path_utf8).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("path utf-8: {e}"));
            return -1;
        }
    };
    let f = match std::fs::File::open(path_str) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(&e.to_string());
            return -1;
        }
    };
    let mut br = BufReader::new(f);
    let (sum, _) = match scan_gguf_with_layout(&mut br) {
        Ok(x) => x,
        Err(e) => {
            set_last_error(&e.to_string());
            return -1;
        }
    };
    *out_params = estimate_param_elements(&sum);
    0
}

#[no_mangle]
pub unsafe extern "C" fn fd4_gguf_sample_lift(
    path_utf8: *const libc::c_char,
    max_sample: libc::size_t,
    out_ptr: *mut libc::c_float,
    out_cap: libc::size_t,
    n_written: *mut libc::size_t,
    n_params: *mut u64,
) -> libc::c_int {
    clear_last_error();
    if path_utf8.is_null() || out_ptr.is_null() || n_written.is_null() || n_params.is_null() {
        set_last_error("fd4_gguf_sample_lift: null pointer");
        return -1;
    }
    let path_str = match CStr::from_ptr(path_utf8).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("path utf-8: {e}"));
            return -1;
        }
    };
    let path = Path::new(path_str);
    let max_sample = max_sample as usize;
    let (sample, pcount) = match sample_f32_weights_from_path(path, max_sample) {
        Ok(x) => x,
        Err(e) => {
            set_last_error(&e.to_string());
            return -1;
        }
    };
    let lifted = lift_to_4d(sample);
    let out_cap = out_cap as usize;
    if lifted.len() > out_cap {
        set_last_error("fd4_gguf_sample_lift: out_cap too small");
        return -1;
    }
    *n_params = pcount;
    *n_written = lifted.len() as libc::size_t;
    std::ptr::copy_nonoverlapping(lifted.as_ptr(), out_ptr, lifted.len());
    0
}

/// Quaternion RoPE over embedding quads; pads input to multiple of 4.
#[no_mangle]
pub unsafe extern "C" fn fd4_rope4d_sequence(
    in_ptr: *const libc::c_float,
    in_len: libc::size_t,
    out_ptr: *mut libc::c_float,
    out_cap: libc::size_t,
    n_out: *mut libc::size_t,
) -> libc::c_int {
    clear_last_error();
    if out_ptr.is_null() || n_out.is_null() {
        set_last_error("fd4_rope4d_sequence: null pointer");
        return -1;
    }
    let in_len = in_len as usize;
    let out_cap = out_cap as usize;
    let slice = if in_len == 0 || in_ptr.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(in_ptr, in_len)
    };
    let output = apply_quaternion_rope_sequence(slice.to_vec());
    if output.len() > out_cap {
        set_last_error("fd4_rope4d_sequence: out_cap too small");
        return -1;
    }
    if !output.is_empty() {
        std::ptr::copy_nonoverlapping(output.as_ptr(), out_ptr, output.len());
    }
    *n_out = output.len() as libc::size_t;
    0
}

/// Causal quaternion attention along the sequence axis.
#[no_mangle]
pub unsafe extern "C" fn fd4_spacetime_attention(
    q_ptr: *const libc::c_float,
    k_ptr: *const libc::c_float,
    v_ptr: *const libc::c_float,
    n_floats: libc::size_t,
    seq_len: libc::size_t,
    out_ptr: *mut libc::c_float,
    out_cap: libc::size_t,
    n_out: *mut libc::size_t,
) -> libc::c_int {
    clear_last_error();
    if q_ptr.is_null() || k_ptr.is_null() || v_ptr.is_null() || out_ptr.is_null() || n_out.is_null() {
        set_last_error("fd4_spacetime_attention: null pointer");
        return -1;
    }
    let n = n_floats as usize;
    let sl = seq_len as usize;
    let need = sl.saturating_mul(4);
    if need == 0 || n < need {
        set_last_error("fd4_spacetime_attention: invalid seq_len or n_floats");
        return -1;
    }
    let q = std::slice::from_raw_parts(q_ptr, need);
    let k = std::slice::from_raw_parts(k_ptr, need);
    let v = std::slice::from_raw_parts(v_ptr, need);
    let output = gpu::spacetime_attention_dispatch(q, k, v, sl);
    if output.is_empty() {
        set_last_error("fd4_spacetime_attention: compute failed");
        return -1;
    }
    let out_cap = out_cap as usize;
    if output.len() > out_cap {
        set_last_error("fd4_spacetime_attention: out_cap too small");
        return -1;
    }
    std::ptr::copy_nonoverlapping(output.as_ptr(), out_ptr, output.len());
    *n_out = output.len() as libc::size_t;
    0
}

#[no_mangle]
pub unsafe extern "C" fn fd4_sample_next_token_4d(
    logits_ptr: *const libc::c_float,
    n_logits: libc::size_t,
    temperature: libc::c_float,
    top_k: libc::size_t,
) -> u32 {
    clear_last_error();
    if logits_ptr.is_null() || n_logits == 0 {
        return 0;
    }
    let n = n_logits as usize;
    let logits = std::slice::from_raw_parts(logits_ptr, n);
    sample_next_token_4d(logits, temperature, top_k as usize)
}

#[no_mangle]
pub unsafe extern "C" fn fd4_sample_next_token_flat(
    logits_ptr: *const libc::c_float,
    n_logits: libc::size_t,
    temperature: libc::c_float,
    top_k: libc::size_t,
) -> u32 {
    clear_last_error();
    if logits_ptr.is_null() || n_logits == 0 {
        return 0;
    }
    let n = n_logits as usize;
    let logits = std::slice::from_raw_parts(logits_ptr, n);
    sample_next_token_flat(logits, temperature, top_k as usize)
}

#[no_mangle]
pub unsafe extern "C" fn fd4_project_logits_stub(
    last4_ptr: *const libc::c_float,
    last_len: libc::size_t,
    lifted_ptr: *const libc::c_float,
    lifted_len: libc::size_t,
    vocab_size: libc::uint32_t,
    logits_out: *mut libc::c_float,
    logits_cap: libc::size_t,
    n_logits: *mut libc::size_t,
    step: libc::uint32_t,
    log_first: libc::c_int,
) -> libc::c_int {
    clear_last_error();
    if last4_ptr.is_null() || logits_out.is_null() || n_logits.is_null() {
        set_last_error("fd4_project_logits_stub: null pointer");
        return -1;
    }
    if last_len < 4 {
        set_last_error("fd4_project_logits_stub: last_len < 4");
        return -1;
    }
    let last = std::slice::from_raw_parts(last4_ptr, 4);
    let lifted = if lifted_ptr.is_null() || lifted_len == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(lifted_ptr, lifted_len as usize)
    };
    let vs = vocab_size as usize;
    let cap = logits_cap as usize;
    if vs == 0 || cap < vs {
        set_last_error("fd4_project_logits_stub: bad vocab_size or logits_cap");
        return -1;
    }
    let out = std::slice::from_raw_parts_mut(logits_out, cap);
    let nw = project_logits_stub(
        last,
        lifted,
        vs,
        step,
        log_first != 0,
        &mut out[..vs],
    );
    *n_logits = nw as libc::size_t;
    0
}

#[no_mangle]
pub unsafe extern "C" fn fd4_gemm4d(
    a_ptr: *const libc::c_float,
    na: libc::size_t,
    b_ptr: *const libc::c_float,
    nb: libc::size_t,
    c_ptr: *mut libc::c_float,
    nc: libc::size_t,
    m: libc::size_t,
    k: libc::size_t,
    n: libc::size_t,
) -> libc::c_int {
    clear_last_error();
    if a_ptr.is_null() || b_ptr.is_null() || c_ptr.is_null() {
        set_last_error("fd4_gemm4d: null pointer");
        return -1;
    }
    let m = m as usize;
    let kdim = k as usize;
    let n = n as usize;
    let need_a = m.saturating_mul(kdim);
    let need_b = kdim.saturating_mul(n);
    let need_c = m.saturating_mul(n);
    if na as usize < need_a || nb as usize < need_b || nc as usize < need_c {
        set_last_error("fd4_gemm4d: buffer too small");
        return -1;
    }
    let a = std::slice::from_raw_parts(a_ptr, need_a);
    let b = std::slice::from_raw_parts(b_ptr, need_b);
    let c = std::slice::from_raw_parts_mut(c_ptr, need_c);
    gpu::gemm4d_dispatch(a, b, c, m, kdim, n);
    0
}

#[no_mangle]
pub extern "C" fn fd4_gpu_backend_name() -> *const libc::c_char {
    match gpu::active_backend() {
        GpuBackend::Metal => b"metal\0".as_ptr().cast(),
        GpuBackend::Cuda => b"cuda\0".as_ptr().cast(),
        GpuBackend::Cpu => b"cpu\0".as_ptr().cast(),
    }
}

/// Frees pointers returned by `fd4_capabilities_json` or `fd4_gguf_inspect_json`.
#[no_mangle]
pub extern "C" fn fd4_free_string(p: *mut libc::c_char) {
    if p.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(p);
    }
}

#[no_mangle]
pub extern "C" fn fd4_gguf_inspect_json(path_utf8: *const libc::c_char) -> *mut libc::c_char {
    clear_last_error();
    if path_utf8.is_null() {
        set_last_error("null path");
        return null_mut();
    }
    let path_str = unsafe { CStr::from_ptr(path_utf8) };
    let path_str = match path_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid utf-8 path: {e}"));
            return null_mut();
        }
    };
    let path = Path::new(path_str);
    match inspect_gguf_path(path) {
        Ok(rep) => {
            let v = json!({
                "engine": rep.engine,
                "path": rep.path,
                "gguf_version": rep.gguf.version,
                "tensor_count": rep.gguf.tensor_count,
                "kv_count": rep.gguf.kv_count,
                "architecture": rep.gguf.architecture,
                "lift_preview": rep.lift_preview,
                "tensors_preview": rep.gguf.tensors.iter().take(32).collect::<Vec<_>>(),
                "tensors_truncated": rep.gguf.tensors.len() > 32,
            });
            match CString::new(v.to_string()) {
                Ok(c) => c.into_raw(),
                Err(e) => {
                    set_last_error(&format!("cstring: {e}"));
                    null_mut()
                }
            }
        }
        Err(e) => {
            set_last_error(&e.to_string());
            null_mut()
        }
    }
}
