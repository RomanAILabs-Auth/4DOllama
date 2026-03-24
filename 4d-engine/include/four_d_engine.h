/* C ABI for 4DOllama — consumed by Go cgo. */
#ifndef FOUR_D_ENGINE_H
#define FOUR_D_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Returns static version string (do not free). */
const char *fd4_version(void);

/**
 * Runtime capability manifest (JSON). Caller must free with fd4_free_string.
 * Describes quaternion/4D tensor features actually compiled into this binary.
 */
char *fd4_capabilities_json(void);

/**
 * Inspect a GGUF file: version, tensors, selected metadata, 4D lift preview.
 * Returns malloc-allocated UTF-8 JSON, or NULL on error (see fd4_last_error).
 * Caller must free with fd4_free_string.
 */
char *fd4_gguf_inspect_json(const char *path_utf8);

void fd4_free_string(char *p);

/**
 * Quaternion rotation demo on float triples (input padded internally to len % 3 == 0).
 * Returns 0 on success, -1 on error. *n_out is number of floats written.
 */
int fd4_compute_demo(const float *in, size_t n_in, float *out, size_t out_cap, size_t *n_out);

int fd4_gguf_param_count(const char *path, uint64_t *out_params);

int fd4_gguf_sample_lift(const char *path, size_t max_sample, float *out, size_t out_cap,
                         size_t *n_written, uint64_t *n_params);

/** Per-token quaternion RoPE on embedding quads (pad len % 4 == 0). */
int fd4_rope4d_sequence(const float *in, size_t n_in, float *out, size_t out_cap, size_t *n_out);

/**
 * Causal 4D spacetime attention: n_floats must equal seq_len * 4; q, k, v each have at least n floats.
 */
int fd4_spacetime_attention(const float *q, const float *k, const float *v, size_t n_floats, size_t seq_len,
                            float *out, size_t out_cap, size_t *n_out);

/** 4D-projected softmax sample; returns quaternion-row index (map to 4 logits per row on host). */
uint32_t fd4_sample_next_token_4d(const float *logits, size_t n_logits, float temperature, size_t top_k);

/** Flat softmax sample over full-vocab logits (stub LM-head path). */
uint32_t fd4_sample_next_token_flat(const float *logits, size_t n_logits, float temperature, size_t top_k);

/**
 * Stub output projection: last4 (len>=4) + optional lifted weights -> vocab_size logits.
 * Writes exactly vocab_size floats. log_first!=0 prints one stderr debug line when step==0.
 * Returns 0 on success.
 */
int fd4_project_logits_stub(const float *last4, size_t last_len,
                            const float *lifted, size_t lifted_len,
                            uint32_t vocab_size,
                            float *logits_out, size_t logits_cap, size_t *n_logits,
                            uint32_t step, int log_first);

/** Row-major GEMM C[m,n] = A[m,k] * B[k,n]; na >= m*k, nb >= k*n, nc >= m*n. */
int fd4_gemm4d(const float *a, size_t na, const float *b, size_t nb, float *c, size_t nc,
               size_t m, size_t k, size_t n);

/** Static string: "cpu" | "cuda" | "metal" — do not free. */
const char *fd4_gpu_backend_name(void);

/** Thread-local last error message (UTF-8), empty if none. */
const char *fd4_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* FOUR_D_ENGINE_H */
