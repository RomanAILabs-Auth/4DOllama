// Copyright RomanAILabs - Daniel Harding
/*
 * RomanAI Pass 3 — full-file GGUF buffer + per-tensor byte spans (ggml_row_size).
 * Pass 1 vocab: tokenizer.ggml.tokens (GGUF array of strings) → UTF-8 lookup for mir_romanai_vocab_print.
 */

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long long roma4d_i64;

typedef struct {
    const uint8_t *b;
    size_t n;
    size_t p;
} RaBuf;

typedef struct {
    uint64_t abs_off;
    int64_t nbytes;
    int64_t vec4_slots;
    uint32_t ggml_type;
    int64_t nelems;
    int valid;
} RaLayTensor;

typedef struct {
    uint64_t offset_rel;
    uint32_t ggml_type;
    int64_t nelems;
} RaLayTmp;

static uint8_t *g_ra_file = NULL;
static size_t g_ra_file_len = 0;
static RaLayTensor *g_ra_tensors = NULL;
static uint64_t g_ra_ntensors = 0;
static uint64_t g_ra_loop_count = 0;
static char g_ra_loaded_path[4096];

/* tokenizer.ggml.tokens — contiguous UTF-8 bytes; off[i]..off[i+1] is token i */
static char *g_ra_vocab_blob = NULL;
static size_t *g_ra_vocab_off = NULL;
static size_t g_ra_vocab_ntokens = 0;

static int ra_u32(RaBuf *r, uint32_t *o) {
    if (r->p + 4 > r->n) {
        return -1;
    }
    *o = (uint32_t)r->b[r->p] | ((uint32_t)r->b[r->p + 1] << 8) | ((uint32_t)r->b[r->p + 2] << 16)
       | ((uint32_t)r->b[r->p + 3] << 24);
    r->p += 4;
    return 0;
}

static int ra_u64(RaBuf *r, uint64_t *o) {
    if (r->p + 8 > r->n) {
        return -1;
    }
    *o = (uint64_t)r->b[r->p] | ((uint64_t)r->b[r->p + 1] << 8) | ((uint64_t)r->b[r->p + 2] << 16)
       | ((uint64_t)r->b[r->p + 3] << 24) | ((uint64_t)r->b[r->p + 4] << 32) | ((uint64_t)r->b[r->p + 5] << 40)
       | ((uint64_t)r->b[r->p + 6] << 48) | ((uint64_t)r->b[r->p + 7] << 56);
    r->p += 8;
    return 0;
}

static int ra_skip_string(RaBuf *r) {
    uint64_t len;
    if (ra_u64(r, &len) != 0) {
        return -1;
    }
    if (len > (1ull << 30) || r->p + len > r->n) {
        return -1;
    }
    r->p += (size_t)len;
    return 0;
}

static int ra_skip_gguf_value(RaBuf *r, uint32_t ty);

static int ra_skip_gguf_value(RaBuf *r, uint32_t ty) {
    uint32_t u32;
    uint64_t u64;
    uint32_t et;
    uint64_t ne;
    uint64_t i;

    switch (ty) {
    case 0:
        if (r->p >= r->n) {
            return -1;
        }
        r->p++;
        return 0;
    case 1:
        if (r->p >= r->n) {
            return -1;
        }
        r->p++;
        return 0;
    case 2:
    case 3:
        if (r->p + 2 > r->n) {
            return -1;
        }
        r->p += 2;
        return 0;
    case 4:
    case 5:
        return ra_u32(r, &u32);
    case 6:
        if (r->p + 4 > r->n) {
            return -1;
        }
        r->p += 4;
        return 0;
    case 7:
        if (r->p >= r->n) {
            return -1;
        }
        r->p++;
        return 0;
    case 8:
        return ra_skip_string(r);
    case 9:
        if (ra_u32(r, &et) != 0) {
            return -1;
        }
        if (ra_u64(r, &ne) != 0) {
            return -1;
        }
        for (i = 0; i < ne; i++) {
            if (ra_skip_gguf_value(r, et) != 0) {
                return -1;
            }
        }
        return 0;
    case 10:
    case 11:
        return ra_u64(r, &u64);
    case 12:
        if (r->p + 8 > r->n) {
            return -1;
        }
        r->p += 8;
        return 0;
    default:
        return -1;
    }
}

static void ra_vocab_free(void) {
    free(g_ra_vocab_blob);
    g_ra_vocab_blob = NULL;
    free(g_ra_vocab_off);
    g_ra_vocab_off = NULL;
    g_ra_vocab_ntokens = 0;
}

/* Read GGUF length-prefixed key into out (NUL-terminated); skip oversize keys without overflow. */
static int ra_read_kv_key(RaBuf *r, char *out, size_t out_sz) {
    uint64_t len;
    if (ra_u64(r, &len) != 0) {
        return -1;
    }
    if (len > (1ull << 30) || r->p + len > r->n) {
        return -1;
    }
    if (len >= out_sz) {
        r->p += (size_t)len;
        out[0] = '\0';
        return 1;
    }
    memcpy(out, r->b + r->p, (size_t)len);
    out[len] = '\0';
    r->p += (size_t)len;
    return 0;
}

/*
 * r->p must point at first element of a GGUF value of type ARRAY (9) whose key was tokenizer.ggml.tokens.
 * If element type is not STRING (8), skips the array without storing.
 * On parse error, restores r->p to save and skips the whole array value.
 */
static int ra_load_vocab_tokens_array(RaBuf *r) {
    size_t save = r->p;
    uint32_t et;
    uint64_t ne;
    uint64_t i;
    size_t elems_start;
    size_t total;
    char *blob;
    size_t pos;
    RaBuf scan;

    if (ra_u32(r, &et) != 0) {
        goto bail_skip;
    }
    if (ra_u64(r, &ne) != 0) {
        goto bail_skip;
    }
    if (et != 8) {
        for (i = 0; i < ne; i++) {
            if (ra_skip_gguf_value(r, et) != 0) {
                goto bail_skip;
            }
        }
        return 0;
    }
    if (ne > (1ull << 24)) {
        goto bail_skip;
    }

    elems_start = r->p;
    scan.b = r->b;
    scan.n = r->n;
    scan.p = elems_start;
    total = 0;
    for (i = 0; i < ne; i++) {
        uint64_t slen;
        if (ra_u64(&scan, &slen) != 0) {
            goto bail_skip;
        }
        if (slen > (1ull << 28) || scan.p + slen > scan.n) {
            goto bail_skip;
        }
        total += (size_t)slen;
        scan.p += (size_t)slen;
    }

    ra_vocab_free();

    if (ne == 0) {
        r->p = scan.p;
        return 0;
    }

    g_ra_vocab_off = (size_t *)calloc((size_t)(ne + 1), sizeof(size_t));
    if (!g_ra_vocab_off) {
        goto bail_skip;
    }
    blob = (char *)malloc(total + 1);
    if (!blob) {
        free(g_ra_vocab_off);
        g_ra_vocab_off = NULL;
        goto bail_skip;
    }

    scan.p = elems_start;
    pos = 0;
    for (i = 0; i < ne; i++) {
        uint64_t slen;
        if (ra_u64(&scan, &slen) != 0) {
            free(blob);
            free(g_ra_vocab_off);
            g_ra_vocab_off = NULL;
            r->p = save;
            return ra_skip_gguf_value(r, 9);
        }
        g_ra_vocab_off[i] = pos;
        memcpy(blob + pos, scan.b + scan.p, (size_t)slen);
        pos += (size_t)slen;
        scan.p += (size_t)slen;
    }
    g_ra_vocab_off[ne] = pos;
    blob[pos] = '\0';
    g_ra_vocab_blob = blob;
    g_ra_vocab_ntokens = (size_t)ne;
    r->p = scan.p;
    return 0;

bail_skip:
    r->p = save;
    return ra_skip_gguf_value(r, 9);
}

static int ra_shape_n_elems(uint32_t ndims, const uint64_t *dims, int64_t *out) {
    int64_t n = 1;
    uint32_t i;
    for (i = 0; i < ndims; i++) {
        uint64_t d = dims[i];
        if (d == 0) {
            *out = 0;
            return 0;
        }
        if ((uint64_t)n > (uint64_t)LLONG_MAX / d) {
            return -1;
        }
        n *= (int64_t)d;
    }
    *out = n;
    return 0;
}

static int ra_ggml_type_traits(uint32_t t, int64_t *bl, int64_t *ts) {
    switch (t) {
    case 0:
        *bl = 1;
        *ts = 4;
        return 0;
    case 1:
    case 30:
        *bl = 1;
        *ts = 2;
        return 0;
    case 2:
        *bl = 32;
        *ts = 18;
        return 0;
    case 3:
        *bl = 32;
        *ts = 20;
        return 0;
    case 6:
        *bl = 32;
        *ts = 22;
        return 0;
    case 7:
        *bl = 32;
        *ts = 24;
        return 0;
    case 8:
        *bl = 32;
        *ts = 34;
        return 0;
    case 9:
        *bl = 32;
        *ts = 36;
        return 0;
    case 10:
        *bl = 256;
        *ts = 2 * 2 + 256 / 16 + 256 / 4;
        return 0;
    case 11:
        *bl = 256;
        *ts = 2 + 256 / 4 + 256 / 8 + 12;
        return 0;
    case 12:
        *bl = 256;
        *ts = 144;
        return 0;
    case 13:
        *bl = 256;
        *ts = 160;
        return 0;
    case 14:
        *bl = 256;
        *ts = 210;
        return 0;
    case 15:
        *bl = 256;
        *ts = 4 + 256 + (256 / 16) * 2;
        return 0;
    default:
        return -1;
    }
}

static int ra_ggml_row_size(uint32_t typ, int64_t ne, int64_t *out) {
    int64_t bl;
    int64_t ts;
    if (ne < 0) {
        return -1;
    }
    if (ra_ggml_type_traits(typ, &bl, &ts) != 0) {
        return -1;
    }
    if (bl == 0 || ts == 0) {
        return -1;
    }
    if (ne % bl != 0) {
        return -1;
    }
    *out = (ne / bl) * ts;
    return 0;
}

static void ra_layout_free_internal(void) {
    ra_vocab_free();
    free(g_ra_file);
    g_ra_file = NULL;
    g_ra_file_len = 0;
    free(g_ra_tensors);
    g_ra_tensors = NULL;
    g_ra_ntensors = 0;
    g_ra_loop_count = 0;
    g_ra_loaded_path[0] = '\0';
}

int32_t mir_romanai_gguf_layout_free(void) {
    ra_layout_free_internal();
    return 0;
}

/*
 * UTF-8 span for token_id from tokenizer.ggml.tokens (no NUL inside [utf8, utf8+len)).
 * If layout has no vocab or id is out of range, *utf8_out is NULL and *byte_len_out is 0.
 */
void mir_romanai_gguf_vocab_lookup(roma4d_i64 token_id, const char **utf8_out, size_t *byte_len_out) {
    long long id = (long long)token_id;
    *utf8_out = NULL;
    *byte_len_out = 0;
    if (!g_ra_vocab_blob || !g_ra_vocab_off || g_ra_vocab_ntokens == 0) {
        return;
    }
    if (id < 0 || (uint64_t)id >= (uint64_t)g_ra_vocab_ntokens) {
        return;
    }
    {
        size_t a = g_ra_vocab_off[(size_t)id];
        size_t b = g_ra_vocab_off[(size_t)id + 1];
        *utf8_out = g_ra_vocab_blob + a;
        *byte_len_out = b - a;
    }
}

/*
 * Returns: 0 ok, 1 path/open/read, 2 bad magic/version, 3 parse, 4 file too large (ROMANAI_GGUF_MAX_BYTES).
 */
int32_t mir_romanai_gguf_layout_load(const char *path) {
    FILE *f;
    long sz;
    RaBuf r;
    char magic[4];
    uint32_t ver;
    uint64_t tensor_count;
    uint64_t kv_count;
    uint64_t ki;
    uint64_t ti;
    RaLayTmp *tmp = NULL;
    uint64_t data_base;
    unsigned long max_file = 0;
    const char *maxb_env;
    unsigned long max_tensors = 0;
    const char *maxt_env;
    int64_t max_vec4_slots = 0;
    const char *maxv_env;
    int log_pass3 = 0;

    if (!path || !*path || strcmp(path, "not_set") == 0) {
        return 1;
    }

    if (g_ra_loaded_path[0] && strcmp(path, g_ra_loaded_path) == 0 && g_ra_file) {
        return 0;
    }
    ra_layout_free_internal();

    maxb_env = getenv("ROMANAI_GGUF_MAX_BYTES");
    if (maxb_env && *maxb_env) {
        char *end = NULL;
        unsigned long v = strtoul(maxb_env, &end, 10);
        if (end != maxb_env && v > 0) {
            max_file = v;
        }
    }
    maxt_env = getenv("ROMANAI_PASS3_MAX_TENSORS");
    if (maxt_env && *maxt_env) {
        char *end = NULL;
        unsigned long v = strtoul(maxt_env, &end, 10);
        if (end != maxt_env) {
            max_tensors = v;
        }
    }
    maxv_env = getenv("ROMANAI_PASS3_MAX_VEC4_SLOTS");
    if (maxv_env && *maxv_env) {
        char *end = NULL;
        unsigned long v = strtoul(maxv_env, &end, 10);
        if (end != maxv_env && v > 0 && v < (unsigned long)LLONG_MAX) {
            max_vec4_slots = (int64_t)v;
        }
    }
    {
        const char *pl = getenv("ROMANAI_PASS3_LOG");
        if (pl && pl[0] != '\0' && pl[0] != '0') {
            log_pass3 = 1;
        }
    }

    f = fopen(path, "rb");
    if (!f) {
        return 1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 1;
    }
    sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return 1;
    }
    if (max_file > 0 && (unsigned long long)sz > (unsigned long long)max_file) {
        fprintf(stderr, "ROMANAI_GGUF_WORLDTUBE\terror\tfile_too_large\tmax_bytes=%lu\tsize=%ld\n", max_file, sz);
        fclose(f);
        return 4;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return 1;
    }
    g_ra_file = (uint8_t *)malloc((size_t)sz);
    if (!g_ra_file) {
        fclose(f);
        return 1;
    }
    if (fread(g_ra_file, 1, (size_t)sz, f) != (size_t)sz) {
        free(g_ra_file);
        g_ra_file = NULL;
        fclose(f);
        return 1;
    }
    fclose(f);
    g_ra_file_len = (size_t)sz;
    strncpy(g_ra_loaded_path, path, sizeof g_ra_loaded_path - 1);
    g_ra_loaded_path[sizeof g_ra_loaded_path - 1] = '\0';

    r.b = g_ra_file;
    r.n = g_ra_file_len;
    r.p = 0;

    if (r.p + 4 > r.n || memcmp(r.b + r.p, "GGUF", 4) != 0) {
        ra_layout_free_internal();
        return 2;
    }
    r.p += 4;
    if (ra_u32(&r, &ver) != 0 || ver < 2 || ver > 4) {
        ra_layout_free_internal();
        return 2;
    }
    if (ra_u64(&r, &tensor_count) != 0 || ra_u64(&r, &kv_count) != 0) {
        ra_layout_free_internal();
        return 3;
    }

    for (ki = 0; ki < kv_count; ki++) {
        char kbuf[512];
        int kr = ra_read_kv_key(&r, kbuf, sizeof kbuf);
        if (kr < 0) {
            ra_layout_free_internal();
            return 3;
        }
        {
            uint32_t ty;
            if (ra_u32(&r, &ty) != 0) {
                ra_layout_free_internal();
                return 3;
            }
            if (kr == 0 && strcmp(kbuf, "tokenizer.ggml.tokens") == 0 && ty == 9) {
                if (ra_load_vocab_tokens_array(&r) != 0) {
                    ra_layout_free_internal();
                    return 3;
                }
            } else {
                if (ra_skip_gguf_value(&r, ty) != 0) {
                    ra_layout_free_internal();
                    return 3;
                }
            }
        }
    }

    if (tensor_count > (1ull << 40)) {
        ra_layout_free_internal();
        return 3;
    }
    tmp = (RaLayTmp *)calloc((size_t)tensor_count, sizeof(RaLayTmp));
    if (!tmp && tensor_count > 0) {
        ra_layout_free_internal();
        return 3;
    }

    for (ti = 0; ti < tensor_count; ti++) {
        uint32_t ndims;
        uint64_t shape[64];
        uint32_t d;
        if (ra_skip_string(&r) != 0) {
            free(tmp);
            ra_layout_free_internal();
            return 3;
        }
        if (ra_u32(&r, &ndims) != 0 || ndims > 64) {
            free(tmp);
            ra_layout_free_internal();
            return 3;
        }
        for (d = 0; d < ndims; d++) {
            if (ra_u64(&r, &shape[d]) != 0) {
                free(tmp);
                ra_layout_free_internal();
                return 3;
            }
        }
        if (ra_u32(&r, &tmp[ti].ggml_type) != 0 || ra_u64(&r, &tmp[ti].offset_rel) != 0) {
            free(tmp);
            ra_layout_free_internal();
            return 3;
        }
        if (ra_shape_n_elems(ndims, shape, &tmp[ti].nelems) != 0) {
            tmp[ti].nelems = -1;
        }
    }

    data_base = (uint64_t)(((r.p + 31u) / 32u) * 32u);
    if (data_base > g_ra_file_len) {
        free(tmp);
        ra_layout_free_internal();
        return 3;
    }

    g_ra_ntensors = tensor_count;
    g_ra_tensors = (RaLayTensor *)calloc((size_t)tensor_count, sizeof(RaLayTensor));
    if (!g_ra_tensors && tensor_count > 0) {
        free(tmp);
        ra_layout_free_internal();
        return 3;
    }

    for (ti = 0; ti < tensor_count; ti++) {
        int64_t nbytes;
        uint64_t abs_off;
        int64_t slots;

        if (tmp[ti].nelems < 0) {
            continue;
        }
        if (ra_ggml_row_size(tmp[ti].ggml_type, tmp[ti].nelems, &nbytes) != 0) {
            if (log_pass3) {
                fprintf(stderr, "ROMANAI_GGUF_WORLDTUBE\tskip\ti=%llu\treason=row_size\n", (unsigned long long)ti);
            }
            continue;
        }
        abs_off = data_base + tmp[ti].offset_rel;
        if (abs_off > g_ra_file_len || (uint64_t)nbytes > g_ra_file_len - abs_off) {
            if (log_pass3) {
                fprintf(stderr, "ROMANAI_GGUF_WORLDTUBE\tskip\ti=%llu\treason=out_of_file\n", (unsigned long long)ti);
            }
            continue;
        }
        slots = nbytes / (int64_t)(4 * sizeof(double));
        if (max_vec4_slots > 0 && slots > max_vec4_slots) {
            slots = max_vec4_slots;
        }
        g_ra_tensors[ti].valid = 1;
        g_ra_tensors[ti].abs_off = abs_off;
        g_ra_tensors[ti].nbytes = nbytes;
        g_ra_tensors[ti].vec4_slots = slots;
        g_ra_tensors[ti].ggml_type = tmp[ti].ggml_type;
        g_ra_tensors[ti].nelems = tmp[ti].nelems;
        if (log_pass3) {
            fprintf(stderr,
                    "ROMANAI_GGUF_WORLDTUBE\ttensor\ti=%llu\tvec4_slots=%lld\tnbytes=%lld\tabs_off=%llu\tggml_type=%u\n",
                    (unsigned long long)ti, (long long)g_ra_tensors[ti].vec4_slots, (long long)nbytes,
                    (unsigned long long)abs_off, tmp[ti].ggml_type);
        }
    }
    free(tmp);

    g_ra_loop_count = g_ra_ntensors;
    if (max_tensors > 0 && g_ra_loop_count > max_tensors) {
        g_ra_loop_count = max_tensors;
    }

    fprintf(stderr,
            "ROMANAI_GGUF_WORLDTUBE\tfooter\tfile_bytes=%zu\tdata_base=%llu\ttensors=%llu\tloop_tensors=%llu\tmmap_mode=full_buffer\n",
            g_ra_file_len, (unsigned long long)data_base, (unsigned long long)g_ra_ntensors,
            (unsigned long long)g_ra_loop_count);
#ifdef _WIN32
    fprintf(stderr, "ROMANAI_GGUF_WORLDTUBE\tplatform\tnote=Windows uses full fread buffer (not CreateFileMapping); POSIX uses same buffer path in Pass 3\n");
#else
    fprintf(stderr, "ROMANAI_GGUF_WORLDTUBE\tplatform\tnote=full buffer read; mir_mmap_gguf may still cap separately for legacy demos\n");
#endif
    fflush(stderr);

    return 0;
}

roma4d_i64 mir_romanai_gguf_layout_tensor_count(void) {
    return (roma4d_i64)g_ra_loop_count;
}

roma4d_i64 mir_romanai_gguf_layout_tensor_vec4_slots(roma4d_i64 idx) {
    if (!g_ra_tensors || idx < 0 || (uint64_t)idx >= g_ra_ntensors) {
        return 0;
    }
    if (!g_ra_tensors[idx].valid) {
        return 0;
    }
    return g_ra_tensors[idx].vec4_slots;
}

void *mir_romanai_gguf_layout_tensor_rawptr(roma4d_i64 idx) {
    if (!g_ra_file || !g_ra_tensors || idx < 0 || (uint64_t)idx >= g_ra_ntensors) {
        return NULL;
    }
    if (!g_ra_tensors[idx].valid) {
        return NULL;
    }
    return (void *)(g_ra_file + g_ra_tensors[idx].abs_off);
}

/* Phase 2 — quantization bridge: GGML type id + scratch F64 buffer as vec4 lanes for Roma4D GA. */
#define RA_QBRIDGE_MAX_FLOATS (262144u) /* 65536 vec4 lanes max */

static double g_ra_qbridge[RA_QBRIDGE_MAX_FLOATS];

roma4d_i64 mir_romanai_gguf_layout_tensor_ggml_type(roma4d_i64 idx) {
    if (!g_ra_tensors || idx < 0 || (uint64_t)idx >= g_ra_ntensors) {
        return -1;
    }
    if (!g_ra_tensors[idx].valid) {
        return -1;
    }
    return (roma4d_i64)g_ra_tensors[idx].ggml_type;
}

/*
 * Fills g_ra_qbridge with float64 values (4 = one vec4) for mir_cast_to_vec4_list.
 * ggml_type 0 (F32): copy min(nelems, cap) floats from tensor.
 * Else (Q4_K_M, etc.): linear byte→float stub (not full block decode — coherent host-side expansion for 4D prep).
 * Returns vec4 slot count, or 0 on failure.
 */
roma4d_i64 mir_romanai_gguf_tensor_quant_bridge_prepare(roma4d_i64 idx) {
    size_t i;
    size_t nf = 0;

    if (!g_ra_file || !g_ra_tensors || idx < 0 || (uint64_t)idx >= g_ra_ntensors) {
        return 0;
    }
    if (!g_ra_tensors[idx].valid) {
        return 0;
    }
    {
        RaLayTensor *t = &g_ra_tensors[idx];
        uint64_t off = t->abs_off;
        int64_t nb = t->nbytes;
        if (nb < 1 || off > g_ra_file_len || off + (uint64_t)nb > g_ra_file_len) {
            return 0;
        }
        if (t->ggml_type == 0u) {
            int64_t ne = t->nelems;
            size_t maxf;
            if (ne < 1) {
                return 0;
            }
            maxf = (size_t)ne;
            if (maxf > RA_QBRIDGE_MAX_FLOATS) {
                maxf = RA_QBRIDGE_MAX_FLOATS;
            }
            if ((uint64_t)nb < maxf * sizeof(float)) {
                maxf = (size_t)((uint64_t)nb / sizeof(float));
            }
            if (maxf == 0) {
                return 0;
            }
            {
                const float *fp = (const float *)(g_ra_file + off);
                for (i = 0; i < maxf && nf < RA_QBRIDGE_MAX_FLOATS; i++) {
                    g_ra_qbridge[nf++] = (double)fp[i];
                }
            }
        } else {
            size_t nbz = (size_t)nb;
            if (nbz == 0) {
                return 0;
            }
            for (i = 0; i < nbz && nf < RA_QBRIDGE_MAX_FLOATS; i++) {
                uint8_t b = g_ra_file[off + (uint64_t)i];
                double v = ((double)(int)b - 127.5) / 127.5;
                g_ra_qbridge[nf++] = v;
            }
        }
        if (nf == 0) {
            return 0;
        }
        while (nf % 4u != 0u && nf < RA_QBRIDGE_MAX_FLOATS) {
            g_ra_qbridge[nf++] = 0.0;
        }
        return (roma4d_i64)(nf / 4u);
    }
}

void *mir_romanai_gguf_tensor_quant_bridge_ptr(void) {
    return (void *)g_ra_qbridge;
}

/*
 * Pass 4 — one measurable “forward” step: fixed 4×4 mix (synthetic) + first F32 tensor dot probe.
 * Requires layout loaded for path (same buffer as Pass 3). Stderr prefix: ROMANAI_DECODE_GRAPH
 * Returns 0 ok, 1 bad path, 2 layout not loaded or path mismatch.
 */
int32_t mir_romanai_decode_graph_step(const char *path) {
    static const double M[4][4] = {
        {0.5, 0.5, 0.0, 0.0},
        {0.0, 0.7071067811865475, 0.7071067811865475, 0.0},
        {0.0, 0.0, 0.8660254037844386, 0.5},
        {0.25, 0.25, 0.25, 0.25},
    };
    double x[4] = {1.0, -0.5, 0.25, 0.125};
    double y[4];
    double syn_l2 = 0.0;
    int i;
    int j;
    uint64_t ti;

    if (!path || !*path || strcmp(path, "not_set") == 0) {
        return 1;
    }
    /* One GGUF buffer per process; do not rely on C string identity across R4D/MIR. */
    if (!g_ra_file || !g_ra_tensors || g_ra_ntensors == 0) {
        fprintf(stderr, "ROMANAI_DECODE_GRAPH\terror\tlayout_not_loaded\n");
        fflush(stderr);
        return 2;
    }

    for (i = 0; i < 4; i++) {
        double s = 0.0;
        for (j = 0; j < 4; j++) {
            s += M[i][j] * x[j];
        }
        y[i] = s;
        syn_l2 += s * s;
    }
    fprintf(stderr, "ROMANAI_DECODE_GRAPH\tsynthetic\tl2_y=%.12f\n", syn_l2);
    fflush(stderr);

    for (ti = 0; ti < g_ra_ntensors; ti++) {
        if (!g_ra_tensors[ti].valid) {
            continue;
        }
        if (g_ra_tensors[ti].ggml_type != 0u) {
            continue;
        }
        if (g_ra_tensors[ti].nelems < 4) {
            continue;
        }
        if (g_ra_tensors[ti].nbytes < (int64_t)(4 * sizeof(float))) {
            continue;
        }
        {
            float f[4];
            memcpy(f, g_ra_file + g_ra_tensors[ti].abs_off, sizeof f);
            double acc = 0.0;
            for (i = 0; i < 4; i++) {
                acc += (double)f[i] * x[i];
            }
            fprintf(stderr,
                    "ROMANAI_DECODE_GRAPH\tgguf_f32_probe\ti=%llu\tnelems=%lld\tdot4_fixed_x=%.12f\n",
                    (unsigned long long)ti, (long long)g_ra_tensors[ti].nelems, acc);
            fflush(stderr);
            return 0;
        }
    }

    fprintf(stderr, "ROMANAI_DECODE_GRAPH\tgguf_f32_probe\tskip=no_f32_tensor_nelems_ge_4\n");
    fflush(stderr);
    return 0;
}

/* ---- Pass 5 — 4D periodic lattice + injection → logit bias (honest micro-model) ---------- */

#define RALAT_NX 6
#define RALAT_NY 6
#define RALAT_NZ 6
#define RALAT_NW 4
#define RALAT_LEN ((RALAT_NX) * (RALAT_NY) * (RALAT_NZ) * (RALAT_NW))

static double ralat_cur[RALAT_LEN];
static double ralat_nxt[RALAT_LEN];
static double ralat_src[RALAT_LEN];

static size_t ralat_idx(int ix, int iy, int iz, int iw) {
    return (size_t)(unsigned)ix
         + (size_t)RALAT_NX
               * ((size_t)(unsigned)iy
                  + (size_t)RALAT_NY * ((size_t)(unsigned)iz + (size_t)RALAT_NZ * (size_t)(unsigned)iw));
}

static int ralat_wrap(int v, int n) {
    int r = v % n;
    if (r < 0) {
        r += n;
    }
    return r;
}

static double ralat_lap_periodic(const double *c, int ix, int iy, int iz, int iw) {
    double v = c[ralat_idx(ix, iy, iz, iw)];
    double s = 0.0;
    int d;

    d = ralat_wrap(ix + 1, RALAT_NX);
    s += c[ralat_idx(d, iy, iz, iw)] - v;
    d = ralat_wrap(ix - 1, RALAT_NX);
    s += c[ralat_idx(d, iy, iz, iw)] - v;
    d = ralat_wrap(iy + 1, RALAT_NY);
    s += c[ralat_idx(ix, d, iz, iw)] - v;
    d = ralat_wrap(iy - 1, RALAT_NY);
    s += c[ralat_idx(ix, d, iz, iw)] - v;
    d = ralat_wrap(iz + 1, RALAT_NZ);
    s += c[ralat_idx(ix, iy, d, iw)] - v;
    d = ralat_wrap(iz - 1, RALAT_NZ);
    s += c[ralat_idx(ix, iy, d, iw)] - v;
    d = ralat_wrap(iw + 1, RALAT_NW);
    s += c[ralat_idx(ix, iy, iz, d)] - v;
    d = ralat_wrap(iw - 1, RALAT_NW);
    s += c[ralat_idx(ix, iy, iz, d)] - v;
    return s;
}

/* Sum |f_i| over first four floats of first eligible F32 tensor (same rule as decode probe). */
static int ralat_f32_injection_energy(double *out_e) {
    uint64_t ti;
    *out_e = 0.0;
    for (ti = 0; ti < g_ra_ntensors; ti++) {
        if (!g_ra_tensors[ti].valid) {
            continue;
        }
        if (g_ra_tensors[ti].ggml_type != 0u) {
            continue;
        }
        if (g_ra_tensors[ti].nelems < 4) {
            continue;
        }
        if (g_ra_tensors[ti].nbytes < (int64_t)(4 * sizeof(float))) {
            continue;
        }
        {
            float f[4];
            int i;
            memcpy(f, g_ra_file + g_ra_tensors[ti].abs_off, sizeof f);
            for (i = 0; i < 4; i++) {
                *out_e += fabs((double)f[i]);
            }
            return 0;
        }
    }
    return -1;
}

/*
 * One lattice tick: inject GGUF-derived signal at center (w-aware 4-torus), diffuse, project bias.
 * Defaults align loosely with 4DOllama FOURD_LATTICE_KAPPA / internal D,dt (see coupling_lattice.go).
 * Env: ROMANAI_LATTICE_DISABLE (non-0), ROMANAI_LATTICE_KAPPA, ROMANAI_LATTICE_D, ROMANAI_LATTICE_DT,
 *      ROMANAI_LATTICE_STEPS (unsigned, default 1).
 * Stderr: ROMANAI_LATTICE
 * Returns 0 ok, 1 bad path, 2 layout not loaded.
 */
int32_t mir_romanai_lattice_coupling_step(const char *path) {
    const char *dis = getenv("ROMANAI_LATTICE_DISABLE");
    double kappa = 0.0015;
    double D = 0.014;
    double dt = 0.18;
    unsigned long nsteps = 1;
    const char *es;
    double inj = 0.0;
    int inj_ok;
    unsigned long s;
    int ix;
    int iy;
    int iz;
    int iw;
    int cx;
    int cy;
    int cz;
    int cw;
    size_t ci;
    double center_field;
    double logit_bias;

    if (dis && dis[0] != '\0' && dis[0] != '0') {
        fprintf(stderr, "ROMANAI_LATTICE\tskip\tdisabled\n");
        fflush(stderr);
        return 0;
    }
    if (!path || !*path || strcmp(path, "not_set") == 0) {
        return 1;
    }
    if (!g_ra_file || !g_ra_tensors || g_ra_ntensors == 0) {
        fprintf(stderr, "ROMANAI_LATTICE\terror\tlayout_not_loaded\n");
        fflush(stderr);
        return 2;
    }

    es = getenv("ROMANAI_LATTICE_KAPPA");
    if (es && *es) {
        char *e2 = NULL;
        double v = strtod(es, &e2);
        if (e2 != es && v >= 0.0) {
            kappa = v;
        }
    }
    es = getenv("ROMANAI_LATTICE_D");
    if (es && *es) {
        char *e2 = NULL;
        double v = strtod(es, &e2);
        if (e2 != es && v >= 0.0) {
            D = v;
        }
    }
    es = getenv("ROMANAI_LATTICE_DT");
    if (es && *es) {
        char *e2 = NULL;
        double v = strtod(es, &e2);
        if (e2 != es && v >= 0.0) {
            dt = v;
        }
    }
    es = getenv("ROMANAI_LATTICE_STEPS");
    if (es && *es) {
        char *e2 = NULL;
        unsigned long v = strtoul(es, &e2, 10);
        if (e2 != es && v > 0 && v <= 10000) {
            nsteps = v;
        }
    }

    inj_ok = ralat_f32_injection_energy(&inj);
    if (inj_ok != 0) {
        inj = 1e-9;
    }

    memset(ralat_cur, 0, sizeof ralat_cur);
    memset(ralat_nxt, 0, sizeof ralat_nxt);
    memset(ralat_src, 0, sizeof ralat_src);

    cx = RALAT_NX / 2;
    cy = RALAT_NY / 2;
    cz = RALAT_NZ / 2;
    cw = RALAT_NW / 2;
    ci = ralat_idx(cx, cy, cz, cw);
    ralat_src[ci] = kappa * inj * (1.0 + 0.01);

    for (s = 0; s < nsteps; s++) {
        for (ix = 0; ix < RALAT_NX; ix++) {
            for (iy = 0; iy < RALAT_NY; iy++) {
                for (iz = 0; iz < RALAT_NZ; iz++) {
                    for (iw = 0; iw < RALAT_NW; iw++) {
                        size_t i = ralat_idx(ix, iy, iz, iw);
                        double lap = ralat_lap_periodic(ralat_cur, ix, iy, iz, iw);
                        ralat_nxt[i] = ralat_cur[i] + dt * (D * lap + ralat_src[i]);
                    }
                }
            }
        }
        memcpy(ralat_cur, ralat_nxt, sizeof ralat_cur);
        memset(ralat_src, 0, sizeof ralat_src);
    }

    center_field = ralat_cur[ci];
    logit_bias = tanh(center_field * 0.5) * 8.0;

    fprintf(stderr,
            "ROMANAI_LATTICE\tcoupling\tgrid=%d,%d,%d,%d\tkappa=%g\tD=%g\tdt=%g\tsteps=%lu\t"
            "injection_energy=%g\tf32_tensor_used=%s\tcenter_field=%g\tlogit_bias=%g\n",
            RALAT_NX, RALAT_NY, RALAT_NZ, RALAT_NW, kappa, D, dt, nsteps, inj,
            inj_ok == 0 ? "yes" : "no", center_field, logit_bias);
    fflush(stderr);
    return 0;
}
