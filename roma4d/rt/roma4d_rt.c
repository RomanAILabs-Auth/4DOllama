/* Roma4D CPU runtime stubs for LLVM linking (geometry, list, bump).
 *
 * Builtin constructors and print — declared by LLVM codegen with C linkage.
 * libc: only `puts` is referenced (forward-declared); MinGW/MSVC supply it at link time.
 *
 * Spacetime: compile-time only — no runtime 4D overhead.
 */
typedef long long roma4d_i64;

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Directory scan: POSIX dirent (available with MinGW/MSYS2). File mapping: mmap on Unix;
 * on native Windows with Clang/UCRT, sys/mman.h is often missing — use fread (prototype). */
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

int puts(const char *s);
int system(const char *command);

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- 4-qubit statevector (16 amps), q0 = LSB of basis index ---------------- */
static double qs_re[16];
static double qs_im[16];

static void qs_init_zero(void) {
    int i;
    for (i = 0; i < 16; i++) {
        qs_re[i] = qs_im[i] = 0.0;
    }
    qs_re[0] = 1.0;
}

static void qs_normalize(void) {
    double n = 0.0;
    int i;
    for (i = 0; i < 16; i++) {
        n += qs_re[i] * qs_re[i] + qs_im[i] * qs_im[i];
    }
    n = sqrt(n);
    if (n < 1e-15) {
        return;
    }
    for (i = 0; i < 16; i++) {
        qs_re[i] /= n;
        qs_im[i] /= n;
    }
}

static void qs_h(int q) {
    double nr[16], ni[16];
    int mask = 1 << q;
    int i;
    const double s = 1.0 / sqrt(2.0);
    memcpy(nr, qs_re, sizeof(nr));
    memcpy(ni, qs_im, sizeof(ni));
    for (i = 0; i < 16; i++) {
        if ((i & mask) != 0) {
            continue;
        }
        {
            int j = i ^ mask;
            double ar = qs_re[i], ai = qs_im[i];
            double br = qs_re[j], bi = qs_im[j];
            nr[i] = s * (ar + br);
            ni[i] = s * (ai + bi);
            nr[j] = s * (ar - br);
            ni[j] = s * (ai - bi);
        }
    }
    memcpy(qs_re, nr, sizeof(qs_re));
    memcpy(qs_im, ni, sizeof(qs_im));
}

static void qs_cnot(int c, int t) {
    double nr[16], ni[16];
    int i;
    if (c == t) {
        return;
    }
    /* |...c...t...> -> |...c...t^c...>: amplitude at new basis index */
    for (i = 0; i < 16; i++) {
        int bitc = (i >> c) & 1;
        int bitt = (i >> t) & 1;
        int new_bitt = bitt ^ bitc;
        int j = (i & ~(1 << t)) | (new_bitt << t);
        nr[j] = qs_re[i];
        ni[j] = qs_im[i];
    }
    memcpy(qs_re, nr, sizeof(qs_re));
    memcpy(qs_im, ni, sizeof(qs_im));
}

static void qs_rz(int q, double phi) {
    double cr = cos(phi);
    double sr = sin(phi);
    int i;
    int mask = 1 << q;
    for (i = 0; i < 16; i++) {
        if ((i & mask) == 0) {
            continue;
        }
        {
            double r = qs_re[i], im = qs_im[i];
            qs_re[i] = r * cr - im * sr;
            qs_im[i] = r * sr + im * cr;
        }
    }
}

static double qs_bloch_z_expect(int q) {
    double p0 = 0.0, p1 = 0.0;
    int i;
    int mask = 1 << q;
    for (i = 0; i < 16; i++) {
        double p = qs_re[i] * qs_re[i] + qs_im[i] * qs_im[i];
        if ((i & mask) == 0) {
            p0 += p;
        } else {
            p1 += p;
        }
    }
    return p0 - p1;
}

/* P(q0 = q1) minus P(q0 != q1) for two-qubit marginal (indices 0,1) */
static double qs_zz_corr_01(void) {
    double psame = 0.0, pdiff = 0.0;
    int i;
    for (i = 0; i < 16; i++) {
        double p = qs_re[i] * qs_re[i] + qs_im[i] * qs_im[i];
        int b0 = i & 1;
        int b1 = (i >> 1) & 1;
        if (b0 == b1) {
            psame += p;
        } else {
            pdiff += p;
        }
    }
    return psame - pdiff;
}

static void append_json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            fputc('\\', f);
            fputc((int)c, f);
        } else if (c == '\n') {
            fputs("\\n", f);
        } else if (c == '\r') {
            fputs("\\r", f);
        } else if (c < 32) {
            /* skip control */
        } else {
            fputc((int)c, f);
        }
    }
    fputc('"', f);
}

static void sanitize_query(char *dst, size_t dstsz, const char *src) {
    size_t j = 0;
    if (!src) {
        src = "Summarize entanglement in this state using only the listed probabilities.";
    }
    for (; *src && j + 1 < dstsz; src++) {
        unsigned char c = (unsigned char)*src;
        if (c == '"' || c == '\\') {
            continue;
        }
        if (c < 32 && c != '\n') {
            continue;
        }
        dst[j++] = (char)c;
    }
    dst[j] = '\0';
}

static void qs_fmt_probs(char *buf, size_t bufsz, const double *re, const double *im) {
    int i;
    buf[0] = '\0';
    for (i = 0; i < 16; i++) {
        double p = re[i] * re[i] + im[i] * im[i];
        char line[96];
        if (p < 1e-8) {
            continue;
        }
        snprintf(line, sizeof line, "|%d%d%d%d>: p=%.5f  ",
                 (i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1, p);
        strncat(buf, line, bufsz - strlen(buf) - 1);
    }
}

static void qs_normalize_buf(double *re, double *im) {
    double n = 0.0;
    int i;
    for (i = 0; i < 16; i++) {
        n += re[i] * re[i] + im[i] * im[i];
    }
    n = sqrt(n);
    if (n < 1e-15) {
        return;
    }
    for (i = 0; i < 16; i++) {
        re[i] /= n;
        im[i] /= n;
    }
}

static void qs_rz_buf(int q, double phi, double *re, double *im) {
    double cr = cos(phi);
    double sr = sin(phi);
    int mask = 1 << q;
    int i;
    for (i = 0; i < 16; i++) {
        if ((i & mask) == 0) {
            continue;
        }
        {
            double r = re[i], imv = im[i];
            re[i] = r * cr - imv * sr;
            im[i] = r * sr + imv * cr;
        }
    }
}

static int qs_load_state(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return 0;
    }
    if (fread(qs_re, sizeof(double), 16, f) != 16u) {
        fclose(f);
        return 0;
    }
    if (fread(qs_im, sizeof(double), 16, f) != 16u) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

static void qs_save_state(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        return;
    }
    fwrite(qs_re, sizeof(double), 16, f);
    fwrite(qs_im, sizeof(double), 16, f);
    fclose(f);
}

/*
 * quantum_server_demo — 4-qubit unitary track + Ollama qwen2.5 grounded on probabilities.
 * QUANTUM_QUERY — optional user question (sanitized for JSON).
 * QUANTUM_CONTINUE=1 — load prior amplitudes from TEMP/TMPDIR (same run writes them back).
 * Requires: ollama serve, ollama pull qwen2.5, curl on PATH.
 */
int quantum_server_demo(void) {
    char probbuf[2048];
    char pastbuf[2048];
    char futurebuf[2048];
    char querybuf[512];
    char fullprompt[12288];
    char path[512];
    char statepath[512];
    char cmd[768];
    FILE *fp;
    const char *td;
    const char *qenv;
    const char *qcont;
    int loaded = 0;
    double fut_re[16], fut_im[16];

#ifdef _WIN32
    td = getenv("TEMP");
    if (!td || !*td) {
        td = ".";
    }
    snprintf(statepath, sizeof statepath, "%s\\roma4d_quantum_state.bin", td);
    snprintf(path, sizeof path, "%s\\roma4d_quantum_payload.json", td);
#else
    td = getenv("TMPDIR");
    if (!td || !*td) {
        td = "/tmp";
    }
    snprintf(statepath, sizeof statepath, "%s/roma4d_quantum_state.bin", td);
    snprintf(path, sizeof path, "%s/roma4d_quantum_payload.json", td);
#endif

    qcont = getenv("QUANTUM_CONTINUE");
    if (qcont && qcont[0] == '1' && qs_load_state(statepath)) {
        loaded = 1;
    }

    if (!loaded) {
        qs_init_zero();
        qs_h(0);
        qs_h(1);
        qs_cnot(0, 1);
        qs_fmt_probs(pastbuf, sizeof pastbuf, qs_re, qs_im);
        qs_h(2);
        qs_cnot(1, 2);
        qs_rz(3, M_PI / 4.0);
        qs_cnot(2, 3);
        qs_normalize();
    } else {
        snprintf(pastbuf, sizeof pastbuf,
                 "(state restored via QUANTUM_CONTINUE=1; past slab not recomputed)");
    }

    qs_fmt_probs(probbuf, sizeof probbuf, qs_re, qs_im);
    memcpy(fut_re, qs_re, sizeof fut_re);
    memcpy(fut_im, qs_im, sizeof fut_im);
    qs_rz_buf(0, M_PI / 2.0, fut_re, fut_im);
    qs_normalize_buf(fut_re, fut_im);
    qs_fmt_probs(futurebuf, sizeof futurebuf, fut_re, fut_im);

    puts("");
    puts("  === 4D QUANTUM SERVER — simulated basis snapshot (4 qubits) ===");
    printf("  mode: %s\n", loaded ? "QUANTUM_CONTINUE (state from disk)" : "fresh unitary evolution");
    printf("  simulation_tau: 4 gate epochs (see spacetime: regions in demos/quantum_server.r4d)\n");
    printf("  <Z> q0..q3:  %+.4f  %+.4f  %+.4f  %+.4f\n",
           qs_bloch_z_expect(0), qs_bloch_z_expect(1), qs_bloch_z_expect(2), qs_bloch_z_expect(3));
    printf("  ZZ_corr(q0,q1) marginal: %+.4f  (1=max same-bit bias, -1=max opposite)\n", qs_zz_corr_01());
    puts("  dominant basis probs (present):");
    printf("  %s\n", probbuf);
    puts("  state file (next QUANTUM_CONTINUE=1):");
    printf("  %s\n", statepath);
    puts("  ===============================================================");

    qenv = getenv("QUANTUM_QUERY");
    sanitize_query(querybuf, sizeof querybuf, qenv);

    snprintf(fullprompt, sizeof fullprompt,
             "You are the Offline 4D Quantum Server. You MUST base your answer ONLY on the "
             "simulated quantum data below (real 4-qubit statevector; Hadamard, CNOT, Rz). "
             "Do not invent probabilities. If something is not in the data, say so.\n\n"
             "PAST (after H on q0,q1 and CNOT q0->q1):\n%s\n\n"
             "PRESENT (full circuit):\n%s\n\n"
             "FUTURE (hypothetical one-step: extra Rz(pi/2) on q0 applied to PRESENT copy):\n%s\n\n"
             "<Z> expectations q0..q3: %+.4f %+.4f %+.4f %+.4f\n"
             "ZZ_corr(q0,q1) marginal: %+.4f\n\n"
             "USER QUESTION:\n%s\n\nAnswer clearly and ground claims in the numbers.",
             pastbuf,
             probbuf,
             futurebuf,
             qs_bloch_z_expect(0), qs_bloch_z_expect(1), qs_bloch_z_expect(2), qs_bloch_z_expect(3),
             qs_zz_corr_01(),
             querybuf);

    fp = fopen(path, "wb");
    if (!fp) {
        puts("  [quantum_server] could not write JSON payload file");
        return -1;
    }
    fputs("{\"model\":\"qwen2.5\",\"prompt\":", fp);
    append_json_string(fp, fullprompt);
    fputs(",\"stream\":false}\n", fp);
    fclose(fp);

#ifdef _WIN32
    snprintf(cmd, sizeof cmd,
             "curl -s -S -X POST http://127.0.0.1:11434/api/generate "
             "-H \"Content-Type: application/json\" "
             "-d \"@%s\"",
             path);
#else
    snprintf(cmd, sizeof cmd,
             "curl -s -S -X POST http://127.0.0.1:11434/api/generate "
             "-H 'Content-Type: application/json' "
             "-d @'%s'",
             path);
#endif

    puts("");
    puts("  --- Ollama qwen2.5 (grounded on PAST/PRESENT/FUTURE blocks above) ---");
    fflush(stdout);
    qs_save_state(statepath);
    return system(cmd);
}

static unsigned roma4d_pool_i;
static double roma4d_v4_pool[48][4];

static void *roma4d_next_vec4_slot(void) {
    unsigned i = (roma4d_pool_i++) % 48u;
    return (void *)roma4d_v4_pool[i];
}

int bump(int x) { return x + 1; }

void *vec4(double a0, double a1, double a2, double a3) {
    double *p = (double *)roma4d_next_vec4_slot();
    p[0] = a0;
    p[1] = a1;
    p[2] = a2;
    p[3] = a3;
    return (void *)p;
}

void *rotor(double angle, void *plane_str) {
    double *p;
    (void)angle;
    (void)plane_str;
    p = (double *)roma4d_next_vec4_slot();
    p[0] = 1.0;
    p[1] = 0.0;
    p[2] = 0.0;
    p[3] = 0.0;
    return (void *)p;
}

void *multivector(void) {
    double *p = (double *)roma4d_next_vec4_slot();
    p[0] = 0.0;
    p[1] = 0.0;
    p[2] = 0.0;
    p[3] = 0.0;
    return (void *)p;
}

void *Particle(void) {
    static struct {
        double pos[4];
        double vel[4];
    } cell;
    return (void *)&cell;
}

int print(const char *s) {
    if (!s) {
        s = "";
    }
    return puts(s);
}

void identity_v4(double *out, const double *v) {
    int k;
    if (out && v) {
        for (k = 0; k < 4; k++) {
            out[k] = v[k];
        }
    }
}

void roma4d_geometric_mul_vec4_rotor(double *out, const double *v, const double *r) {
    int k;
    (void)r;
    if (out && v) {
        for (k = 0; k < 4; k++) {
            out[k] = v[k];
        }
    }
}

/*
 * Demo hook: POST to local Ollama /api/generate with model qwen2.5.
 * Requires: `ollama serve` running and `ollama pull qwen2.5`, and `curl` on PATH.
 * JSON is fixed in C because Roma4D has no host string runtime for dynamic bodies yet.
 */
int ollama_demo(void) {
    const char *cmd =
        "curl -s -S -X POST http://127.0.0.1:11434/api/generate "
        "-H \"Content-Type: application/json\" "
        "-d \"{\\\"model\\\":\\\"qwen2.5\\\",\\\"prompt\\\":\\\"You are the Roma4D Causal Oracle. "
        "A spacetime simulation has: (1) a list[vec4] worldtube evolved with par for and rotors, "
        "(2) timetravel_borrow on a causal rotor, (3) compile-time t and expr @ t slices. "
        "Q1: In plain language, what could have caused a collision at t=42? "
        "Q2: What might change if we altered velocity at t=-10? "
        "Answer both in under 120 words.\\\",\\\"stream\\\":false}\"";
    return system(cmd);
}

/* Must match LLVM lowerViewVec4List { double*; i64; i64 } layout (64-bit). */
typedef struct {
    double *data;
    roma4d_i64 len;
    roma4d_i64 cap;
} roma4d_list_vec4_hdr;

void roma4d_list_get_vec4(void *lst, roma4d_i64 i, double *out) {
    int k;
    roma4d_list_vec4_hdr *h = (roma4d_list_vec4_hdr *)lst;
    if (!out) {
        return;
    }
    if (!lst || !h->data || i < 0 || i >= h->len) {
        for (k = 0; k < 4; k++) {
            out[k] = 0.0;
        }
        return;
    }
    memcpy(out, h->data + i * 4, 4 * sizeof(double));
}

void roma4d_list_set_vec4(void *lst, roma4d_i64 i, const double *v) {
    roma4d_list_vec4_hdr *h = (roma4d_list_vec4_hdr *)lst;
    if (!lst || !h->data || !v || i < 0 || i >= h->len) {
        return;
    }
    memcpy(h->data + i * 4, v, 4 * sizeof(double));
}

/* ---- mmap GGUF / Ollama blob path / interactive qwen chat ------------------ */

void *mir_mmap_gguf(const char *path) {
    if (!path || strcmp(path, "not_found") == 0) {
        return NULL;
    }
#ifndef _WIN32
    {
        int fd;
        struct stat st;
        void *p;
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            return NULL;
        }
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            close(fd);
            return NULL;
        }
        p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) {
            return NULL;
        }
        return p;
    }
#else
    {
        FILE *fp;
        long sz;
        void *p;
        size_t cap = (size_t)1000000 * 4 * sizeof(double);
        size_t to_read;
        fp = fopen(path, "rb");
        if (!fp) {
            return NULL;
        }
        if (fseek(fp, 0, SEEK_END) != 0) {
            fclose(fp);
            return NULL;
        }
        sz = ftell(fp);
        if (sz <= 0) {
            fclose(fp);
            return NULL;
        }
        if (fseek(fp, 0, SEEK_SET) != 0) {
            fclose(fp);
            return NULL;
        }
        to_read = (size_t)sz > cap ? cap : (size_t)sz;
        p = malloc(to_read);
        if (!p) {
            fclose(fp);
            return NULL;
        }
        if (fread(p, 1, to_read, fp) != to_read) {
            free(p);
            fclose(fp);
            return NULL;
        }
        fclose(fp);
        return p;
    }
#endif
}

static char g_qwen_blob_path[1024];
static const char g_not_found_lit[] = "not_found";

const char *mir_get_ollama_qwen_path(void) {
    char dir[768];
    DIR *d;
    struct dirent *e;
    unsigned long long best = 0;

#ifdef _WIN32
    {
        const char *up = getenv("USERPROFILE");
        if (!up || !*up) {
            return g_not_found_lit;
        }
        snprintf(dir, sizeof dir, "%s\\.ollama\\models\\blobs", up);
    }
#else
    {
        const char *home = getenv("HOME");
        if (!home || !*home) {
            return g_not_found_lit;
        }
        snprintf(dir, sizeof dir, "%s/.ollama/models/blobs", home);
    }
#endif

    d = opendir(dir);
    if (!d) {
        return g_not_found_lit;
    }
    g_qwen_blob_path[0] = '\0';
    while ((e = readdir(d)) != NULL) {
        char full[1024];
        struct stat st;
        if (e->d_name[0] == '.') {
            continue;
        }
#ifdef _WIN32
        snprintf(full, sizeof full, "%s\\%s", dir, e->d_name);
#else
        snprintf(full, sizeof full, "%s/%s", dir, e->d_name);
#endif
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }
        {
            unsigned long long sz = (unsigned long long)st.st_size;
            if (sz > best) {
                best = sz;
                snprintf(g_qwen_blob_path, sizeof g_qwen_blob_path, "%s", full);
            }
        }
    }
    closedir(d);
    if (best == 0 || g_qwen_blob_path[0] == '\0') {
        return g_not_found_lit;
    }
    return g_qwen_blob_path;
}

/* Ollama /api/generate with stream:true returns NDJSON lines; each has a "response" string fragment. */
static void print_ollama_stream_response_field(const char *line) {
    const char *k = strstr(line, "\"response\"");
    const char *p;
    if (!k) {
        return;
    }
    p = strchr(k, ':');
    if (!p) {
        return;
    }
    p++;
    while (*p == ' ' || *p == '\t') {
        p++;
    }
    if (*p != '"') {
        return;
    }
    p++;
    while (*p) {
        if (*p == '\\') {
            p++;
            if (*p == 'n') {
                putchar('\n');
            } else if (*p == 't') {
                putchar('\t');
            } else if (*p == 'r') {
                putchar('\r');
            } else if (*p == '"' || *p == '\\') {
                putchar((unsigned char)*p);
            } else if (*p) {
                putchar((unsigned char)*p);
            }
            if (*p) {
                p++;
            }
            continue;
        }
        if (*p == '"') {
            break;
        }
        putchar((unsigned char)*p);
        p++;
    }
    fflush(stdout);
}

#ifdef _WIN32
#define R4D_popen  _popen
#define R4D_pclose _pclose
#else
#define R4D_popen  popen
#define R4D_pclose pclose
#endif

int mir_qwen_chat_loop(void) {
    char line[2048];
    char path[512];
    char cmd[2048];
    char streambuf[65536];
    FILE *fp;
    FILE *pipef;
    const char *td;

#ifdef _WIN32
    td = getenv("TEMP");
    if (!td || !*td) {
        td = ".";
    }
    snprintf(path, sizeof path, "%s\\roma4d_qwen_chat.json", td);
#else
    td = getenv("TMPDIR");
    if (!td || !*td) {
        td = "/tmp";
    }
    snprintf(path, sizeof path, "%s/roma4d_qwen_chat.json", td);
#endif

    puts("");
    puts("Roma4D mir_qwen_chat_loop — streaming tokens (curl -N). Type 'exit' to quit.");
    puts("  Ollama: http://127.0.0.1:11434  model: qwen2.5  (keep_alive 10m keeps weights hot.)");
    puts("  Note: reply speed is Ollama/GPU-bound; the Roma4D par pass above is a separate demo, not NN inference.");
    fflush(stdout);

    while (fgets(line, sizeof line, stdin)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
            line[--n] = '\0';
        }
        if (strcmp(line, "exit") == 0) {
            break;
        }
        fp = fopen(path, "wb");
        if (!fp) {
            puts("[mir_qwen_chat_loop] could not write JSON payload file");
            continue;
        }
        fputs("{\"model\":\"qwen2.5\",\"prompt\":", fp);
        append_json_string(fp, line);
        fputs(",\"stream\":true,\"keep_alive\":\"10m\"}\n", fp);
        fclose(fp);

#ifdef _WIN32
        snprintf(cmd, sizeof cmd,
                 "curl -sS -N -X POST http://127.0.0.1:11434/api/generate "
                 "-H \"Content-Type: application/json\" "
                 "-d \"@%s\"",
                 path);
#else
        snprintf(cmd, sizeof cmd,
                 "curl -sS -N -X POST http://127.0.0.1:11434/api/generate "
                 "-H 'Content-Type: application/json' "
                 "-d @'%s'",
                 path);
#endif
        pipef = R4D_popen(cmd, "r");
        if (!pipef) {
            puts("[mir_qwen_chat_loop] popen(curl) failed; is curl on PATH?");
            continue;
        }
        while (fgets(streambuf, (int)sizeof streambuf, pipef) != NULL) {
            print_ollama_stream_response_field(streambuf);
        }
        R4D_pclose(pipef);
        putchar('\n');
        fflush(stdout);
    }
    return 0;
}
