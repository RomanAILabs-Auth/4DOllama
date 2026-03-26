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
    printf("  simulation_tau: 4 gate epochs (see spacetime: regions in demos/quantum_server.roma4d)\n");
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

void roma4d_list_get_vec4(void *lst, roma4d_i64 i, double *out) {
    int k;
    (void)lst;
    (void)i;
    if (out) {
        for (k = 0; k < 4; k++) {
            out[k] = 0.0;
        }
    }
}
