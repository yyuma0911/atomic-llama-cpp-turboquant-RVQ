/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit) and
 * GGML_TYPE_TURBO4_0 (4-bit) for use as --cache-type-k turboN in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Global: WHT group size for CPU quantize path (set by CPU SET_ROWS handler) */
GGML_API int turbo3_cpu_wht_group_size = 0;

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.145560f) return 0;
    if (val < -0.103361f) return 1;
    if (val < -0.079142f) return 2;
    if (val < -0.060009f) return 3;
    if (val < -0.043430f) return 4;
    if (val < -0.028293f) return 5;
    if (val < -0.013963f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.013963f) return 8;
    if (val <  0.028293f) return 9;
    if (val <  0.043430f) return 10;
    if (val <  0.060009f) return 11;
    if (val <  0.079142f) return 12;
    if (val <  0.103361f) return 13;
    if (val <  0.145560f) return 14;
    return 15;
}

/* ---------- WHT sign arrays (must match CUDA/Metal, seed=42) ---------- */

static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

/* ---------- CPU forward WHT (in-place, group_size elements) ---------- */

static void turbo_cpu_fwht(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

/* ---------- CPU inverse WHT (in-place, group_size elements) ----------
 *
 * Forward is  y = D(s2) * N * H * D(s1) * x   (N = 1/sqrt(group_size))
 * H is the unnormalized Hadamard butterfly with H*H = group_size * I, so
 * (N*H) is self-inverse.  s1 and s2 are ±1 diagonals, also self-inverse.
 * The inverse therefore has the same structure with s1 and s2 swapped:
 *     x = D(s1) * N * H * D(s2) * y
 */
GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs2 (undoes the s2 that was applied last in the forward pass)
    for (int i = 0; i < group_size; i++) x[i] *= s2[i];

    // butterfly stages (same as forward — self-inverse up to the inv_sqrt scaling below)
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs1
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s1[i];
}

/* ---------- TURBO3_0: 3-bit PolarQuant with WHT rotation ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    // Read WHT group size from global (set by CPU SET_ROWS handler before each call).
    // Fallback: 128 if row is 128-aligned, else 64.
    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo3_0 * grp_dst = y + g * blocks_per_group;

        // 1. L2 norm over the group
        float norm_sq = 0.0f;
        float buf[128];  // max group_size
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        // 2. Normalize
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        // 3. Forward WHT rotation
        turbo_cpu_fwht(buf, group_size);

        // 4. Quantize + pack into sub-blocks
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO3;

            memset(blk->qs, 0, QK_TURBO3 / 4);
            memset(blk->signs, 0, QK_TURBO3 / 8);

            for (int j = 0; j < QK_TURBO3; j++) {
                int idx = nearest_centroid_3bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                recon_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
            }
        }

        // 5. Corrected norm: grp_norm / recon_norm (matching CUDA kernel)
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_0: 2-bit PolarQuant (no QJL) ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO2;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo2_0 * grp_dst = y + g * blocks_per_group;

        /* 1. L2 norm over the group */
        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        /* 3. Forward WHT rotation */
        turbo_cpu_fwht(buf, group_size);

        /* 4. Quantize + pack into sub-blocks */
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo2_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO2;

            memset(blk->qs, 0, QK_TURBO2 / 4);

            for (int j = 0; j < QK_TURBO2; j++) {
                int idx = nearest_centroid_2bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
            }
        }

        /* 5. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 3-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Forward WHT rotation (matches CUDA set_rows) */
        float rotated[TURBO_D];
        memcpy(rotated, normalized, d * sizeof(float));
        turbo_cpu_fwht(rotated, d);

#if TURBO4_USE_4BIT
        /* Step 3: 4-bit quantization (16 centroids) */
        static const float CENTROIDS_4BIT[16] = {
            -0.173926f, -0.117195f, -0.089527f, -0.068756f,
            -0.051262f, -0.035597f, -0.020989f, -0.006938f,
             0.006938f,  0.020989f,  0.035597f,  0.051262f,
             0.068756f,  0.089527f,  0.117195f,  0.173926f
        };
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Norm correction */
        float recon_norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            recon_norm_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);
#else
        /* Step 3: 3-bit quantization (8 centroids) */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Step 4: Residual */
        float reconstructed[TURBO_D];
        for (int i = 0; i < d; i++) {
            reconstructed[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[TURBO_D];
        for (int i = 0; i < d; i++) {
            residual[i] = normalized[i] - mse_recon[i];
        }

        /* Step 5: QJL */
        float projected[TURBO_D];
        matvec(turbo_qjl_matrix, residual, projected, d);
#endif

        /* Pack */
#if !TURBO4_USE_4BIT
        y[block].norm  = GGML_FP32_TO_FP16(norm);
#endif

#if TURBO4_USE_4BIT
        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
#else
        /* Legacy 3-bit + QJL: pack 3-bit indices + QJL signs */
        memset(y[block].qs, 0, d * 3 / 8);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t val   = (uint16_t)(indices[i] & 0x7);
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
            }
        }
        memset(y[block].signs, 0, d / 8);
        for (int i = 0; i < d; i++) {
            if (projected[i] >= 0.0f) {
                y[block].signs[i / 8] |= (1 << (i % 8));
            }
        }
#endif
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

#if TURBO4_USE_4BIT
    /* 4-bit PolarQuant: nibble unpack → centroid → inverse rotate → scale */
    /* TODO: add proper 4-bit centroid table to C code (currently only in Metal) */
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
        }
        /* No inverse WHT, dequant stays in the rotated domain.
        * Q is WHT-rotated by the graph, so <Q_rot, K_rot> gives correct attention scores.
        * The inverse WHT is applied to the attention output via GGML_OP_TURBO_WHT (direction=1) in the graph. 
        */
    }
#else
    /* Legacy 3-bit + QJL dequant */
    turbo_init_qjl();
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);

        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t raw   = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        float signs[TURBO_D];
        for (int i = 0; i < d; i++) {
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        }

        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;

        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        float qjl_recon[TURBO_D];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) {
            qjl_recon[i] *= qjl_scale;
        }

        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
        }
    }
#endif
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ================================================================== */
/* TQ3_1S / TQ4_1S: WHT-rotated weight quantization                  */
/* ================================================================== */

/* Lloyd-Max centroids for N(0,1) — shared with Metal shaders */
static const float TQ3_0_CENTROIDS[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};

static const float TQ4_0_CENTROIDS[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};

/* WHT sign pattern (golden ratio hash, 32-element blocks) — shared by TQ3 and TQ4 */
static const float TQ3_0_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

#define TQ_BLOCK_SIZE 32
#define TQ_INV_SQRT32 0.17677669529663688f  /* 1/sqrt(32) */

/* Forward RHT: sign flips -> WHT butterfly -> normalize */
static void tq3_0_rht_forward(float * buf) {
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ3_0_SIGNS[i];
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32;
}

/* Inverse RHT: WHT butterfly -> normalize + unsign */
static void tq3_0_rht_inverse(float * buf) {
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32 * TQ3_0_SIGNS[i];
}

/* Nearest centroid for TQ3 (8 centroids) */
static int tq3_0_choose_index(float val) {
    /* Binary search on midpoints of TQ3_0_CENTROIDS */
    if (val < -1.644041f) return 0;
    if (val < -1.015870f) return 1;
    if (val < -0.493925f) return 2;
    if (val < -0.008701f) return 3;
    if (val <  0.477664f) return 4;
    if (val <  1.001363f) return 5;
    if (val <  1.633223f) return 6;
    return 7;
}

/* Nearest centroid for TQ4 (16 centroids) */
static int tq4_0_choose_index(float val) {
    /* Binary search on midpoints of TQ4_0_CENTROIDS */
    if (val < -2.400804f) return 0;
    if (val < -1.843532f) return 1;
    if (val < -1.437139f) return 2;
    if (val < -1.099286f) return 3;
    if (val < -0.799550f) return 4;
    if (val < -0.522404f) return 5;
    if (val < -0.258222f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.258222f) return 8;
    if (val <  0.522404f) return 9;
    if (val <  0.799550f) return 10;
    if (val <  1.099286f) return 11;
    if (val <  1.437139f) return 12;
    if (val <  1.843532f) return 13;
    if (val <  2.400804f) return 14;
    return 15;
}

/* ---------- TQ3_1S quantization ---------- */

void quantize_row_tq3_1s_ref(const float * GGML_RESTRICT x, block_tq3_1s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;

    for (int block = 0; block < nb; block++) {
        const float * src_blk = x + block * QK_TQ3_0;
        block_tq3_1s * blk = &y[block];

        /* 1. Forward RHT */
        float buf[TQ_BLOCK_SIZE];
        memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
        tq3_0_rht_forward(buf);

        /* 2. Split into two halves, compute RMS per half */
        float rms0 = 0.0f, rms1 = 0.0f;
        for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
        for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
        rms0 = sqrtf(rms0 / 16.0f);
        rms1 = sqrtf(rms1 / 16.0f);

        /* 3. Scale search (9 points) */
        static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
        float best_d0 = rms0, best_d1 = rms1;
        float best_err = 1e30f;

        for (int si = 0; si < 9; si++) {
            float d0 = rms0 * scales[si];
            float d1 = rms1 * scales[si];
            float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
            float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

            float err = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv0);
                float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d0;
                err += diff * diff;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv1);
                float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d1;
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_d0 = d0;
                best_d1 = d1;
            }
        }

        /* 4. Iterative refinement (6 iterations) */
        for (int iter = 0; iter < 6; iter++) {
            float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
            float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

            float num0 = 0.0f, den0 = 0.0f;
            float num1 = 0.0f, den1 = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv0);
                float c = TQ3_0_CENTROIDS[idx];
                num0 += buf[j] * c;
                den0 += c * c;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv1);
                float c = TQ3_0_CENTROIDS[idx];
                num1 += buf[j] * c;
                den1 += c * c;
            }
            if (den0 > 1e-10f) best_d0 = num0 / den0;
            if (den1 > 1e-10f) best_d1 = num1 / den1;
        }

        /* 5. Final quantize + pack */
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        blk->d0 = GGML_FP32_TO_FP16(best_d0);
        blk->d1 = GGML_FP32_TO_FP16(best_d1);
        memset(blk->qs, 0, QK_TQ3_0 * 3 / 8);

        /* TQ3 packing: 4 groups of 8 indices packed into 3 bytes each */
        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            for (int i = 0; i < 8; i++) {
                int j = g * 8 + i;
                float inv = (j < 16) ? inv0 : inv1;
                indices[i] = (uint8_t)tq3_0_choose_index(buf[j] * inv);
            }
            uint8_t * qp = blk->qs + g * 3;
            qp[0] = (indices[0] & 7) | ((indices[1] & 7) << 3) | ((indices[2] & 3) << 6);
            qp[1] = ((indices[2] >> 2) & 1) | ((indices[3] & 7) << 1) | ((indices[4] & 7) << 4) | ((indices[5] & 1) << 7);
            qp[2] = ((indices[5] >> 1) & 3) | ((indices[6] & 7) << 2) | ((indices[7] & 7) << 5);
        }
    }
}

void dequantize_row_tq3_1s(const block_tq3_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        /* Unpack 3-bit indices */
        float buf[32];
        for (int g = 0; g < 4; g++) {
            const uint8_t * qp = x[blk_i].qs + g * 3;
            uint8_t idx[8];
            idx[0] =  qp[0]       & 7;
            idx[1] = (qp[0] >> 3) & 7;
            idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
            idx[3] = (qp[1] >> 1) & 7;
            idx[4] = (qp[1] >> 4) & 7;
            idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
            idx[6] = (qp[2] >> 2) & 7;
            idx[7] = (qp[2] >> 5) & 7;

            for (int i = 0; i < 8; i++) {
                int j = g * 8 + i;
                float d = (j < 16) ? d0 : d1;
                buf[j] = TQ3_0_CENTROIDS[idx[i]] * d;
            }
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_TQ3_0, buf, QK_TQ3_0 * sizeof(float));
    }
}

size_t quantize_tq3_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TQ3_0 == 0);

    size_t row_size = (n_per_row / QK_TQ3_0) * sizeof(block_tq3_1s);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq3_1s_ref(
            src + row * n_per_row,
            (block_tq3_1s *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TQ4_1S quantization ---------- */

void quantize_row_tq4_1s_ref(const float * GGML_RESTRICT x, block_tq4_1s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4_1S == 0);
    const int nb = k / QK_TQ4_1S;

    for (int block = 0; block < nb; block++) {
        const float * src_blk = x + block * QK_TQ4_1S;
        block_tq4_1s * blk = &y[block];

        /* 1. Forward RHT */
        float buf[TQ_BLOCK_SIZE];
        memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
        tq3_0_rht_forward(buf);

        /* 2. Split into two halves, compute RMS per half */
        float rms0 = 0.0f, rms1 = 0.0f;
        for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
        for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
        rms0 = sqrtf(rms0 / 16.0f);
        rms1 = sqrtf(rms1 / 16.0f);

        /* 3. Scale search (9 points) */
        static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
        float best_d0 = rms0, best_d1 = rms1;
        float best_err = 1e30f;

        for (int si = 0; si < 9; si++) {
            float d0 = rms0 * scales[si];
            float d1 = rms1 * scales[si];
            float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
            float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

            float err = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv0);
                float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d0;
                err += diff * diff;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv1);
                float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d1;
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_d0 = d0;
                best_d1 = d1;
            }
        }

        /* 4. Iterative refinement (6 iterations) */
        for (int iter = 0; iter < 6; iter++) {
            float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
            float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

            float num0 = 0.0f, den0 = 0.0f;
            float num1 = 0.0f, den1 = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv0);
                float c = TQ4_0_CENTROIDS[idx];
                num0 += buf[j] * c;
                den0 += c * c;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv1);
                float c = TQ4_0_CENTROIDS[idx];
                num1 += buf[j] * c;
                den1 += c * c;
            }
            if (den0 > 1e-10f) best_d0 = num0 / den0;
            if (den1 > 1e-10f) best_d1 = num1 / den1;
        }

        /* 5. Final quantize + pack (nibble packing) */
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        blk->d0 = GGML_FP32_TO_FP16(best_d0);
        blk->d1 = GGML_FP32_TO_FP16(best_d1);
        memset(blk->qs, 0, QK_TQ4_1S / 2);

        for (int j = 0; j < QK_TQ4_1S; j++) {
            float inv = (j < 16) ? inv0 : inv1;
            int idx = tq4_0_choose_index(buf[j] * inv);
            blk->qs[j / 2] |= (uint8_t)((idx & 0xF) << ((j & 1) * 4));
        }
    }
}

void dequantize_row_tq4_1s(const block_tq4_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4_1S == 0);
    const int nb = k / QK_TQ4_1S;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        float buf[32];
        for (int j = 0; j < 32; j++) {
            uint8_t idx = (x[blk_i].qs[j / 2] >> ((j & 1) * 4)) & 0xF;
            float d = (j < 16) ? d0 : d1;
            buf[j] = TQ4_0_CENTROIDS[idx] * d;
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_TQ4_1S, buf, QK_TQ4_1S * sizeof(float));
    }
}

size_t quantize_tq4_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TQ4_1S == 0);

    size_t row_size = (n_per_row / QK_TQ4_1S) * sizeof(block_tq4_1s);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq4_1s_ref(
            src + row * n_per_row,
            (block_tq4_1s *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

// ============================================================
// TQ3_RVQ: Residual Vector Quantization for 3-bit weights
// ============================================================

// Default codebooks for TQ3_RVQ
// Stage 2 (16-level): multiplicative scale correction for 32-element blocks
// Pre-optimized universal default for LLM linear layers
static const float TQ3_RVQ_CB2_DEFAULT[16] = {
    0.70f, 0.78f, 0.85f, 0.90f, 0.94f, 0.97f, 0.99f, 1.00f,
    1.02f, 1.04f, 1.07f, 1.11f, 1.16f, 1.23f, 1.31f, 1.40f
};

// Stage 3 (4-level): fine-grained multiplicative scale for 8-element sub-blocks
static const float TQ3_RVQ_CB3_DEFAULT[4] = {
    0.88f, 0.95f, 1.05f, 1.12f
};

// Base 3-bit centroids (initialized with Lloyd-Max N(0,1) defaults)
static float tq3_rvq_centroids[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};
static float tq3_rvq_k = 1.07f; // Default shape parameter k (Lloyd-Max N(0,1) optimal)

// Weak link to CUDA codebook sync function (defined in mmvq-tq.cu and convert.cu)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak)) void ggml_cuda_tq3_rvq_sync_codebook_mmvq(const float * centroids, const float * cb2, const float * cb3);
__attribute__((weak)) void ggml_cuda_tq3_rvq_sync_codebook_convert(const float * centroids, const float * cb2, const float * cb3);
__attribute__((weak)) void ggml_cuda_tq3_rvq_sync_codebook_getrows(const float * centroids, const float * cb2, const float * cb3);
#else
static void (*ggml_cuda_tq3_rvq_sync_codebook_mmvq)(const float * centroids, const float * cb2, const float * cb3) = NULL;
static void (*ggml_cuda_tq3_rvq_sync_codebook_convert)(const float * centroids, const float * cb2, const float * cb3) = NULL;
static void (*ggml_cuda_tq3_rvq_sync_codebook_getrows)(const float * centroids, const float * cb2, const float * cb3) = NULL;
#endif

// Global codebooks for RVQ stages (initialized with static defaults at compile time)
// Use tq3_rvq_set_codebook() to override with model-specific values at runtime
static float tq3_rvq_cb2[16] = {
    0.70f, 0.78f, 0.85f, 0.90f, 0.94f, 0.97f, 0.99f, 1.00f,
    1.02f, 1.04f, 1.07f, 1.11f, 1.16f, 1.23f, 1.31f, 1.40f
};
static float tq3_rvq_cb3[4] = {
    0.88f, 0.95f, 1.05f, 1.12f
};
static bool  tq3_rvq_cb_initialized = true;
static bool  tq3_rvq_global_fitted = false;

void tq3_rvq_set_codebook(const float * cb2, const float * cb3) {
    memcpy(tq3_rvq_cb2, cb2, sizeof(tq3_rvq_cb2));
    memcpy(tq3_rvq_cb3, cb3, sizeof(tq3_rvq_cb3));
    tq3_rvq_cb_initialized = true;

    if (ggml_cuda_tq3_rvq_sync_codebook_mmvq) {
        ggml_cuda_tq3_rvq_sync_codebook_mmvq(tq3_rvq_centroids, tq3_rvq_cb2, tq3_rvq_cb3);
    }
    if (ggml_cuda_tq3_rvq_sync_codebook_convert) {
        ggml_cuda_tq3_rvq_sync_codebook_convert(tq3_rvq_centroids, tq3_rvq_cb2, tq3_rvq_cb3);
    }
    if (ggml_cuda_tq3_rvq_sync_codebook_getrows) {
        ggml_cuda_tq3_rvq_sync_codebook_getrows(tq3_rvq_centroids, tq3_rvq_cb2, tq3_rvq_cb3);
    }
}

void tq3_rvq_get_codebook(float * cb2_out, float * cb3_out) {
    memcpy(cb2_out, tq3_rvq_cb2, sizeof(tq3_rvq_cb2));
    memcpy(cb3_out, tq3_rvq_cb3, sizeof(tq3_rvq_cb3));
}

// Forward declaration for K-means helper (defined below)
void tq3_rvq_kmeans(const float * samples, int n_samples, float * centroids, int k, int n_iter);

void tq3_rvq_set_centroids(const float * centroids) {
    memcpy(tq3_rvq_centroids, centroids, sizeof(tq3_rvq_centroids));

    if (ggml_cuda_tq3_rvq_sync_codebook_mmvq) {
        ggml_cuda_tq3_rvq_sync_codebook_mmvq(tq3_rvq_centroids, tq3_rvq_cb2, tq3_rvq_cb3);
    }
    if (ggml_cuda_tq3_rvq_sync_codebook_convert) {
        ggml_cuda_tq3_rvq_sync_codebook_convert(tq3_rvq_centroids, tq3_rvq_cb2, tq3_rvq_cb3);
    }
    if (ggml_cuda_tq3_rvq_sync_codebook_getrows) {
        ggml_cuda_tq3_rvq_sync_codebook_getrows(tq3_rvq_centroids, tq3_rvq_cb2, tq3_rvq_cb3);
    }
}

void tq3_rvq_get_centroids(float * centroids_out) {
    memcpy(centroids_out, tq3_rvq_centroids, sizeof(tq3_rvq_centroids));
}

void tq3_rvq_fit_centroids(const float * samples, int n_samples) {
    if (n_samples <= 0) {
        fprintf(stderr, "[TQ3_RVQ] Warning: tq3_rvq_fit_centroids called with %d samples, skipping\n", n_samples);
        return;
    }
    float new_centroids[8];
    tq3_rvq_kmeans(samples, n_samples, new_centroids, 8, 20);
    tq3_rvq_set_centroids(new_centroids);

    fprintf(stderr, "[TQ3_RVQ] Fitted data-driven centroids via K-means:\n  ");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "%.4f ", new_centroids[i]);
    }
    fprintf(stderr, "\n");
}


static int tq3_rvq_choose_index(float val) {
    float m0 = (tq3_rvq_centroids[0] + tq3_rvq_centroids[1]) * 0.5f;
    float m1 = (tq3_rvq_centroids[1] + tq3_rvq_centroids[2]) * 0.5f;
    float m2 = (tq3_rvq_centroids[2] + tq3_rvq_centroids[3]) * 0.5f;
    float m3 = (tq3_rvq_centroids[3] + tq3_rvq_centroids[4]) * 0.5f;
    float m4 = (tq3_rvq_centroids[4] + tq3_rvq_centroids[5]) * 0.5f;
    float m5 = (tq3_rvq_centroids[5] + tq3_rvq_centroids[6]) * 0.5f;
    float m6 = (tq3_rvq_centroids[6] + tq3_rvq_centroids[7]) * 0.5f;

    if (val < m0) return 0;
    if (val < m1) return 1;
    if (val < m2) return 2;
    if (val < m3) return 3;
    if (val < m4) return 4;
    if (val < m5) return 5;
    if (val < m6) return 6;
    return 7;
}

void tq3_rvq_init_from_k(float k) {
    tq3_rvq_k = k;
    for (int i = 0; i < 8; i++) {
        // Equal-spaced index t_i in [-1.0, 1.0]
        float t = -1.0f + i * (2.0f / 7.0f);
        float sign = (t >= 0.0f) ? 1.0f : -1.0f;
        // Scale factor: Lloyd-Max maximum centroid magnitude (1.996684f)
        tq3_rvq_centroids[i] = sign * powf(fabsf(t), k) * 1.996684f;
    }

    fprintf(stderr, "[TQ3_RVQ] Generated k-sparsity centroids with k = %.3f:\n", k);
    fprintf(stderr, "  ");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "%.4f ", tq3_rvq_centroids[i]);
    }
    fprintf(stderr, "\n");

    if (ggml_cuda_tq3_rvq_sync_codebook_mmvq) {
        ggml_cuda_tq3_rvq_sync_codebook_mmvq(tq3_rvq_centroids, tq3_rvq_cb2, tq3_rvq_cb3);
    }
    if (ggml_cuda_tq3_rvq_sync_codebook_convert) {
        ggml_cuda_tq3_rvq_sync_codebook_convert(tq3_rvq_centroids, tq3_rvq_cb2, tq3_rvq_cb3);
    }
    if (ggml_cuda_tq3_rvq_sync_codebook_getrows) {
        ggml_cuda_tq3_rvq_sync_codebook_getrows(tq3_rvq_centroids, tq3_rvq_cb2, tq3_rvq_cb3);
    }
}

static void quantize_row_tq3_rvq_impl(const float * GGML_RESTRICT x, block_tq3_rvq * GGML_RESTRICT y, int64_t k, const float * imatrix,
                                       const float * cb2, const float * cb3) {
    assert(k % QK_TQ3_RVQ == 0);
    const int nb = k / QK_TQ3_RVQ;
    for (int blk = 0; blk < nb; blk++) {
        const float * src = x + blk * QK_TQ3_RVQ;
        const float * blk_imatrix = imatrix ? (imatrix + blk * QK_TQ3_RVQ) : NULL;
        block_tq3_rvq * dst = &y[blk];

        // 1. Forward RHT per 32-element sub-block (same pattern as TQ3_1S/TQ4_1S)
        float buf[QK_TQ3_RVQ];
        memcpy(buf, src, QK_TQ3_RVQ * sizeof(float));
        for (int g = 0; g < 8; g++) {
            tq3_0_rht_forward(buf + g * 32);
        }

        // Compute rotated importance weights (mean over each 32-element sub-block)
        float w[QK_TQ3_RVQ];
        if (blk_imatrix) {
            for (int g = 0; g < 8; g++) {
                float sum_w = 0.0f;
                for (int i = 0; i < 32; i++) {
                    sum_w += blk_imatrix[g * 32 + i];
                }
                float mean_w = sum_w / 32.0f;
                for (int i = 0; i < 32; i++) {
                    w[g * 32 + i] = mean_w;
                }
            }
        } else {
            for (int i = 0; i < QK_TQ3_RVQ; i++) {
                w[i] = 1.0f;
            }
        }

        // 2. Compute super-block scale (MAE-based robust initialization)
        float sum_abs = 0.0f;
        for (int i = 0; i < QK_TQ3_RVQ; i++) {
            sum_abs += fabsf(buf[i]);
        }
        float mae = sum_abs / QK_TQ3_RVQ;
        // Convert robust MAE to estimated standard deviation (RMS) for normal distribution
        float rms = mae * 1.253314f;

        // Smart initialization of s2_vals and s3_vals using local RMS ratios
        float s2_vals[8];
        float s3_vals[32];
        for (int b2 = 0; b2 < 8; b2++) {
            float local_sum = 0.0f;
            for (int i = 0; i < 32; i++) {
                local_sum += buf[b2 * 32 + i] * buf[b2 * 32 + i];
            }
            float local_rms = sqrtf(local_sum / 32.0f);
            float ratio = (rms > 1e-10f) ? (local_rms / rms) : 1.0f;
            int best_idx = 0;
            float best_diff = FLT_MAX;
            for (int ci = 0; ci < 16; ci++) {
                float diff = fabsf(cb2[ci] - ratio);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_idx = ci;
                }
            }
            s2_vals[b2] = cb2[best_idx];
        }
        for (int b3 = 0; b3 < 32; b3++) {
            int b2 = b3 / 4;
            float sub_sum = 0.0f;
            for (int i = 0; i < 8; i++) {
                sub_sum += buf[b3 * 8 + i] * buf[b3 * 8 + i];
            }
            float sub_rms = sqrtf(sub_sum / 8.0f);
            float target_rms = rms * s2_vals[b2];
            float ratio = (target_rms > 1e-10f) ? (sub_rms / target_rms) : 1.0f;
            int best_idx = 0;
            float best_diff = FLT_MAX;
            for (int ci = 0; ci < 4; ci++) {
                float diff = fabsf(cb3[ci] - ratio);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_idx = ci;
                }
            }
            s3_vals[b3] = cb3[best_idx];
        }

        // 3. Grid search on initial scale (RMS) with imatrix support (9 points)
        static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
        float best_rms = rms;
        float best_err = FLT_MAX;
        for (int si = 0; si < 9; si++) {
            float test_rms = rms * scales[si];
            float err = 0.0f;
            for (int i = 0; i < QK_TQ3_RVQ; i++) {
                float scale = test_rms * s2_vals[i / 32] * s3_vals[i / 8];
                float val = (scale > 1e-10f) ? (buf[i] / scale) : 0.0f;
                int q = tq3_rvq_choose_index(val);
                float restored = tq3_rvq_centroids[q] * scale;
                float diff = buf[i] - restored;
                err += w[i] * diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_rms = test_rms;
            }
        }
        rms = best_rms;

        // Local flat arrays for holding temporary states
        uint8_t q_vals[QK_TQ3_RVQ];

        // 4. Iterative refinement (6 iterations)
        const int N_ITER = 6;
        for (int iter = 0; iter < N_ITER; iter++) {
            // Step A: Optimize 3-bit base indices q_vals
            for (int i = 0; i < QK_TQ3_RVQ; i++) {
                float scale = rms * s2_vals[i / 32] * s3_vals[i / 8];
                float val = (scale > 1e-10f) ? (buf[i] / scale) : 0.0f;
                q_vals[i] = tq3_rvq_choose_index(val);
            }

            // Step B: Optimize s2_vals (RVQ Stage 2, 4-bit per 32-element block)
            // Mathematically rigorous O(1) quadratic search per centroid config
            for (int b2 = 0; b2 < 8; b2++) {
                float sum_wXY = 0.0f;
                float sum_wYY = 0.0f;
                for (int i = 0; i < 32; i++) {
                    int idx = b2 * 32 + i;
                    float Y = tq3_rvq_centroids[q_vals[idx]] * rms * s3_vals[idx / 8];
                    float wY = w[idx] * Y;
                    sum_wXY += buf[idx] * wY;
                    sum_wYY += Y * wY;
                }
                int best_idx = 0;
                float best_s2_err = FLT_MAX;
                for (int ci = 0; ci < 16; ci++) {
                    float s2 = cb2[ci];
                    float err = s2 * s2 * sum_wYY - 2.0f * s2 * sum_wXY;
                    if (err < best_s2_err) {
                        best_s2_err = err;
                        best_idx = ci;
                    }
                }
                s2_vals[b2] = cb2[best_idx];
            }

            // Step C: Optimize s3_vals (RVQ Stage 3, 2-bit per 8-element sub-block)
            // Mathematically rigorous O(1) quadratic search per centroid config
            for (int b3 = 0; b3 < 32; b3++) {
                float sum_wXY = 0.0f;
                float sum_wYY = 0.0f;
                for (int i = 0; i < 8; i++) {
                    int idx = b3 * 8 + i;
                    float Y = tq3_rvq_centroids[q_vals[idx]] * rms * s2_vals[idx / 32];
                    float wY = w[idx] * Y;
                    sum_wXY += buf[idx] * wY;
                    sum_wYY += Y * wY;
                }
                int best_idx = 0;
                float best_s3_err = FLT_MAX;
                for (int ci = 0; ci < 4; ci++) {
                    float s3 = cb3[ci];
                    float err = s3 * s3 * sum_wYY - 2.0f * s3 * sum_wXY;
                    if (err < best_s3_err) {
                        best_s3_err = err;
                        best_idx = ci;
                    }
                }
                s3_vals[b3] = cb3[best_idx];
            }

            // Step D: Wiener filter for optimal super-block scale d (simulating FP16 rounding error)
            float num_d = 0.0f, den_d = 0.0f;
            for (int i = 0; i < QK_TQ3_RVQ; i++) {
                float c = tq3_rvq_centroids[q_vals[i]] * s2_vals[i / 32] * s3_vals[i / 8];
                float wc = w[i] * c;
                num_d += buf[i] * wc;
                den_d += c * wc;
            }
            if (den_d > 1e-10f) {
                rms = num_d / den_d;
                // Simulate FP16 rounding error in iterative loop
                dst->d = GGML_FP32_TO_FP16(rms);
                rms = GGML_FP16_TO_FP32(dst->d);
            }
        }

        // 5. Final one-time bit-packing into dst
        // Pack 3-bit base indices (dst->qs): 8 values per 3 bytes, LSB-first
        memset(dst->qs, 0, sizeof(dst->qs));
        for (int i = 0; i < QK_TQ3_RVQ; i++) {
            int q = q_vals[i];
            int byte_idx = (i * 3) / 8;
            int bit_off = (i * 3) % 8;
            dst->qs[byte_idx] |= (q & 0x7) << bit_off;
            if (bit_off > 5) {
                dst->qs[byte_idx + 1] |= (q & 0x7) >> (8 - bit_off);
            }
        }

        // Pack 4-bit scales2 (dst->scales2): 2 blocks per byte, LSB-first
        memset(dst->scales2, 0, sizeof(dst->scales2));
        for (int b2 = 0; b2 < 8; b2++) {
            int best_idx = 0;
            for (int ci = 0; ci < 16; ci++) {
                if (cb2[ci] == s2_vals[b2]) {
                    best_idx = ci;
                    break;
                }
            }
            int byte_idx = b2 >> 1;
            int nibble_off = (b2 & 1) * 4;
            dst->scales2[byte_idx] |= (best_idx & 0xF) << nibble_off;
        }

        // Pack 2-bit scales3 (dst->scales3): 4 sub-blocks per byte, LSB-first
        memset(dst->scales3, 0, sizeof(dst->scales3));
        for (int b3 = 0; b3 < 32; b3++) {
            int best_idx = 0;
            for (int ci = 0; ci < 4; ci++) {
                if (cb3[ci] == s3_vals[b3]) {
                    best_idx = ci;
                    break;
                }
            }
            int byte_idx = b3 >> 2;
            int bit_off = (b3 & 3) * 2;
            dst->scales3[byte_idx] |= (best_idx & 0x3) << bit_off;
        }

        // Save final rms
        dst->d = GGML_FP32_TO_FP16(rms);
    }
}

void quantize_row_tq3_rvq_ref(const float * GGML_RESTRICT x, block_tq3_rvq * GGML_RESTRICT y, int64_t k) {
    quantize_row_tq3_rvq_impl(x, y, k, NULL, tq3_rvq_cb2, tq3_rvq_cb3);
}

void dequantize_row_tq3_rvq(const block_tq3_rvq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_RVQ == 0);
    const int nb = k / QK_TQ3_RVQ;
    for (int blk = 0; blk < nb; blk++) {
        const block_tq3_rvq * blk_ptr = &x[blk];
        const float d = GGML_FP16_TO_FP32(blk_ptr->d);

        // Dequantize all 256 elements into a local buffer
        float buf[QK_TQ3_RVQ];
        for (int i = 0; i < QK_TQ3_RVQ; i++) {
            // Unpack 3-bit, LSB-first
            int byte_idx = (i * 3) / 8;
            int bit_off = (i * 3) % 8;
            int q = (blk_ptr->qs[byte_idx] >> bit_off) & 0x7;
            if (bit_off > 5) {
                q |= (blk_ptr->qs[byte_idx + 1] << (8 - bit_off)) & 0x7;
            }
            // RVQ stage 2: 4-bit per 32-element block
            int block2 = i / 32;
            int byte2 = block2 >> 1;
            int nib_off = (block2 & 1) * 4;
            int idx2 = (blk_ptr->scales2[byte2] >> nib_off) & 0xF;
            float s2 = tq3_rvq_cb2[idx2];
            // RVQ stage 3: 2-bit per 8-element sub-block
            int block3 = i / 8;
            int byte3 = block3 >> 2;
            int bit_off3 = (block3 & 3) * 2;
            int idx3 = (blk_ptr->scales3[byte3] >> bit_off3) & 0x3;
            float s3 = tq3_rvq_cb3[idx3];
            // Decode in rotated domain: w_rot = centroid[q] * d * s2 * s3
            buf[i] = tq3_rvq_centroids[q] * d * s2 * s3;
        }

        // Inverse RHT per 32-element sub-block (same pattern as TQ3_1S/TQ4_1S)
        for (int g = 0; g < 8; g++) {
            tq3_0_rht_inverse(buf + g * 32);
        }

        memcpy(y + blk * QK_TQ3_RVQ, buf, QK_TQ3_RVQ * sizeof(float));
    }
}

/* ---------- K-means helper for RVQ codebook learning ---------- */

static int tq3_rvq_compare_float(const void * a, const void * b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

// Fits k centroids to the given samples using Lloyd's algorithm.
// Centroids are initialised by evenly-spaced percentile picks then refined.
void tq3_rvq_kmeans(const float * samples, int n_samples, float * centroids, int k, int n_iter) {
    if (n_samples <= 0 || k <= 0) return;

    // Sort a copy to initialise centroids at evenly-spaced percentiles
    float * sorted = (float *)malloc(n_samples * sizeof(float));
    if (!sorted) {
        // Fallback: evenly-spaced init from min/max of samples
        float fmin = samples[0], fmax = samples[0];
        for (int i = 1; i < n_samples; i++) {
            if (samples[i] < fmin) fmin = samples[i];
            if (samples[i] > fmax) fmax = samples[i];
        }
        for (int ci = 0; ci < k; ci++) {
            centroids[ci] = fmin + (fmax - fmin) * (float)ci / (float)(k - 1);
        }
        goto do_lloyd;
    }
    memcpy(sorted, samples, n_samples * sizeof(float));

    // O(n log n) sort via C stdlib
    qsort(sorted, n_samples, sizeof(float), tq3_rvq_compare_float);

    for (int ci = 0; ci < k; ci++) {
        int idx = (int)((float)ci / (float)(k - 1) * (float)(n_samples - 1) + 0.5f);
        if (idx >= n_samples) idx = n_samples - 1;
        centroids[ci] = sorted[idx];
    }
    free(sorted);

do_lloyd:
    ; // null statement (label must precede a statement, not a declaration)
    // Lloyd iterations
    float sums[16]  = {0};  // k <= 16 always
    int   counts[16] = {0};
    for (int iter = 0; iter < n_iter; iter++) {
        memset(sums,   0, k * sizeof(float));
        memset(counts, 0, k * sizeof(int));

        for (int i = 0; i < n_samples; i++) {
            float best_dist = FLT_MAX;
            int   best_ci   = 0;
            for (int ci = 0; ci < k; ci++) {
                float d = fabsf(samples[i] - centroids[ci]);
                if (d < best_dist) { best_dist = d; best_ci = ci; }
            }
            sums[best_ci]   += samples[i];
            counts[best_ci] += 1;
        }

        for (int ci = 0; ci < k; ci++) {
            if (counts[ci] > 0) centroids[ci] = sums[ci] / (float)counts[ci];
        }
    }

    // Sort centroids ascending so codebook lookups are cache-friendly
    for (int i = 1; i < k; i++) {
        float key = centroids[i];
        int j = i - 1;
        while (j >= 0 && centroids[j] > key) { centroids[j + 1] = centroids[j]; j--; }
        centroids[j + 1] = key;
    }
}

size_t quantize_tq3_rvq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_TQ3_RVQ == 0);
    size_t row_size = (n_per_row / QK_TQ3_RVQ) * sizeof(block_tq3_rvq);

    const int nb_per_row = (int)(n_per_row / QK_TQ3_RVQ);  // 256-element blocks per row
    const int n_blocks    = (int)(nrows * nb_per_row);       // total blocks across all rows

    // ---------------------------------------------------------------
    // Pass 1: collect scale-ratio samples for K-means codebook fitting
    // We only do this ONCE for the first representative linear layer (square, >= 1024)
    // ---------------------------------------------------------------
    if (!tq3_rvq_global_fitted && nrows == n_per_row && nrows >= 1024) {
        // cb2: ratio of local_rms(32-elem sub-block) / global_rms(256-elem block), 8 per block
        // cb3: ratio of local_rms(8-elem sub-block)  / (global_rms * s2_approx),   32 per block
        int n_samples2 = n_blocks * 8;
        int n_samples3 = n_blocks * 32;

        float * ratios2 = (float *)malloc(n_samples2 * sizeof(float));
        float * ratios3 = (float *)malloc(n_samples3 * sizeof(float));

        if (ratios2 && ratios3) {
            int sample2_idx = 0;
            int sample3_idx = 0;

            for (int64_t row = 0; row < nrows; row++) {
                const float * row_src = src + row * n_per_row;

                for (int blk = 0; blk < nb_per_row; blk++) {
                    const float * blk_src = row_src + blk * QK_TQ3_RVQ;

                    // WHT-rotate each 32-element sub-block (same as quantizer)
                    float buf[QK_TQ3_RVQ];
                    memcpy(buf, blk_src, QK_TQ3_RVQ * sizeof(float));
                    for (int g = 0; g < 8; g++) tq3_0_rht_forward(buf + g * 32);

                    // Global RMS of the 256-element rotated block
                    float sum_sq = 0.0f;
                    for (int i = 0; i < QK_TQ3_RVQ; i++) sum_sq += buf[i] * buf[i];
                    float global_rms = sqrtf(sum_sq / QK_TQ3_RVQ);
                    if (global_rms < 1e-10f) {
                        for (int g = 0; g < 8; g++) ratios2[sample2_idx++] = 1.0f;
                        for (int g = 0; g < 32; g++) ratios3[sample3_idx++] = 1.0f;
                        continue;
                    }

                    // cb2 samples: local RMS of each 32-element sub-block / global_rms
                    for (int g = 0; g < 8; g++) {
                        float local_sq = 0.0f;
                        for (int i = 0; i < 32; i++) local_sq += buf[g*32 + i] * buf[g*32 + i];
                        float local_rms = sqrtf(local_sq / 32.0f);
                        ratios2[sample2_idx++] = local_rms / global_rms;
                    }

                    // cb3 samples: local RMS of each 8-element sub-block / (global_rms * s2_approx)
                    for (int g = 0; g < 8; g++) {
                        float s2_sum = 0.0f;
                        for (int i = 0; i < 32; i++) s2_sum += buf[g*32 + i] * buf[g*32 + i];
                        float s2_approx = sqrtf(s2_sum / 32.0f) / global_rms;
                        float denom = global_rms * s2_approx;
                        if (denom < 1e-10f) denom = 1e-10f;

                        for (int sb = 0; sb < 4; sb++) {
                            float sub_sq = 0.0f;
                            for (int i = 0; i < 8; i++) sub_sq += buf[g*32 + sb*8 + i] * buf[g*32 + sb*8 + i];
                            float sub_rms = sqrtf(sub_sq / 8.0f);
                            ratios3[sample3_idx++] = sub_rms / denom;
                        }
                    }
                }
            }

            // Fit codebooks via K-means (20 iterations is plenty for 16/4 centroids)
            float new_cb2[16], new_cb3[4];
            tq3_rvq_kmeans(ratios2, n_samples2, new_cb2, 16, 20);
            tq3_rvq_kmeans(ratios3, n_samples3, new_cb3,  4, 20);
            tq3_rvq_set_codebook(new_cb2, new_cb3);
            
            tq3_rvq_global_fitted = true;

            fprintf(stderr, "[TQ3_RVQ] global adaptive codebook fitted on tensor (%lld x %lld):\n",
                    (long long)n_per_row, (long long)nrows);
            fprintf(stderr, "  cb2: ");
            for (int ci = 0; ci < 16; ci++) fprintf(stderr, "%.3f ", new_cb2[ci]);
            fprintf(stderr, "\n  cb3: ");
            for (int ci = 0; ci <  4; ci++) fprintf(stderr, "%.3f ", new_cb3[ci]);
            fprintf(stderr, "\n");
        }

        free(ratios2);
        free(ratios3);
    }

    // ---------------------------------------------------------------
    // Pass 2: actual quantization with the model-adaptive codebook
    // ---------------------------------------------------------------
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq3_rvq_impl(
            src + row * n_per_row,
            (block_tq3_rvq *)((char *)dst + row * row_size),
            n_per_row,
            imatrix,
            tq3_rvq_cb2,
            tq3_rvq_cb3
        );
    }
    return nrows * row_size;
}
