#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

/* TQ3_1S: 16-byte block = d0(fp16) + d1(fp16) + qs[12].
 * The WHT sign-pattern index is stored in the low 2 bits of the d0 half
 * (see quantize_row_tq3_1s_impl / dequantize_row_tq3_1s in ggml-turbo-quant.c). */
extern void quantize_row_tq3_1s_ref(const float * x, void * y, long long k);
extern void dequantize_row_tq3_1s(const void * x, float * y, long long k);
extern size_t quantize_tq3_1s(const float * src, void * dst, long long nrows, long long n_per_row, const float * imatrix);

/* Read the 2-bit sign-pattern index from the low bits of a block's d0 half.
 * Use memcpy so the read is endian-neutral (matches the CPU/CUDA decoders). */
static int tq3_1s_block_pattern(const char * blk) {
    uint16_t d0raw;
    memcpy(&d0raw, blk, sizeof(d0raw));
    return (int)(d0raw & 0x3u);
}

int main(void) {
    int failures = 0;
    const int d = 128;
    char buf[256];
    float input[128], output[128];
    float mse, cosv, ni, no;

    printf("=== TurboQuant C Round-Trip Test ===\n\n");

    /* Test 1: basis vector
     *
     * dequantize_row_turbo3_0 leaves output in the WHT-rotated domain (Q is
     * also rotated by the graph, so <Q_rot, K_rot> yields correct attention
     * scores without an explicit inverse). To verify the round-trip, apply
     * the inverse WHT before comparing against the original input. */
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 1 (turbo3): e0 = [1, 0, ...]\n");
    printf("  In:  [%.6f, %.6f, %.6f, %.6f]\n", input[0], input[1], input[2], input[3]);
    printf("  Out: [%.6f, %.6f, %.6f, %.6f]\n", output[0], output[1], output[2], output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f OutNorm=%.6f\n\n", mse/d, ni > 0 && no > 0 ? cosv/sqrtf(ni)/sqrtf(no) : 0, sqrtf(no));

    /* Test 2: large-norm vector */
    for (int i = 0; i < d; i++) input[i] = sinf(i*0.1f+0.5f) * 10.0f;
    quantize_row_turbo3_0_ref(input, buf, d);
    dequantize_row_turbo3_0(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 2 (turbo3): sin*10\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", input[0], input[1], input[2], input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", output[0], output[1], output[2], output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f InNorm=%.2f OutNorm=%.2f\n\n", mse/d, cosv/sqrtf(ni)/sqrtf(no), sqrtf(ni), sqrtf(no));

    /* Test 3: turbo4
     *
     * Same convention as turbo3: dequant leaves output in the rotated domain
     * (see comment in dequantize_row_turbo4_0 @ ggml-turbo-quant.c). Apply
     * the inverse WHT before comparing. */
    for (int i = 0; i < d; i++) input[i] = cosf(i*0.2f) * 5.0f;
    quantize_row_turbo4_0_ref(input, buf, d);
    dequantize_row_turbo4_0(buf, output, d);
    turbo_cpu_fwht_inverse(output, d);
    printf("Test 3 (turbo4): cos*5\n");
    printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", input[0], input[1], input[2], input[3]);
    printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", output[0], output[1], output[2], output[3]);
    mse = cosv = ni = no = 0;
    for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
    printf("  MSE=%.8f Cosine=%.6f\n\n", mse/d, cosv/sqrtf(ni)/sqrtf(no));

    /* Test 4: TQ3_1S round-trip (plain reference path, no imatrix)
     *
     * TQ3_1S dequant leaves the output in the original (x) domain, so no
     * inverse-WHT step is needed before comparing against the input. */
    {
        const int d = 256; /* 8 blocks of 32 */
        char buf[256/32 * 16];
        float input[256], output[256];
        float mse, cosv, ni, no;

        srand(1234);
        for (int i = 0; i < d; i++) input[i] = (float)(rand() % 2000 - 1000) / 500.0f;
        quantize_row_tq3_1s_ref(input, buf, d);
        dequantize_row_tq3_1s(buf, output, d);
        mse = cosv = ni = no = 0;
        for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
        printf("Test 4 (tq3_1s): random, no imatrix\n");
        printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", input[0], input[1], input[2], input[3]);
        printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", output[0], output[1], output[2], output[3]);
        printf("  MSE=%.8f Cosine=%.6f\n\n", mse/d, ni > 0 && no > 0 ? cosv/sqrtf(ni)/sqrtf(no) : 0);

        if (mse/d > 0.05f) {
            printf("  FAIL: TQ3_1S round-trip MSE too high\n");
            failures++;
        }
        if (ni > 0 && no > 0 && cosv/sqrtf(ni)/sqrtf(no) < 0.98f) {
            printf("  FAIL: TQ3_1S round-trip cosine too low\n");
            failures++;
        }
    }

    /* Test 5: TQ3_1S sign-pattern selection (imatrix path)
     *
     * quantize_tq3_1s picks the best of 4 WHT sign patterns per block and
     * stores the 2-bit index in the low bits of d0. Verify that (a) the
     * imatrix-weighted round-trip stays accurate, (b) the stored pattern
     * index is always in [0,3], and (c) random blocks actually exercise
     * more than one pattern (i.e. the selector is not degenerate). */
    {
        const int d = 1024; /* 32 blocks of 32 */
        const int nb = d/32;
        char buf[1024/32 * 16];
        float input[1024], output[1024], imatrix[1024];
        int   pat_count[4] = {0, 0, 0, 0};
        float mse, cosv, ni, no;

        srand(5678);
        for (int i = 0; i < d; i++) {
            input[i]   = (float)(rand() % 2000 - 1000) / 500.0f;
            imatrix[i] = 1.0f + 0.5f*sinf(0.3f*i); /* column importance varies */
        }
        size_t row_size = (d/32) * 16;
        size_t nbytes = quantize_tq3_1s(input, buf, 1, d, imatrix);
        dequantize_row_tq3_1s(buf, output, d);
        for (int b = 0; b < nb; b++) {
            const int pat = tq3_1s_block_pattern(buf + b*16);
            if (pat < 0 || pat > 3) {
                printf("  FAIL: block %d pattern index %d out of range\n", b, pat);
                failures++;
            } else {
                pat_count[pat]++;
            }
        }
        mse = cosv = ni = no = 0;
        for (int i = 0; i < d; i++) { mse += (input[i]-output[i])*(input[i]-output[i]); cosv += input[i]*output[i]; ni += input[i]*input[i]; no += output[i]*output[i]; }
        printf("Test 5 (tq3_1s): sign-pattern selection, imatrix\n");
        printf("  In:  [%.4f, %.4f, %.4f, %.4f]\n", input[0], input[1], input[2], input[3]);
        printf("  Out: [%.4f, %.4f, %.4f, %.4f]\n", output[0], output[1], output[2], output[3]);
        printf("  MSE=%.8f Cosine=%.6f\n", mse/d, ni > 0 && no > 0 ? cosv/sqrtf(ni)/sqrtf(no) : 0);
        printf("  pattern distribution: p0=%d p1=%d p2=%d p3=%d (bytes=%zu)\n\n",
               pat_count[0], pat_count[1], pat_count[2], pat_count[3], nbytes);

        if (nbytes != row_size) {
            printf("  FAIL: quantize_tq3_1s wrote %zu bytes, expected %zu\n", nbytes, row_size);
            failures++;
        }
        if (mse/d > 0.05f) {
            printf("  FAIL: TQ3_1S imatrix round-trip MSE too high\n");
            failures++;
        }
        if (ni > 0 && no > 0 && cosv/sqrtf(ni)/sqrtf(no) < 0.98f) {
            printf("  FAIL: TQ3_1S imatrix round-trip cosine too low\n");
            failures++;
        }
        int n_distinct = (pat_count[0] > 0) + (pat_count[1] > 0) + (pat_count[2] > 0) + (pat_count[3] > 0);
        if (n_distinct < 2) {
            printf("  FAIL: sign-pattern selector used only %d distinct pattern(s)\n", n_distinct);
            failures++;
        }
    }

    printf("=== Done ===\n");
    if (failures > 0) {
        printf("%d test(s) FAILED\n", failures);
    }
    return failures;
}
