#ifdef __AVX512F__

#define BATCH_SIZE_512 16

static const __m512 ABS_MASK_512 = {
    .m512_u32 = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
                 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
                 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF,
                 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF}
};

typedef struct {
    ALIGN64 __m512 buf[W];
    __m512 nsigma_k;
    int idx;
    int count;
} Hampel9_AVX512;

#define MINMAX_SIMD512(a, b) do { \
    __m512 tmp_min = _mm512_min_ps(a, b); \
    __m512 tmp_max = _mm512_max_ps(a, b); \
    a = tmp_min; \
    b = tmp_max; \
} while(0)

FORCE_INLINE __m512 fabs_simd512(__m512 x) {
    return _mm512_and_ps(x, ABS_MASK_512);
}

FORCE_INLINE __m512 median9_simd512_registers(__m512 v0, __m512 v1, __m512 v2,
                                               __m512 v3, __m512 v4, __m512 v5,
                                               __m512 v6, __m512 v7, __m512 v8) {
    MINMAX_SIMD512(v1, v2); MINMAX_SIMD512(v4, v5); MINMAX_SIMD512(v7, v8);
    MINMAX_SIMD512(v0, v1); MINMAX_SIMD512(v3, v4); MINMAX_SIMD512(v6, v7);
    MINMAX_SIMD512(v1, v2); MINMAX_SIMD512(v4, v5); MINMAX_SIMD512(v7, v8);
    MINMAX_SIMD512(v0, v3); MINMAX_SIMD512(v5, v8); MINMAX_SIMD512(v4, v7);
    MINMAX_SIMD512(v3, v6); MINMAX_SIMD512(v1, v4); MINMAX_SIMD512(v2, v5);
    MINMAX_SIMD512(v4, v7); MINMAX_SIMD512(v4, v2); MINMAX_SIMD512(v6, v4);
    MINMAX_SIMD512(v4, v2);
    return v4;
}

FORCE_INLINE void hampel9_simd512_init(Hampel9_AVX512 *h) {
    h->idx = 0;
    h->count = 0;
    h->nsigma_k = _mm512_set1_ps(NSIGMA_K);
    __m512 zero = _mm512_setzero_ps();
    for (int i = 0; i < W; i++) {
        h->buf[i] = zero;
    }
}

FORCE_INLINE __m512 hampel9_simd512_update(Hampel9_AVX512 *h, __m512 x_batch) {
    h->buf[h->idx] = x_batch;
    h->idx = h->idx + 1;
    int ge = (h->idx >= W);
    h->idx -= ge * W;
    
    if (h->count < W) {
        h->count++;
        return x_batch;
    }
    
    // Pre-load all registers
    __m512 b0 = h->buf[0], b1 = h->buf[1], b2 = h->buf[2];
    __m512 b3 = h->buf[3], b4 = h->buf[4], b5 = h->buf[5];
    __m512 b6 = h->buf[6], b7 = h->buf[7], b8 = h->buf[8];
    
    __m512 med = median9_simd512_registers(b0, b1, b2, b3, b4, b5, b6, b7, b8);
    
    __m512 ad0 = fabs_simd512(_mm512_sub_ps(b0, med));
    __m512 ad1 = fabs_simd512(_mm512_sub_ps(b1, med));
    __m512 ad2 = fabs_simd512(_mm512_sub_ps(b2, med));
    __m512 ad3 = fabs_simd512(_mm512_sub_ps(b3, med));
    __m512 ad4 = fabs_simd512(_mm512_sub_ps(b4, med));
    __m512 ad5 = fabs_simd512(_mm512_sub_ps(b5, med));
    __m512 ad6 = fabs_simd512(_mm512_sub_ps(b6, med));
    __m512 ad7 = fabs_simd512(_mm512_sub_ps(b7, med));
    __m512 ad8 = fabs_simd512(_mm512_sub_ps(b8, med));
    
    __m512 mad = median9_simd512_registers(ad0, ad1, ad2, ad3, ad4, ad5, ad6, ad7, ad8);
    
    // ✅ Inlined comparison with OS signaling
    __mmask16 mask = _mm512_cmp_ps_mask(
        fabs_simd512(_mm512_sub_ps(x_batch, med)),
        _mm512_mul_ps(h->nsigma_k, mad),
        _CMP_GT_OS  // ✅ Signaling NaN handling
    );
    
    return _mm512_mask_blend_ps(mask, x_batch, med);
}

// Aligned batch processing for AVX-512
FORCE_INLINE void hampel9_process_batch512_aligned(Hampel9_AVX512 *filters, int num_batches,
                                                   const float *inputs, float *outputs) {
    for (int batch = 0; batch < num_batches; batch++) {
        __m512 x = _mm512_load_ps(&inputs[batch * 16]);
        __m512 result = hampel9_simd512_update(&filters[batch], x);
        _mm512_store_ps(&outputs[batch * 16], result);
    }
}

#endif // __AVX512F__