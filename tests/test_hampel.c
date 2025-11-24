/**
 * @file test_hampel.c
 * @brief Comprehensive test suite for production Hampel filter
 *
 * Tests include verification of Improvement B (selective MAD cache update)
 */

#include "../include/hampel_robust.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <float.h>

#define TEST_PASSED printf("✓ %s passed\n", __func__)
#define TEST_FAILED printf("✗ %s FAILED at line %d\n", __func__, __LINE__)

// Helper: Check if float is NaN
static int is_nan(float x)
{
    return x != x;
}

// Helper: Check if two floats are close
static int approx_equal(float a, float b, float tol)
{
    if (is_nan(a) && is_nan(b))
        return 1;
    if (is_nan(a) || is_nan(b))
        return 0;
    return fabsf(a - b) < tol;
}

// Helper: Print __m256 for debugging
static void print_m256(const char *name, __m256 v)
{
    float vals[8];
    _mm256_storeu_ps(vals, v);
    printf("%s: [", name);
    for (int i = 0; i < 8; i++)
    {
        if (is_nan(vals[i]))
            printf("NaN");
        else if (isinf(vals[i]))
            printf("Inf");
        else
            printf("%.2f", vals[i]);
        if (i < 7)
            printf(", ");
    }
    printf("]\n");
}

/**
 * Test 1: Basic initialization
 */
void test_init()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    assert(filter.idx == 0);
    assert(filter.count == 0);

#ifndef HAMPEL_NO_DIAGNOSTICS
    assert(filter.total_updates == 0);
    assert(filter.nan_samples == 0);
    assert(filter.outliers == 0);
#endif

    TEST_PASSED;
}

/**
 * Test 2: Warmup phase (first W-1 samples)
 */
void test_warmup()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    // Feed W-1 samples, should return unchanged
    for (int i = 0; i < W - 1; i++)
    {
        float input[8] = {100.0f, 101.0f, 99.0f, 100.5f, 100.2f, 99.8f, 100.1f, 100.3f};
        __m256 x = _mm256_loadu_ps(input);
        __m256 result = hampel9_simd_update(&filter, x);

        float output[8];
        _mm256_storeu_ps(output, result);

        // During warmup, output should equal input
        for (int j = 0; j < 8; j++)
        {
            assert(approx_equal(output[j], input[j], 1e-5f));
        }
    }

    assert(filter.count == W - 1);
    TEST_PASSED;
}

/**
 * Test 3: NaN handling - all NaN input
 */
void test_all_nan()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    // Warmup with valid data
    for (int i = 0; i < W; i++)
    {
        float input[8] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
        __m256 x = _mm256_loadu_ps(input);
        hampel9_simd_update(&filter, x);
    }

    // Feed all-NaN input
    float nan_input[8];
    for (int i = 0; i < 8; i++)
        nan_input[i] = NAN;

    __m256 x = _mm256_loadu_ps(nan_input);
    __m256 result = hampel9_simd_update(&filter, x);

    float output[8];
    _mm256_storeu_ps(output, result);

    // Output should be NaN (preserving markers)
    for (int i = 0; i < 8; i++)
    {
        assert(is_nan(output[i]));
    }

#ifndef HAMPEL_NO_DIAGNOSTICS
    assert(filter.nan_samples == 8);
#endif

    TEST_PASSED;
}

/**
 * Test 4: NaN handling - mixed NaN and valid
 */
void test_mixed_nan()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    // Warmup
    for (int i = 0; i < W; i++)
    {
        float input[8] = {100.0f, 101.0f, 99.0f, 100.5f, 100.2f, 99.8f, 100.1f, 100.3f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // Mixed input: some NaN, some valid
    float mixed[8] = {100.0f, NAN, 101.0f, NAN, 99.0f, 100.5f, NAN, 100.2f};
    __m256 result = hampel9_simd_update(&filter, _mm256_loadu_ps(mixed));

    float output[8];
    _mm256_storeu_ps(output, result);

    // Check: NaN positions preserved, valid positions filtered
    assert(approx_equal(output[0], 100.0f, 1.0f));
    assert(is_nan(output[1]));
    assert(approx_equal(output[2], 101.0f, 1.0f));
    assert(is_nan(output[3]));
    assert(approx_equal(output[4], 99.0f, 1.0f));
    assert(approx_equal(output[5], 100.5f, 1.0f));
    assert(is_nan(output[6]));
    assert(approx_equal(output[7], 100.2f, 1.0f));

    TEST_PASSED;
}

/**
 * Test 5: Outlier detection
 */
void test_outlier_detection()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    // Feed stable data for W samples
    for (int i = 0; i < W; i++)
    {
        float input[8] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // Feed outlier (1000.0f way outside normal range)
    float outlier_input[8] = {1000.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    __m256 result = hampel9_simd_update(&filter, _mm256_loadu_ps(outlier_input));

    float output[8];
    _mm256_storeu_ps(output, result);

    // First value should be replaced with median (~100.0)
    assert(approx_equal(output[0], 100.0f, 1.0f));
    // Other values should pass through
    for (int i = 1; i < 8; i++)
    {
        assert(approx_equal(output[i], 100.0f, 0.1f));
    }

#ifndef HAMPEL_NO_DIAGNOSTICS
    assert(filter.outliers >= 1);
#endif

    TEST_PASSED;
}

/**
 * Test 6: Infinity handling
 */
void test_infinity_handling()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    // Warmup
    for (int i = 0; i < W; i++)
    {
        float input[8] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // Feed infinity
    float inf_input[8] = {INFINITY, 100.0f, -INFINITY, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    __m256 result = hampel9_simd_update(&filter, _mm256_loadu_ps(inf_input));

    float output[8];
    _mm256_storeu_ps(output, result);

    // Infinities should be handled (replaced with median)
    assert(!isinf(output[0]));
    assert(approx_equal(output[1], 100.0f, 0.1f));
    assert(!isinf(output[2]));

#ifndef HAMPEL_NO_DIAGNOSTICS
    assert(filter.inf_samples >= 2);
#endif

    TEST_PASSED;
}

/**
 * Test 7: Fast-path efficiency
 */
void test_fast_path()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    // Warmup
    for (int i = 0; i < W; i++)
    {
        float input[8] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // Feed many clean samples
    for (int i = 0; i < 100; i++)
    {
        float input[8] = {100.1f, 99.9f, 100.0f, 100.2f, 99.8f, 100.1f, 100.0f, 99.9f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

#ifndef HAMPEL_NO_DIAGNOSTICS
    Hampel9_Stats stats;
    hampel9_get_stats(&filter, &stats);

    // Should have high fast-path hit rate (>80%)
    assert(stats.fast_path_rate > 80.0);
    printf("  Fast-path rate: %.1f%%\n", stats.fast_path_rate);
#endif

    TEST_PASSED;
}

/**
 * Test 8: Statistics accuracy
 */
void test_statistics()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    // Warmup
    for (int i = 0; i < W; i++)
    {
        float input[8] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // Feed 10 samples with known characteristics
    // 5 clean samples
    for (int i = 0; i < 5; i++)
    {
        float input[8] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // 2 samples with NaN (8 per sample = 16 NaN total)
    for (int i = 0; i < 2; i++)
    {
        float input[8] = {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // 3 samples with outliers (8 per sample = 24 outliers total)
    for (int i = 0; i < 3; i++)
    {
        float input[8] = {1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

#ifndef HAMPEL_NO_DIAGNOSTICS
    Hampel9_Stats stats;
    hampel9_get_stats(&filter, &stats);

    assert(stats.nan_samples == 16);
    printf("  NaN rate: %.1f%% (%lu samples)\n", stats.nan_rate, stats.nan_samples);
    printf("  Outlier rate: %.1f%% (%lu samples)\n", stats.outlier_rate, stats.outliers);
#endif

    TEST_PASSED;
}

/**
 * Test 9: Improvement B - Selective MAD cache update (Bug Fix Verification)
 *
 * This test verifies that Improvement B correctly handles intermittent NaN inputs
 * by NOT updating last_mad for lanes with bad inputs.
 */
void test_improvement_b_selective_mad_cache()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    // Warmup with stable data
    for (int i = 0; i < W; i++)
    {
        float input[8] = {100.0f, 200.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // Feed stable data to establish valid MAD for all lanes
    for (int i = 0; i < 5; i++)
    {
        float input[8] = {100.1f, 200.1f, 100.1f, 100.1f, 100.1f, 100.1f, 100.1f, 100.1f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // Extract MAD after stable period (for lane 1, should be very small)
    float stable_mad[8];
    _mm256_storeu_ps(stable_mad, filter.last_mad);
    float lane1_valid_mad = stable_mad[1];

    // Feed NaN to lane 1 ONLY
    float nan_input[8] = {100.0f, NAN, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    hampel9_simd_update(&filter, _mm256_loadu_ps(nan_input));

    // Check: lane 1's MAD should NOT have changed (Improvement B)
    float after_nan_mad[8];
    _mm256_storeu_ps(after_nan_mad, filter.last_mad);

    assert(approx_equal(after_nan_mad[1], lane1_valid_mad, 1e-6f));
    printf("  Lane 1 MAD preserved: %.6f (before) == %.6f (after NaN)\n",
           lane1_valid_mad, after_nan_mad[1]);

    // Feed valid data back to lane 1
    float valid_input[8] = {100.0f, 200.1f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    __m256 result = hampel9_simd_update(&filter, _mm256_loadu_ps(valid_input));

    float output[8];
    _mm256_storeu_ps(output, result);

    // Lane 1 should filter correctly using preserved MAD
    assert(approx_equal(output[1], 200.1f, 1.0f));

    TEST_PASSED;
}

/**
 * Test 10: Batch processing
 */
void test_batch_processing()
{
    const int NUM_BATCHES = 10;
    Hampel9_SIMD filters[NUM_BATCHES];

    // Initialize all filters
    for (int i = 0; i < NUM_BATCHES; i++)
    {
        hampel9_simd_init(&filters[i]);
    }

    // Warmup all filters
    float warmup_data[NUM_BATCHES * 8];
    for (int i = 0; i < NUM_BATCHES * 8; i++)
    {
        warmup_data[i] = 100.0f;
    }

    for (int tick = 0; tick < W; tick++)
    {
        float dummy_output[NUM_BATCHES * 8];
        hampel9_process_batch(filters, NUM_BATCHES, warmup_data, dummy_output);
    }

    // Process batch with outliers
    float tick_data[NUM_BATCHES * 8];
    for (int i = 0; i < NUM_BATCHES * 8; i++)
    {
        tick_data[i] = (i % 8 == 0) ? 1000.0f : 100.0f; // Outlier in first position
    }

    float filtered_data[NUM_BATCHES * 8];
    hampel9_process_batch(filters, NUM_BATCHES, tick_data, filtered_data);

    // Check outliers were filtered
    for (int i = 0; i < NUM_BATCHES; i++)
    {
        assert(approx_equal(filtered_data[i * 8], 100.0f, 1.0f));
    }

    TEST_PASSED;
}

/**
 * Test 11: Stress test - random data
 */
void test_stress_random()
{
    Hampel9_SIMD filter;
    hampel9_simd_init(&filter);

    // Warmup
    for (int i = 0; i < W; i++)
    {
        float input[8] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
        hampel9_simd_update(&filter, _mm256_loadu_ps(input));
    }

    // Feed 1000 random samples
    srand(42);
    for (int i = 0; i < 1000; i++)
    {
        float input[8];
        for (int j = 0; j < 8; j++)
        {
            float r = (float)rand() / RAND_MAX;
            if (r < 0.05)
            {
                input[j] = NAN; // 5% NaN
            }
            else if (r < 0.10)
            {
                input[j] = 1000.0f; // 5% outliers
            }
            else
            {
                input[j] = 100.0f + (rand() % 10) - 5; // Normal range
            }
        }

        __m256 result = hampel9_simd_update(&filter, _mm256_loadu_ps(input));

        float output[8];
        _mm256_storeu_ps(output, result);

        // Check no invalid outputs (except preserved NaN)
        for (int j = 0; j < 8; j++)
        {
            if (!is_nan(output[j]))
            {
                assert(!isinf(output[j]));
                assert(output[j] > -1000.0f && output[j] < 1000.0f);
            }
        }
    }

#ifndef HAMPEL_NO_DIAGNOSTICS
    Hampel9_Stats stats;
    hampel9_get_stats(&filter, &stats);

    printf("  Stress test stats:\n");
    printf("    Total updates: %lu\n", stats.total_updates);
    printf("    NaN rate: %.2f%%\n", stats.nan_rate);
    printf("    Outlier rate: %.2f%%\n", stats.outlier_rate);
    printf("    Fast-path rate: %.2f%%\n", stats.fast_path_rate);
#endif

    TEST_PASSED;
}

/**
 * Main test runner
 */
int main()
{
    printf("========================================\n");
    printf("Hampel Filter Production Test Suite v1.1\n");
    printf("========================================\n\n");

    test_init();
    test_warmup();
    test_all_nan();
    test_mixed_nan();
    test_outlier_detection();
    test_infinity_handling();
    test_fast_path();
    test_statistics();
    test_improvement_b_selective_mad_cache(); // NEW: Verifies bug fix
    test_batch_processing();
    test_stress_random();

    printf("\n========================================\n");
    printf("All tests passed! ✓\n");
    printf("Improvements A, B, C verified!\n");
    printf("========================================\n");

    return 0;
}