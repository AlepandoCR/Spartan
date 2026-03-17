//
// Created by Alepando on 10/3/2026.
//
#pragma once

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    #define SPARTAN_USE_AVX512
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define SPARTAN_USE_AVX2
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    #define SPARTAN_USE_NEON
    #include <arm_neon.h>
#else
    #define SPARTAN_USE_SCALAR
    #include <cmath>
#endif

#include <algorithm>

namespace org::spartan::internal::math::simd {

    /**
     * Hardware Abstraction Layer for Single Instruction Multiple Data operations.
     *
     * This layer provides a unified interface for vector math across different CPU
     * architectures. It detects the target hardware at compile time and maps our
     * generic SimdFloat type to the appropriate intrinsic registers.
     *
     * By writing algorithms using these wrappers, the neural network code remains
     * clean, cross-platform, and mathematically identical, while extracting
     * maximum performance from the host processor.
     */

#if defined(SPARTAN_USE_AVX512)

    // Advanced Vector Extensions 512 Implementation for Zen 5 / Modern x86_64

    using SimdFloat = __m512d;
    constexpr int simdLaneCount = 8;

    /**
     * Loads consecutive double-precision numbers from memory into the hardware register.
     * This uses unaligned loads to ensure it never crashes if the Java Virtual Machine
     * or std::vector allocates memory that does not align perfectly to a 64-byte boundary.
     *
     * SAFE MODE: Explicitly using _mm512_loadu_pd instead of _mm512_load_pd.
     */
    inline SimdFloat simdLoad(const double* sourcePointer) {
        return _mm512_loadu_pd(sourcePointer);
    }

    /**
     * Writes the contents of the hardware register directly back into main memory.
     * Uses unaligned store to support generic heap allocations.
     *
     * SAFE MODE: Explicitly using _mm512_storeu_pd instead of _mm512_store_pd.
     */
    inline void simdStore(double* targetPointer, const SimdFloat vectorToStore) {
        _mm512_storeu_pd(targetPointer, vectorToStore);
    }

    /**
     * Creates a vector completely filled with absolute zeros.
     * This is optimized by the CPU using an XOR operation against itself.
     */
    inline SimdFloat simdSetZero() {
        return _mm512_setzero_pd();
    }

    /**
     * Broadcasts a single scalar value across all lanes of the vector register.
     */
    inline SimdFloat simdBroadcast(const double scalarValue) {
        return _mm512_set1_pd(scalarValue);
    }

    /**
     * Performs an element-wise addition of two vectors.
     */
    inline SimdFloat simdAdd(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm512_add_pd(firstVector, secondVector);
    }

    /**
     * Performs an element-wise subtraction of two vectors.
     */
    inline SimdFloat simdSubtract(const SimdFloat minuend, const SimdFloat subtrahend) {
        return _mm512_sub_pd(minuend, subtrahend);
    }

    /**
     * Performs an element-wise multiplication of two vectors.
     */
    inline SimdFloat simdMultiply(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm512_mul_pd(firstVector, secondVector);
    }

    /**
     * Performs an element-wise division of two vectors.
     */
    inline SimdFloat simdDivide(const SimdFloat dividend, const SimdFloat divisor) {
        return _mm512_div_pd(dividend, divisor);
    }

    /**
     * Executes a Fused Multiply-Add operation: (multiplier * multiplicand) + addend.
     * The hardware calculates the multiplication and the addition simultaneously.
     */
    inline SimdFloat simdFusedMultiplyAdd(const SimdFloat multiplier, const SimdFloat multiplicand, const SimdFloat addend) {
        return _mm512_fmadd_pd(multiplier, multiplicand, addend);
    }

    /**
     * Compares two vectors element by element and returns a vector containing the largest values.
     */
    inline SimdFloat simdMax(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm512_max_pd(firstVector, secondVector);
    }

    /**
     * Compares two vectors element by element and returns a vector containing the smallest values.
     */
    inline SimdFloat simdMin(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm512_min_pd(firstVector, secondVector);
    }

    /**
     * Calculates the square root of every element in the vector.
     */
    inline SimdFloat simdSqrt(const SimdFloat vectorToRoot) {
        return _mm512_sqrt_pd(vectorToRoot);
    }

    /**
     * Computes the absolute value of every element in the vector.
     * Formula: |x|
     */
    inline SimdFloat simdAbs(const SimdFloat vectorToAbs) {
        return _mm512_abs_pd(vectorToAbs);
    }

    /**
     * Compares if the elements of the first vector are strictly greater than the second.
     * Returns a __mmask8 mask, which is then used by blend operations.
     */
    inline SimdFloat simdCompareGreaterThan(const SimdFloat firstVector, const SimdFloat secondVector) {
        // In AVX-512, comparisons return a dedicated mask register type
        // We cast/store it in the SimdFloat context for compatibility with our Blend abstraction
        auto mask = _mm512_cmp_pd_mask(firstVector, secondVector, _CMP_GT_OQ);
        // We return a "fake" vector where bits are mapped for the blend function
        return _mm512_maskz_mov_pd(mask, _mm512_set1_pd(1.0));
    }

    /**
     * Selects elements from either the trueValue vector or the falseValue vector
     * based on the mask provided.
     */
    inline SimdFloat simdBlend(const SimdFloat trueValue, const SimdFloat falseValue, const SimdFloat maskVector) {
        // Convert the mask vector back to a hardware mask
        __mmask8 mask = _mm512_cmp_pd_mask(maskVector, _mm512_setzero_pd(), _CMP_NEQ_OQ);
        return _mm512_mask_blend_pd(mask, falseValue, trueValue);
    }

    /**
     * Collapses all internal lanes of the register into a single scalar number.
     */
    inline double simdHorizontalSum(const SimdFloat vectorToSum) {
        return _mm512_reduce_add_pd(vectorToSum);
    }


#elif defined(SPARTAN_USE_AVX2)

    // Advanced Vector Extensions 2 Implementation for x86_64 processors

    using SimdFloat = __m256d;
    constexpr int simdLaneCount = 4;

    /**
     * Loads consecutive double-precision numbers from memory into the hardware register.
     * This uses unaligned loads to ensure it never crashes if the Java Virtual Machine
     * or std::vector allocates memory that does not align perfectly to a 32-byte boundary.
     *
     * SAFE MODE: Explicitly using _mm256_loadu_pd instead of _mm256_load_pd.
     */
    inline SimdFloat simdLoad(const double* sourcePointer) {
        return _mm256_loadu_pd(sourcePointer);
    }

    /**
     * Writes the contents of the hardware register directly back into main memory.
     * Uses unaligned store to support generic heap allocations (16-byte aligned on Windows).
     *
     * SAFE MODE: Explicitly using _mm256_storeu_pd instead of _mm256_store_pd.
     */
    inline void simdStore(double* targetPointer, const SimdFloat vectorToStore) {
        _mm256_storeu_pd(targetPointer, vectorToStore);
    }

    /**
     * Creates a vector completely filled with absolute zeros.
     * This is optimized by the CPU using an XOR operation against itself.
     */
    inline SimdFloat simdSetZero() {
        return _mm256_setzero_pd();
    }

    /**
     * Broadcasts a single scalar value across all lanes of the vector register.
     */
    inline SimdFloat simdBroadcast(const double scalarValue) {
        return _mm256_set1_pd(scalarValue);
    }

    /**
     * Performs an element-wise addition of two vectors.
     */
    inline SimdFloat simdAdd(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm256_add_pd(firstVector, secondVector);
    }

    /**
     * Performs an element-wise subtraction of two vectors.
     */
    inline SimdFloat simdSubtract(const SimdFloat minuend, const SimdFloat subtrahend) {
        return _mm256_sub_pd(minuend, subtrahend);
    }

    /**
     * Performs an element-wise multiplication of two vectors.
     */
    inline SimdFloat simdMultiply(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm256_mul_pd(firstVector, secondVector);
    }

    /**
     * Performs an element-wise division of two vectors.
     */
    inline SimdFloat simdDivide(const SimdFloat dividend, const SimdFloat divisor) {
        return _mm256_div_pd(dividend, divisor);
    }

    /**
     * Executes a Fused Multiply-Add operation: (multiplier * multiplicand) + addend.
     * The hardware calculates the multiplication and the addition simultaneously.
     */
    inline SimdFloat simdFusedMultiplyAdd(const SimdFloat multiplier, const SimdFloat multiplicand, const SimdFloat addend) {
        return _mm256_fmadd_pd(multiplier, multiplicand, addend);
    }

    /**
     * Compares two vectors element by element and returns a vector containing the largest values.
     */
    inline SimdFloat simdMax(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm256_max_pd(firstVector, secondVector);
    }

    /**
     * Compares two vectors element by element and returns a vector containing the smallest values.
     */
    inline SimdFloat simdMin(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm256_min_pd(firstVector, secondVector);
    }

    /**
     * Calculates the square root of every element in the vector.
     */
    inline SimdFloat simdSqrt(const SimdFloat vectorToRoot) {
        return _mm256_sqrt_pd(vectorToRoot);
    }

    /**
     * Computes the absolute value of every element in the vector.
     * Formula: |x|
     */
    inline SimdFloat simdAbs(const SimdFloat vectorToAbs) {
        // AVX2 doesn't have a direct abs for doubles, so we use a branchless approach
        // Create a mask with all bits set except the sign bit
        const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        return _mm256_and_pd(vectorToAbs, sign_mask);
    }

    /**
     * Compares if the elements of the first vector are strictly greater than the second.
     * Returns a bitmask vector.
     */
    inline SimdFloat simdCompareGreaterThan(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm256_cmp_pd(firstVector, secondVector, _CMP_GT_OQ);
    }

    /**
     * Selects elements from either the trueValue vector or the falseValue vector
     * based on the bits provided by the mask vector.
     */
    inline SimdFloat simdBlend(const SimdFloat trueValue, const SimdFloat falseValue, const SimdFloat mask) {
        return _mm256_blendv_pd(falseValue, trueValue, mask);
    }

    /**
     * Collapses all internal lanes of the register into a single scalar number.
     */
    inline double simdHorizontalSum(const SimdFloat vectorToSum) {
        const __m256d horizontalAddition = _mm256_hadd_pd(vectorToSum, vectorToSum);
        const __m256d permutedAddition = _mm256_permute2f128_pd(horizontalAddition, horizontalAddition, 1);
        const __m256d finalSum = _mm256_add_pd(horizontalAddition, permutedAddition);
        return _mm256_cvtsd_f64(finalSum);
    }

#elif defined(SPARTAN_USE_NEON)

    // NEON Implementation for ARM64 and Apple Silicon processors

    using SimdFloat = float64x2_t;
    constexpr int simdLaneCount = 2;

    inline SimdFloat simdLoad(const double* sourcePointer) {
        return vld1q_f64(sourcePointer);
    }

    inline void simdStore(double* targetPointer, const SimdFloat vectorToStore) {
        vst1q_f64(targetPointer, vectorToStore);
    }

    inline SimdFloat simdSetZero() {
        return vdupq_n_f64(0.0);
    }

    inline SimdFloat simdBroadcast(const double scalarValue) {
        return vdupq_n_f64(scalarValue);
    }

    inline SimdFloat simdAdd(const SimdFloat firstVector, const SimdFloat secondVector) {
        return vaddq_f64(firstVector, secondVector);
    }

    inline SimdFloat simdSubtract(const SimdFloat minuend, const SimdFloat subtrahend) {
        return vsubq_f64(minuend, subtrahend);
    }

    inline SimdFloat simdMultiply(const SimdFloat firstVector, const SimdFloat secondVector) {
        return vmulq_f64(firstVector, secondVector);
    }

    inline SimdFloat simdDivide(const SimdFloat dividend, const SimdFloat divisor) {
        return vdivq_f64(dividend, divisor);
    }

    inline SimdFloat simdFusedMultiplyAdd(const SimdFloat multiplier, const SimdFloat multiplicand, const SimdFloat addend) {
        return vfmaq_f64(addend, multiplier, multiplicand);
    }

    inline SimdFloat simdMax(const SimdFloat firstVector, const SimdFloat secondVector) {
        return vmaxq_f64(firstVector, secondVector);
    }

    inline SimdFloat simdMin(const SimdFloat firstVector, const SimdFloat secondVector) {
        return vminq_f64(firstVector, secondVector);
    }

    inline SimdFloat simdSqrt(const SimdFloat vectorToRoot) {
        return vsqrtq_f64(vectorToRoot);
    }

    inline SimdFloat simdAbs(const SimdFloat vectorToAbs) {
        return vabsq_f64(vectorToAbs);
    }

    inline SimdFloat simdCompareGreaterThan(const SimdFloat firstVector, const SimdFloat secondVector) {
        // NEON comparisons return an unsigned integer vector mask.
        // We reinterpret it as a float vector so the abstraction type remains completely transparent.
        return vreinterpretq_f64_u64(vcgtq_f64(firstVector, secondVector));
    }

    inline SimdFloat simdBlend(const SimdFloat trueValue, const SimdFloat falseValue, const SimdFloat mask) {
        return vbslq_f64(vreinterpretq_u64_f64(mask), trueValue, falseValue);
    }

    inline double simdHorizontalSum(const SimdFloat vectorToSum) {
        return vaddvq_f64(vectorToSum);
    }

#else

    // Scalar Fallback for legacy processors without vectorization support

    struct SimdFallback { double value[1]; };
    using SimdFloat = SimdFallback;
    constexpr int simdLaneCount = 1;

    inline SimdFloat simdLoad(const double* sourcePointer) {
        return {sourcePointer[0]};
    }

    inline void simdStore(double* targetPointer, const SimdFloat vectorToStore) {
        targetPointer[0] = vectorToStore.value[0];
    }

    inline SimdFloat simdSetZero() {
        return {0.0};
    }

    inline SimdFloat simdBroadcast(const double scalarValue) {
        return {scalarValue};
    }

    inline SimdFloat simdAdd(const SimdFloat firstVector, const SimdFloat secondVector) {
        return {firstVector.value[0] + secondVector.value[0]};
    }

    inline SimdFloat simdSubtract(const SimdFloat minuend, const SimdFloat subtrahend) {
        return {minuend.value[0] - subtrahend.value[0]};
    }

    inline SimdFloat simdMultiply(const SimdFloat firstVector, const SimdFloat secondVector) {
        return {firstVector.value[0] * secondVector.value[0]};
    }

    inline SimdFloat simdDivide(const SimdFloat dividend, const SimdFloat divisor) {
        return {dividend.value[0] / divisor.value[0]};
    }

    inline SimdFloat simdFusedMultiplyAdd(const SimdFloat multiplier, const SimdFloat multiplicand, const SimdFloat addend) {
        return {(multiplier.value[0] * multiplicand.value[0]) + addend.value[0]};
    }

    inline SimdFloat simdMax(const SimdFloat firstVector, const SimdFloat secondVector) {
        return {std::max(firstVector.value[0], secondVector.value[0])};
    }

    inline SimdFloat simdMin(const SimdFloat firstVector, const SimdFloat secondVector) {
        return {std::min(firstVector.value[0], secondVector.value[0])};
    }

    inline SimdFloat simdSqrt(const SimdFloat vectorToRoot) {
        return {std::sqrt(vectorToRoot.value[0])};
    }

    inline SimdFloat simdAbs(const SimdFloat vectorToAbs) {
        return {std::abs(vectorToAbs.value[0])};
    }

    inline SimdFloat simdCompareGreaterThan(const SimdFloat firstVector, const SimdFloat secondVector) {
        // Emulate a mask by returning 1.0 for true and 0.0 for false.
        return {firstVector.value[0] > secondVector.value[0] ? 1.0 : 0.0};
    }

    inline SimdFloat simdBlend(const SimdFloat trueValue, const SimdFloat falseValue, const SimdFloat mask) {
        return {mask.value[0] > 0.5 ? trueValue.value[0] : falseValue.value[0]};
    }

    inline double simdHorizontalSum(const SimdFloat vectorToSum) {
        return vectorToSum.value[0];
    }

#endif

}

