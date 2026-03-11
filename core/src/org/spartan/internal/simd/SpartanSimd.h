//
// Created by Alepando on 10/3/2026.
//
#pragma once

#if defined(__AVX2__)
    #define SPARTAN_USE_AVX2
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    #define SPARTAN_USE_NEON
    #include <arm_neon.h>
#else
    #define SPARTAN_USE_SCALAR
    #include <cmath>
#endif

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

#if defined(SPARTAN_USE_AVX2)

    // Advanced Vector Extensions 2 Implementation for x86_64 processors

    using SimdFloat = __m256d;
    constexpr int simdLaneCount = 4;

    /**
     * Loads consecutive double-precision numbers from memory into the hardware register.
     * This uses unaligned loads to ensure it never crashes if the Java Virtual Machine
     * allocates memory that does not align perfectly to a 32-byte boundary.
     */
    inline SimdFloat simdLoad(const double* sourcePointer) {
        return _mm256_loadu_pd(sourcePointer);
    }

    /**
     * Writes the contents of the hardware register directly back into main memory.
     */
    inline void simdStore(double* targetPointer, const SimdFloat vectorToStore) {
        _mm256_storeu_pd(targetPointer, vectorToStore);
    }

    /**
     * Creates a vector completely filled with absolute zeros.
     * This is optimized by the CPU using an XOR operation against itself, which takes zero clock cycles.
     */
    inline SimdFloat simdSetZero() {
        return _mm256_setzero_pd();
    }

    /**
     * Broadcasts a single scalar value across all lanes of the vector register.
     * Useful for applying a constant multiplier (like a learning rate or alpha) to an entire array.
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
     * The hardware calculates the multiplication and the addition simultaneously in a
     * single clock cycle without losing precision during rounding.
     */
    inline SimdFloat simdFusedMultiplyAdd(const SimdFloat multiplier, const SimdFloat multiplicand, const SimdFloat addend) {
        return _mm256_fmadd_pd(multiplier, multiplicand, addend);
    }

    /**
     * Compares two vectors element by element and returns a vector containing the largest values.
     * Essential for clipping operations like the Rectified Linear Unit (ReLU) or fuzzy union.
     */
    inline SimdFloat simdMax(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm256_max_pd(firstVector, secondVector);
    }

    /**
     * Compares two vectors element by element and returns a vector containing the smallest values.
     * Essential for bounding constraints or fuzzy intersection.
     */
    inline SimdFloat simdMin(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm256_min_pd(firstVector, secondVector);
    }

    /**
     * Calculates the square root of every element in the vector.
     * Required for stabilizing the gradients in the Adam optimizer or fuzzy dilation.
     */
    inline SimdFloat simdSqrt(const SimdFloat vectorToRoot) {
        return _mm256_sqrt_pd(vectorToRoot);
    }

    /**
     * Compares if the elements of the first vector are strictly greater than the second.
     * Returns a bitmask vector where lanes are filled with all 1s (true) or all 0s (false).
     */
    inline SimdFloat simdCompareGreaterThan(const SimdFloat firstVector, const SimdFloat secondVector) {
        return _mm256_cmp_pd(firstVector, secondVector, _CMP_GT_OQ);
    }

    /**
     * Selects elements from either the trueValue vector or the falseValue vector
     * based on the bits provided by the mask vector.
     * This avoids costly CPU branch prediction failures (if-statements) during activation functions.
     */
    inline SimdFloat simdBlend(const SimdFloat trueValue, const SimdFloat falseValue, const SimdFloat mask) {
        return _mm256_blendv_pd(falseValue, trueValue, mask);
    }

    /**
     * Collapses all internal lanes of the register into a single scalar number by adding them together.
     * This maintains the data inside the L1 cache until the absolute final sum is computed.
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