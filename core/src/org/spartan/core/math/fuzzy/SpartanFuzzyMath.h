//
// Created by Alepando on 23/2/2026.
//

#ifndef SPARTAN_CORE_SPARTANFUZZYMATH_H
#define SPARTAN_CORE_SPARTANFUZZYMATH_H

#include <immintrin.h>
#include <algorithm>
#include <cmath>

/**
 * @namespace org::spartan::core::math::fuzzy
 * @brief Provides high-performance Zero-Allocation Fuzzy Logic operations.
 * * All operations utilize AVX2 (Advanced Vector Extensions) SIMD instructions
 * to process 4 double-precision floats (256 bits) per clock cycle.
 * Designed to operate directly on raw memory pointers provided by the JVM via FFM.
 */
namespace org::spartan::core::math::fuzzy {

    /**
     * A collection of fuzzy math functions for use in the Spartan Engine.
     */
   /**
     * @class FuzzySetOps
     * @brief Core mathematical operations for Fuzzy Sets.
     * * Implements standard Zadeh operators for fuzzy logic:
     * Union (Maximum), Intersection (Minimum), and Complement (1.0 - x).
     */
    class FuzzySetOps {
    public:
        // Delete constructor to prevent instantiation (Static class)
        FuzzySetOps() = delete;

        /**
         * @brief Computes the Fuzzy Union (MAX) of two sets: A = A U B
         * * @param targetSet Pointer to the first set. This array will be mutated to store the result.
         * @param sourceSet Pointer to the second set. Read-only.
         * @param size The number of valid elements to process.
         */
        static void unionSets(double* targetSet, const double* sourceSet, int size) {
            int i = 0;
            // AVX Unrolled Loop (Process 4 doubles at a time)
            for (; i <= size - 4; i += 4) {
                __m256d vecA = _mm256_loadu_pd(&targetSet[i]);
                __m256d vecB = _mm256_loadu_pd(&sourceSet[i]);
                __m256d result = _mm256_max_pd(vecA, vecB);
                _mm256_storeu_pd(&targetSet[i], result);
            }
            // Tail loop for remaining elements
            for (; i < size; ++i) {
                targetSet[i] = std::max(targetSet[i], sourceSet[i]);
            }
        }

        /**
         * @brief Computes the Fuzzy Intersection (MIN) of two sets: A = A ∩ B
         * * @param targetSet Pointer to the first set. This array will be mutated to store the result.
         * @param sourceSet Pointer to the second set. Read-only.
         * @param size The number of valid elements to process.
         */
        static void intersectSets(double* targetSet, const double* sourceSet, int size) {
            int i = 0;
            for (; i <= size - 4; i += 4) {
                __m256d vecA = _mm256_loadu_pd(&targetSet[i]);
                __m256d vecB = _mm256_loadu_pd(&sourceSet[i]);
                __m256d result = _mm256_min_pd(vecA, vecB);
                _mm256_storeu_pd(&targetSet[i], result);
            }
            for (; i < size; ++i) {
                targetSet[i] = std::min(targetSet[i], sourceSet[i]);
            }
        }

        /**
         * @brief Computes the Fuzzy Complement of a set: A = 1.0 - A
         * * @param targetSet Pointer to the set to be complemented. Mutated in place.
         * @param size The number of valid elements to process.
         */
        static void complementSet(double* targetSet, int size) {
            int i = 0;
            // Create a vector where all 4 slots contain 1.0
            __m256d vecOne = _mm256_set1_pd(1.0);

            for (; i <= size - 4; i += 4) {
                __m256d vecA = _mm256_loadu_pd(&targetSet[i]);
                __m256d result = _mm256_sub_pd(vecOne, vecA);
                _mm256_storeu_pd(&targetSet[i], result);
            }
            for (; i < size; ++i) {
                targetSet[i] = 1.0 - targetSet[i];
            }
        }
    };

    /**
     * @class FuzzyModifiers
     * @brief Linguistic hedges (modifiers) to shift the gravity of fuzzy sets.
     * * Operations that modify the intensity of a fuzzy truth value.
     */
    class FuzzyModifiers {
    public:
        FuzzyModifiers() = delete;

        /**
         * @brief Applies "Concentration" (Linguistic "VERY"). Computes A = A^2
         * Reduces the truth value of elements that are not fully 1.0.
         * * @param targetSet Pointer to the set to be modified. Mutated in place.
         * @param size The number of valid elements to process.
         */
        static void applyConcentration(double* targetSet, int size) {
            int i = 0;
            for (; i <= size - 4; i += 4) {
                __m256d vecA = _mm256_loadu_pd(&targetSet[i]);
                // Multiply the vector by itself
                __m256d result = _mm256_mul_pd(vecA, vecA);
                _mm256_storeu_pd(&targetSet[i], result);
            }
            for (; i < size; ++i) {
                targetSet[i] = targetSet[i] * targetSet[i];
            }
        }

        /**
         * @brief Applies "Dilation" (Linguistic "SOMEWHAT"). Computes A = sqrt(A)
         * Increases the truth value of weak elements.
         * * @param targetSet Pointer to the set to be modified. Mutated in place.
         * @param size The number of valid elements to process.
         */
        static void applyDilation(double* targetSet, int size) {
            int i = 0;
            for (; i <= size - 4; i += 4) {
                __m256d vecA = _mm256_loadu_pd(&targetSet[i]);
                // Hardware-accelerated square root
                __m256d result = _mm256_sqrt_pd(vecA);
                _mm256_storeu_pd(&targetSet[i], result);
            }
            for (; i < size; ++i) {
                targetSet[i] = std::sqrt(targetSet[i]);
            }
        }
    };

}

#endif //SPARTAN_CORE_SPARTANFUZZYMATH_H