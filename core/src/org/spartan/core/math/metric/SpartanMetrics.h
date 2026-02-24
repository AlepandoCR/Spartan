//
// Created by Alepando on 23/2/2026.
//

#ifndef SPARTAN_CORE_SPARTANMETRICS_H
#define SPARTAN_CORE_SPARTANMETRICS_H

#include <immintrin.h>
#include <algorithm>
#include <cmath>

/**
 * @namespace org::spartan::core::math::metric
 * @brief Hardware-accelerated distance and similarity metrics for vector spaces.
 */
namespace org::spartan::core::math::reinforcement {

    /**
     * @class VectorMetrics
     * @brief Computes spatial and logical similarities between raw context arrays.
     * Essential for temporal coherence (tracking entities across ticks without using IDs).
     */
    class VectorMetrics {
    public:
        VectorMetrics() = delete;

        /**
         * @brief Computes the Cosine Similarity between two context vectors.
         * Measures the cosine of the angle between two multi-dimensional vectors.
         * Useful for variable context elements (SpartanVariableContextElement).
         * * @return A value between -1.0 (opposite) and 1.0 (identical).
         */
        static double cosineSimilarity(const double* vecA, const double* vecB, int size) {
            // Initialize accumulators for dot product and magnitudes
            __m256d sum_dot = _mm256_setzero_pd();
            __m256d sum_sqA = _mm256_setzero_pd();
            __m256d sum_sqB = _mm256_setzero_pd();


            // AVX Unrolled Loop (Process 4 doubles at a time)
            int i = 0;
            for (; i <= size - 4; i += 4) {
                __m256d a = _mm256_loadu_pd(&vecA[i]); // Load 4 elements from vecA
                __m256d b = _mm256_loadu_pd(&vecB[i]); // Load 4 elements from vecB

                // sum_dot += a * b
                sum_dot = _mm256_add_pd(sum_dot, _mm256_mul_pd(a, b));
                // sum_sqA += a * a
                sum_sqA = _mm256_add_pd(sum_sqA, _mm256_mul_pd(a, a));
                // sum_sqB += b * b
                sum_sqB = _mm256_add_pd(sum_sqB, _mm256_mul_pd(b, b));
            }

            // Horizontal add to sum the elements in the AVX registers
            double dot_arr[4], sqA_arr[4], sqB_arr[4];

            // Store the results back to arrays for final summation
            _mm256_storeu_pd(dot_arr, sum_dot);
            _mm256_storeu_pd(sqA_arr, sum_sqA);
            _mm256_storeu_pd(sqB_arr, sum_sqB);

            // Sum the partial results from the AVX registers
            double dot = dot_arr[0] + dot_arr[1] + dot_arr[2] + dot_arr[3];
            double magA = sqA_arr[0] + sqA_arr[1] + sqA_arr[2] + sqA_arr[3];
            double magB = sqB_arr[0] + sqB_arr[1] + sqB_arr[2] + sqB_arr[3];

            // Tail loop for remaining elements
            for (; i < size; ++i) {
                dot += vecA[i] * vecB[i];
                magA += vecA[i] * vecA[i];
                magB += vecB[i] * vecB[i];
            }

            // Handle edge case where magnitude is zero to avoid division by zero
            if (magA == 0.0 || magB == 0.0) return 0.0;

            //          (sqrt(A) x sqrt(B) = sqrt(A x B))
            return dot / std::sqrt(magA * magB);
        }

        /**
         * @brief Computes the Fuzzy Jaccard Index: |A ∩ B| / |A U B|
         * Measures the overlap between two fuzzy sets.
         * Excellent for hot-encoded features (SpartanHotContextElement).
         * * @return A coherence value between 0.0 (disjoint) and 1.0 (identical).
         */
        static double fuzzyJaccard(const double* vecA, const double* vecB, int size) {
            __m256d sum_min = _mm256_setzero_pd();
            __m256d sum_max = _mm256_setzero_pd();

            int i = 0;
            for (; i <= size - 4; i += 4) {
                __m256d a = _mm256_loadu_pd(&vecA[i]);
                __m256d b = _mm256_loadu_pd(&vecB[i]);

                sum_min = _mm256_add_pd(sum_min, _mm256_min_pd(a, b));
                sum_max = _mm256_add_pd(sum_max, _mm256_max_pd(a, b));
            }

            double min_arr[4], max_arr[4];
            _mm256_storeu_pd(min_arr, sum_min);
            _mm256_storeu_pd(max_arr, sum_max);

            double intersectionSum = min_arr[0] + min_arr[1] + min_arr[2] + min_arr[3];
            double unionSum = max_arr[0] + max_arr[1] + max_arr[2] + max_arr[3];

            for (; i < size; ++i) {
                intersectionSum += std::min(vecA[i], vecB[i]);
                unionSum += std::max(vecA[i], vecB[i]);
            }

            if (unionSum == 0.0) return 1.0;
            return intersectionSum / unionSum;
        }
    };

}

#endif //SPARTAN_CORE_SPARTANMETRICS_H