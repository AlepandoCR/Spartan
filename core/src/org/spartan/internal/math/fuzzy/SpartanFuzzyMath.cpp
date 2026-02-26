//
// Created by Alepando on 23/2/2026.
//

#include "SpartanFuzzyMath.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace org::spartan::internal::math::fuzzy {

    // --- FuzzySetOps ---

    void FuzzySetOps::unionSets(double* targetSet, const double* sourceSet, const int arrayLength) {
        int elementIndex = 0;
        for (; elementIndex <= arrayLength - 4; elementIndex += 4) {
            const __m256d simdTargetValues = _mm256_loadu_pd(&targetSet[elementIndex]);
            const __m256d simdSourceValues = _mm256_loadu_pd(&sourceSet[elementIndex]);
            const __m256d simdUnionResult = _mm256_max_pd(simdTargetValues, simdSourceValues);
            _mm256_storeu_pd(&targetSet[elementIndex], simdUnionResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = std::max(targetSet[elementIndex], sourceSet[elementIndex]);
        }
    }

    void FuzzySetOps::intersectSets(double* targetSet, const double* sourceSet, const int arrayLength) {
        int elementIndex = 0;
        for (; elementIndex <= arrayLength - 4; elementIndex += 4) {
            const __m256d simdTargetValues = _mm256_loadu_pd(&targetSet[elementIndex]);
            const __m256d simdSourceValues = _mm256_loadu_pd(&sourceSet[elementIndex]);
            const __m256d simdIntersectionResult = _mm256_min_pd(simdTargetValues, simdSourceValues);
            _mm256_storeu_pd(&targetSet[elementIndex], simdIntersectionResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = std::min(targetSet[elementIndex], sourceSet[elementIndex]);
        }
    }

    void FuzzySetOps::complementSet(double* targetSet, const int arrayLength) {
        int elementIndex = 0;
        const __m256d simdIdentityValue = _mm256_set1_pd(1.0);
        for (; elementIndex <= arrayLength - 4; elementIndex += 4) {
            const __m256d simdTargetValues = _mm256_loadu_pd(&targetSet[elementIndex]);
            const __m256d simdComplementResult = _mm256_sub_pd(simdIdentityValue, simdTargetValues);
            _mm256_storeu_pd(&targetSet[elementIndex], simdComplementResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = 1.0 - targetSet[elementIndex];
        }
    }

    void FuzzyModifiers::applyConcentration(double* targetSet, const int arrayLength) {
        int elementIndex = 0;
        for (; elementIndex <= arrayLength - 4; elementIndex += 4) {
            const __m256d simdTargetValues = _mm256_loadu_pd(&targetSet[elementIndex]);
            const __m256d simdConcentratedResult = _mm256_mul_pd(simdTargetValues, simdTargetValues);
            _mm256_storeu_pd(&targetSet[elementIndex], simdConcentratedResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = targetSet[elementIndex] * targetSet[elementIndex];
        }
    }

    void FuzzyModifiers::applyDilation(double* targetSet, const int arrayLength) {
        int elementIndex = 0;
        for (; elementIndex <= arrayLength - 4; elementIndex += 4) {
            const __m256d simdTargetValues = _mm256_loadu_pd(&targetSet[elementIndex]);
            const __m256d simdDilatedResult = _mm256_sqrt_pd(simdTargetValues);
            _mm256_storeu_pd(&targetSet[elementIndex], simdDilatedResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = std::sqrt(targetSet[elementIndex]);
        }
    }

}

