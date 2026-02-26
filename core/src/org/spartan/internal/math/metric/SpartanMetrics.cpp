//
// Created by Alepando on 23/2/2026.
//

#include "SpartanMetrics.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace org::spartan::internal::math::metric {

    double VectorMetrics::cosineSimilarity(const double* firstVector, const double* secondVector, const int arrayLength) {
        __m256d simdDotProductAccumulator = _mm256_setzero_pd();
        __m256d simdFirstMagnitudeSquaredAccumulator = _mm256_setzero_pd();
        __m256d simdSecondMagnitudeSquaredAccumulator = _mm256_setzero_pd();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - 4; elementIndex += 4) {
            const __m256d simdFirstChunk = _mm256_loadu_pd(&firstVector[elementIndex]);
            const __m256d simdSecondChunk = _mm256_loadu_pd(&secondVector[elementIndex]);

            simdDotProductAccumulator = _mm256_add_pd(simdDotProductAccumulator, _mm256_mul_pd(simdFirstChunk, simdSecondChunk));
            simdFirstMagnitudeSquaredAccumulator = _mm256_add_pd(simdFirstMagnitudeSquaredAccumulator, _mm256_mul_pd(simdFirstChunk, simdFirstChunk));
            simdSecondMagnitudeSquaredAccumulator = _mm256_add_pd(simdSecondMagnitudeSquaredAccumulator, _mm256_mul_pd(simdSecondChunk, simdSecondChunk));
        }

        double dotProductLanes[4], firstMagnitudeSquaredLanes[4], secondMagnitudeSquaredLanes[4];
        _mm256_storeu_pd(dotProductLanes, simdDotProductAccumulator);
        _mm256_storeu_pd(firstMagnitudeSquaredLanes, simdFirstMagnitudeSquaredAccumulator);
        _mm256_storeu_pd(secondMagnitudeSquaredLanes, simdSecondMagnitudeSquaredAccumulator);

        double dotProduct = dotProductLanes[0] + dotProductLanes[1] + dotProductLanes[2] + dotProductLanes[3];
        double firstMagnitudeSquared = firstMagnitudeSquaredLanes[0] + firstMagnitudeSquaredLanes[1] + firstMagnitudeSquaredLanes[2] + firstMagnitudeSquaredLanes[3];
        double secondMagnitudeSquared = secondMagnitudeSquaredLanes[0] + secondMagnitudeSquaredLanes[1] + secondMagnitudeSquaredLanes[2] + secondMagnitudeSquaredLanes[3];

        for (; elementIndex < arrayLength; ++elementIndex) {
            dotProduct += firstVector[elementIndex] * secondVector[elementIndex];
            firstMagnitudeSquared += firstVector[elementIndex] * firstVector[elementIndex];
            secondMagnitudeSquared += secondVector[elementIndex] * secondVector[elementIndex];
        }

        if (firstMagnitudeSquared == 0.0 || secondMagnitudeSquared == 0.0) return 0.0;

        return dotProduct / std::sqrt(firstMagnitudeSquared * secondMagnitudeSquared);
    }

    double VectorMetrics::fuzzyJaccard(const double* firstVector, const double* secondVector, const int arrayLength) {
        __m256d simdIntersectionAccumulator = _mm256_setzero_pd();
        __m256d simdUnionAccumulator = _mm256_setzero_pd();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - 4; elementIndex += 4) {
            const __m256d simdFirstChunk = _mm256_loadu_pd(&firstVector[elementIndex]);
            const __m256d simdSecondChunk = _mm256_loadu_pd(&secondVector[elementIndex]);

            simdIntersectionAccumulator = _mm256_add_pd(simdIntersectionAccumulator, _mm256_min_pd(simdFirstChunk, simdSecondChunk));
            simdUnionAccumulator = _mm256_add_pd(simdUnionAccumulator, _mm256_max_pd(simdFirstChunk, simdSecondChunk));
        }

        double intersectionLanes[4], unionLanes[4];
        _mm256_storeu_pd(intersectionLanes, simdIntersectionAccumulator);
        _mm256_storeu_pd(unionLanes, simdUnionAccumulator);

        double intersectionSum = intersectionLanes[0] + intersectionLanes[1] + intersectionLanes[2] + intersectionLanes[3];
        double unionSum = unionLanes[0] + unionLanes[1] + unionLanes[2] + unionLanes[3];

        for (; elementIndex < arrayLength; ++elementIndex) {
            intersectionSum += std::min(firstVector[elementIndex], secondVector[elementIndex]);
            unionSum += std::max(firstVector[elementIndex], secondVector[elementIndex]);
        }

        if (unionSum == 0.0) return 1.0;
        return intersectionSum / unionSum;
    }

}

