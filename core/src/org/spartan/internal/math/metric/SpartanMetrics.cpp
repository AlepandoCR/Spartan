//
// Created by Alepando on 23/2/2026.
//

#include "SpartanMetrics.h"
#include "../../simd/SpartanSimd.h"

#include <algorithm>
#include <cmath>

namespace org::spartan::internal::math::metric {

    using namespace org::spartan::internal::math::simd;

    double VectorMetrics::cosineSimilarity(const double* firstVector, const double* secondVector, const int arrayLength) {
        SimdFloat simdDotProductAccumulator = simdSetZero();
        SimdFloat simdFirstMagnitudeSquaredAccumulator = simdSetZero();
        SimdFloat simdSecondMagnitudeSquaredAccumulator = simdSetZero();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - simdLaneCount; elementIndex += simdLaneCount) {
            const SimdFloat simdFirstChunk = simdLoad(&firstVector[elementIndex]);
            const SimdFloat simdSecondChunk = simdLoad(&secondVector[elementIndex]);

            simdDotProductAccumulator = simdFusedMultiplyAdd(simdFirstChunk, simdSecondChunk, simdDotProductAccumulator);
            simdFirstMagnitudeSquaredAccumulator = simdFusedMultiplyAdd(simdFirstChunk, simdFirstChunk, simdFirstMagnitudeSquaredAccumulator);
            simdSecondMagnitudeSquaredAccumulator = simdFusedMultiplyAdd(simdSecondChunk, simdSecondChunk, simdSecondMagnitudeSquaredAccumulator);
        }

        double dotProduct = simdHorizontalSum(simdDotProductAccumulator);
        double firstMagnitudeSquared = simdHorizontalSum(simdFirstMagnitudeSquaredAccumulator);
        double secondMagnitudeSquared = simdHorizontalSum(simdSecondMagnitudeSquaredAccumulator);

        for (; elementIndex < arrayLength; ++elementIndex) {
            dotProduct += firstVector[elementIndex] * secondVector[elementIndex];
            firstMagnitudeSquared += firstVector[elementIndex] * firstVector[elementIndex];
            secondMagnitudeSquared += secondVector[elementIndex] * secondVector[elementIndex];
        }

        if (firstMagnitudeSquared == 0.0 || secondMagnitudeSquared == 0.0) return 0.0;

        return dotProduct / std::sqrt(firstMagnitudeSquared * secondMagnitudeSquared);
    }

    double VectorMetrics::fuzzyJaccard(const double* firstVector, const double* secondVector, const int arrayLength) {
        SimdFloat simdIntersectionAccumulator = simdSetZero();
        SimdFloat simdUnionAccumulator = simdSetZero();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - simdLaneCount; elementIndex += simdLaneCount) {
            const SimdFloat simdFirstChunk = simdLoad(&firstVector[elementIndex]);
            const SimdFloat simdSecondChunk = simdLoad(&secondVector[elementIndex]);

            simdIntersectionAccumulator = simdAdd(simdIntersectionAccumulator, simdMin(simdFirstChunk, simdSecondChunk));
            simdUnionAccumulator = simdAdd(simdUnionAccumulator, simdMax(simdFirstChunk, simdSecondChunk));
        }

        double intersectionSum = simdHorizontalSum(simdIntersectionAccumulator);
        double unionSum = simdHorizontalSum(simdUnionAccumulator);

        for (; elementIndex < arrayLength; ++elementIndex) {
            intersectionSum += std::min(firstVector[elementIndex], secondVector[elementIndex]);
            unionSum += std::max(firstVector[elementIndex], secondVector[elementIndex]);
        }

        if (unionSum == 0.0) return 1.0;
        return intersectionSum / unionSum;
    }

}

