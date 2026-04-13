//
// Created by Alepando on 23/2/2026.
//

#include "SpartanMetrics.h"
#include "../../simd/SpartanSimdOps.h"
#include "../../simd/SpartanSimdDispatcher.h"

#include <algorithm>
#include <cmath>

namespace org::spartan::internal::math::metric {

    double VectorMetrics::cosineSimilarity(const double* firstVector, const double* secondVector, const int arrayLength) {
        auto& ops = simd::getSelectedSimdOperations();
        int laneCount = simd::getSimdLaneCount();

        simd::SimdFloat simdDotProductAccumulator = ops.setZero();
        simd::SimdFloat simdFirstMagnitudeSquaredAccumulator = ops.setZero();
        simd::SimdFloat simdSecondMagnitudeSquaredAccumulator = ops.setZero();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - laneCount; elementIndex += laneCount) {
            const simd::SimdFloat simdFirstChunk = ops.load(&firstVector[elementIndex]);
            const simd::SimdFloat simdSecondChunk = ops.load(&secondVector[elementIndex]);

            simdDotProductAccumulator = ops.fusedMultiplyAdd(simdFirstChunk, simdSecondChunk, simdDotProductAccumulator);
            simdFirstMagnitudeSquaredAccumulator = ops.fusedMultiplyAdd(simdFirstChunk, simdFirstChunk, simdFirstMagnitudeSquaredAccumulator);
            simdSecondMagnitudeSquaredAccumulator = ops.fusedMultiplyAdd(simdSecondChunk, simdSecondChunk, simdSecondMagnitudeSquaredAccumulator);
        }

        double dotProduct = ops.horizontalSum(simdDotProductAccumulator);
        double firstMagnitudeSquared = ops.horizontalSum(simdFirstMagnitudeSquaredAccumulator);
        double secondMagnitudeSquared = ops.horizontalSum(simdSecondMagnitudeSquaredAccumulator);

        for (; elementIndex < arrayLength; ++elementIndex) {
            dotProduct += firstVector[elementIndex] * secondVector[elementIndex];
            firstMagnitudeSquared += firstVector[elementIndex] * firstVector[elementIndex];
            secondMagnitudeSquared += secondVector[elementIndex] * secondVector[elementIndex];
        }

        if (firstMagnitudeSquared == 0.0 || secondMagnitudeSquared == 0.0) return 0.0;

        return dotProduct / std::sqrt(firstMagnitudeSquared * secondMagnitudeSquared);
    }

    double VectorMetrics::fuzzyJaccard(const double* firstVector, const double* secondVector, const int arrayLength) {
        auto& ops = simd::getSelectedSimdOperations();
        int laneCount = simd::getSimdLaneCount();

        simd::SimdFloat simdIntersectionAccumulator = ops.setZero();
        simd::SimdFloat simdUnionAccumulator = ops.setZero();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - laneCount; elementIndex += laneCount) {
            const simd::SimdFloat simdFirstChunk = ops.load(&firstVector[elementIndex]);
            const simd::SimdFloat simdSecondChunk = ops.load(&secondVector[elementIndex]);

            simdIntersectionAccumulator = ops.add(simdIntersectionAccumulator, ops.minimum(simdFirstChunk, simdSecondChunk));
            simdUnionAccumulator = ops.add(simdUnionAccumulator, ops.maximum(simdFirstChunk, simdSecondChunk));
        }

        double intersectionSum = ops.horizontalSum(simdIntersectionAccumulator);
        double unionSum = ops.horizontalSum(simdUnionAccumulator);

        for (; elementIndex < arrayLength; ++elementIndex) {
            intersectionSum += std::min(firstVector[elementIndex], secondVector[elementIndex]);
            unionSum += std::max(firstVector[elementIndex], secondVector[elementIndex]);
        }

        if (unionSum == 0.0) return 1.0;
        return intersectionSum / unionSum;
    }

}

