//
// Created by Alepando on 23/2/2026.
//

#include "SpartanFuzzyMath.h"
#include "../../simd/SpartanSimdOps.h"
#include "../../simd/SpartanSimdDispatcher.h"

#include <algorithm>
#include <cmath>

namespace org::spartan::internal::math::fuzzy {

    void FuzzySetOps::unionSets(double* targetSet, const double* sourceSet, const int arrayLength) {
        auto& ops = simd::getSelectedSimdOperations();
        int laneCount = simd::getSimdLaneCount();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - laneCount; elementIndex += laneCount) {
            const simd::SimdFloat simdTargetValues = ops.load(&targetSet[elementIndex]);
            const simd::SimdFloat simdSourceValues = ops.load(&sourceSet[elementIndex]);
            const simd::SimdFloat simdUnionResult = ops.maximum(simdTargetValues, simdSourceValues);
            ops.store(&targetSet[elementIndex], simdUnionResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = std::max(targetSet[elementIndex], sourceSet[elementIndex]);
        }
    }

    void FuzzySetOps::intersectSets(double* targetSet, const double* sourceSet, const int arrayLength) {
        auto& ops = simd::getSelectedSimdOperations();
        int laneCount = simd::getSimdLaneCount();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - laneCount; elementIndex += laneCount) {
            const simd::SimdFloat simdTargetValues = ops.load(&targetSet[elementIndex]);
            const simd::SimdFloat simdSourceValues = ops.load(&sourceSet[elementIndex]);
            const simd::SimdFloat simdIntersectionResult = ops.minimum(simdTargetValues, simdSourceValues);
            ops.store(&targetSet[elementIndex], simdIntersectionResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = std::min(targetSet[elementIndex], sourceSet[elementIndex]);
        }
    }

    void FuzzySetOps::complementSet(double* targetSet, const int arrayLength) {
        auto& ops = simd::getSelectedSimdOperations();
        int laneCount = simd::getSimdLaneCount();

        int elementIndex = 0;
        const simd::SimdFloat simdIdentityValue = ops.broadcast(1.0);
        for (; elementIndex <= arrayLength - laneCount; elementIndex += laneCount) {
            const simd::SimdFloat simdTargetValues = ops.load(&targetSet[elementIndex]);
            const simd::SimdFloat simdComplementResult = ops.subtract(simdIdentityValue, simdTargetValues);
            ops.store(&targetSet[elementIndex], simdComplementResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = 1.0 - targetSet[elementIndex];
        }
    }

    void FuzzyModifiers::applyConcentration(double* targetSet, const int arrayLength) {
        auto& ops = simd::getSelectedSimdOperations();
        int laneCount = simd::getSimdLaneCount();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - laneCount; elementIndex += laneCount) {
            const simd::SimdFloat simdTargetValues = ops.load(&targetSet[elementIndex]);
            const simd::SimdFloat simdConcentratedResult = ops.multiply(simdTargetValues, simdTargetValues);
            ops.store(&targetSet[elementIndex], simdConcentratedResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = targetSet[elementIndex] * targetSet[elementIndex];
        }
    }

    void FuzzyModifiers::applyDilation(double* targetSet, const int arrayLength) {
        auto& ops = simd::getSelectedSimdOperations();
        int laneCount = simd::getSimdLaneCount();

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - laneCount; elementIndex += laneCount) {
            const simd::SimdFloat simdTargetValues = ops.load(&targetSet[elementIndex]);
            const simd::SimdFloat simdDilatedResult = ops.sqrt(simdTargetValues);
            ops.store(&targetSet[elementIndex], simdDilatedResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = std::sqrt(targetSet[elementIndex]);
        }
    }

}

