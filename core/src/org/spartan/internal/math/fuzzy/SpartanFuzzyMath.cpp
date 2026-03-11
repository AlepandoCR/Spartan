//
// Created by Alepando on 23/2/2026.
//

#include "SpartanFuzzyMath.h"
#include "../../simd/SpartanSimd.h"

#include <algorithm>
#include <cmath>

namespace org::spartan::internal::math::fuzzy {

    using namespace org::spartan::internal::math::simd;

    void FuzzySetOps::unionSets(double* targetSet, const double* sourceSet, const int arrayLength) {
        int elementIndex = 0;
        for (; elementIndex <= arrayLength - simdLaneCount; elementIndex += simdLaneCount) {
            const SimdFloat simdTargetValues = simdLoad(&targetSet[elementIndex]);
            const SimdFloat simdSourceValues = simdLoad(&sourceSet[elementIndex]);
            const SimdFloat simdUnionResult = simdMax(simdTargetValues, simdSourceValues);
            simdStore(&targetSet[elementIndex], simdUnionResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = std::max(targetSet[elementIndex], sourceSet[elementIndex]);
        }
    }

    void FuzzySetOps::intersectSets(double* targetSet, const double* sourceSet, const int arrayLength) {
        int elementIndex = 0;
        for (; elementIndex <= arrayLength - simdLaneCount; elementIndex += simdLaneCount) {
            const SimdFloat simdTargetValues = simdLoad(&targetSet[elementIndex]);
            const SimdFloat simdSourceValues = simdLoad(&sourceSet[elementIndex]);
            const SimdFloat simdIntersectionResult = simdMin(simdTargetValues, simdSourceValues);
            simdStore(&targetSet[elementIndex], simdIntersectionResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = std::min(targetSet[elementIndex], sourceSet[elementIndex]);
        }
    }

    void FuzzySetOps::complementSet(double* targetSet, const int arrayLength) {
        int elementIndex = 0;
        const SimdFloat simdIdentityValue = simdBroadcast(1.0);
        for (; elementIndex <= arrayLength - simdLaneCount; elementIndex += simdLaneCount) {
            const SimdFloat simdTargetValues = simdLoad(&targetSet[elementIndex]);
            const SimdFloat simdComplementResult = simdSubtract(simdIdentityValue, simdTargetValues);
            simdStore(&targetSet[elementIndex], simdComplementResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = 1.0 - targetSet[elementIndex];
        }
    }

    void FuzzyModifiers::applyConcentration(double* targetSet, const int arrayLength) {
        int elementIndex = 0;
        for (; elementIndex <= arrayLength - simdLaneCount; elementIndex += simdLaneCount) {
            const SimdFloat simdTargetValues = simdLoad(&targetSet[elementIndex]);
            const SimdFloat simdConcentratedResult = simdMultiply(simdTargetValues, simdTargetValues);
            simdStore(&targetSet[elementIndex], simdConcentratedResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = targetSet[elementIndex] * targetSet[elementIndex];
        }
    }

    void FuzzyModifiers::applyDilation(double* targetSet, const int arrayLength) {
        int elementIndex = 0;
        for (; elementIndex <= arrayLength - simdLaneCount; elementIndex += simdLaneCount) {
            const SimdFloat simdTargetValues = simdLoad(&targetSet[elementIndex]);
            const SimdFloat simdDilatedResult = simdSqrt(simdTargetValues);
            simdStore(&targetSet[elementIndex], simdDilatedResult);
        }
        for (; elementIndex < arrayLength; ++elementIndex) {
            targetSet[elementIndex] = std::sqrt(targetSet[elementIndex]);
        }
    }

}

