//
// Created by Alepando on 23/2/2026.
//

#include "SpartanReinforcement.h"
#include "../../simd/SpartanSimdOps.h"
#include "../../simd/SpartanSimdDispatcher.h"

namespace org::spartan::internal::math::reinforcement {

    void GradientOps::applyRemorseUpdate(double* weights, const double* features, const double remorse, const double learningRate, const int arrayLength) {

        auto& ops = simd::getSelectedSimdOperations();
        int laneCount = simd::getSimdLaneCount();

        // Pre-multiply the remorse and learning rate to save some cycles in the loop
        const double adjustmentFactor = learningRate * remorse;
        const simd::SimdFloat simdAdjustmentFactor = ops.broadcast(adjustmentFactor);

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - laneCount; elementIndex += laneCount) {
            const simd::SimdFloat currentWeights = ops.load(&weights[elementIndex]);
            const simd::SimdFloat currentFeatures = ops.load(&features[elementIndex]);

            // weightDelta = adjustmentFactor * features
            const simd::SimdFloat weightDelta = ops.multiply(simdAdjustmentFactor, currentFeatures);

            // updatedWeights = currentWeights + weightDelta
            const simd::SimdFloat updatedWeights = ops.add(currentWeights, weightDelta);

            ops.store(&weights[elementIndex], updatedWeights);
        }

        for (; elementIndex < arrayLength; ++elementIndex) {
            weights[elementIndex] += adjustmentFactor * features[elementIndex];
        }
    }

}

