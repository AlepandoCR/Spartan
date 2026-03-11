//
// Created by Alepando on 23/2/2026.
//

#include "SpartanReinforcement.h"
#include "../../simd/SpartanSimd.h"

namespace org::spartan::internal::math::reinforcement {

    using namespace org::spartan::internal::math::simd;

    void GradientOps::applyRemorseUpdate(double* weights, const double* features, const double remorse, const double learningRate, const int arrayLength) {

        // Pre-multiply the remorse and learning rate to save some cycles in the loop
        const double adjustmentFactor = learningRate * remorse;
        const SimdFloat simdAdjustmentFactor = simdBroadcast(adjustmentFactor);

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - simdLaneCount; elementIndex += simdLaneCount) {
            const SimdFloat currentWeights = simdLoad(&weights[elementIndex]);
            const SimdFloat currentFeatures = simdLoad(&features[elementIndex]);

            // weightDelta = adjustmentFactor * features
            const SimdFloat weightDelta = simdMultiply(simdAdjustmentFactor, currentFeatures);

            // updatedWeights = currentWeights + weightDelta
            const SimdFloat updatedWeights = simdAdd(currentWeights, weightDelta);

            simdStore(&weights[elementIndex], updatedWeights);
        }

        for (; elementIndex < arrayLength; ++elementIndex) {
            weights[elementIndex] += adjustmentFactor * features[elementIndex];
        }
    }

}

