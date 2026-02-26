//
// Created by Alepando on 23/2/2026.
//

#include "SpartanReinforcement.h"

#include <immintrin.h>

namespace org::spartan::core::math::reinforcement {

    void GradientOps::applyRemorseUpdate(double* weights, const double* features, const double remorse, const double learningRate, const int arrayLength) {

        // pre multiply the remorse and learning rate to save some cycles in the loop
        const double adjustmentFactor = learningRate * remorse;
        const __m256d simdAdjustmentFactor = _mm256_set1_pd(adjustmentFactor);

        int elementIndex = 0;
        for (; elementIndex <= arrayLength - 4; elementIndex += 4) {
            const __m256d currentWeights = _mm256_loadu_pd(&weights[elementIndex]);
            const __m256d currentFeatures = _mm256_loadu_pd(&features[elementIndex]);

            // weightDelta = adjustmentFactor * features
            const __m256d weightDelta = _mm256_mul_pd(simdAdjustmentFactor, currentFeatures);

            // updatedWeights = currentWeights + weightDelta
            const __m256d updatedWeights = _mm256_add_pd(currentWeights, weightDelta);

            _mm256_storeu_pd(&weights[elementIndex], updatedWeights);
        }

        for (; elementIndex < arrayLength; ++elementIndex) {
            weights[elementIndex] += adjustmentFactor * features[elementIndex];
        }
    }

}

