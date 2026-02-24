//
// Created by aocam on 23/2/2026.
//

#ifndef SPARTAN_CORE_SPARTANREINFORCEMENT_H
#define SPARTAN_CORE_SPARTANREINFORCEMENT_H
#include <immintrin.h>

/**
 * @namespace org::spartan::core::math::reinforcement
 * @brief Reinforcement Learning mathematical operations and gradient adjustments.
 */
namespace org::spartan::core::math::reinforcement {

    /**
     * @class GradientOps
     * @brief High-speed operations for adjusting neural/heuristic weights.
     */
    class GradientOps {
    public:
        GradientOps() = delete;

        /**
         * @brief Applies a Regret-based update (Remorse) to a synaptic weight array.
         * Formula: W_new = W_old + (learning_rate * remorse * input_features)
         * * @param weights Pointer to the decision weights (Mutated in place).
         * @param features The sensory context that led to the action.
         * @param remorse The calculated regret (Optimal_Reward - Actual_Reward).
         * @param learningRate Scaling factor for the adjustment.
         * @param size Length of the arrays.
         */
        static void applyRemorseUpdate(double* weights, const double* features, double remorse, double learningRate, int size) {

            // Scalar factor, pre-multiplied for efficiency in the SIMD loop
            double adjustmentFactor = learningRate * remorse;
            __m256d vecFactor = _mm256_set1_pd(adjustmentFactor);

            int i = 0;
            for (; i <= size - 4; i += 4) {
                __m256d w = _mm256_loadu_pd(&weights[i]);
                __m256d x = _mm256_loadu_pd(&features[i]);

                // delta = factor * features
                __m256d delta = _mm256_mul_pd(vecFactor, x);

                 // new_w = old_w + delta
                __m256d new_w = _mm256_add_pd(w, delta);

                _mm256_storeu_pd(&weights[i], new_w);
            }

            for (; i < size; ++i) {
                weights[i] += adjustmentFactor * features[i];
            }
        }
    };

}
#endif //SPARTAN_CORE_SPARTANREINFORCEMENT_H