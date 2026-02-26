//
// Created by Alepando on 23/2/2026.
//

#pragma once

/**
 * @namespace org::spartan::core::math::reinforcement
 * @brief Reinforcement Learning mathematical operations and gradient adjustments.
 */
namespace org::spartan::internal::math::reinforcement {

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
         *
         * @param weights Pointer to the decision weights (Mutated in place).
         * @param features The sensory context that led to the action.
         * @param remorse The calculated regret (Optimal_Reward - Actual_Reward).
         * @param learningRate Scaling factor for the adjustment.
         * @param arrayLength Length of the weight and feature arrays.
         *
         * TODO ask for {@link SpartanBaseModel} instead of raw arrays and extract spans internally
         */
        static void applyRemorseUpdate(double* weights, const double* features, double remorse, double learningRate, int arrayLength);
    };

}
