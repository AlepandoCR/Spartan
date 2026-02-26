//
// Created by Alepando on 24/2/2026.
//

#pragma once

#include <cstdint>

/**
 * @file ModelHyperparameterConfig.h
 * @brief C-compatible struct holding all tunable hyperparameters for an ML model.
 *
 * This struct uses Standard Layout to guarantee ABI compatibility with C
 * and direct memory mapping via Java FFM (Foreign Function & Memory API).
 * No constructors, no virtual methods, no inheritance — pure POD.
 */

extern "C" {

    /**
     * @struct


     * @brief Standard Layout hyperparameter block shared between the JVM and native engine.
     *
     * The JVM allocates this struct in off-heap memory via MemorySegment and passes
     * a raw pointer to the C++ side. Both sides read/write the same memory region
     * with zero serialization overhead.
     *
     * @note All fields are intentionally public and trivially copyable.
     */
    struct ModelHyperparameterConfig {

        /** @brief Step size for gradient descent updates. Typical range: [1e-5, 1e-1]. */
        double learningRate;

        /** @brief Discount factor for future rewards in temporal-difference learning. Range: [0.0, 1.0]. */
        double gamma;

        /** @brief Current exploration probability for epsilon-greedy policies. Range: [0.0, 1.0]. */
        double epsilon;

        double epsilonMin;

        /** @brief Multiplicative decay applied to epsilon after each episode. Range: [0.0, 1.0]. */
        double epsilonDecay;

        /** @brief Flag indicating whether the model is in training mode (true) or inference-only (false). */
        bool isTraining;
    };

}


