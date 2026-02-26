//
// Created by Alepando on 24/2/2026.
//

#pragma once

#include <span>

/**
 * @file SpartanAbstractCritic.h
 * @brief Abstract interface for value-function critics in the Spartan pipeline.
 *
 * A Critic evaluates a given state is by producing a scalar value
 * estimate
 *
 * @note This class follows the Zero-Copy / Zero-Allocation architecture:
 *       it holds a non-owning std::span over weight memory allocated and
 *       managed by the JVM via FFM.
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class SpartanAbstractCritic
     * @brief Non-owning view-based interface for state-value estimation.
     *
     * The critic receives a span over its weight buffer at construction time.
     * It never allocates or deallocates memory — the JVM retains ownership
     * of the underlying MemorySegment for the lifetime of the tick.
     */
    class SpartanAbstractCritic {
    public:
        /**
         * @brief Constructs the critic with a non-owning view over its weight buffer.
         *
         * @param weights A span over the weight array managed by the JVM.
         *                Must remain valid for the lifetime of this object.
         */
        explicit SpartanAbstractCritic(std::span<double> weights);

        /** @brief Virtual destructor for safe polymorphic deletion. */
        virtual ~SpartanAbstractCritic() = default;

        SpartanAbstractCritic(const SpartanAbstractCritic&) = default;
        SpartanAbstractCritic& operator=(const SpartanAbstractCritic&) = default;
        SpartanAbstractCritic(SpartanAbstractCritic&&) noexcept = default;
        SpartanAbstractCritic& operator=(SpartanAbstractCritic&&) noexcept = default;

        /**
         * @brief Estimates the value of the given state.
         *
         * @param state A read-only span over the current observation vector.
         *              Its size must match ModelHyperparameterConfig::stateSize.
         * @return The estimated scalar value V(state).
         */
        [[nodiscard]] virtual double evaluate(std::span<const double> state) const = 0;

    protected:
        /** @brief Non-owning view over the critic's weight buffer. */
        std::span<double> weights_;
    };

}

