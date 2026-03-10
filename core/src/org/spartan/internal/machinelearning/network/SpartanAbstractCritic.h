//
// Created by Alepando on 24/2/2026.
//

#pragma once

#include <span>

/**
 * @file SpartanAbstractCritic.h
 * @brief Curiously Recurring Template Pattern base for value-function critic networks (Frontier B).
 *
 * Uses the Curiously Recurring Template Pattern to achieve
 * **static polymorphism**  -  all calls to @c evaluate are resolved at
 * compile time, eliminating vtable lookups in the hot Advanced Vector Extensions math path.
 *
 * This is an internal Frontier-B network component, on the same level as
 * @c ContinuousQNetwork, @c GaussianPolicyNetwork, and
 * @c GatedRecurrentUnitLayer.  It is composed **inside** concrete
 * Frontier-A models (agents, standalone critics) it does NOT inherit
 * from @c SpartanModel.
 *
 * @tparam DerivedCritic The concrete critic type (e.g., @c TemporalDifferenceCritic).
 *
 * @note This class follows the Zero-Copy / Zero-Allocation architecture:
 *       it holds a non-owning std::span over weight memory allocated and
 *       managed by the Java Virtual Machine via the Foreign Function and Memory interface.
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class SpartanAbstractCritic
     * @brief Curiously Recurring Template Pattern non-owning view-based base for state-value estimation.
     *
     * Concrete critics must implement:
     * @code
     *   double evaluateImpl(std::span<const double> state) const;
     * @endcode
     *
     * @tparam DerivedCritic The concrete critic class that inherits this base.
     */
    template <typename DerivedCritic>
    class SpartanAbstractCritic {
    public:
        /**
         * @brief Constructs the critic with non-owning views over its buffers.
         *
         * @param criticWeights Span over the weight array (Java Virtual Machine-owned).
         * @param criticBiases  Span over the bias array (Java Virtual Machine-owned).
         */
        SpartanAbstractCritic(const std::span<double> criticWeights,
                              const std::span<double> criticBiases)
            : criticWeights_(criticWeights),
              criticBiases_(criticBiases) {}

        ~SpartanAbstractCritic() = default;

        SpartanAbstractCritic(const SpartanAbstractCritic&) = default;
        SpartanAbstractCritic& operator=(const SpartanAbstractCritic&) = default;
        SpartanAbstractCritic(SpartanAbstractCritic&&) noexcept = default;
        SpartanAbstractCritic& operator=(SpartanAbstractCritic&&) noexcept = default;

        /**
         * @brief Estimates the value of the given state via static dispatch.
         *
         * @param observationState A read-only span over the current observation vector.
         * @return The estimated scalar value V(state).
         */
        [[nodiscard]] double evaluate(const std::span<const double> observationState) const {
            return static_cast<const DerivedCritic*>(this)->evaluateImpl(observationState);
        }

        /**
         * @brief Rebinds the critic to new weight and bias buffers.
         *
         * @param newCriticWeights Replacement weight span.
         * @param newCriticBiases  Replacement bias span.
         */
        void rebindCriticBuffers(const std::span<double> newCriticWeights,
                                 const std::span<double> newCriticBiases) {
            criticWeights_ = newCriticWeights;
            criticBiases_ = newCriticBiases;
        }

    protected:
        std::span<double> criticWeights_;
        std::span<double> criticBiases_;
    };

}



