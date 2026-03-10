//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>

/**
 * @file ContinuousQNetwork.h
 * @brief Curiously Recurring Template Pattern base for continuous-action Q-value networks (Frontier B).
 *
 * Provides a compile-time-dispatched @c computeQValue method that
 * concrete Q-network implementations (e.g., the twin critics inside a
 * Recurrent Soft Actor-Critic) must fulfil via @c computeQValueImpl.
 *
 * No vtable, no virtual overhead  -  the compiler can inline and
 * auto-vectorise the underlying Advanced Vector Extensions math in the hot path.
 *
 * @tparam DerivedNetwork The concrete Q-network class.
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class ContinuousQNetwork
     * @brief Curiously Recurring Template Pattern base for state-action value estimation (Q-function).
     *
     * Concrete implementations must provide:
     * @code
     *   double computeQValueImpl(std::span<const double> observationState,
     *                            std::span<const double> actionVector) const;
     * @endcode
     *
     * @tparam DerivedNetwork The concrete network that inherits this base.
     */
    template <typename DerivedNetwork>
    class ContinuousQNetwork {
    public:
        /**
         * @brief Constructs the Q-network with non-owning views over its buffers.
         *
         * @param networkWeights Span over the weight array (Java Virtual Machine-owned).
         * @param networkBiases  Span over the bias array (Java Virtual Machine-owned).
         */
        ContinuousQNetwork(const std::span<double> networkWeights,
                           const std::span<double> networkBiases)
            : networkWeights_(networkWeights),
              networkBiases_(networkBiases) {}

        ~ContinuousQNetwork() = default;

        ContinuousQNetwork(const ContinuousQNetwork&) = default;
        ContinuousQNetwork& operator=(const ContinuousQNetwork&) = default;
        ContinuousQNetwork(ContinuousQNetwork&&) noexcept = default;
        ContinuousQNetwork& operator=(ContinuousQNetwork&&) noexcept = default;

        /**
         * @brief Computes Q(state, action) via static dispatch.
         *
         * @param observationState Read-only span over the current state vector.
         * @param actionVector     Read-only span over the action vector.
         * @return The estimated Q-value for the given state-action pair.
         */
        [[nodiscard]] double computeQValue(
                const std::span<const double> observationState,
                const std::span<const double> actionVector) const {
            return static_cast<const DerivedNetwork*>(this)
                ->computeQValueImpl(observationState, actionVector);
        }

        /**
         * @brief Rebinds to new Java Virtual Machine-owned weight and bias buffers.
         *
         * @param newNetworkWeights Replacement weight span.
         * @param newNetworkBiases  Replacement bias span.
         */
        void rebindNetworkBuffers(const std::span<double> newNetworkWeights,
                                  const std::span<double> newNetworkBiases) {
            networkWeights_ = newNetworkWeights;
            networkBiases_ = newNetworkBiases;
        }

    protected:
        std::span<double> networkWeights_;
        std::span<double> networkBiases_;
    };

}





