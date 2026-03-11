//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>
#include <utility>

/**
 * @file ContinuousQNetwork.h
 * @brief Curiously Recurring Template Pattern base for continuous-action Q-value networks.
 *
 * Provides a compile-time-dispatched computeQValue method that
 * concrete Q-network implementations must fulfil via computeQValueImpl.
 *
 * No vtable, no virtual overhead - the compiler can inline and
 * auto-vectorize the underlying math in the hot path.
 *
 * @tparam DerivedNetwork The concrete Q-network class.
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class ContinuousQNetwork
     * @brief Base for state-action value estimation.
     *
     * Concrete implementations must provide a computeQValueImpl method.
     * Using variadic templates allows the model agent to pass dynamic
     * scratchpads and configurations without forcing this base class to
     * know about specific hyperparameter struct types.
     *
     * @tparam DerivedNetwork The concrete network that inherits this base.
     */
    template <typename DerivedNetwork>
    class ContinuousQNetwork {
    public:
        /**
         * @brief Constructs the Q-network with non-owning views over its buffers.
         *
         * @param networkWeights Span over the weight array.
         * @param networkBiases  Span over the bias array.
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
         * @brief Computes the estimated value via static dispatch.
         *
         * Forwards the state, action, and any additional memory buffers
         * (like zero-allocation scratchpads) directly to the concrete implementation.
         *
         * @param observationState Read-only span over the current state vector.
         * @param actionVector     Read-only span over the action vector.
         * @param additionalArgs   Dynamic buffers and configs passed from the agent.
         * @return The estimated Q-value for the given state-action pair.
         */
        template <typename... AdditionalArgs>
        [[nodiscard]] double computeQValue(
                const std::span<const double> observationState,
                const std::span<const double> actionVector,
                AdditionalArgs&&... additionalArgs) const {

            return static_cast<const DerivedNetwork*>(this)
                ->computeQValueImpl(
                    observationState,
                    actionVector,
                    std::forward<AdditionalArgs>(additionalArgs)...
                );
        }

        /**
         * @brief Rebinds to new weight and bias buffers.
         *
         * @param newNetworkWeights Replacement weight span.
         * @param newNetworkBiases  Replacement bias span.
         */
        void rebindNetworkBuffers(const std::span<double> newNetworkWeights,
                                  const std::span<double> newNetworkBiases) {
            networkWeights_ = newNetworkWeights;
            networkBiases_ = newNetworkBiases;
        }

        /** @brief Returns a mutable span over the network weights for optimizer updates. */
        [[nodiscard]] std::span<double> getNetworkWeights() noexcept { return networkWeights_; }

        /** @brief Returns a mutable span over the network biases for optimizer updates. */
        [[nodiscard]] std::span<double> getNetworkBiases() noexcept { return networkBiases_; }

    protected:
        std::span<double> networkWeights_;
        std::span<double> networkBiases_;
    };

}