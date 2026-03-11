//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>

/**
 * @file GaussianPolicyNetwork.h
 * @brief Curiously Recurring Template Pattern base for stochastic Gaussian policy networks (Frontier B).
 *
 * The policy network outputs a mean and log-standard-deviation vector
 * for each action dimension, enabling reparameterised sampling for
 * continuous-action RL algorithms (e.g., Soft Actor-Critic).
 *
 * @tparam DerivedPolicy The concrete policy class.
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class GaussianPolicyNetwork
     * @brief Curiously Recurring Template Pattern base for continuous-action stochastic policies.
     *
     * Concrete implementations must provide:
     * @code
     *   void computePolicyOutputImpl(std::span<const double> observationState,
     *                                std::span<double> actionMeanOutput,
     *                                std::span<double> actionLogStdOutput) const;
     * @endcode
     *
     * @tparam DerivedPolicy The concrete policy that inherits this base.
     */
    template <typename DerivedPolicy>
    class GaussianPolicyNetwork {
    public:
        /**
         * @brief Constructs the policy network with non-owning views.
         *
         * @param policyWeights Span over the policy weight array (Java Virtual Machine-owned).
         * @param policyBiases  Span over the policy bias array (Java Virtual Machine-owned).
         */
        GaussianPolicyNetwork(const std::span<double> policyWeights,
                              const std::span<double> policyBiases)
            : policyWeights_(policyWeights),
              policyBiases_(policyBiases) {}

        ~GaussianPolicyNetwork() = default;

        GaussianPolicyNetwork(const GaussianPolicyNetwork&) = default;
        GaussianPolicyNetwork& operator=(const GaussianPolicyNetwork&) = default;
        GaussianPolicyNetwork(GaussianPolicyNetwork&&) noexcept = default;
        GaussianPolicyNetwork& operator=(GaussianPolicyNetwork&&) noexcept = default;

        /**
         * @brief Computes the policy output (mean + log-std) via static dispatch.
         *
         * @param observationState   Read-only span over the current state vector.
         * @param actionMeanOutput   Writable span where the action means are stored.
         * @param actionLogStdOutput Writable span where the log-std values are stored.
         * @param additionalArgs     Dynamic buffers and configs passed from the agent.
         */
        template <typename... AdditionalArgs>
        void computePolicyOutput(
                const std::span<const double> observationState,
                const std::span<double> actionMeanOutput,
                const std::span<double> actionLogStdOutput,
                AdditionalArgs&&... additionalArgs) const {

            static_cast<const DerivedPolicy*>(this)
                ->computePolicyOutputImpl(
                    observationState,
                    actionMeanOutput,
                    actionLogStdOutput,
                    std::forward<AdditionalArgs>(additionalArgs)...
                );
        }

        /**
         * @brief Rebinds to new JVM-owned weight and bias buffers.
         *
         * @param newPolicyWeights Replacement weight span.
         * @param newPolicyBiases  Replacement bias span.
         */
        void rebindPolicyBuffers(const std::span<double> newPolicyWeights,
                                 const std::span<double> newPolicyBiases) {
            policyWeights_ = newPolicyWeights;
            policyBiases_ = newPolicyBiases;
        }

        /** @brief Returns a mutable span over the policy weights for optimizer updates. */
        [[nodiscard]] std::span<double> getPolicyWeights() noexcept { return policyWeights_; }

        /** @brief Returns a mutable span over the policy biases for optimizer updates. */
        [[nodiscard]] std::span<double> getPolicyBiases() noexcept { return policyBiases_; }

    protected:
        std::span<double> policyWeights_;
        std::span<double> policyBiases_;
    };

}



