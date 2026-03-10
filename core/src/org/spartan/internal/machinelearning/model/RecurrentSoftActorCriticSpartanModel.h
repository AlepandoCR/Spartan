//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>
#include <cstdint>

#include "SpartanAgent.h"
#include "../ModelHyperparameterConfig.h"
#include "../network/ContinuousQNetwork.h"
#include "../network/GaussianPolicyNetwork.h"
#include "../network/GatedRecurrentUnitLayer.h"
#include "../network/SpartanAbstractCritic.h"

/**
 * @file RecurrentSoftActorCriticSpartanModel.h
 * @brief Concrete Recurrent Soft Actor-Critic agent  -  Frontier A facade.
 *
 * This is the **public-facing** model that the registry and Java Foreign Function and Memory see
 * through the @c SpartanModel / @c SpartanAgent virtual interface.
 * Internally it composes Frontier-B components (Curiously Recurring Template Pattern networks) that are
 * resolved at compile time for maximum Advanced Vector Extensions throughput.
 *
 * Internal architecture:
 *   - 1× GatedRecurrentUnitLayer   -  temporal context encoding
 *   - 1× GaussianPolicyNetwork     -  stochastic actor (pi)
 *   - 2× ContinuousQNetwork        -  twin soft Q-critics (Q1, Q2)
 *
 * All weight/bias/hidden-state buffers are Java Virtual Machine-owned and mapped via
 * @c std::span  -  zero copy, zero allocation on the C++ side.
 */
namespace org::spartan::internal::machinelearning {

    // Forward-declare concrete Curiously Recurring Template Pattern leaf types ─────
    // These are private implementation details; Java never sees them.

    class RecurrentSoftActorCriticGruLayer;
    class RecurrentSoftActorCriticPolicyNetwork;
    class RecurrentSoftActorCriticFirstQNetwork;
    class RecurrentSoftActorCriticSecondQNetwork;

    //
    //  Concrete Curiously Recurring Template Pattern leaves (Frontier B  -  zero-overhead components)
    //

    /**
     * @class RecurrentSoftActorCriticGruLayer
     * @brief Curiously Recurring Template Pattern leaf for the Recurrent Soft Actor-Critic's internal Gated Recurrent Unit layer.
     */
    class RecurrentSoftActorCriticGruLayer final
        : public GatedRecurrentUnitLayer<RecurrentSoftActorCriticGruLayer> {
    public:
        using GatedRecurrentUnitLayer::GatedRecurrentUnitLayer;

        /** @brief Executes one Gated Recurrent Unit time-step (Advanced Vector Extensions-eligible). */
        void forwardPassImpl(
                [[maybe_unused]] std::span<const double> inputVector,
                [[maybe_unused]] std::span<double> hiddenStateInOut) const {
            // TODO: Implement Gated Recurrent Unit gate math (update, reset, candidate).
        }
    };

    /**
     * @class RecurrentSoftActorCriticPolicyNetwork
     * @brief Curiously Recurring Template Pattern leaf for the Recurrent Soft Actor-Critic's stochastic Gaussian actor.
     */
    class RecurrentSoftActorCriticPolicyNetwork final
        : public GaussianPolicyNetwork<RecurrentSoftActorCriticPolicyNetwork> {
    public:
        using GaussianPolicyNetwork::GaussianPolicyNetwork;

        /** @brief Computes action mean and log-std from the observation (AVX-eligible). */
        void computePolicyOutputImpl(
                [[maybe_unused]] std::span<const double> observationState,
                [[maybe_unused]] std::span<double> actionMeanOutput,
                [[maybe_unused]] std::span<double> actionLogStdOutput) const {
            // TODO: Implement forward pass through policy layers.
        }
    };

    /**
     * @class RecurrentSoftActorCriticFirstQNetwork
     * @brief Curiously Recurring Template Pattern leaf for the first soft Q-critic (Q1).
     */
    class RecurrentSoftActorCriticFirstQNetwork final
        : public ContinuousQNetwork<RecurrentSoftActorCriticFirstQNetwork> {
    public:
        using ContinuousQNetwork::ContinuousQNetwork;

        [[nodiscard]] double computeQValueImpl(
                [[maybe_unused]] std::span<const double> observationState,
                [[maybe_unused]] std::span<const double> actionVector) const {
            // TODO: Implement Q-network forward pass.
            return 0.0;
        }
    };

    /**
     * @class RecurrentSoftActorCriticSecondQNetwork
     * @brief Curiously Recurring Template Pattern leaf for the second soft Q-critic (Q2).
     */
    class RecurrentSoftActorCriticSecondQNetwork final
        : public ContinuousQNetwork<RecurrentSoftActorCriticSecondQNetwork> {
    public:
        using ContinuousQNetwork::ContinuousQNetwork;

        [[nodiscard]] double computeQValueImpl(
                [[maybe_unused]] std::span<const double> observationState,
                [[maybe_unused]] std::span<const double> actionVector) const {
            // TODO: Implement Q-network forward pass.
            return 0.0;
        }
    };

    //
    //  Frontier A facade
    //

    /**
     * @class RecurrentSoftActorCriticSpartanModel
     * @brief Public-facing Recurrent Soft Actor-Critic agent registered in the model registry.
     *
     * The registry invokes @c processTick() through a single virtual call.
     * Inside that call, all math is dispatched statically through the
     * Curiously Recurring Template Pattern network members  -  no further vtable lookups.
     */
    class RecurrentSoftActorCriticSpartanModel final : public SpartanAgent {
    public:
        /**
         * @brief Constructs the Recurrent Soft Actor-Critic model, binding all Java Virtual Machine-owned buffers.
         *
         * @param agentIdentifier              Unique agent identifier.
         * @param opaqueHyperparameterConfig   Pointer to a Java Virtual Machine-owned
         *        @c RecurrentSoftActorCriticHyperparameterConfig.
         * @param modelWeights                 Span over the full weight buffer.
         * @param contextBuffer                Span over the observation input.
         * @param actionOutputBuffer           Span over the action output.
         * @param gruGateWeights               Span over Gated Recurrent Unit gate weight matrix.
         * @param gruGateBiases                Span over Gated Recurrent Unit gate bias vector.
         * @param gruHiddenState               Span over Gated Recurrent Unit hidden-state vector.
         * @param policyWeights                Span over policy weight buffer.
         * @param policyBiases                 Span over policy bias buffer.
         * @param firstCriticWeights           Span over Q1 weight buffer.
         * @param firstCriticBiases            Span over Q1 bias buffer.
         * @param secondCriticWeights          Span over Q2 weight buffer.
         * @param secondCriticBiases           Span over Q2 bias buffer.
         */
        RecurrentSoftActorCriticSpartanModel(
                uint64_t agentIdentifier,
                void* opaqueHyperparameterConfig,
                std::span<double> modelWeights,
                std::span<const double> contextBuffer,
                std::span<double> actionOutputBuffer,
                std::span<double> gruGateWeights,
                std::span<double> gruGateBiases,
                std::span<double> gruHiddenState,
                std::span<double> policyWeights,
                std::span<double> policyBiases,
                std::span<double> firstCriticWeights,
                std::span<double> firstCriticBiases,
                std::span<double> secondCriticWeights,
                std::span<double> secondCriticBiases);

        ~RecurrentSoftActorCriticSpartanModel() override = default;

        RecurrentSoftActorCriticSpartanModel(RecurrentSoftActorCriticSpartanModel&&) noexcept = default;
        RecurrentSoftActorCriticSpartanModel& operator=(RecurrentSoftActorCriticSpartanModel&&) noexcept = default;

        // Frontier A virtual interface
        void processTick() override;
        void applyReward(double rewardSignal) override;
        void decayExploration() override;

    private:
        /**
         * @brief Returns the typed config, cast from the opaque base pointer.
         */
        [[nodiscard]] const RecurrentSoftActorCriticHyperparameterConfig* typedConfig() const noexcept {
            return static_cast<const RecurrentSoftActorCriticHyperparameterConfig*>(
                opaqueHyperparameterConfig_);
        }

        // Frontier B components (static dispatch, no vtable)
        RecurrentSoftActorCriticGruLayer recurrentLayer_;
        RecurrentSoftActorCriticPolicyNetwork policyNetwork_;
        RecurrentSoftActorCriticFirstQNetwork firstCriticNetwork_;
        RecurrentSoftActorCriticSecondQNetwork secondCriticNetwork_;
    };

}












