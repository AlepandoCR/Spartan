//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>
#include <cstdint>

#include "SpartanAgent.h"
#include "../ModelHyperparameterConfig.h"
#include "../network/ContinuousQNetwork.h"

/**
 * @file DoubleDeepQNetworkSpartanModel.h
 * @brief Concrete Double Deep Q-Network agent  -  Frontier A facade.
 *
 * Exposes the standard @c SpartanAgent virtual interface to the registry
 * while composing Frontier-B Curiously Recurring Template Pattern Q-networks internally for compile-time
 * dispatch and Advanced Vector Extensions vectorisation.
 *
 * Internal architecture:
 *   - 1× ContinuousQNetwork   -  online Q-network  (selects actions)
 *   - 1× ContinuousQNetwork   -  target Q-network  (stabilises training)
 *
 * All weight/bias buffers are JVM-owned and mapped via @c std::span.
 */
namespace org::spartan::internal::machinelearning {

    // Concrete Curiously Recurring Template Pattern leaves

    /**
     * @class DoubleDeepQNetworkOnlineNetwork
     * @brief Curiously Recurring Template Pattern leaf for the Double Deep Q-Network online (behaviour) Q-network.
     */
    class DoubleDeepQNetworkOnlineNetwork final
        : public ContinuousQNetwork<DoubleDeepQNetworkOnlineNetwork> {
    public:
        using ContinuousQNetwork::ContinuousQNetwork;

        [[nodiscard]] double computeQValueImpl(
                [[maybe_unused]] std::span<const double> observationState,
                [[maybe_unused]] std::span<const double> actionVector) const {
            // TODO: Implement online Q-network forward pass.
            return 0.0;
        }
    };

    /**
     * @class DoubleDeepQNetworkTargetNetwork
     * @brief Curiously Recurring Template Pattern leaf for the Double Deep Q-Network target (frozen) Q-network.
     */
    class DoubleDeepQNetworkTargetNetwork final
        : public ContinuousQNetwork<DoubleDeepQNetworkTargetNetwork> {
    public:
        using ContinuousQNetwork::ContinuousQNetwork;

        [[nodiscard]] double computeQValueImpl(
                [[maybe_unused]] std::span<const double> observationState,
                [[maybe_unused]] std::span<const double> actionVector) const {
            // TODO: Implement target Q-network forward pass.
            return 0.0;
        }
    };

    //
    //  Frontier A facade
    //

    /**
     * @class DoubleDeepQNetworkSpartanModel
     * @brief Public-facing Double Deep Q-Network agent registered in the model registry.
     *
     * The registry invokes @c processTick() through a single virtual call.
     * Internally, the online and target Q-networks are Curiously Recurring Template Pattern  -  zero vtable
     * overhead in the math path.
     */
    class DoubleDeepQNetworkSpartanModel final : public SpartanAgent {
    public:
        /**
         * @brief Constructs the Double Deep Q-Network model, binding all Java Virtual Machine-owned buffers.
         *
         * @param agentIdentifier              Unique agent identifier.
         * @param opaqueHyperparameterConfig   Pointer to a Java Virtual Machine-owned
         *        @c DoubleDeepQNetworkHyperparameterConfig.
         * @param modelWeights                 Span over the full weight buffer.
         * @param contextBuffer                Span over the observation input.
         * @param actionOutputBuffer           Span over the action output.
         * @param onlineNetworkWeights         Span over online Q-network weights.
         * @param onlineNetworkBiases          Span over online Q-network biases.
         * @param targetNetworkWeights         Span over target Q-network weights.
         * @param targetNetworkBiases          Span over target Q-network biases.
         */
        DoubleDeepQNetworkSpartanModel(
                uint64_t agentIdentifier,
                void* opaqueHyperparameterConfig,
                std::span<double> modelWeights,
                std::span<const double> contextBuffer,
                std::span<double> actionOutputBuffer,
                std::span<double> onlineNetworkWeights,
                std::span<double> onlineNetworkBiases,
                std::span<double> targetNetworkWeights,
                std::span<double> targetNetworkBiases);

        ~DoubleDeepQNetworkSpartanModel() override = default;

        DoubleDeepQNetworkSpartanModel(DoubleDeepQNetworkSpartanModel&&) noexcept = default;
        DoubleDeepQNetworkSpartanModel& operator=(DoubleDeepQNetworkSpartanModel&&) noexcept = default;

        // Frontier A virtual interface
        void processTick() override;
        void applyReward(double rewardSignal) override;
        void decayExploration() override;

    private:
        [[nodiscard]] const DoubleDeepQNetworkHyperparameterConfig* typedConfig() const noexcept {
            return static_cast<const DoubleDeepQNetworkHyperparameterConfig*>(
                opaqueHyperparameterConfig_);
        }

        /** @brief Tick counter for target-network sync scheduling. */
        int32_t ticksSinceLastTargetSync_ = 0;

        // Frontier B components (static dispatch, no vtable)
        DoubleDeepQNetworkOnlineNetwork onlineNetwork_;
        DoubleDeepQNetworkTargetNetwork targetNetwork_;
    };

}






