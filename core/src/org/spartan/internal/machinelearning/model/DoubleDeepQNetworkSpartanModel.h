//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>

#include "SpartanAgent.h"
#include "../ModelHyperparameterConfig.h"
#include "../network/ContinuousQNetwork.h"
#include "../replay/ExperienceReplayBuffer.h"
#include "internal/math/tensor/SpartanTensorMath.h"

/**
 * @file DoubleDeepQNetworkSpartanModel.h
 * @brief Concrete Double Deep Q-Network agent  -  Frontier A facade.
 *
 * Exposes the standard @c SpartanAgent virtual interface to the registry
 * while composing Frontier-B Curiously Recurring Template Pattern Q-networks internally for compile-time
 * dispatch and Advanced Vector Extensions vectorisation.
 *
 * Internal architecture:
 * - 1× ContinuousQNetwork   -  online Q-network  (selects actions)
 * - 1× ContinuousQNetwork   -  target Q-network  (stabilises training)
 *
 * All weight/bias buffers are JVM-owned and mapped via @c std::span.
 */
namespace org::spartan::internal::machinelearning {

    using namespace org::spartan::internal::math::tensor;

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
                const std::span<const double> observationState,
                const std::span<const double> actionVector,
                const DoubleDeepQNetworkHyperparameterConfig* config,
                std::span<double> scratchpadA,
                std::span<double> scratchpadB) const {

            const int inputSize = static_cast<int>(observationState.size() + actionVector.size());
            const int hiddenSize = config->hiddenLayerNeuronCount;

            assert(scratchpadA.size() >= hiddenSize && "Scratchpad A is too small");
            assert(scratchpadB.size() >= hiddenSize && "Scratchpad B is too small");

            auto currentInput = scratchpadA.subspan(0, inputSize);
            std::copy(observationState.begin(), observationState.end(), currentInput.begin());
            std::copy(actionVector.begin(), actionVector.end(),
                      currentInput.begin() + static_cast<std::ptrdiff_t>(observationState.size()));

            size_t weightOffset = 0;
            size_t biasOffset = 0;

            auto currentOutput = scratchpadB.subspan(0, hiddenSize);

            // Input to First Hidden Layer
            const std::span<const double> w1 = this->networkWeights_.subspan(weightOffset, hiddenSize * inputSize);
            const std::span<const double> b1 = this->networkBiases_.subspan(biasOffset, hiddenSize);

            TensorOps::denseForwardPass(currentInput, w1, b1, currentOutput);
            TensorOps::applyLeakyReLU(currentOutput, 0.01);

            weightOffset += hiddenSize * inputSize;
            biasOffset += hiddenSize;

            // Hidden Layers Propagation
            for (int layer = 1; layer < config->hiddenLayerCount; ++layer) {
                currentInput = currentOutput;
                currentOutput = (layer % 2 == 1) ? scratchpadA.subspan(0, hiddenSize) : scratchpadB.subspan(0, hiddenSize);

                const std::span<const double> wHidden = this->networkWeights_.subspan(weightOffset, hiddenSize * hiddenSize);
                const std::span<const double> bHidden = this->networkBiases_.subspan(biasOffset, hiddenSize);

                TensorOps::denseForwardPass(currentInput, wHidden, bHidden, currentOutput);
                TensorOps::applyLeakyReLU(currentOutput, 0.01);

                weightOffset += hiddenSize * hiddenSize;
                biasOffset += hiddenSize;
            }

            // Final Output Layer
            double finalQValue = 0.0;
            auto finalOutputSpan = std::span(&finalQValue, 1);

            const std::span<const double> wOut = this->networkWeights_.subspan(weightOffset, hiddenSize);
            const std::span<const double> bOut = this->networkBiases_.subspan(biasOffset, 1);

            TensorOps::denseForwardPass(currentOutput, wOut, bOut, finalOutputSpan);

            return finalQValue;
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
                const std::span<const double> observationState,
                const std::span<const double> actionVector,
                const DoubleDeepQNetworkHyperparameterConfig* config,
                std::span<double> scratchpadA,
                std::span<double> scratchpadB) const {

            const int inputSize = static_cast<int>(observationState.size() + actionVector.size());
            const int hiddenSize = config->hiddenLayerNeuronCount;

            auto currentInput = scratchpadA.subspan(0, inputSize);
            std::ranges::copy(observationState, currentInput.begin());
            std::ranges::copy(actionVector,
                              currentInput.begin() + static_cast<std::ptrdiff_t>(observationState.size()));

            size_t weightOffset = 0;
            size_t biasOffset = 0;

            auto currentOutput = scratchpadB.subspan(0, hiddenSize);

            const std::span<const double> w1 = this->networkWeights_.subspan(weightOffset, hiddenSize * inputSize);
            const std::span<const double> b1 = this->networkBiases_.subspan(biasOffset, hiddenSize);

            TensorOps::denseForwardPass(currentInput, w1, b1, currentOutput);
            TensorOps::applyLeakyReLU(currentOutput, 0.01);

            weightOffset += hiddenSize * inputSize;
            biasOffset += hiddenSize;

            for (int layer = 1; layer < config->hiddenLayerCount; ++layer) {
                currentInput = currentOutput;
                currentOutput = (layer % 2 == 1) ? scratchpadA.subspan(0, hiddenSize) : scratchpadB.subspan(0, hiddenSize);

                const std::span<const double> wHidden = this->networkWeights_.subspan(weightOffset, hiddenSize * hiddenSize);
                const std::span<const double> bHidden = this->networkBiases_.subspan(biasOffset, hiddenSize);

                TensorOps::denseForwardPass(currentInput, wHidden, bHidden, currentOutput);
                TensorOps::applyLeakyReLU(currentOutput, 0.01);

                weightOffset += hiddenSize * hiddenSize;
                biasOffset += hiddenSize;
            }

            double finalQValue = 0.0;
            auto finalOutputSpan = std::span(&finalQValue, 1);

            const std::span<const double> wOut = this->networkWeights_.subspan(weightOffset, hiddenSize);
            const std::span<const double> bOut = this->networkBiases_.subspan(biasOffset, 1);

            TensorOps::denseForwardPass(currentOutput, wOut, bOut, finalOutputSpan);

            return finalQValue;
        }

        std::span<double> getTargetWeights() { return this->networkWeights_; }
        std::span<double> getTargetBiases() { return this->networkBiases_; }
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
         * @c DoubleDeepQNetworkHyperparameterConfig.
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

        /** @brief Global training step counter for Adam bias correction. */
        int32_t trainingStepCounter_ = 0;

        // Frontier B components (static dispatch, no vtable)
        DoubleDeepQNetworkOnlineNetwork onlineNetwork_;
        DoubleDeepQNetworkTargetNetwork targetNetwork_;

        // Needed for Polyak updates
        std::span<const double> rawOnlineWeights_;
        std::span<const double> rawOnlineBiases_;

        // Experience replay for off-policy learning
        replay::ExperienceReplayBuffer replayBuffer_;

        // Snapshot of the previous tick's state for storing transitions
        std::vector<double> previousStateSnapshot_;

        // Snapshot of the previous tick's action for storing transitions
        std::vector<double> previousActionSnapshot_;

        // Pre-allocated batch index buffer for mini-batch sampling
        std::vector<int32_t> batchIndicesBuffer_;

        // Q-value output scratchpads (one per action for the full Q-vector)
        std::vector<double> onlineQValuesScratchpad_;
        std::vector<double> targetQValuesScratchpad_;

        // Gradient scratchpads for backpropagation
        std::vector<double> onlineWeightGradients_;
        std::vector<double> onlineBiasGradients_;
        std::vector<double> outputGradientScratchpad_;
        std::vector<double> inputGradientScratchpad_;

        // Adam optimizer state for online network weights
        std::vector<double> adamWeightMomentum_;
        std::vector<double> adamWeightVelocity_;
        std::vector<double> adamBiasMomentum_;
        std::vector<double> adamBiasVelocity_;

        // General-purpose inference scratchpads
        std::vector<double> scratchpadA_;
        std::vector<double> scratchpadB_;

        // Flag indicating whether we have a valid previous state to store
        bool hasPreviousState_ = false;
    };

}