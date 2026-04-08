//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>
#include <cstdint>
#include <vector>
#include <memory>

#include "SpartanAgent.h"
#include "../ModelHyperparameterConfig.h"
#include "../network/ContinuousQNetwork.h"
#include "../network/GaussianPolicyNetwork.h"
#include "../network/GatedRecurrentUnitLayer.h"
#include "../network/SpartanAbstractCritic.h"
#include "../network/NestedAutoEncoderUnit.h"
#include "../trace/RemorseTraceBuffer.h"
#include "internal/math/tensor/SpartanTensorMath.h"
#include <format>
#include "internal/logging/SpartanLogger.h"

/**
 * @file RecurrentSoftActorCriticSpartanModel.h
 * @brief Concrete Recurrent Soft Actor-Critic agent  -  Frontier A facade.
 *
 * This is the **public-facing** model that the registry and Java Foreign Function and Memory see
 * through the @c SpartanModel / @c SpartanAgent virtual interface.
 * Internally it composes Frontier-B components (Curiously Recurring Template Pattern networks) that are
 * resolved at compile time for maximum throughput.
 *
 * Internal architecture:
 *   - N x NestedAutoEncoderUnit  -  variable-context dimensionality reduction
 *   - 1 x GatedRecurrentUnitLayer   -  temporal context encoding
 *   - 1 x GaussianPolicyNetwork     -  stochastic actor (pi)
 *   - 2 x ContinuousQNetwork        -  twin soft Q-critics (Q1, Q2)
 *   - 1 x RemorseTraceBuffer        -  temporal credit assignment
 *
 * All weight/bias/hidden-state buffers are Java Virtual Machine-owned and mapped via
 * @c std::span  -  zero copy, zero allocation on the C++ side.
 */
namespace org::spartan::internal::machinelearning {

    // Forward-declare concrete Curiously Recurring Template Pattern leaf types
    class RecurrentSoftActorCriticGruLayer;
    class RecurrentSoftActorCriticPolicyNetwork;
    class RecurrentSoftActorCriticFirstQNetwork;
    class RecurrentSoftActorCriticSecondQNetwork;

    //
    //  Concrete Curiously Recurring Template Pattern leaves (Frontier B  -  zero-overhead components)
    //

    /**
     * @class RecurrentSoftActorCriticGruLayer
     * @brief Concrete implementation of a Gated Recurrent Unit using the CRTP base.
     *
     * Processes temporal sequences by maintaining a hidden state. It calculates
     * what to forget, what to remember, and outputs the updated context vector.
     */
    class RecurrentSoftActorCriticGruLayer final
        : public GatedRecurrentUnitLayer<RecurrentSoftActorCriticGruLayer> {
    public:
        using GatedRecurrentUnitLayer::GatedRecurrentUnitLayer;

        /**
         * @brief Executes the GRU forward pass using pre-allocated zero-allocation memory.
         *
         * @param inputVector The raw observation state from the environment.
         * @param hiddenStateInOut The memory state from the previous tick, mutated for the next tick.
         * @param config Structural hyperparameters for matrix slicing.
         * @param gateMemoryBuffer Dynamic scratchpad to store concatenations and gate outputs.
         */
        void forwardPassImpl(
                const std::span<const double> inputVector,
                const std::span<double> hiddenStateInOut,
                const RecurrentSoftActorCriticHyperparameterConfig* config,
                std::span<double> gateMemoryBuffer) const {

            const int inputSize = static_cast<int>(inputVector.size());
            const int hiddenSize = config->hiddenStateSize;
            const int concatSize = hiddenSize + inputSize;

            // Guard: Check if concatenatedInput size matches available source data size
            if (gateMemoryBuffer.size() < static_cast<size_t>(concatSize + (hiddenSize * 3))) {
                 return; // Fail safe
            }

            // Memory Slicing for the Scratchpad
            auto concatenatedInput = gateMemoryBuffer.subspan(0, concatSize);
            auto updateGateZ = gateMemoryBuffer.subspan(concatSize, hiddenSize);
            auto resetGateR = gateMemoryBuffer.subspan(concatSize + hiddenSize, hiddenSize);
            auto candidateStateH = gateMemoryBuffer.subspan(concatSize + (hiddenSize * 2), hiddenSize);

            // Create [h_{t-1}, x_t] concatenation
            std::copy(hiddenStateInOut.begin(), hiddenStateInOut.end(), concatenatedInput.begin());
            std::copy(inputVector.begin(), inputVector.end(), concatenatedInput.begin() + hiddenSize);

            size_t weightOffset = 0;
            size_t biasOffset = 0;

            // Update Gate (Z): Decides how much of the past memory to keep.
            // Z = Sigmoid(W_z * [h_{t-1}, x_t] + b_z)
            std::span<const double> updateGateWeights = this->gateWeights_.subspan(weightOffset, hiddenSize * concatSize);
            std::span<const double> updateGateBiases = this->gateBiases_.subspan(biasOffset, hiddenSize);

            math::tensor::TensorOps::denseForwardPass(concatenatedInput, updateGateWeights, updateGateBiases, updateGateZ);
            if (config->baseConfig.isTraining) {
                math::tensor::TensorOps::applySigmoidExact(updateGateZ);
            } else {
                math::tensor::TensorOps::applySigmoidFast(updateGateZ);
            }

            weightOffset += hiddenSize * concatSize;
            biasOffset += hiddenSize;

            // Reset Gate (R): Decides how much of the past memory to forget before calculating the new state.
            // R = Sigmoid(W_r * [h_{t-1}, x_t] + b_r)
            const std::span<const double> resetGateWeights = this->gateWeights_.subspan(weightOffset, hiddenSize * concatSize);
            const std::span<const double> resetGateBiases = this->gateBiases_.subspan(biasOffset, hiddenSize);

            TensorOps::denseForwardPass(concatenatedInput, resetGateWeights, resetGateBiases, resetGateR);
            if (config->baseConfig.isTraining) {
                TensorOps::applySigmoidExact(resetGateR);
            } else {
                TensorOps::applySigmoidFast(resetGateR);
            }

            weightOffset += hiddenSize * concatSize;
            biasOffset += hiddenSize;

            // Candidate State (H~): The new information to potentially write into memory.
            // First, apply the reset gate to the previous hidden state
            for (int neuronIndex = 0; neuronIndex < hiddenSize; ++neuronIndex) {
                concatenatedInput[neuronIndex] = resetGateR[neuronIndex] * hiddenStateInOut[neuronIndex];
            }

            // H~ = Tanh(W_h * [R * h_{t-1}, x_t] + b_h)
            std::span<const double> candidateWeights = this->gateWeights_.subspan(weightOffset, hiddenSize * concatSize);
            std::span<const double> candidateBiases = this->gateBiases_.subspan(biasOffset, hiddenSize);

            TensorOps::denseForwardPass(concatenatedInput, candidateWeights, candidateBiases, candidateStateH);
            if (config->baseConfig.isTraining) {
                TensorOps::applyTanhExact(candidateStateH);
            } else {
                TensorOps::applyTanh(candidateStateH);
            }

            // Final Hidden State (h_t): Blend the old state and candidate state using the update gate.
            // h_t = (1 - Z) * h_{t-1} + Z * H~
            for (int neuronIndex = 0; neuronIndex < hiddenSize; ++neuronIndex) {
                const double gateValue = updateGateZ[neuronIndex];
                hiddenStateInOut[neuronIndex] = ((1.0 - gateValue) * hiddenStateInOut[neuronIndex]) + (gateValue * candidateStateH[neuronIndex]);
            }
        }
    };

    /**
     * @class RecurrentSoftActorCriticPolicyNetwork
     * @brief Curiously Recurring Template Pattern leaf for the stochastic Gaussian actor.
     *
     * Architecture: hiddenState -> Dense(hidden, hidden) -> Tanh -> Dense(hidden, actionSize*2)
     * The output is split into actionMean and actionLogStd.
     */
    class RecurrentSoftActorCriticPolicyNetwork final
        : public GaussianPolicyNetwork<RecurrentSoftActorCriticPolicyNetwork> {
    public:
        using GaussianPolicyNetwork::GaussianPolicyNetwork;

        /**
         * @brief Computes action mean and log-standard-deviation from the observation state.
         *
         * @param observationState  The GRU hidden state output (read-only).
         * @param actionMeanOutput  Writable span for the action means.
         * @param actionLogStdOutput Writable span for the log-standard-deviations.
         * @param config            Typed hyperparameter config.
         * @param scratchpadBuffer  Pre-allocated working memory (size >= actorHiddenLayerNeuronCount).
         */
        void computePolicyOutputImpl(
                const std::span<const double> observationState,
                const std::span<double> actionMeanOutput,
                const std::span<double> actionLogStdOutput,
                const RecurrentSoftActorCriticHyperparameterConfig* config,
                const std::span<double> scratchpadBuffer) const {

            const int hiddenSize = config->actorHiddenLayerNeuronCount;
            const int actionSize = config->baseConfig.actionSize;
            const int inputSize = static_cast<int>(observationState.size());

            size_t weightOffset = 0;
            size_t biasOffset = 0;

            // Layer 1 Input -> Hidden (Dense + Tanh)
            const auto hiddenActivation = scratchpadBuffer.subspan(0, hiddenSize);

            const std::span<const double> layer1Weights = this->policyWeights_.subspan(weightOffset, hiddenSize * inputSize);
            const std::span<const double> layer1Biases = this->policyBiases_.subspan(biasOffset, hiddenSize);

            TensorOps::denseForwardPass(observationState, layer1Weights, layer1Biases, hiddenActivation);
            if (config->baseConfig.isTraining) {
                TensorOps::applyTanhExact(hiddenActivation);
            } else {
                TensorOps::applyTanh(hiddenActivation);
            }

            weightOffset += hiddenSize * inputSize;
            biasOffset += hiddenSize;

            // Layer 2 Hidden -> Mean output (Dense, linear - no activation)
            const std::span<const double> meanWeights = this->policyWeights_.subspan(weightOffset, actionSize * hiddenSize);
            const std::span<const double> meanBiases = this->policyBiases_.subspan(biasOffset, actionSize);

            const std::span<const double> hiddenView = hiddenActivation;
            TensorOps::denseForwardPass(hiddenView, meanWeights, meanBiases, actionMeanOutput);

            weightOffset += actionSize * hiddenSize;
            biasOffset += actionSize;

            // Layer 3 Hidden -> Log-Std output (Dense, linear - clamped later)
            const std::span<const double> logStdWeights = this->policyWeights_.subspan(weightOffset, actionSize * hiddenSize);
            const std::span<const double> logStdBiases = this->policyBiases_.subspan(biasOffset, actionSize);

            TensorOps::denseForwardPass(hiddenView, logStdWeights, logStdBiases, actionLogStdOutput);

            // Clamp log-std to [-20, 2] to prevent numerical instability
            for (int actionIndex = 0; actionIndex < actionSize; ++actionIndex) {
                if (actionLogStdOutput[actionIndex] < -20.0) actionLogStdOutput[actionIndex] = -20.0;
                if (actionLogStdOutput[actionIndex] > 2.0) actionLogStdOutput[actionIndex] = 2.0;
            }
        }
    };

    /**
     * @class RecurrentSoftActorCriticFirstQNetwork
     * @brief Curiously Recurring Template Pattern leaf for the first soft Q-critic (Q1).
     *
     * Architecture: [state, action] -> Dense(combined, hidden) -> LeakyReLU -> Dense(hidden, 1)
     */
    class RecurrentSoftActorCriticFirstQNetwork final
        : public ContinuousQNetwork<RecurrentSoftActorCriticFirstQNetwork> {
    public:
        using ContinuousQNetwork::ContinuousQNetwork;

        /**
         * @brief Computes the Q-value for a state-action pair.
         *
         * @param observationState  The GRU hidden state (read-only).
         * @param actionVector      The selected action (read-only).
         * @param config            Typed hyperparameter config.
         * @param scratchpadA       Pre-allocated working memory A.
         * @param scratchpadB       Pre-allocated working memory B.
         * @return The estimated Q-value scalar.
         */
        [[nodiscard]] double computeQValueImpl(
                const std::span<const double> observationState,
                const std::span<const double> actionVector,
                const RecurrentSoftActorCriticHyperparameterConfig* config,
                std::span<double> scratchpadA,
                std::span<double> scratchpadB) const {

            const int stateSize = static_cast<int>(observationState.size());
            const int actionSize = static_cast<int>(actionVector.size());
            const int combinedInputSize = stateSize + actionSize;
            const int hiddenSize = config->criticHiddenLayerNeuronCount;

            // Concatenate [state, action] into scratchpadA
            auto combinedInput = scratchpadA.subspan(0, combinedInputSize);
            std::ranges::copy(observationState, combinedInput.begin());
            std::ranges::copy(actionVector, combinedInput.begin() + stateSize);

            size_t weightOffset = 0;
            size_t biasOffset = 0;

            // Layer 1: Combined -> Hidden (Dense + LeakyReLU)
            const auto hiddenActivation = scratchpadB.subspan(0, hiddenSize);

            std::span<const double> layer1Weights = this->networkWeights_.subspan(weightOffset, hiddenSize * combinedInputSize);
            std::span<const double> layer1Biases = this->networkBiases_.subspan(biasOffset, hiddenSize);

            TensorOps::denseForwardPass(combinedInput, layer1Weights, layer1Biases, hiddenActivation);
            TensorOps::applyLeakyReLU(hiddenActivation, 0.01);

            weightOffset += hiddenSize * combinedInputSize;
            biasOffset += hiddenSize;

            // Layer 2 Hidden -> Scalar Q-value (Dense, linear)
            double qValueResult = 0.0;
            auto qValueSpan = std::span(&qValueResult, 1);

            const std::span<const double> outputWeights = this->networkWeights_.subspan(weightOffset, hiddenSize);
            const std::span<const double> outputBiases = this->networkBiases_.subspan(biasOffset, 1);

            const std::span<const double> hiddenView = hiddenActivation;
            TensorOps::denseForwardPass(hiddenView, outputWeights, outputBiases, qValueSpan);

            return qValueResult;
        }
    };

    /**
     * @class RecurrentSoftActorCriticSecondQNetwork
     * @brief Curiously Recurring Template Pattern leaf for the second soft Q-critic (Q2).
     * Identical architecture to Q1 but with independent weights.
     */
    class RecurrentSoftActorCriticSecondQNetwork final
        : public ContinuousQNetwork<RecurrentSoftActorCriticSecondQNetwork> {
    public:
        using ContinuousQNetwork::ContinuousQNetwork;

        [[nodiscard]] double computeQValueImpl(
                const std::span<const double> observationState,
                const std::span<const double> actionVector,
                const RecurrentSoftActorCriticHyperparameterConfig* config,
                std::span<double> scratchpadA,
                std::span<double> scratchpadB) const {

            const int stateSize = static_cast<int>(observationState.size());
            const int actionSize = static_cast<int>(actionVector.size());
            const int combinedInputSize = stateSize + actionSize;
            const int hiddenSize = config->criticHiddenLayerNeuronCount;

            auto combinedInput = scratchpadA.subspan(0, combinedInputSize);
            std::ranges::copy(observationState, combinedInput.begin());
            std::ranges::copy(actionVector, combinedInput.begin() + stateSize);

            size_t weightOffset = 0;
            size_t biasOffset = 0;

            const auto hiddenActivation = scratchpadB.subspan(0, hiddenSize);

            const std::span<const double> layer1Weights = this->networkWeights_.subspan(weightOffset, hiddenSize * combinedInputSize);
            const std::span<const double> layer1Biases = this->networkBiases_.subspan(biasOffset, hiddenSize);

            TensorOps::denseForwardPass(combinedInput, layer1Weights, layer1Biases, hiddenActivation);
            TensorOps::applyLeakyReLU(hiddenActivation, 0.01);

            weightOffset += hiddenSize * combinedInputSize;
            biasOffset += hiddenSize;

            double qValueResult = 0.0;
            const auto qValueSpan = std::span(&qValueResult, 1);

            const std::span<const double> outputWeights = this->networkWeights_.subspan(weightOffset, hiddenSize);
            const std::span<const double> outputBiases = this->networkBiases_.subspan(biasOffset, 1);

            const std::span<const double> hiddenView = hiddenActivation;
            TensorOps::denseForwardPass(hiddenView, outputWeights, outputBiases, qValueSpan);

            return qValueResult;
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
         * @param encoderWeightPool            Span over the flat weight buffer for all nested encoders.
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
                std::span<double> secondCriticBiases,
                std::span<double> encoderWeightPool);

        ~RecurrentSoftActorCriticSpartanModel() override = default;

        RecurrentSoftActorCriticSpartanModel(RecurrentSoftActorCriticSpartanModel&&) noexcept = default;
        RecurrentSoftActorCriticSpartanModel& operator=(RecurrentSoftActorCriticSpartanModel&&) noexcept = default;

        // Frontier A virtual interface
        void processTick() override;
        void applyReward(double rewardSignal) override;
        void decayExploration() override;

        /**
         * @brief Returns the critic weight buffer (GRU + Q1 + Q2) for persistence.
         *
         * The critic weights span is the JVM-owned buffer containing all
         * Gated Recurrent Unit gate weights, gate biases, hidden state,
         * and both Q-critic weight/bias arrays in a flat contiguous layout.
         */
        [[nodiscard]] std::span<const double> getCriticWeights() const noexcept override;

        [[nodiscard]] std::span<double> getCriticWeightsMutable() noexcept override;

    private:
        /**
         * @brief Returns the typed config, cast from the opaque base pointer.
         */
        [[nodiscard]] const RecurrentSoftActorCriticHyperparameterConfig* typedConfig() const noexcept {
            return static_cast<const RecurrentSoftActorCriticHyperparameterConfig*>(
                opaqueHyperparameterConfig_);
        }

        using AlignedMemoryDeleter = void(*)(void*);

        // Frontier B components (static dispatch, no vtable)
        RecurrentSoftActorCriticGruLayer recurrentLayer_;
        RecurrentSoftActorCriticPolicyNetwork policyNetwork_;
        RecurrentSoftActorCriticFirstQNetwork firstCriticNetwork_;
        RecurrentSoftActorCriticSecondQNetwork secondCriticNetwork_;

        // Target critic networks for stable Bellman bootstrap (Polyak-averaged copies).
        // Weights are C++ owned (pre-allocated in constructor), not JVM spans.
        RecurrentSoftActorCriticFirstQNetwork firstTargetCriticNetwork_;
        RecurrentSoftActorCriticSecondQNetwork secondTargetCriticNetwork_;

        // Strictly Aligned Memory Block for AVX-512 Safety
        std::unique_ptr<void, AlignedMemoryDeleter> alignedInternalMemory_;

        // Pre-allocated storage for target critic weights and biases (mapped to aligned block).
        std::span<double> firstTargetCriticWeightStorage_;
        std::span<double> firstTargetCriticBiasStorage_;
        std::span<double> secondTargetCriticWeightStorage_;
        std::span<double> secondTargetCriticBiasStorage_;

        // Nested AutoEncoder bank for variable-context compression
        std::vector<NestedAutoEncoderUnit> nestedEncoderBank_;

        // Temporal credit assignment via hidden-state similarity
        RemorseTraceBuffer remorseTraceBuffer_;

        // Concatenated compressed observation: [fixed context | latent0 | latent1 | ...]
        std::span<double> compressedObservationBuffer_;

        // Allocated strictly once during construction based on hyperparameter config.
        std::span<double> inferenceScratchpadA_;
        std::span<double> inferenceScratchpadB_;

        // Dedicated scratchpads for action mean and log-standard-deviation.
        std::span<double> actionMeanScratchpad_;
        std::span<double> actionLogStdScratchpad_;
        std::span<double> actionStdScratchpad_;
        std::span<double> actionNoiseScratchpad_;
        std::span<double> actionGradientScratchpad_;
        std::span<double> policyLogStdCache_;
        std::span<double> policyHiddenActivationCache_;

        // Additional dedicated memory for the Gated Recurrent Unit's complex gate calculations
        std::span<double> recurrentGateMemoryBuffer_;

        // Scratchpad for remorse trace blame scores
        std::span<double> blameScoresScratchpad_;

        // Per-encoder working memory (flat pool, sliced per encoder)
        std::span<double> encoderScratchpadPool_;

        // Adam optimizer state for Q1 critic weights
        std::span<double> firstCriticWeightMomentum_;
        std::span<double> firstCriticWeightVelocity_;
        std::span<double> firstCriticBiasMomentum_;
        std::span<double> firstCriticBiasVelocity_;

        // Adam optimizer state for Q2 critic weights
        std::span<double> secondCriticWeightMomentum_;
        std::span<double> secondCriticWeightVelocity_;
        std::span<double> secondCriticBiasMomentum_;
        std::span<double> secondCriticBiasVelocity_;

        // Policy optimizer state and gradients
        std::span<double> policyWeightMomentum_;
        std::span<double> policyWeightVelocity_;
        std::span<double> policyBiasMomentum_;
        std::span<double> policyBiasVelocity_;
        std::span<double> policyWeightGradientScratchpad_;
        std::span<double> policyBiasGradientScratchpad_;
        std::span<double> policyHiddenGradientScratchpad_;

        // Gradient scratchpads for critic backward pass
        std::span<double> criticWeightGradientScratchpad_;
        std::span<double> criticBiasGradientScratchpad_;
        std::span<double> criticInputGradientScratchpad_;

        // Critic training buffers (multi-layer backprop)
        std::span<double> criticCombinedInputBuffer_;
        std::span<double> firstCriticActivationCache_;
        std::span<double> secondCriticActivationCache_;

        // Non-owning span over the full JVM-owned critic weight buffer for persistence.
        std::span<const double> criticWeightsSpan_;

        // Running tick counter for remorse trace
        uint64_t currentTickNumber_ = 0;

        // Training step counter for Adam bias correction in critics
        int32_t criticTrainingStepCounter_ = 0;

        // Training step counter for Adam bias correction in the policy
        int32_t policyTrainingStepCounter_ = 0;

        bool hasPolicySnapshot_ = false;
    };

}
