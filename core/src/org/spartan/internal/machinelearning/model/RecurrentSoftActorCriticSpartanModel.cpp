//
// Created by Alepando on 9/3/2026.
//

#include "RecurrentSoftActorCriticSpartanModel.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "internal/math/reinforcement/SpartanReinforcement.h"

namespace org::spartan::internal::machinelearning {

    RecurrentSoftActorCriticSpartanModel::RecurrentSoftActorCriticSpartanModel(
            const uint64_t agentIdentifier,
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
            std::span<double> encoderWeightPool)
        : SpartanAgent(agentIdentifier,
                        opaqueHyperparameterConfig,
                        modelWeights,
                        contextBuffer,
                        actionOutputBuffer),
          recurrentLayer_(gruGateWeights, gruGateBiases, gruHiddenState),
          policyNetwork_(policyWeights, policyBiases),
          firstCriticNetwork_(firstCriticWeights, firstCriticBiases),
          secondCriticNetwork_(secondCriticWeights, secondCriticBiases),
          remorseTraceBuffer_(0, 0) {

        const auto* config = typedConfig();
        if (!config) return;

        const int hiddenSize = config->actorHiddenLayerNeuronCount;
        const int actionSize = config->baseConfig.actionSize;
        const int contextSize = static_cast<int>(contextBuffer.size());
        const int encoderCount = config->nestedEncoderCount;

        // Pre-allocate the Remorse Trace Buffer
        const int traceCapacity = config->remorseTraceBufferCapacity > 0
            ? config->remorseTraceBufferCapacity : 256;
        remorseTraceBuffer_ = RemorseTraceBuffer(traceCapacity, hiddenSize);

        // Calculate total latent output size from all nested encoders
        int totalLatentDimensions = 0;
        for (int encoderIndex = 0; encoderIndex < encoderCount; ++encoderIndex) {
            totalLatentDimensions += config->encoderSlots[encoderIndex].latentDimensionSize;
        }

        // The compressed observation is: full contextBuffer + all latent vectors concatenated
        const int compressedObservationSize = contextSize + totalLatentDimensions;
        compressedObservationBuffer_.resize(compressedObservationSize);

        // Build the nested encoder bank by slicing the encoderWeightPool
        // Per encoder weight layout:
        //   encoderHiddenWeights  = hiddenNeurons * inputDim
        //   encoderHiddenBiases   = hiddenNeurons
        //   encoderLatentWeights  = latentDim * hiddenNeurons
        //   encoderLatentBiases   = latentDim
        //   decoderHiddenWeights  = hiddenNeurons * latentDim
        //   decoderHiddenBiases   = hiddenNeurons
        //   decoderOutputWeights  = inputDim * hiddenNeurons
        //   decoderOutputBiases   = inputDim
        int totalEncoderScratchpadSize = 0;
        size_t encoderWeightOffset = 0;

        for (int encoderIndex = 0; encoderIndex < encoderCount; ++encoderIndex) {
            const auto& slot = config->encoderSlots[encoderIndex];
            const int inputDim = slot.contextSliceElementCount;
            const int latentDim = slot.latentDimensionSize;
            const int hiddenNeurons = slot.hiddenNeuronCount;

            // Track the max scratchpad needed per encoder: max(hiddenNeurons, inputDim)
            const int encoderScratchSize = hiddenNeurons + latentDim + inputDim;
            totalEncoderScratchpadSize += encoderScratchSize;
        }

        // Pre-allocate all encoder scratchpads in a single flat pool
        encoderScratchpadPool_.resize(totalEncoderScratchpadSize);

        // Now actually construct the encoder units
        size_t scratchpadOffset = 0;
        for (int encoderIndex = 0; encoderIndex < encoderCount; ++encoderIndex) {
            const auto& slot = config->encoderSlots[encoderIndex];
            const int inputDim = slot.contextSliceElementCount;
            const int latentDim = slot.latentDimensionSize;
            const int hiddenNeurons = slot.hiddenNeuronCount;

            const size_t encHiddenWeightCount = static_cast<size_t>(hiddenNeurons) * inputDim;
            const size_t encLatentWeightCount = static_cast<size_t>(latentDim) * hiddenNeurons;
            const size_t decHiddenWeightCount = static_cast<size_t>(hiddenNeurons) * latentDim;
            const size_t decOutputWeightCount = static_cast<size_t>(inputDim) * hiddenNeurons;

            // Slice weights from the pool
            auto encoderHiddenWeights = encoderWeightPool.subspan(encoderWeightOffset, encHiddenWeightCount);
            encoderWeightOffset += encHiddenWeightCount;

            auto encoderHiddenBiases = encoderWeightPool.subspan(encoderWeightOffset, hiddenNeurons);
            encoderWeightOffset += hiddenNeurons;

            auto encoderLatentWeights = encoderWeightPool.subspan(encoderWeightOffset, encLatentWeightCount);
            encoderWeightOffset += encLatentWeightCount;

            auto encoderLatentBiases = encoderWeightPool.subspan(encoderWeightOffset, latentDim);
            encoderWeightOffset += latentDim;

            auto decoderHiddenWeights = encoderWeightPool.subspan(encoderWeightOffset, decHiddenWeightCount);
            encoderWeightOffset += decHiddenWeightCount;

            auto decoderHiddenBiases = encoderWeightPool.subspan(encoderWeightOffset, hiddenNeurons);
            encoderWeightOffset += hiddenNeurons;

            auto decoderOutputWeights = encoderWeightPool.subspan(encoderWeightOffset, decOutputWeightCount);
            encoderWeightOffset += decOutputWeightCount;

            auto decoderOutputBiases = encoderWeightPool.subspan(encoderWeightOffset, inputDim);
            encoderWeightOffset += inputDim;

            // Slice scratchpads from the flat pool
            auto hiddenScratch = std::span(encoderScratchpadPool_).subspan(scratchpadOffset, hiddenNeurons);
            scratchpadOffset += hiddenNeurons;

            auto latentScratch = std::span(encoderScratchpadPool_).subspan(scratchpadOffset, latentDim);
            scratchpadOffset += latentDim;

            auto reconstructionScratch = std::span(encoderScratchpadPool_).subspan(scratchpadOffset, inputDim);
            scratchpadOffset += inputDim;

            nestedEncoderBank_.emplace_back(
                encoderHiddenWeights, encoderHiddenBiases,
                encoderLatentWeights, encoderLatentBiases,
                decoderHiddenWeights, decoderHiddenBiases,
                decoderOutputWeights, decoderOutputBiases,
                latentScratch,        // latent buffer
                hiddenScratch,        // hidden scratchpad
                reconstructionScratch,// reconstruction scratchpad
                inputDim, latentDim, hiddenNeurons
            );
        }

        // Pre-allocate inference scratchpads (large enough for the biggest layer)
        const int criticCombinedSize = hiddenSize + actionSize;
        const int maxScratchSize = std::max({
            hiddenSize,
            config->criticHiddenLayerNeuronCount,
            criticCombinedSize,
            compressedObservationSize
        });

        inferenceScratchpadA_.resize(maxScratchSize);
        inferenceScratchpadB_.resize(maxScratchSize);

        // Action output scratchpads
        actionMeanScratchpad_.resize(actionSize);
        actionLogStdScratchpad_.resize(actionSize);

        // GRU gate memory: [concat(hidden, input)] + [Z] + [R] + [H~]
        recurrentGateMemoryBuffer_.resize(
            (hiddenSize + compressedObservationSize) + (hiddenSize * 3));

        // Blame scores scratchpad (one per trace entry)
        blameScoresScratchpad_.resize(traceCapacity);

        // Pre-allocate Adam optimizer state for twin Q-critics
        const size_t criticWeightCount = firstCriticWeights.size();
        const size_t criticBiasCount = firstCriticBiases.size();

        firstCriticWeightMomentum_.resize(criticWeightCount);
        firstCriticWeightVelocity_.resize(criticWeightCount);
        firstCriticBiasMomentum_.resize(criticBiasCount);
        firstCriticBiasVelocity_.resize(criticBiasCount);

        secondCriticWeightMomentum_.resize(criticWeightCount);
        secondCriticWeightVelocity_.resize(criticWeightCount);
        secondCriticBiasMomentum_.resize(criticBiasCount);
        secondCriticBiasVelocity_.resize(criticBiasCount);

        // Gradient scratchpads for critic backward pass
        criticWeightGradientScratchpad_.resize(criticWeightCount);
        criticInputGradientScratchpad_.resize(maxScratchSize);
    }

    void RecurrentSoftActorCriticSpartanModel::processTick() {
        const auto* config = typedConfig();
        if (!config) return;

        const int actionSize = config->baseConfig.actionSize;
        const int hiddenSize = config->actorHiddenLayerNeuronCount;
        const int contextSize = static_cast<int>(contextBuffer_.size());
        const int encoderCount = config->nestedEncoderCount;

        //
        // Phase A: Build the observation vector for the GRU.
        //
        // If no encoders exist, the raw contextBuffer_ is passed directly to the GRU
        // without any copy. When encoders exist, the fixed context is memcpy'd once
        // into the compressed buffer and latent outputs are appended after it.
        //
        std::span<const double> gruInputObservation;

        if (encoderCount == 0) {
            // Zero-copy fast path: pass the JVM-owned context directly to the GRU.
            gruInputObservation = contextBuffer_;
        } else {
            // Copy the fixed context block via memcpy (single burst write to L1 cache)
            std::memcpy(
                compressedObservationBuffer_.data(),
                contextBuffer_.data(),
                static_cast<size_t>(contextSize) * sizeof(double));

            const bool hasCleanSizes = !cleanSizesBuffer_.empty()
                && static_cast<int>(cleanSizesBuffer_.size()) >= encoderCount;

            int latentWriteOffset = contextSize;
            for (int encoderIndex = 0; encoderIndex < encoderCount; ++encoderIndex) {
                const auto& slot = config->encoderSlots[encoderIndex];

                // Determine the actual valid element count for this encoder's input slice.
                // If clean sizes are available, use the minimum of (cleanSize, slotCapacity)
                // to guard against Java sending a count larger than the pre-allocated slot.
                const int slotCapacity = slot.contextSliceElementCount;
                const int validElementCount = hasCleanSizes
                    ? std::min(static_cast<int>(cleanSizesBuffer_[encoderIndex]), slotCapacity)
                    : slotCapacity;

                // Create a clean view spanning only the valid elements.
                // Any padding beyond validElementCount is ignored by the encoder.
                const auto cleanContextSlice = contextBuffer_.subspan(
                    slot.contextSliceStartIndex, validElementCount);

                nestedEncoderBank_[encoderIndex].encode(cleanContextSlice);

                const auto latentOutput = nestedEncoderBank_[encoderIndex].getLatentOutput();
                std::memcpy(
                    compressedObservationBuffer_.data() + latentWriteOffset,
                    latentOutput.data(),
                    static_cast<size_t>(slot.latentDimensionSize) * sizeof(double));
                latentWriteOffset += slot.latentDimensionSize;
            }

            gruInputObservation = std::span<const double>(compressedObservationBuffer_);
        }

        //
        // Phase B Pass the observation through the Gated Recurrent Unit
        //          to update the temporal hidden state.
        //
        recurrentLayer_.forwardPass(
            gruInputObservation,
            recurrentLayer_.getMutableHiddenState(),
            config,
            std::span(recurrentGateMemoryBuffer_));

        //
        //  Run the Gaussian policy network to produce action mean and log-std.
        //
        const std::span<const double> hiddenStateView = recurrentLayer_.getHiddenState();
        policyNetwork_.computePolicyOutput(
            hiddenStateView,
            std::span(actionMeanScratchpad_),
            std::span(actionLogStdScratchpad_),
            config,
            std::span(inferenceScratchpadA_));

        //
        // Convert log-std to std in-place via exp, then apply Gaussian noise
        //          (reparameterization trick) to produce the final stochastic action.
        //
        // The log-std values are destroyed here since they are not needed again this tick.
        // This eliminates a redundant copy to a separate actionStdScratchpad_.
        //
        TensorOps::applyExpFast(std::span(actionLogStdScratchpad_));

        //
        // Phase E: Write the final noisy action into the JVM-owned action output buffer.
        //
        TensorOps::applyGaussianNoise(
            std::span<const double>(actionMeanScratchpad_),
            std::span<const double>(actionLogStdScratchpad_),
            actionOutputBuffer_);

        //
        // Phase F: Record the current cognitive snapshot for temporal credit assignment.
        //
        const int selectedActionIndex = static_cast<int>(
            TensorOps::findArgmax(
                std::span<const double>(actionOutputBuffer_.data(), actionSize)));

        remorseTraceBuffer_.recordSnapshot(
            currentTickNumber_,
            selectedActionIndex,
            hiddenStateView);

        //
        // Phase G (training only): Evaluate twin Q-critics and update the actor policy.
        //
        // SAC uses the minimum of two independent Q-estimates (Twin Critics) to
        // combat the overestimation bias inherent in single-critic methods.
        // The actor is updated to maximize: Q_min(s, a) + alpha * H(pi)
        //
        if (config->baseConfig.isTraining) {
            const std::span<const double> actionView(actionOutputBuffer_.data(), actionSize);

            const double firstQValue = firstCriticNetwork_.computeQValue(
                hiddenStateView, actionView, config,
                std::span(inferenceScratchpadA_),
                std::span(inferenceScratchpadB_));

            const double secondQValue = secondCriticNetwork_.computeQValue(
                hiddenStateView, actionView, config,
                std::span(inferenceScratchpadA_),
                std::span(inferenceScratchpadB_));

            // Twin Clipped Q-value: take the pessimistic estimate
            const double minimumQValue = std::min(firstQValue, secondQValue);

            // Entropy bonus: H(pi) = -sum(log_std) (per-dimension Gaussian entropy)
            // Since actionLogStdScratchpad_ was overwritten by exp() in Phase D,
            // we approximate the entropy from the action standard deviations.
            // H(Gaussian) = 0.5 * ln(2*pi*e*sigma^2) per dimension
            // For the gradient, we only need the sign: positive entropy = more exploration.
            double entropyEstimate = 0.0;
            for (int dimensionIndex = 0; dimensionIndex < actionSize; ++dimensionIndex) {
                // actionLogStdScratchpad_ now contains exp(logStd) = std
                const double standardDeviation = actionLogStdScratchpad_[dimensionIndex];
                if (standardDeviation > 1e-8) {
                    entropyEstimate += std::log(standardDeviation);
                }
            }

            // Actor objective: maximize Q_min + alpha * entropy
            // Gradient direction for the policy weights:
            //   dJ/dW_policy = dQ/da * da/dW + alpha * dH/dW
            // We approximate with a scaled remorse-style update where the "regret"
            // signal is the negative of the actor objective (we want to ascend).
            const double actorObjective = minimumQValue + config->entropyTemperatureAlpha * entropyEstimate;

            // Scale the update by the policy learning rate
            // Positive actorObjective means the current policy is doing well;
            // negative means it needs stronger correction.
            math::reinforcement::GradientOps::applyRemorseUpdate(
                policyNetwork_.getPolicyWeights().data(),
                hiddenStateView.data(),
                actorObjective,
                config->policyNetworkLearningRate,
                hiddenSize);
        }

        ++currentTickNumber_;
    }

    void RecurrentSoftActorCriticSpartanModel::applyReward(const double rewardSignal) {
        const auto* config = typedConfig();
        if (!config || !config->baseConfig.isTraining) return;

        const int hiddenSize = config->actorHiddenLayerNeuronCount;
        const int actionSize = config->baseConfig.actionSize;
        const int traceEntryCount = remorseTraceBuffer_.currentEntryCount();

        //
        // Blame distribution for the Actor via Remorse Trace.
        //
        if (traceEntryCount > 0) {
            const std::span<const double> currentHiddenState = recurrentLayer_.getHiddenState();

            remorseTraceBuffer_.computeBlameScores(
                currentHiddenState,
                std::span(blameScoresScratchpad_),
                config->remorseMinimumSimilarityThreshold);

            for (int32_t entryIndex = 0; entryIndex < traceEntryCount; ++entryIndex) {
                const double blameScore = blameScoresScratchpad_[entryIndex];
                if (blameScore <= 0.0) continue;

                const double effectiveReward = rewardSignal * blameScore;

                const double* archivedStatePointer =
                    remorseTraceBuffer_.getArchivedHiddenStatePointer(entryIndex);
                math::reinforcement::GradientOps::applyRemorseUpdate(
                    policyNetwork_.getPolicyWeights().data(),
                    archivedStatePointer,
                    effectiveReward,
                    config->policyNetworkLearningRate,
                    hiddenSize);
            }
        }

        //
        // Train both Q-critics using Temporal Difference error.
        //
        // The TD target for SAC is:
        //   y = r + gamma * (min(Q1_target, Q2_target)(s', a') - alpha * log pi(a'|s'))
        //
        // Since we don't have separate target networks for the critics yet (future phase),
        // we use the current Q-values as a self-consistent bootstrap:
        //   y = r + gamma * min(Q1, Q2) - alpha * entropy_estimate
        //
        // Each critic is updated independently to minimize (Q_i(s, a) - y)^2.
        //
        const std::span<const double> hiddenStateView = recurrentLayer_.getHiddenState();
        const std::span<const double> actionView(actionOutputBuffer_.data(), actionSize);

        const double firstQValue = firstCriticNetwork_.computeQValue(
            hiddenStateView, actionView, config,
            std::span(inferenceScratchpadA_),
            std::span(inferenceScratchpadB_));

        const double secondQValue = secondCriticNetwork_.computeQValue(
            hiddenStateView, actionView, config,
            std::span(inferenceScratchpadA_),
            std::span(inferenceScratchpadB_));

        const double minimumQValue = std::min(firstQValue, secondQValue);

        // Estimate entropy from the current standard deviations (already exp'd in processTick)
        double entropyEstimate = 0.0;
        for (int dimensionIndex = 0; dimensionIndex < actionSize; ++dimensionIndex) {
            if (const double standardDeviation = actionLogStdScratchpad_[dimensionIndex]; standardDeviation > 1e-8) {
                entropyEstimate += std::log(standardDeviation);
            }
        }

        // TD target: y = r + gamma * (Q_min_bootstrap - alpha * entropy)
        const double gamma = config->baseConfig.gamma;
        const double alpha = config->entropyTemperatureAlpha;
        const double temporalDifferenceTarget = rewardSignal
            + gamma * (minimumQValue - alpha * entropyEstimate);

        ++criticTrainingStepCounter_;

        // Train Q1 critic
        {
            const double temporalDifferenceErrorQ1 = 2.0 * (firstQValue - temporalDifferenceTarget);

            std::ranges::fill(criticWeightGradientScratchpad_, 0.0);

            criticInputGradientScratchpad_[0] = temporalDifferenceErrorQ1;

            math::tensor::TensorOps::denseBackwardPass(
                hiddenStateView,
                std::span<const double>(criticInputGradientScratchpad_.data(), 1),
                std::span<const double>(firstCriticNetwork_.getNetworkWeights()),
                std::span(criticWeightGradientScratchpad_),
                std::span(inferenceScratchpadA_));

            TensorOps::applyAdamUpdate(
                firstCriticNetwork_.getNetworkWeights(),
                std::span<const double>(criticWeightGradientScratchpad_),
                std::span(firstCriticWeightMomentum_),
                std::span(firstCriticWeightVelocity_),
                config->firstCriticLearningRate, 0.9, 0.999, 1e-8,
                criticTrainingStepCounter_);
        }

        // Train Q2 critic
        {
            const double temporalDifferenceErrorQ2 = 2.0 * (secondQValue - temporalDifferenceTarget);

            std::ranges::fill(criticWeightGradientScratchpad_, 0.0);

            criticInputGradientScratchpad_[0] = temporalDifferenceErrorQ2;

            TensorOps::denseBackwardPass(
                hiddenStateView,
                std::span<const double>(criticInputGradientScratchpad_.data(), 1),
                std::span<const double>(secondCriticNetwork_.getNetworkWeights()),
                std::span(criticWeightGradientScratchpad_),
                std::span(inferenceScratchpadA_));

            TensorOps::applyAdamUpdate(
                secondCriticNetwork_.getNetworkWeights(),
                std::span<const double>(criticWeightGradientScratchpad_),
                std::span(secondCriticWeightMomentum_),
                std::span(secondCriticWeightVelocity_),
                config->secondCriticLearningRate, 0.9, 0.999, 1e-8,
                criticTrainingStepCounter_);
        }
    }

    void RecurrentSoftActorCriticSpartanModel::decayExploration() {
        const auto* config = typedConfig();
        if (!config) return;

        // Soft Actor-Critic uses entropy temperature (alpha) rather than epsilon-greedy.
        // The entropy temperature auto-tuning will be implemented when the full
        // dual-gradient SAC training loop is wired.  For now, decay the base epsilon
        // for hybrid exploration policies.
        if (auto* mutableConfig = const_cast<BaseHyperparameterConfig*>(&config->baseConfig); mutableConfig->epsilon > mutableConfig->epsilonMin) {
            mutableConfig->epsilon *= mutableConfig->epsilonDecay;
            if (mutableConfig->epsilon < mutableConfig->epsilonMin) {
                mutableConfig->epsilon = mutableConfig->epsilonMin;
            }
        }

        // Reset the remorse trace at episode boundaries
        remorseTraceBuffer_.reset();
    }

}

