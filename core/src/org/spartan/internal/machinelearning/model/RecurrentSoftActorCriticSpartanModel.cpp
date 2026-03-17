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
          firstTargetCriticNetwork_({}, {}),
          secondTargetCriticNetwork_({}, {}),
          alignedInternalMemory_(nullptr, [](void* ptr) {
              if (ptr) {
#if defined(_WIN32)
                  _aligned_free(ptr);
#else
                  free(ptr);
#endif
              }
          }),
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

        // Helper to ensure SIMD alignment (64 bytes = 8 doubles)
        auto alignSize = [](size_t size) -> size_t {
            return (size + 7) & ~static_cast<size_t>(7);
        };

        // Calculate total latent output size from all nested encoders
        int totalLatentDimensions = 0;
        for (int encoderIndex = 0; encoderIndex < encoderCount; ++encoderIndex) {
            totalLatentDimensions += config->encoderSlots[encoderIndex].latentDimensionSize;
        }

        const int compressedObservationSize = contextSize + totalLatentDimensions;

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

        // --- ALLOCATION STRATEGY: Single Flat Aligned Block for all Vectors ---

        size_t totalDoublesNeeded = 0;
        totalDoublesNeeded += alignSize(compressedObservationSize);
        totalDoublesNeeded += alignSize(totalEncoderScratchpadSize);

        // Pre-allocate inference scratchpads (large enough for the biggest layer)
        const int criticCombinedSize = hiddenSize + actionSize;
        const int maxScratchSize = std::max({
            hiddenSize,
            config->criticHiddenLayerNeuronCount,
            criticCombinedSize,
            compressedObservationSize
        });

        totalDoublesNeeded += alignSize(maxScratchSize); // inferenceScratchpadA_
        totalDoublesNeeded += alignSize(maxScratchSize); // inferenceScratchpadB_
        totalDoublesNeeded += alignSize(actionSize);     // actionMeanScratchpad_
        totalDoublesNeeded += alignSize(actionSize);     // actionLogStdScratchpad_

        // GRU gate memory: [concat(hidden, input)] + [Z] + [R] + [H~]
        size_t gruMemSize = (hiddenSize + compressedObservationSize) + (hiddenSize * 3);
        totalDoublesNeeded += alignSize(gruMemSize);

        // Blame scores scratchpad (one per trace entry)
        totalDoublesNeeded += alignSize(traceCapacity);

        // Pre-allocate Adam optimizer state for twin Q-critics
        const size_t criticWeightCount = firstCriticWeights.size();
        const size_t criticBiasCount = firstCriticBiases.size();

        totalDoublesNeeded += alignSize(criticWeightCount); // firstCriticWeightMomentum_
        totalDoublesNeeded += alignSize(criticWeightCount); // firstCriticWeightVelocity_
        totalDoublesNeeded += alignSize(criticBiasCount);   // firstCriticBiasMomentum_
        totalDoublesNeeded += alignSize(criticBiasCount);   // firstCriticBiasVelocity_

        totalDoublesNeeded += alignSize(criticWeightCount); // secondCriticWeightMomentum_
        totalDoublesNeeded += alignSize(criticWeightCount); // secondCriticWeightVelocity_
        totalDoublesNeeded += alignSize(criticBiasCount);   // secondCriticBiasMomentum_
        totalDoublesNeeded += alignSize(criticBiasCount);   // secondCriticBiasVelocity_

        // Gradient scratchpads for critic backward pass
        totalDoublesNeeded += alignSize(criticWeightCount); // criticWeightGradientScratchpad_
        totalDoublesNeeded += alignSize(criticBiasCount);   // criticBiasGradientScratchpad_
        totalDoublesNeeded += alignSize(maxScratchSize);    // criticInputGradientScratchpad_

        // Target Critic Storage
        totalDoublesNeeded += alignSize(criticWeightCount); // firstTargetCriticWeightStorage_
        totalDoublesNeeded += alignSize(criticBiasCount);   // firstTargetCriticBiasStorage_
        totalDoublesNeeded += alignSize(criticWeightCount); // secondTargetCriticWeightStorage_
        totalDoublesNeeded += alignSize(criticBiasCount);   // secondTargetCriticBiasStorage_

        // Allocate
        void* rawMemory = nullptr;
#if defined(_WIN32)
        rawMemory = _aligned_malloc(totalDoublesNeeded * sizeof(double), 64);
#else
        if (posix_memalign(&rawMemory, 64, totalDoublesNeeded * sizeof(double)) != 0) {
            rawMemory = nullptr;
        }
#endif
        if (!rawMemory) return; // Allocation failed

        // Initialize to Zero (important for Momentum/Velocity/Gradients)
        std::memset(rawMemory, 0, totalDoublesNeeded * sizeof(double));

        alignedInternalMemory_.reset(rawMemory);
        auto* memoryCursor = static_cast<double*>(alignedInternalMemory_.get());

        auto bindSpan = [&](size_t size) -> std::span<double> {
            std::span<double> boundSpan(memoryCursor, size);
            memoryCursor += alignSize(size);
            return boundSpan;
        };

        // Bind Spans
        compressedObservationBuffer_ = bindSpan(compressedObservationSize);
        encoderScratchpadPool_ = bindSpan(totalEncoderScratchpadSize);

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
            auto hiddenScratch = encoderScratchpadPool_.subspan(scratchpadOffset, hiddenNeurons);
            scratchpadOffset += hiddenNeurons;

            auto latentScratch = encoderScratchpadPool_.subspan(scratchpadOffset, latentDim);
            scratchpadOffset += latentDim;

            auto reconstructionScratch = encoderScratchpadPool_.subspan(scratchpadOffset, inputDim);
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

        inferenceScratchpadA_ = bindSpan(maxScratchSize);
        inferenceScratchpadB_ = bindSpan(maxScratchSize);
        actionMeanScratchpad_ = bindSpan(actionSize);
        actionLogStdScratchpad_ = bindSpan(actionSize);
        recurrentGateMemoryBuffer_ = bindSpan(gruMemSize);
        blameScoresScratchpad_ = bindSpan(traceCapacity);

        firstCriticWeightMomentum_ = bindSpan(criticWeightCount);
        firstCriticWeightVelocity_ = bindSpan(criticWeightCount);
        firstCriticBiasMomentum_ = bindSpan(criticBiasCount);
        firstCriticBiasVelocity_ = bindSpan(criticBiasCount);

        secondCriticWeightMomentum_ = bindSpan(criticWeightCount);
        secondCriticWeightVelocity_ = bindSpan(criticWeightCount);
        secondCriticBiasMomentum_ = bindSpan(criticBiasCount);
        secondCriticBiasVelocity_ = bindSpan(criticBiasCount);

        criticWeightGradientScratchpad_ = bindSpan(criticWeightCount);
        criticBiasGradientScratchpad_ = bindSpan(criticBiasCount);
        criticInputGradientScratchpad_ = bindSpan(maxScratchSize);

        firstTargetCriticWeightStorage_ = bindSpan(criticWeightCount);
        firstTargetCriticBiasStorage_ = bindSpan(criticBiasCount);
        secondTargetCriticWeightStorage_ = bindSpan(criticWeightCount);
        secondTargetCriticBiasStorage_ = bindSpan(criticBiasCount);

        // Pre-allocate target critic weight/bias storage as C++-owned copies.
        // Initialise from the online critic weights so target starts identical.
        std::copy(firstCriticWeights.begin(), firstCriticWeights.end(), firstTargetCriticWeightStorage_.begin());
        std::copy(firstCriticBiases.begin(), firstCriticBiases.end(), firstTargetCriticBiasStorage_.begin());
        std::copy(secondCriticWeights.begin(), secondCriticWeights.end(), secondTargetCriticWeightStorage_.begin());
        std::copy(secondCriticBiases.begin(), secondCriticBiases.end(), secondTargetCriticBiasStorage_.begin());

        // Rebind the target critic networks to point at the C++-owned storage.
        firstTargetCriticNetwork_.rebindNetworkBuffers(
            std::span(firstTargetCriticWeightStorage_),
            std::span(firstTargetCriticBiasStorage_));
        secondTargetCriticNetwork_.rebindNetworkBuffers(
            std::span(secondTargetCriticWeightStorage_),
            std::span(secondTargetCriticBiasStorage_));

        // Store a non-owning view over the full JVM-owned critic buffer for persistence.
        this->criticWeightsSpan_ = std::span<const double>(gruGateWeights.data(),
             gruGateWeights.size() + gruGateBiases.size() + gruHiddenState.size()
             + firstCriticWeights.size() + firstCriticBiases.size()
             + secondCriticWeights.size() + secondCriticBiases.size());
    }

    void RecurrentSoftActorCriticSpartanModel::processTick() {
        logging::SpartanLogger::debug("[RSAC-INTERNAL] processTick() START");

        const auto* config = typedConfig();
        if (!config) {
             logging::SpartanLogger::error("[RSAC-INTERNAL] Config is null!");
             return;
        }

        const int actionSize = config->baseConfig.actionSize;
        const int hiddenSize = config->actorHiddenLayerNeuronCount;
        const int contextSize = static_cast<int>(contextBuffer_.size());
        const int encoderCount = config->nestedEncoderCount;

        logging::SpartanLogger::debug("[RSAC-INTERNAL] Phase A: Observation Build");

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

        logging::SpartanLogger::debug("[RSAC-INTERNAL] Phase B: GRU Forward");

        //
        // Phase B Pass the observation through the Gated Recurrent Unit
        //          to update the temporal hidden state.
        //
        recurrentLayer_.forwardPass(
            gruInputObservation,
            recurrentLayer_.getMutableHiddenState(),
            config,
            std::span(recurrentGateMemoryBuffer_));

        logging::SpartanLogger::debug("[RSAC-INTERNAL] Phase C: Policy Forward");

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

        //
        // Phase H (training only): Sync online critics -> target critics via Polyak averaging.
        //
        if (config->baseConfig.isTraining) {
            constexpr double targetCriticSmoothingCoefficient = 0.005;

            TensorOps::applyPolyakAveraging(
                std::span<const double>(firstCriticNetwork_.getNetworkWeights()),
                std::span(firstTargetCriticWeightStorage_),
                targetCriticSmoothingCoefficient);

            TensorOps::applyPolyakAveraging(
                std::span<const double>(firstCriticNetwork_.getNetworkBiases()),
                std::span(firstTargetCriticBiasStorage_),
                targetCriticSmoothingCoefficient);

            TensorOps::applyPolyakAveraging(
                std::span<const double>(secondCriticNetwork_.getNetworkWeights()),
                std::span(secondTargetCriticWeightStorage_),
                targetCriticSmoothingCoefficient);

            TensorOps::applyPolyakAveraging(
                std::span<const double>(secondCriticNetwork_.getNetworkBiases()),
                std::span(secondTargetCriticBiasStorage_),
                targetCriticSmoothingCoefficient);
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
        // The TD target for SAC uses separate target networks for stable bootstrap:
        //   y = r + gamma * (min(Q1_target, Q2_target)(s, a) - alpha * entropy)
        //
        // Each online critic is updated independently to minimize (Q_i(s, a) - y)^2.
        //
        const std::span<const double> hiddenStateView = recurrentLayer_.getHiddenState();
        const std::span<const double> actionView(actionOutputBuffer_.data(), actionSize);

        // Evaluate online critics for the current state-action pair.
        const double firstQValue = firstCriticNetwork_.computeQValue(
            hiddenStateView, actionView, config,
            std::span(inferenceScratchpadA_),
            std::span(inferenceScratchpadB_));

        const double secondQValue = secondCriticNetwork_.computeQValue(
            hiddenStateView, actionView, config,
            std::span(inferenceScratchpadA_),
            std::span(inferenceScratchpadB_));

        // Evaluate TARGET critics for the bootstrap (stable TD target).
        const double firstTargetQValue = firstTargetCriticNetwork_.computeQValue(
            hiddenStateView, actionView, config,
            std::span(inferenceScratchpadA_),
            std::span(inferenceScratchpadB_));

        const double secondTargetQValue = secondTargetCriticNetwork_.computeQValue(
            hiddenStateView, actionView, config,
            std::span(inferenceScratchpadA_),
            std::span(inferenceScratchpadB_));

        const double minimumTargetQValue = std::min(firstTargetQValue, secondTargetQValue);

        // Estimate entropy from the current standard deviations (already exp'd in processTick)
        double entropyEstimate = 0.0;
        for (int dimensionIndex = 0; dimensionIndex < actionSize; ++dimensionIndex) {
            if (const double standardDeviation = actionLogStdScratchpad_[dimensionIndex]; standardDeviation > 1e-8) {
                entropyEstimate += std::log(standardDeviation);
            }
        }

        // TD target: y = r + gamma * (Q_min_target - alpha * entropy)
        const double gamma = config->baseConfig.gamma;
        const double alpha = config->entropyTemperatureAlpha;
        const double temporalDifferenceTarget = rewardSignal
            + gamma * (minimumTargetQValue - alpha * entropyEstimate);

        ++criticTrainingStepCounter_;

        // Train Q1 critic (weights + biases)
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

            // Bias gradient for the output layer: dL/dB = dL/dY (the TD error itself).
            std::ranges::fill(criticBiasGradientScratchpad_, 0.0);
            criticBiasGradientScratchpad_[0] = temporalDifferenceErrorQ1;

            TensorOps::applyAdamUpdate(
                firstCriticNetwork_.getNetworkWeights(),
                std::span<const double>(criticWeightGradientScratchpad_),
                std::span(firstCriticWeightMomentum_),
                std::span(firstCriticWeightVelocity_),
                config->firstCriticLearningRate, 0.9, 0.999, 1e-8,
                criticTrainingStepCounter_);

            TensorOps::applyAdamUpdate(
                firstCriticNetwork_.getNetworkBiases(),
                std::span<const double>(criticBiasGradientScratchpad_),
                std::span(firstCriticBiasMomentum_),
                std::span(firstCriticBiasVelocity_),
                config->firstCriticLearningRate, 0.9, 0.999, 1e-8,
                criticTrainingStepCounter_);
        }

        // Train Q2 critic (weights + biases)
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

            // Bias gradient for the output layer.
            std::ranges::fill(criticBiasGradientScratchpad_, 0.0);
            criticBiasGradientScratchpad_[0] = temporalDifferenceErrorQ2;

            TensorOps::applyAdamUpdate(
                secondCriticNetwork_.getNetworkWeights(),
                std::span<const double>(criticWeightGradientScratchpad_),
                std::span(secondCriticWeightMomentum_),
                std::span(secondCriticWeightVelocity_),
                config->secondCriticLearningRate, 0.9, 0.999, 1e-8,
                criticTrainingStepCounter_);

            TensorOps::applyAdamUpdate(
                secondCriticNetwork_.getNetworkBiases(),
                std::span<const double>(criticBiasGradientScratchpad_),
                std::span(secondCriticBiasMomentum_),
                std::span(secondCriticBiasVelocity_),
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

    std::span<const double> RecurrentSoftActorCriticSpartanModel::getCriticWeights() const noexcept {
        return criticWeightsSpan_;
    }

}

