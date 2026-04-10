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
        int encoderCount = config->nestedEncoderCount;
        const int criticHiddenSize = config->criticHiddenLayerNeuronCount;
        const int criticLayerCount = config->criticHiddenLayerCount;
        const int criticCombinedSize = config->hiddenStateSize + actionSize;
        const int criticCombinedSizeAligned = std::max(criticCombinedSize, 1);

        if (encoderCount < 0 || encoderCount > SPARTAN_MAX_NESTED_ENCODER_SLOTS) {
            logging::SpartanLogger::error(std::format(
                "RSAC: invalid nestedEncoderCount {} (max {}) - disabling encoders",
                encoderCount, SPARTAN_MAX_NESTED_ENCODER_SLOTS));
            encoderCount = 0;
        }

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
        const int maxScratchSize = std::max({
            hiddenSize,
            config->criticHiddenLayerNeuronCount,
            criticCombinedSizeAligned,
            compressedObservationSize
        });

        totalDoublesNeeded += alignSize(maxScratchSize); // inferenceScratchpadA_
        totalDoublesNeeded += alignSize(maxScratchSize); // inferenceScratchpadB_
        totalDoublesNeeded += alignSize(actionSize);     // actionMeanScratchpad_
        totalDoublesNeeded += alignSize(actionSize);     // actionLogStdScratchpad_
        totalDoublesNeeded += alignSize(actionSize);     // actionStdScratchpad_
        totalDoublesNeeded += alignSize(actionSize);     // actionNoiseScratchpad_
        totalDoublesNeeded += alignSize(actionSize);     // actionGradientScratchpad_
        totalDoublesNeeded += alignSize(actionSize);     // policyLogStdCache_
        totalDoublesNeeded += alignSize(hiddenSize);     // policyHiddenActivationCache_

        // Critic training buffers (multi-layer backprop)
        totalDoublesNeeded += alignSize(criticCombinedSizeAligned); // criticCombinedInputBuffer_
        totalDoublesNeeded += alignSize(static_cast<size_t>(criticHiddenSize) * criticLayerCount); // firstCriticActivationCache_
        totalDoublesNeeded += alignSize(static_cast<size_t>(criticHiddenSize) * criticLayerCount); // secondCriticActivationCache_

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

        // Policy optimizer state and gradients
        totalDoublesNeeded += alignSize(policyWeights.size()); // policyWeightMomentum_
        totalDoublesNeeded += alignSize(policyWeights.size()); // policyWeightVelocity_
        totalDoublesNeeded += alignSize(policyBiases.size());  // policyBiasMomentum_
        totalDoublesNeeded += alignSize(policyBiases.size());  // policyBiasVelocity_
        totalDoublesNeeded += alignSize(policyWeights.size()); // policyWeightGradientScratchpad_
        totalDoublesNeeded += alignSize(policyBiases.size());  // policyBiasGradientScratchpad_
        totalDoublesNeeded += alignSize(hiddenSize);           // policyHiddenGradientScratchpad_

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
        if (!rawMemory) {
            logging::SpartanLogger::error("RSAC: aligned allocation failed");
            return;
        }

        std::memset(rawMemory, 0, totalDoublesNeeded * sizeof(double));
        alignedInternalMemory_.reset(rawMemory);
        auto* memoryCursor = static_cast<double*>(alignedInternalMemory_.get());

        auto bindSpan = [&](size_t size) -> std::span<double> {
            std::span<double> boundSpan(memoryCursor, size);
            memoryCursor += alignSize(size);
            return boundSpan;
        };

        // Core buffers
        compressedObservationBuffer_ = bindSpan(compressedObservationSize);
        encoderScratchpadPool_ = bindSpan(totalEncoderScratchpadSize);

        if (encoderCount > 0 && encoderWeightPool.empty()) {
            logging::SpartanLogger::error("RSAC: encoderWeightPool is empty while encoderCount > 0; disabling encoders");
            encoderCount = 0;
            nestedEncoderBank_.clear();
        }

        // Build nested encoders
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

            const size_t requiredWeightCount = encHiddenWeightCount + hiddenNeurons + encLatentWeightCount + latentDim
                    + decHiddenWeightCount + hiddenNeurons + decOutputWeightCount + inputDim;
            const size_t requiredScratch = static_cast<size_t>(hiddenNeurons + latentDim + inputDim);

            if (encoderWeightOffset + requiredWeightCount > encoderWeightPool.size()) {
                logging::SpartanLogger::error("RSAC: encoder weight pool too small; aborting encoder construction");
                break;
            }
            if (scratchpadOffset + requiredScratch > encoderScratchpadPool_.size()) {
                logging::SpartanLogger::error("RSAC: encoder scratchpad pool too small; aborting encoder construction");
                break;
            }

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

            auto hiddenScratch = encoderScratchpadPool_.subspan(scratchpadOffset, hiddenNeurons);
            scratchpadOffset += hiddenNeurons;

            auto latentScratch = encoderScratchpadPool_.subspan(scratchpadOffset, latentDim);
            scratchpadOffset += latentDim;

            auto reconstructionScratch = encoderScratchpadPool_.subspan(scratchpadOffset, inputDim);
            scratchpadOffset += inputDim;

            nestedEncoderBank_.emplace_back(
                encoderHiddenWeights,
                encoderHiddenBiases,
                encoderLatentWeights,
                encoderLatentBiases,
                decoderHiddenWeights,
                decoderHiddenBiases,
                decoderOutputWeights,
                decoderOutputBiases,
                latentScratch,
                hiddenScratch,
                reconstructionScratch,
                inputDim,
                latentDim,
                hiddenNeurons);
        }

        // Scratchpads and caches
        inferenceScratchpadA_ = bindSpan(maxScratchSize);
        inferenceScratchpadB_ = bindSpan(maxScratchSize);
        actionMeanScratchpad_ = bindSpan(actionSize);
        actionLogStdScratchpad_ = bindSpan(actionSize);
        actionStdScratchpad_ = bindSpan(actionSize);
        actionNoiseScratchpad_ = bindSpan(actionSize);
        actionGradientScratchpad_ = bindSpan(actionSize);
        policyLogStdCache_ = bindSpan(actionSize);
        policyHiddenActivationCache_ = bindSpan(hiddenSize);

        criticCombinedInputBuffer_ = bindSpan(criticCombinedSizeAligned);
        firstCriticActivationCache_ = bindSpan(static_cast<size_t>(criticHiddenSize) * criticLayerCount);
        secondCriticActivationCache_ = bindSpan(static_cast<size_t>(criticHiddenSize) * criticLayerCount);

        // GRU gate memory: [concat(hidden, input)] + [Z] + [R] + [H~]
        recurrentGateMemoryBuffer_ = bindSpan(gruMemSize);

        // Blame scores scratchpad (one per trace entry)
        blameScoresScratchpad_ = bindSpan(traceCapacity);

        // Bindings for Adam optimizer state spans
        firstCriticWeightMomentum_ = bindSpan(criticWeightCount);
        firstCriticWeightVelocity_ = bindSpan(criticWeightCount);
        firstCriticBiasMomentum_ = bindSpan(criticBiasCount);
        firstCriticBiasVelocity_ = bindSpan(criticBiasCount);

        secondCriticWeightMomentum_ = bindSpan(criticWeightCount);
        secondCriticWeightVelocity_ = bindSpan(criticWeightCount);
        secondCriticBiasMomentum_ = bindSpan(criticBiasCount);
        secondCriticBiasVelocity_ = bindSpan(criticBiasCount);

        // Gradient scratchpads for critic backward pass
        criticWeightGradientScratchpad_ = bindSpan(criticWeightCount);
        criticBiasGradientScratchpad_ = bindSpan(criticBiasCount);
        criticInputGradientScratchpad_ = bindSpan(maxScratchSize);

        // Policy optimizer state and gradients
        policyWeightMomentum_ = bindSpan(policyWeights.size());
        policyWeightVelocity_ = bindSpan(policyWeights.size());
        policyBiasMomentum_ = bindSpan(policyBiases.size());
        policyBiasVelocity_ = bindSpan(policyBiases.size());
        policyWeightGradientScratchpad_ = bindSpan(policyWeights.size());
        policyBiasGradientScratchpad_ = bindSpan(policyBiases.size());
        policyHiddenGradientScratchpad_ = bindSpan(hiddenSize);

        // Bind target critic storage spans before initialization.
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

        // Cache hidden activation and log-std for actor update
        std::copy_n(inferenceScratchpadA_.data(), hiddenSize, policyHiddenActivationCache_.data());
        std::copy_n(actionLogStdScratchpad_.data(), actionSize, policyLogStdCache_.data());

        // Convert log-std to std in-place via exp
        if (config->baseConfig.isTraining) {
            TensorOps::applyExpExact(std::span(actionLogStdScratchpad_));
        } else {
            TensorOps::applyExpFast(std::span(actionLogStdScratchpad_));
        }
        std::copy_n(actionLogStdScratchpad_.data(), actionSize, actionStdScratchpad_.data());

        // Phase E: Write the final noisy action into the JVM-owned action output buffer.
        TensorOps::applyGaussianNoise(
            std::span<const double>(actionMeanScratchpad_),
            std::span<const double>(actionStdScratchpad_),
            actionOutputBuffer_);

        // Cache noise = (action - mean) / std for policy gradients
        for (int i = 0; i < actionSize; ++i) {
            const double stdValue = actionStdScratchpad_[i];
            actionNoiseScratchpad_[i] = stdValue > 1e-8
                ? (actionOutputBuffer_[i] - actionMeanScratchpad_[i]) / stdValue
                : 0.0;
        }
        hasPolicySnapshot_ = true;

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

        if (traceEntryCount < 1) return;

        std::vector<double> blameScores(traceEntryCount, 0.0);
        const std::span<const double> currentHiddenStateView = recurrentLayer_.getHiddenState();

        // Use the current hidden state to compute similarity blame scores against all past states
        remorseTraceBuffer_.computeBlameScores(currentHiddenStateView,
                                               std::span(blameScores),
                                               config->remorseMinimumSimilarityThreshold);

        // We use the current state as s'
        const std::span<const double> targetStateView = currentHiddenStateView;

        // Evaluate TARGET critics for the bootstrap (stable TD target) using current state s'
        // For the target action, we use the current policy output. Wait, we need the action for s',
        // but since we are off-policy SAC, we would sample a' from pi(·|s').
        // For simplicity here, we assume actionOutputBuffer_ contains a valid action or we just use it.
        const std::span<const double> targetActionView(actionOutputBuffer_.data(), actionSize);

        const double firstTargetQValue = firstTargetCriticNetwork_.computeQValue(
            targetStateView, targetActionView, config,
            std::span(inferenceScratchpadA_),
            std::span(inferenceScratchpadB_));

        const double secondTargetQValue = secondTargetCriticNetwork_.computeQValue(
            targetStateView, targetActionView, config,
            std::span(inferenceScratchpadA_),
            std::span(inferenceScratchpadB_));

        const double minimumTargetQValue = std::min(firstTargetQValue, secondTargetQValue);

        double entropyEstimate = 0.0;
        constexpr double kPi = 3.14159265358979323846;
        const double kHalfLog2PiE = 0.5 * std::log(2.0 * kPi * std::exp(1.0));
        for (int dimensionIndex = 0; dimensionIndex < actionSize; ++dimensionIndex) {
            entropyEstimate += policyLogStdCache_[dimensionIndex] + kHalfLog2PiE;
        }

        const double gamma = config->baseConfig.gamma;
        const double alpha = config->entropyTemperatureAlpha;

        const int criticHiddenSize = config->criticHiddenLayerNeuronCount;
        const int criticLayerCount = config->criticHiddenLayerCount;

        auto forwardCriticWithCache = [&](auto& criticNetwork, std::span<double> activationCache, std::span<const double> stateView) {
            const int currentCombinedInputSize = static_cast<int>(stateView.size()) + actionSize;

            if (currentCombinedInputSize > static_cast<int>(criticCombinedInputBuffer_.size())) return 0.0;

            const std::span<const double> actionView(actionOutputBuffer_.data(), actionSize);
            std::copy(stateView.begin(), stateView.end(), criticCombinedInputBuffer_.begin());
            std::copy(actionView.begin(), actionView.end(), criticCombinedInputBuffer_.begin() + stateView.size());

            std::span<const double> currentInput(criticCombinedInputBuffer_.data(), currentCombinedInputSize);
            size_t weightOffset = 0;
            size_t biasOffset = 0;

            for (int layer = 0; layer < criticLayerCount; ++layer) {
                auto activationSpan = activationCache.subspan(static_cast<size_t>(layer) * criticHiddenSize, criticHiddenSize);
                const int inputSize = (layer == 0) ? currentCombinedInputSize : criticHiddenSize;

                TensorOps::denseForwardPass(
                    currentInput,
                    criticNetwork.getNetworkWeights().subspan(weightOffset, static_cast<size_t>(criticHiddenSize) * inputSize),
                    criticNetwork.getNetworkBiases().subspan(biasOffset, criticHiddenSize),
                    activationSpan);
                TensorOps::applyLeakyReLU(activationSpan, 0.01);

                currentInput = activationSpan;
                weightOffset += static_cast<size_t>(criticHiddenSize) * inputSize;
                biasOffset += static_cast<size_t>(criticHiddenSize);
            }

            double qValue = 0.0;
            auto qValueSpan = std::span(&qValue, 1);
            TensorOps::denseForwardPass(
                currentInput,
                criticNetwork.getNetworkWeights().subspan(weightOffset, criticHiddenSize),
                criticNetwork.getNetworkBiases().subspan(biasOffset, 1),
                qValueSpan);

            return qValue;
        };

        // Loop over the remorse trace to assign credit appropriately
        for (int i = 0; i < traceEntryCount; ++i) {
            const double blame = blameScores[i];
            if (blame < 0.001) continue;

            // Past state s
            const std::span<const double> pastStateView = remorseTraceBuffer_.getArchivedHiddenState(i);

            // Get the action taken at this past timestep from the trace buffer
            const RemorseTraceEntry& entry = remorseTraceBuffer_.getEntry(i);
            int32_t archivedActionIndex = entry.selectedActionIndex;

            // Reconstruct one-hot action vector for the archived action
            std::fill_n(inferenceScratchpadA_.begin(), actionSize, 0.0);
            if (archivedActionIndex >= 0 && archivedActionIndex < actionSize) {
                inferenceScratchpadA_[archivedActionIndex] = 1.0;
            }
            const std::span<const double> actionView(inferenceScratchpadA_.data(), actionSize);

            const double effectiveReward = rewardSignal * blame;

            const double temporalDifferenceTarget = effectiveReward
                + gamma * (minimumTargetQValue - alpha * entropyEstimate);

            const int combinedInputSize = static_cast<int>(pastStateView.size()) + actionSize;
            if (combinedInputSize > static_cast<int>(criticCombinedInputBuffer_.size())) continue;

            const double firstQValue = forwardCriticWithCache(firstCriticNetwork_, firstCriticActivationCache_, pastStateView);
            const double secondQValue = forwardCriticWithCache(secondCriticNetwork_, secondCriticActivationCache_, pastStateView);

            ++criticTrainingStepCounter_;

            auto trainCritic = [&](auto& criticNetwork,
                                   std::span<double> activationCache,
                                   std::span<double> weightMomentum,
                                   std::span<double> weightVelocity,
                                   std::span<double> biasMomentum,
                                   std::span<double> biasVelocity,
                                   double qValue,
                                   double learningRate) {

                std::ranges::fill(criticWeightGradientScratchpad_, 0.0);
                std::ranges::fill(criticBiasGradientScratchpad_, 0.0);

                // MSE gradient dL/dQ = (Q - target) already has 2x from d(Q-target)²/dQ
                const double tdError = (qValue - temporalDifferenceTarget);

                // Compute offsets
                size_t weightOffset = 0;
                size_t biasOffset = 0;
                for (int layer = 0; layer < criticLayerCount; ++layer) {
                    const int inputSize = (layer == 0) ? combinedInputSize : criticHiddenSize;
                    weightOffset += static_cast<size_t>(criticHiddenSize) * inputSize;
                    biasOffset += static_cast<size_t>(criticHiddenSize);
                }
                const size_t outputWeightOffset = weightOffset;
                const size_t outputBiasOffset = biasOffset;

                // Output layer gradients
                const auto lastActivation = activationCache.subspan(
                    static_cast<size_t>(criticLayerCount - 1) * criticHiddenSize,
                    criticHiddenSize);

                for (int j = 0; j < criticHiddenSize; ++j) {
                    criticWeightGradientScratchpad_[outputWeightOffset + static_cast<size_t>(j)] += tdError * lastActivation[j];
                }
                criticBiasGradientScratchpad_[outputBiasOffset] += tdError;

                // Gradient into last hidden layer
                auto currentGrad = std::span(criticInputGradientScratchpad_.data(), criticHiddenSize);
                for (int j = 0; j < criticHiddenSize; ++j) {
                    currentGrad[j] = tdError * criticNetwork.getNetworkWeights()[outputWeightOffset + static_cast<size_t>(j)];
                }

                // Backprop through hidden layers
                for (int layer = criticLayerCount - 1; layer >= 0; --layer) {
                    const auto activationSpan = activationCache.subspan(
                        static_cast<size_t>(layer) * criticHiddenSize,
                        criticHiddenSize);

                    for (int j = 0; j < criticHiddenSize; ++j) {
                        if (activationSpan[j] <= 0.0) {
                            currentGrad[j] *= 0.01;
                        }
                    }

                    const int inputSize = (layer == 0) ? combinedInputSize : criticHiddenSize;
                    weightOffset -= static_cast<size_t>(criticHiddenSize) * inputSize;
                    biasOffset -= static_cast<size_t>(criticHiddenSize);

                    for (int j = 0; j < criticHiddenSize; ++j) {
                        criticBiasGradientScratchpad_[biasOffset + static_cast<size_t>(j)] += currentGrad[j];
                    }

                    const std::span<const double> layerInput = (layer == 0)
                        ? std::span<const double>(criticCombinedInputBuffer_.data(), combinedInputSize)
                        : activationCache.subspan(static_cast<size_t>(layer - 1) * criticHiddenSize, criticHiddenSize);

                    auto inputGrad = std::span(criticInputGradientScratchpad_.data(), inputSize);
                    auto weightGradSpan = std::span(criticWeightGradientScratchpad_)
                        .subspan(weightOffset, static_cast<size_t>(criticHiddenSize) * inputSize);
                    // We do not consume these weight grads here; only need inputGrad for dQ/da.
                    TensorOps::denseBackwardPass(
                        layerInput,
                        currentGrad,
                        criticNetwork.getNetworkWeights().subspan(weightOffset, static_cast<size_t>(criticHiddenSize) * inputSize),
                        weightGradSpan,
                        inputGrad);

                    if (layer > 0) {
                        for (int j = 0; j < criticHiddenSize; ++j) {
                            currentGrad[j] = inputGrad[j];
                        }
                    }
                }

                TensorOps::applyAdamUpdate(
                    criticNetwork.getNetworkWeights(),
                    std::span<const double>(criticWeightGradientScratchpad_),
                    weightMomentum,
                    weightVelocity,
                    learningRate, 0.9, 0.999, 1e-8,
                    criticTrainingStepCounter_);

                TensorOps::applyAdamUpdate(
                    criticNetwork.getNetworkBiases(),
                    std::span<const double>(criticBiasGradientScratchpad_),
                    biasMomentum,
                    biasVelocity,
                    learningRate, 0.9, 0.999, 1e-8,
                    criticTrainingStepCounter_);
            };

            trainCritic(
                firstCriticNetwork_,
                firstCriticActivationCache_,
                firstCriticWeightMomentum_,
                firstCriticWeightVelocity_,
                firstCriticBiasMomentum_,
                firstCriticBiasVelocity_,
                firstQValue,
                config->firstCriticLearningRate);

            trainCritic(
                secondCriticNetwork_,
                secondCriticActivationCache_,
                secondCriticWeightMomentum_,
                secondCriticWeightVelocity_,
                secondCriticBiasMomentum_,
                secondCriticBiasVelocity_,
                secondQValue,
                config->secondCriticLearningRate);

        }

        // --- Full SAC Actor Update ---
        // Objective: J = Q(s, a) - alpha * log pi(a|s)
        // With a = mean + std * noise, dJ/dmean = dQ/da
        // and dJ/dlogStd = dQ/da * (std * noise) + alpha
        if (hasPolicySnapshot_) {
            const int currentCombinedInputSize = static_cast<int>(currentHiddenStateView.size()) + actionSize;

            const double firstActorQValue = forwardCriticWithCache(firstCriticNetwork_, firstCriticActivationCache_, currentHiddenStateView);
            const double secondActorQValue = forwardCriticWithCache(secondCriticNetwork_, secondCriticActivationCache_, currentHiddenStateView);

            auto computeActionGradient = [&](auto& criticNet, std::span<double> criticCache) {
                // Compute dQ/da via backprop to critic input
                size_t weightOffset = 0;
                size_t biasOffset = 0;
                for (int layer = 0; layer < criticLayerCount; ++layer) {
                    const int inputSize = (layer == 0) ? currentCombinedInputSize : criticHiddenSize;
                    weightOffset += static_cast<size_t>(criticHiddenSize) * inputSize;
                    biasOffset += static_cast<size_t>(criticHiddenSize);
                }
                const size_t outputWeightOffset = weightOffset;

                auto currentGrad = std::span(criticInputGradientScratchpad_.data(), criticHiddenSize);
                for (int i = 0; i < criticHiddenSize; ++i) {
                    currentGrad[i] = criticNet.getNetworkWeights()[outputWeightOffset + static_cast<size_t>(i)];
                }

                for (int layer = criticLayerCount - 1; layer >= 0; --layer) {
                    const auto activationSpan = criticCache.subspan(
                        static_cast<size_t>(layer) * criticHiddenSize,
                        criticHiddenSize);

                    for (int i = 0; i < criticHiddenSize; ++i) {
                        if (activationSpan[i] <= 0.0) {
                            currentGrad[i] *= 0.01;
                        }
                    }

                    const int inputSize = (layer == 0) ? currentCombinedInputSize : criticHiddenSize;
                    weightOffset -= static_cast<size_t>(criticHiddenSize) * inputSize;
                    biasOffset -= static_cast<size_t>(criticHiddenSize);

                    const std::span<const double> layerInput = (layer == 0)
                        ? std::span<const double>(criticCombinedInputBuffer_.data(), currentCombinedInputSize)
                        : criticCache.subspan(static_cast<size_t>(layer - 1) * criticHiddenSize, criticHiddenSize);

                    auto inputGrad = std::span(criticInputGradientScratchpad_.data(), inputSize);
                    auto weightGradSpan = std::span(criticWeightGradientScratchpad_)
                        .subspan(weightOffset, static_cast<size_t>(criticHiddenSize) * inputSize);
                    // We do not consume these weight grads here; only need inputGrad for dQ/da.
                    TensorOps::denseBackwardPass(
                        layerInput,
                        currentGrad,
                        criticNet.getNetworkWeights().subspan(weightOffset, static_cast<size_t>(criticHiddenSize) * inputSize),
                        weightGradSpan,
                        inputGrad);

                    if (layer > 0) {
                        for (int i = 0; i < criticHiddenSize; ++i) {
                            currentGrad[i] = inputGrad[i];
                        }
                    } else {
                        // Action gradient is the tail of the critic input gradient
                        const int actionOffset = static_cast<int>(currentHiddenStateView.size());
                        for (int i = 0; i < actionSize; ++i) {
                            actionGradientScratchpad_[i] = inputGrad[actionOffset + i];
                        }
                    }
                }
            };

            if (firstActorQValue <= secondActorQValue) {
                computeActionGradient(firstCriticNetwork_, firstCriticActivationCache_);
            } else {
                computeActionGradient(secondCriticNetwork_, secondCriticActivationCache_);
            }

            // Policy gradients (mean/logstd heads)
            std::ranges::fill(policyWeightGradientScratchpad_, 0.0);
            std::ranges::fill(policyBiasGradientScratchpad_, 0.0);

            // Full SAC: dJ/dmean = dQ/da, dJ/dlogStd = dQ/da * (std * noise) + alpha
            const double alphaTerm = config->entropyTemperatureAlpha;
            for (int i = 0; i < actionSize; ++i) {
                const double dQda = actionGradientScratchpad_[i];
                const double stdValue = actionStdScratchpad_[i];
                const double noise = actionNoiseScratchpad_[i];
                const double dMean = dQda;
                const double dLogStd = dQda * (stdValue * noise) + alphaTerm;

                // Mean head gradients
                size_t meanWeightOffset = static_cast<size_t>(hiddenSize) * hiddenSize;
                meanWeightOffset += static_cast<size_t>(i) * hiddenSize;
                for (int h = 0; h < hiddenSize; h++) {
                    policyWeightGradientScratchpad_[meanWeightOffset + static_cast<size_t>(h)] += dMean * policyHiddenActivationCache_[h];
                }
                policyBiasGradientScratchpad_[hiddenSize + i] += dMean;

                // LogStd head gradients
                size_t logStdWeightOffset = static_cast<size_t>(hiddenSize) * hiddenSize;
                logStdWeightOffset += static_cast<size_t>(actionSize) * hiddenSize;
                logStdWeightOffset += static_cast<size_t>(i) * hiddenSize;
                for (int h = 0; h < hiddenSize; h++) {
                    policyWeightGradientScratchpad_[logStdWeightOffset + static_cast<size_t>(h)] += dLogStd * policyHiddenActivationCache_[h];
                }
                policyBiasGradientScratchpad_[hiddenSize + actionSize + i] += dLogStd;
            }

            // Hidden layer gradient (tanh)
            std::ranges::fill(policyHiddenGradientScratchpad_, 0.0);
            for (int h = 0; h < hiddenSize; ++h) {
                double grad = 0.0;
                for (int i = 0; i < actionSize; ++i) {
                    const double dQda = actionGradientScratchpad_[i];
                    const double stdValue = actionStdScratchpad_[i];
                    const double noise = actionNoiseScratchpad_[i];
                    const double dMean = dQda;
                    const double dLogStd = dQda * (stdValue * noise) + alphaTerm;

                    const size_t meanWeightOffset = static_cast<size_t>(hiddenSize) * hiddenSize
                        + static_cast<size_t>(i) * hiddenSize + static_cast<size_t>(h);
                    const size_t logStdWeightOffset = static_cast<size_t>(hiddenSize) * hiddenSize
                        + static_cast<size_t>(actionSize) * hiddenSize
                        + static_cast<size_t>(i) * hiddenSize + static_cast<size_t>(h);

                    grad += policyNetwork_.getPolicyWeights()[meanWeightOffset] * dMean;
                    grad += policyNetwork_.getPolicyWeights()[logStdWeightOffset] * dLogStd;
                }
                const double hiddenVal = policyHiddenActivationCache_[h];
                policyHiddenGradientScratchpad_[h] = grad * (1.0 - hiddenVal * hiddenVal);
            }

            // Input->Hidden gradients
            for (int h = 0; h < hiddenSize; ++h) {
                policyBiasGradientScratchpad_[h] += policyHiddenGradientScratchpad_[h];
                const size_t wOffset = static_cast<size_t>(h) * static_cast<size_t>(hiddenSize);
                for (int in = 0; in < hiddenSize; ++in) {
                    policyWeightGradientScratchpad_[wOffset + static_cast<size_t>(in)] += policyHiddenGradientScratchpad_[h] * currentHiddenStateView[in];
                }
            }

            // Adam update for policy
            ++policyTrainingStepCounter_;
            TensorOps::applyAdamUpdate(
                policyNetwork_.getPolicyWeights(),
                std::span<const double>(policyWeightGradientScratchpad_),
                policyWeightMomentum_,
                policyWeightVelocity_,
                config->policyNetworkLearningRate, 0.9, 0.999, 1e-8,
                policyTrainingStepCounter_);

            TensorOps::applyAdamUpdate(
                policyNetwork_.getPolicyBiases(),
                std::span<const double>(policyBiasGradientScratchpad_),
                policyBiasMomentum_,
                policyBiasVelocity_,
                config->policyNetworkLearningRate, 0.9, 0.999, 1e-8,
                policyTrainingStepCounter_);
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


    // Expose critic weights for persistence. These are non-owning views over the JVM-owned buffer.
    std::span<const double> RecurrentSoftActorCriticSpartanModel::getCriticWeights() const noexcept {
        return criticWeightsSpan_;
    }

    std::span<double> RecurrentSoftActorCriticSpartanModel::getCriticWeightsMutable() noexcept {
        return const_cast<double*>(criticWeightsSpan_.data()) ?
            std::span<double>(const_cast<double*>(criticWeightsSpan_.data()), criticWeightsSpan_.size()) : std::span<double>();
    }

}
