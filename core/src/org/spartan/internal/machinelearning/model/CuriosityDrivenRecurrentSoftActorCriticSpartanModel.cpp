//
// Created by Alepando on 12/3/2026.
//

#include "CuriosityDrivenRecurrentSoftActorCriticSpartanModel.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <format>
#include <random>

#ifdef _WIN32
#endif

#include "internal/math/tensor/SpartanTensorMath.h"
#include "internal/logging/SpartanLogger.h"

namespace org::spartan::internal::machinelearning {

    using math::tensor::TensorOps;

    namespace {
        void sanitizeFinite(std::span<double> values) {
            for (double& value : values) {
                if (!std::isfinite(value)) {
                    value = 0.0;
                }
            }
        }

        bool isAllZero(std::span<const double> values) {
            for (const double value : values) {
                if (std::abs(value) > 1e-12) {
                    return false;
                }
            }
            return true;
        }

        void fillSmallRandom(std::span<double> values, std::mt19937& rng) {
            std::uniform_real_distribution<double> dist(-0.01, 0.01);
            for (double& value : values) {
                value = dist(rng);
            }
        }
    }

    CuriosityDrivenRecurrentSoftActorCriticSpartanModel::CuriosityDrivenRecurrentSoftActorCriticSpartanModel(
            const uint64_t agentIdentifier,
            void* opaqueHyperparameterConfig,
            const std::span<double> modelWeights,
            const std::span<const double> contextBuffer,
            const std::span<double> actionOutputBuffer,
            const std::span<double> recurrentSoftActorCriticCriticWeights,
            const std::span<double> forwardDynamicsWeights,
            const std::span<double> forwardDynamicsBiases,
            std::unique_ptr<RecurrentSoftActorCriticSpartanModel> internalRecurrentSoftActorCriticModel)
        : SpartanAgent(agentIdentifier,
                       opaqueHyperparameterConfig,
                       modelWeights,
                       contextBuffer,
                       actionOutputBuffer),
          internalRecurrentSoftActorCriticModel_(std::move(internalRecurrentSoftActorCriticModel)),
          forwardDynamicsWeights_(forwardDynamicsWeights),
          forwardDynamicsBiases_(forwardDynamicsBiases),
          alignedScratchpadMemory_(nullptr, [](void* ptr) {
              if (ptr) {
                #if defined(_WIN32)
                  _aligned_free(ptr);
                #else
                  free(ptr);
                #endif
              }
          }) {

        const auto* javaConfig = static_cast<const CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig*>(
            opaqueHyperparameterConfig);

        // DEBUG: Log config read from Java
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-CONSTRUCT] Config loaded from Java (addr={})",
            reinterpret_cast<uintptr_t>(javaConfig)));

        localConfig_ = *javaConfig;

        const auto* config = typedConfig();
        const auto stateSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.stateSize);
        const auto actionSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.actionSize);
        const auto hiddenSize = static_cast<size_t>(config->forwardDynamicsHiddenLayerDimensionSize);

        // DEBUG: Log extracted dimensions
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-CONSTRUCT] Dimensions: stateSize={}, actionSize={}, hiddenSize={}",
            stateSize, actionSize, hiddenSize));

        const size_t totalWeightCount = (stateSize + actionSize) * hiddenSize + hiddenSize * stateSize;
        const size_t totalBiasCount = hiddenSize + stateSize;

        // DEBUG: Log weight and bias counts
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-CONSTRUCT] Weight layout: totalWeightCount={}, totalBiasCount={}",
            totalWeightCount, totalBiasCount));

        // Helper to ensure SIMD alignment (64 bytes = 8 doubles)
        auto alignSize = [](size_t size) -> size_t {
            return (size + 7) & ~static_cast<size_t>(7);
        };

        // Calculate total memory with mandatory alignment padding for SIMD safety
        size_t totalDoublesNeeded = 0;
        totalDoublesNeeded += alignSize(stateSize);                 // previousStateBuffer_
        totalDoublesNeeded += alignSize(actionSize);                // previousActionBuffer_
        totalDoublesNeeded += alignSize(stateSize);                 // predictedNextStateBuffer_
        totalDoublesNeeded += alignSize(stateSize + actionSize);    // forwardNetworkInputBuffer_
        totalDoublesNeeded += alignSize(hiddenSize);               // forwardNetworkHiddenBuffer_
        totalDoublesNeeded += alignSize(stateSize);                 // forwardNetworkOutputGradient_
        totalDoublesNeeded += alignSize(hiddenSize);               // forwardDynamicsHiddenActivationGradients_
        totalDoublesNeeded += alignSize(stateSize + actionSize);    // forwardNetworkInputGradientDummy_
        totalDoublesNeeded += alignSize(totalWeightCount);         // forwardDynamicsWeightGradients_
        totalDoublesNeeded += alignSize(totalBiasCount);           // forwardDynamicsBiasGradients_
        totalDoublesNeeded += alignSize(totalWeightCount);         // forwardWeightsFirstMoment_
        totalDoublesNeeded += alignSize(totalWeightCount);         // forwardWeightsSecondMoment_
        totalDoublesNeeded += alignSize(totalBiasCount);           // forwardBiasesFirstMoment_
        totalDoublesNeeded += alignSize(totalBiasCount);           // forwardBiasesSecondMoment_
        totalDoublesNeeded += alignSize(totalWeightCount);         // internalForwardDynamicsWeights
        totalDoublesNeeded += alignSize(totalBiasCount);           // internalForwardDynamicsBiases

        // DEBUG: Log total memory allocation
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-CONSTRUCT] Total aligned memory: {} doubles ({} MB)",
            totalDoublesNeeded, (totalDoublesNeeded * sizeof(double)) / (1024.0 * 1024.0)));

        void* rawMemory = nullptr;
        #if defined(_WIN32)
        rawMemory = _aligned_malloc(totalDoublesNeeded * sizeof(double), 64);
        #else
        if (posix_memalign(&rawMemory, 64, totalDoublesNeeded * sizeof(double)) != 0) {
            rawMemory = nullptr;
        }
        #endif

        alignedScratchpadMemory_.reset(rawMemory);

        // DEBUG: Log aligned memory allocation
        if (rawMemory) {
            logging::SpartanLogger::debug(std::format(
                "[CURIOSITY-CONSTRUCT] Aligned memory allocated: {} (64-byte aligned)",
                reinterpret_cast<uintptr_t>(rawMemory)));
        } else {
            logging::SpartanLogger::error("[CURIOSITY-CONSTRUCT] FAILED to allocate aligned memory!");
        }

        if (rawMemory) {

            auto* data = static_cast<double*>(rawMemory);
            for(size_t i = 0; i < totalDoublesNeeded; ++i) {
                data[i] = (static_cast<double>(rand()) / (RAND_MAX)) * 0.02 - 0.01;
            }
        }

        auto* memoryCursor = static_cast<double*>(alignedScratchpadMemory_.get());

        // Aligned binding logic to prevent EXCEPTION_ACCESS_VIOLATION in SIMD ops
        auto bindSpan = [&](size_t size) -> std::span<double> {
            std::span<double> boundSpan(memoryCursor, size);
            logging::SpartanLogger::debug(std::format(
                "[CURIOSITY-CONSTRUCT] Bound span: size={}, addr={}",
                size, reinterpret_cast<uintptr_t>(memoryCursor)));
            memoryCursor += alignSize(size);
            return boundSpan;
        };

        previousStateBuffer_ = bindSpan(stateSize);
        previousActionBuffer_ = bindSpan(actionSize);
        predictedNextStateBuffer_ = bindSpan(stateSize);
        forwardNetworkInputBuffer_ = bindSpan(stateSize + actionSize);
        forwardNetworkHiddenBuffer_ = bindSpan(hiddenSize);
        forwardNetworkOutputGradient_ = bindSpan(stateSize);
        forwardDynamicsHiddenActivationGradients_ = bindSpan(hiddenSize);
        forwardNetworkInputGradientDummy_ = bindSpan(stateSize + actionSize);
        forwardDynamicsWeightGradients_ = bindSpan(totalWeightCount);
        forwardDynamicsBiasGradients_ = bindSpan(totalBiasCount);
        forwardWeightsFirstMoment_ = bindSpan(totalWeightCount);
        forwardWeightsSecondMoment_ = bindSpan(totalWeightCount);
        forwardBiasesFirstMoment_ = bindSpan(totalBiasCount);
        forwardBiasesSecondMoment_ = bindSpan(totalBiasCount);

        std::span<double> internalWeights = bindSpan(totalWeightCount);
        std::span<double> internalBiases = bindSpan(totalBiasCount);

        if (!forwardDynamicsWeights.empty() && forwardDynamicsWeights.size() <= totalWeightCount) {
            logging::SpartanLogger::debug(std::format(
                "[CURIOSITY-CONSTRUCT] Copying weights: size={}", forwardDynamicsWeights.size()));
            std::copy(forwardDynamicsWeights.begin(), forwardDynamicsWeights.end(), internalWeights.begin());
        } else {
            logging::SpartanLogger::warn(std::format(
                "[CURIOSITY-CONSTRUCT] Warning: forwardDynamicsWeights empty or oversized (size={}, expected<={})",
                forwardDynamicsWeights.empty() ? 0 : forwardDynamicsWeights.size(), totalWeightCount));
        }

        if (!forwardDynamicsBiases.empty() && forwardDynamicsBiases.size() <= totalBiasCount) {
            logging::SpartanLogger::debug(std::format(
                "[CURIOSITY-CONSTRUCT] Copying biases: size={}", forwardDynamicsBiases.size()));
            std::copy(forwardDynamicsBiases.begin(), forwardDynamicsBiases.end(), internalBiases.begin());
        } else {
            logging::SpartanLogger::warn(std::format(
                "[CURIOSITY-CONSTRUCT] Warning: forwardDynamicsBiases empty or oversized (size={}, expected<={})",
                forwardDynamicsBiases.empty() ? 0 : forwardDynamicsBiases.size(), totalBiasCount));
        }

        if (isAllZero(internalWeights) && isAllZero(internalBiases)) {
            std::mt19937 rng(static_cast<uint32_t>(agentIdentifier));
            fillSmallRandom(internalWeights, rng);
            fillSmallRandom(internalBiases, rng);
            logging::SpartanLogger::debug("[CURIOSITY-CONSTRUCT] Forward dynamics params initialized (was all-zero)");
        }

        forwardDynamicsWeights_ = internalWeights;
        forwardDynamicsBiases_ = internalBiases;
        sanitizeFinite(forwardDynamicsWeights_);
        sanitizeFinite(forwardDynamicsBiases_);

        logging::SpartanLogger::info(std::format(
            "[CURIOSITY-CONSTRUCT] Constructor complete for agent {}", agentIdentifier));
    }

    void CuriosityDrivenRecurrentSoftActorCriticSpartanModel::processTick() {
        logging::SpartanLogger::debug("[CURIOSITY-TICK] processTick() START");

        if (internalRecurrentSoftActorCriticModel_) {
             internalRecurrentSoftActorCriticModel_->processTick();
        } else {
             logging::SpartanLogger::error("[CURIOSITY-TICK] FATAL: Internal RSAC model is null!");
             return;
        }

        if (!hasValidPreviousTick_) {
            hasValidPreviousTick_ = true;
            logging::SpartanLogger::debug("[CURIOSITY-TICK] First tick (seed), copying context/action buffers");
            std::ranges::copy(contextBuffer_, previousStateBuffer_.begin());
            std::ranges::copy(actionOutputBuffer_, previousActionBuffer_.begin());
            sanitizeFinite(previousStateBuffer_);
            sanitizeFinite(previousActionBuffer_);
            logging::SpartanLogger::debug("[CURIOSITY-TICK] processTick() END (First Tick)");
            return;
        }

        sanitizeFinite(previousStateBuffer_);
        sanitizeFinite(previousActionBuffer_);
        logging::SpartanLogger::debug("[CURIOSITY-TICK] Running Forward Dynamics Inference");
        runForwardDynamicsNetworkInference();


        double mse = 0.0;
        const auto* config = typedConfig();
        const size_t stateSize = contextBuffer_.size();

        logging::SpartanLogger::debug(std::format("[CURIOSITY-TICK] Calculating MSE. State size: {}", stateSize));

        for (size_t i = 0; i < stateSize; ++i) {

            double diff = contextBuffer_[i] - predictedNextStateBuffer_[i];
            forwardNetworkOutputGradient_[i] = diff;
            mse += diff * diff;
        }
        mse /= static_cast<double>(stateSize);

        lastIntrinsicReward_ = std::clamp(
            mse * config->intrinsicRewardScale,
            config->intrinsicRewardClampingMinimum,
            config->intrinsicRewardClampingMaximum
        );

        logging::SpartanLogger::debug(std::format("[CURIOSITY-TICK] Intrinsic Reward: {}", lastIntrinsicReward_));


        if (config->recurrentSoftActorCriticConfig.baseConfig.isTraining) {
            logging::SpartanLogger::debug("[CURIOSITY-TICK] Training Forward Dynamics Network");
            trainForwardDynamicsNetwork(mse);
        }


        std::ranges::copy(contextBuffer_, previousStateBuffer_.begin());
        std::ranges::copy(actionOutputBuffer_, previousActionBuffer_.begin());
        sanitizeFinite(previousStateBuffer_);
        sanitizeFinite(previousActionBuffer_);
        logging::SpartanLogger::debug("[CURIOSITY-TICK] processTick() END");
    }

    void CuriosityDrivenRecurrentSoftActorCriticSpartanModel::applyReward(const double extrinsicReward) {
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-REWARD] applyReward() called: extrinsicReward={}, lastIntrinsicReward_={}",
            extrinsicReward, lastIntrinsicReward_));

        const double totalReward = extrinsicReward + lastIntrinsicReward_;

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-REWARD] Total reward (extrinsic + intrinsic): {}", totalReward));

        internalRecurrentSoftActorCriticModel_->applyReward(totalReward);

        logging::SpartanLogger::debug("[CURIOSITY-REWARD] applyReward() END");
    }

    void CuriosityDrivenRecurrentSoftActorCriticSpartanModel::decayExploration() {
        logging::SpartanLogger::debug("[CURIOSITY-DECAY] decayExploration() called");
        internalRecurrentSoftActorCriticModel_->decayExploration();
        logging::SpartanLogger::debug("[CURIOSITY-DECAY] decayExploration() END");
    }

    std::span<const double> CuriosityDrivenRecurrentSoftActorCriticSpartanModel::getCriticWeights() const noexcept {
        const auto innerCriticWeights = internalRecurrentSoftActorCriticModel_->getCriticWeights();
        const size_t totalSize = innerCriticWeights.size() + forwardDynamicsWeights_.size() + forwardDynamicsBiases_.size();
        fullCriticSaveBuffer_.resize(totalSize);

        size_t offset = 0;
        std::copy_n(innerCriticWeights.begin(), innerCriticWeights.size(), fullCriticSaveBuffer_.begin() + offset);
        offset += innerCriticWeights.size();

        std::copy_n(forwardDynamicsWeights_.begin(), forwardDynamicsWeights_.size(), fullCriticSaveBuffer_.begin() + offset);
        offset += forwardDynamicsWeights_.size();

        std::copy_n(forwardDynamicsBiases_.begin(), forwardDynamicsBiases_.size(), fullCriticSaveBuffer_.begin() + offset);

        return fullCriticSaveBuffer_;
    }

    void CuriosityDrivenRecurrentSoftActorCriticSpartanModel::runForwardDynamicsNetworkInference() {
        logging::SpartanLogger::debug("[CURIOSITY-INFERENCE] runForwardDynamicsNetworkInference() START");

        const auto* config = typedConfig();
        const size_t stateSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.stateSize);
        const size_t actionSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.actionSize);
        const size_t hiddenSize = static_cast<size_t>(config->forwardDynamicsHiddenLayerDimensionSize);

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-INFERENCE] Config: stateSize={}, actionSize={}, hiddenSize={}",
            stateSize, actionSize, hiddenSize));

        // Verify buffer sizes
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-INFERENCE] Buffer verification - previousState.size()={}, previousAction.size()={}, forwardNetworkInput.size()={}",
            previousStateBuffer_.size(), previousActionBuffer_.size(), forwardNetworkInputBuffer_.size()));

        const size_t expectedWeightCount = (stateSize + actionSize) * hiddenSize + (hiddenSize * stateSize);
        const size_t expectedBiasCount = hiddenSize + stateSize;
        if (forwardDynamicsWeights_.size() < expectedWeightCount || forwardDynamicsBiases_.size() < expectedBiasCount) {
            logging::SpartanLogger::error(std::format(
                "[CURIOSITY-INFERENCE] Forward dynamics buffer mismatch: weights={} (expected>= {}), biases={} (expected>= {})",
                forwardDynamicsWeights_.size(), expectedWeightCount,
                forwardDynamicsBiases_.size(), expectedBiasCount));
            return;
        }

        std::copy_n(previousStateBuffer_.begin(), stateSize, forwardNetworkInputBuffer_.begin());
        std::copy_n(previousActionBuffer_.begin(), actionSize, forwardNetworkInputBuffer_.begin() + stateSize);

        logging::SpartanLogger::debug("[CURIOSITY-INFERENCE] Input buffer concatenated (state + action)");

        const size_t inputToHiddenWeightCount = (stateSize + actionSize) * hiddenSize;
        const auto inputToHiddenWeights = forwardDynamicsWeights_.subspan(0, inputToHiddenWeightCount);
        const auto hiddenBiases = forwardDynamicsBiases_.subspan(0, hiddenSize);

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-INFERENCE] Input->Hidden: weights.size()={}, biases.size()={}",
            inputToHiddenWeights.size(), hiddenBiases.size()));

        TensorOps::denseForwardPass(
            forwardNetworkInputBuffer_, inputToHiddenWeights, hiddenBiases, forwardNetworkHiddenBuffer_
        );

        logging::SpartanLogger::debug("[CURIOSITY-INFERENCE] Input->Hidden forward pass complete");

        const size_t hiddenToOutputWeightCount = hiddenSize * stateSize;
        const auto hiddenToOutputWeights = forwardDynamicsWeights_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount);
        const auto outputBiases = forwardDynamicsBiases_.subspan(hiddenSize, stateSize);

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-INFERENCE] Hidden->Output: weights.size()={}, biases.size()={}",
            hiddenToOutputWeights.size(), outputBiases.size()));

        TensorOps::denseForwardPass(
            forwardNetworkHiddenBuffer_, hiddenToOutputWeights, outputBiases, predictedNextStateBuffer_
        );
        sanitizeFinite(predictedNextStateBuffer_);

        logging::SpartanLogger::debug("[CURIOSITY-INFERENCE] Hidden->Output forward pass complete");

        // Debug: Log first few predicted values
        if (stateSize > 0) {
            logging::SpartanLogger::debug(std::format(
                "[CURIOSITY-INFERENCE] First predicted state values: [{}, {}, {}]",
                predictedNextStateBuffer_[0],
                stateSize > 1 ? predictedNextStateBuffer_[1] : 0.0,
                stateSize > 2 ? predictedNextStateBuffer_[2] : 0.0));
        }

        logging::SpartanLogger::debug("[CURIOSITY-INFERENCE] runForwardDynamicsNetworkInference() END");
    }

    void CuriosityDrivenRecurrentSoftActorCriticSpartanModel::trainForwardDynamicsNetwork(const double predictionError) {
        logging::SpartanLogger::debug(std::format("[CURIOSITY-TRAIN] trainForwardDynamicsNetwork() START, predictionError={}", predictionError));

        if (!hasValidPreviousTick_) {
            logging::SpartanLogger::warn("[CURIOSITY-TRAIN] WARNING: hasValidPreviousTick_=false, aborting training!");
            return;
        }

        adamTimeStep_++;
        logging::SpartanLogger::debug(std::format("[CURIOSITY-TRAIN] adamTimeStep_ incremented to: {}", adamTimeStep_));

        const auto* config = typedConfig();
        const size_t stateSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.stateSize);
        const size_t actionSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.actionSize);
        const size_t hiddenSize = static_cast<size_t>(config->forwardDynamicsHiddenLayerDimensionSize);

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-TRAIN] Dimensions: stateSize={}, actionSize={}, hiddenSize={}",
            stateSize, actionSize, hiddenSize));

        const size_t inputToHiddenWeightCount = (stateSize + actionSize) * hiddenSize;
        const size_t hiddenToOutputWeightCount = hiddenSize * stateSize;
        const size_t expectedWeightCount = inputToHiddenWeightCount + hiddenToOutputWeightCount;
        const size_t expectedBiasCount = hiddenSize + stateSize;
        if (forwardDynamicsWeights_.size() < expectedWeightCount || forwardDynamicsBiases_.size() < expectedBiasCount) {
            logging::SpartanLogger::error(std::format(
                "[CURIOSITY-TRAIN] Forward dynamics buffer mismatch: weights={} (expected>= {}), biases={} (expected>= {})",
                forwardDynamicsWeights_.size(), expectedWeightCount,
                forwardDynamicsBiases_.size(), expectedBiasCount));
            return;
        }

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-TRAIN] Weight counts: inputToHidden={}, hiddenToOutput={}",
            inputToHiddenWeightCount, hiddenToOutputWeightCount));

        // Guard: Verify buffer sizes before operations
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-TRAIN] Guard check - forwardDynamicsWeightGradients_.size()={}, expected={}",
            forwardDynamicsWeightGradients_.size(), inputToHiddenWeightCount + hiddenToOutputWeightCount));
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-TRAIN] Guard check - forwardNetworkHiddenBuffer_.size()={}, expected={}",
            forwardNetworkHiddenBuffer_.size(), hiddenSize));
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-TRAIN] Guard check - forwardNetworkInputBuffer_.size()={}, expected={}",
            forwardNetworkInputBuffer_.size(), stateSize + actionSize));

        std::ranges::fill(forwardDynamicsWeightGradients_, 0.0);
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Weight gradients zeroed");

        std::ranges::fill(forwardDynamicsHiddenActivationGradients_, 0.0);
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Hidden activation gradients zeroed");

        std::ranges::fill(forwardNetworkInputGradientDummy_, 0.0);
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Input gradient dummy zeroed");

        const auto hiddenToOutputWeights = forwardDynamicsWeights_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount);
        auto hiddenToOutputWeightGradients = forwardDynamicsWeightGradients_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount);

        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Calling denseBackwardPass for Hidden->Output layer...");
        TensorOps::denseBackwardPass(
            forwardNetworkHiddenBuffer_,
            forwardNetworkOutputGradient_,
            hiddenToOutputWeights,
            hiddenToOutputWeightGradients,
            forwardDynamicsHiddenActivationGradients_
        );
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Hidden->Output backward pass complete");

        const auto inputToHiddenWeights = forwardDynamicsWeights_.subspan(0, inputToHiddenWeightCount);
        auto inputToHiddenWeightGradients = forwardDynamicsWeightGradients_.subspan(0, inputToHiddenWeightCount);

        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Calling denseBackwardPass for Input->Hidden layer...");
        TensorOps::denseBackwardPass(
            forwardNetworkInputBuffer_,
            forwardDynamicsHiddenActivationGradients_,
            inputToHiddenWeights,
            inputToHiddenWeightGradients,
            forwardNetworkInputGradientDummy_
        );
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Input->Hidden backward pass complete");

        // Update bias gradients
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Copying bias gradients...");
        std::copy_n(forwardDynamicsHiddenActivationGradients_.begin(), hiddenSize, forwardDynamicsBiasGradients_.begin());
        std::copy_n(forwardNetworkOutputGradient_.begin(), stateSize, forwardDynamicsBiasGradients_.begin() + hiddenSize);
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Bias gradients copied");

        const double learningRate = config->forwardDynamicsLearningRate;
        const double beta1 = 0.9;
        const double beta2 = 0.999;
        const double epsilon = 1e-8;

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-TRAIN] Adam hyperparams: lr={}, beta1={}, beta2={}, epsilon={}, t={}",
            learningRate, beta1, beta2, epsilon, adamTimeStep_));

        // Adam for Weights - Input->Hidden
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Applying Adam update to Input->Hidden weights...");
        TensorOps::applyAdamUpdate(
            forwardDynamicsWeights_.subspan(0, inputToHiddenWeightCount),
            inputToHiddenWeightGradients,
            forwardWeightsFirstMoment_.subspan(0, inputToHiddenWeightCount),
            forwardWeightsSecondMoment_.subspan(0, inputToHiddenWeightCount),
            learningRate, beta1, beta2, epsilon, adamTimeStep_
        );
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Input->Hidden weights updated");

        // Adam for Weights - Hidden->Output
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Applying Adam update to Hidden->Output weights...");
        TensorOps::applyAdamUpdate(
            forwardDynamicsWeights_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount),
            hiddenToOutputWeightGradients,
            forwardWeightsFirstMoment_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount),
            forwardWeightsSecondMoment_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount),
            learningRate, beta1, beta2, epsilon, adamTimeStep_
        );
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Hidden->Output weights updated");

        // Adam for Biases - Hidden
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Applying Adam update to hidden biases...");
        TensorOps::applyAdamUpdate(
            forwardDynamicsBiases_.subspan(0, hiddenSize),
            forwardDynamicsBiasGradients_.subspan(0, hiddenSize),
            forwardBiasesFirstMoment_.subspan(0, hiddenSize),
            forwardBiasesSecondMoment_.subspan(0, hiddenSize),
            learningRate, beta1, beta2, epsilon, adamTimeStep_
        );
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Hidden biases updated");

        // Adam for Biases - Output
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Applying Adam update to output biases...");
        TensorOps::applyAdamUpdate(
            forwardDynamicsBiases_.subspan(hiddenSize, stateSize),
            forwardDynamicsBiasGradients_.subspan(hiddenSize, stateSize),
            forwardBiasesFirstMoment_.subspan(hiddenSize, stateSize),
            forwardBiasesSecondMoment_.subspan(hiddenSize, stateSize),
            learningRate, beta1, beta2, epsilon, adamTimeStep_
        );
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Output biases updated");

        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] trainForwardDynamicsNetwork() END");
    }
}

