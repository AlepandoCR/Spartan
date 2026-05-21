//
// Created by Alepando on 12/3/2026.
//

#include "CuriosityDrivenRecurrentSoftActorCriticSpartanModel.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <cmath>
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

        bool isAllZero(const std::span<const double> values) {
            return std::ranges::all_of(values, [](const double value) {
                return std::abs(value) <= 1e-12;
            });
        }

        void fillSmallRandom(std::span<double> values, std::mt19937& rng) {
            std::uniform_real_distribution dist(-0.01, 0.01);
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
            const std::span<double> curiosityWeights,
            const std::span<double> curiosityBiases,
            std::unique_ptr<RecurrentSoftActorCriticSpartanModel> internalRecurrentSoftActorCriticModel)
        : SpartanAgent(agentIdentifier,
                       opaqueHyperparameterConfig,
                       modelWeights,
                       contextBuffer,
                       actionOutputBuffer),
          internalRecurrentSoftActorCriticModel_(std::move(internalRecurrentSoftActorCriticModel)),
          criticWeightsSpan_(recurrentSoftActorCriticCriticWeights.data(), recurrentSoftActorCriticCriticWeights.size()),
          alignedScratchpadMemory_(nullptr, [](void* ptr) {
              if (!ptr) {
                  return;
              }
#if defined(_WIN32)
              _aligned_free(ptr);
#else
              free(ptr);
#endif
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

        // Ensure all buffer sizes are valid
        if (stateSize <= 0) {
            logging::SpartanLogger::error(std::format(
                "[CURIOSITY-CONSTRUCT] Invalid stateSize {} (must be > 0)", stateSize));
            throw std::invalid_argument("stateSize must be > 0");
        }
        if (actionSize <= 0) {
            logging::SpartanLogger::error(std::format(
                "[CURIOSITY-CONSTRUCT] Invalid actionSize {} (must be > 0)", actionSize));
            throw std::invalid_argument("actionSize must be > 0");
        }
        if (hiddenSize <= 0) {
            logging::SpartanLogger::error(std::format(
                "[CURIOSITY-CONSTRUCT] Invalid forwardDynamicsHiddenLayerDimensionSize {} (must be > 0)", hiddenSize));
            throw std::invalid_argument("hiddenSize must be > 0");
        }

        // Log extracted dimensions
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-CONSTRUCT] Dimensions: stateSize={}, actionSize={}, hiddenSize={}",
            stateSize, actionSize, hiddenSize));

        // Forward dynamics counts
        const size_t forwardWeightCount = (stateSize + actionSize) * hiddenSize + hiddenSize * stateSize;
        const size_t forwardBiasCount = hiddenSize + stateSize;

        // Inverse dynamics counts (input: state + next_state)
        const size_t inverseInputSize = stateSize * 2;
        const size_t inverseHiddenSize = localConfig_.inverseDynamicsHiddenLayerDimensionSize > 0
                                             ? localConfig_.inverseDynamicsHiddenLayerDimensionSize : hiddenSize;
        const size_t inverseWeightCount = inverseInputSize * inverseHiddenSize + inverseHiddenSize * actionSize;
        const size_t inverseBiasCount = inverseHiddenSize + actionSize;

        const size_t totalWeightCount = forwardWeightCount + inverseWeightCount;
        const size_t totalBiasCount = forwardBiasCount + inverseBiasCount;

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
        // Forward dynamics gradients/moments and internal params
        totalDoublesNeeded += alignSize(forwardWeightCount);         // forwardDynamicsWeightGradients_
        totalDoublesNeeded += alignSize(forwardBiasCount);           // forwardDynamicsBiasGradients_
        totalDoublesNeeded += alignSize(forwardWeightCount);         // forwardWeightsFirstMoment_
        totalDoublesNeeded += alignSize(forwardWeightCount);         // forwardWeightsSecondMoment_
        totalDoublesNeeded += alignSize(forwardBiasCount);           // forwardBiasesFirstMoment_
        totalDoublesNeeded += alignSize(forwardBiasCount);           // forwardBiasesSecondMoment_
        // Inverse network primary buffers
        totalDoublesNeeded += alignSize(stateSize * 2);              // inverseNetworkInputBuffer_
        totalDoublesNeeded += alignSize(inverseHiddenSize);          // inverseNetworkHiddenBuffer_
        totalDoublesNeeded += alignSize(actionSize);                 // predictedActionBuffer_
        totalDoublesNeeded += alignSize(inverseHiddenSize);          // inverseDynamicsHiddenActivationGradients_
        totalDoublesNeeded += alignSize(stateSize * 2);              // inverseNetworkInputGradientDummy_
        totalDoublesNeeded += alignSize(forwardWeightCount);         // internalForwardDynamicsWeights
        totalDoublesNeeded += alignSize(forwardBiasCount);           // internalForwardDynamicsBiases

        // Inverse dynamics gradients/moments and internal params
        totalDoublesNeeded += alignSize(inverseWeightCount);         // inverseDynamicsWeightGradients_
        totalDoublesNeeded += alignSize(inverseBiasCount);           // inverseDynamicsBiasGradients_
        totalDoublesNeeded += alignSize(inverseWeightCount);         // inverseWeightsFirstMoment_
        totalDoublesNeeded += alignSize(inverseWeightCount);         // inverseWeightsSecondMoment_
        totalDoublesNeeded += alignSize(inverseBiasCount);           // inverseBiasesFirstMoment_
        totalDoublesNeeded += alignSize(inverseBiasCount);           // inverseBiasesSecondMoment_
        totalDoublesNeeded += alignSize(inverseWeightCount);         // internalInverseDynamicsWeights
        totalDoublesNeeded += alignSize(inverseBiasCount);           // internalInverseDynamicsBiases

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

        if (rawMemory == nullptr) {
            logging::SpartanLogger::error("[CURIOSITY-CONSTRUCT] FAILED to allocate aligned memory! aborting constructor");
            throw std::bad_alloc();
        }

        alignedScratchpadMemory_.reset(rawMemory);

        // DEBUG: Log aligned memory allocation

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-CONSTRUCT] Aligned memory allocated: {} (64-byte aligned)",
            reinterpret_cast<uintptr_t>(rawMemory)));

        if (rawMemory) {

            auto* data = static_cast<double*>(rawMemory);
            for(size_t i = 0; i < totalDoublesNeeded; ++i) {
                data[i] = (static_cast<double>(rand()) / (RAND_MAX)) * 0.02 - 0.01;
            }
        }

        auto* memoryCursor = static_cast<double*>(alignedScratchpadMemory_.get());

        // Aligned binding logic to prevent EXCEPTION_ACCESS_VIOLATION in SIMD ops
        auto bindSpan = [&](size_t size) -> std::span<double> {
            const std::span boundSpan(memoryCursor, size);
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

        // Inverse network primary buffers
        inverseNetworkInputBuffer_ = bindSpan(stateSize * 2);
        inverseNetworkHiddenBuffer_ = bindSpan(inverseHiddenSize);
        predictedActionBuffer_ = bindSpan(actionSize);
        // Inverse network auxiliary gradients
        inverseDynamicsHiddenActivationGradients_ = bindSpan(inverseHiddenSize);
        inverseNetworkInputGradientDummy_ = bindSpan(stateSize * 2);
        // Forward dynamics allocations
        forwardDynamicsWeightGradients_ = bindSpan(forwardWeightCount);
        forwardDynamicsBiasGradients_ = bindSpan(forwardBiasCount);
        forwardWeightsFirstMoment_ = bindSpan(forwardWeightCount);
        forwardWeightsSecondMoment_ = bindSpan(forwardWeightCount);
        forwardBiasesFirstMoment_ = bindSpan(forwardBiasCount);
        forwardBiasesSecondMoment_ = bindSpan(forwardBiasCount);

        std::span<double> internalForwardWeights = bindSpan(forwardWeightCount);
        std::span<double> internalForwardBiases = bindSpan(forwardBiasCount);

        // Inverse dynamics allocations
        inverseDynamicsWeightGradients_ = bindSpan(inverseWeightCount);
        inverseDynamicsBiasGradients_ = bindSpan(inverseBiasCount);
        inverseWeightsFirstMoment_ = bindSpan(inverseWeightCount);
        inverseWeightsSecondMoment_ = bindSpan(inverseWeightCount);
        inverseBiasesFirstMoment_ = bindSpan(inverseBiasCount);
        inverseBiasesSecondMoment_ = bindSpan(inverseBiasCount);

        std::span<double> internalInverseWeights = bindSpan(inverseWeightCount);
        std::span<double> internalInverseBiases = bindSpan(inverseBiasCount);

        // Copy provided curiosity weights/biases into internal forward/inverse storages
        if (!curiosityWeights.empty() && curiosityWeights.size() >= forwardWeightCount + inverseWeightCount) {
            logging::SpartanLogger::debug(std::format(
                "[CURIOSITY-CONSTRUCT] Copying curiosity weights: provided={}, expected={}", curiosityWeights.size(), forwardWeightCount + inverseWeightCount));
            // forward then inverse
            std::copy_n(curiosityWeights.begin(), forwardWeightCount, internalForwardWeights.begin());
            std::copy_n(curiosityWeights.begin() + static_cast<std::ptrdiff_t>(forwardWeightCount), inverseWeightCount, internalInverseWeights.begin());
        } else {
            logging::SpartanLogger::warn(std::format(
                "[CURIOSITY-CONSTRUCT] Warning: curiosityWeights empty or undersized (provided={}, expected={})",
                curiosityWeights.empty() ? 0 : curiosityWeights.size(), forwardWeightCount + inverseWeightCount));
        }

        if (!curiosityBiases.empty() && curiosityBiases.size() >= forwardBiasCount + inverseBiasCount) {
            logging::SpartanLogger::debug(std::format(
                "[CURIOSITY-CONSTRUCT] Copying curiosity biases: provided={}, expected={}", curiosityBiases.size(), forwardBiasCount + inverseBiasCount));
            std::copy_n(curiosityBiases.begin(), forwardBiasCount, internalForwardBiases.begin());
            std::copy_n(curiosityBiases.begin() + static_cast<std::ptrdiff_t>(forwardBiasCount), inverseBiasCount, internalInverseBiases.begin());
        } else {
            logging::SpartanLogger::warn(std::format(
                "[CURIOSITY-CONSTRUCT] Warning: curiosityBiases empty or undersized (provided={}, expected={})",
                curiosityBiases.empty() ? 0 : curiosityBiases.size(), forwardBiasCount + inverseBiasCount));
        }

        // If params are zero, initialize with small random values
        if (isAllZero(internalForwardWeights) && isAllZero(internalForwardBiases)) {
            std::mt19937 rng(static_cast<uint32_t>(agentIdentifier));
            fillSmallRandom(internalForwardWeights, rng);
            fillSmallRandom(internalForwardBiases, rng);
            logging::SpartanLogger::debug("[CURIOSITY-CONSTRUCT] Forward dynamics params initialized (was all-zero)");
        }

        if (isAllZero(internalInverseWeights) && isAllZero(internalInverseBiases)) {
            std::mt19937 rng(static_cast<uint32_t>(agentIdentifier) ^ 0xfeedu);
            fillSmallRandom(internalInverseWeights, rng);
            fillSmallRandom(internalInverseBiases, rng);
            logging::SpartanLogger::debug("[CURIOSITY-CONSTRUCT] Inverse dynamics params initialized (was all-zero)");
        }

        forwardDynamicsWeights_ = internalForwardWeights;
        forwardDynamicsBiases_ = internalForwardBiases;
        sanitizeFinite(forwardDynamicsWeights_);
        sanitizeFinite(forwardDynamicsBiases_);

        // Bind inverse internal params
        inverseDynamicsWeights_ = internalInverseWeights;
        inverseDynamicsBiases_ = internalInverseBiases;
        sanitizeFinite(inverseDynamicsWeights_);
        sanitizeFinite(inverseDynamicsBiases_);

        logging::SpartanLogger::info(std::format(
            "[CURIOSITY-CONSTRUCT] Constructor complete for agent {}", agentIdentifier));
    }

    void CuriosityDrivenRecurrentSoftActorCriticSpartanModel::processTick() {
        logging::SpartanLogger::debug("[CURIOSITY-TICK] processTick() START");

        // Defensive check: ensure buffers are properly bound
        if (contextBuffer_.empty() || actionOutputBuffer_.empty()) {
            logging::SpartanLogger::warn(
                std::format("[CURIOSITY] WARNING: contextBuffer size={}, actionBuffer size={} - skipping processTick",
                contextBuffer_.size(), actionOutputBuffer_.size()));
            return;
        }

        if (internalRecurrentSoftActorCriticModel_) {
             internalRecurrentSoftActorCriticModel_->processTick();
        } else {
             logging::SpartanLogger::error("[CURIOSITY-TICK] FATAL: Internal RSAC model is null!");
             return;
        }

        if (!hasValidPreviousTick_) {
            hasValidPreviousTick_ = true;
            logging::SpartanLogger::debug("[CURIOSITY-TICK] First tick (seed), copying context/action buffers");
            std::copy_n(contextBuffer_.data(), std::min(contextBuffer_.size(), previousStateBuffer_.size()), previousStateBuffer_.data());
            std::copy_n(actionOutputBuffer_.data(), std::min(actionOutputBuffer_.size(), previousActionBuffer_.size()), previousActionBuffer_.data());
            sanitizeFinite(previousStateBuffer_);
            sanitizeFinite(previousActionBuffer_);
            logging::SpartanLogger::debug("[CURIOSITY-TICK] processTick() END (First Tick)");
            return;
        }

        sanitizeFinite(previousStateBuffer_);
        sanitizeFinite(previousActionBuffer_);
        logging::SpartanLogger::debug("[CURIOSITY-TICK] Running Forward Dynamics Inference");
        runForwardDynamicsNetworkInference();

        // Also run inverse dynamics inference to predict the action taken between previousState -> current state
        logging::SpartanLogger::debug("[CURIOSITY-TICK] Running Inverse Dynamics Inference");
        runInverseDynamicsNetworkInference();

        double mse = 0.0;
        double inverseMse = 0.0;
        const auto* config = typedConfig();
        const size_t stateSize = contextBuffer_.size();
        const size_t actionSize = previousActionBuffer_.size();
        const double mseScale = stateSize > 0 ? (2.0 / static_cast<double>(stateSize)) : 0.0;

        logging::SpartanLogger::debug(std::format("[CURIOSITY-TICK] Calculating Forward MSE. State size: {}", stateSize));

        for (size_t i = 0; i < stateSize; ++i) {
            const double diff = contextBuffer_[i] - predictedNextStateBuffer_[i];
            // Exact MSE gradient: 2*(pred-target)/N
            forwardNetworkOutputGradient_[i] = diff * mseScale;
            mse += diff * diff;
        }
        mse /= static_cast<double>(stateSize);

        // Inverse MSE between predicted action and recorded previous action
        for (size_t i = 0; i < actionSize; ++i) {
            const double idiff = predictedActionBuffer_[i] - previousActionBuffer_[i];
            inverseMse += idiff * idiff;
        }
        inverseMse = actionSize > 0 ? inverseMse / static_cast<double>(actionSize) : 0.0;

        lastIntrinsicReward_ = std::clamp(
            mse * config->intrinsicRewardScale,
            config->intrinsicRewardClampingMinimum,
            config->intrinsicRewardClampingMaximum
        );

        logging::SpartanLogger::debug(std::format("[CURIOSITY-TICK] Intrinsic Reward (forward mse): {}", lastIntrinsicReward_));
        logging::SpartanLogger::debug(std::format("[CURIOSITY-TICK] Inverse MSE: {}", inverseMse));

        if (config->recurrentSoftActorCriticConfig.baseConfig.isTraining) {
            logging::SpartanLogger::debug("[CURIOSITY-TICK] Training Forward + Inverse Dynamics Networks");
            trainForwardDynamicsNetwork(mse);
            trainInverseDynamicsNetwork(inverseMse);
        }


        std::copy_n(contextBuffer_.data(), std::min(contextBuffer_.size(), previousStateBuffer_.size()), previousStateBuffer_.data());
        std::copy_n(actionOutputBuffer_.data(), std::min(actionOutputBuffer_.size(), previousActionBuffer_.size()), previousActionBuffer_.data());
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

        // Append forward dynamics params
        std::copy_n(forwardDynamicsWeights_.begin(), forwardDynamicsWeights_.size(), fullCriticSaveBuffer_.begin() + offset);
        offset += forwardDynamicsWeights_.size();

        std::copy_n(forwardDynamicsBiases_.begin(), forwardDynamicsBiases_.size(), fullCriticSaveBuffer_.begin() + offset);
        offset += forwardDynamicsBiases_.size();

        // Append inverse dynamics params
        std::copy_n(inverseDynamicsWeights_.begin(), inverseDynamicsWeights_.size(), fullCriticSaveBuffer_.begin() + offset);
        offset += inverseDynamicsWeights_.size();

        std::copy_n(inverseDynamicsBiases_.begin(), inverseDynamicsBiases_.size(), fullCriticSaveBuffer_.begin() + offset);

        return fullCriticSaveBuffer_;
    }

    void CuriosityDrivenRecurrentSoftActorCriticSpartanModel::runForwardDynamicsNetworkInference() {
        logging::SpartanLogger::debug("[CURIOSITY-INFERENCE] runForwardDynamicsNetworkInference() START");

        const auto* config = typedConfig();
        const auto stateSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.stateSize);
        const auto actionSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.actionSize);
        const auto hiddenSize = static_cast<size_t>(config->forwardDynamicsHiddenLayerDimensionSize);

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-INFERENCE] Config: stateSize={}, actionSize={}, hiddenSize={}",
            stateSize, actionSize, hiddenSize));

        // Verify buffer sizes
        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-INFERENCE] Buffer verification - previousState.size()={}, previousAction.size()={}, forwardNetworkInput.size()={}",
            previousStateBuffer_.size(), previousActionBuffer_.size(), forwardNetworkInputBuffer_.size()));

        const size_t expectedWeightCount = (stateSize + actionSize) * hiddenSize + (hiddenSize * stateSize);
        if (const size_t expectedBiasCount = hiddenSize + stateSize; forwardDynamicsWeights_.size() < expectedWeightCount || forwardDynamicsBiases_.size() < expectedBiasCount) {
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


    void CuriosityDrivenRecurrentSoftActorCriticSpartanModel::runInverseDynamicsNetworkInference() {
        logging::SpartanLogger::debug("[CURIOSITY-INFERENCE] runInverseDynamicsNetworkInverse() START");

        const auto* config = typedConfig();
        const auto stateSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.stateSize);
        const auto actionSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.actionSize);
        const auto invHiddenSize = static_cast<size_t>(config->inverseDynamicsHiddenLayerDimensionSize > 0
            ? config->inverseDynamicsHiddenLayerDimensionSize : config->forwardDynamicsHiddenLayerDimensionSize);

        const size_t inverseInputSize = stateSize * 2;
        const size_t inputToHiddenWeightCount = inverseInputSize * invHiddenSize;
        const size_t hiddenToOutputWeightCount = invHiddenSize * actionSize;

        // Build inverse network input: previousState || currentState (contextBuffer_)
        std::copy_n(previousStateBuffer_.begin(), stateSize, inverseNetworkInputBuffer_.begin());
        std::copy_n(contextBuffer_.begin(), std::min(
            contextBuffer_.size(),
            inverseNetworkInputBuffer_.size() - stateSize),
            inverseNetworkInputBuffer_.begin() + stateSize
            );

        // Forward pass Input -> Hidden
        const auto invInputToHiddenWeights = inverseDynamicsWeights_.subspan(0, inputToHiddenWeightCount);
        const auto invHiddenBiases = inverseDynamicsBiases_.subspan(0, invHiddenSize);
        TensorOps::denseForwardPass(inverseNetworkInputBuffer_, invInputToHiddenWeights, invHiddenBiases, inverseNetworkHiddenBuffer_);
        TensorOps::applyLeakyReLU(inverseNetworkHiddenBuffer_, 0.01);

        // Forward pass Hidden -> Output (predict action)
        const auto invHiddenToOutputWeights = inverseDynamicsWeights_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount);
        const auto invOutputBiases = inverseDynamicsBiases_.subspan(invHiddenSize, actionSize);
        TensorOps::denseForwardPass(inverseNetworkHiddenBuffer_, invHiddenToOutputWeights, invOutputBiases, predictedActionBuffer_);
        sanitizeFinite(predictedActionBuffer_);

        logging::SpartanLogger::debug("[CURIOSITY-INFERENCE] runInverseDynamicsNetworkInverse() END");
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
        const auto stateSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.stateSize);
        const auto actionSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.actionSize);
        const auto hiddenSize = static_cast<size_t>(config->forwardDynamicsHiddenLayerDimensionSize);

        logging::SpartanLogger::debug(std::format(
            "[CURIOSITY-TRAIN] Dimensions: stateSize={}, actionSize={}, hiddenSize={}",
            stateSize, actionSize, hiddenSize));

        const size_t inputToHiddenWeightCount = (stateSize + actionSize) * hiddenSize;
        const size_t hiddenToOutputWeightCount = hiddenSize * stateSize;
        const size_t expectedWeightCount = inputToHiddenWeightCount + hiddenToOutputWeightCount;
        if (const size_t expectedBiasCount = hiddenSize + stateSize; forwardDynamicsWeights_.size() < expectedWeightCount || forwardDynamicsBiases_.size() < expectedBiasCount) {
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
        const auto hiddenToOutputWeightGradients = forwardDynamicsWeightGradients_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount);

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
        const auto inputToHiddenWeightGradients = forwardDynamicsWeightGradients_.subspan(0, inputToHiddenWeightCount);

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
        constexpr double beta1 = 0.9;
        constexpr double beta2 = 0.999;
        constexpr double epsilon = 1e-8;

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
            learningRate,
            beta1,
            beta2,
            epsilon,
            adamTimeStep_
        );
        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] Output biases updated");

        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] trainForwardDynamicsNetwork() END");
    }


    void CuriosityDrivenRecurrentSoftActorCriticSpartanModel::trainInverseDynamicsNetwork(const double predictionError) {
        logging::SpartanLogger::debug(std::format("[CURIOSITY-TRAIN] trainInverseDynamicsNetwork() START, predictionError={}", predictionError));

        if (!hasValidPreviousTick_) {
            logging::SpartanLogger::warn("[CURIOSITY-TRAIN] WARNING: hasValidPreviousTick_=false, aborting inverse training!");
            return;
        }

        // Compute inverse output gradients (MSE): grad = 2*(pred - target)/N
        const auto* config = typedConfig();
        const auto actionSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.actionSize);
        const auto invHiddenSize = static_cast<size_t>(config->inverseDynamicsHiddenLayerDimensionSize > 0
            ? config->inverseDynamicsHiddenLayerDimensionSize : config->forwardDynamicsHiddenLayerDimensionSize);
        const size_t inverseInputSize = static_cast<size_t>(config->recurrentSoftActorCriticConfig.baseConfig.stateSize) * 2;

        std::span<double> outGrad(predictedActionBuffer_.data(), actionSize);
        for (size_t i = 0; i < actionSize; ++i) {
            outGrad[i] = 2.0 * (predictedActionBuffer_[i] - previousActionBuffer_[i]) / static_cast<double>(actionSize);
        }

        // Zero inverse gradients accumulators
        std::ranges::fill(inverseDynamicsWeightGradients_, 0.0);
        std::ranges::fill(inverseDynamicsBiasGradients_, 0.0);

        const size_t inputToHiddenWeightCount = inverseInputSize * invHiddenSize;
        const size_t hiddenToOutputWeightCount = invHiddenSize * actionSize;

        const auto invHiddenToOutputWeights = inverseDynamicsWeights_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount);
        auto invHiddenToOutputWeightGradients = inverseDynamicsWeightGradients_.subspan(inputToHiddenWeightCount, hiddenToOutputWeightCount);

        // Backprop Hidden->Output
        TensorOps::denseBackwardPass(
            inverseNetworkHiddenBuffer_,
            outGrad,
            invHiddenToOutputWeights,
            invHiddenToOutputWeightGradients,
            inverseDynamicsHiddenActivationGradients_ /* reusing forward buffer name for hidden grads */
        );

        // Backprop Input->Hidden
        const auto invInputToHiddenWeights = inverseDynamicsWeights_.subspan(0, inputToHiddenWeightCount);
        auto invInputToHiddenWeightGradients = inverseDynamicsWeightGradients_.subspan(0, inputToHiddenWeightCount);

        TensorOps::denseBackwardPass(
            inverseNetworkInputBuffer_,
            inverseDynamicsHiddenActivationGradients_,
            invInputToHiddenWeights,
            invInputToHiddenWeightGradients,
            inverseNetworkInputBuffer_ /* reuse as dummy input grad */
        );

        // Copy bias gradients: hidden then output
        std::copy_n(inverseDynamicsHiddenActivationGradients_.begin(), invHiddenSize, inverseDynamicsBiasGradients_.begin());
        std::copy_n(outGrad.begin(), actionSize, inverseDynamicsBiasGradients_.begin() + invHiddenSize);

        // Apply Adam updates to inverse params
        const double invLr = config->inverseDynamicsLearningRate;
        TensorOps::applyAdamUpdate(
            inverseDynamicsWeights_,
            std::span<const double>(inverseDynamicsWeightGradients_),
            inverseWeightsFirstMoment_,
            inverseWeightsSecondMoment_,
            invLr, 0.9, 0.999, 1e-8, ++adamTimeStep_);

        TensorOps::applyAdamUpdate(
            inverseDynamicsBiases_,
            std::span<const double>(inverseDynamicsBiasGradients_),
            inverseBiasesFirstMoment_,
            inverseBiasesSecondMoment_,
            invLr, 0.9, 0.999, 1e-8, adamTimeStep_);

        logging::SpartanLogger::debug("[CURIOSITY-TRAIN] trainInverseDynamicsNetwork() END");
    }
}

