//
// Created by Alepando on 25/2/2026.
//
#include "SpartanEngine.h"
#include <memory>
#include <span>
#include <format>

#include "math/fuzzy/SpartanFuzzyMath.h"
#include "memory/ArrayCleaners.h"
#include "machinelearning/model/DefaultSpartanAgent.h"
#include "machinelearning/model/RecurrentSoftActorCriticSpartanModel.h"
#include "machinelearning/model/DoubleDeepQNetworkSpartanModel.h"
#include "machinelearning/model/AutoEncoderCompressorSpartanModel.h"
#include "machinelearning/model/CuriosityDrivenRecurrentSoftActorCriticSpartanModel.h"
#include "machinelearning/persistence/SpartanPersistence.h"
#include "machinelearning/persistence/SpartanPersistence.h"

namespace org::spartan::internal {

    namespace {
        size_t simdPad(size_t count) {
            return (count + 7) & ~static_cast<size_t>(7);
        }
    }

    long SpartanEngine::computeFuzzySetUnion(double* targetFuzzySet,
                                             double* sourceFuzzySet,
                                             const int targetSetSize,
                                             const int sourceSetSize) {
        const auto timerStart = std::chrono::high_resolution_clock::now();
        std::span<double> targetCleanView =
            memory::MemoryUtils::cleanView(targetFuzzySet, targetSetSize);
        const std::span<double> sourceCleanView =
            memory::MemoryUtils::cleanView(sourceFuzzySet, sourceSetSize);
        math::fuzzy::FuzzySetOps::unionSets(targetCleanView.data(),
                                            sourceCleanView.data(),
                                            std::min(targetSetSize, sourceSetSize));
        const auto timerEnd = std::chrono::high_resolution_clock::now();
        const auto elapsedNanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(timerEnd - timerStart).count();
        return elapsedNanoseconds;
    }

    /**
     * Constructs a Recurrent Soft Actor-Critic model by slicing the flat critic and model
     * weight buffers into the individual sub-network spans expected by the constructor.
     */
    static std::unique_ptr<machinelearning::SpartanModel> constructRecurrentSoftActorCriticModel(
            const uint64_t agentIdentifier,
            void* opaqueHyperparameterConfig,
            double* criticWeightsBuffer,
            const int32_t criticWeightsCount,
            double* modelWeightsBuffer,
            const int32_t modelWeightsCount,
            double* contextBuffer,
            const int32_t contextCount,
            double* actionOutputBuffer,
            const int32_t actionOutputCount) {

        const auto* config = static_cast<const RecurrentSoftActorCriticHyperparameterConfig*>(
            opaqueHyperparameterConfig);

        const int gruHiddenSize = config->hiddenStateSize;
        const int gruInputSize = config->recurrentInputFeatureCount > 0
            ? config->recurrentInputFeatureCount : config->baseConfig.stateSize;
        const int gruConcatSize = gruHiddenSize + gruInputSize;

        const int actorHiddenSize = config->actorHiddenLayerNeuronCount;
        const int actorLayerCount = config->actorHiddenLayerCount;
        const int actionSize = config->baseConfig.actionSize;

        const int criticHiddenSize = config->criticHiddenLayerNeuronCount;
        const int criticLayerCount = config->criticHiddenLayerCount;

        const size_t gruGateWeightCount = static_cast<size_t>(3) * gruHiddenSize * gruConcatSize;
        const size_t gruGateBiasCount = static_cast<size_t>(3) * gruHiddenSize;
        const size_t gruHiddenStateCount = gruHiddenSize;

        // MULTI-LAYER CRITIC MATH
        const size_t criticCombinedInput = gruHiddenSize + actionSize;
        size_t criticWeightCountPerNetwork = criticHiddenSize * criticCombinedInput; // Input -> L1
        if (criticLayerCount > 1) {
             criticWeightCountPerNetwork += static_cast<size_t>(criticHiddenSize) * criticHiddenSize * (criticLayerCount - 1);
        }
        criticWeightCountPerNetwork += criticHiddenSize; // Ln -> Output

        size_t criticBiasCountPerNetwork = static_cast<size_t>(criticHiddenSize) * criticLayerCount; // Biases for all hidden layers
        criticBiasCountPerNetwork += 1; // Output bias

        size_t criticOffset = 0;
        const auto criticSpan = std::span(criticWeightsBuffer, criticWeightsCount);

        auto gruGateWeights = criticSpan.subspan(criticOffset, gruGateWeightCount);
        criticOffset += gruGateWeightCount;

        auto gruGateBiases = criticSpan.subspan(criticOffset, gruGateBiasCount);
        criticOffset += gruGateBiasCount;

        auto gruHiddenState = criticSpan.subspan(criticOffset, gruHiddenStateCount);
        criticOffset += gruHiddenStateCount;

        criticOffset = simdPad(criticOffset);

        auto firstCriticWeights = criticSpan.subspan(criticOffset, criticWeightCountPerNetwork);
        criticOffset += criticWeightCountPerNetwork;

        auto firstCriticBiases = criticSpan.subspan(criticOffset, criticBiasCountPerNetwork);
        criticOffset += criticBiasCountPerNetwork;

        criticOffset = simdPad(criticOffset);

        auto secondCriticWeights = criticSpan.subspan(criticOffset, criticWeightCountPerNetwork);
        criticOffset += criticWeightCountPerNetwork;

        auto secondCriticBiases = criticSpan.subspan(criticOffset, criticBiasCountPerNetwork);

        // MULTI-LAYER ACTOR MATH
        size_t totalPolicyWeightCount = static_cast<size_t>(actorHiddenSize) * gruHiddenSize; // Input -> L1
        if (actorLayerCount > 1) {
            totalPolicyWeightCount += static_cast<size_t>(actorHiddenSize) * actorHiddenSize * (actorLayerCount - 1);
        }
        totalPolicyWeightCount += static_cast<size_t>(actionSize) * actorHiddenSize; // Ln -> Mean
        totalPolicyWeightCount += static_cast<size_t>(actionSize) * actorHiddenSize; // Ln -> LogStd

        size_t totalPolicyBiasCount = static_cast<size_t>(actorHiddenSize) * actorLayerCount; // Biases for all hidden layers
        totalPolicyBiasCount += actionSize + actionSize; // Mean + LogStd biases

        const auto modelSpan = std::span(modelWeightsBuffer, modelWeightsCount);
        size_t modelOffset = 0;

        auto policyWeights = modelSpan.subspan(modelOffset, totalPolicyWeightCount);
        modelOffset += totalPolicyWeightCount;

        auto policyBiases = modelSpan.subspan(modelOffset, totalPolicyBiasCount);
        modelOffset += totalPolicyBiasCount;

        modelOffset = simdPad(modelOffset);
        auto encoderWeightPool = modelSpan.subspan(modelOffset);

        auto contextSpan = std::span<const double>(contextBuffer, contextCount);
        auto actionSpan = std::span(actionOutputBuffer, actionOutputCount);
        auto fullModelSpan = std::span(modelWeightsBuffer, modelWeightsCount);

        return std::make_unique<machinelearning::RecurrentSoftActorCriticSpartanModel>(
            agentIdentifier,
            opaqueHyperparameterConfig,
            fullModelSpan,
            contextSpan,
            actionSpan,
            gruGateWeights,
            gruGateBiases,
            gruHiddenState,
            policyWeights,
            policyBiases,
            firstCriticWeights,
            firstCriticBiases,
            secondCriticWeights,
            secondCriticBiases,
            encoderWeightPool);
    }


    static std::unique_ptr<machinelearning::SpartanModel> constructDoubleDeepQNetworkModel(
            const uint64_t agentIdentifier,
            void* opaqueHyperparameterConfig,
            double* criticWeightsBuffer,
            const int32_t criticWeightsCount,
            double* modelWeightsBuffer,
            const int32_t modelWeightsCount,
            double* contextBuffer,
            const int32_t contextCount,
            double* actionOutputBuffer,
            const int32_t actionOutputCount) {

        const auto* config = static_cast<const DoubleDeepQNetworkHyperparameterConfig*>(
            opaqueHyperparameterConfig);

        const int stateSize = config->baseConfig.stateSize;
        const int actionSize = config->baseConfig.actionSize;
        const int hiddenSize = config->hiddenLayerNeuronCount;

        const size_t onlineWeightCount = static_cast<size_t>(hiddenSize) * stateSize + static_cast<size_t>(actionSize) * hiddenSize;
        const size_t onlineBiasCount = static_cast<size_t>(hiddenSize) + actionSize;

        const auto modelSpan = std::span(modelWeightsBuffer, modelWeightsCount);
        size_t modelOffset = 0;

        auto onlineWeights = modelSpan.subspan(modelOffset, onlineWeightCount);
        modelOffset += onlineWeightCount;
        auto onlineBiases = modelSpan.subspan(modelOffset, onlineBiasCount);

        auto criticSpan = std::span(criticWeightsBuffer, criticWeightsCount);
        size_t criticOffset = 0;

        auto targetWeights = criticSpan.subspan(criticOffset, onlineWeightCount);
        criticOffset += onlineWeightCount;
        auto targetBiases = criticSpan.subspan(criticOffset, onlineBiasCount);

        return std::make_unique<machinelearning::DoubleDeepQNetworkSpartanModel>(
            agentIdentifier,
            opaqueHyperparameterConfig,
            std::span(modelWeightsBuffer, modelWeightsCount),
            std::span<const double>(contextBuffer, contextCount),
            std::span(actionOutputBuffer, actionOutputCount),
            onlineWeights, onlineBiases,
            targetWeights, targetBiases);
    }

    static std::unique_ptr<machinelearning::SpartanModel> constructAutoEncoderCompressorModel(
            const uint64_t agentIdentifier,
            void* opaqueHyperparameterConfig,
            double* modelWeightsBuffer,
            const int32_t modelWeightsCount,
            double* contextBuffer,
            const int32_t contextCount,
            double* actionOutputBuffer,
            const int32_t actionOutputCount) {

        const auto* config = static_cast<const AutoEncoderCompressorHyperparameterConfig*>(
            opaqueHyperparameterConfig);

        const int stateSize = config->baseConfig.stateSize;
        const int latentSize = config->latentDimensionSize;
        const int hiddenSize = config->encoderHiddenNeuronCount;

        const size_t encoderLayer1Weights = static_cast<size_t>(hiddenSize) * stateSize;
        const size_t encoderLayer2Weights = static_cast<size_t>(latentSize) * hiddenSize;
        const size_t encoderWeightCount = simdPad(encoderLayer1Weights + encoderLayer2Weights);
        const size_t encoderBiasCount = simdPad(static_cast<size_t>(hiddenSize) + latentSize);

        const size_t decoderLayer1Weights = static_cast<size_t>(hiddenSize) * latentSize;
        const size_t decoderLayer2Weights = static_cast<size_t>(stateSize) * hiddenSize;
        const size_t decoderWeightCount = simdPad(decoderLayer1Weights + decoderLayer2Weights);
        const size_t decoderBiasCount = simdPad(static_cast<size_t>(hiddenSize) + stateSize);

        const size_t latentCount = simdPad(latentSize);

        const auto modelSpan = std::span(modelWeightsBuffer, modelWeightsCount);
        size_t offset = 0;

        auto encoderWeights = modelSpan.subspan(offset, encoderWeightCount);
        offset += encoderWeightCount;
        auto encoderBiases = modelSpan.subspan(offset, encoderBiasCount);
        offset += encoderBiasCount;
        auto decoderWeights = modelSpan.subspan(offset, decoderWeightCount);
        offset += decoderWeightCount;
        auto decoderBiases = modelSpan.subspan(offset, decoderBiasCount);
        offset += decoderBiasCount;
        auto latentBuffer = modelSpan.subspan(offset, latentCount);

        return std::make_unique<machinelearning::AutoEncoderCompressorSpartanModel>(
            agentIdentifier,
            opaqueHyperparameterConfig,
            std::span(modelWeightsBuffer, modelWeightsCount),
            std::span<const double>(contextBuffer, contextCount),
            std::span(actionOutputBuffer, actionOutputCount),
            encoderWeights, encoderBiases,
            decoderWeights, decoderBiases,
            latentBuffer);
    }

        /**
     * Constructs a Curiosity-Driven RSAC model by surgically slicing flat buffers.
     * This function ensures total memory parity between Java's FFM allocations and C++ spans.
     */
    static std::unique_ptr<machinelearning::SpartanModel> constructCuriosityDrivenRecurrentSoftActorCriticModel(
            const uint64_t agentIdentifier,
            void* opaqueHyperparameterConfig,
            double* criticWeightsBuffer,
            const int32_t criticWeightsCount,
            double* modelWeightsBuffer,
            const int32_t modelWeightsCount,
            double* contextBuffer,
            const int32_t contextCount,
            double* actionOutputBuffer,
            const int32_t actionOutputCount) {

        const auto* config = static_cast<const CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig*>(
            opaqueHyperparameterConfig);

        // 1. Extract Curiosity (Forward Dynamics) dimensions
        const int32_t stateSize = config->recurrentSoftActorCriticConfig.baseConfig.stateSize;
        const int32_t actionSize = config->recurrentSoftActorCriticConfig.baseConfig.actionSize;
        const int32_t curiosityHiddenSize = config->forwardDynamicsHiddenLayerDimensionSize;

        const size_t curiosityWeights = static_cast<size_t>(stateSize + actionSize) * curiosityHiddenSize +
                                        static_cast<size_t>(curiosityHiddenSize) * stateSize;
        const size_t curiosityBiases = static_cast<size_t>(curiosityHiddenSize) + stateSize;

        // Extract RSAC internal dimensions
        const auto* rsacConfig = &config->recurrentSoftActorCriticConfig;
        const int32_t gruHiddenSize = rsacConfig->hiddenStateSize;
        const int32_t criticHiddenSize = rsacConfig->criticHiddenLayerNeuronCount;
        const int32_t criticLayerCount = rsacConfig->criticHiddenLayerCount;
        const int32_t actorHiddenSize = rsacConfig->actorHiddenLayerNeuronCount;
        const int32_t actorLayerCount = rsacConfig->actorHiddenLayerCount;

        // DEBUG: Verify if C++ is reading the struct layout correctly from Java
        logging::SpartanLogger::debug(std::format(
            "[DEBUG-CONFIG] stateSize={}, actionSize={}, criticHidden={}, criticLayers={}, actorHidden={}, actorLayers={}",
            stateSize, actionSize, criticHiddenSize, criticLayerCount, actorHiddenSize, actorLayerCount));

        // RSAC Component Math (Line-by-line parity with Java Allocator)
        const int32_t gruInputSize = rsacConfig->recurrentInputFeatureCount > 0
            ? rsacConfig->recurrentInputFeatureCount : stateSize;

        const size_t gruW_Count = static_cast<size_t>(3) * gruHiddenSize * (gruHiddenSize + gruInputSize);
        const size_t gruB_Count = static_cast<size_t>(3) * gruHiddenSize;
        const size_t gruS_Count = static_cast<size_t>(gruHiddenSize);

        const size_t criticIn = static_cast<size_t>(gruHiddenSize) + actionSize;
        size_t criticW_Count = (criticIn * criticHiddenSize) +
                               (criticLayerCount > 1 ? static_cast<size_t>(criticHiddenSize) * criticHiddenSize * (criticLayerCount - 1) : 0) +
                               criticHiddenSize;
        size_t criticB_Count = static_cast<size_t>(criticHiddenSize) * criticLayerCount + 1;

        // --- CRITIC SLICING ---
        auto criticSpan = std::span(criticWeightsBuffer, static_cast<size_t>(criticWeightsCount));
        size_t cOffset = 0;

        auto safeCriticSlice = [&](size_t size, std::string_view name) -> std::span<double> {
            if (size == 0) {
                logging::SpartanLogger::error(std::format("Critic slice size is zero: {}", name));
                return {};
            }
            if (cOffset + size > static_cast<size_t>(criticWeightsCount)) {
                logging::SpartanLogger::error(std::format("OUT OF BOUNDS (Critic): {} needs {}, but only {} left", name, size, criticWeightsCount - cOffset));
                return {};
            }
            const auto s = criticSpan.subspan(cOffset, size);
            cOffset += size;
            return s;
        };

        auto gruW = safeCriticSlice(gruW_Count, "GRU weights");
        auto gruB = safeCriticSlice(gruB_Count, "GRU biases");
        auto gruS = safeCriticSlice(gruS_Count, "GRU state");

        // Important: Java might apply SIMD padding between GRU and Critics
        cOffset = simdPad(cOffset);
        if (cOffset > static_cast<size_t>(criticWeightsCount)) {
            logging::SpartanLogger::error("Critic padding exceeded buffer size before Critic 1 slices");
            return nullptr;
        }

        auto c1W = safeCriticSlice(criticW_Count, "Critic 1 weights");
        auto c1B = safeCriticSlice(criticB_Count, "Critic 1 biases");

        cOffset = simdPad(cOffset);
        if (cOffset > static_cast<size_t>(criticWeightsCount)) {
            logging::SpartanLogger::error("Critic padding exceeded buffer size before Critic 2 slices");
            return nullptr;
        }

        auto c2W = safeCriticSlice(criticW_Count, "Critic 2 weights");
        auto c2B = safeCriticSlice(criticB_Count, "Critic 2 biases");

        cOffset = simdPad(cOffset);
        if (cOffset > static_cast<size_t>(criticWeightsCount)) {
            logging::SpartanLogger::error("Critic padding exceeded buffer size before Curiosity slices");
            return nullptr;
        }

        auto curW = safeCriticSlice(curiosityWeights, "Curiosity weights");
        auto curB = safeCriticSlice(curiosityBiases, "Curiosity biases");

        if (gruW.empty() || gruB.empty() || gruS.empty() || c1W.empty() || c1B.empty() || c2W.empty() || c2B.empty() || curW.empty() || curB.empty()) {
            logging::SpartanLogger::error("Curiosity critic slicing failed; aborting model construction");
            return nullptr;
        }

        // --- MODEL (ACTOR) SLICING ---
        auto modelSpan = std::span(modelWeightsBuffer, static_cast<size_t>(modelWeightsCount));
        size_t mOffset = 0;

        auto safeModelSlice = [&](size_t size, std::string_view name) -> std::span<double> {
            if (size == 0) {
                logging::SpartanLogger::error(std::format("Model slice size is zero: {}", name));
                return {};
            }
            if (mOffset + size > static_cast<size_t>(modelWeightsCount)) {
                logging::SpartanLogger::error(std::format("OUT OF BOUNDS (Model): {} needs {}, but only {} left", name, size, modelWeightsCount - mOffset));
                return {};
            }
            const auto s = modelSpan.subspan(mOffset, size);
            mOffset += size;
            return s;
        };

        size_t policyW_Count = (static_cast<size_t>(actorHiddenSize) * gruHiddenSize) +
                               (actorLayerCount > 1 ? static_cast<size_t>(actorHiddenSize) * actorHiddenSize * (actorLayerCount - 1) : 0) +
                               (static_cast<size_t>(actionSize) * actorHiddenSize * 2); // Mean + LogStd

        size_t policyB_Count = (static_cast<size_t>(actorHiddenSize) * actorLayerCount) + (actionSize * 2);

        auto pW = safeModelSlice(policyW_Count, "Policy weights");
        auto pB = safeModelSlice(policyB_Count, "Policy biases");

        if (pW.empty() || pB.empty()) {
            logging::SpartanLogger::error("Curiosity actor slicing failed; aborting model construction");
            return nullptr;
        }

        mOffset = simdPad(mOffset);
        if (mOffset > static_cast<size_t>(modelWeightsCount)) {
            logging::SpartanLogger::error("Model padding exceeded buffer size before encoder pool");
            return nullptr;
        }
        auto encoderPool = modelSpan.subspan(mOffset);

        // 4. Final Object Construction
        const uint64_t rsacId = (agentIdentifier << 1) | 1;
        auto contextSpan = std::span<const double>(contextBuffer, static_cast<size_t>(contextCount));
        auto actionSpan = std::span(actionOutputBuffer, static_cast<size_t>(actionOutputCount));

        if (cOffset < (curiosityWeights + curiosityBiases)) {
            logging::SpartanLogger::error("Curiosity critic offset underflow; aborting model construction");
            return nullptr;
        }

        auto internalRSAC = std::make_unique<machinelearning::RecurrentSoftActorCriticSpartanModel>(
            rsacId, const_cast<RecurrentSoftActorCriticHyperparameterConfig*>(rsacConfig),
            modelSpan, contextSpan, actionSpan,
            gruW, gruB, gruS,
            pW, pB,
            c1W, c1B, c2W, c2B,
            encoderPool);

        return std::make_unique<machinelearning::CuriosityDrivenRecurrentSoftActorCriticSpartanModel>(
            agentIdentifier, opaqueHyperparameterConfig,
            modelSpan, contextSpan, actionSpan,
            criticSpan.subspan(0, cOffset - (curiosityWeights + curiosityBiases)),
            curW, curB, std::move(internalRSAC));
    }

    void SpartanEngine::registerAgent(const uint64_t agentIdentifier,
                                      void* opaqueHyperparameterConfig,
                                      double* criticWeightsBuffer,
                                      const int32_t criticWeightsCount,
                                      double* modelWeightsBuffer,
                                      const int32_t modelWeightsCount,
                                      double* contextBuffer,
                                      const int32_t contextCount,
                                      double* actionOutputBuffer,
                                      const int32_t actionOutputCount) {

        const auto* baseConfig = static_cast<const BaseHyperparameterConfig*>(opaqueHyperparameterConfig);
        const int32_t modelType = baseConfig->modelTypeIdentifier;
        logging::SpartanLogger::setDebugEnabled(baseConfig->debugLogging);

        if (modelType == SPARTAN_MODEL_TYPE_DEFAULT) {
            const std::span contextSpan(contextBuffer, static_cast<size_t>(contextCount));
            const std::span modelWeightsSpan(modelWeightsBuffer, static_cast<size_t>(modelWeightsCount));
            const std::span actionOutputSpan(actionOutputBuffer, static_cast<size_t>(actionOutputCount));

            if (auto recycledModel = modelRegistry_.getIdleModelToRebind()) {
                recycledModel->rebind(agentIdentifier,
                                      opaqueHyperparameterConfig,
                                      modelWeightsSpan,
                                      contextSpan,
                                      actionOutputSpan);
                modelRegistry_.registerModel(std::move(recycledModel));
                logging::SpartanLogger::info(
                    std::format("Rebound idle model to agent {}", agentIdentifier));
                return;
            }

            auto model = std::make_unique<machinelearning::DefaultSpartanAgent>(
                agentIdentifier,
                opaqueHyperparameterConfig,
                modelWeightsSpan,
                contextSpan,
                actionOutputSpan);
            modelRegistry_.registerModel(std::move(model));
            logging::SpartanLogger::info(
                std::format("Registered default agent {}", agentIdentifier));
            return;
        }

        std::unique_ptr<machinelearning::SpartanModel> model;

        switch (modelType) {
            case SPARTAN_MODEL_TYPE_RECURRENT_SOFT_ACTOR_CRITIC:
                model = constructRecurrentSoftActorCriticModel(
                    agentIdentifier, opaqueHyperparameterConfig,
                    criticWeightsBuffer, criticWeightsCount,
                    modelWeightsBuffer, modelWeightsCount,
                    contextBuffer, contextCount,
                    actionOutputBuffer, actionOutputCount);
                logging::SpartanLogger::info(
                    std::format("Registered RecurrentSoftActorCritic agent {}", agentIdentifier));
                break;

            case SPARTAN_MODEL_TYPE_DOUBLE_DEEP_Q_NETWORK:
                model = constructDoubleDeepQNetworkModel(
                    agentIdentifier, opaqueHyperparameterConfig,
                    criticWeightsBuffer, criticWeightsCount,
                    modelWeightsBuffer, modelWeightsCount,
                    contextBuffer, contextCount,
                    actionOutputBuffer, actionOutputCount);
                logging::SpartanLogger::info(
                    std::format("Registered DoubleDeepQNetwork agent {}", agentIdentifier));
                break;

            case SPARTAN_MODEL_TYPE_AUTO_ENCODER_COMPRESSOR:
                model = constructAutoEncoderCompressorModel(
                    agentIdentifier, opaqueHyperparameterConfig,
                    modelWeightsBuffer, modelWeightsCount,
                    contextBuffer, contextCount,
                    actionOutputBuffer, actionOutputCount);
                logging::SpartanLogger::info(
                    std::format("Registered AutoEncoderCompressor agent {}", agentIdentifier));
                break;

            case SPARTAN_MODEL_TYPE_CURIOSITY_DRIVEN_RECURRENT_SOFT_ACTOR_CRITIC:
                model = constructCuriosityDrivenRecurrentSoftActorCriticModel(
                    agentIdentifier, opaqueHyperparameterConfig,
                    criticWeightsBuffer, criticWeightsCount,
                    modelWeightsBuffer, modelWeightsCount,
                    contextBuffer, contextCount,
                    actionOutputBuffer, actionOutputCount);
                logging::SpartanLogger::info(
                    std::format("Registered CuriosityDrivenRecurrentSoftActorCritic agent {}", agentIdentifier));
                break;

            default:
                logging::SpartanLogger::error(
                    std::format("Unknown model type {} for agent {}", modelType, agentIdentifier));
                return;
        }

        if (!model) {
            logging::SpartanLogger::error(
                std::format("Failed to construct model type {} for agent {}", modelType, agentIdentifier));
            return;
        }

        modelRegistry_.registerModel(std::move(model));
    }

    void SpartanEngine::unregisterAgent(const uint64_t agentIdentifier) {
        modelRegistry_.unregisterModel(agentIdentifier);
        logging::SpartanLogger::info(std::format("Unregistered agent {}", agentIdentifier));
    }

    void SpartanEngine::tickAllAgents(const uint64_t* agentIdentifiersBuffer,
                                      const double* rewardSignalsBuffer,
                                      const int32_t rewardEntryCount) {
        if (agentIdentifiersBuffer != nullptr && rewardSignalsBuffer != nullptr && rewardEntryCount > 0) {
            modelRegistry_.distributeRewardsByIdentifier(
                std::span<const uint64_t>(agentIdentifiersBuffer, rewardEntryCount),
                std::span<const double>(rewardSignalsBuffer, rewardEntryCount));
        }
        modelRegistry_.tickAll();
    }

    void SpartanEngine::updateContextPointer(const uint64_t agentIdentifier,
                                             double* newPointer,
                                             const int newCapacity) {
        const auto cleanContextSpan = memory::MemoryUtils::cleanView(newPointer, newCapacity);
        modelRegistry_.updateModelContext(agentIdentifier, cleanContextSpan);
    }

    void SpartanEngine::updateCleanSizes(const uint64_t agentIdentifier,
                                         const int32_t* cleanSizesBuffer,
                                         const int32_t slotCount) {
        modelRegistry_.updateModelCleanSizes(agentIdentifier,
            std::span(cleanSizesBuffer, slotCount));
    }

    bool SpartanEngine::saveModel(const uint64_t agentIdentifier, const char* filePath) {
        if (modelRegistry_.saveModelToFile(agentIdentifier, filePath)) {
            return true;
        }

        // Try searching in multi-agent groups if not found in regular registry
        bool foundAndSaved = false;
        multiAgentRegistry_.forEach([&](const std::unique_ptr<machinelearning::SpartanMultiAgentGroup>& group) {
            if (foundAndSaved) return;
            machinelearning::SpartanAgent* agent = group->getAgent(agentIdentifier);
            if (agent) {
                // Determine model type
                const auto* baseConfig = static_cast<const BaseHyperparameterConfig*>(
                    agent->getOpaqueHyperparameterConfig());
                const uint32_t modelType = baseConfig
                    ? static_cast<uint32_t>(baseConfig->modelTypeIdentifier)
                    : machinelearning::persistence::MODEL_TYPE_RECURRENT_SOFT_ACTOR_CRITIC;

                // Build topology and extract
                const std::span<const double> modelWeightBlob = agent->getModelWeights();
                const std::span<const double> criticWeightBlob = agent->getCriticWeights();

                machinelearning::persistence::SubModelTopologyEntry modelTopologyEntry{};
                modelTopologyEntry.subModelRole = machinelearning::persistence::SUB_MODEL_GAUSSIAN_POLICY;
                modelTopologyEntry.subModelIndex = 0;
                modelTopologyEntry.weightByteOffsetRelative = 0;
                modelTopologyEntry.weightElementCount = modelWeightBlob.size();
                modelTopologyEntry.biasesByteOffsetRelative = 0;
                modelTopologyEntry.biasElementCount = 0;

                std::vector<machinelearning::persistence::SubModelTopologyEntry> topologyEntries;
                topologyEntries.push_back(modelTopologyEntry);

                if (!criticWeightBlob.empty()) {
                    machinelearning::persistence::SubModelTopologyEntry criticTopologyEntry{};
                    criticTopologyEntry.subModelRole = machinelearning::persistence::SUB_MODEL_Q_CRITIC_FIRST;
                    criticTopologyEntry.subModelIndex = 1;
                    criticTopologyEntry.weightByteOffsetRelative = modelWeightBlob.size() * sizeof(double);
                    criticTopologyEntry.weightElementCount = criticWeightBlob.size();
                    criticTopologyEntry.biasesByteOffsetRelative = 0;
                    criticTopologyEntry.biasElementCount = 0;
                    topologyEntries.push_back(criticTopologyEntry);
                }

                std::vector<double> concatenatedWeightBlob;
                concatenatedWeightBlob.reserve(modelWeightBlob.size() + criticWeightBlob.size());
                concatenatedWeightBlob.insert(concatenatedWeightBlob.end(),
                    modelWeightBlob.begin(), modelWeightBlob.end());
                concatenatedWeightBlob.insert(concatenatedWeightBlob.end(),
                    criticWeightBlob.begin(), criticWeightBlob.end());

                foundAndSaved = machinelearning::persistence::saveModel(
                    filePath,
                    modelType,
                    std::span<const machinelearning::persistence::SubModelTopologyEntry>(topologyEntries),
                    std::span<const double>(concatenatedWeightBlob));
            }
        });

        if (!foundAndSaved) {
            logging::SpartanLogger::error(
                std::format("saveModel: No active model or nested agent found for ID {}", agentIdentifier));
        }

        return foundAndSaved;
    }

    bool SpartanEngine::loadModel(const uint64_t agentIdentifier, const char* filePath) {
        using namespace machinelearning::persistence;

        // Try searching in multi-agent groups first
        machinelearning::SpartanModel* targetModel = nullptr;

        multiAgentRegistry_.forEach([&](const std::unique_ptr<machinelearning::SpartanMultiAgentGroup>& group) {
            if (targetModel) return;
            machinelearning::SpartanAgent* agent = group->getAgent(agentIdentifier);
            if (agent) {
                targetModel = agent;
            }
        });

        // If not found, try regular registry
        if (!targetModel) {
            targetModel = modelRegistry_.getModel(agentIdentifier);
        }

        if (!targetModel) {
            logging::SpartanLogger::error(
                std::format("loadModel: No active model or nested agent found for ID {}", agentIdentifier));
            return false;
        }

        SpartanFileHeader header{};
        if (!loadHeader(filePath, header)) {
            logging::SpartanLogger::error("loadModel: Failed to read or validate .spartan file header.");
            return false;
        }

        // Retrieve both weight buffers from the active model
        const std::span<double> modelWeightBlob = targetModel->getModelWeightsMutable();
        const std::span<double> criticWeightBlob = targetModel->getCriticWeightsMutable();

        const uint64_t expectedWeightCount = modelWeightBlob.size() + criticWeightBlob.size();
        if (header.totalWeightCount != expectedWeightCount) {
            logging::SpartanLogger::error(
                std::format("loadModel: Weight count mismatch. File has {}, agent expects {}.",
                            header.totalWeightCount, expectedWeightCount));
            return false;
        }

        std::vector<double> concatBuffer(expectedWeightCount);
        auto weightSpan = std::span(concatBuffer);
        if (!loadWeights(filePath, header, weightSpan)) {
            logging::SpartanLogger::error("loadModel: CRC-32 mismatch or I/O error reading weights.");
            return false;
        }

        // Unpack into the active model's buffers
        std::copy(concatBuffer.begin(), concatBuffer.begin() + modelWeightBlob.size(), modelWeightBlob.begin());
        if (!criticWeightBlob.empty()) {
            std::copy(concatBuffer.begin() + modelWeightBlob.size(), concatBuffer.end(), criticWeightBlob.begin());
        }

        logging::SpartanLogger::info(
            std::format("Loaded {} weights into agent {} from {}", header.totalWeightCount, agentIdentifier, filePath));
        return true;
    }

    void SpartanEngine::decayExploration(const uint64_t agentIdentifier) {
        modelRegistry_.decayExplorationForAgent(agentIdentifier);
    }

    bool SpartanEngine::tickAgent(const uint64_t agentIdentifier, const double rewardSignal) {
        return modelRegistry_.tickSingleAgent(agentIdentifier, rewardSignal);
    }

    void SpartanEngine::registerMultiAgentGroup(const uint64_t groupIdentifier,
                                                double* sharedContextBuffer,
                                                const int32_t sharedContextCount,
                                                double* sharedActionsBuffer,
                                                const int32_t sharedActionsCount,
                                                const int32_t stateSize,
                                                const int32_t actionSize,
                                                const int32_t maxAgents) {
        if (sharedContextBuffer == nullptr || sharedActionsBuffer == nullptr) {
            logging::SpartanLogger::error("registerMultiAgentGroup: received null shared buffers.");
            return;
        }
        if (sharedContextCount <= 0 || sharedActionsCount <= 0 || stateSize <= 0 || actionSize <= 0 || maxAgents <= 0) {
            logging::SpartanLogger::error("registerMultiAgentGroup: received invalid sizes.");
            return;
        }

        auto contextSpan = std::span<const double>(sharedContextBuffer, sharedContextCount);
        auto actionsSpan = std::span<double>(sharedActionsBuffer, sharedActionsCount);

        auto group = std::make_unique<machinelearning::SpartanMultiAgentGroup>(
            groupIdentifier,
            contextSpan,
            actionsSpan,
            stateSize,
            actionSize,
            maxAgents);

        // Ensure group was created successfully
        if (!group) {
            logging::SpartanLogger::error("registerMultiAgentGroup: Failed to allocate SpartanMultiAgentGroup.");
            return;
        }

        multiAgentRegistry_.insert(groupIdentifier, std::move(group));
        logging::SpartanLogger::info(
            std::format("Registered multi-agent group {}", groupIdentifier));
    }

    void SpartanEngine::tickMultiAgentGroup(const uint64_t groupIdentifier) {
        auto groupPtr = multiAgentRegistry_.get(groupIdentifier);
        if (groupPtr && *groupPtr) {
            (*groupPtr)->processTick();
            return;
        }
        logging::SpartanLogger::error(
            std::format("tickMultiAgentGroup: No active group found for ID {}", groupIdentifier));
    }

    void SpartanEngine::unregisterMultiAgentGroup(const uint64_t groupIdentifier) {
        auto groupPtr = multiAgentRegistry_.get(groupIdentifier);
        if (groupPtr && *groupPtr) {
            multiAgentRegistry_.erase(groupIdentifier);
            logging::SpartanLogger::info(
                std::format("Unregistered multi-agent group {}", groupIdentifier));
            return;
        }
        logging::SpartanLogger::error(
            std::format("unregisterMultiAgentGroup: No active group found for ID {}", groupIdentifier));
    }

    void SpartanEngine::multiAgentApplyRewards(const uint64_t groupIdentifier,
                                               const double* rewardsBuffer,
                                               const int32_t rewardCount) {
        auto groupPtr = multiAgentRegistry_.get(groupIdentifier);
        if (groupPtr && *groupPtr) {
            (*groupPtr)->applyRewardsToAll(std::span<const double>(rewardsBuffer, rewardCount));
            return;
        }
        logging::SpartanLogger::error(
            std::format("multiAgentApplyRewards: No active group found for ID {}", groupIdentifier));
    }

    bool SpartanEngine::addAgentToMultiAgentGroup(const uint64_t groupIdentifier,
                                                  const uint64_t agentIdentifier,
                                                  void* opaqueConfig,
                                                  double* modelWeights,
                                                  int32_t modelWeightsCount,
                                                  double* criticWeights,
                                                  int32_t criticWeightsCount) {
        auto groupPtr = multiAgentRegistry_.get(groupIdentifier);
        if (!groupPtr || !*groupPtr) {
            logging::SpartanLogger::error(
                std::format("addAgentToMultiAgentGroup: No active group found for ID {}", groupIdentifier));
            return false;
        }

        const auto* baseConfig = static_cast<const BaseHyperparameterConfig*>(opaqueConfig);
        std::unique_ptr<machinelearning::SpartanModel> model;

        switch (baseConfig->modelTypeIdentifier) {
            case SPARTAN_MODEL_TYPE_RECURRENT_SOFT_ACTOR_CRITIC: {
                model = constructRecurrentSoftActorCriticModel(
                    agentIdentifier,
                    opaqueConfig,
                    criticWeights,
                    criticWeightsCount,
                    modelWeights,
                    modelWeightsCount,
                    nullptr, 0, // Context handled by group
                    nullptr, 0  // Actions handled by group
                );
                break;
            }
            case SPARTAN_MODEL_TYPE_DOUBLE_DEEP_Q_NETWORK: {
                model = constructDoubleDeepQNetworkModel(
                    agentIdentifier,
                    opaqueConfig,
                    criticWeights,
                    criticWeightsCount,
                    modelWeights,
                    modelWeightsCount,
                    nullptr, 0,
                    nullptr, 0
                );
                break;
            }
            // AutoEncoder is a SpartanCompressor, not a SpartanAgent, so it cannot be added here.

            case SPARTAN_MODEL_TYPE_CURIOSITY_DRIVEN_RECURRENT_SOFT_ACTOR_CRITIC: {
                model = constructCuriosityDrivenRecurrentSoftActorCriticModel(
                    agentIdentifier,
                    opaqueConfig,
                    criticWeights,
                    criticWeightsCount,
                    modelWeights,
                    modelWeightsCount,
                    nullptr, 0,
                    nullptr, 0
                );
                break;
            }
            default:
                logging::SpartanLogger::error(
                    std::format("addAgentToMultiAgentGroup: Unsupported or unknown model type {} for agent {}",
                    baseConfig->modelTypeIdentifier, agentIdentifier));
                return false;
        }

        if (!model) {
            logging::SpartanLogger::error("addAgentToMultiAgentGroup: Failed to construct agent.");
            return false;
        }

        // Transfer ownership and cast to SpartanAgent
        // We know these types inherit from SpartanAgent.
        auto* agentPtr = dynamic_cast<machinelearning::SpartanAgent*>(model.release());
        std::unique_ptr<machinelearning::SpartanAgent> agent(agentPtr);

        (*groupPtr)->addAgent(agentIdentifier, std::move(agent));

        logging::SpartanLogger::info(
            std::format("Added agent {} to group {}", agentIdentifier, groupIdentifier));
        return true;
    }

    bool SpartanEngine::removeAgentFromMultiAgentGroup(const uint64_t groupIdentifier,
                                                       const uint64_t agentIdentifier) {
        auto groupPtr = multiAgentRegistry_.get(groupIdentifier);
        if (!groupPtr || !*groupPtr) {
            logging::SpartanLogger::error(
                std::format("removeAgentFromMultiAgentGroup: No active group found for ID {}", groupIdentifier));
            return false;
        }

        // Remove agent from the group
        (*groupPtr)->removeAgent(agentIdentifier);
        logging::SpartanLogger::info(
            std::format("Removed agent {} from group {}", agentIdentifier, groupIdentifier));
        return true;
    }

    bool SpartanEngine::saveModel(const uint64_t agentIdentifier,
                                  const char* filePath,
                                  const uint32_t modelTypeId) {
        auto modelPtr = modelRegistry_.getModel(agentIdentifier);
        if (!modelPtr) {
            logging::SpartanLogger::error(
                std::format("saveModel: Agent {} not found in registry", agentIdentifier));
            return false;
        }

        if (filePath == nullptr) {
            logging::SpartanLogger::error("saveModel: filePath is null");
            return false;
        }

        bool success = machinelearning::persistence::saveModelWithModule(filePath, modelPtr, modelTypeId);
        if (success) {
            logging::SpartanLogger::info(
                std::format("saveModel: Successfully saved agent {} to {}", agentIdentifier, filePath));
        } else {
            logging::SpartanLogger::error(
                std::format("saveModel: Failed to save agent {} to {}", agentIdentifier, filePath));
        }
        return success;
    }

    bool SpartanEngine::loadModel(const uint64_t agentIdentifier,
                                  const char* filePath,
                                  const uint32_t modelTypeId) {
        auto modelPtr = modelRegistry_.getModel(agentIdentifier);
        if (!modelPtr) {
            logging::SpartanLogger::error(
                std::format("loadModel: Agent {} not found in registry", agentIdentifier));
            return false;
        }

        if (filePath == nullptr) {
            logging::SpartanLogger::error("loadModel: filePath is null");
            return false;
        }

        bool success = machinelearning::persistence::loadModelWithModule(filePath, modelPtr, modelTypeId);
        if (success) {
            logging::SpartanLogger::info(
                std::format("loadModel: Successfully loaded agent {} from {}", agentIdentifier, filePath));
        } else {
            logging::SpartanLogger::error(
                std::format("loadModel: Failed to load agent {} from {}", agentIdentifier, filePath));
        }
        return success;
    }

}
