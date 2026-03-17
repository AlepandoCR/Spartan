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
        auto criticSpan = std::span(criticWeightsBuffer, criticWeightsCount);

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

        // 2. Extract RSAC internal dimensions
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

        // 3. RSAC Component Math (Line-by-line parity with Java Allocator)
        const int32_t gruInputSize = rsacConfig->recurrentInputFeatureCount > 0
            ? rsacConfig->recurrentInputFeatureCount : stateSize;

        const size_t gruW_Count = static_cast<size_t>(3) * gruHiddenSize * (gruHiddenSize + gruInputSize);
        const size_t gruB_Count = static_cast<size_t>(3) * gruHiddenSize;
        const size_t gruS_Count = static_cast<size_t>(gruHiddenSize);

        const size_t criticIn = static_cast<size_t>(gruHiddenSize) + actionSize;
        size_t criticW_Count = (criticIn * criticHiddenSize) +
                               (criticLayerCount > 1 ? (size_t)criticHiddenSize * criticHiddenSize * (criticLayerCount - 1) : 0) +
                               criticHiddenSize;
        size_t criticB_Count = (size_t)criticHiddenSize * criticLayerCount + 1;

        // --- CRITIC SLICING ---
        auto criticSpan = std::span(criticWeightsBuffer, static_cast<size_t>(criticWeightsCount));
        size_t cOffset = 0;

        auto safeCriticSlice = [&](size_t size, std::string_view name) -> std::span<double> {
            if (cOffset + size > static_cast<size_t>(criticWeightsCount)) {
                logging::SpartanLogger::error(std::format("OUT OF BOUNDS (Critic): {} needs {}, but only {} left", name, size, criticWeightsCount - cOffset));
                return {};
            }
            auto s = criticSpan.subspan(cOffset, size);
            cOffset += size;
            return s;
        };

        auto gruW = safeCriticSlice(gruW_Count, "GRU weights");
        auto gruB = safeCriticSlice(gruB_Count, "GRU biases");
        auto gruS = safeCriticSlice(gruS_Count, "GRU state");

        // Important: Java might apply SIMD padding between GRU and Critics
        cOffset = simdPad(cOffset);

        auto c1W = safeCriticSlice(criticW_Count, "Critic 1 weights");
        auto c1B = safeCriticSlice(criticB_Count, "Critic 1 biases");

        cOffset = simdPad(cOffset);

        auto c2W = safeCriticSlice(criticW_Count, "Critic 2 weights");
        auto c2B = safeCriticSlice(criticB_Count, "Critic 2 biases");

        cOffset = simdPad(cOffset);

        auto curW = safeCriticSlice(curiosityWeights, "Curiosity weights");
        auto curB = safeCriticSlice(curiosityBiases, "Curiosity biases");

        if (gruW.empty() || c1W.empty() || curW.empty()) return nullptr;

        // --- MODEL (ACTOR) SLICING ---
        auto modelSpan = std::span(modelWeightsBuffer, static_cast<size_t>(modelWeightsCount));
        size_t mOffset = 0;

        auto safeModelSlice = [&](size_t size, std::string_view name) -> std::span<double> {
            if (mOffset + size > static_cast<size_t>(modelWeightsCount)) {
                logging::SpartanLogger::error(std::format("OUT OF BOUNDS (Model): {} needs {}, but only {} left", name, size, modelWeightsCount - mOffset));
                return {};
            }
            auto s = modelSpan.subspan(mOffset, size);
            mOffset += size;
            return s;
        };

        size_t policyW_Count = (static_cast<size_t>(actorHiddenSize) * gruHiddenSize) +
                               (actorLayerCount > 1 ? (size_t)actorHiddenSize * actorHiddenSize * (actorLayerCount - 1) : 0) +
                               (static_cast<size_t>(actionSize) * actorHiddenSize * 2); // Mean + LogStd

        size_t policyB_Count = (static_cast<size_t>(actorHiddenSize) * actorLayerCount) + (actionSize * 2);

        auto pW = safeModelSlice(policyW_Count, "Policy weights");
        auto pB = safeModelSlice(policyB_Count, "Policy biases");

        mOffset = simdPad(mOffset);
        auto encoderPool = modelSpan.subspan(mOffset);

        // 4. Final Object Construction
        const uint64_t rsacId = (agentIdentifier << 1) | 1;
        auto contextSpan = std::span<const double>(contextBuffer, static_cast<size_t>(contextCount));
        auto actionSpan = std::span(actionOutputBuffer, static_cast<size_t>(actionOutputCount));

        auto internalRSAC = std::make_unique<machinelearning::RecurrentSoftActorCriticSpartanModel>(
            rsacId, const_cast<RecurrentSoftActorCriticHyperparameterConfig*>(rsacConfig),
            modelSpan, contextSpan, actionSpan,
            gruW, gruB, gruS,
            pW, pB, // Correctly sliced Actor spans
            c1W, c1B, c2W, c2B,
            encoderPool);

        return std::make_unique<machinelearning::CuriosityDrivenRecurrentSoftActorCriticSpartanModel>(
            agentIdentifier, opaqueHyperparameterConfig,
            modelSpan, contextSpan, actionSpan,
            criticSpan.subspan(0, cOffset - (curiosityWeights + curiosityBiases)), // RSAC Critic block
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
        return modelRegistry_.saveModelToFile(agentIdentifier, filePath);
    }

    bool SpartanEngine::loadModel(const char* filePath,
                                  double* targetWeightBuffer,
                                  const int32_t targetWeightCount) {
        using namespace machinelearning::persistence;

        SpartanFileHeader header{};
        if (!loadHeader(filePath, header)) {
            logging::SpartanLogger::error("loadModel: Failed to read or validate .spartan file header.");
            return false;
        }

        auto weightSpan = std::span(targetWeightBuffer, targetWeightCount);
        if (!loadWeights(filePath, header, weightSpan)) {
            logging::SpartanLogger::error("loadModel: CRC-32 mismatch or I/O error reading weights.");
            return false;
        }

        logging::SpartanLogger::info(
            std::format("Loaded {} weights from {}", header.totalWeightCount, filePath));
        return true;
    }

    void SpartanEngine::decayExploration(const uint64_t agentIdentifier) {
        modelRegistry_.decayExplorationForAgent(agentIdentifier);
    }

    bool SpartanEngine::tickAgent(const uint64_t agentIdentifier, const double rewardSignal) {
        return modelRegistry_.tickSingleAgent(agentIdentifier, rewardSignal);
    }

}