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
#include "machinelearning/persistence/SpartanPersistence.h"

namespace org::spartan::internal {

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
     *
     * Weight layout in criticWeightsBuffer (JVM side):
     *   [GRU gate weights | GRU gate biases | GRU hidden state | Q1 weights | Q1 biases | Q2 weights | Q2 biases]
     *
     * Weight layout in modelWeightsBuffer (JVM side):
     *   [Policy weights | Policy biases | Encoder weight pool (if nestedEncoderCount > 0)]
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

        const int hiddenSize = config->actorHiddenLayerNeuronCount;
        const int inputSize = config->recurrentInputFeatureCount > 0
            ? config->recurrentInputFeatureCount : config->baseConfig.stateSize;
        const int actionSize = config->baseConfig.actionSize;
        const int criticHiddenSize = config->criticHiddenLayerNeuronCount;
        const int concatSize = hiddenSize + inputSize;

        // Slice the critic weights buffer into GRU + Q1 + Q2 sub-spans.
        // GRU has 3 gates (update, reset, candidate), each with weights and biases.
        const size_t gruGateWeightCount = static_cast<size_t>(3) * hiddenSize * concatSize;
        const size_t gruGateBiasCount = static_cast<size_t>(3) * hiddenSize;
        const size_t gruHiddenStateCount = hiddenSize;

        // Q-critics: combined input = hidden + action, one hidden layer, one scalar output
        const size_t criticCombinedInput = hiddenSize + actionSize;
        const size_t criticWeightCountPerNetwork = (criticHiddenSize * criticCombinedInput) + criticHiddenSize;
        const size_t criticBiasCountPerNetwork = criticHiddenSize + 1;

        size_t criticOffset = 0;
        auto criticSpan = std::span(criticWeightsBuffer, criticWeightsCount);

        auto gruGateWeights = criticSpan.subspan(criticOffset, gruGateWeightCount);
        criticOffset += gruGateWeightCount;

        auto gruGateBiases = criticSpan.subspan(criticOffset, gruGateBiasCount);
        criticOffset += gruGateBiasCount;

        auto gruHiddenState = criticSpan.subspan(criticOffset, gruHiddenStateCount);
        criticOffset += gruHiddenStateCount;

        auto firstCriticWeights = criticSpan.subspan(criticOffset, criticWeightCountPerNetwork);
        criticOffset += criticWeightCountPerNetwork;

        auto firstCriticBiases = criticSpan.subspan(criticOffset, criticBiasCountPerNetwork);
        criticOffset += criticBiasCountPerNetwork;

        auto secondCriticWeights = criticSpan.subspan(criticOffset, criticWeightCountPerNetwork);
        criticOffset += criticWeightCountPerNetwork;

        auto secondCriticBiases = criticSpan.subspan(criticOffset, criticBiasCountPerNetwork);

        // Slice the model weights buffer into Policy + Encoder pool sub-spans.
        // Policy: input-to-hidden weights + hidden biases + mean-output weights + mean biases
        //         + log-std-output weights + log-std biases
        const size_t policyLayer1WeightCount = static_cast<size_t>(hiddenSize) * hiddenSize;
        const size_t policyMeanWeightCount = static_cast<size_t>(actionSize) * hiddenSize;
        const size_t policyLogStdWeightCount = static_cast<size_t>(actionSize) * hiddenSize;
        const size_t totalPolicyWeightCount = policyLayer1WeightCount + policyMeanWeightCount + policyLogStdWeightCount;
        const size_t totalPolicyBiasCount = static_cast<size_t>(hiddenSize) + actionSize + actionSize;

        const auto modelSpan = std::span(modelWeightsBuffer, modelWeightsCount);
        size_t modelOffset = 0;

        auto policyWeights = modelSpan.subspan(modelOffset, totalPolicyWeightCount);
        modelOffset += totalPolicyWeightCount;

        auto policyBiases = modelSpan.subspan(modelOffset, totalPolicyBiasCount);
        modelOffset += totalPolicyBiasCount;

        // Remaining model weights belong to the nested encoder pool
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


    /**
     * Constructs a Double Deep Q-Network model by slicing the weight buffers.
     *
     * Weight layout in criticWeightsBuffer:
     *   [Target network weights | Target network biases]
     *
     * Weight layout in modelWeightsBuffer:
     *   [Online network weights | Online network biases]
     */
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

        // Online network: layer 1 weights + hidden layer bias + output weights + output bias
        const size_t onlineWeightCount = static_cast<size_t>(hiddenSize) * stateSize + static_cast<size_t>(actionSize) * hiddenSize;
        const size_t onlineBiasCount = static_cast<size_t>(hiddenSize) + actionSize;

        const auto modelSpan = std::span(modelWeightsBuffer, modelWeightsCount);
        size_t modelOffset = 0;

        auto onlineWeights = modelSpan.subspan(modelOffset, onlineWeightCount);
        modelOffset += onlineWeightCount;
        auto onlineBiases = modelSpan.subspan(modelOffset, onlineBiasCount);

        // Target network mirrors online network dimensions
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

    /**
     * Constructs an AutoEncoder Compressor model by slicing the weight buffers.
     *
     * Weight layout in modelWeightsBuffer:
     *   [Encoder weights | Encoder biases | Decoder weights | Decoder biases | Latent buffer]
     */
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

        const size_t encoderWeightCount = static_cast<size_t>(hiddenSize) * stateSize;
        const size_t encoderBiasCount = hiddenSize;
        const size_t decoderWeightCount = static_cast<size_t>(stateSize) * hiddenSize;
        const size_t decoderBiasCount = stateSize;
        const size_t latentCount = latentSize;

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

        // Read the model type discriminator from the first field of the base config.
        const auto* baseConfig = static_cast<const BaseHyperparameterConfig*>(opaqueHyperparameterConfig);
        const int32_t modelType = baseConfig->modelTypeIdentifier;

        // Attempt to recycle an idle model before allocating a new one.
        // Recycling only works for DefaultSpartanAgent (model type 0) since
        // concrete model types have different internal topologies.
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

        // Construct the appropriate concrete model based on the type discriminator.
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

    void SpartanEngine::tickAllAgents(double* globalRewardsBuffer,
                                      const int32_t globalRewardsCount) {
        // Phase 1: Execute parallel inference across all models.
        modelRegistry_.tickAll();

        // Distribute global rewards to agents that support reward-based learning.
        // Each element in the rewards buffer corresponds to the agent at the same
        // index in the registry's iteration order. Agents that are not SpartanAgent
        // subclasses simply ignore the reward signal.
        if (globalRewardsBuffer != nullptr && globalRewardsCount > 0) {
            modelRegistry_.applyGlobalRewards(
                std::span<const double>(globalRewardsBuffer, globalRewardsCount));
        }
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

}
