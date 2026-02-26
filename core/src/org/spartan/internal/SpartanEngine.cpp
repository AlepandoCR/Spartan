//
// Created by Alepando on 25/2/2026.
//
#include "SpartanEngine.h"
#include <memory>
#include <span>
#include <format>

#include "math/fuzzy/SpartanFuzzyMath.h"
#include "memory/ArrayCleaners.h"

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
    void SpartanEngine::registerAgent(const uint64_t agentIdentifier,
                                      ModelHyperparameterConfig* hyperparameterConfig,
                                      double* criticWeightsBuffer,
                                      const int32_t criticWeightsCount,
                                      double* modelWeightsBuffer,
                                      const int32_t modelWeightsCount,
                                      double* contextBuffer,
                                      const int32_t contextCount,
                                      double* actionOutputBuffer,
                                      const int32_t actionOutputCount) {

        std::span contextSpan(contextBuffer, static_cast<size_t>(contextCount));

        std::span modelWeightsSpan(modelWeightsBuffer,
                                           static_cast<size_t>(modelWeightsCount));
        std::span actionOutputSpan(actionOutputBuffer,
                                           static_cast<size_t>(actionOutputCount));

        // TODO: Construct a concrete critic using criticWeightsBuffer/criticWeightsCount
        //       once a non-abstract implementation is available.
        machinelearning::SpartanAbstractCritic* criticInstance = nullptr;

        // Check for idle models to reuse before constructing a new one

        if (const auto idleModel = modelRegistry_.getIdleModelToRebind()) {
            idleModel->rebind(
                agentIdentifier,
                hyperparameterConfig,
                criticInstance,
                modelWeightsSpan,
                contextSpan,
                actionOutputSpan
            );
        }

        // No idle model available, construct a new one
        auto model = std::make_unique<machinelearning::SpartanBaseModel>(
        agentIdentifier,
        hyperparameterConfig,
        criticInstance,
        modelWeightsSpan,
        contextSpan,
        actionOutputSpan
        );
        modelRegistry_.registerModel(std::move(model));
    }
    void SpartanEngine::unregisterAgent(const uint64_t agentIdentifier) {
        modelRegistry_.unregisterModel(agentIdentifier);
        logging::SpartanLogger::info(std::format("Unregistered agent {}", agentIdentifier));
    }
    void SpartanEngine::tickAllAgents(double* globalRewardsBuffer,
                                      const int32_t globalRewardsCount) {
        std::span<const double> globalRewardsSpan(globalRewardsBuffer,
                                                   static_cast<size_t>(globalRewardsCount));
        modelRegistry_.tickAll();
    }
} // namespace org::spartan::core
