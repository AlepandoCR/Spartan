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
                                      void* opaqueHyperparameterConfig,
                                      double* criticWeightsBuffer,
                                      const int32_t criticWeightsCount,
                                      double* modelWeightsBuffer,
                                      const int32_t modelWeightsCount,
                                      double* contextBuffer,
                                      const int32_t contextCount,
                                      double* actionOutputBuffer,
                                      const int32_t actionOutputCount) {

        const std::span contextSpan(contextBuffer, static_cast<size_t>(contextCount));
        const std::span modelWeightsSpan(modelWeightsBuffer, static_cast<size_t>(modelWeightsCount));
        const std::span actionOutputSpan(actionOutputBuffer, static_cast<size_t>(actionOutputCount));

        // TODO -> use critic weights for value-based agents once implemented.
        (void)criticWeightsBuffer;
        (void)criticWeightsCount;

        // Attempt to recycle an idle model before allocating a new one.
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

        // No idle model available  -  construct a new DefaultSpartanAgent.
        auto model = std::make_unique<machinelearning::DefaultSpartanAgent>(
            agentIdentifier,
            opaqueHyperparameterConfig,
            modelWeightsSpan,
            contextSpan,
            actionOutputSpan
        );
        modelRegistry_.registerModel(std::move(model));
        logging::SpartanLogger::info(
            std::format("Registered new agent {}", agentIdentifier));
    }

    void SpartanEngine::unregisterAgent(const uint64_t agentIdentifier) {
        modelRegistry_.unregisterModel(agentIdentifier);
        logging::SpartanLogger::info(std::format("Unregistered agent {}", agentIdentifier));
    }

    void SpartanEngine::tickAllAgents(double* globalRewardsBuffer,
                                      const int32_t globalRewardsCount) {
        // TODO: Distribute per-agent reward slices from the global buffer.
        (void)globalRewardsBuffer;
        (void)globalRewardsCount;
        modelRegistry_.tickAll();
    }

    void SpartanEngine::updateContextPointer(const uint64_t agentIdentifier,
                                             double* newPointer,
                                             const int newCapacity) {
        const auto cleanContextSpan = memory::MemoryUtils::cleanView(newPointer, newCapacity);
        modelRegistry_.updateModelContext(agentIdentifier, cleanContextSpan);
    }

}
