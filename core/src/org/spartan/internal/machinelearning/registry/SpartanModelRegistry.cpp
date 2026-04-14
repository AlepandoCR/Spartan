//
// Created by Alepando on 25/2/2026.
//

#include "SpartanModelRegistry.h"

#include <vector>
#include <execution>
#include <algorithm>
#include <format>
#include <ranges>

#include "internal/logging/SpartanLogger.h"
#include "internal/machinelearning/model/SpartanModel.h"
#include "internal/machinelearning/model/SpartanAgent.h"
#include "internal/machinelearning/ModelHyperparameterConfig.h"
#include "internal/machinelearning/persistence/SpartanPersistence.h"

namespace org::spartan::internal::machinelearning {

    void SpartanModelRegistry::registerModel(std::unique_ptr<SpartanModel> model) {
        std::lock_guard lock(registryMutex_);
        activeModels_[model->getIdentifier()] = std::move(model);

        auto newSnapshot = std::make_shared<std::vector<SpartanModel*>>();
        newSnapshot->reserve(activeModels_.size());
        for (auto& modelEntry : activeModels_ | std::views::values) {
            newSnapshot->push_back(modelEntry.get());
        }
        tickSnapshot_ = newSnapshot;
    }

    void SpartanModelRegistry::unregisterModel(const uint64_t agentIdentifier) {
        std::lock_guard lock(registryMutex_);

        if (const auto iterator = activeModels_.find(agentIdentifier); iterator != activeModels_.end()) {
            iterator->second->unbind();
            idleModels_.push_back(std::move(iterator->second));
            activeModels_.erase(iterator);

            auto newSnapshot = std::make_shared<std::vector<SpartanModel*>>();
            newSnapshot->reserve(activeModels_.size());
            for (auto& modelEntry : activeModels_ | std::views::values) {
                newSnapshot->push_back(modelEntry.get());
            }
            tickSnapshot_ = newSnapshot;
        }
    }

    void SpartanModelRegistry::tickAll() {
        std::lock_guard lock(registryMutex_);
        auto snapshot = tickSnapshot_;
        lock.~lock_guard();  // Unlock before parallel processing

        if (!snapshot) return;

        // Single virtual call per model  -  Frontier A overhead is O(1) per agent.
        // Use parallel execution on supported platforms, sequential otherwise.
#if defined(__cpp_lib_execution) && !defined(__clang__)
        std::for_each(std::execution::par, snapshot->begin(), snapshot->end(),
            [](SpartanModel* model) {
                model->processTick();
            });
#else
        std::for_each(snapshot->begin(), snapshot->end(),
            [](SpartanModel* model) {
                model->processTick();
            });
#endif
    }

    bool SpartanModelRegistry::hasIdleModelAvailable() const noexcept {
        std::lock_guard lock(registryMutex_);
        return !idleModels_.empty();
    }

    std::unique_ptr<SpartanModel> SpartanModelRegistry::getIdleModelToRebind() noexcept {
        std::lock_guard lock(registryMutex_);

        if (idleModels_.empty()) {
            return nullptr;
        }

        auto model = std::move(idleModels_.back());
        idleModels_.pop_back();

        return model;
    }

    void SpartanModelRegistry::updateModelContext(const uint64_t agentIdentifier, const std::span<const double> newPtr) {
        std::lock_guard lock(registryMutex_);

        if (const auto it = activeModels_.find(agentIdentifier); it != activeModels_.end()) {
            it->second->setContextBuffer(newPtr);
        } else {
            logging::SpartanLogger::error(
                std::format("Failed to update context pointer: No active model found for agent ID {}", agentIdentifier));
        }
    }

    void SpartanModelRegistry::updateModelCleanSizes(const uint64_t agentIdentifier,
                                                      const std::span<const int32_t> cleanSizesBuffer) {
        std::lock_guard lock(registryMutex_);

        if (const auto it = activeModels_.find(agentIdentifier); it != activeModels_.end()) {
            it->second->setCleanSizesBuffer(cleanSizesBuffer);
        } else {
            logging::SpartanLogger::error(
                std::format("Failed to update clean sizes: No active model found for agent ID {}", agentIdentifier));
        }
    }

    void SpartanModelRegistry::distributeRewardsByIdentifier(
            const std::span<const uint64_t> agentIdentifiers,
            const std::span<const double> rewardSignals) {

        std::lock_guard lock(registryMutex_);

        const auto pairCount = static_cast<int32_t>(
            std::min(agentIdentifiers.size(), rewardSignals.size()));

        for (int32_t rewardIndex = 0; rewardIndex < pairCount; ++rewardIndex) {
            const uint64_t agentIdentifier = agentIdentifiers[rewardIndex];
            const double rewardSignal = rewardSignals[rewardIndex];

            const auto iterator = activeModels_.find(agentIdentifier);
            if (iterator == activeModels_.end()) {
                continue;
            }

            // Only SpartanAgent subclasses support reward-based learning.
            // The dynamic_cast is performed once per reward entry per tick.
            // This is the Frontier A boundary where dynamic dispatch overhead
            // is acceptable.
            if (auto* agentPointer = dynamic_cast<SpartanAgent*>(iterator->second.get())) {
                agentPointer->applyReward(rewardSignal);
            }
        }
    }

    void SpartanModelRegistry::decayExplorationForAgent(const uint64_t agentIdentifier) {
        std::lock_guard lock(registryMutex_);

        const auto iterator = activeModels_.find(agentIdentifier);
        if (iterator == activeModels_.end()) {
            return;
        }

        if (auto* agentPointer = dynamic_cast<SpartanAgent*>(iterator->second.get())) {
            agentPointer->decayExploration();
        }
    }

    SpartanModel* SpartanModelRegistry::getModel(const uint64_t agentIdentifier) {
        std::lock_guard lock(registryMutex_);
        const auto iterator = activeModels_.find(agentIdentifier);
        if (iterator != activeModels_.end()) {
            return iterator->second.get();
        }
        return nullptr;
    }

    bool SpartanModelRegistry::tickSingleAgent(const uint64_t agentIdentifier,
                                                const double rewardSignal) {
        std::lock_guard lock(registryMutex_);

        const auto iterator = activeModels_.find(agentIdentifier);
        if (iterator == activeModels_.end()) {
            return false;
        }

        SpartanModel* model = iterator->second.get();

        // Apply reward if the model supports reward-based learning.
        if (auto* agentPointer = dynamic_cast<SpartanAgent*>(model)) {
            agentPointer->applyReward(rewardSignal);
        }

        model->processTick();
        return true;
    }

    bool SpartanModelRegistry::saveModelToFile(const uint64_t agentIdentifier, const char* filePath) {
        std::lock_guard lock(registryMutex_);

        const auto iterator = activeModels_.find(agentIdentifier);
        if (iterator == activeModels_.end()) {
            logging::SpartanLogger::error(
                std::format("saveModelToFile: No active model found for agent ID {}", agentIdentifier));
            return false;
        }

        const SpartanModel* model = iterator->second.get();

        // Retrieve both weight buffers for full model persistence.
        const std::span<const double> modelWeightBlob = model->getModelWeights();
        const std::span<const double> criticWeightBlob = model->getCriticWeights();

        // Build TOC entries: one for model weights, one for critic weights (if present).
        persistence::SubModelTopologyEntry modelTopologyEntry{};
        modelTopologyEntry.subModelRole = persistence::SUB_MODEL_GAUSSIAN_POLICY;
        modelTopologyEntry.subModelIndex = 0;
        modelTopologyEntry.weightByteOffsetRelative = 0;
        modelTopologyEntry.weightElementCount = modelWeightBlob.size();
        modelTopologyEntry.biasesByteOffsetRelative = 0;
        modelTopologyEntry.biasElementCount = 0;

        std::vector<persistence::SubModelTopologyEntry> topologyEntries;
        topologyEntries.push_back(modelTopologyEntry);

        if (!criticWeightBlob.empty()) {
            persistence::SubModelTopologyEntry criticTopologyEntry{};
            criticTopologyEntry.subModelRole = persistence::SUB_MODEL_Q_CRITIC_FIRST;
            criticTopologyEntry.subModelIndex = 1;
            criticTopologyEntry.weightByteOffsetRelative =
                modelWeightBlob.size() * sizeof(double);
            criticTopologyEntry.weightElementCount = criticWeightBlob.size();
            criticTopologyEntry.biasesByteOffsetRelative = 0;
            criticTopologyEntry.biasElementCount = 0;
            topologyEntries.push_back(criticTopologyEntry);
        }

        // Concatenate model and critic weights into a single contiguous blob.
        // This allocation is acceptable because persistence is a cold-path operation.
        std::vector<double> concatenatedWeightBlob;
        concatenatedWeightBlob.reserve(modelWeightBlob.size() + criticWeightBlob.size());
        concatenatedWeightBlob.insert(concatenatedWeightBlob.end(),
            modelWeightBlob.begin(), modelWeightBlob.end());
        concatenatedWeightBlob.insert(concatenatedWeightBlob.end(),
            criticWeightBlob.begin(), criticWeightBlob.end());

        // Determine the model type from the base config for the file header.
        const auto* baseConfig = static_cast<const BaseHyperparameterConfig*>(
            model->getOpaqueHyperparameterConfig());
        const uint32_t modelType = baseConfig
            ? static_cast<uint32_t>(baseConfig->modelTypeIdentifier)
            : persistence::MODEL_TYPE_RECURRENT_SOFT_ACTOR_CRITIC;

        return persistence::saveModel(
            filePath,
            modelType,
            std::span<const persistence::SubModelTopologyEntry>(topologyEntries),
            std::span<const double>(concatenatedWeightBlob));
    }


}