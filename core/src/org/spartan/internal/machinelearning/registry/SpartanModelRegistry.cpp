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
#include "internal/machinelearning/persistence/SpartanPersistence.h"

namespace org::spartan::internal::machinelearning {

    void SpartanModelRegistry::registerModel(std::unique_ptr<SpartanModel> model) {
        std::lock_guard lock(registryMutex_);
        activeModels_[model->getIdentifier()] = std::move(model);
    }

    void SpartanModelRegistry::unregisterModel(const uint64_t agentIdentifier) {
        std::lock_guard lock(registryMutex_);

        if (const auto iterator = activeModels_.find(agentIdentifier); iterator != activeModels_.end()) {
            iterator->second->unbind();
            idleModels_.push_back(std::move(iterator->second));
            activeModels_.erase(iterator);
        }
    }

    void SpartanModelRegistry::tickAll() {
        std::lock_guard lock(registryMutex_);

        // the map structure; each model's processTick() is thread-safe by contract
        std::vector<SpartanModel*> modelsToTick;
        modelsToTick.reserve(activeModels_.size());
        for (auto& modelEntry : activeModels_ | std::views::values) {
            modelsToTick.push_back(modelEntry.get());
        }

        // Single virtual call per model  -  Frontier A overhead is O(1) per agent.
        std::for_each(std::execution::par, modelsToTick.begin(), modelsToTick.end(),
            [](SpartanModel* model) {
                model->processTick();
            });
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

    void SpartanModelRegistry::applyGlobalRewards(const std::span<const double> globalRewardsBuffer) {
        std::lock_guard lock(registryMutex_);

        // Build a flat snapshot of agent pointers in stable iteration order.
        // The reward buffer index matches the insertion/iteration order of the map.
        int32_t rewardIndex = 0;
        const auto rewardCount = static_cast<int32_t>(globalRewardsBuffer.size());

        for (auto& modelEntry : activeModels_ | std::views::values) {
            if (rewardIndex >= rewardCount) break;

            // Only SpartanAgent subclasses support reward-based learning.
            // The dynamic_cast is done once per tick per agent  -  this is the
            // Frontier A boundary where dynamic dispatch overhead is acceptable.
            if (auto* agentPointer = dynamic_cast<SpartanAgent*>(modelEntry.get())) {
                const double rewardSignal = globalRewardsBuffer[rewardIndex];
                agentPointer->applyReward(rewardSignal);
            }
            ++rewardIndex;
        }
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

        // The model's full weight buffer (modelWeights_) contains all trainable
        // parameters in a flat contiguous layout. We save the entire blob.
        const std::span<const double> weightBlob = model->getModelWeights();

        // For now we write a single TOC entry representing the full model.
        // Future phases will decompose into sub-model entries.
        persistence::SubModelTopologyEntry topologyEntry{};
        topologyEntry.subModelRole = persistence::SUB_MODEL_GAUSSIAN_POLICY;
        topologyEntry.subModelIndex = 0;
        topologyEntry.weightByteOffsetRelative = 0;
        topologyEntry.weightElementCount = weightBlob.size();
        topologyEntry.biasesByteOffsetRelative = 0;
        topologyEntry.biasElementCount = 0;

        const auto entries = std::span<const persistence::SubModelTopologyEntry>(&topologyEntry, 1);

        return persistence::saveModel(
            filePath,
            persistence::MODEL_TYPE_RECURRENT_SOFT_ACTOR_CRITIC,
            entries,
            weightBlob);
    }


}