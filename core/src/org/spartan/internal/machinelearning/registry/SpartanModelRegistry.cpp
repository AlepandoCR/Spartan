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

        bool foundModel = false;

        if (const auto it = activeModels_.find(agentIdentifier); it != activeModels_.end()) {
            foundModel = true;
            it->second->setContextBuffer(newPtr);
        }

        if (!foundModel) {
            logging::SpartanLogger::error(std::format("Failed to update context pointer: No active model found for agent ID {}", agentIdentifier));
        }
    }


}