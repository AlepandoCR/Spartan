//
// Created by Alepando on 25/2/2026.
//

#include "SpartanModelRegistry.h"

#include <vector>
#include <execution>
#include <algorithm>
#include <ranges>

#include "internal/machinelearning/model/SpartanBaseModel.h"

namespace org::spartan::internal::machinelearning {

    void SpartanModelRegistry::registerModel(std::unique_ptr<SpartanBaseModel> model) {
        std::lock_guard lock(registryMutex_);
        activeModels_[model->getId()] = std::move(model);
    }

    void SpartanModelRegistry::unregisterModel(const uint64_t agentIdentifier) {
        std::lock_guard lock(registryMutex_);
        activeModels_.erase(agentIdentifier);
    }

    void SpartanModelRegistry::tickAll() {
        std::lock_guard lock(registryMutex_);

        // Extract raw pointers for iteration
        std::vector<SpartanBaseModel*> modelsToTick;
        modelsToTick.reserve(activeModels_.size());
        for (auto &val: activeModels_ | std::views::values) {
            modelsToTick.push_back(val.get());
        }

        // use parallel execution policy to process all models concurrently
        std::for_each(std::execution::par, modelsToTick.begin(), modelsToTick.end(),
            [&](SpartanBaseModel* model) {
                // TODO: calc the correct slice of the global state and reward buffers for this model based on its agentId and hyperparameters
                // model->tick;
            });
    }

    bool SpartanModelRegistry::hasIdleModelAvailable() const noexcept {
        std::lock_guard lock(registryMutex_);
        return !idleModels_.empty();
    }

    std::unique_ptr<SpartanBaseModel> SpartanModelRegistry::getIdleModelToRebind() noexcept {
        std::lock_guard lock(registryMutex_);

        if (idleModels_.empty()) {
            return nullptr;
        }

        auto model = std::move(idleModels_.back());
        idleModels_.pop_back();

        return model;
    }


} // namespace org::spartan::core::machinelearning