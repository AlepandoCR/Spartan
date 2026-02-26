//
// Created by Alepando on 24/2/2026.
//

#pragma once

#include <unordered_map>
#include <memory>
#include <mutex>
#include <span>
#include <cstdint>
#include <optional>
#include <vector>

#include "core/machinelearning/model/SpartanBaseModel.h"

/**
 * @file SpartanModelRegistry.h
 * @brief Registry for managing active AI agents within the Spartan engine.
 */
namespace org::spartan::core::machinelearning {

    /**
     * @class SpartanModelRegistry
     * @brief Manages the lifecycle and parallel execution of ML models.
     *
     * A single instance is owned by
     * SpartanEngine (the top-level engine facade) and its lifetime is
     * bound to the engine's lifetime.  Copy and move are deleted because
     * the registry holds unique ownership of model objects.
     */
    class SpartanModelRegistry {
    public:
        SpartanModelRegistry() = default;
        ~SpartanModelRegistry() = default;

        // Non-copyable / non-movable — owns unique_ptrs internally.
        SpartanModelRegistry(const SpartanModelRegistry&) = delete;
        SpartanModelRegistry& operator=(const SpartanModelRegistry&) = delete;
        SpartanModelRegistry(SpartanModelRegistry&&) = delete;
        SpartanModelRegistry& operator=(SpartanModelRegistry&&) = delete;

        /**
         * @brief Registers a new model.
         *
         * C++ takes ownership of the C++ object, but **not** the underlying
         * memory buffers — the JVM retains ownership of those.
         *
         * @param model Unique pointer to the constructed model.
         */
        void registerModel(std::unique_ptr<SpartanBaseModel> model);

        /**
         * @brief Unregisters a model
         *
         * @param agentIdentifier The unique identifier of the agent.
         */
        void unregisterModel(uint64_t agentIdentifier);

        /**
         * @brief processes all agents in parallel.
         */
        void tickAll();

        /**
         * @brief Checks if there are any models in the idle pool ready for recycling.
         * @return true if a model can be reused without allocation.
         */
        [[nodiscard]] bool hasIdleModelAvailable() const noexcept;

        /**
        * @brief Attempts to retrieve a model from the idle pool.
        * @return A unique_ptr to the model, or nullptr if the pool is empty.
        * @warning The caller takes ownership and MUST rebind/re-register it.
        */
        [[nodiscard]] std::unique_ptr<SpartanBaseModel> getIdleModelToRebind() noexcept;
    private:
        /** @brief Guards concurrent access to the model map. */
        mutable std::mutex registryMutex_;

        /** @brief Map from agent identifier to its owned model instance. */
        std::unordered_map<uint64_t, std::unique_ptr<SpartanBaseModel>> activeModels_;

        /** @brief Pool of idle models ready for reuse to minimize allocations. */
        std::vector<std::unique_ptr<SpartanBaseModel>> idleModels_;
    };

} // namespace org::spartan::core::machinelearning