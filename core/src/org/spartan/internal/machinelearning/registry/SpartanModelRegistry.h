//
// Created by Alepando on 24/2/2026.
//

#pragma once

#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstdint>
#include <vector>

#include "internal/machinelearning/model/SpartanModel.h"

/**
 * @file SpartanModelRegistry.h
 * @brief Registry for managing active AI agents within the Spartan engine.
 */
namespace org::spartan::internal::machinelearning {

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

        // Non-copyable / non-movable  -  owns unique_ptrs internally.
        SpartanModelRegistry(const SpartanModelRegistry&) = delete;
        SpartanModelRegistry& operator=(const SpartanModelRegistry&) = delete;
        SpartanModelRegistry(SpartanModelRegistry&&) = delete;
        SpartanModelRegistry& operator=(SpartanModelRegistry&&) = delete;

        /**
         * @brief Registers a new model.
         *
         * C++ takes ownership of the C++ object, but **not** the underlying
         * memory buffers  -  the JVM retains ownership of those.
         *
         * @param model Unique pointer to the constructed model.
         */
        void registerModel(std::unique_ptr<SpartanModel> model);

        /**
         * @brief Unregisters a model
         *
         * @param agentIdentifier The unique identifier of the agent.
         */
        void unregisterModel(uint64_t agentIdentifier);

        /**
         * @brief Processes all agents in parallel.
         */
        void tickAll();

        /**
         * @brief Distributes reward signals to specific agents by identifier.
         *
         * Uses parallel arrays of agent identifiers and reward values.
         * For each pair, the corresponding model is looked up in the registry.
         * If the model exists and is a SpartanAgent, applyReward is invoked
         * with the associated reward signal. Non-agent models and unknown
         * identifiers are silently skipped.
         *
         * This approach is safe with unordered containers because distribution
         * is driven by explicit identifier lookup rather than iteration order.
         *
         * @param agentIdentifiers Read-only span over the JVM-owned agent ID array.
         * @param rewardSignals    Read-only span over the JVM-owned reward array.
         *                         Must have the same length as @p agentIdentifiers.
         */
        void distributeRewardsByIdentifier(std::span<const uint64_t> agentIdentifiers,
                                           std::span<const double> rewardSignals);

        /**
         * @brief Saves a model's weights to a .spartan binary file.
         *
         * @param agentIdentifier  The unique ID of the model to save.
         * @param filePath         Null-terminated path to the output file.
         * @return True on success, false if the model was not found or I/O failed.
         */
        bool saveModelToFile(uint64_t agentIdentifier, const char* filePath);

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
        [[nodiscard]] std::unique_ptr<SpartanModel> getIdleModelToRebind() noexcept;

        void updateModelContext(uint64_t agentIdentifier, std::span<const double> newPtr);

        /**
         * @brief Updates the clean sizes buffer for an agent's variable-length encoder slots.
         *
         * @param agentIdentifier   The unique ID of the agent.
         * @param cleanSizesBuffer  Read-only span over the JVM-owned int32_t array.
         */
        void updateModelCleanSizes(uint64_t agentIdentifier, std::span<const int32_t> cleanSizesBuffer);

        /**
         * @brief Triggers exploration decay for a specific agent.
         *
         * Looks up the model by identifier.  If the model is a SpartanAgent,
         * invokes its decayExploration() method.  Non-agent models and unknown
         * identifiers are silently skipped.
         *
         * @param agentIdentifier The unique ID of the agent.
         */
        void decayExplorationForAgent(uint64_t agentIdentifier);

        /**
         * @brief Applies a reward and executes a single tick for one specific agent.
         *
         * Locks the registry for lookup, applies the reward if the model is a
         * SpartanAgent, then calls processTick().  The JVM guarantees that the
         * same agent is never ticked concurrently from multiple threads.
         *
         * @param agentIdentifier The unique ID of the agent to tick.
         * @param rewardSignal    The scalar reward to apply before inference.
         * @return True if the agent was found and ticked, false otherwise.
         */
        bool tickSingleAgent(uint64_t agentIdentifier, double rewardSignal);
    private:
        /** @brief Guards concurrent access to the model map. */
        mutable std::mutex registryMutex_;

        /** @brief Map from agent identifier to its owned model instance. */
        std::unordered_map<uint64_t, std::unique_ptr<SpartanModel>> activeModels_;

        /** @brief Pool of idle models ready for reuse to minimize allocations. */
        std::vector<std::unique_ptr<SpartanModel>> idleModels_;

        /** @brief RCU snapshot of models for lock-free parallel ticking. */
        std::shared_ptr<std::vector<SpartanModel*>> tickSnapshot_;
    };

} // namespace org::spartan::core::machinelearning