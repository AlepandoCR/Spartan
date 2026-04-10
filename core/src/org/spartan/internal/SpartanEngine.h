//
// Created by Alepando on 25/2/2026.
//

#pragma once

#include <chrono>
#include <string_view>
#include <cstdint>
#include <memory>

#include "logging/SpartanLogger.h"
#include "machinelearning/registry/SpartanModelRegistry.h"
#include "machinelearning/registry/SpartanSlotMap.h"
#include "machinelearning/model/SpartanMultiAgentGroup.h"

/**
 * @file SpartanEngine.h
 * @brief Top-level engine facade for the Spartan native core.
 *
 * SpartanEngine is the single root object that owns every subsystem
 * (model registry, math utilities, etc.).  The API layer (extern "C")
 * holds one static instance and delegates all domain logic here.
 */
namespace org::spartan::internal {

    /**
     * @class SpartanEngine
     * @brief Central facade that owns all engine subsystems.
     *
     * All business logic lives here; the C API surface (SpartanApi.cpp)
     * is a thin validation / logging shim that forwards to this class.
     */
    class SpartanEngine {
    public:
        SpartanEngine() = default;
        ~SpartanEngine() = default;

        // Non-copyable / non-movable  -  owns the registry.
        SpartanEngine(const SpartanEngine&) = delete;
        SpartanEngine& operator=(const SpartanEngine&) = delete;
        SpartanEngine(SpartanEngine&&) = delete;
        SpartanEngine& operator=(SpartanEngine&&) = delete;


        /**
         * @brief Logs an informational message through the centralized logger.
         *
         * @param message The string view to log (UTF-8).
         */
        static void log(const std::string_view message) {
            logging::SpartanLogger::info(message);
        }

        /**
         * @brief Logs an error message through the centralized logger.
         *
         * @param message The error description (UTF-8).
         */
        static void logError(const std::string_view message) {
            logging::SpartanLogger::error(message);
        }


        /**
         * @brief Computes the fuzzy-set union of two membership arrays in-place.
         *
         * @param targetFuzzySet  Pointer to the target membership array (modified in-place).
         * @param sourceFuzzySet  Pointer to the source membership array (read-only semantics).
         * @param targetSetSize   Number of elements in @p targetFuzzySet.
         * @param sourceSetSize   Number of elements in @p sourceFuzzySet.
         * @return Elapsed nanoseconds.
         */
        static long computeFuzzySetUnion(double* targetFuzzySet,
                                         double* sourceFuzzySet,
                                         int targetSetSize,
                                         int sourceSetSize);


        /**
         * @brief Constructs and registers a new ML agent in the engine.
         *
         * All memory segments (hyperparameters, weights, action buffer) are
         * owned by the JVM.  The C++ side wraps them into non-owning
         * std::span views  -  **zero copy, zero allocation**.
         *
         * @param agentIdentifier              Unique 64-bit identifier for the agent.
         * @param opaqueHyperparameterConfig   Opaque pointer to a JVM-owned config struct.
         * @param criticWeightsBuffer          Pointer to the critic's weight array.
         * @param criticWeightsCount           Number of doubles in @p criticWeightsBuffer.
         * @param modelWeightsBuffer           Pointer to the model's weight array.
         * @param modelWeightsCount            Number of doubles in @p modelWeightsBuffer.
         * @param contextBuffer                Pointer to the context/state input array.
         * @param contextCount                 Number of doubles in @p contextBuffer.
         * @param actionOutputBuffer           Pointer to the action output array.
         * @param actionOutputCount            Number of doubles in @p actionOutputBuffer.
         */
        void registerAgent(uint64_t agentIdentifier,
                           void* opaqueHyperparameterConfig,
                           double* criticWeightsBuffer,
                           int32_t criticWeightsCount,
                           double* modelWeightsBuffer,
                           int32_t modelWeightsCount,
                           double* contextBuffer,
                           int32_t contextCount,
                           double* actionOutputBuffer,
                           int32_t actionOutputCount);

        /**
         * @brief Removes an agent from the registry.
         *
         * Releases the C++ model object but does **not** free the underlying
         * JVM-owned memory buffers.
         *
         * @param agentIdentifier The unique identifier of the agent to remove.
         */
        void unregisterAgent(uint64_t agentIdentifier);

        /**
         * @brief Executes a global engine tick across every registered agent.
         *
         * Distributes rewards to the correct agents by identifier lookup before
         * running the parallel inference pass.  Using explicit identifier arrays
         * avoids any dependency on the iteration order of the internal
         * @c unordered_map.
         *
         * @param agentIdentifiersBuffer Pointer to the JVM-owned agent ID array.
         * @param rewardSignalsBuffer    Pointer to the JVM-owned reward values array.
         * @param rewardEntryCount       Number of entries in both parallel arrays.
         */
        void tickAllAgents(const uint64_t* agentIdentifiersBuffer,
                           const double* rewardSignalsBuffer,
                           int32_t rewardEntryCount);

        void updateContextPointer(uint64_t agentIdentifier, double* newPointer, int newCapacity);

        /**
         * @brief Updates the clean sizes buffer for an agent's variable-length encoder slots.
         *
         * @param agentIdentifier   The unique ID of the agent.
         * @param cleanSizesBuffer  Pointer to the JVM-owned int32_t array.
         * @param slotCount         Number of entries.
         */
        void updateCleanSizes(uint64_t agentIdentifier, const int32_t* cleanSizesBuffer, int32_t slotCount);

        /**
         * @brief Saves a model's complete state to a .spartan binary file.
         *
         * @param agentIdentifier  The unique identifier of the model to save.
         * @param filePath         Null-terminated path to the output file.
         * @return True on success, false on failure.
         */
        bool saveModel(uint64_t agentIdentifier, const char* filePath);

        /**
         * @brief Loads model weights from a .spartan binary file into a pre-allocated buffer.
         *
         * @param filePath             Null-terminated path to the input .spartan file.
         * @param targetWeightBuffer   Pointer to the JVM-owned weight buffer to populate.
         * @param targetWeightCount    Number of doubles in the target buffer.
         * @return True on success and CRC-32 validation, false otherwise.
         */
        bool loadModel(uint64_t agentIdentifier, const char* filePath);

        /**
         * @brief Triggers exploration decay for a specific agent at episode boundaries.
         *
         * @param agentIdentifier The unique ID of the agent.
         */
        void decayExploration(uint64_t agentIdentifier);

        /**
         * @brief Applies a reward and executes a single inference tick for one agent.
         *
         * This is the event-driven counterpart to tickAllAgents(). The reward
         * is applied first (if the model is a SpartanAgent), then processTick()
         * is invoked. The JVM must guarantee that the same agent is not ticked
         * concurrently from multiple threads.
         *
         * @param agentIdentifier The unique ID of the agent to tick.
         * @param rewardSignal    The scalar reward to apply before inference.
         * @return True if the agent was found and ticked, false otherwise.
         */
        bool tickAgent(uint64_t agentIdentifier, double rewardSignal);

         /**
          * @brief Registers a new multi-agent group with shared buffers.
          *
          * @param groupIdentifier       Unique identifier for the group.
          * @param sharedContextBuffer   Pointer to the shared context buffer [stateSize * N].
          * @param sharedContextCount    Number of doubles in the shared context buffer.
          * @param sharedActionsBuffer   Pointer to the shared action buffer [actionSize * N].
          * @param sharedActionsCount    Number of doubles in the shared action buffer.
          * @param stateSize             State size per agent.
          * @param actionSize            Action size per agent.
          * @param maxAgents             Maximum number of agents in the group.
          */
         void registerMultiAgentGroup(uint64_t groupIdentifier,
                                      double* sharedContextBuffer,
                                      int32_t sharedContextCount,
                                      double* sharedActionsBuffer,
                                      int32_t sharedActionsCount,
                                      int32_t stateSize,
                                      int32_t actionSize,
                                      int32_t maxAgents);

         /**
          * @brief Adds an agent to an existing multi-agent group.
          *
          * @param groupIdentifier    Unique identifier for the group.
          * @param agentIdentifier    Unique identifier for the agent.
          * @param opaqueConfig       Opaque pointer to the agent's hyperparameter config.
          * @param modelWeights       Pointer to model weights buffer.
          * @param modelWeightsCount  Size of model weights buffer.
          * @param criticWeights      Pointer to critic weights buffer (optional/shared).
          * @param criticWeightsCount Size of critic weights buffer.
          * @return True on success, false if group not found or config invalid.
          */
         bool addAgentToMultiAgentGroup(uint64_t groupIdentifier,
                                        uint64_t agentIdentifier,
                                        void* opaqueConfig,
                                        double* modelWeights,
                                        int32_t modelWeightsCount,
                                        double* criticWeights,
                                        int32_t criticWeightsCount);

         /**
          * @brief Removes an agent from a multi-agent group.
          *
          * @param groupIdentifier Unique identifier for the group.
          * @param agentIdentifier Unique identifier for the agent.
          * @return True on success, false if group or agent not found.
          */
         bool removeAgentFromMultiAgentGroup(uint64_t groupIdentifier,
                                             uint64_t agentIdentifier);

         /**
          * @brief Executes a tick for a specific multi-agent group.
          *
          * @param groupIdentifier Unique identifier for the group.
          */
         void tickMultiAgentGroup(uint64_t groupIdentifier);

         /**
          * @brief Unregisters a multi-agent group and releases its C++ object.
          *
          * @param groupIdentifier Unique identifier for the group.
          */
         void unregisterMultiAgentGroup(uint64_t groupIdentifier);

         /**
          * @brief Applies rewards to all agents in a multi-agent group.
          *
          * @param groupIdentifier Unique identifier for the group.
          * @param rewardsBuffer   Pointer to the buffer containing sequential reward values.
          * @param rewardCount     Number of reward values in the buffer.
          */
         void multiAgentApplyRewards(uint64_t groupIdentifier, const double* rewardsBuffer, int32_t rewardCount);

          /**
          * @brief Saves a model to a .spartan file using its persistence module.
          *
          * @param agentIdentifier    The 64-bit identifier of the agent to save.
          * @param filePath           Null-terminated path to output .spartan file.
          * @param modelTypeId        The model type (1=RSAC, 2=DDQN, 3=AutoEncoder, 4=Curiosity).
          * @return True on success, false on failure.
          */
         bool saveModel(uint64_t agentIdentifier, const char* filePath, uint32_t modelTypeId);

         /**
          * @brief Loads a model from a .spartan file using its persistence module.
          *
          * @param agentIdentifier    The 64-bit identifier of the agent to load into.
          * @param filePath           Null-terminated path to input .spartan file.
          * @param modelTypeId        Expected model type (must match file header).
          * @return True on success, false on failure.
          */
         bool loadModel(uint64_t agentIdentifier, const char* filePath, uint32_t modelTypeId);

    private:
        /** @brief The model registry owned by this engine instance. */
        machinelearning::SpartanModelRegistry modelRegistry_;

        /** @brief The multi-agent registry owned by this engine instance. */
        machinelearning::SpartanSlotMap<std::unique_ptr<machinelearning::SpartanMultiAgentGroup>> multiAgentRegistry_;
    };

}

