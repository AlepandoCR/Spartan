//
// Created by Alepando on 25/2/2026.
//


#include <cstdint>
#include <cstring>


#if defined(_WIN32)
#define SPARTAN_API_EXPORT __declspec(dllexport)
#else
#define SPARTAN_API_EXPORT __attribute__((visibility("default")))
#endif

#include "internal/SpartanEngine.h"
#include "internal/machinelearning/ModelHyperparameterConfig.h"
#include "internal/machinelearning/persistence/PersistenceModuleRegistration.h"
#include "internal/simd/SpartanSimdDispatcher.h"

using namespace org::spartan::internal;

/** @brief Single engine instance for the lifetime of the shared library. */
static SpartanEngine engine;

namespace {
    uint32_t fnv1a(uint32_t hash, uint32_t value) {
        hash ^= value;
        return hash * 0x01000193u;
    }

    uint32_t compute_layout_signature() {
        uint32_t hash = 0x811C9DC5u;
        hash = fnv1a(hash, static_cast<uint32_t>(sizeof(BaseHyperparameterConfig)));
        hash = fnv1a(hash, static_cast<uint32_t>(sizeof(RecurrentSoftActorCriticHyperparameterConfig)));
        hash = fnv1a(hash, static_cast<uint32_t>(sizeof(CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig)));
        hash = fnv1a(hash, static_cast<uint32_t>(offsetof(RecurrentSoftActorCriticHyperparameterConfig, recurrentInputFeatureCount)));
        hash = fnv1a(hash, static_cast<uint32_t>(offsetof(RecurrentSoftActorCriticHyperparameterConfig, targetSmoothingCoefficient)));
        hash = fnv1a(hash, static_cast<uint32_t>(offsetof(CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig, forwardDynamicsHiddenLayerDimensionSize)));
        return hash;
    }
}

extern "C" {

    // TODO: Add contracts once C++26 is supported by all major compilers.


    /**
     * @brief Initializes the Spartan native engine.
     *
     * This is the first function the JVM must invoke after loading the shared
     * library.  It performs any one-time global setup required by the engine.
     */
    SPARTAN_API_EXPORT void spartan_init() {
        logging::SpartanLogger::setDebugEnabled(false);
        SpartanEngine::log("Detected C++ Spartan Core...");
        simd::initializeSIMDDispatcher();
        machinelearning::persistence::initializePersistenceModules();
    }


    /**
     * @brief Logs a UTF-8 message through the Spartan console pipeline.
     *
     * @param message A null-terminated C string (UTF-8).
     */
    SPARTAN_API_EXPORT void spartan_log(const char* message) {
        if (message == nullptr) {
            SpartanEngine::logError("Received null message pointer.");
            return;
        }
        SpartanEngine::log(message);
    }



    /**
     * @brief Computes the fuzzy-set union of two membership arrays in-place.
     *
     * @param targetFuzzySet  Pointer to the target membership array (modified in-place).
     * @param sourceFuzzySet  Pointer to the source membership array (read-only semantics).
     * @param targetSetSize   Number of elements in @p targetFuzzySet.
     * @param sourceSetSize   Number of elements in @p sourceFuzzySet.
     * @return Elapsed nanoseconds, or -1 on invalid input.
     */
    SPARTAN_API_EXPORT long spartan_test_vector_union(double* targetFuzzySet,
                                                         double* sourceFuzzySet,
                                                         const int targetSetSize,
                                                         const int sourceSetSize) {
        if (targetFuzzySet == nullptr || sourceFuzzySet == nullptr) {
            SpartanEngine::logError("Received null pointer for targetFuzzySet or sourceFuzzySet.");
            return -1;
        }
        if (targetSetSize <= 0 || sourceSetSize <= 0) {
            SpartanEngine::logError("Received non-positive size for targetFuzzySet or sourceFuzzySet.");
            return -1;
        }
        return SpartanEngine::computeFuzzySetUnion(targetFuzzySet, sourceFuzzySet,
                                                                       targetSetSize, sourceSetSize);
    }

    /**
     * @brief Registers a new ML agent in the engine's model registry.
     *
     * The JVM allocates all memory segments (hyperparameters, critic weights,
     * model weights, action buffer) via the Foreign Function & Memory API and
     * passes raw pointers plus sizes.  The config pointer is opaque (@c void*)
     * because each model type expects a different Standard Layout struct.
     *
     * @param agentIdentifier              Unique 64-bit identifier for the agent.
     * @param opaqueHyperparameterConfig   Opaque pointer to a JVM-owned config struct.
     * @param criticWeightsBuffer          Pointer to the critic's weight array (JVM-owned).
     * @param criticWeightsCount           Number of doubles in @p criticWeightsBuffer.
     * @param modelWeightsBuffer           Pointer to the model's weight array (JVM-owned).
     * @param modelWeightsCount            Number of doubles in @p modelWeightsBuffer.
     * @param contextBuffer                Pointer to the context/state input array (JVM-owned).
     * @param contextCount                 Number of doubles in @p contextBuffer.
     * @param actionOutputBuffer           Pointer to the action output array (JVM-owned).
     * @param actionOutputCount            Number of doubles in @p actionOutputBuffer.
     * @return 0 on success, -1 on invalid arguments.
     */
    SPARTAN_API_EXPORT int spartan_register_model(
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

        SpartanEngine::log(std::format(
            "spartan_register_model: id={}, cfg={}, criticPtr={}, criticCount={}, modelPtr={}, modelCount={}, ctxPtr={}, ctxCount={}, actionPtr={}, actionCount={}",
            agentIdentifier,
            reinterpret_cast<uintptr_t>(opaqueHyperparameterConfig),
            reinterpret_cast<uintptr_t>(criticWeightsBuffer), criticWeightsCount,
            reinterpret_cast<uintptr_t>(modelWeightsBuffer), modelWeightsCount,
            reinterpret_cast<uintptr_t>(contextBuffer), contextCount,
            reinterpret_cast<uintptr_t>(actionOutputBuffer), actionOutputCount));

        if (opaqueHyperparameterConfig == nullptr) {
            SpartanEngine::logError("spartan_register_model: opaqueHyperparameterConfig is null.");
            return -1;
        }
        if ((reinterpret_cast<uintptr_t>(opaqueHyperparameterConfig) % alignof(double)) != 0) {
            SpartanEngine::logError("spartan_register_model: opaqueHyperparameterConfig is misaligned.");
            return -1;
        }

        uint32_t providedSignature = 0;
        std::memcpy(&providedSignature,
                    reinterpret_cast<const uint8_t*>(opaqueHyperparameterConfig) + 60,
                    sizeof(providedSignature));
        const uint32_t expectedSignature = compute_layout_signature();
        if (providedSignature == 0) {
            logging::SpartanLogger::warn(
                "spartan_register_model: Config buffer not properly initialized (signature=0). "
                "This should not happen if SpartanModelAllocator.serialize() was called correctly. "
                "Proceeding with caution - config may contain garbage values.");
        } else if (providedSignature != expectedSignature) {
            SpartanEngine::logError(std::format(
                "spartan_register_model: layout signature mismatch (provided={}, expected={}).",
                providedSignature, expectedSignature));
            return -1;
        }

        engine.registerAgent(agentIdentifier,
                             opaqueHyperparameterConfig,
                             criticWeightsBuffer,
                             criticWeightsCount,
                             modelWeightsBuffer,
                             modelWeightsCount,
                             contextBuffer,
                             contextCount,
                             actionOutputBuffer,
                             actionOutputCount);
        return 0;
    }

    /**
     * @brief Removes an agent from the registry (e.g., entity death / disconnect).
     *
     * @param agentIdentifier The unique identifier of the agent to remove.
     * @return 0 on success (or if the agent was already absent).
     */
    SPARTAN_API_EXPORT int spartan_unregister_model(const uint64_t agentIdentifier) {
        engine.unregisterAgent(agentIdentifier);
        return 0;
    }

    /**
     * @brief Executes a global engine tick across every registered agent.
     *
     * Accepts parallel arrays that map each agent identifier to its
     * corresponding reward signal.  Rewards are distributed by explicit
     * identifier lookup before the parallel inference pass executes,
     * eliminating any dependency on the internal map iteration order.
     *
     * @param agentIdentifiersBuffer Pointer to the JVM-owned uint64_t agent ID array.
     * @param rewardSignalsBuffer    Pointer to the JVM-owned double reward array.
     * @param rewardEntryCount       Number of entries in both parallel arrays.
     * @return 0 on success, -1 on invalid input.
     */
    SPARTAN_API_EXPORT int spartan_tick_all(const uint64_t* agentIdentifiersBuffer,
                                               const double* rewardSignalsBuffer,
                                               const int32_t rewardEntryCount) {
        // Allow empty ticks (count=0) - just tick all agents without rewards
        if (rewardEntryCount <= 0) {
            engine.tickAllAgents(nullptr, nullptr, 0);
            return 0;
        }

        // For non-empty reward distribution, buffers must be valid
        if (agentIdentifiersBuffer == nullptr || rewardSignalsBuffer == nullptr) {
            SpartanEngine::logError("spartan_tick_all: received null pointer for agent identifiers or reward signals.");
            return -1;
        }

        engine.tickAllAgents(agentIdentifiersBuffer, rewardSignalsBuffer, rewardEntryCount);
        return 0;
    }


    SPARTAN_API_EXPORT void updateContextPointer(const uint64_t agentIdentifier, double* newPointer, const int newCapacity) {
        if (newPointer == nullptr || newCapacity <= 0) {
            SpartanEngine::logError("updateContextPointer: invalid new pointer or capacity.");
            return;
        }
        engine.updateContextPointer(agentIdentifier, newPointer, newCapacity);
    }

    /**
     * @brief Updates the clean sizes buffer for an active agent's variable-length context slices.
     *
     * Java writes one int32_t per nested encoder slot, indicating how many
     * elements in that slot are valid (non-padding) this tick. The C++ side
     * uses these sizes to create clean views that ignore dirty padding.
     *
     * @param agentIdentifier  The unique ID of the agent.
     * @param cleanSizesBuffer Pointer to the JVM-owned int32_t array.
     * @param slotCount        Number of int32_t entries in the buffer.
     */
    SPARTAN_API_EXPORT void spartan_update_clean_sizes(
            const uint64_t agentIdentifier,
            const int32_t* cleanSizesBuffer,
            const int32_t slotCount) {
        if (cleanSizesBuffer == nullptr || slotCount <= 0) {
            SpartanEngine::logError("spartan_update_clean_sizes: invalid buffer or count.");
            return;
        }
        engine.updateCleanSizes(agentIdentifier, cleanSizesBuffer, slotCount);
    }


    /**
     * @brief Saves a model's complete state (model weights + critic weights) to a .spartan file.
     *
     * Serializes both the model weight buffer and the critic/secondary weight buffer
     * into a single binary file with a 48-byte header, table of contents, concatenated
     * weight blob, and a trailing CRC-32 checksum.
     *
     * @param agentIdentifier The unique ID of the agent to save.
     * @param filePath        Null-terminated path to the output .spartan file.
     * @param modelTypeId     The model type (1=RSAC, 2=DDQN, 3=AutoEncoder, 4=Curiosity).
     * @return 0 on success, -1 on invalid input or I/O failure.
     */
    SPARTAN_API_EXPORT int spartan_save_model(const uint64_t agentIdentifier, const char* filePath, const uint32_t modelTypeId) {
        if (filePath == nullptr) {
            SpartanEngine::logError("spartan_save_model: received null file path.");
            return -1;
        }
        const bool success = engine.saveModel(agentIdentifier, filePath, modelTypeId);
        return success ? 0 : -1;
    }

    /**
     * @brief Loads model weights from a .spartan binary file into the given agent.
     *
     * Uses the model's specialized persistence module for deserialization.
     * The trailing CRC-32 checksum is verified without heap allocation.
     *
     * @param agentIdentifier     The unique ID of the target agent.
     * @param filePath            Null-terminated path to the input .spartan file.
     * @param modelTypeId         The model type (1=RSAC, 2=DDQN, 3=AutoEncoder, 4=Curiosity).
     * @return 0 on success, -1 on invalid input, CRC mismatch, or I/O failure.
     */
    SPARTAN_API_EXPORT int spartan_load_model(const uint64_t agentIdentifier, const char* filePath, const uint32_t modelTypeId) {
        if (filePath == nullptr) {
            SpartanEngine::logError("spartan_load_model: received null file path.");
            return -1;
        }

        const bool success = engine.loadModel(agentIdentifier, filePath, modelTypeId);
        return success ? 0 : -1;
    }

    /**
     * @brief Triggers exploration rate decay for a specific agent.
     *
     * Intended to be called at episode boundaries from the Java host.
     * For epsilon-greedy agents (Double Deep Q-Network), this decays epsilon.
     * For entropy-based agents (Recurrent Soft Actor-Critic), this decays the
     * base exploration parameter and resets the remorse trace buffer.
     *
     * @param agentIdentifier The unique ID of the agent.
     */
    SPARTAN_API_EXPORT void spartan_decay_exploration(const uint64_t agentIdentifier) {
        engine.decayExploration(agentIdentifier);
    }


    /**
     * @brief Applies a reward and executes a single inference tick for one agent.
     *
     * This is the event-driven counterpart to spartan_tick_all(). It applies
     * the reward first (if the model supports reward-based learning), then
     * executes processTick() to produce the next action.
     *
     * The JVM must guarantee that the same agent is not ticked concurrently
     * from multiple threads. Different agents may be ticked in parallel.
     *
     * @param agentIdentifier The unique ID of the agent to tick.
     * @param rewardSignal    The scalar reward to apply before inference.
     * @return 0 on success, -1 if the agent was not found in the registry.
     */
    SPARTAN_API_EXPORT int spartan_tick_agent(const uint64_t agentIdentifier,
                                                  const double rewardSignal) {
        const bool success = engine.tickAgent(agentIdentifier, rewardSignal);
        if (!success) {
            SpartanEngine::logError("spartan_tick_agent: no active model found for the given agent identifier.");
            return -1;
        }
        return 0;
    }

    /**
     * @brief Registers a multi-agent group with the Spartan engine.
     *
     * Creates a SpartanMultiAgentGroup in the engine's registry to manage
     * N homogeneous agents that share a single context buffer.
     *
     * @param multiAgentId      Unique identifier for the agent group
     * @param contextBuffer     Pointer to shared context buffer (Java-owned)
     * @param contextSize       Total size of context buffer (stateSize * N agents)
     * @param agentCount        Number of agents in the group
     * @return 0 on success, -1 on invalid input
     */
    SPARTAN_API_EXPORT int spartan_register_multi_agent(
            const uint64_t multiAgentId,
            double* contextBuffer,
            const int32_t contextSize,
            double* actionBuffer,
            const int32_t actionBufferSize,
            const int32_t actionFieldSize,
            const int32_t stateSize,
            const int32_t maxAgents) {

        if (contextBuffer == nullptr || contextSize <= 0 ||
            actionBuffer == nullptr || actionBufferSize <= 0 ||
            actionFieldSize <= 0 ||
            stateSize <= 0 || maxAgents <= 0) {
            SpartanEngine::logError("spartan_register_multi_agent: invalid parameters.");
            return -1;
        }

        engine.registerMultiAgentGroup(
            multiAgentId,
            contextBuffer,
            contextSize,
            actionBuffer,
            actionBufferSize,
            stateSize,
            actionFieldSize,
            maxAgents
        );

        SpartanEngine::log(std::format(
            "Registered multi-agent group {} with stateSize={}, actionFieldSize={}, maxAgents={}",
            multiAgentId, stateSize, actionFieldSize, maxAgents));

        return 0;
    }

    /**
     * @brief Adds an agent to an existing multi-agent group.
     *
     * Constructs a SpartanAgent internally within the group's context, binding it
     * to the appropriate subspans of the shared context and action buffers.
     *
     * @param multiAgentId       Unique identifier for the multi-agent group
     * @param agentIdentifier    Unique identifier for the new agent
     * @param opaqueConfig       Opaque pointer to the agent's hyperparameter config
     * @param modelWeights       Pointer to model weights buffer
     * @param modelWeightsCount  Size of model weights buffer
     * @param criticWeights      Pointer to critic weights buffer
     * @param criticWeightsCount Size of critic weights buffer
     * @return 0 on success, -1 on failure
     */
    SPARTAN_API_EXPORT int spartan_multi_agent_add_agent(
            const uint64_t multiAgentId,
            const uint64_t agentIdentifier,
            void* opaqueConfig,
            double* modelWeights,
            const int32_t modelWeightsCount,
            double* criticWeights,
            const int32_t criticWeightsCount) {

        if (opaqueConfig == nullptr) {
            SpartanEngine::logError("spartan_multi_agent_add_agent: opaqueConfig is null.");
            return -1;
        }

        const bool success = engine.addAgentToMultiAgentGroup(
            multiAgentId,
            agentIdentifier,
            opaqueConfig,
            modelWeights,
            modelWeightsCount,
            criticWeights,
            criticWeightsCount
        );

        if (!success) {
            SpartanEngine::logError(std::format(
                "spartan_multi_agent_add_agent: Failed to add agent {} to group {}",
                agentIdentifier, multiAgentId));
            return -1;
        }

        SpartanEngine::log(std::format(
            "Added agent {} to multi-agent group {}", agentIdentifier, multiAgentId));
        return 0;
    }

    /**
     * @brief Removes an agent from a multi-agent group.
     *
     * The agent is unbound from the group's shared buffers and destroyed.
     *
     * @param multiAgentId    Unique identifier for the multi-agent group
     * @param agentIdentifier Unique identifier for the agent to remove
     * @return 0 on success, -1 if group or agent not found
     */
    SPARTAN_API_EXPORT int spartan_multi_agent_remove_agent(
            const uint64_t multiAgentId,
            const uint64_t agentIdentifier) {

        const bool success = engine.removeAgentFromMultiAgentGroup(multiAgentId, agentIdentifier);
        if (!success) {
            SpartanEngine::logError(std::format(
                "spartan_multi_agent_remove_agent: Failed to remove agent {} from group {}",
                agentIdentifier, multiAgentId));
            return -1;
        }

        SpartanEngine::log(std::format(
            "Removed agent {} from multi-agent group {}", agentIdentifier, multiAgentId));
        return 0;
    }

    /**
     * @brief Executes a tick for all agents in a multi-agent group.
     *
     * Orchestrates MARL CTDE inference:
     * 1. All agents read their context subspans in parallel and perform inference
     * 2. Global critic evaluates joint state and joint actions
     * 3. Returns success/failure to caller
     *
     * @param multiAgentId  Unique identifier for the agent group
     * @return 0 on success, -1 if group not found
     */
    SPARTAN_API_EXPORT int spartan_tick_multi_agent(const uint64_t multiAgentId) {
        engine.tickMultiAgentGroup(multiAgentId);
        return 0;
    }

    /**
     * @brief Applies rewards to all agents in a multi-agent group.
     *
     * @param multiAgentId    Unique identifier for the multi-agent group
     * @param rewardsBuffer   Pointer to reward values
     * @param rewardCount     Number of reward values
     * @return 0 on success, -1 on failure
     */
    SPARTAN_API_EXPORT int spartan_multi_agent_apply_rewards(
            const uint64_t multiAgentId,
            const double* rewardsBuffer,
            const int32_t rewardCount) {

        if (rewardsBuffer == nullptr || rewardCount <= 0) {
            SpartanEngine::logError("spartan_multi_agent_apply_rewards: invalid parameters.");
            return -1;
        }

        engine.multiAgentApplyRewards(multiAgentId, rewardsBuffer, rewardCount);
        return 0;
    }

    /**
     * @brief Unregisters a multi-agent group from the engine.
     *
     * Removes and destroys the multi-agent group, freeing all associated resources
     * and agents within the group.
     *
     * @param multiAgentId Unique identifier for the multi-agent group to unregister
     * @return 0 on success, -1 if group not found
     */
    SPARTAN_API_EXPORT int spartan_unregister_multi_agent(const uint64_t multiAgentId) {
        engine.unregisterMultiAgentGroup(multiAgentId);
        return 0;
    }
}
