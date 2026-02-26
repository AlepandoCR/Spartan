//
// Created by Alepando on 25/2/2026.
//

#include "../core/SpartanCore.h"

#include <cstdint>

using namespace org::spartan::core;

/** @brief Single engine instance for the lifetime of the shared library. */
static SpartanEngine engine;

extern "C" {

    // TODO: Add contracts once C++26 is supported by all major compilers.


    /**
     * @brief Initializes the Spartan native engine.
     *
     * This is the first function the JVM must invoke after loading the shared
     * library.  It performs any one-time global setup required by the engine.
     */
    __declspec(dllexport) void spartan_init() {
        SpartanEngine::log("Detected C++ Spartan Core...");
    }


    /**
     * @brief Logs a UTF-8 message through the Spartan console pipeline.
     *
     * @param message A null-terminated C string (UTF-8).
     */
    __declspec(dllexport) void spartan_log(const char* message) {
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
    __declspec(dllexport) long spartan_test_vector_union(double* targetFuzzySet,
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
     * passes raw pointers plus sizes.  All domain logic is delegated to
     * SpartanEngine — this function only validates inputs and logs errors.
     *
     * @param agentIdentifier       Unique 64-bit identifier for the agent.
     * @param hyperparameterConfig  Pointer to a JVM-owned ModelHyperparameterConfig.
     * @param criticWeightsBuffer   Pointer to the critic's weight array (JVM-owned).
     * @param criticWeightsCount    Number of doubles in @p criticWeightsBuffer.
     * @param modelWeightsBuffer    Pointer to the model's weight array (JVM-owned).
     * @param modelWeightsCount     Number of doubles in @p modelWeightsBuffer.
     * @param contextBuffer         Pointer to the context/state input array (JVM-owned).
     * @param contextCount          Number of doubles in @p contextBuffer.
     * @param actionOutputBuffer    Pointer to the action output array (JVM-owned).
     * @param actionOutputCount     Number of doubles in @p actionOutputBuffer.
     * @return 0 on success, -1 on invalid arguments.
     */
    __declspec(dllexport) int spartan_register_model(
            const uint64_t agentIdentifier,
            ModelHyperparameterConfig* hyperparameterConfig,
            double* criticWeightsBuffer,
            const int32_t criticWeightsCount,
            double* modelWeightsBuffer,
            const int32_t modelWeightsCount,
            double* contextBuffer,
            const int32_t contextCount,
            double* actionOutputBuffer,
            const int32_t actionOutputCount) {

        if (hyperparameterConfig == nullptr) {
            SpartanEngine::logError("spartan_register_model: hyperparameterConfig is null.");
            return -1;
        }
        if (criticWeightsBuffer == nullptr || criticWeightsCount <= 0) {
            SpartanEngine::logError("spartan_register_model: invalid critic weights buffer.");
            return -1;
        }
        if (modelWeightsBuffer == nullptr || modelWeightsCount <= 0) {
            SpartanEngine::logError("spartan_register_model: invalid model weights buffer.");
            return -1;
        }
        if (actionOutputBuffer == nullptr || actionOutputCount <= 0) {
            SpartanEngine::logError("spartan_register_model: invalid action output buffer.");
            return -1;
        }
        if (contextBuffer == nullptr || contextCount <= 0) {
            SpartanEngine::logError("spartan_register_model: invalid context buffer.");
            return -1;
        }

        engine.registerAgent(agentIdentifier,
                             hyperparameterConfig,
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
    __declspec(dllexport) int spartan_unregister_model(const uint64_t agentIdentifier) {
        engine.unregisterAgent(agentIdentifier);
        return 0;
    }

    /**
     * @brief Executes a global engine tick across every registered agent.
     *
     * @param globalRewardsBuffer  Pointer to the flat reward array (JVM-owned).
     * @param globalRewardsCount   Number of doubles in @p globalRewardsBuffer.
     * @return 0 on success, -1 on invalid input.
     */
    __declspec(dllexport) int spartan_tick_all(double* globalRewardsBuffer,
                                               const int32_t globalRewardsCount) {
        if (globalRewardsBuffer == nullptr || globalRewardsCount <= 0) {
            SpartanEngine::logError("spartan_tick_all: invalid global rewards buffer.");
            return -1;
        }

        engine.tickAllAgents(globalRewardsBuffer, globalRewardsCount);
        return 0;
    }

}