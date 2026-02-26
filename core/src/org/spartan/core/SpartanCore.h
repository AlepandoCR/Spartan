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
#include "machinelearning/ModelHyperparameterConfig.h"

/**
 * @file SpartanCore.h
 * @brief Top-level engine facade for the Spartan native core.
 *
 * SpartanEngine is the single root object that owns every subsystem
 * (model registry, math utilities, etc.).  The API layer (extern "C")
 * holds one static instance and delegates all domain logic here.
 */
namespace org::spartan::core {

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

        // Non-copyable / non-movable — owns the registry.
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
         * std::span views — **zero copy, zero allocation**.
         *
         * @param agentIdentifier       Unique 64-bit identifier for the agent.
         * @param hyperparameterConfig  Pointer to a JVM-owned config struct.
         * @param criticWeightsBuffer   Pointer to the critic's weight array.
         * @param criticWeightsCount    Number of doubles in @p criticWeightsBuffer.
         * @param modelWeightsBuffer    Pointer to the model's weight array.
         * @param modelWeightsCount     Number of doubles in @p modelWeightsBuffer.
         * @param actionOutputBuffer    Pointer to the action output array.
         * @param contextBuffer         Pointer to the context/state input array.
         * @param contextCount          Number of doubles in @p contextBuffer.
         * @param actionOutputCount     Number of doubles in @p actionOutputBuffer.
         */
        void registerAgent(uint64_t agentIdentifier,
                           ModelHyperparameterConfig* hyperparameterConfig,
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
         * @param globalRewardsBuffer Pointer to the flat reward array (JVM-owned).
         * @param globalRewardsCount  Number of doubles in @p globalRewardsBuffer.
         */
        void tickAllAgents(double* globalRewardsBuffer, int32_t globalRewardsCount);

    private:
        /** @brief The model registry owned by this engine instance. */
        machinelearning::SpartanModelRegistry modelRegistry_;
    };

} // namespace org::spartan::core

