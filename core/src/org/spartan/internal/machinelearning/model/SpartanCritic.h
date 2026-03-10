//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>

#include "SpartanModel.h"

/**
 * @file SpartanCritic.h
 * @brief Intermediate abstract interface for value-estimation critic models.
 *
 * Critics are full models in the Spartan hierarchy  -  they own weight
 * buffers, receive context, and produce scalar value estimates as their
 * "action" output.  The registry can register, tick, and manage them
 * through the same @c SpartanModel virtual interface as any agent or
 * compressor.
 *
 * Concrete critic model families (e.g., a standalone Temporal Difference Critic deployed
 * as its own entity) inherit from @c SpartanCritic and implement
 * @c processTick() + @c evaluateState().
 *
 * @note When a critic is used as an **internal component** of a larger
 *       model (e.g., the twin Q-networks inside a Recurrent Soft Actor-Critic agent), the
 *       Frontier-B Curiously Recurring Template Pattern base @c ContinuousQNetwork in @c network/ is
 *       used instead  -  that path has zero virtual overhead.
 *       @c SpartanCritic is only for critics that are first-class
 *       registry citizens with their own lifecycle.
 *
 * @note Still part of **Frontier A** (dynamic polymorphism).
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class SpartanCritic
     * @brief Abstract interface for models whose primary output is a
     *        scalar value estimate V(s) or Q(s, a).
     *
     * Concrete critics must implement @c processTick() (from SpartanModel)
     * and @c evaluateState().
     */
    class SpartanCritic : public SpartanModel {
    public:
        ~SpartanCritic() override = default;

        // Non-copyable / move-only (inherited)
        SpartanCritic(const SpartanCritic&) = delete;
        SpartanCritic& operator=(const SpartanCritic&) = delete;
        SpartanCritic(SpartanCritic&&) noexcept = default;
        SpartanCritic& operator=(SpartanCritic&&) noexcept = default;

        //  Critic contract

        /**
         * @brief Estimates the value of the current observation state.
         *
         * @param observationState A read-only span over the state vector.
         * @return The estimated scalar value V(state).
         */
        [[nodiscard]] virtual double evaluateState(std::span<const double> observationState) const = 0;

        /**
         * @brief Performs a soft (Polyak) or hard parameter update towards
         *        a target network's weights.
         *
         * @param targetWeights Read-only span over the target weight buffer.
         * @param smoothingCoefficient Polyak averaging τ. 1.0 = hard copy.
         */
        virtual void softUpdateWeights(std::span<const double> targetWeights,
                                       double smoothingCoefficient) = 0;

    protected:
        /**
         * @brief Protected constructor  -  delegates to @c SpartanModel.
         *
         * @param agentIdentifier              Unique 64-bit agent identifier.
         * @param opaqueHyperparameterConfig   Opaque pointer to Java Virtual Machine-owned config struct.
         * @param modelWeights                 Span over the trainable-weight buffer.
         * @param contextBuffer                Span over the observation/state input buffer.
         * @param actionOutputBuffer           Span over the value output buffer.
         */
        SpartanCritic(const uint64_t agentIdentifier,
                      void* opaqueHyperparameterConfig,
                      const std::span<double> modelWeights,
                      const std::span<const double> contextBuffer,
                      const std::span<double> actionOutputBuffer)
            : SpartanModel(agentIdentifier,
                           opaqueHyperparameterConfig,
                           modelWeights,
                           contextBuffer,
                           actionOutputBuffer) {}

        /** @brief Default constructor for deferred initialisation via rebind(). */
        SpartanCritic() = default;
    };

}



