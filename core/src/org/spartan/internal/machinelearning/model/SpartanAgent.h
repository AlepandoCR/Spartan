//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>
#include <cstdlib>
#include <new>
#include <format>

#if defined(_WIN32)
#endif

#include "SpartanModel.h"

#include "internal/logging/SpartanLogger.h"

/**
 * @file SpartanAgent.h
 * @brief Intermediate abstract interface for decision-making Reinforcement Learning models.
 *
 * Models that interact with a reward signal (e.g., Recurrent Soft
 * Actor-Critic, Double Deep Q-Network) inherit from @c SpartanAgent.
 * The interface adds the reward-feedback contract on top of the base
 * @c SpartanModel tick lifecycle.
 *
 * @note Still part of **Frontier A** (dynamic polymorphism).
 * The @c processTick() call remains virtual so the registry can
 * invoke it uniformly, while all internal math should use
 * Curiously Recurring Template Pattern / static dispatch (Frontier B).
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class SpartanAgent
     * @brief Abstract interface for models that learn from scalar rewards.
     *
     * Concrete agents must implement @c processTick() (from SpartanModel),
     * @c applyReward(), and @c decayExploration().
     */
    class SpartanAgent : public SpartanModel {
    public:
        ~SpartanAgent() override = default;

        // ── Memory Alignment (AVX-512 Support) C++17/20/26 ───────────

        /**
         * @brief Fallback for non-over-aligned allocations.
         */
        void* operator new(size_t size) {
#if defined(_WIN32)
            void* ptr = _aligned_malloc(size, 64);
            if (!ptr) throw std::bad_alloc();
            return ptr;
#else
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 64, size) != 0) {
                throw std::bad_alloc();
            }
            return ptr;
#endif
        }

        /**
         * @brief Modern C++ allocator for strictly aligned types.
         *
         * C++26 automatically calls this overload when instantiating objects
         * with an alignas() requirement larger than __STDCPP_DEFAULT_NEW_ALIGNMENT__.
         */
        void* operator new(size_t size, std::align_val_t alignment) {
#if defined(_WIN32)
            void* ptr = _aligned_malloc(size, static_cast<size_t>(alignment));
            if (!ptr) throw std::bad_alloc();
            return ptr;
#else
            void* ptr = nullptr;
            if (posix_memalign(&ptr, static_cast<size_t>(alignment), size) != 0) {
                throw std::bad_alloc();
            }
            return ptr;
#endif
        }

        /**
         * @brief Fallback deallocator.
         */
        void operator delete(void* ptr) noexcept {
            if (!ptr) return;
            logging::SpartanLogger::info(std::format("freed memory at {}", reinterpret_cast<std::uintptr_t>(ptr)));
#if defined(_WIN32)
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }

        /**
         * @brief Modern C++ deallocator for strictly aligned types.
         */
        void operator delete(void* ptr, std::align_val_t) noexcept {
            if (!ptr) return;
#if defined(_WIN32)
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }

        // ── Non-copyable / move-only (inherited) ─────────────────────
        SpartanAgent(const SpartanAgent&) = delete;
        SpartanAgent& operator=(const SpartanAgent&) = delete;
        SpartanAgent(SpartanAgent&&) noexcept = default;
        SpartanAgent& operator=(SpartanAgent&&) noexcept = default;

        // Reward contract

        /**
         * @brief Applies a scalar reward signal and triggers a learning step.
         *
         * Called once per tick **after** @c processTick() has written the
         * action buffer.  The concrete implementation is expected to
         * update internal value estimates and/or policy weights.
         *
         * @param rewardSignal The scalar reward received from the environment.
         */
        virtual void applyReward(double rewardSignal) = 0;

        /**
         * @brief Decays the exploration rate according to the configured policy.
         *
         * Typically called at episode boundaries.  Concrete agents decide
         * the decay schedule (linear, exponential, etc.).
         */
        virtual void decayExploration() = 0;

    protected:
        /**
         * @brief Protected constructor  -  delegates to @c SpartanModel.
         *
         * @param agentIdentifier              Unique 64-bit agent identifier.
         * @param opaqueHyperparameterConfig   Opaque pointer to Java Virtual Machine-owned config struct.
         * @param modelWeights                 Span over the trainable-weight buffer.
         * @param contextBuffer                Span over the observation/state input buffer.
         * @param actionOutputBuffer           Span over the action output buffer.
         */
        SpartanAgent(const uint64_t agentIdentifier,
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
        SpartanAgent() = default;
    };

}