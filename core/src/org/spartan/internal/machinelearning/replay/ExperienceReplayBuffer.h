//
// Created by Alepando on 10/3/2026.
//

#pragma once

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <span>

/**
 * @file ExperienceReplayBuffer.h
 * @brief Pre-allocated circular buffer for off-policy experience replay.
 *
 * Stores (state, action, reward, nextState) transition tuples in a flat
 * contiguous memory layout. All memory is allocated once during construction.
 * The ring buffer uses a write head with branchless wrap-around (no modulo).
 *
 * Sampling uses a thread-local random number generator to avoid lock
 * contention during parallel agent training.
 *
 * @note This buffer does NOT own JVM memory. It copies transition data into
 *       its own pre-allocated storage so that past experiences remain valid
 *       even after the JVM updates the context/action buffers.
 */
namespace org::spartan::internal::machinelearning::replay {

    /**
     * @struct ExperienceTransition
     * @brief Metadata for a single stored transition in the replay buffer.
     *
     * The actual state/action data lives in flat arrays indexed by these offsets.
     */
    struct ExperienceTransition {
        /** @brief Byte offset into the state archive for this transition's state. */
        size_t stateArchiveOffset;

        /** @brief Byte offset into the state archive for this transition's next state. */
        size_t nextStateArchiveOffset;

        /** @brief Byte offset into the action archive for this transition's action. */
        size_t actionArchiveOffset;

        /** @brief The scalar reward received after taking the action. */
        double reward;

        /** @brief Whether this transition is terminal (episode ended). */
        bool isTerminal;
    };

    /**
     * @class ExperienceReplayBuffer
     * @brief Fixed-capacity ring buffer for storing and sampling experience transitions.
     */
    class ExperienceReplayBuffer {
    public:
        /**
         * @brief Constructs the replay buffer with pre-allocated flat storage.
         *
         * @param maxTransitionCount Maximum number of transitions to store.
         * @param stateDimensionSize Number of doubles per state vector.
         * @param actionDimensionSize Number of doubles per action vector.
         */
        ExperienceReplayBuffer(
                const int32_t maxTransitionCount,
                const int32_t stateDimensionSize,
                const int32_t actionDimensionSize)
            : maxTransitionCount_(maxTransitionCount),
              stateDimensionSize_(stateDimensionSize),
              actionDimensionSize_(actionDimensionSize) {

            transitionRing_.resize(maxTransitionCount);

            // Two state snapshots per transition: current state + next state
            stateArchive_.resize(
                static_cast<size_t>(maxTransitionCount) * 2 * stateDimensionSize);
            actionArchive_.resize(
                static_cast<size_t>(maxTransitionCount) * actionDimensionSize);
        }

        /** @brief Default constructor for deferred initialisation. */
        ExperienceReplayBuffer() = default;

        /**
         * @brief Stores a complete transition tuple in the ring buffer.
         *
         * Copies state and action data into the internal archive via memcpy.
         * Overwrites the oldest transition when the buffer is full.
         *
         * @param currentState  Read-only span of the state before the action.
         * @param action        Read-only span of the action taken.
         * @param reward        Scalar reward received.
         * @param nextState     Read-only span of the resulting state.
         * @param isTerminal    Whether the episode ended after this transition.
         */
        void storeTransition(
                const std::span<const double> currentState,
                const std::span<const double> action,
                const double reward,
                const std::span<const double> nextState,
                const bool isTerminal) {

            const size_t stateByteSize =
                static_cast<size_t>(stateDimensionSize_) * sizeof(double);
            const size_t actionByteSize =
                static_cast<size_t>(actionDimensionSize_) * sizeof(double);

            // Compute archive offsets for this slot
            const size_t stateOffset =
                static_cast<size_t>(writeHead_) * 2 * stateDimensionSize_;
            const size_t nextStateOffset = stateOffset + stateDimensionSize_;
            const size_t actionOffset =
                static_cast<size_t>(writeHead_) * actionDimensionSize_;

            // Copy transition data into flat archives
            std::memcpy(&stateArchive_[stateOffset], currentState.data(), stateByteSize);
            std::memcpy(&stateArchive_[nextStateOffset], nextState.data(), stateByteSize);
            std::memcpy(&actionArchive_[actionOffset], action.data(), actionByteSize);

            transitionRing_[writeHead_] = ExperienceTransition{
                .stateArchiveOffset = stateOffset,
                .nextStateArchiveOffset = nextStateOffset,
                .actionArchiveOffset = actionOffset,
                .reward = reward,
                .isTerminal = isTerminal
            };

            // Advance write head with branchless wrap-around
            ++writeHead_;
            if (writeHead_ >= maxTransitionCount_) {
                writeHead_ = 0;
            }
            if (currentCount_ < maxTransitionCount_) {
                ++currentCount_;
            }
        }

        /**
         * @brief Samples a batch of random transition indices without replacement.
         *
         * Uses a thread-local random generator for lock-free parallel sampling.
         * The output buffer must be pre-allocated by the caller.
         *
         * @param batchIndicesOutput Pre-allocated span to receive sampled indices.
         * @param batchSize          Number of transitions to sample.
         */
        void sampleBatchIndices(
                const std::span<int32_t> batchIndicesOutput,
                const int32_t batchSize) const {

            thread_local std::mt19937 generator(std::random_device{}());
            std::uniform_int_distribution distribution(0, currentCount_ - 1);

            const int32_t actualBatchSize = std::min(batchSize, currentCount_);
            for (int32_t sampleIndex = 0; sampleIndex < actualBatchSize; ++sampleIndex) {
                batchIndicesOutput[sampleIndex] = distribution(generator);
            }
        }

        /**
         * @brief Returns a read-only pointer to the state data for a given transition index.
         * @param transitionIndex Index into the transition ring.
         * @return Pointer to the first element of the archived state.
         */
        [[nodiscard]] inline const double* getStatePointer(const int32_t transitionIndex) const {
            return &stateArchive_[transitionRing_[transitionIndex].stateArchiveOffset];
        }

        /**
         * @brief Returns a read-only pointer to the next-state data for a given transition index.
         * @param transitionIndex Index into the transition ring.
         * @return Pointer to the first element of the archived next state.
         */
        [[nodiscard]] inline const double* getNextStatePointer(const int32_t transitionIndex) const {
            return &stateArchive_[transitionRing_[transitionIndex].nextStateArchiveOffset];
        }

        /**
         * @brief Returns a read-only pointer to the action data for a given transition index.
         * @param transitionIndex Index into the transition ring.
         * @return Pointer to the first element of the archived action.
         */
        [[nodiscard]] inline const double* getActionPointer(const int32_t transitionIndex) const {
            return &actionArchive_[transitionRing_[transitionIndex].actionArchiveOffset];
        }

        /**
         * @brief Returns the reward for a given transition index.
         * @param transitionIndex Index into the transition ring.
         * @return The scalar reward value.
         */
        [[nodiscard]] inline double getReward(const int32_t transitionIndex) const {
            return transitionRing_[transitionIndex].reward;
        }

        /**
         * @brief Returns whether the transition at the given index is terminal.
         * @param transitionIndex Index into the transition ring.
         * @return True if the episode ended after this transition.
         */
        [[nodiscard]] inline bool isTerminal(const int32_t transitionIndex) const {
            return transitionRing_[transitionIndex].isTerminal;
        }

        /** @brief Returns the number of transitions currently stored. */
        [[nodiscard]] int32_t currentTransitionCount() const noexcept { return currentCount_; }

        /** @brief Returns true if enough transitions are stored to fill a batch. */
        [[nodiscard]] bool hasEnoughTransitions(const int32_t requiredBatchSize) const noexcept {
            return currentCount_ >= requiredBatchSize;
        }

        /** @brief Returns the state dimension size. */
        [[nodiscard]] int32_t getStateDimensionSize() const noexcept { return stateDimensionSize_; }

        /** @brief Returns the action dimension size. */
        [[nodiscard]] int32_t getActionDimensionSize() const noexcept { return actionDimensionSize_; }

        /**
         * @brief Retroactively assigns a reward to the most recently stored transition.
         *
         * This is called by applyReward() after the transition was stored with
         * reward=0 during processTick(). The write head has already advanced,
         * so we access the slot immediately behind it.
         *
         * @param reward The scalar reward to assign.
         */
        void updateLatestTransitionReward(const double reward) {
            if (currentCount_ == 0) return;
            int32_t latestIndex = writeHead_ - 1;
            if (latestIndex < 0) latestIndex = maxTransitionCount_ - 1;
            transitionRing_[latestIndex].reward = reward;
        }

        /**
         * @brief Marks the most recently stored transition as terminal (episode ended).
         *
         * Called at episode boundaries so the Bellman backup knows not to
         * bootstrap the Q-value from the next state.
         */
        void markLatestTransitionTerminal() {
            if (currentCount_ == 0) return;
            int32_t latestIndex = writeHead_ - 1;
            if (latestIndex < 0) latestIndex = maxTransitionCount_ - 1;
            transitionRing_[latestIndex].isTerminal = true;
        }

    private:
        std::vector<ExperienceTransition> transitionRing_;
        std::vector<double> stateArchive_;
        std::vector<double> actionArchive_;

        int32_t maxTransitionCount_ = 0;
        int32_t stateDimensionSize_ = 0;
        int32_t actionDimensionSize_ = 0;
        int32_t writeHead_ = 0;
        int32_t currentCount_ = 0;
    };

}


