//
// Created by Alepando on 10/3/2026.
//

#pragma once

#include <span>
#include <cstdint>
#include <vector>
#include <cstring>

#include "internal/math/metric/SpartanMetrics.h"

/**
 * @file RemorseTraceBuffer.h
 * @brief Ring buffer for temporal credit assignment via hidden-state similarity.
 *
 * Stores snapshots of the Gated Recurrent Unit hidden state and the action taken
 * at each tick.  When a global outcome event arrives (e.g., agent death), the
 * buffer is scanned to compute per-tick blame scores using cosine similarity
 * between the "blame" hidden state and all archived states.
 *
 * Memory is pre-allocated once during agent construction.  The ring buffer
 * overwrites the oldest entries when full, providing a natural adaptive
 * horizon that does not require a fixed lookback window.
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @struct RemorseTraceEntry
     * @brief Metadata for a single hidden-state snapshot in the ring buffer.
     */
    struct RemorseTraceEntry {
        /** @brief The tick number when this snapshot was recorded. */
        uint64_t tickNumber = 0;

        /** @brief The zero-based index of the action selected at this tick. */
        int32_t selectedActionIndex = 0;

        /**
         * @brief Offset (in doubles) into the flat hidden state archive where
         *        this entry's snapshot begins.
         */
        size_t hiddenStateArchiveOffset = 0;
    };

    /**
     * @class RemorseTraceBuffer
     * @brief Fixed-capacity ring buffer for temporal credit assignment.
     *
     * All memory is allocated once in the constructor.  No heap activity
     * occurs during recordSnapshot() or computeBlameScores().
     */
    class RemorseTraceBuffer {
    public:
        /**
         * @brief Constructs the buffer with a fixed capacity.
         *
         * @param maxEntryCount             Maximum number of ticks to remember.
         * @param hiddenStateDimensionSize  Number of doubles in each hidden state vector.
         */
        RemorseTraceBuffer(int32_t maxEntryCount, int32_t hiddenStateDimensionSize);

        ~RemorseTraceBuffer() = default;
        RemorseTraceBuffer(RemorseTraceBuffer&&) noexcept = default;
        RemorseTraceBuffer& operator=(RemorseTraceBuffer&&) noexcept = default;

        RemorseTraceBuffer(const RemorseTraceBuffer&) = delete;
        RemorseTraceBuffer& operator=(const RemorseTraceBuffer&) = delete;

        /**
         * @brief Records the current hidden state and action into the ring buffer.
         *
         * @param tickNumber           The current global tick number.
         * @param selectedActionIndex  The action that was chosen at this tick.
         * @param currentHiddenState   Read-only view of the Gated Recurrent Unit hidden state.
         */
        void recordSnapshot(uint64_t tickNumber,
                            int32_t selectedActionIndex,
                            std::span<const double> currentHiddenState);

        /**
         * @brief Computes per-entry blame scores using cosine similarity.
         *
         * @param blameHiddenState              The hidden state at the moment of the global outcome.
         * @param blameScoresOutput             Writable span for per-entry scores.
         * @param minimumSimilarityThreshold    Entries below this similarity receive zero blame.
         */
        void computeBlameScores(std::span<const double> blameHiddenState,
                                std::span<double> blameScoresOutput,
                                double minimumSimilarityThreshold) const;

        /** @brief Returns the number of valid entries currently stored. */
        [[nodiscard]] int32_t currentEntryCount() const noexcept { return currentCount_; }

        /** @brief Returns entry metadata at the given logical index. */
        [[nodiscard]] const RemorseTraceEntry& getEntry(int32_t index) const;

        /** @brief Returns a read-only view of the archived hidden state for the given entry. */
        [[nodiscard]] std::span<const double> getArchivedHiddenState(int32_t index) const;

        /**
         * @brief Returns a raw pointer to the archived hidden state for the given logical index.
         *
         * This avoids constructing a std::span object in tight loops where only the
         * pointer is needed (e.g., applyReward blame distribution).
         *
         * @param index Logical index (0 = oldest valid entry).
         * @return Pointer to the first element of the archived hidden state.
         */
        [[nodiscard]] inline const double* getArchivedHiddenStatePointer(int32_t index) const {
            const size_t archiveOffset = resolvePhysicalArchiveOffset(index);
            return &hiddenStateArchive_[archiveOffset];
        }

        /** @brief Clears all entries (e.g., at episode boundaries). */
        void reset() noexcept;

    private:
        /**
         * @brief Resolves the physical archive byte offset for a given logical index.
         *
         * Uses branchless wrap-around arithmetic instead of the modulo operator
         * to avoid the expensive integer division instruction in hot loops.
         *
         * @param logicalIndex The logical index (0 = oldest valid).
         * @return The byte offset into hiddenStateArchive_.
         */
        [[nodiscard]] inline size_t resolvePhysicalArchiveOffset(int32_t logicalIndex) const noexcept {
            int32_t physicalIndex;
            if (currentCount_ < maxEntryCount_) {
                physicalIndex = logicalIndex;
            } else {
                physicalIndex = writeHead_ + logicalIndex;
                if (physicalIndex >= maxEntryCount_) {
                    physicalIndex -= maxEntryCount_;
                }
            }
            return static_cast<size_t>(physicalIndex) * static_cast<size_t>(hiddenStateDimensionSize_);
        }

        std::vector<RemorseTraceEntry> entryRing_;
        std::vector<double> hiddenStateArchive_;

        /** @brief Pre-allocated buffer for active (above-threshold) entry indices during blame. */
        mutable std::vector<int32_t> activeEntryIndices_;

        int32_t maxEntryCount_;
        int32_t hiddenStateDimensionSize_;
        int32_t writeHead_ = 0;
        int32_t currentCount_ = 0;
    };

}

