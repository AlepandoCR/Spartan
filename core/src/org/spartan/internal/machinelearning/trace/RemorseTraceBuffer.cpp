//
// Created by Alepando on 10/3/2026.
//

#include "RemorseTraceBuffer.h"

namespace org::spartan::internal::machinelearning {

    using math::metric::VectorMetrics;

    RemorseTraceBuffer::RemorseTraceBuffer(
            const int32_t maxEntryCount,
            const int32_t hiddenStateDimensionSize)
        : maxEntryCount_(maxEntryCount),
          hiddenStateDimensionSize_(hiddenStateDimensionSize) {

        // Pre-allocate all memory once. No further heap activity after this point.
        entryRing_.resize(maxEntryCount);
        hiddenStateArchive_.resize(
            static_cast<size_t>(maxEntryCount) * static_cast<size_t>(hiddenStateDimensionSize));
        activeEntryIndices_.resize(maxEntryCount);
    }

    void RemorseTraceBuffer::recordSnapshot(
            const uint64_t tickNumber,
            const int32_t selectedActionIndex,
            const std::span<const double> currentHiddenState) {

        const size_t archiveOffset =
            static_cast<size_t>(writeHead_) * static_cast<size_t>(hiddenStateDimensionSize_);

        // Write the entry metadata into the ring slot
        entryRing_[writeHead_] = RemorseTraceEntry{
            .tickNumber = tickNumber,
            .selectedActionIndex = selectedActionIndex,
            .hiddenStateArchiveOffset = archiveOffset
        };

        // Snapshot the hidden state into the pre-allocated archive via memcpy
        std::memcpy(
            &hiddenStateArchive_[archiveOffset],
            currentHiddenState.data(),
            static_cast<size_t>(hiddenStateDimensionSize_) * sizeof(double));

        // Advance the ring buffer write head using branchless wrap-around
        ++writeHead_;
        if (writeHead_ >= maxEntryCount_) {
            writeHead_ = 0;
        }
        if (currentCount_ < maxEntryCount_) {
            ++currentCount_;
        }
    }

    void RemorseTraceBuffer::computeBlameScores(
            const std::span<const double> blameHiddenState,
            const std::span<double> blameScoresOutput,
            const double minimumSimilarityThreshold) const {

        double totalBlameAccumulator = 0.0;
        int32_t activeCount = 0;

        // Pre-compute the physical start index once to avoid modulo per iteration.
        // When the buffer is not full, logical index maps directly to physical index.
        // When full, the oldest entry sits at writeHead_.
        int32_t physicalIndex = (currentCount_ < maxEntryCount_) ? 0 : writeHead_;

        const double* blameStatePointer = blameHiddenState.data();

        // Single pass: compute similarity, apply threshold gate, and record active indices.
        for (int32_t entryIndex = 0; entryIndex < currentCount_; ++entryIndex) {

            const size_t archiveOffset =
                static_cast<size_t>(physicalIndex) * static_cast<size_t>(hiddenStateDimensionSize_);

            const double* archivedStatePointer = &hiddenStateArchive_[archiveOffset];

            const double similarity = VectorMetrics::cosineSimilarity(
                blameStatePointer,
                archivedStatePointer,
                hiddenStateDimensionSize_);

            if (similarity >= minimumSimilarityThreshold) {
                blameScoresOutput[entryIndex] = similarity;
                totalBlameAccumulator += similarity;
                activeEntryIndices_[activeCount] = entryIndex;
                ++activeCount;
            } else {
                blameScoresOutput[entryIndex] = 0.0;
            }

            // Advance physical index with branchless wrap-around (1-2 cycles vs 20-40 for modulo)
            ++physicalIndex;
            if (physicalIndex >= maxEntryCount_) {
                physicalIndex = 0;
            }
        }

        // Normalize only the active entries to sum to 1.0.
        // This avoids iterating over entries that are already zero.
        if (totalBlameAccumulator > 0.0) {
            const double normalisationFactor = 1.0 / totalBlameAccumulator;
            for (int32_t activeIndex = 0; activeIndex < activeCount; ++activeIndex) {
                blameScoresOutput[activeEntryIndices_[activeIndex]] *= normalisationFactor;
            }
        }
    }

    const RemorseTraceEntry& RemorseTraceBuffer::getEntry(const int32_t index) const {
        int32_t physicalIndex;
        if (currentCount_ < maxEntryCount_) {
            physicalIndex = index;
        } else {
            physicalIndex = writeHead_ + index;
            if (physicalIndex >= maxEntryCount_) {
                physicalIndex -= maxEntryCount_;
            }
        }
        return entryRing_[physicalIndex];
    }

    std::span<const double> RemorseTraceBuffer::getArchivedHiddenState(const int32_t index) const {
        const size_t archiveOffset = resolvePhysicalArchiveOffset(index);
        return {
            &hiddenStateArchive_[archiveOffset],
            static_cast<size_t>(hiddenStateDimensionSize_)};
    }

    void RemorseTraceBuffer::reset() noexcept {
        writeHead_ = 0;
        currentCount_ = 0;
    }

}
