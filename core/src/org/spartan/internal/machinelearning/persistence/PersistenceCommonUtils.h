#pragma once

#include <span>
#include <vector>
#include <string>

namespace org::spartan::internal::machinelearning::persistence {

    /**
     * @class PersistenceCommonUtils
     * @brief Shared utilities for model persistence across all types.
     */
    class PersistenceCommonUtils {
    public:
        /**
         * Safely extracts a subspan with validation.
         *
         * @param source The source span
         * @param offset Starting offset in elements
         * @param count Number of elements to extract
         * @param label Debug label for error messages
         * @return Vector copy of the requested span
         * @throws std::out_of_range if bounds exceed source size
         */
        static std::vector<double> safeExtractSpan(
            const std::span<const double>& source,
            size_t offset,
            size_t count,
            const std::string& label);

        /**
         * Computes the size of a fully-connected layer weights + biases.
         *
         * @param inputSize Input neuron count
         * @param outputSize Output neuron count
         * @return Total number of doubles (weights + biases)
         */
        static size_t denseLayerTotalSize(int inputSize, int outputSize);

        /**
         * Computes SIMD-aligned size (64 bytes = 8 doubles).
         *
         * @param sizeInDoubles Unaligned size
         * @return Aligned size
         */
        static size_t computeSimdAlignedSize(size_t sizeInDoubles);

        /**
         * Removes SIMD padding from a weight blob (for file size optimization).
         *
         * @param weights Input weights with potential padding
         * @param originalCount Original element count before padding
         * @return Vector without SIMD padding
         */
        static std::vector<double> removeSimdPadding(
            const std::vector<double>& weights,
            size_t originalCount);

        /**
         * Adds SIMD padding back to weights (for reconstruction on load).
         *
         * @param weights Input weights without padding
         * @param targetAlignedCount Target aligned count
         * @return Vector with SIMD padding
         */
        static std::vector<double> addSimdPadding(
            const std::vector<double>& weights,
            size_t targetAlignedCount);
    };

}
