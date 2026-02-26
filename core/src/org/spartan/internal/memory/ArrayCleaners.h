//
// Created by Alepando on 23/2/2026.
//

#pragma once

#include <span>
#include <vector>

/**
 * @namespace org::spartan::core::memory
 * @brief Zero-cost and high-speed memory utilities for FFM buffers.
 */
namespace org::spartan::internal::memory {

    /**
     * @class MemoryUtils
     * @brief Utility class for creating safe views and copies over JVM-managed memory.
     */
    class MemoryUtils {
    public:
        MemoryUtils() = delete;

        /**
         * @brief Creates a clean "window" over the dirty array. Takes 0 nanoseconds.
         * For immediate math, comparisons, and reading.
         *
         * @param rawBufferPointer Pointer to the raw JVM-managed memory buffer.
         * @param validElementCount Number of valid elements within the buffer.
         */
        static std::span<double> cleanView(double* rawBufferPointer, int validElementCount);

        /**
         * @brief Physically creates a new, perfectly sized array and copies the valid data.
         * To save the context.
         *
         * @param rawBufferPointer Pointer to the raw JVM-managed memory buffer (read-only).
         * @param validElementCount Number of valid elements to copy from the buffer.
         */
        static std::vector<double> cleanCopy(const double* rawBufferPointer, int validElementCount);
    };

}
