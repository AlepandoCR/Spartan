//
// Created by Alepando on 23/2/2026.
//

#ifndef SPARTAN_CORE_ARRAYCLEANERS_H
#define SPARTAN_CORE_ARRAYCLEANERS_H

#include <span>
#include <vector>
#include <cstring>

/**
 * @namespace org::spartan::core::memory
 * @brief Zero-cost and high-speed memory utilities for FFM buffers.
 */
namespace org::spartan::core::memory {

    class MemoryUtils {
    public:
        MemoryUtils() = delete;

        /**
         * @brief Creates a clean "window" over the dirty array. Takes 0 nanoseconds.
         *  For immediate math, comparisons, and reading.
         */
        static std::span<double> cleanView(double* dirtyArray, int validSize) {
            // We trust the caller since the object comes from Java,
            // the JVM has control over the memory until the tick ends
            //NOLINTNEXTLINE
            return std::span(dirtyArray, validSize);
        }

        /**
         * @brief Physically creates a new, perfectly sized array and copies the valid data.
         * to save the context
         */
        static std::vector<double> cleanCopy(const double* dirtyArray, int validSize) {

            std::vector<double> cleanArray(validSize);

            std::memcpy(cleanArray.data(), dirtyArray, validSize * sizeof(double));

            return cleanArray;
        }
    };

}
#endif //SPARTAN_CORE_ARRAYCLEANERS_H