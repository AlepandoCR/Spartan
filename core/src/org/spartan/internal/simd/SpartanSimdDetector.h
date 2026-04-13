//
// Created by Alepando on 12/4/2026.
//

#pragma once

#include <cstdint>
#include <string_view>

/**
 * @file SpartanSimdDetector.h
 * @brief Runtime CPU capability detection for SIMD optimizations.
 *
 * This module detects CPU capabilities at runtime (x86/x64 only via CPUID)
 * or compile-time (ARM64 via preprocessor). Provides a zero-cost abstraction
 * for querying supported SIMD instruction sets.
 */

namespace org::spartan::internal::simd {

    /**
     * @enum SimdCapability
     * @brief Enumeration of supported SIMD instruction sets in priority order.
     */
    enum class SimdCapability : uint32_t {
        SCALAR = 0,        ///< Fallback: no SIMD
        SSE42 = 1,         ///< Streaming SIMD Extensions 4.2 (x86/x64)
        AVX = 2,           ///< Advanced Vector Extensions (x86/x64)
        AVX2 = 3,          ///< Advanced Vector Extensions 2 (x86/x64)
        AVX512 = 4,        ///< Advanced Vector Extensions 512-bit (x86/x64)
        NEON = 5,          ///< ARM NEON (ARM64/Apple Silicon)
    };

    /**
     * @struct CpuCapabilities
     * @brief Aggregates detected CPU capabilities and metadata.
     */
    struct CpuCapabilities {
        /// Which SIMD instruction set was selected as optimal
        SimdCapability selectedCapability = SimdCapability::SCALAR;

        /// Individual capability flags
        bool hasSSE42 = false;
        bool hasAVX = false;
        bool hasAVX2 = false;
        bool hasAVX512F = false;  ///< AVX-512 Foundation
        bool hasAVX512DQ = false; ///< AVX-512 DQ extension

        /// For ARM64: detected at compile-time
        bool hasNEON = false;

        /// Architecture identifier
        std::string_view architecture;

        /// Number of SIMD lanes in optimal ISA
        int optimalLaneCount = 1;

        /**
         * @brief Detects CPU capabilities at runtime (x86/x64) or returns
         *        compile-time detected values (ARM64).
         *
         * This is a one-time operation; results are cached by the caller.
         *
         * @return CpuCapabilities structure with detected or compile-time values
         */
        static CpuCapabilities detect();

        /**
         * @brief Returns a human-readable name for the selected capability.
         */
        std::string_view capabilityName() const;

        /**
         * @brief Returns a human-readable name for the architecture.
         */
        std::string_view architectureName() const;
    };

    /**
     * @brief Detects and caches CPU capabilities (thread-safe singleton).
     *
     * Subsequent calls return the cached result without re-detecting.
     *
     * @return Reference to the global CpuCapabilities instance
     */
    CpuCapabilities& getDetectedCapabilities();

    /**
     * @brief Logs detected capabilities via SpartanLogger.
     *
     * @param capabilities The capabilities to log
     */
    void logDetectedCapabilities(const CpuCapabilities& capabilities);

}

