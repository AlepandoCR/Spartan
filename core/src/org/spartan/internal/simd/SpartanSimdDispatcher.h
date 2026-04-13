//
// Created by Alepando on 12/4/2026.
//

#pragma once

#include "SpartanSimdDetector.h"

/**
 * @file SpartanSimdDispatcher.h
 * @brief Runtime dispatcher for SIMD implementation selection.
 *
 * This module acts as a bridge between runtime detection and compile-time
 * SIMD implementations. It ensures that the correct SIMD backend is selected
 * based on detected CPU capabilities at library initialization time.
 *
 * Zero-cost abstraction via inline functions and template
 * specialization. The dispatcher makes one decision at startup; all subsequent
 * operations are direct calls with full inline optimization potential.
 */

namespace org::spartan::internal::simd {

    /**
     * @brief Initializes SIMD dispatcher at library startup.
     *
     * Should be called exactly once before any SIMD operations. Detects CPU
     * capabilities and logs results for debugging. Thread-safe via lazy static
     * initialization.
     *
     * @return The detected and selected SIMD capability
     */
    SimdCapability initializeSIMDDispatcher();

    /**
     * @brief Gets the currently selected SIMD capability.
     *
     * Returns the capability that was selected during dispatcher initialization.
     * Safe to call from any thread after initialization.
     *
     * @return The selected SimdCapability
     */
    SimdCapability getSelectedSimdCapability();

    /**
     * @brief Gets the number of SIMD lanes for optimal operations.
     *
     * Returns the number of double-precision values that can fit in the
     * optimal SIMD register for the current platform.
     *
     * @return Number of lanes (1 for SCALAR, 2 for NEON, 4 for AVX2, 8 for AVX-512)
     */
    int getSimdLaneCount();

}
