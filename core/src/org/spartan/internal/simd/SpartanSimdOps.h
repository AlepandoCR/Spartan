//
// Created by Alepando on 12/4/2026.
//

#pragma once

#include <span>
#include <cstdint>
#include "SpartanSimdDetector.h"

/**
 * @file SpartanSimdOps.h
 * @brief Runtime-dispatched PRIMITIVE SIMD operations.
 *
 * This layer provides basic vectorial operations (load, store, add, multiply, etc.)
 * selected at runtime based on detected CPU capabilities (via CPUID detection).
 */

namespace org::spartan::internal::simd {

    /**
     * @struct SimdFloat
     * @brief Runtime-selected SIMD float type.
     *
     * Opaque handle to the best available SIMD register type.
     * Size depends on detected capability (8 lanes for AVX-512, 4 for AVX2, 2 for NEON, 1 for SCALAR).
     */
    struct SimdFloat {
        // Opaque data - actual ISA-specific type is hidden at runtime
        alignas(64) double data[8];  // Max 8 doubles for AVX-512
    };

    /**
     * @struct SimdOperations
     * @brief Virtual table of PRIMITIVE vectorial operations selected at runtime.
     *
     * IMPORTANT: These are ONLY basic vector operations, NOT AI logic.
     * Machine Learning algorithms use these primitives.
     */
    struct SimdOperations {
        // Primitive load/store
        SimdFloat (*load)(const double* ptr) = nullptr;
        void (*store)(double* ptr, SimdFloat value) = nullptr;

        // Basic arithmetic
        SimdFloat (*add)(SimdFloat a, SimdFloat b) = nullptr;
        SimdFloat (*subtract)(SimdFloat a, SimdFloat b) = nullptr;
        SimdFloat (*multiply)(SimdFloat a, SimdFloat b) = nullptr;
        SimdFloat (*divide)(SimdFloat a, SimdFloat b) = nullptr;
        SimdFloat (*fusedMultiplyAdd)(SimdFloat mul1, SimdFloat mul2, SimdFloat add) = nullptr;

        // Min/max
        SimdFloat (*maximum)(SimdFloat a, SimdFloat b) = nullptr;
        SimdFloat (*minimum)(SimdFloat a, SimdFloat b) = nullptr;

        // Utility
        SimdFloat (*setZero)(void) = nullptr;
        SimdFloat (*broadcast)(double scalar) = nullptr;
        double (*horizontalSum)(SimdFloat value) = nullptr;

        // Advanced
        SimdFloat (*sqrt)(SimdFloat value) = nullptr;
        SimdFloat (*abs)(SimdFloat value) = nullptr;
        SimdFloat (*compareGreaterThan)(SimdFloat a, SimdFloat b) = nullptr;
        SimdFloat (*blend)(SimdFloat trueValue, SimdFloat falseValue, SimdFloat mask) = nullptr;
    };

    /**
     * @brief Gets the SIMD operations table selected for the current CPU.
     *
     * Returns function pointers to primitive operations based on runtime detection.
     * Cached after first initialization.
     *
     * @return Reference to the global SimdOperations instance
     */
    SimdOperations& getSelectedSimdOperations();

    /**
     * @brief Initializes the SIMD operations dispatcher.
     *
     * Should be called after initializeSIMDDispatcher().
     */
    void initializeSimdOperations();

}


