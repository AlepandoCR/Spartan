//
// Created by Alepando on 12/4/2026.
//

#include "SpartanSimdDispatcher.h"
#include "SpartanSimdOps.h"

namespace org::spartan::internal::simd {

    namespace {
        // Global state: cached capability decision
        SimdCapability g_selectedCapability = SimdCapability::SCALAR;
        int g_simdLaneCount = 1;
        bool g_initialized = false;
    }

    SimdCapability initializeSIMDDispatcher() {
        // Thread-safe initialization via static
        static SimdCapability initialized = [&]() {
            if (!g_initialized) {
                const auto& capabilities = getDetectedCapabilities();
                g_selectedCapability = capabilities.selectedCapability;
                g_simdLaneCount = capabilities.optimalLaneCount;
                g_initialized = true;

                // Log capabilities for debugging
                logDetectedCapabilities(capabilities);

                // Initialize SIMD operations dispatcher (runtime function pointer selection)
                initializeSimdOperations();
            }
            return g_selectedCapability;
        }();

        return initialized;
    }

    SimdCapability getSelectedSimdCapability() {
        // Ensure initialization
        if (!g_initialized) {
            initializeSIMDDispatcher();
        }
        return g_selectedCapability;
    }

    int getSimdLaneCount() {
        // Ensure initialization
        if (!g_initialized) {
            initializeSIMDDispatcher();
        }
        return g_simdLaneCount;
    }

}


