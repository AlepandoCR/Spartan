//
// Created by Alepando on 12/4/2026.
//

#include "SpartanSimdOps.h"
#include "../logging/SpartanLogger.h"
#include <format>
#include <mutex>
#include <atomic>

namespace org::spartan::internal::simd {

    namespace implementations {
        // Forward declarations for ISA-specific implementations.


        extern SimdOperations createAVX512Operations();
        extern SimdOperations createAVX2Operations();
        extern SimdOperations createScalarOperations();

        // NEON is only available on ARM platforms
        #if defined(__aarch64__) || defined(_M_ARM64)
            extern SimdOperations createNEONOperations();
        #endif
    }

    namespace {
        // Thread-safe globals with mutex protection and atomic flag
        std::mutex g_simd_mutex;
        std::atomic<bool> g_operations_initialized(false);
        SimdOperations g_selectedOperations;
    }

    SimdOperations& getSelectedSimdOperations() {
        // Lazy initialization with double-checked locking pattern
        // First check: avoid lock contention on happy path
        if (!g_operations_initialized.load(std::memory_order_acquire)) {
            initializeSimdOperations();
        }
        return g_selectedOperations;
    }

    void initializeSimdOperations() {
        // First check without lock - fast path
        if (g_operations_initialized.load(std::memory_order_acquire)) {
            return;
        }

        std::lock_guard<std::mutex> lock(g_simd_mutex);

        // Double-check inside lock: another thread might have initialized while we waited
        if (g_operations_initialized.load(std::memory_order_acquire)) {
            return;
        }

        const auto& capabilities = getDetectedCapabilities();

        using logging::SpartanLogger;

        // Select the best available implementation based on detected capabilities
        switch (capabilities.selectedCapability) {
            case SimdCapability::AVX512:
                SpartanLogger::info("[SIMD-OPS] Initializing AVX-512 operations");
                g_selectedOperations = implementations::createAVX512Operations();
                break;

            case SimdCapability::AVX2:
                SpartanLogger::info("[SIMD-OPS] Initializing AVX2 operations");
                g_selectedOperations = implementations::createAVX2Operations();
                break;

            case SimdCapability::NEON:
                #if defined(__aarch64__) || defined(_M_ARM64)
                    SpartanLogger::info("[SIMD-OPS] Initializing NEON operations");
                    g_selectedOperations = implementations::createNEONOperations();
                #else
                    SpartanLogger::warn("[SIMD-OPS] NEON requested but not compiled for this platform. Using SCALAR.");
                    g_selectedOperations = implementations::createScalarOperations();
                #endif
                break;

            case SimdCapability::SCALAR:
            default:
                SpartanLogger::info("[SIMD-OPS] Using SCALAR operations");
                g_selectedOperations = implementations::createScalarOperations();
                break;
        }

        // Validate that all function pointers are set
        if (g_selectedOperations.load == nullptr ||
            g_selectedOperations.store == nullptr ||
            g_selectedOperations.add == nullptr) {
            SpartanLogger::error("[SIMD-OPS] ERROR: Selected primitive operations have null function pointers!");
        } else {
            SpartanLogger::debug("[SIMD-OPS] All primitive operation pointers initialized successfully");
        }

        // Signal initialization complete with proper memory ordering
        g_operations_initialized.store(true, std::memory_order_release);
    }

}





