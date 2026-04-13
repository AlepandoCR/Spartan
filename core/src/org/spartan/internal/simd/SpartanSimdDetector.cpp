//
// Created by Alepando on 12/4/2026.
//

#include "SpartanSimdDetector.h"
#include "../logging/SpartanLogger.h"
#include <format>

#if defined(_WIN32)
    #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
    // cpuid.h is only available on x86/x64 architectures
    #include <cpuid.h>
#endif

namespace org::spartan::internal::simd {

    std::string_view CpuCapabilities::capabilityName() const {
        switch (selectedCapability) {
            case SimdCapability::SCALAR:
                return "SCALAR (no SIMD)";
            case SimdCapability::SSE42:
                return "SSE4.2";
            case SimdCapability::AVX:
                return "AVX";
            case SimdCapability::AVX2:
                return "AVX2";
            case SimdCapability::AVX512:
                return "AVX-512";
            case SimdCapability::NEON:
                return "NEON";
            default:
                return "UNKNOWN";
        }
    }

    std::string_view CpuCapabilities::architectureName() const {
        return architecture;
    }

    CpuCapabilities CpuCapabilities::detect() {
        CpuCapabilities caps;

#if defined(__aarch64__) || defined(_M_ARM64)
        // ARM64: Runtime detection via NEON intrinsics availability
        // NEON is standard on ARMv8-A but we check if it's usable
        caps.hasNEON = true;  // Always available on ARM64
        caps.selectedCapability = SimdCapability::NEON;
        caps.optimalLaneCount = 2;  // 2 doubles in float64x2_t
        caps.architecture = "ARM64 (NEON, runtime selected)";

#elif (defined(_WIN32) || defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
         // x86/x64: Runtime CPUID detection
         caps.architecture = "x86/x64 (runtime CPUID detection)";

  #if defined(_WIN32)
         // Windows: Use __cpuid and __cpuidex
         int cpuInfo[4] = {-1};

         // Get vendor and max CPUID level
         __cpuid(cpuInfo, 0);
         int maxLevel = cpuInfo[0];

         // Basic capabilities (SSE4.2, AVX)
         __cpuid(cpuInfo, 1);
         caps.hasSSE42 = (cpuInfo[2] & (1 << 20)) != 0;  // ECX bit 20
         caps.hasAVX = (cpuInfo[2] & (1 << 28)) != 0;    // ECX bit 28

         // Extended capabilities (AVX2, AVX-512)
         if (maxLevel >= 7) {
             __cpuidex(cpuInfo, 7, 0);
             caps.hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;      // EBX bit 5
             caps.hasAVX512F = (cpuInfo[1] & (1 << 16)) != 0;   // EBX bit 16
             caps.hasAVX512DQ = (cpuInfo[1] & (1 << 17)) != 0;  // EBX bit 17
         }
  #elif defined(__GNUC__)
         // GCC/Clang: Use __cpuid and __cpuid_count
         unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

         // Get max CPUID level
         __cpuid(0, eax, ebx, ecx, edx);
         unsigned int maxLevel = eax;

         // Level 1: Basic capabilities
         __cpuid(1, eax, ebx, ecx, edx);
         caps.hasSSE42 = (ecx & (1 << 20)) != 0;
         caps.hasAVX = (ecx & (1 << 28)) != 0;

         // Level 7: Extended capabilities
         if (maxLevel >= 7) {
             __cpuid_count(7, 0, eax, ebx, ecx, edx);
             caps.hasAVX2 = (ebx & (1 << 5)) != 0;
             caps.hasAVX512F = (ebx & (1 << 16)) != 0;
             caps.hasAVX512DQ = (ebx & (1 << 17)) != 0;
         }
  #endif

          // Select optimal capability (in priority order)
          if (caps.hasAVX512F && caps.hasAVX512DQ) {
              caps.selectedCapability = SimdCapability::AVX512;
              caps.optimalLaneCount = 8;  // 8 doubles in __m512d
          } else if (caps.hasAVX2) {
              caps.selectedCapability = SimdCapability::AVX2;
              caps.optimalLaneCount = 4;  // 4 doubles in __m256d
          } else if (caps.hasAVX) {
              caps.selectedCapability = SimdCapability::AVX;
              caps.optimalLaneCount = 4;  // Still 4 doubles (128-bit for FP)
          } else if (caps.hasSSE42) {
              caps.selectedCapability = SimdCapability::SSE42;
              caps.optimalLaneCount = 2;  // 2 doubles in __m128d
          } else {
              caps.selectedCapability = SimdCapability::SCALAR;
               caps.optimalLaneCount = 1;
           }

#else
         // Unknown architecture: fallback to SCALAR
         caps.selectedCapability = SimdCapability::SCALAR;
         caps.optimalLaneCount = 1;
         caps.architecture = "Unknown (SCALAR fallback)";
#endif

        return caps;
    }

    CpuCapabilities& getDetectedCapabilities() {
        static CpuCapabilities cached = CpuCapabilities::detect();
        return cached;
    }

    void logDetectedCapabilities(const CpuCapabilities& capabilities) {
        using logging::SpartanLogger;

        SpartanLogger::info(std::format("[SIMD-DETECT] Architecture: {}", capabilities.architectureName()));
        SpartanLogger::info(std::format("[SIMD-DETECT] Selected SIMD: {} (lanes={})",
                                        capabilities.capabilityName(),
                                        capabilities.optimalLaneCount));

        if (capabilities.selectedCapability != SimdCapability::NEON &&
            capabilities.selectedCapability != SimdCapability::SCALAR) {
            SpartanLogger::debug(std::format(
                "[SIMD-DETECT] x86/x64 flags: SSE4.2={}, AVX={}, AVX2={}, AVX512F={}, AVX512DQ={}",
                capabilities.hasSSE42,
                capabilities.hasAVX,
                capabilities.hasAVX2,
                capabilities.hasAVX512F,
                capabilities.hasAVX512DQ
            ));
        }
    }

}






