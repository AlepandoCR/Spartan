#pragma once
#include <chrono>
#include <string_view>
#include <print>
#include <span>


#include "math/fuzzy/SpartanFuzzyMath.h"
#include "memory/ArrayCleaners.h"

namespace org::spartan::core {

    class SpartanEngine {

    public:
        /**
         * Logs a message to the console using std::print.
         * * @param message The string view to log.
         */
        void log(std::string_view message) {
            std::println("[Spartan-Core] >> {}", message);
        }


        static long testVectorUnion(double* setA, double* setB, const int sizeA, const int sizeB) {
            auto start = std::chrono::high_resolution_clock::now();

            std::span<double> cleanA = memory::MemoryUtils::cleanView(setA, sizeA);
            std::span<double> cleanB = memory::MemoryUtils::cleanView(setB, sizeB);

            math::fuzzy::FuzzySetOps::unionSets(cleanA.data(), cleanB.data(), std::min(sizeA, sizeB));

            const auto end = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            return static_cast<long>(duration);
        }

    };

}
