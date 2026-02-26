//
// Created by Alepando on 23/2/2026.
//

#include "ArrayCleaners.h"

#include <cstring>

namespace org::spartan::internal::memory {

    std::span<double> MemoryUtils::cleanView(double* rawBufferPointer, const int validElementCount) {
        // We trust the caller since the object comes from Java,
        // the JVM has control over the memory until the tick ends
        //NOLINTNEXTLINE
        return std::span(rawBufferPointer, validElementCount);
    }

    std::vector<double> MemoryUtils::cleanCopy(const double* rawBufferPointer, const int validElementCount) {
        std::vector<double> cleanArray(validElementCount);
        std::memcpy(cleanArray.data(), rawBufferPointer, validElementCount * sizeof(double));
        return cleanArray;
    }

}

