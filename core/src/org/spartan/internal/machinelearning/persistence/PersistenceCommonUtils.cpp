#include "PersistenceCommonUtils.h"
#include "../../logging/SpartanLogger.h"
#include <stdexcept>
#include <algorithm>
#include <string>

namespace org::spartan::internal::machinelearning::persistence {

    std::vector<double> PersistenceCommonUtils::safeExtractSpan(
            const std::span<const double>& source,
            size_t offset,
            size_t count,
            const std::string& label) {

        if (offset + count > source.size()) {
            std::string error = label + ": Requested bounds [" + std::to_string(offset) +
                               ", " + std::to_string(offset + count) + ") exceed source size " +
                               std::to_string(source.size());
            logging::SpartanLogger::error("PersistenceCommonUtils::safeExtractSpan: " + error);
            throw std::out_of_range(error);
        }

        std::vector<double> result(source.begin() + offset, source.begin() + offset + count);
        return result;
    }

    size_t PersistenceCommonUtils::denseLayerTotalSize(int inputSize, int outputSize) {
        // weights: outputSize * inputSize
        // biases: outputSize
        return static_cast<size_t>(outputSize) * (static_cast<size_t>(inputSize) + 1);
    }

    size_t PersistenceCommonUtils::computeSimdAlignedSize(size_t sizeInDoubles) {
        // SIMD alignment: 64 bytes = 8 doubles
        constexpr size_t SIMD_LANE_COUNT = 8;
        return (sizeInDoubles + (SIMD_LANE_COUNT - 1)) & ~(SIMD_LANE_COUNT - 1);
    }

    std::vector<double> PersistenceCommonUtils::removeSimdPadding(
            const std::vector<double>& weights,
            size_t originalCount) {

        if (weights.size() < originalCount) {
            logging::SpartanLogger::error("PersistenceCommonUtils::removeSimdPadding: " +
                                        std::string("Weights size ") + std::to_string(weights.size()) +
                                        " is less than original count " + std::to_string(originalCount));
            return weights;
        }

        std::vector<double> result(weights.begin(), weights.begin() + originalCount);
        return result;
    }

    std::vector<double> PersistenceCommonUtils::addSimdPadding(
            const std::vector<double>& weights,
            size_t targetAlignedCount) {

        if (weights.size() > targetAlignedCount) {
            logging::SpartanLogger::error("PersistenceCommonUtils::addSimdPadding: " +
                                        std::string("Weights size ") + std::to_string(weights.size()) +
                                        " exceeds target aligned count " + std::to_string(targetAlignedCount));
            return weights;
        }

        std::vector<double> result = weights;
        result.resize(targetAlignedCount, 0.0);  // Pad with zeros
        return result;
    }

}
