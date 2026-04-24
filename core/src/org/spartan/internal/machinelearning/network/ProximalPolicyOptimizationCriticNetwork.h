#pragma once

#include <span>
#include <algorithm>
#include <cassert>
#include "../../math/tensor/SpartanTensorMath.h"
#include "internal/machinelearning/ModelHyperparameterConfig.h"

extern "C" {
    struct ProximalPolicyOptimizationHyperparameterConfig;
}

namespace org::spartan::internal::machinelearning::network {

    using namespace org::spartan::internal::math::tensor;

    class ProximalPolicyOptimizationCriticNetwork final {
    public:
        ProximalPolicyOptimizationCriticNetwork() = default;

        void initialize(
        const std::span<double> networkWeights,
        const std::span<double> networkBiases
        ) {
            networkWeights_ = networkWeights;
            networkBiases_ = networkBiases;
        }

        [[nodiscard]] double computeValueImpl(
                const std::span<const double> observationState,
                const void* opaqueConfig,
                const std::span<double> scratchpadA,
                const std::span<double> scratchpadB) const {

            auto* config = static_cast<const ProximalPolicyOptimizationHyperparameterConfig*>(opaqueConfig);
            const int inputSize = static_cast<int>(observationState.size());
            const int hiddenSize = config->criticHiddenNeuronCount;

            assert(scratchpadA.size() >= hiddenSize && "Scratchpad A too small");
            assert(scratchpadB.size() >= hiddenSize && "Scratchpad B too small");

            auto currentInput = scratchpadA.subspan(0, inputSize);
            std::ranges::copy(observationState, currentInput.begin());

            size_t weightOffset = 0;
            size_t biasOffset = 0;

            auto currentOutput = scratchpadB.subspan(0, hiddenSize);

            for (int layer = 0; layer < config->criticHiddenLayerCount; ++layer) {
                const int layerInputSize = (layer == 0) ? inputSize : hiddenSize;
                const std::span<const double> weights = networkWeights_.subspan(
                    weightOffset, hiddenSize * layerInputSize);
                const std::span<const double> biases = networkBiases_.subspan(
                    biasOffset, hiddenSize);

                TensorOps::denseForwardPass(currentInput, weights, biases, currentOutput);
                TensorOps::applyLeakyReLU(currentOutput, 0.01);

                currentInput = currentOutput;
                currentOutput = (layer % 2 == 1) ?
                    scratchpadA.subspan(0, hiddenSize) :
                    scratchpadB.subspan(0, hiddenSize);

                weightOffset += hiddenSize * layerInputSize;
                biasOffset += hiddenSize;
            }

            double finalValue = 0.0;
            auto finalOutputSpan = std::span(&finalValue, 1);

            const std::span<const double> finalWeights = networkWeights_.subspan(
                weightOffset, hiddenSize);
            const std::span<const double> finalBiases = networkBiases_.subspan(
                biasOffset, 1);

            TensorOps::denseForwardPass(currentInput, finalWeights, finalBiases, finalOutputSpan);

            return finalValue;
        }

        [[nodiscard]] std::span<const double> getWeights() const { return networkWeights_; }
        [[nodiscard]] std::span<const double> getBiases() const { return networkBiases_; }
        [[nodiscard]] std::span<double> getWeightsMutable() const { return networkWeights_; }
        [[nodiscard]] std::span<double> getBiasesMutable() const { return networkBiases_; }

    private:
        std::span<double> networkWeights_;
        std::span<double> networkBiases_;
    };

}


