#pragma once

#include <span>
#include <cstdint>
#include <vector>
#include "GaussianPolicyNetwork.h"
#include "../../math/tensor/SpartanTensorMath.h"
#include "internal/machinelearning/ModelHyperparameterConfig.h"

extern "C" {
    struct ProximalPolicyOptimizationHyperparameterConfig;
}

namespace org::spartan::internal::machinelearning::network {

    using namespace org::spartan::internal::math::tensor;

    class ProximalPolicyOptimizationActorNetwork final
        : public GaussianPolicyNetwork<ProximalPolicyOptimizationActorNetwork> {
    public:
        using GaussianPolicyNetwork::GaussianPolicyNetwork;

        void computePolicyOutputImpl(
                const std::span<const double> observationState,
                const void* opaqueConfig,
                const std::span<double> outMeans,
                const std::span<double> outLogStdDevs,
                const std::span<double> scratchpadA,
                const std::span<double> scratchpadB) const {

            auto* config = static_cast<const ProximalPolicyOptimizationHyperparameterConfig*>(opaqueConfig);
            const int inputSize = static_cast<int>(observationState.size());
            const int hiddenSize = config->actorHiddenNeuronCount;
            const int outputSize = config->baseConfig.actionSize;

            auto currentInput = scratchpadA.subspan(0, inputSize);
            std::ranges::copy(observationState, currentInput.begin());

            size_t weightOffset = 0;
            size_t biasOffset = 0;

            auto currentOutput = scratchpadB.subspan(0, hiddenSize);

            for (int layer = 0; layer < config->actorHiddenLayerCount; ++layer) {
                const int layerInputSize = (layer == 0) ? inputSize : hiddenSize;
                const std::span<const double> weights = this->policyWeights_.subspan(
                    weightOffset, hiddenSize * layerInputSize);
                const std::span<const double> biases = this->policyBiases_.subspan(
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

            const std::span<const double> meanWeights = this->policyWeights_.subspan(
                weightOffset, outputSize * hiddenSize);
            const std::span<const double> meanBiases = this->policyBiases_.subspan(
                biasOffset, outputSize);

            TensorOps::denseForwardPass(currentInput, meanWeights, meanBiases, outMeans);

            weightOffset += outputSize * hiddenSize;
            biasOffset += outputSize;

            const std::span<const double> stdWeights = this->policyWeights_.subspan(
                weightOffset, outputSize * hiddenSize);
            const std::span<const double> stdBiases = this->policyBiases_.subspan(
                biasOffset, outputSize);

            TensorOps::denseForwardPass(currentInput, stdWeights, stdBiases, outLogStdDevs);
        }
    };

}


