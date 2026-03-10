//
// Created by Alepando on 9/3/2026.
//

#include "AutoEncoderCompressorSpartanModel.h"

namespace org::spartan::internal::machinelearning {

    AutoEncoderCompressorSpartanModel::AutoEncoderCompressorSpartanModel(
            const uint64_t agentIdentifier,
            void* opaqueHyperparameterConfig,
            std::span<double> modelWeights,
            std::span<const double> contextBuffer,
            std::span<double> actionOutputBuffer,
            std::span<double> encoderWeights,
            std::span<double> encoderBiases,
            std::span<double> decoderWeights,
            std::span<double> decoderBiases,
            std::span<double> latentBuffer)
        : SpartanCompressor(agentIdentifier,
                             opaqueHyperparameterConfig,
                             modelWeights,
                             contextBuffer,
                             actionOutputBuffer),
          encoderWeights_(encoderWeights),
          encoderBiases_(encoderBiases),
          decoderWeights_(decoderWeights),
          decoderBiases_(decoderBiases),
          latentBuffer_(latentBuffer) {}

    void AutoEncoderCompressorSpartanModel::processTick() {
        const auto* config = typedConfig();
        if (!config) {
            return;
        }

        // Encoder pass: contextBuffer_ -> latentBuffer_
        // TODO: Implement encoder forward pass (matrix multiply + activation).

        // Copy latent to action output (what Java reads)
        // TODO: Copy latentBuffer_ contents into actionOutputBuffer_.

        // Decoder pass (training only): latentBuffer_ -> reconstruction
        if (config->baseConfig.isTraining) {
            // TODO: Implement decoder forward pass.
            // TODO: Compute MSE reconstruction loss vs contextBuffer_.
            // lastReconstructionLoss_ = ...;
        }
    }

    std::span<const double> AutoEncoderCompressorSpartanModel::getLatentRepresentation() const {
        return latentBuffer_;
    }

    double AutoEncoderCompressorSpartanModel::getReconstructionLoss() const {
        return lastReconstructionLoss_;
    }

}

