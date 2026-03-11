//
// Created by Alepando on 9/3/2026.
//

#include "AutoEncoderCompressorSpartanModel.h"

#include <algorithm>
#include <cstring>

#include "internal/math/tensor/SpartanTensorMath.h"

namespace org::spartan::internal::machinelearning {

    using math::tensor::TensorOps;

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
          latentBuffer_(latentBuffer) {

        const auto* config = typedConfig();
        if (!config) return;

        const int stateSize = config->baseConfig.stateSize;
        const int latentSize = config->latentDimensionSize;
        const int hiddenSize = config->encoderHiddenNeuronCount;

        // Pre-allocate forward pass scratchpads
        encoderHiddenActivation_.resize(hiddenSize);
        decoderHiddenActivation_.resize(hiddenSize);
        reconstructionBuffer_.resize(stateSize);

        // Pre-allocate gradient scratchpads
        reconstructionGradient_.resize(stateSize);
        encoderWeightGradients_.resize(encoderWeights.size());
        decoderWeightGradients_.resize(decoderWeights.size());
        encoderBiasGradients_.resize(encoderBiases.size());
        decoderBiasGradients_.resize(decoderBiases.size());
        inputGradientScratchpad_.resize(std::max({stateSize, hiddenSize, latentSize}));

        // Pre-allocate Adam optimizer state (zero-initialised by resize)
        encoderWeightMomentum_.resize(encoderWeights.size());
        encoderWeightVelocity_.resize(encoderWeights.size());
        encoderBiasMomentum_.resize(encoderBiases.size());
        encoderBiasVelocity_.resize(encoderBiases.size());

        decoderWeightMomentum_.resize(decoderWeights.size());
        decoderWeightVelocity_.resize(decoderWeights.size());
        decoderBiasMomentum_.resize(decoderBiases.size());
        decoderBiasVelocity_.resize(decoderBiases.size());
    }

    void AutoEncoderCompressorSpartanModel::processTick() {
        const auto* config = typedConfig();
        if (!config) return;

        const int stateSize = config->baseConfig.stateSize;
        const int latentSize = config->latentDimensionSize;
        const int hiddenSize = config->encoderHiddenNeuronCount;

        //
        // Phase A: Encoder forward pass.
        //   Input -> Dense(stateSize, hiddenSize) -> LeakyReLU -> Dense(hiddenSize, latentSize) -> Latent
        //

        // Layer 1: Input -> Hidden
        TensorOps::denseForwardPass(
            contextBuffer_,
            encoderWeights_.subspan(0, static_cast<size_t>(hiddenSize) * stateSize),
            encoderBiases_.subspan(0, hiddenSize),
            std::span(encoderHiddenActivation_));
        TensorOps::applyLeakyReLU(std::span(encoderHiddenActivation_), 0.01);

        // Layer 2: Hidden -> Latent (linear projection)
        const size_t encoderLayer2WeightOffset = static_cast<size_t>(hiddenSize) * stateSize;
        TensorOps::denseForwardPass(
            std::span<const double>(encoderHiddenActivation_),
            encoderWeights_.subspan(encoderLayer2WeightOffset, static_cast<size_t>(latentSize) * hiddenSize),
            encoderBiases_.subspan(hiddenSize, latentSize),
            latentBuffer_);

        //
        // Phase B: Copy latent representation to the action output buffer.
        //          This is what Java reads as the compressed observation.
        //
        std::memcpy(actionOutputBuffer_.data(), latentBuffer_.data(),
                     static_cast<size_t>(latentSize) * sizeof(double));

        //
        // Phase C (training only): Decoder forward pass + MSE loss + backward pass + Adam.
        //
        if (!config->baseConfig.isTraining) return;

        // Decoder Layer 1: Latent -> Hidden
        TensorOps::denseForwardPass(
            std::span<const double>(latentBuffer_.data(), latentSize),
            decoderWeights_.subspan(0, static_cast<size_t>(hiddenSize) * latentSize),
            decoderBiases_.subspan(0, hiddenSize),
            std::span(decoderHiddenActivation_));
        TensorOps::applyLeakyReLU(std::span(decoderHiddenActivation_), 0.01);

        // Decoder Layer 2: Hidden -> Reconstruction (linear)
        const size_t decoderLayer2WeightOffset = static_cast<size_t>(hiddenSize) * latentSize;
        TensorOps::denseForwardPass(
            std::span<const double>(decoderHiddenActivation_),
            decoderWeights_.subspan(decoderLayer2WeightOffset, static_cast<size_t>(stateSize) * hiddenSize),
            decoderBiases_.subspan(hiddenSize, stateSize),
            std::span(reconstructionBuffer_));

        //
        // Compute MSE loss and its gradient.
        //   Loss = mean((reconstruction - input)^2)
        //   Gradient = 2 * (reconstruction - input) / N
        //
        lastReconstructionLoss_ = TensorOps::computeMeanSquaredErrorLoss(
            std::span<const double>(reconstructionBuffer_),
            contextBuffer_);

        TensorOps::computeMeanSquaredErrorGradient(
            std::span<const double>(reconstructionBuffer_),
            contextBuffer_,
            std::span(reconstructionGradient_));

        //
        //  Backward pass through the decoder.
        //
        // Zero gradient accumulators
        std::ranges::fill(decoderWeightGradients_, 0.0);
        std::ranges::fill(decoderBiasGradients_, 0.0);

        // Bias gradient for decoder layer 2 = the reconstruction gradient (dL/dB = dL/dY)
        std::memcpy(decoderBiasGradients_.data() + hiddenSize,
                     reconstructionGradient_.data(),
                     static_cast<size_t>(stateSize) * sizeof(double));

        // Decoder Layer 2 backward: reconstruction gradient -> hidden gradient
        TensorOps::denseBackwardPass(
            std::span<const double>(decoderHiddenActivation_),
            std::span<const double>(reconstructionGradient_.data(), stateSize),
            decoderWeights_.subspan(decoderLayer2WeightOffset, static_cast<size_t>(stateSize) * hiddenSize),
            std::span(decoderWeightGradients_.data() + decoderLayer2WeightOffset,
                              static_cast<size_t>(stateSize) * hiddenSize),
            std::span(inputGradientScratchpad_.data(), hiddenSize));

        // Apply LeakyReLU derivative to the hidden gradient
        for (int neuronIndex = 0; neuronIndex < hiddenSize; ++neuronIndex) {
            if (decoderHiddenActivation_[neuronIndex] <= 0.0) {
                inputGradientScratchpad_[neuronIndex] *= 0.01;
            }
        }

        // Bias gradient for decoder layer 1 = hidden gradient after activation derivative
        std::memcpy(decoderBiasGradients_.data(),
                     inputGradientScratchpad_.data(),
                     static_cast<size_t>(hiddenSize) * sizeof(double));

        // Decoder Layer 1 backward: hidden gradient -> latent gradient
        TensorOps::denseBackwardPass(
            std::span<const double>(latentBuffer_.data(), latentSize),
            std::span<const double>(inputGradientScratchpad_.data(), hiddenSize),
            decoderWeights_.subspan(0, static_cast<size_t>(hiddenSize) * latentSize),
            std::span(decoderWeightGradients_.data(),
                              static_cast<size_t>(hiddenSize) * latentSize),
            std::span(inputGradientScratchpad_.data(), latentSize));

        // The latent gradient is now in inputGradientScratchpad_[0..latentSize)

        //
        // Backward pass through the encoder.
        //
        std::ranges::fill(encoderWeightGradients_, 0.0);
        std::ranges::fill(encoderBiasGradients_, 0.0);

        // Bias gradient for encoder layer 2 = the latent gradient
        std::memcpy(encoderBiasGradients_.data() + hiddenSize,
                     inputGradientScratchpad_.data(),
                     static_cast<size_t>(latentSize) * sizeof(double));

        // Encoder Layer 2 backward: latent gradient -> hidden gradient
        TensorOps::denseBackwardPass(
            std::span<const double>(encoderHiddenActivation_),
            std::span<const double>(inputGradientScratchpad_.data(), latentSize),
            encoderWeights_.subspan(encoderLayer2WeightOffset, static_cast<size_t>(latentSize) * hiddenSize),
            std::span(encoderWeightGradients_.data() + encoderLayer2WeightOffset,
                              static_cast<size_t>(latentSize) * hiddenSize),
            std::span(inputGradientScratchpad_.data(), hiddenSize));

        // Apply LeakyReLU derivative to the encoder hidden gradient
        for (int neuronIndex = 0; neuronIndex < hiddenSize; ++neuronIndex) {
            if (encoderHiddenActivation_[neuronIndex] <= 0.0) {
                inputGradientScratchpad_[neuronIndex] *= 0.01;
            }
        }

        // Bias gradient for encoder layer 1 = hidden gradient after activation derivative
        std::memcpy(encoderBiasGradients_.data(),
                     inputGradientScratchpad_.data(),
                     static_cast<size_t>(hiddenSize) * sizeof(double));

        // Encoder Layer 1 backward: hidden gradient -> input gradient (not used further)
        TensorOps::denseBackwardPass(
            contextBuffer_,
            std::span<const double>(inputGradientScratchpad_.data(), hiddenSize),
            encoderWeights_.subspan(0, static_cast<size_t>(hiddenSize) * stateSize),
            std::span(encoderWeightGradients_.data(),
                              static_cast<size_t>(hiddenSize) * stateSize),
            std::span(inputGradientScratchpad_.data(), stateSize));

        //
        //Apply Adam optimizer to all weights and biases simultaneously.
        //
        ++trainingStepCounter_;
        const double learningRate = config->baseConfig.learningRate;

        // Encoder weights
        TensorOps::applyAdamUpdate(
            encoderWeights_,
            std::span<const double>(encoderWeightGradients_),
            std::span(encoderWeightMomentum_),
            std::span(encoderWeightVelocity_),
            learningRate, 0.9, 0.999, 1e-8, trainingStepCounter_);

        // Encoder biases
        TensorOps::applyAdamUpdate(
            encoderBiases_,
            std::span<const double>(encoderBiasGradients_),
            std::span(encoderBiasMomentum_),
            std::span(encoderBiasVelocity_),
            learningRate, 0.9, 0.999, 1e-8, trainingStepCounter_);

        // Decoder weights
        TensorOps::applyAdamUpdate(
            decoderWeights_,
            std::span<const double>(decoderWeightGradients_),
            std::span(decoderWeightMomentum_),
            std::span(decoderWeightVelocity_),
            learningRate, 0.9, 0.999, 1e-8, trainingStepCounter_);

        // Decoder biases
        TensorOps::applyAdamUpdate(
            decoderBiases_,
            std::span<const double>(decoderBiasGradients_),
            std::span(decoderBiasMomentum_),
            std::span(decoderBiasVelocity_),
            learningRate, 0.9, 0.999, 1e-8, trainingStepCounter_);
    }

    std::span<const double> AutoEncoderCompressorSpartanModel::getLatentRepresentation() const {
        return latentBuffer_;
    }

    double AutoEncoderCompressorSpartanModel::getReconstructionLoss() const {
        return lastReconstructionLoss_;
    }

}

