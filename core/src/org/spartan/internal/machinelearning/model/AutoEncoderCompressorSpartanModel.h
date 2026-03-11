//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>
#include <cstdint>
#include <vector>

#include "SpartanCompressor.h"
#include "../ModelHyperparameterConfig.h"

/**
 * @file AutoEncoderCompressorSpartanModel.h
 * @brief Concrete AutoEncoder compressor model  -  Frontier A facade.
 *
 * Exposes the standard @c SpartanCompressor virtual interface to the
 * registry while internally performing encoding / decoding passes
 * through Frontier-B static-dispatch components.
 *
 * The encoder compresses the high-dimensional @c contextBuffer_ into a
 * compact latent vector written to @c actionOutputBuffer_.  The decoder
 * reconstructs the input for loss computation during training.
 *
 * All weight/bias buffers are JVM-owned and mapped via @c std::span.
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class AutoEncoderCompressorSpartanModel
     * @brief Public-facing AutoEncoder compressor registered in the model registry.
     */
    class AutoEncoderCompressorSpartanModel final : public SpartanCompressor {
    public:
        /**
         * @brief Constructs the AutoEncoder model, binding all JVM-owned buffers.
         *
         * @param agentIdentifier              Unique agent ID.
         * @param opaqueHyperparameterConfig   Pointer to a JVM-owned
         *        @c AutoEncoderCompressorHyperparameterConfig.
         * @param modelWeights                 Span over the full weight buffer.
         * @param contextBuffer                Span over the observation input.
         * @param actionOutputBuffer           Span over the latent output.
         * @param encoderWeights               Span over the encoder weight buffer.
         * @param encoderBiases                Span over the encoder bias buffer.
         * @param decoderWeights               Span over the decoder weight buffer.
         * @param decoderBiases                Span over the decoder bias buffer.
         * @param latentBuffer                 Span over the bottleneck latent vector.
         */
        AutoEncoderCompressorSpartanModel(
                uint64_t agentIdentifier,
                void* opaqueHyperparameterConfig,
                std::span<double> modelWeights,
                std::span<const double> contextBuffer,
                std::span<double> actionOutputBuffer,
                std::span<double> encoderWeights,
                std::span<double> encoderBiases,
                std::span<double> decoderWeights,
                std::span<double> decoderBiases,
                std::span<double> latentBuffer);

        ~AutoEncoderCompressorSpartanModel() override = default;

        AutoEncoderCompressorSpartanModel(AutoEncoderCompressorSpartanModel&&) noexcept = default;
        AutoEncoderCompressorSpartanModel& operator=(AutoEncoderCompressorSpartanModel&&) noexcept = default;

        // Frontier A virtual interface (SpartanModel)
        void processTick() override;

        // Frontier A virtual interface (SpartanCompressor)
        [[nodiscard]] std::span<const double> getLatentRepresentation() const override;
        [[nodiscard]] double getReconstructionLoss() const override;

    private:
        [[nodiscard]] const AutoEncoderCompressorHyperparameterConfig* typedConfig() const noexcept {
            return static_cast<const AutoEncoderCompressorHyperparameterConfig*>(
                opaqueHyperparameterConfig_);
        }

        // Internal network buffers (non-owning, externally managed)
        std::span<double> encoderWeights_;
        std::span<double> encoderBiases_;
        std::span<double> decoderWeights_;
        std::span<double> decoderBiases_;
        std::span<double> latentBuffer_;

        /** @brief Cached reconstruction loss from the last training tick. */
        double lastReconstructionLoss_ = 0.0;

        /** @brief Training step counter for Adam bias correction. */
        int32_t trainingStepCounter_ = 0;

        // Pre-allocated scratchpads for forward/backward pass (allocated once in constructor)

        /** @brief Encoder hidden layer activation buffer. */
        std::vector<double> encoderHiddenActivation_;

        /** @brief Decoder hidden layer activation buffer. */
        std::vector<double> decoderHiddenActivation_;

        /** @brief Reconstructed output buffer (compared against contextBuffer_ for loss). */
        std::vector<double> reconstructionBuffer_;

        // Gradient scratchpads for backpropagation

        /** @brief MSE gradient at the reconstruction output layer. */
        std::vector<double> reconstructionGradient_;

        /** @brief Weight gradients for all encoder weights. */
        std::vector<double> encoderWeightGradients_;

        /** @brief Weight gradients for all decoder weights. */
        std::vector<double> decoderWeightGradients_;

        /** @brief Bias gradients for the encoder (one per encoder bias element). */
        std::vector<double> encoderBiasGradients_;

        /** @brief Bias gradients for the decoder (one per decoder bias element). */
        std::vector<double> decoderBiasGradients_;

        /** @brief Input gradient scratchpad (propagated back through each layer). */
        std::vector<double> inputGradientScratchpad_;

        // Adam optimizer state for encoder weights and biases

        /** @brief First moment (momentum) for encoder weights. */
        std::vector<double> encoderWeightMomentum_;
        /** @brief Second moment (velocity) for encoder weights. */
        std::vector<double> encoderWeightVelocity_;
        /** @brief First moment for encoder biases. */
        std::vector<double> encoderBiasMomentum_;
        /** @brief Second moment for encoder biases. */
        std::vector<double> encoderBiasVelocity_;

        // Adam optimizer state for decoder weights and biases

        /** @brief First moment (momentum) for decoder weights. */
        std::vector<double> decoderWeightMomentum_;
        /** @brief Second moment (velocity) for decoder weights. */
        std::vector<double> decoderWeightVelocity_;
        /** @brief First moment for decoder biases. */
        std::vector<double> decoderBiasMomentum_;
        /** @brief Second moment for decoder biases. */
        std::vector<double> decoderBiasVelocity_;
    };

}

