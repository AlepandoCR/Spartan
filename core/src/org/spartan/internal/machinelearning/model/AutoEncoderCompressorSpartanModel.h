//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>
#include <cstdint>

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
    };

}

