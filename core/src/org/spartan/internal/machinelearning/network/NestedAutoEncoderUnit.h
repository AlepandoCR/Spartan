//
// Created by Alepando on 10/3/2026.
//

#pragma once

#include <span>
#include <cstdint>

#include "internal/math/tensor/SpartanTensorMath.h"

/**
 * @file NestedAutoEncoderUnit.h
 * @brief Lightweight CRTP-based encoder for compressing variable-context slices.
 *
 * Each NestedAutoEncoderUnit is a Frontier B component that lives entirely
 * inside a parent agent (e.g., RecurrentSoftActorCriticSpartanModel).
 * Java never sees these units.  They compress a slice of the context buffer
 * into a compact latent vector using a single dense layer with Leaky ReLU
 * activation, followed by a projection to the latent dimension.
 *
 * The unit does NOT own memory.  All weight, bias, and latent buffers are
 * non-owning std::span views over regions of the parent agent's pre-allocated
 * weight pool.  Zero allocation in the hot path.
 *
 * Architecture per encoder:
 *   Input Slice -> Dense(inputDim, hiddenDim) -> LeakyReLU -> Dense(hiddenDim, latentDim) -> Latent
 *
 * During training, a mirror decoder reconstructs the input for loss:
 *   Latent -> Dense(latentDim, hiddenDim) -> LeakyReLU -> Dense(hiddenDim, inputDim) -> Reconstruction
 */
namespace org::spartan::internal::machinelearning {

    using math::tensor::TensorOps;

    /**
     * @class NestedAutoEncoderUnit
     * @brief Single-layer encoder unit with non-owning buffer views.
     *
     * Constructed once during agent initialisation and reused every tick.
     */
    class NestedAutoEncoderUnit {
    public:
        /**
         * @brief Constructs the encoder unit with pre-sliced buffer views.
         *
         * @param encoderHiddenWeights  Span over encoder input-to-hidden weight matrix.
         *                               Size: hiddenNeuronCount * inputDimensionSize.
         * @param encoderHiddenBiases   Span over encoder hidden bias vector.
         *                               Size: hiddenNeuronCount.
         * @param encoderLatentWeights  Span over encoder hidden-to-latent weight matrix.
         *                               Size: latentDimensionSize * hiddenNeuronCount.
         * @param encoderLatentBiases   Span over encoder latent bias vector.
         *                               Size: latentDimensionSize.
         * @param decoderHiddenWeights  Span over decoder latent-to-hidden weight matrix.
         *                               Size: hiddenNeuronCount * latentDimensionSize.
         * @param decoderHiddenBiases   Span over decoder hidden bias vector.
         *                               Size: hiddenNeuronCount.
         * @param decoderOutputWeights  Span over decoder hidden-to-output weight matrix.
         *                               Size: inputDimensionSize * hiddenNeuronCount.
         * @param decoderOutputBiases   Span over decoder output bias vector.
         *                               Size: inputDimensionSize.
         * @param latentBuffer          Span over the latent representation output.
         *                               Size: latentDimensionSize.
         * @param hiddenScratchpad      Span over working memory for hidden activations.
         *                               Size: hiddenNeuronCount.
         * @param reconstructionScratchpad Span over working memory for decoder output.
         *                               Size: inputDimensionSize.
         * @param inputDimensionSize    Number of doubles in the context slice.
         * @param latentDimensionSize   Number of doubles in the compressed output.
         * @param hiddenNeuronCount     Number of neurons in the hidden layer.
         */
        NestedAutoEncoderUnit(
                const std::span<double> encoderHiddenWeights,
                const std::span<double> encoderHiddenBiases,
                const std::span<double> encoderLatentWeights,
                const std::span<double> encoderLatentBiases,
                const std::span<double> decoderHiddenWeights,
                const std::span<double> decoderHiddenBiases,
                const std::span<double> decoderOutputWeights,
                const std::span<double> decoderOutputBiases,
                const std::span<double> latentBuffer,
                const std::span<double> hiddenScratchpad,
                const std::span<double> reconstructionScratchpad,
                const int32_t inputDimensionSize,
                const int32_t latentDimensionSize,
                const int32_t hiddenNeuronCount)
            : encoderHiddenWeights_(encoderHiddenWeights),
              encoderHiddenBiases_(encoderHiddenBiases),
              encoderLatentWeights_(encoderLatentWeights),
              encoderLatentBiases_(encoderLatentBiases),
              decoderHiddenWeights_(decoderHiddenWeights),
              decoderHiddenBiases_(decoderHiddenBiases),
              decoderOutputWeights_(decoderOutputWeights),
              decoderOutputBiases_(decoderOutputBiases),
              latentBuffer_(latentBuffer),
              hiddenScratchpad_(hiddenScratchpad),
              reconstructionScratchpad_(reconstructionScratchpad),
              inputDimensionSize_(inputDimensionSize),
              latentDimensionSize_(latentDimensionSize),
              hiddenNeuronCount_(hiddenNeuronCount) {}

        ~NestedAutoEncoderUnit() = default;
        NestedAutoEncoderUnit(const NestedAutoEncoderUnit&) = default;
        NestedAutoEncoderUnit& operator=(const NestedAutoEncoderUnit&) = default;
        NestedAutoEncoderUnit(NestedAutoEncoderUnit&&) noexcept = default;
        NestedAutoEncoderUnit& operator=(NestedAutoEncoderUnit&&) noexcept = default;

        /**
         * @brief Encodes a context slice into the latent buffer.
         *
         * Pipeline: Input -> Dense -> LeakyReLU -> Dense -> Latent
         *
         * All operations use TensorOps (SIMD-vectorised).
         * Zero allocations.  The hiddenScratchpad is reused every call.
         *
         * @param contextSlice Read-only view into the relevant portion of contextBuffer.
         */
        void encode(const std::span<const double> contextSlice) const {
            // Input -> Hidden (Dense + LeakyReLU)
            TensorOps::denseForwardPass(
                    contextSlice,
                    encoderHiddenWeights_,
                    encoderHiddenBiases_,
                    hiddenScratchpad_);
            TensorOps::applyLeakyReLU(hiddenScratchpad_, 0.01);

            // Hidden -> Latent (Dense, no activation - linear projection)
            TensorOps::denseForwardPass(
                    hiddenScratchpad_,
                    encoderLatentWeights_,
                    encoderLatentBiases_,
                    latentBuffer_);
        }


        /**
         * @brief Decodes the latent buffer back to input space and computes reconstruction loss.
         *
         * Pipeline: Latent -> Dense -> LeakyReLU -> Dense -> Reconstruction
         * Loss: Mean Squared Error between reconstruction and original input.
         *
         * Only called during training.  The reconstructionScratchpad is reused.
         *
         * @param originalContextSlice Read-only view of the original input for loss computation.
         * @return The Mean Squared Error reconstruction loss.
         */
        [[nodiscard]] double decodeAndComputeLoss(const std::span<const double> originalContextSlice) const {
            // Layer 1: Latent -> Hidden (Dense + LeakyReLU)
            const std::span<const double> latentView = latentBuffer_;
            TensorOps::denseForwardPass(
                    latentView,
                    decoderHiddenWeights_,
                    decoderHiddenBiases_,
                    hiddenScratchpad_);
            TensorOps::applyLeakyReLU(hiddenScratchpad_, 0.01);

            // Layer 2 Hidden -> Reconstruction (Dense, linear)
            const std::span<const double> hiddenView = hiddenScratchpad_;
            TensorOps::denseForwardPass(
                    hiddenView,
                    decoderOutputWeights_,
                    decoderOutputBiases_,
                    reconstructionScratchpad_);

            // Compute Mean Squared Error using the SIMD-vectorized implementation
            // This avoids a scalar loop and uses FusedMultiplyAdd accumulators
            // with a single horizontal reduction at the end
            return TensorOps::computeMeanSquaredErrorLoss(
                    reconstructionScratchpad_, originalContextSlice);
        }

        /**
         * @brief Returns the compressed latent representation from the last encode() call.
         * @return Read-only view over the latent buffer.
         */
        [[nodiscard]] std::span<const double> getLatentOutput() const noexcept {
            return latentBuffer_;
        }

        /** @brief Returns the number of elements in the latent output. */
        [[nodiscard]] int32_t getLatentDimensionSize() const noexcept {
            return latentDimensionSize_;
        }

    private:
        // Encoder weights (non-owning, sliced from parent agent's weight pool)
        std::span<double> encoderHiddenWeights_;
        std::span<double> encoderHiddenBiases_;
        std::span<double> encoderLatentWeights_;
        std::span<double> encoderLatentBiases_;

        // Decoder weights (non-owning, sliced from parent agent's weight pool)
        std::span<double> decoderHiddenWeights_;
        std::span<double> decoderHiddenBiases_;
        std::span<double> decoderOutputWeights_;
        std::span<double> decoderOutputBiases_;

        // Output and working memory (non-owning)
        std::span<double> latentBuffer_;
        std::span<double> hiddenScratchpad_;
        std::span<double> reconstructionScratchpad_;

        // Topology dimensions
        int32_t inputDimensionSize_;
        int32_t latentDimensionSize_;
        int32_t hiddenNeuronCount_;
    };

}


