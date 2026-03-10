//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>

#include "SpartanModel.h"

/**
 * @file SpartanCompressor.h
 * @brief Intermediate abstract interface for representation-learning models.
 *
 * Models that compress high-dimensional observations into a compact
 * latent vector (e.g., AutoEncoder, Variational AutoEncoder) inherit from @c SpartanCompressor.
 * They do **not** interact with rewards  -  their sole purpose is
 * dimensionality reduction for downstream agents.
 *
 * @note Still part of **Frontier A** (dynamic polymorphism).
 *       The @c processTick() call remains virtual so the registry can
 *       invoke it uniformly, while all internal math should use
 *       Curiously Recurring Template Pattern / static dispatch (Frontier B).
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class SpartanCompressor
     * @brief Abstract interface for models that produce latent representations.
     *
     * Concrete compressors (e.g., @c SpartanAutoEncoder) must implement
     * both @c processTick() (inherited from SpartanModel) and
     * @c getLatentRepresentation().
     */
    class SpartanCompressor : public SpartanModel {
    public:
        ~SpartanCompressor() override = default;

        // Non-copyable / move-only (inherited)
        SpartanCompressor(const SpartanCompressor&) = delete;
        SpartanCompressor& operator=(const SpartanCompressor&) = delete;
        SpartanCompressor(SpartanCompressor&&) noexcept = default;
        SpartanCompressor& operator=(SpartanCompressor&&) noexcept = default;

        // Compression contract

        /**
         * @brief Returns the compressed latent representation of the last input.
         *
         * After @c processTick() encodes the context buffer, this method
         * provides a read-only view into the resulting latent vector.
         * The span points to memory inside the action output buffer
         * (or an internal staging area), so it is valid until the next tick.
         *
         * @return A read-only span over the latent representation.
         */
        [[nodiscard]] virtual std::span<const double> getLatentRepresentation() const = 0;

        /**
         * @brief Returns the reconstruction loss from the last encoding pass.
         *
         * Used for monitoring convergence without requiring reward signals.
         *
         * @return Scalar reconstruction loss (lower is better).
         */
        [[nodiscard]] virtual double getReconstructionLoss() const = 0;

    protected:
        /**
         * @brief Protected constructor  -  delegates to @c SpartanModel.
         *
         * @param agentIdentifier              Unique 64-bit agent identifier.
         * @param opaqueHyperparameterConfig   Opaque pointer to Java Virtual Machine-owned config struct.
         * @param modelWeights                 Span over the trainable-weight buffer.
         * @param contextBuffer                Span over the observation/state input buffer.
         * @param actionOutputBuffer           Span over the action output buffer.
         */
        SpartanCompressor(const uint64_t agentIdentifier,
                          void* opaqueHyperparameterConfig,
                          const std::span<double> modelWeights,
                          const std::span<const double> contextBuffer,
                          const std::span<double> actionOutputBuffer)
            : SpartanModel(agentIdentifier,
                           opaqueHyperparameterConfig,
                           modelWeights,
                           contextBuffer,
                           actionOutputBuffer) {}

        /** @brief Default constructor for deferred initialisation via rebind(). */
        SpartanCompressor() = default;
    };

}





