//
// Created by Alepando on 9/3/2026.
//

#pragma once

#include <span>

/**
 * @file GatedRecurrentUnitLayer.h
 * @brief Curiously Recurring Template Pattern base for Gated Recurrent Unit recurrent layers (Frontier B).
 *
 * Provides a compile-time-dispatched @c forwardPass method that concrete
 * Gated Recurrent Unit implementations must fulfil via @c forwardPassImpl.  The Gated Recurrent Unit
 * maintains a non-owning view over its hidden-state buffer so that temporal context
 * persists across ticks without any heap allocation.
 *
 * No vtable, no virtual overhead - the compiler can inline and
 * auto-vectorise the underlying Advanced Vector Extensions gate computations.
 *
 * @tparam DerivedLayer The concrete Gated Recurrent Unit layer class.
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class GatedRecurrentUnitLayer
     * @brief Curiously Recurring Template Pattern base for recurrent hidden-state propagation.
     *
     * Concrete implementations must provide:
     * @code
     *   void forwardPassImpl(std::span<const double> inputVector,
     *                        std::span<double> hiddenStateInOut) const;
     * @endcode
     *
     * @tparam DerivedLayer The concrete layer that inherits this base.
     */
    template <typename DerivedLayer>
    class GatedRecurrentUnitLayer {
    public:
        /**
         * @brief Constructs the Gated Recurrent Unit layer with non-owning views over its buffers.
         *
         * @param gateWeights      Span over the concatenated gate weight matrix (Java Virtual Machine-owned).
         *                         Layout: [W_z | W_r | W_h] flattened row-major.
         * @param gateBiases       Span over the concatenated gate bias vector (Java Virtual Machine-owned).
         * @param hiddenStateBuffer Span over the persistent hidden-state vector (Java Virtual Machine-owned).
         *                         Modified in-place on every @c forwardPass call.
         */
        GatedRecurrentUnitLayer(const std::span<double> gateWeights,
                                const std::span<double> gateBiases,
                                const std::span<double> hiddenStateBuffer)
            : gateWeights_(gateWeights),
              gateBiases_(gateBiases),
              hiddenStateBuffer_(hiddenStateBuffer) {}

        ~GatedRecurrentUnitLayer() = default;

        GatedRecurrentUnitLayer(const GatedRecurrentUnitLayer&) = default;
        GatedRecurrentUnitLayer& operator=(const GatedRecurrentUnitLayer&) = default;
        GatedRecurrentUnitLayer(GatedRecurrentUnitLayer&&) noexcept = default;
        GatedRecurrentUnitLayer& operator=(GatedRecurrentUnitLayer&&) noexcept = default;

        /**
         * @brief Executes one Gated Recurrent Unit time-step via static dispatch.
         *
         * Reads @p inputVector and the current hidden state, computes the
         * update/reset/candidate gates, and writes the new hidden state
         * back into @c hiddenStateBuffer_ (in-place, zero-copy to Java).
         *
         * @param inputVector      Read-only span over the input features for this tick.
         * @param hiddenStateInOut Writable span over the hidden state  -  updated in-place.
         */
        void forwardPass(const std::span<const double> inputVector,
                         const std::span<double> hiddenStateInOut) const {
            static_cast<const DerivedLayer*>(this)
                ->forwardPassImpl(inputVector, hiddenStateInOut);
        }

        /**
         * @brief Resets the hidden state to zeros (episode boundary).
         */
        void resetHiddenState() const {
            for (auto& element : hiddenStateBuffer_) {
                element = 0.0;
            }
        }

        /**
         * @brief Rebinds to new Java Virtual Machine-owned buffers.
         *
         * @param newGateWeights       Replacement gate weight span.
         * @param newGateBiases        Replacement gate bias span.
         * @param newHiddenStateBuffer Replacement hidden-state span.
         */
        void rebindLayerBuffers(const std::span<double> newGateWeights,
                                const std::span<double> newGateBiases,
                                const std::span<double> newHiddenStateBuffer) {
            gateWeights_ = newGateWeights;
            gateBiases_ = newGateBiases;
            hiddenStateBuffer_ = newHiddenStateBuffer;
        }

    protected:
        std::span<double> gateWeights_;
        std::span<double> gateBiases_;
        std::span<double> hiddenStateBuffer_;
    };

}
