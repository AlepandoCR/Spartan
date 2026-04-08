//
// Created by Alepando on 24/2/2026.
//

#pragma once

#include <span>
#include <cstdint>

/**
 * @file SpartanModel.h
 * @brief Pure abstract interface for all Machine Learning models in the Spartan engine.
 *
 * This is the **Frontier A** (dynamic-polymorphism boundary).  The model
 * registry stores @c unique_ptr<SpartanModel> and invokes @c processTick()
 * through a single virtual call per tick  -  O(1) overhead that is acceptable
 * at the Application Programming Interface / scheduler layer.
 *
 * Concrete model families derive through one of the three intermediate
 * interfaces:
 *   - @c SpartanAgent       -  decision-making models (Recurrent Soft Actor-Critic, Double Deep Q-Network, ...)
 *   - @c SpartanCritic      -  value-estimation models (standalone critics)
 *   - @c SpartanCompressor  -  representation-learning models (AutoEncoder, ...)
 *
 * The hyperparameter config pointer is stored as @c void* so that each
 * concrete model can @c static_cast to its own specialised Plain Old Data struct
 * without polluting the base interface with template parameters.
 *
 * @note All memory is owned by the Java Virtual Machine.  This class only stores non-owning
 *       @c std::span views for zero-copy Foreign Function and Memory interop.
 */
namespace org::spartan::internal::machinelearning {

    /**
     * @class SpartanModel
     * @brief Abstract root of the Spartan model hierarchy.
     */
    class SpartanModel {
    public:
        virtual ~SpartanModel() = default;

        // Non-copyable / move-only
        SpartanModel(const SpartanModel&) = delete;
        SpartanModel& operator=(const SpartanModel&) = delete;
        SpartanModel(SpartanModel&&) noexcept = default;
        SpartanModel& operator=(SpartanModel&&) noexcept = default;

        // Virtual API (Frontier A)

        /**
         * @brief Executes one inference cycle and writes results to the action buffer.
         *
         * This is the single virtual entry-point invoked by the registry's
         * parallel scheduler.  All heavy math behind this call should use
         * static dispatch internally
         */
        virtual void processTick() = 0;

        // Buffer management (non-virtual, shared by all models)

        /**
         * @brief Replaces the context (state) buffer after a Java Virtual Machine arena resize.
         *
         * @param newContextBuffer Updated span over the Java Virtual Machine-owned state memory.
         */
        void setContextBuffer(const std::span<const double> newContextBuffer) {
            contextBuffer_ = newContextBuffer;
        }

        /**
         * @brief Updates the clean sizes buffer for dynamic context slicing.
         *
         * Java writes one int32_t per nested encoder slot, indicating the actual
         * number of valid (non-padding) elements that were populated this tick.
         * Models that use nested encoders read this buffer to create clean views
         * over only the valid portion of each variable-length context slice.
         *
         * @param newCleanSizesBuffer Read-only span over the JVM-owned clean sizes array.
         */
        void setCleanSizesBuffer(const std::span<const int32_t> newCleanSizesBuffer) {
            cleanSizesBuffer_ = newCleanSizesBuffer;
        }

        /**
         * @brief Rebinds every Java Virtual Machine-owned buffer to a new set of pointers.
         *
         * The config pointer is @c void* because each concrete model type
         * expects a different Plain Old Data struct.  The concrete class is responsible
         * for casting it to the correct type.
         *
         * @param agentIdentifier          New unique identifier.
         * @param opaqueHyperparameterConfig Opaque pointer to a Java Virtual Machine-owned config struct.
         * @param modelWeights             Span over the model's weight array.
         * @param contextBuffer            Span over the observation/state array.
         * @param actionOutputBuffer       Span over the output/action array.
         */
        virtual void rebind(const uint64_t agentIdentifier,
                            void* opaqueHyperparameterConfig,
                            const std::span<double> modelWeights,
                            const std::span<const double> contextBuffer,
                            const std::span<double> actionOutputBuffer) {
            agentIdentifier_ = agentIdentifier;
            opaqueHyperparameterConfig_ = opaqueHyperparameterConfig;
            modelWeights_ = modelWeights;
            contextBuffer_ = contextBuffer;
            actionOutputBuffer_ = actionOutputBuffer;
        }

        /**
         * @brief Detaches the model from all Java Virtual Machine buffers, making it inert.
         *
         * After this call the model is safe to place in the idle pool
         * and later rebind to a different agent.
         */
        virtual void unbind() {
            opaqueHyperparameterConfig_ = nullptr;
            modelWeights_ = std::span<double>();
            contextBuffer_ = std::span<const double>();
            actionOutputBuffer_ = std::span<double>();
        }

        /** @brief Returns the unique agent identifier bound to this model. */
        [[nodiscard]] uint64_t getIdentifier() const noexcept { return agentIdentifier_; }

        /** @brief Returns the opaque hyperparameter config pointer for type introspection. */
        [[nodiscard]] const void* getOpaqueHyperparameterConfig() const noexcept {
            return opaqueHyperparameterConfig_;
        }

        /** @brief Returns a read-only view of the model's weight buffer for persistence. */
        [[nodiscard]] std::span<const double> getModelWeights() const noexcept {
            return {modelWeights_.data(), modelWeights_.size()};
        }

        /** @brief Returns a mutable view of the model's weight buffer for rebind. */
        [[nodiscard]] std::span<double> getModelWeightsMutable() noexcept {
            return modelWeights_;
        }

        /**
         * @brief Returns a read-only view of the critic/secondary weight buffer for persistence.
         *
         * Models that store trainable parameters in a separate critic buffer
         * (e.g., Recurrent Soft Actor-Critic stores GRU + Q1 + Q2 weights,
         * Double Deep Q-Network stores target network weights) override this
         * to expose that buffer.  The persistence layer serializes both
         * getModelWeights() and getCriticWeights() into the .spartan file.
         *
         * The default implementation returns an empty span for models that
         * have no secondary weight buffer (e.g., AutoEncoder).
         *
         * @return Read-only span over the critic weight buffer, or empty.
         */
        [[nodiscard]] virtual std::span<const double> getCriticWeights() const noexcept {
            return {};
        }

        [[nodiscard]] virtual std::span<double> getCriticWeightsMutable() noexcept {
            return {};
        }

    protected:
        /**
         * @brief Protected constructor  -  only concrete subclasses can instantiate.
         *
         * @param agentIdentifier              Unique 64-bit agent identifier.
         * @param opaqueHyperparameterConfig   Opaque pointer to Java Virtual Machine-owned config struct.
         * @param modelWeights                 Span over the trainable-weight buffer.
         * @param contextBuffer                Span over the observation/state input buffer.
         * @param actionOutputBuffer           Span over the action output buffer.
         */
        SpartanModel(const uint64_t agentIdentifier,
                     void* opaqueHyperparameterConfig,
                     const std::span<double> modelWeights,
                     const std::span<const double> contextBuffer,
                     const std::span<double> actionOutputBuffer)
            : agentIdentifier_(agentIdentifier),
              opaqueHyperparameterConfig_(opaqueHyperparameterConfig),
              modelWeights_(modelWeights),
              contextBuffer_(contextBuffer),
              actionOutputBuffer_(actionOutputBuffer) {}

        /** @brief Default constructor for deferred initialisation via rebind(). */
        SpartanModel() = default;

        // Shared state (non-owning views over Java Virtual Machine memory)
        uint64_t agentIdentifier_ = 0;

        /** @brief Opaque config pointer  -  concrete models static_cast to their Plain Old Data type. */
        void* opaqueHyperparameterConfig_ = nullptr;

        std::span<double> modelWeights_;
        std::span<const double> contextBuffer_;
        std::span<double> actionOutputBuffer_;

        /** @brief Per-encoder clean element counts for dynamic context slicing (JVM-owned). */
        std::span<const int32_t> cleanSizesBuffer_;
    };

}