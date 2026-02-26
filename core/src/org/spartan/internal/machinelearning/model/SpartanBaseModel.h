#pragma once

#include <span>
#include <cstdint> // Para uint64_t

#include "../ModelHyperparameterConfig.h"
#include "../critic/SpartanAbstractCritic.h"

namespace org::spartan::internal::machinelearning {

    class SpartanBaseModel {
    public:
        /**
         * @param agentId Unique identifier (typically the UUID most significant bits).
         * @param params Hyperparameters shared from JVM.
         * @param critic Value estimator.
         * @param weights Trainable weights buffer.
         * @param contextBuffer Memory segment where C++ will read the current state/context from Java.
         * @param actionBuffer Memory segment where C++ will write its output decisions.
         */
        SpartanBaseModel(uint64_t agentId,
                         ModelHyperparameterConfig* params,
                         SpartanAbstractCritic* critic,
                         std::span<double> weights,
                         std::span<const double> contextBuffer,
                         std::span<double> actionBuffer);

        ~SpartanBaseModel() = default;


        /**
         * @brief Executes the inference loop and writes to actionBuffer.
         */
        void processTick() const;

        void rebind(
            uint64_t agentId,
            ModelHyperparameterConfig* params,
            SpartanAbstractCritic* critic,
            std::span<double> weights,
            std::span<const double> contextBuffer,
            std::span<double> actionBuffer
            );

        /**
         * @brief unbinds the model spans from the JVM buffers, effectively making it inactive and ready for reuse.
         */
        void unbind();



        [[nodiscard]] uint64_t getId() const { return uuid_; }

    protected:
        uint64_t uuid_;
        ModelHyperparameterConfig* params_;
        SpartanAbstractCritic* critic_;
        std::span<double> weights_;
        std::span<const double> contextBuffer_;
        
        /** @brief Output buffer. Modifying this instantly updates Java's memory. */
        std::span<double> actionBuffer_;
    };

}