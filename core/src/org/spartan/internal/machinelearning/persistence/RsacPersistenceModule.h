#pragma once

#include "ModelPersistenceModule.h"
#include <vector>

namespace org::spartan::internal::machinelearning::persistence {

    /**
     * @class RsacPersistenceModule
     * @brief Persistence handler for RSAC models.
     *
     * Handles serialization/deserialization of:
     * - GRU recurrent layer (weights, biases, hidden state)
     * - GRU Adam optimizer state
     * - Policy network (actor)
     * - Critic networks (Q1, Q2) with Adam state
     * - Policy head (mean + logstd)
     */
    class RsacPersistenceModule : public ModelPersistenceModule {
    public:
        RsacPersistenceModule();
        ~RsacPersistenceModule() override = default;

        std::vector<double> serialize(
            const org::spartan::internal::machinelearning::SpartanModel* model) override;

        bool deserialize(org::spartan::internal::machinelearning::SpartanModel* model,
                        const std::vector<double>& weights) override;

        bool canHandle(uint32_t modelTypeIdentifier) const override;
        uint32_t modelTypeId() const override;
    };

}
