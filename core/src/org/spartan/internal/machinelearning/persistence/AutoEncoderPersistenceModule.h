#pragma once

#include "ModelPersistenceModule.h"

namespace org::spartan::internal::machinelearning::persistence {

    class AutoEncoderPersistenceModule : public ModelPersistenceModule {
    public:
        AutoEncoderPersistenceModule();
        ~AutoEncoderPersistenceModule() override = default;

        std::vector<double> serialize(
            const org::spartan::internal::machinelearning::SpartanModel* model) override;

        bool deserialize(org::spartan::internal::machinelearning::SpartanModel* model,
                        const std::vector<double>& weights) override;

        bool canHandle(uint32_t modelTypeIdentifier) const override;
        uint32_t modelTypeId() const override;
    };

}
