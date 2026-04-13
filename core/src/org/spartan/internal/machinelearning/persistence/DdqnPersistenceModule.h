#pragma once

#include "ModelPersistenceModule.h"

namespace org::spartan::internal::machinelearning::persistence {

    class DdqnPersistenceModule : public ModelPersistenceModule {
    public:
        DdqnPersistenceModule();
        ~DdqnPersistenceModule() override = default;

        std::vector<double> serialize(
            const org::spartan::internal::machinelearning::SpartanModel* model) override;

        bool deserialize(org::spartan::internal::machinelearning::SpartanModel* model,
                        const std::vector<double>& weights) override;

        bool canHandle(uint32_t modelTypeIdentifier) const override;
        uint32_t modelTypeId() const override;

        /**
         * @brief Initializes and registers the module singleton.
         */
        static void initializeAndRegister();
    };

}
