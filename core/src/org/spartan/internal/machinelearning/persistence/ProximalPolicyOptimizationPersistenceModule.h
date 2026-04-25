#pragma once

#include "ModelPersistenceModule.h"

namespace org::spartan::internal::machinelearning::persistence {

    class ProximalPolicyOptimizationPersistenceModule : public ModelPersistenceModule {
    public:
        ProximalPolicyOptimizationPersistenceModule();
        ~ProximalPolicyOptimizationPersistenceModule() override = default;

        std::vector<double> serialize(
            const SpartanModel* model) override;

        bool deserialize(SpartanModel* model,
                        const std::vector<double>& weights) override;

        [[nodiscard]] bool canHandle(uint32_t modelTypeIdentifier) const override;
        [[nodiscard]] uint32_t modelTypeId() const override;

        static void initializeAndRegister();
    };

}
