#include "DdqnPersistenceModule.h"
#include "../model/DoubleDeepQNetworkSpartanModel.h"
#include "../../logging/SpartanLogger.h"
#include <string>

namespace org::spartan::internal::machinelearning::persistence {

    DdqnPersistenceModule::DdqnPersistenceModule() {
        // Empty constructor - registration handled by initializeAndRegister()
    }

    void DdqnPersistenceModule::initializeAndRegister() {
        static bool initialized = false;
        if (!initialized) {
            initialized = true;
            ModelPersistenceRegistry::getInstance().registerModule(
                std::make_unique<DdqnPersistenceModule>());
        }
    }

    std::vector<double> DdqnPersistenceModule::serialize(
            const machinelearning::SpartanModel* model) {

        const auto* ddqnModel = dynamic_cast<const DoubleDeepQNetworkSpartanModel*>(model);
        if (!ddqnModel) {
            logging::SpartanLogger::error("DdqnPersistenceModule::serialize: Model is not DDQN type");
            return {};
        }

        std::vector<double> weights;

        // Serialize both online and target networks
        const auto criticWeights = ddqnModel->getCriticWeights();
        const auto modelWeights = ddqnModel->getModelWeights();

        weights.insert(weights.end(), criticWeights.begin(), criticWeights.end());
        weights.insert(weights.end(), modelWeights.begin(), modelWeights.end());

        logging::SpartanLogger::debug("DdqnPersistenceModule::serialize: Serialized " +
                                    std::to_string(weights.size()) + " doubles");

        return weights;
    }

    bool DdqnPersistenceModule::deserialize(
            machinelearning::SpartanModel* model,
            const std::vector<double>& weights) {

        auto* ddqnModel = dynamic_cast<DoubleDeepQNetworkSpartanModel*>(model);
        if (!ddqnModel) {
            logging::SpartanLogger::error("DdqnPersistenceModule::deserialize: Model is not DDQN type");
            return false;
        }

        const auto criticWeights = ddqnModel->getCriticWeights();
        const auto modelWeights = ddqnModel->getModelWeightsMutable();

        size_t expectedSize = criticWeights.size() + modelWeights.size();
        if (weights.size() != expectedSize) {
            logging::SpartanLogger::error("DdqnPersistenceModule::deserialize: Size mismatch");
            return false;
        }

        std::copy(weights.begin(), weights.begin() + criticWeights.size(),
                 ddqnModel->getCriticWeightsMutable().begin());

        std::copy(weights.begin() + criticWeights.size(), weights.end(),
                 modelWeights.begin());

        return true;
    }

    bool DdqnPersistenceModule::canHandle(uint32_t modelTypeIdentifier) const {
        return modelTypeIdentifier == MODEL_TYPE_DDQN;
    }

    uint32_t DdqnPersistenceModule::modelTypeId() const {
        return MODEL_TYPE_DDQN;
    }

}
