#include "RsacPersistenceModule.h"
#include "../model/RecurrentSoftActorCriticSpartanModel.h"
#include "../../logging/SpartanLogger.h"

namespace org::spartan::internal::machinelearning::persistence {

    RsacPersistenceModule::RsacPersistenceModule() {
        ModelPersistenceRegistry::getInstance().registerModule(
            std::make_unique<RsacPersistenceModule>());
    }

    std::vector<double> RsacPersistenceModule::serialize(
            const machinelearning::SpartanModel* model) {

        const auto* rsacModel = dynamic_cast<const RecurrentSoftActorCriticSpartanModel*>(model);
        if (!rsacModel) {
            logging::SpartanLogger::error("RsacPersistenceModule::serialize: Model is not RSAC type");
            return {};
        }

        std::vector<double> weights;

        // Serialize:
        // 1. GRU weights + biases (from criticWeights)
        // 2. Critic1 weights + biases (from criticWeights)
        // 3. Critic2 weights + biases (from criticWeights)
        // 4. Actor weights + biases (from modelWeights)
        // 5. Adam moments for all networks

        // For now, return all weights from both buffers
        const auto criticWeights = rsacModel->getCriticWeights();
        const auto modelWeights = rsacModel->getModelWeights();

        weights.insert(weights.end(), criticWeights.begin(), criticWeights.end());
        weights.insert(weights.end(), modelWeights.begin(), modelWeights.end());

        logging::SpartanLogger::debug("RsacPersistenceModule::serialize: Serialized " +
                                    std::to_string(weights.size()) + " doubles");

        return weights;
    }

    bool RsacPersistenceModule::deserialize(
            machinelearning::SpartanModel* model,
            const std::vector<double>& weights) {

        auto* rsacModel = dynamic_cast<RecurrentSoftActorCriticSpartanModel*>(model);
        if (!rsacModel) {
            logging::SpartanLogger::error("RsacPersistenceModule::deserialize: Model is not RSAC type");
            return false;
        }

        // Validate total size matches critic + model weights
        const auto criticWeights = rsacModel->getCriticWeights();
        const auto modelWeights = rsacModel->getModelWeightsMutable();

        size_t expectedSize = criticWeights.size() + modelWeights.size();
        if (weights.size() != expectedSize) {
            logging::SpartanLogger::error(std::format(
                "RsacPersistenceModule::deserialize: Size mismatch. Expected {} got {}",
                expectedSize, weights.size()));
            return false;
        }

        // Restore critic weights
        std::copy(weights.begin(), weights.begin() + criticWeights.size(),
                 rsacModel->getCriticWeightsMutable().begin());

        // Restore model weights
        std::copy(weights.begin() + criticWeights.size(), weights.end(),
                 modelWeights.begin());

        logging::SpartanLogger::debug("RsacPersistenceModule::deserialize: Restored " +
                                    std::to_string(weights.size()) + " doubles");

        return true;
    }

    bool RsacPersistenceModule::canHandle(uint32_t modelTypeIdentifier) const {
        return modelTypeIdentifier == MODEL_TYPE_RSAC;
    }

    uint32_t RsacPersistenceModule::modelTypeId() const {
        return MODEL_TYPE_RSAC;
    }

}
