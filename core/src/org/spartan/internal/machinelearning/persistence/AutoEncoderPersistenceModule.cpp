#include "AutoEncoderPersistenceModule.h"
#include "../model/AutoEncoderCompressorSpartanModel.h"
#include "../../logging/SpartanLogger.h"
#include <string>

namespace org::spartan::internal::machinelearning::persistence {

    AutoEncoderPersistenceModule::AutoEncoderPersistenceModule() {
        ModelPersistenceRegistry::getInstance().registerModule(
            std::make_unique<AutoEncoderPersistenceModule>());
    }

    std::vector<double> AutoEncoderPersistenceModule::serialize(
            const machinelearning::SpartanModel* model) {

        const auto* aeModel = dynamic_cast<const AutoEncoderCompressorSpartanModel*>(model);
        if (!aeModel) {
            logging::SpartanLogger::error("AutoEncoderPersistenceModule::serialize: Model is not AutoEncoder type");
            return {};
        }

        std::vector<double> weights;

        // Serialize encoder and decoder weights (NOT latent buffer)
        const auto modelWeights = aeModel->getModelWeights();
        weights.insert(weights.end(), modelWeights.begin(), modelWeights.end());

        logging::SpartanLogger::debug("AutoEncoderPersistenceModule::serialize: Serialized " +
                                    std::to_string(weights.size()) + " doubles");

        return weights;
    }

    bool AutoEncoderPersistenceModule::deserialize(
            machinelearning::SpartanModel* model,
            const std::vector<double>& weights) {

        auto* aeModel = dynamic_cast<AutoEncoderCompressorSpartanModel*>(model);
        if (!aeModel) {
            logging::SpartanLogger::error("AutoEncoderPersistenceModule::deserialize: Model is not AutoEncoder type");
            return false;
        }

        const auto modelWeights = aeModel->getModelWeightsMutable();

        if (weights.size() != modelWeights.size()) {
            logging::SpartanLogger::error("AutoEncoderPersistenceModule::deserialize: Size mismatch");
            return false;
        }

        std::copy(weights.begin(), weights.end(), modelWeights.begin());

        return true;
    }

    bool AutoEncoderPersistenceModule::canHandle(uint32_t modelTypeIdentifier) const {
        return modelTypeIdentifier == MODEL_TYPE_AUTOENCODER;
    }

    uint32_t AutoEncoderPersistenceModule::modelTypeId() const {
        return MODEL_TYPE_AUTOENCODER;
    }

}
