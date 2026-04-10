#include "CuriosityRsacPersistenceModule.h"
#include "../model/CuriosityDrivenRecurrentSoftActorCriticSpartanModel.h"
#include "../../logging/SpartanLogger.h"

namespace org::spartan::internal::machinelearning::persistence {

    CuriosityRsacPersistenceModule::CuriosityRsacPersistenceModule() {
        ModelPersistenceRegistry::getInstance().registerModule(
            std::make_unique<CuriosityRsacPersistenceModule>());
    }

    std::vector<double> CuriosityRsacPersistenceModule::serialize(
            const machinelearning::SpartanModel* model) {

        const auto* curiosityModel =
            dynamic_cast<const CuriosityDrivenRecurrentSoftActorCriticSpartanModel*>(model);
        if (!curiosityModel) {
            logging::SpartanLogger::error("CuriosityRsacPersistenceModule::serialize: Model is not Curiosity RSAC type");
            return {};
        }

        std::vector<double> weights;

        // Serialize: RSAC + Forward Dynamics
        // Get all weights from curiosity model (includes internal RSAC + forward dynamics)
        const auto modelWeights = curiosityModel->getModelWeights();
        const auto criticWeights = curiosityModel->getCriticWeights();

        weights.insert(weights.end(), criticWeights.begin(), criticWeights.end());
        weights.insert(weights.end(), modelWeights.begin(), modelWeights.end());

        logging::SpartanLogger::debug("CuriosityRsacPersistenceModule::serialize: Serialized " +
                                    std::to_string(weights.size()) + " doubles");

        return weights;
    }

    bool CuriosityRsacPersistenceModule::deserialize(
            machinelearning::SpartanModel* model,
            const std::vector<double>& weights) {

        auto* curiosityModel =
            dynamic_cast<CuriosityDrivenRecurrentSoftActorCriticSpartanModel*>(model);
        if (!curiosityModel) {
            logging::SpartanLogger::error("CuriosityRsacPersistenceModule::deserialize: Model is not Curiosity RSAC type");
            return false;
        }

        const auto criticWeights = curiosityModel->getCriticWeights();
        const auto modelWeights = curiosityModel->getModelWeightsMutable();

        size_t expectedSize = criticWeights.size() + modelWeights.size();
        if (weights.size() != expectedSize) {
            logging::SpartanLogger::error("CuriosityRsacPersistenceModule::deserialize: Size mismatch");
            return false;
        }

        std::copy(weights.begin(), weights.begin() + criticWeights.size(),
                 curiosityModel->getCriticWeightsMutable().begin());

        std::copy(weights.begin() + criticWeights.size(), weights.end(),
                 modelWeights.begin());

        return true;
    }

    bool CuriosityRsacPersistenceModule::canHandle(uint32_t modelTypeIdentifier) const {
        return modelTypeIdentifier == MODEL_TYPE_CURIOSITY_RSAC;
    }

    uint32_t CuriosityRsacPersistenceModule::modelTypeId() const {
        return MODEL_TYPE_CURIOSITY_RSAC;
    }

}
