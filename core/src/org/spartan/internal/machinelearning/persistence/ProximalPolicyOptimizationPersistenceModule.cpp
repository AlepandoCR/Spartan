#include "ProximalPolicyOptimizationPersistenceModule.h"

#include "../model/ProximalPolicyOptimizationSpartanModel.h"
#include "../../logging/SpartanLogger.h"

#include <algorithm>
#include <cstddef>
#include <string>

namespace org::spartan::internal::machinelearning::persistence {

    ProximalPolicyOptimizationPersistenceModule::ProximalPolicyOptimizationPersistenceModule() {
        // Empty constructor; registration is handled by initializeAndRegister().
    }

    void ProximalPolicyOptimizationPersistenceModule::initializeAndRegister() {
        static bool initialized = false;
        if (!initialized) {
            initialized = true;
            ModelPersistenceRegistry::getInstance().registerModule(
                std::make_unique<ProximalPolicyOptimizationPersistenceModule>());
        }
    }

    std::vector<double> ProximalPolicyOptimizationPersistenceModule::serialize(
            const SpartanModel* model) {

        const auto* proximalPolicyOptimizationModel =
            dynamic_cast<const ProximalPolicyOptimizationSpartanModel*>(model);
        if (!proximalPolicyOptimizationModel) {
            logging::SpartanLogger::error(
                "ProximalPolicyOptimizationPersistenceModule::serialize: Model is not ProximalPolicyOptimization type");
            return {};
        }

        std::vector<double> weights;

        const auto actorWeights = proximalPolicyOptimizationModel->getActorWeights();
        const auto actorBiases = proximalPolicyOptimizationModel->getActorBiases();
        const auto criticWeights = proximalPolicyOptimizationModel->getCriticNetworkWeights();
        const auto criticBiases = proximalPolicyOptimizationModel->getCriticBiases();

        weights.reserve(actorWeights.size() + actorBiases.size() + criticWeights.size() + criticBiases.size());
        weights.insert(weights.end(), actorWeights.begin(), actorWeights.end());
        weights.insert(weights.end(), actorBiases.begin(), actorBiases.end());
        weights.insert(weights.end(), criticWeights.begin(), criticWeights.end());
        weights.insert(weights.end(), criticBiases.begin(), criticBiases.end());

        logging::SpartanLogger::debug(
            "ProximalPolicyOptimizationPersistenceModule::serialize: Serialized " +
            std::to_string(weights.size()) + " doubles");

        return weights;
    }

    bool ProximalPolicyOptimizationPersistenceModule::deserialize(
            SpartanModel* model,
            const std::vector<double>& weights) {

        auto* proximalPolicyOptimizationModel =
            dynamic_cast<ProximalPolicyOptimizationSpartanModel*>(model);
        if (!proximalPolicyOptimizationModel) {
            logging::SpartanLogger::error(
                "ProximalPolicyOptimizationPersistenceModule::deserialize: Model is not ProximalPolicyOptimization type");
            return false;
        }

        const auto actorWeights = proximalPolicyOptimizationModel->getActorWeightsMutable();
        const auto actorBiases = proximalPolicyOptimizationModel->getActorBiasesMutable();
        const auto criticWeights = proximalPolicyOptimizationModel->getCriticNetworkWeightsMutable();
        const auto criticBiases = proximalPolicyOptimizationModel->getCriticBiasesMutable();

        if (const size_t expectedSize = actorWeights.size() + actorBiases.size() + criticWeights.size() + criticBiases.size(); weights.size() != expectedSize) {
            logging::SpartanLogger::error(
                "ProximalPolicyOptimizationPersistenceModule::deserialize: Size mismatch. Expected " +
                std::to_string(expectedSize) + " got " + std::to_string(weights.size()));
            return false;
        }

        auto cursor = weights.begin();
        std::copy_n(cursor, static_cast<std::ptrdiff_t>(actorWeights.size()), actorWeights.begin());
        cursor += static_cast<std::ptrdiff_t>(actorWeights.size());

        std::copy_n(cursor, static_cast<std::ptrdiff_t>(actorBiases.size()), actorBiases.begin());
        cursor += static_cast<std::ptrdiff_t>(actorBiases.size());

        std::copy_n(cursor, static_cast<std::ptrdiff_t>(criticWeights.size()), criticWeights.begin());
        cursor += static_cast<std::ptrdiff_t>(criticWeights.size());

        std::copy_n(cursor, static_cast<std::ptrdiff_t>(criticBiases.size()), criticBiases.begin());

        logging::SpartanLogger::debug(
            "ProximalPolicyOptimizationPersistenceModule::deserialize: Restored " +
            std::to_string(weights.size()) + " doubles");

        return true;
    }

    bool ProximalPolicyOptimizationPersistenceModule::canHandle(uint32_t modelTypeIdentifier) const {
        return modelTypeIdentifier == SPARTAN_MODEL_TYPE_PROXIMAL_POLICY_OPTIMIZATION;
    }

    uint32_t ProximalPolicyOptimizationPersistenceModule::modelTypeId() const {
        return SPARTAN_MODEL_TYPE_PROXIMAL_POLICY_OPTIMIZATION;
    }

}


