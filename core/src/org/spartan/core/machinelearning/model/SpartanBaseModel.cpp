//
// Created by Alepando on 24/2/2026.
//

#include "SpartanBaseModel.h"

namespace org::spartan::core::machinelearning {


    SpartanBaseModel::SpartanBaseModel(
        const uint64_t agentId,
        ModelHyperparameterConfig* params,
        SpartanAbstractCritic* critic,
        const std::span<double> weights,
        const std::span<const double> contextBuffer,
        const std::span<double> actionBuffer
        )
    : uuid_(agentId),
      params_(params),
      critic_(critic),
      weights_(weights),
      contextBuffer_(contextBuffer),
      actionBuffer_(actionBuffer){}

    void SpartanBaseModel::processTick() const {
        if (!params_->isTraining) {
            return;
        }

        // TODO: Implement training logic (forward pass, loss computation, weight update).
    }

    void SpartanBaseModel::rebind(
        const uint64_t agentId,
        ModelHyperparameterConfig* params,
        SpartanAbstractCritic* critic,
        const std::span<double> weights,
        const std::span<const double> contextBuffer,
        const std::span<double> actionBuffer
        ) {
        this->uuid_ = agentId;
        this->params_ = params;
        this->critic_ = critic;
        this->weights_ = weights;
        this->contextBuffer_ = contextBuffer;
        this->actionBuffer_ = actionBuffer;
    }

    void SpartanBaseModel::unbind() {
        this->params_ = nullptr;
        this->critic_ = nullptr;
        this->weights_ = std::span<double>();
        this->contextBuffer_ = std::span<const double>();
        this->actionBuffer_ = std::span<double>();
    }

}

