//
// Created by Alepando on 9/3/2026.
//

#include "DoubleDeepQNetworkSpartanModel.h"

namespace org::spartan::internal::machinelearning {

    DoubleDeepQNetworkSpartanModel::DoubleDeepQNetworkSpartanModel(
            const uint64_t agentIdentifier,
            void* opaqueHyperparameterConfig,
            std::span<double> modelWeights,
            std::span<const double> contextBuffer,
            std::span<double> actionOutputBuffer,
            std::span<double> onlineNetworkWeights,
            std::span<double> onlineNetworkBiases,
            std::span<double> targetNetworkWeights,
            std::span<double> targetNetworkBiases)
        : SpartanAgent(agentIdentifier,
                        opaqueHyperparameterConfig,
                        modelWeights,
                        contextBuffer,
                        actionOutputBuffer),
          onlineNetwork_(onlineNetworkWeights, onlineNetworkBiases),
          targetNetwork_(targetNetworkWeights, targetNetworkBiases) {}

    void DoubleDeepQNetworkSpartanModel::processTick() {
        const auto* config = typedConfig();
        if (!config || !config->baseConfig.isTraining) {
            return;
        }

        // Frontier B: all calls below resolve at compile time

        //  Feed contextBuffer_ through the online Q-network.
        // TODO: Implement epsilon-greedy action selection.

        // Periodically sync online -> target weights.
        ++ticksSinceLastTargetSync_;
        if (ticksSinceLastTargetSync_ >= config->targetNetworkSyncInterval) {
            // TODO: Hard-copy online weights to target weights.
            ticksSinceLastTargetSync_ = 0;
        }
    }

    void DoubleDeepQNetworkSpartanModel::applyReward(
            [[maybe_unused]] const double rewardSignal) {
        // TODO: Store transition in replay buffer and sample mini-batch.
        // TODO: Compute double-Q target and backpropagate through online network.
    }

    void DoubleDeepQNetworkSpartanModel::decayExploration() {
        // TODO: Decay epsilon according to config epsilonDecay / epsilonMin.
    }

}

