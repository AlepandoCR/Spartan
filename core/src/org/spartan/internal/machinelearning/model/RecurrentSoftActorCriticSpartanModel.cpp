//
// Created by Alepando on 9/3/2026.
//

#include "RecurrentSoftActorCriticSpartanModel.h"

namespace org::spartan::internal::machinelearning {

    RecurrentSoftActorCriticSpartanModel::RecurrentSoftActorCriticSpartanModel(
            const uint64_t agentIdentifier,
            void* opaqueHyperparameterConfig,
            std::span<double> modelWeights,
            std::span<const double> contextBuffer,
            std::span<double> actionOutputBuffer,
            std::span<double> gruGateWeights,
            std::span<double> gruGateBiases,
            std::span<double> gruHiddenState,
            std::span<double> policyWeights,
            std::span<double> policyBiases,
            std::span<double> firstCriticWeights,
            std::span<double> firstCriticBiases,
            std::span<double> secondCriticWeights,
            std::span<double> secondCriticBiases)
        : SpartanAgent(agentIdentifier,
                        opaqueHyperparameterConfig,
                        modelWeights,
                        contextBuffer,
                        actionOutputBuffer),
          recurrentLayer_(gruGateWeights, gruGateBiases, gruHiddenState),
          policyNetwork_(policyWeights, policyBiases),
          firstCriticNetwork_(firstCriticWeights, firstCriticBiases),
          secondCriticNetwork_(secondCriticWeights, secondCriticBiases) {}

    void RecurrentSoftActorCriticSpartanModel::processTick() {
        if (const auto* config = typedConfig(); !config || !config->baseConfig.isTraining) {
            return;
        }

        // ── Frontier B: all calls below resolve at compile time ──────

        //  Encode temporal context through the Gated Recurrent Unit layer.
        // TODO: Implement Gated Recurrent Unit forward pass with contextBuffer_ -> hiddenState.

        // Run the Gaussian policy to produce action mean + log-std.
        // TODO: Implement policy forward pass with hiddenState -> actionOutputBuffer_.

        //  Evaluate twin Q-critics for the current state-action pair.
        // TODO: Implement Q-value computation for gradient step.
    }

    void RecurrentSoftActorCriticSpartanModel::applyReward(
            [[maybe_unused]] const double rewardSignal) {
        // TODO: Implement Soft Actor-Critic Bellman backup + dual Q-network gradient step.
    }

    void RecurrentSoftActorCriticSpartanModel::decayExploration() {
        // Soft Actor-Critic uses entropy temperature (alpha) rather than epsilon-greedy,
        // but the base config epsilon can still be decayed for hybrid policies.
        // TODO: Implement alpha / epsilon decay schedule.
    }

}



