//
// Created by Alepando on 9/3/2026.
//

#include "DoubleDeepQNetworkSpartanModel.h"

#include <cstring>
#include <random>

namespace org::spartan::internal::machinelearning {

    using namespace org::spartan::internal::math::tensor;

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
          targetNetwork_(targetNetworkWeights, targetNetworkBiases),
          rawOnlineWeights_(onlineNetworkWeights),
          rawOnlineBiases_(onlineNetworkBiases),
          criticWeightsSpan_(targetNetworkWeights.data(),
                             targetNetworkWeights.size() + targetNetworkBiases.size()) {

        const auto* config = typedConfig();
        if (!config) return;

        const int stateSize = config->baseConfig.stateSize;
        const int actionSize = config->baseConfig.actionSize;
        const int hiddenSize = config->hiddenLayerNeuronCount;
        const int combinedInputSize = stateSize + actionSize;

        // Pre-allocate inference scratchpads large enough for any layer
        const int maxCapacity = std::max(hiddenSize, combinedInputSize);
        scratchpadA_.resize(maxCapacity);
        scratchpadB_.resize(maxCapacity);

        // Pre-allocate Q-value output buffers (one Q-value per discrete action)
        onlineQValuesScratchpad_.resize(actionSize);
        targetQValuesScratchpad_.resize(actionSize);

        // Pre-allocate state/action snapshots for experience replay
        previousStateSnapshot_.resize(stateSize);
        previousActionSnapshot_.resize(actionSize);

        // Pre-allocate gradient scratchpads
        const size_t totalWeightCount = onlineNetworkWeights.size();
        const size_t totalBiasCount = onlineNetworkBiases.size();
        onlineWeightGradients_.resize(totalWeightCount);
        onlineBiasGradients_.resize(totalBiasCount);
        outputGradientScratchpad_.resize(std::max(hiddenSize, actionSize));
        inputGradientScratchpad_.resize(maxCapacity);

        // Pre-allocate Adam optimizer state (zero-initialized by resize)
        adamWeightMomentum_.resize(totalWeightCount);
        adamWeightVelocity_.resize(totalWeightCount);
        adamBiasMomentum_.resize(totalBiasCount);
        adamBiasVelocity_.resize(totalBiasCount);

        // Pre-allocate replay buffer and batch index buffer
        const int replayCapacity = config->replayBufferCapacity > 0
            ? config->replayBufferCapacity : 10000;
        const int batchSize = config->replayBatchSize > 0
            ? config->replayBatchSize : 32;

        replayBuffer_ = replay::ExperienceReplayBuffer(replayCapacity, stateSize, actionSize);
        batchIndicesBuffer_.resize(batchSize);
    }

    void DoubleDeepQNetworkSpartanModel::processTick() {
        const auto* config = typedConfig();
        if (!config) return;

        const int actionSize = config->baseConfig.actionSize;

        //
        //  Store the previous transition in the replay buffer.
        //          We store (s_{t-1}, a_{t-1}, r=0, s_t, terminal=false).
        //          The reward will be retroactively applied via applyReward().
        //          On the first tick, there is no previous state to store.
        //
        if (hasPreviousState_) {
            replayBuffer_.storeTransition(
                std::span<const double>(previousStateSnapshot_),
                std::span<const double>(previousActionSnapshot_),
                0.0,  // Reward injected later via applyReward
                contextBuffer_,
                false);
        }

        //
        // Phase B: Epsilon-greedy action selection.
        //
        // With probability epsilon, choose a random action (exploration).
        // Otherwise, compute Q-values for all actions via the online network
        // and select the action with the highest Q-value (exploitation).
        //
        thread_local std::mt19937 randomGenerator(std::random_device{}());
        std::uniform_real_distribution uniformDistribution(0.0, 1.0);

        const double explorationRoll = uniformDistribution(randomGenerator);

        if (config->baseConfig.isTraining && explorationRoll < config->baseConfig.epsilon) {
            // Exploration: write random action values
            std::uniform_real_distribution actionDistribution(-1.0, 1.0);
            for (int actionIndex = 0; actionIndex < actionSize; ++actionIndex) {
                actionOutputBuffer_[actionIndex] = actionDistribution(randomGenerator);
            }
        } else {
            // Exploitation: compute Q-values for each discrete action using
            // the online network, then select argmax.
            // For continuous action spaces, we evaluate the online network
            // with the full state and write Q-values to the action buffer.
            for (int actionIndex = 0; actionIndex < actionSize; ++actionIndex) {
                // Create a one-hot action vector in the scratchpad
                std::fill_n(scratchpadA_.begin(), actionSize, 0.0);
                scratchpadA_[actionIndex] = 1.0;

                const double qValue = onlineNetwork_.computeQValue(
                    contextBuffer_,
                    std::span<const double>(scratchpadA_.data(), actionSize),
                    config,
                    std::span(scratchpadA_.data() + actionSize, scratchpadA_.size() - actionSize),
                    std::span(scratchpadB_));

                onlineQValuesScratchpad_[actionIndex] = qValue;
            }

            // Write Q-values to action output buffer so Java can read them
            const size_t bestActionIndex = TensorOps::findArgmax(
                std::span<const double>(onlineQValuesScratchpad_.data(), actionSize));

            // Write a one-hot encoding of the best action
            std::ranges::fill(actionOutputBuffer_, 0.0);
            actionOutputBuffer_[bestActionIndex] = 1.0;
        }

        //
        //  Snapshot the current state and chosen action for the next transition.
        //
        std::memcpy(previousStateSnapshot_.data(), contextBuffer_.data(),
                     static_cast<size_t>(config->baseConfig.stateSize) * sizeof(double));
        std::memcpy(previousActionSnapshot_.data(), actionOutputBuffer_.data(),
                     static_cast<size_t>(actionSize) * sizeof(double));
        hasPreviousState_ = true;

        //
        //Periodically sync online -> target weights via Polyak averaging.
        //
        ++ticksSinceLastTargetSync_;
        if (ticksSinceLastTargetSync_ >= config->targetNetworkSyncInterval) {
            TensorOps::applyPolyakAveraging(
                rawOnlineWeights_,
                targetNetwork_.getTargetWeights(),
                0.005);

            TensorOps::applyPolyakAveraging(
                rawOnlineBiases_,
                targetNetwork_.getTargetBiases(),
                0.005);

            ticksSinceLastTargetSync_ = 0;
        }
    }

    void DoubleDeepQNetworkSpartanModel::applyReward(const double rewardSignal) {
        const auto* config = typedConfig();
        if (!config || !config->baseConfig.isTraining) return;

        // Retroactively assign the reward to the most recent transition.
        // The transition was stored with reward=0 during processTick().
        replayBuffer_.updateLatestTransitionReward(rewardSignal);

        const int batchSize = config->replayBatchSize > 0 ? config->replayBatchSize : 32;
        if (!replayBuffer_.hasEnoughTransitions(batchSize)) return;

        const int actionSize = config->baseConfig.actionSize;
        const int stateSize = config->baseConfig.stateSize;
        const double gamma = config->baseConfig.gamma;
        const double learningRate = config->baseConfig.learningRate;

        // Sample a random mini-batch of transition indices
        replayBuffer_.sampleBatchIndices(
            std::span(batchIndicesBuffer_.data(), batchSize), batchSize);

        // Zero the gradient accumulators before the batch
        std::ranges::fill(onlineWeightGradients_, 0.0);
        std::ranges::fill(onlineBiasGradients_, 0.0);

        ++trainingStepCounter_;

        //
        // Double-Q Learning update over the mini-batch:
        //   action* = argmax_a Q_online(s', a)     (action selection by online network)
        //   target  = r + gamma * Q_target(s', action*)  (evaluation by target network)
        //   loss    = (Q_online(s, a) - target)^2
        //
        for (int32_t batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
            const int32_t transitionIndex = batchIndicesBuffer_[batchIndex];

            const double* statePointer = replayBuffer_.getStatePointer(transitionIndex);
            const double* nextStatePointer = replayBuffer_.getNextStatePointer(transitionIndex);
            const double* actionPointer = replayBuffer_.getActionPointer(transitionIndex);
            const double transitionReward = replayBuffer_.getReward(transitionIndex);
            const bool isTerminal = replayBuffer_.isTerminal(transitionIndex);

            const auto stateView = std::span(statePointer, stateSize);
            const auto nextStateView = std::span(nextStatePointer, stateSize);
            const auto actionView = std::span(actionPointer, actionSize);

            // Compute Q_online(s, a) for the taken action
            const double onlineQValue = onlineNetwork_.computeQValue(
                stateView, actionView, config,
                std::span(scratchpadA_),
                std::span(scratchpadB_));

            // Step 2: Find the best next-state action using the ONLINE network (Double-Q trick)
            double bestNextOnlineQValue = -1e30;
            int bestNextActionIndex = 0;

            for (int nextActionIndex = 0; nextActionIndex < actionSize; ++nextActionIndex) {
                std::fill_n(scratchpadA_.begin(), actionSize, 0.0);
                scratchpadA_[nextActionIndex] = 1.0;

                const double nextQValue = onlineNetwork_.computeQValue(
                    nextStateView,
                    std::span<const double>(scratchpadA_.data(), actionSize),
                    config,
                    std::span(scratchpadA_.data() + actionSize, scratchpadA_.size() - actionSize),
                    std::span(scratchpadB_));

                if (nextQValue > bestNextOnlineQValue) {
                    bestNextOnlineQValue = nextQValue;
                    bestNextActionIndex = nextActionIndex;
                }
            }

            //  Evaluate Q_target(s', argmax_a Q_online(s', a)) using the TARGET network
            double bootstrappedValue = 0.0;
            if (!isTerminal) {
                std::fill_n(scratchpadA_.begin(), actionSize, 0.0);
                scratchpadA_[bestNextActionIndex] = 1.0;

                bootstrappedValue = targetNetwork_.computeQValue(
                    nextStateView,
                    std::span<const double>(scratchpadA_.data(), actionSize),
                    config,
                    std::span(scratchpadA_.data() + actionSize, scratchpadA_.size() - actionSize),
                    std::span(scratchpadB_));
            }

            //  Compute the Bellman target: y = r + gamma * Q_target(s', a*)
            const double bellmanTarget = transitionReward + gamma * bootstrappedValue;

            //  Temporal difference error gradient: dL/dQ = 2 * (Q_online - target) / batchSize
            const double temporalDifferenceError = 2.0 * (onlineQValue - bellmanTarget)
                / static_cast<double>(batchSize);

            // Backpropagate the scalar TD error through the online network
            outputGradientScratchpad_[0] = temporalDifferenceError;

            TensorOps::denseBackwardPass(
                stateView,
                std::span<const double>(outputGradientScratchpad_.data(), 1),
                rawOnlineWeights_,
                std::span(onlineWeightGradients_),
                std::span(inputGradientScratchpad_));

            // Bias gradient for the output layer: dL/dB = dL/dY (the TD error).
            // Accumulated across the mini-batch (gradients were zeroed before the loop).
            onlineBiasGradients_[0] += temporalDifferenceError;
        }

        // Apply Adam optimizer to online network weights and biases
        TensorOps::applyAdamUpdate(
            onlineNetwork_.getNetworkWeights(),
            std::span<const double>(onlineWeightGradients_),
            std::span(adamWeightMomentum_),
            std::span(adamWeightVelocity_),
            learningRate, 0.9, 0.999, 1e-8, trainingStepCounter_);

        TensorOps::applyAdamUpdate(
            onlineNetwork_.getNetworkBiases(),
            std::span<const double>(onlineBiasGradients_),
            std::span(adamBiasMomentum_),
            std::span(adamBiasVelocity_),
            learningRate, 0.9, 0.999, 1e-8, trainingStepCounter_);
    }

    void DoubleDeepQNetworkSpartanModel::decayExploration() {
        const auto* config = typedConfig();
        if (!config) return;

        if (auto* mutableConfig = const_cast<BaseHyperparameterConfig*>(&config->baseConfig); mutableConfig->epsilon > mutableConfig->epsilonMin) {
            mutableConfig->epsilon *= mutableConfig->epsilonDecay;
            if (mutableConfig->epsilon < mutableConfig->epsilonMin) {
                mutableConfig->epsilon = mutableConfig->epsilonMin;
            }
        }
    }

    std::span<const double> DoubleDeepQNetworkSpartanModel::getCriticWeights() const noexcept {
        return criticWeightsSpan_;
    }

}