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
        const int hiddenLayers = config->hiddenLayerCount;
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

        // Multi-layer training buffers
        combinedInputBuffer_.resize(combinedInputSize);
        layerActivationBuffer_.resize(static_cast<size_t>(hiddenLayers) * static_cast<size_t>(hiddenSize));

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
        const int hiddenSize = config->hiddenLayerNeuronCount;
        const int hiddenLayers = config->hiddenLayerCount;
        const int combinedInputSize = stateSize + actionSize;
        const double gamma = config->baseConfig.gamma;
        const double learningRate = config->baseConfig.learningRate;

        // Sample a random mini-batch of transition indices
        replayBuffer_.sampleBatchIndices(
            std::span(batchIndicesBuffer_.data(), batchSize), batchSize);

        // Zero the gradient accumulators before the batch
        std::ranges::fill(onlineWeightGradients_, 0.0);
        std::ranges::fill(onlineBiasGradients_, 0.0);

        ++trainingStepCounter_;

        // Precompute layer offsets
        std::vector<size_t> layerWeightOffsets(static_cast<size_t>(hiddenLayers));
        std::vector<size_t> layerBiasOffsets(static_cast<size_t>(hiddenLayers));
        std::vector<int> layerInputSizes(static_cast<size_t>(hiddenLayers));
        size_t weightOffset = 0;
        size_t biasOffset = 0;

        // Compute offsets for each hidden layer's weights and biases in the flat buffer
        for (int layer = 0; layer < hiddenLayers; ++layer) {
            const int inputSize = (layer == 0) ? combinedInputSize : hiddenSize;
            layerWeightOffsets[layer] = weightOffset;
            layerBiasOffsets[layer] = biasOffset;
            layerInputSizes[layer] = inputSize;
            weightOffset += static_cast<size_t>(hiddenSize) * static_cast<size_t>(inputSize);
            biasOffset += static_cast<size_t>(hiddenSize);
        }


        const size_t outputWeightOffset = weightOffset;
        const size_t outputBiasOffset = biasOffset;

        // Double-Q Learning update over the mini-batch
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

            // Forward pass with activation caching for the taken action
            std::copy(stateView.begin(), stateView.end(), combinedInputBuffer_.begin());
            std::copy(actionView.begin(), actionView.end(), combinedInputBuffer_.begin() + stateSize);

            std::span<const double> currentInput = std::span<const double>(combinedInputBuffer_.data(), combinedInputBuffer_.size());

            // Forward pass through hidden layers with activation caching
            for (int layer = 0; layer < hiddenLayers; ++layer) {
                const auto activationSpan = std::span(
                    layerActivationBuffer_.data() + static_cast<size_t>(layer) * hiddenSize,
                    static_cast<size_t>(hiddenSize));

                const size_t wOffset = layerWeightOffsets[layer];
                const size_t bOffset = layerBiasOffsets[layer];
                const int inputSize = layerInputSizes[layer];

                TensorOps::denseForwardPass(
                    currentInput,
                    rawOnlineWeights_.subspan(wOffset, static_cast<size_t>(hiddenSize) * inputSize),
                    rawOnlineBiases_.subspan(bOffset, hiddenSize),
                    activationSpan);
                TensorOps::applyLeakyReLU(activationSpan, 0.01);

                currentInput = activationSpan;
            }

            double onlineQValue = 0.0;
            auto onlineQSpan = std::span(&onlineQValue, 1);
            TensorOps::denseForwardPass(
                currentInput,
                rawOnlineWeights_.subspan(outputWeightOffset, hiddenSize),
                rawOnlineBiases_.subspan(outputBiasOffset, 1),
                onlineQSpan);

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

            // Evaluate Q_target(s', argmax_a Q_online(s', a)) using the TARGET network
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

            // Bellman target
            const double bellmanTarget = transitionReward + gamma * bootstrappedValue;

            // TD error gradient (MSE gradient already has 2x)
            const double temporalDifferenceError = (onlineQValue - bellmanTarget)
                / static_cast<double>(batchSize);

            // Output layer gradients
            const auto lastActivation = std::span(
                layerActivationBuffer_.data() + static_cast<size_t>(hiddenLayers - 1) * hiddenSize,
                static_cast<size_t>(hiddenSize));

            for (int i = 0; i < hiddenSize; ++i) {
                onlineWeightGradients_[outputWeightOffset + static_cast<size_t>(i)] += temporalDifferenceError * lastActivation[i];
            }
            onlineBiasGradients_[outputBiasOffset] += temporalDifferenceError;

            // Gradient into last hidden layer
            auto currentGrad = std::span(outputGradientScratchpad_.data(), hiddenSize);
            for (int i = 0; i < hiddenSize; ++i) {
                currentGrad[i] = temporalDifferenceError * rawOnlineWeights_[outputWeightOffset + static_cast<size_t>(i)];
            }

            // Backprop through hidden layers
            for (int layer = hiddenLayers - 1; layer >= 0; --layer) {
                const auto activationSpan = std::span(
                    layerActivationBuffer_.data() + static_cast<size_t>(layer) * hiddenSize,
                    static_cast<size_t>(hiddenSize));

                for (int i = 0; i < hiddenSize; ++i) {
                    if (activationSpan[i] <= 0.0) {
                        currentGrad[i] *= 0.01;
                    }
                }

                const size_t wOffset = layerWeightOffsets[layer];
                const size_t bOffset = layerBiasOffsets[layer];
                const int inputSize = layerInputSizes[layer];

                for (int i = 0; i < hiddenSize; ++i) {
                    onlineBiasGradients_[bOffset + static_cast<size_t>(i)] += currentGrad[i];
                }

                const std::span<const double> layerInput = (layer == 0)
                    ? std::span<const double>(combinedInputBuffer_.data(), combinedInputBuffer_.size())
                    : std::span<const double>(
                        layerActivationBuffer_.data() + static_cast<size_t>(layer - 1) * hiddenSize,
                        static_cast<size_t>(hiddenSize));

                auto inputGrad = std::span(inputGradientScratchpad_.data(), inputSize);
                auto weightGradSpan = std::span(onlineWeightGradients_.data(), onlineWeightGradients_.size())
                    .subspan(wOffset, static_cast<size_t>(hiddenSize) * inputSize);

                TensorOps::denseBackwardPass(
                    layerInput,
                    currentGrad,
                    rawOnlineWeights_.subspan(wOffset, static_cast<size_t>(hiddenSize) * inputSize),
                    weightGradSpan,
                    inputGrad);

                if (layer > 0) {
                    for (int i = 0; i < hiddenSize; ++i) {
                        currentGrad[i] = inputGrad[i];
                    }
                }
            }
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


}