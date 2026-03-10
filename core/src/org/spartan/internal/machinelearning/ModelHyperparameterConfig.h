//
// Created by Alepando on 24/2/2026.
//

#pragma once

#include <cstdint>

/**
 * @file ModelHyperparameterConfig.h
 * @brief C-compatible POD structs holding all tunable hyperparameters for ML models.
 *
 * Every struct in this file uses Standard Layout to guarantee ABI compatibility
 * with C and direct memory mapping via Java FFM (Foreign Function & Memory API).
 * No constructors, no virtual methods, no inheritance  -  pure POD.
 *
 * Java allocates these in off-heap memory via @c MemorySegment and passes a raw
 * pointer to the C++ side.  Both sides read/write the same memory region with
 * zero serialization overhead.
 *
 * @note All fields are intentionally public and trivially copyable.
 */

extern "C" {

    //
    //  Base Configuration  -  shared by every model type
    //

    /**
     * @struct BaseHyperparameterConfig
     * @brief Generic hyperparameter block common to all ML model families.
     *
     * Embedded **by composition** inside each specialised config struct so
     * that Java only needs to lay out a single contiguous @c MemorySegment.
     */
    struct BaseHyperparameterConfig {

        /** @brief Step size for gradient descent updates. Typical range: [1e-5, 1e-1]. */
        double learningRate;

        /** @brief Discount factor for future rewards in temporal-difference learning. Range: [0.0, 1.0]. */
        double gamma;

        /** @brief Current exploration probability for epsilon-greedy policies. Range: [0.0, 1.0]. */
        double epsilon;

        /** @brief Minimum floor for epsilon after decay. */
        double epsilonMin;

        /** @brief Multiplicative decay applied to epsilon after each episode. Range: [0.0, 1.0]. */
        double epsilonDecay;

        /** @brief Number of elements in the observation/state vector. */
        int32_t stateSize;

        /** @brief Number of elements in the action output vector. */
        int32_t actionSize;

        /** @brief Flag indicating whether the model is in training mode (true) or inference-only (false). */
        bool isTraining;
    };

    //
    //  Recurrent Soft Actor-Critic (RSAC) Configuration
    //

    /**
     * @struct RecurrentSoftActorCriticHyperparameterConfig
     * @brief Extended config for the Recurrent SAC model family.
     *
     * Contains the base config by composition, plus parameters for the
     * GRU recurrent layer, dual Soft Q-Networks, and entropy tuning.
     */
    struct RecurrentSoftActorCriticHyperparameterConfig {

        /** @brief Common hyperparameters inherited by composition. */
        BaseHyperparameterConfig baseConfig;

        /** @brief Dimensionality of the GRU hidden state vector. */
        int32_t hiddenStateSize;

        /** @brief Soft target-network update coefficient (Polyak averaging). Range: [0.001, 0.05]. */
        double targetSmoothingCoefficient;

        /** @brief Weight for the entropy bonus in the SAC objective. Auto-tuned when negative. */
        double entropyTemperatureAlpha;

        /** @brief Learning rate for the first Soft Q-Network critic. */
        double firstCriticLearningRate;

        /** @brief Learning rate for the second Soft Q-Network critic. */
        double secondCriticLearningRate;

        /** @brief Learning rate for the Gaussian policy (actor) network. */
        double policyNetworkLearningRate;

        /** @brief Number of input features fed into the GRU layer. */
        int32_t recurrentInputFeatureCount;

        /** @brief Number of recurrent layers stacked in the GRU. */
        int32_t recurrentLayerDepth;
    };

    //
    //  Double Deep Q-Network (DDQN) Configuration
    //

    /**
     * @struct DoubleDeepQNetworkHyperparameterConfig
     * @brief Extended config for the Double DQN model family.
     *
     * Contains the base config by composition, plus parameters for the
     * target-network synchronisation and experience-replay sizing.
     */
    struct DoubleDeepQNetworkHyperparameterConfig {

        /** @brief Common hyperparameters inherited by composition. */
        BaseHyperparameterConfig baseConfig;

        /** @brief Number of ticks between hard copies from online -> target network. */
        int32_t targetNetworkSyncInterval;

        /** @brief Maximum number of transitions stored in the replay buffer. */
        int32_t replayBufferCapacity;

        /** @brief Number of transitions sampled per learning step. */
        int32_t replayBatchSize;

        /** @brief Number of hidden units in each fully-connected layer. */
        int32_t hiddenLayerNeuronCount;

        /** @brief Number of hidden layers in the Q-network. */
        int32_t hiddenLayerCount;
    };

    //
    //  AutoEncoder Compressor Configuration
    //

    /**
     * @struct AutoEncoderCompressorHyperparameterConfig
     * @brief Extended config for the AutoEncoder compressor model family.
     *
     * Contains the base config by composition, plus the latent-space
     * dimensionality and encoder/decoder layer sizing.
     */
    struct AutoEncoderCompressorHyperparameterConfig {

        /** @brief Common hyperparameters inherited by composition. */
        BaseHyperparameterConfig baseConfig;

        /** @brief Dimensionality of the bottleneck (latent) vector. */
        int32_t latentDimensionSize;

        /** @brief Number of hidden units in each encoder/decoder layer. */
        int32_t encoderHiddenNeuronCount;

        /** @brief Number of layers in the encoder stack. */
        int32_t encoderLayerCount;

        /** @brief Number of layers in the decoder stack (usually mirrors encoder). */
        int32_t decoderLayerCount;

        /** @brief L2 regularisation penalty applied to the bottleneck activations. */
        double bottleneckRegularisationWeight;
    };

}


