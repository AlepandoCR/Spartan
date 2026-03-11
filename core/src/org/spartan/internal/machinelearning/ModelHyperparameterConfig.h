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

    /** @brief Maximum number of nested AutoEncoder slots per agent. */
    constexpr int32_t SPARTAN_MAX_NESTED_ENCODER_SLOTS = 16;

    //
    //  Nested Encoder Slot Descriptor  -  used inside agent configs
    //

    /**
     * @struct NestedEncoderSlotDescriptor
     * @brief Describes where a single nested AutoEncoder reads from the context buffer.
     *
     * Java populates an array of these descriptors contiguously inside the
     * agent's hyperparameter config.  C++ reads them during construction to
     * build the internal encoder bank.  Pure POD, Standard Layout.
     */
    struct NestedEncoderSlotDescriptor {

        /** @brief Zero-based index into the context buffer where this encoder's input slice begins. */
        int32_t contextSliceStartIndex;

        /** @brief Number of double-precision elements this encoder reads from the context. */
        int32_t contextSliceElementCount;

        /** @brief Dimensionality of the compressed latent output vector. */
        int32_t latentDimensionSize;

        /** @brief Number of hidden neurons in the encoder's dense layer. */
        int32_t hiddenNeuronCount;
    };

    /**
     * @enum SpartanModelType
     * @brief Discriminator identifying which model family a config belongs to.
     *
     * Java sets this as the first field of every config struct so that the
     * C++ registration logic can determine which concrete model to construct
     * without inspecting the full struct layout.
     */
    enum SpartanModelType : int32_t {
        SPARTAN_MODEL_TYPE_DEFAULT                       = 0,
        SPARTAN_MODEL_TYPE_RECURRENT_SOFT_ACTOR_CRITIC   = 1,
        SPARTAN_MODEL_TYPE_DOUBLE_DEEP_Q_NETWORK         = 2,
        SPARTAN_MODEL_TYPE_AUTO_ENCODER_COMPRESSOR       = 3,
    };

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

        /**
         * @brief Discriminator identifying the concrete model family.
         *
         * This MUST be the first field in every config struct (by virtue of
         * composition) so that the engine can read it via a simple
         * @c static_cast<BaseHyperparameterConfig*>(opaquePtr)->modelTypeIdentifier
         * without knowing the full struct type.
         */
        int32_t modelTypeIdentifier;

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
     */
    struct RecurrentSoftActorCriticHyperparameterConfig {

        /** @brief Common hyperparameters inherited by composition. */
        BaseHyperparameterConfig baseConfig;

        /** @brief Dimensionality of the GRU hidden state vector. */
        int32_t hiddenStateSize;

        /** @brief Number of recurrent layers stacked in the GRU. */
        int32_t recurrentLayerDepth;

        /** @brief Number of neurons in the dense layers of the Actor. */
        int32_t actorHiddenLayerNeuronCount;

        /** @brief Number of dense layers in the Actor after the GRU. */
        int32_t actorHiddenLayerCount;

        /** @brief Number of neurons in the dense layers of the Critics. */
        int32_t criticHiddenLayerNeuronCount;

        /** @brief Number of dense layers in the Critics. */
        int32_t criticHiddenLayerCount;

        /** @brief Soft target-network update coefficient. */
        double targetSmoothingCoefficient;

        /** @brief Weight for the entropy bonus. */
        double entropyTemperatureAlpha;

        /** @brief Learning rates for the different networks. */
        double firstCriticLearningRate;
        double secondCriticLearningRate;
        double policyNetworkLearningRate;

        /** @brief Number of input features fed into the GRU layer. */
        int32_t recurrentInputFeatureCount;

        /** @brief Number of nested AutoEncoder compressors owned by this agent. Zero if none. */
        int32_t nestedEncoderCount;

        /** @brief Maximum number of ticks the Remorse Trace ring buffer can store. */
        int32_t remorseTraceBufferCapacity;

        /** @brief Minimum cosine similarity threshold for blame assignment. Range: [0.0, 1.0]. */
        double remorseMinimumSimilarityThreshold;

        /**
         * @brief Descriptors for each nested encoder slot.
         *
         * Java lays these out contiguously. C++ reads nestedEncoderCount entries.
         * Unused slots beyond nestedEncoderCount are ignored.
         */
        NestedEncoderSlotDescriptor encoderSlots[SPARTAN_MAX_NESTED_ENCODER_SLOTS];
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


