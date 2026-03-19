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
 * with C and direct memory mapping via Java FFM.
 * Explicit padding fields (_paddingX) are included to ensure Python FFM generators
 * perfectly mirror the C++ ABI layout without guessing.
 */

extern "C" {

    constexpr int32_t SPARTAN_MAX_NESTED_ENCODER_SLOTS = 16;

    struct NestedEncoderSlotDescriptor {
        int32_t contextSliceStartIndex;
        int32_t contextSliceElementCount;
        int32_t latentDimensionSize;
        int32_t hiddenNeuronCount;
    };

    enum SpartanModelType : int32_t {
        SPARTAN_MODEL_TYPE_DEFAULT                                      = 0,
        SPARTAN_MODEL_TYPE_RECURRENT_SOFT_ACTOR_CRITIC                  = 1,
        SPARTAN_MODEL_TYPE_DOUBLE_DEEP_Q_NETWORK                        = 2,
        SPARTAN_MODEL_TYPE_AUTO_ENCODER_COMPRESSOR                      = 3,
        SPARTAN_MODEL_TYPE_CURIOSITY_DRIVEN_RECURRENT_SOFT_ACTOR_CRITIC = 4,
        SPARTAN_MODEL_TYPE_MULTI_AGENT_GROUP                            = 5,
    };

    //
    //  Base Configuration
    //

    struct BaseHyperparameterConfig {
        int32_t modelTypeIdentifier;
        int32_t _padding0; // EXPLICIT: Forces 8-byte alignment for the next double

        double learningRate;
        double gamma;
        double epsilon;
        double epsilonMin;
        double epsilonDecay;
        int32_t stateSize;
        int32_t actionSize;
        bool isTraining;
        bool debugLogging;
        uint8_t _padding1[6]; // EXPLICIT: Pads struct to exactly 64 bytes
    };

    //
    //  Recurrent Soft Actor-Critic (RSAC) Configuration
    //

    struct RecurrentSoftActorCriticHyperparameterConfig {
        BaseHyperparameterConfig baseConfig;

        int32_t hiddenStateSize;
        int32_t recurrentLayerDepth;
        int32_t actorHiddenLayerNeuronCount;
        int32_t actorHiddenLayerCount;
        int32_t criticHiddenLayerNeuronCount;
        int32_t criticHiddenLayerCount;

        double targetSmoothingCoefficient;
        double entropyTemperatureAlpha;
        double firstCriticLearningRate;
        double secondCriticLearningRate;
        double policyNetworkLearningRate;

        int32_t recurrentInputFeatureCount;
        int32_t nestedEncoderCount;
        int32_t remorseTraceBufferCapacity;
        int32_t _padding2; // EXPLICIT: Forces 8-byte alignment for the next double

        double remorseMinimumSimilarityThreshold;

        NestedEncoderSlotDescriptor encoderSlots[SPARTAN_MAX_NESTED_ENCODER_SLOTS];
    };

    //
    //  Double Deep Q-Network (DDQN) Configuration
    //

    struct DoubleDeepQNetworkHyperparameterConfig {
        BaseHyperparameterConfig baseConfig;
        int32_t targetNetworkSyncInterval;
        int32_t replayBufferCapacity;
        int32_t replayBatchSize;
        int32_t hiddenLayerNeuronCount;
        int32_t hiddenLayerCount;
        int32_t _padding3; // EXPLICIT: Pads struct to multiple of 8
    };

    //
    //  AutoEncoder Compressor Configuration
    //

    struct AutoEncoderCompressorHyperparameterConfig {
        BaseHyperparameterConfig baseConfig;
        int32_t latentDimensionSize;
        int32_t encoderHiddenNeuronCount;
        int32_t encoderLayerCount;
        int32_t decoderLayerCount;
        double bottleneckRegularisationWeight;
    };

    //
    //  Curiosity-Driven Recurrent Soft Actor-Critic Configuration
    //

    struct CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig {
        RecurrentSoftActorCriticHyperparameterConfig recurrentSoftActorCriticConfig;

        int32_t forwardDynamicsHiddenLayerDimensionSize;
        int32_t _padding4; // EXPLICIT: Forces 8-byte alignment for the next double

        double intrinsicRewardScale;
        double intrinsicRewardClampingMinimum;
        double intrinsicRewardClampingMaximum;
        double forwardDynamicsLearningRate;
    };

    //
    //  Multi-Agent Group Configuration
    //

    struct SpartanMultiAgentGroupHyperparameterConfig {
        BaseHyperparameterConfig baseConfig;
        int32_t maxAgents;
        int32_t _padding5; // EXPLICIT: Pads struct to 8-byte alignment
    };

}