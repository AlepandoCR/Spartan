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
 *
 * #pragma pack(1) forces BYTE-ALIGNMENT to ensure identical memory layouts
 * across all platforms (Linux, Windows, macOS). This prevents compiler-specific
 * padding that would cause Java FFM offsets to mismatch on different OSes.
 */

//
// Force byte-packing (no padding) to ensure Linux and Windows use identical layouts.
// Without this, struct sizes and offsets differ between platforms.
#pragma pack(push, 1)

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
        int32_t _padding0; // Forces 8-byte alignment for the next double

        double learningRate;
        double gamma;
        double epsilon;
        double epsilonMin;
        double epsilonDecay;
        int32_t stateSize;
        int32_t actionSize;
        bool isTraining;
        bool debugLogging;
        uint8_t _padding1[6]; // Pads struct to exactly 64 bytes
    };

    //
    //  Recurrent Soft Actor-Critic (RSAC) Configuration
    //

    struct RecurrentSoftActorCriticHyperparameterConfig {
        BaseHyperparameterConfig baseConfig;

        // All int32_t fields together (avoids alignment padding)
        int32_t hiddenStateSize;
        int32_t recurrentLayerDepth;
        int32_t actorHiddenLayerNeuronCount;
        int32_t actorHiddenLayerCount;
        int32_t criticHiddenLayerNeuronCount;
        int32_t criticHiddenLayerCount;
        int32_t recurrentInputFeatureCount;
        int32_t nestedEncoderCount;
        int32_t remorseTraceBufferCapacity;
        int32_t _padding2; // Forces 8-byte alignment for the next double

        // All double fields together (8-byte aligned)
        double targetSmoothingCoefficient;
        double entropyTemperatureAlpha;
        double firstCriticLearningRate;
        double secondCriticLearningRate;
        double policyNetworkLearningRate;
        double targetEntropy;
        double alphaLearningRate;
        double remorseMinimumSimilarityThreshold;

        // Array at end
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
        int32_t _padding3; // Pads struct to multiple of 8
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
        int32_t _padding4; // Forces 8-byte alignment for the next double

        double intrinsicRewardScale;
        double intrinsicRewardClampingMinimum;
        double intrinsicRewardClampingMaximum;
        double forwardDynamicsLearningRate;
    };

    //  Compile-Time Layout Validation
    // These static_asserts ensure C++ struct layouts match Java FFM offsets exactly.
    static_assert(sizeof(BaseHyperparameterConfig) == 64, "BaseHyperparameterConfig must be 64 bytes");
    static_assert(sizeof(RecurrentSoftActorCriticHyperparameterConfig) == 424, "RecurrentSoftActorCriticHyperparameterConfig must be 424 bytes");
    static_assert(sizeof(CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig) == 464, "CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig must be 464 bytes");

    //
    //  Multi-Agent Group Configuration
    //

    struct SpartanMultiAgentGroupHyperparameterConfig {
        BaseHyperparameterConfig baseConfig;
        int32_t maxAgents;
        int32_t _padding5; //Pads struct to 8-byte alignment
    };

} // extern "C"

#pragma pack(pop)

