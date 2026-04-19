//
// Created by Alepando on 24/2/2026.
//

#pragma once

#include <cstdint>
#include <cstddef>

/**
 * @file ModelHyperparameterConfig.h
 * @brief C-compatible POD structs holding all tunable hyperparameters for ML models.
 *
 * Every struct in this file uses Standard Layout to guarantee ABI compatibility
 * with C and direct memory mapping via Java FFM.
 * Explicit padding fields (_paddingX) are included to ensure Python FFM generators
 * perfectly mirror the C++ ABI layout without guessing.
 *
 */

// Compiler-agnostic packing macros for both MSVC and GCC/Clang
#if defined(_MSC_VER)
    #define PACK_BEGIN __pragma(pack(push, 1))
    #define PACK_END __pragma(pack(pop))
    #define PACKED
#else
    #define PACK_BEGIN
    #define PACK_END
    #define PACKED __attribute__((packed))
#endif

extern "C" {

    constexpr int32_t SPARTAN_MAX_NESTED_ENCODER_SLOTS = 16;

    PACK_BEGIN
    struct NestedEncoderSlotDescriptor {
        int32_t contextSliceStartIndex;
        int32_t contextSliceElementCount;
        int32_t latentDimensionSize;
        int32_t hiddenNeuronCount;
    } PACKED;
    PACK_END

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

    PACK_BEGIN
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
    } PACKED;
    PACK_END

    //
    //  Recurrent Soft Actor-Critic (RSAC) Configuration
    //

    PACK_BEGIN
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

        //  Array at end
        NestedEncoderSlotDescriptor encoderSlots[SPARTAN_MAX_NESTED_ENCODER_SLOTS];
    } PACKED;
    PACK_END

    //
    //  Double Deep Q-Network (DDQN) Configuration
    //

    PACK_BEGIN
    struct DoubleDeepQNetworkHyperparameterConfig {
        BaseHyperparameterConfig baseConfig;
        int32_t targetNetworkSyncInterval;
        int32_t replayBufferCapacity;
        int32_t replayBatchSize;
        int32_t hiddenLayerNeuronCount;
        int32_t hiddenLayerCount;
        int32_t _padding3; // EXPLICIT: Pads struct to multiple of 8
    } PACKED;
    PACK_END

    //
    //  AutoEncoder Compressor Configuration
    //

    PACK_BEGIN
    struct AutoEncoderCompressorHyperparameterConfig {
        BaseHyperparameterConfig baseConfig;
        int32_t latentDimensionSize;
        int32_t encoderHiddenNeuronCount;
        int32_t encoderLayerCount;
        int32_t decoderLayerCount;
        double bottleneckRegularisationWeight;
    } PACKED;
    PACK_END

    //
    //  Curiosity-Driven Recurrent Soft Actor-Critic Configuration
    //

    PACK_BEGIN
    struct CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig {
        RecurrentSoftActorCriticHyperparameterConfig recurrentSoftActorCriticConfig;

        int32_t forwardDynamicsHiddenLayerDimensionSize;
        int32_t _padding4; // Forces 8-byte alignment for the next double

        double intrinsicRewardScale;
        double intrinsicRewardClampingMinimum;
        double intrinsicRewardClampingMaximum;
        double forwardDynamicsLearningRate;
    } PACKED;
    PACK_END

    //  Compile-Time Layout Validation
    // These static_asserts ensure C++ struct layouts match Java FFM offsets exactly.
    static_assert(sizeof(BaseHyperparameterConfig) == 64, "BaseHyperparameterConfig must be 64 bytes");
    static_assert(sizeof(RecurrentSoftActorCriticHyperparameterConfig) == 424, "RecurrentSoftActorCriticHyperparameterConfig must be 424 bytes");
    static_assert(sizeof(CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig) == 464, "CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig must be 464 bytes");
    static_assert(offsetof(RecurrentSoftActorCriticHyperparameterConfig, hiddenStateSize) == 64, "RSAC.hiddenStateSize offset must be 64 bytes");
    static_assert(offsetof(RecurrentSoftActorCriticHyperparameterConfig, recurrentInputFeatureCount) == 88, "RSAC.recurrentInputFeatureCount offset must be 88 bytes");
    static_assert(offsetof(RecurrentSoftActorCriticHyperparameterConfig, encoderSlots) == 168, "RSAC.encoderSlots offset must be 168 bytes");
    static_assert(offsetof(CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig, forwardDynamicsHiddenLayerDimensionSize) == 424, "CuriosityRSAC.forwardDynamicsHiddenLayerDimensionSize offset must be 424 bytes");

    //
    //  Multi-Agent Group Configuration
    //

    PACK_BEGIN
    struct SpartanMultiAgentGroupHyperparameterConfig {
        BaseHyperparameterConfig baseConfig;
        int32_t maxAgents;
        int32_t _padding5; // EXPLICIT: Pads struct to 8-byte alignment
    } PACKED;
    PACK_END

} // extern "C"
