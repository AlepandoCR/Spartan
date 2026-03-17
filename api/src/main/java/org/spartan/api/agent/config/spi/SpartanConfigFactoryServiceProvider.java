package org.spartan.api.agent.config.spi;

import org.spartan.api.agent.config.*;

/**
 * Service Provider Interface (SPI) for creating concrete configuration implementations.
 * This pattern avoids reflection while keeping API and Internal modules decoupled.
 *
 */
public interface SpartanConfigFactoryServiceProvider {

    RecurrentSoftActorCriticConfig createRecurrentSoftActorCritic(
            double learningRate, double gamma, double epsilon, double epsilonMin, double epsilonDecay,
            boolean debugLogging, boolean isTraining,
            int hiddenStateSize, int recurrentLayerDepth, int actorHiddenLayerNeuronCount, int actorHiddenLayerCount,
            int criticHiddenLayerNeuronCount, int criticHiddenLayerCount, double targetSmoothingCoefficient,
            double entropyTemperatureAlpha, double firstCriticLearningRate, double secondCriticLearningRate,
            double policyNetworkLearningRate, int recurrentInputFeatureCount, int remorseTraceBufferCapacity,
            double remorseMinimumSimilarityThreshold, NestedEncoderSlotDescriptor[] encoderSlots
    );

    CuriosityDrivenRecurrentSoftActorCriticConfig createCuriosityDrivenRecurrentSoftActorCritic(
            RecurrentSoftActorCriticConfig rsacConfig,
            int forwardDynamicsHiddenLayerDimensionSize, double intrinsicRewardScale,
            double intrinsicRewardClampingMinimum, double intrinsicRewardClampingMaximum,
            double forwardDynamicsLearningRate
    );

    DoubleDeepQNetworkConfig createDoubleDeepQNetwork(
            double learningRate, double gamma, double epsilon, double epsilonMin, double epsilonDecay,
            boolean debugLogging, boolean isTraining,
            int targetNetworkSyncInterval, int replayBufferCapacity, int replayBatchSize,
            int hiddenLayerNeuronCount, int hiddenLayerCount
    );

    AutoEncoderCompressorConfig createAutoEncoderCompressor(
            double learningRate, double gamma, double epsilon, double epsilonMin, double epsilonDecay,
            boolean debugLogging, boolean isTraining,
            int latentDimensionSize, int encoderHiddenNeuronCount, int encoderLayerCount,
            int decoderLayerCount, double bottleneckRegularisationWeight
    );
}
