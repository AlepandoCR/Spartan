package org.spartan.api.engine.config.spi;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.*;

/**
 * Service Provider Interface (SPI) for creating concrete configuration implementations.
 * This pattern avoids reflection while keeping API and Internal modules decoupled.
 *
 */
public interface SpartanConfigFactoryServiceProvider {

    @Contract("_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_ -> new")
    @NotNull RecurrentSoftActorCriticConfig createRecurrentSoftActorCriticConfig(
            double learningRate,
            double gamma,
            double epsilon,
            double epsilonMin,
            double epsilonDecay,
            boolean debugLogging,
            boolean isTraining,
            int hiddenStateSize,
            int recurrentLayerDepth,
            int actorHiddenLayerNeuronCount,
            int actorHiddenLayerCount,
            int criticHiddenLayerNeuronCount,
            int criticHiddenLayerCount,
            double targetSmoothingCoefficient,
            double entropyTemperatureAlpha,
            double firstCriticLearningRate,
            double secondCriticLearningRate,
            double policyNetworkLearningRate,
            int recurrentInputFeatureCount,
            int remorseTraceBufferCapacity,
            double remorseMinimumSimilarityThreshold,
            double targetEntropy,
            double alphaLearningRate
    );
    @Contract("_,_,_,_,_,_ -> new")
    @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig createCuriosityDrivenRecurrentSoftActorCriticConfig(
            RecurrentSoftActorCriticConfig rsacConfig,
            int forwardDynamicsHiddenLayerDimensionSize,
            double intrinsicRewardScale,
            double intrinsicRewardClampingMinimum,
            double intrinsicRewardClampingMaximum,
            double forwardDynamicsLearningRate
    );

    @Contract("_,_,_,_,_,_,_,_,_,_,_,_ -> new")
    @NotNull DoubleDeepQNetworkConfig createDoubleDeepQNetworkConfig(
            double learningRate,
            double gamma,
            double epsilon,
            double epsilonMin,
            double epsilonDecay,
            boolean debugLogging,
            boolean isTraining,
            int targetNetworkSyncInterval,
            int replayBufferCapacity,
            int replayBatchSize,
            int hiddenLayerNeuronCount,
            int hiddenLayerCount
    );

    @Contract("_,_,_,_,_,_,_,_,_,_,_,_ -> new")
    @NotNull AutoEncoderCompressorConfig createAutoEncoderCompressorConfig(
            double learningRate,
            double gamma,
            double epsilon,
            double epsilonMin,
            double epsilonDecay,
            boolean debugLogging,
            boolean isTraining,
            int latentDimensionSize,
            int encoderHiddenNeuronCount,
            int encoderLayerCount,
            int decoderLayerCount,
            double bottleneckRegularisationWeight
    );

    @Contract("_,_,_,_,_,_,_,_ -> new")
    @NotNull SpartanMultiAgentGroupConfig createMultiAgentGroupConfig(
            double learningRate,
            double gamma,
            double epsilon,
            double epsilonMin,
            double epsilonDecay,
            boolean debugLogging,
            boolean isTraining,
            int maxAgents
    );
}
