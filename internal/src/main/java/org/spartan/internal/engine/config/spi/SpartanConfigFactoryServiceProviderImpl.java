package org.spartan.internal.engine.config.spi;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.*;
import org.spartan.api.engine.config.spi.SpartanConfigFactoryServiceProvider;
import org.spartan.api.engine.config.spi.SpartanConfigRegistry;
import org.spartan.internal.engine.config.*;

public class SpartanConfigFactoryServiceProviderImpl implements SpartanConfigFactoryServiceProvider {

    public SpartanConfigFactoryServiceProviderImpl() {
        SpartanConfigRegistry.set(this);
    }

    @Override
    public @NotNull RecurrentSoftActorCriticConfig createRecurrentSoftActorCriticConfig(
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
    ) {

        return new RecurrentSoftActorCriticConfigImpl(
                learningRate,
                gamma,
                epsilon,
                epsilonMin,
                epsilonDecay,
                debugLogging,
                isTraining,
                hiddenStateSize,
                recurrentLayerDepth,
                actorHiddenLayerNeuronCount,
                actorHiddenLayerCount,
                criticHiddenLayerNeuronCount,
                criticHiddenLayerCount,
                targetSmoothingCoefficient,
                entropyTemperatureAlpha,
                firstCriticLearningRate,
                secondCriticLearningRate,
                policyNetworkLearningRate,
                recurrentInputFeatureCount,
                remorseTraceBufferCapacity,
                remorseMinimumSimilarityThreshold,
                targetEntropy,
                alphaLearningRate
        );
    }

    @Override
    public @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig createCuriosityDrivenRecurrentSoftActorCriticConfig(
            @NotNull RecurrentSoftActorCriticConfig rsacConfig,
            int forwardDynamicsHiddenLayerDimensionSize,
            double intrinsicRewardScale,
            double intrinsicRewardClampingMinimum,
            double intrinsicRewardClampingMaximum,
            double forwardDynamicsLearningRate
    ) {

        return new CuriosityDrivenRecurrentSoftActorCriticConfigImpl(
                rsacConfig,
                forwardDynamicsHiddenLayerDimensionSize,
                intrinsicRewardScale,
                intrinsicRewardClampingMinimum,
                intrinsicRewardClampingMaximum,
                forwardDynamicsLearningRate
        );
    }

    @Override
    public @NotNull DoubleDeepQNetworkConfig createDoubleDeepQNetworkConfig(
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
    ) {

        return new DoubleDeepQNetworkConfigImpl(
                learningRate,
                gamma,
                epsilon,
                epsilonMin,
                epsilonDecay,
                debugLogging,
                isTraining,
                targetNetworkSyncInterval,
                replayBufferCapacity,
                replayBatchSize,
                hiddenLayerNeuronCount,
                hiddenLayerCount
        );
    }

    @Override
    public @NotNull AutoEncoderCompressorConfig createAutoEncoderCompressorConfig(
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
    ) {

        return new AutoEncoderCompressorConfigImpl(
                learningRate,
                gamma,
                epsilon,
                epsilonMin,
                epsilonDecay,
                debugLogging,
                isTraining,
                latentDimensionSize,
                encoderHiddenNeuronCount,
                encoderLayerCount,
                decoderLayerCount,
                bottleneckRegularisationWeight
        );
    }

    @Override
    public @NotNull SpartanMultiAgentGroupConfig createMultiAgentGroupConfig(
            double learningRate,
            double gamma,
            double epsilon,
            double epsilonMin,
            double epsilonDecay,
            boolean debugLogging,
            boolean isTraining,
            int maxAgents) {

        return new SpartanMultiAgentGroupConfigImpl(
                learningRate,
                gamma,
                epsilon,
                epsilonMin,
                epsilonDecay,
                debugLogging,
                isTraining,
                maxAgents
        );
    }

    @Override
    public @NotNull ProximalPolicyOptimizationConfig createProximalPolicyOptimizationConfig(
            double learningRate,
            double gamma,
            double epsilon,
            double epsilonMin,
            double epsilonDecay,
            boolean debugLogging,
            boolean isTraining,
            int actorHiddenNeuronCount,
            int actorHiddenLayerCount,
            int criticHiddenNeuronCount,
            int criticHiddenLayerCount,
            int trajectoryBufferCapacity,
            int trainingEpochCount,
            int miniBatchSize,
            double clipRange,
            double gaeGamma,
            double gaeLambda,
            double entropyCoefficient,
            double valueLossCoefficient,
            double maxGradientNorm
    ) {
        return new ProximalPolicyOptimizationConfigImpl(
                learningRate,
                gamma,
                epsilon,
                epsilonMin,
                epsilonDecay,
                debugLogging,
                isTraining,
                actorHiddenNeuronCount,
                actorHiddenLayerCount,
                criticHiddenNeuronCount,
                criticHiddenLayerCount,
                trajectoryBufferCapacity,
                trainingEpochCount,
                miniBatchSize,
                clipRange,
                gaeGamma,
                gaeLambda,
                entropyCoefficient,
                valueLossCoefficient,
                maxGradientNorm
        );
    }
}
