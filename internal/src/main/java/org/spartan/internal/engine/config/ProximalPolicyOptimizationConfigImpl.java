package org.spartan.internal.engine.config;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.ProximalPolicyOptimizationConfig;
import org.spartan.api.engine.config.SpartanModelType;

public record ProximalPolicyOptimizationConfigImpl(

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
) implements ProximalPolicyOptimizationConfig {

    @Contract(pure = true)
    @Override
    public @NotNull SpartanModelType modelType() {
        return SpartanModelType.PROXIMAL_POLICY_OPTIMIZATION;
    }
}