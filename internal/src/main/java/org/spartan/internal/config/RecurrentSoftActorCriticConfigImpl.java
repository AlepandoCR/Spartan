package org.spartan.internal.config;
import org.jetbrains.annotations.Nullable;
import org.spartan.api.agent.config.NestedEncoderSlotDescriptor;
import org.spartan.api.agent.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.agent.config.SpartanModelType;
public record RecurrentSoftActorCriticConfigImpl(
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
        @Nullable NestedEncoderSlotDescriptor[] encoderSlots
) implements RecurrentSoftActorCriticConfig {
}
