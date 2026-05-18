package org.spartan.internal.engine.config;
import org.spartan.api.engine.config.CuriosityDrivenRecurrentSoftActorCriticConfig;
import org.spartan.api.engine.config.RecurrentSoftActorCriticConfig;

public record CuriosityDrivenRecurrentSoftActorCriticConfigImpl(
        RecurrentSoftActorCriticConfig recurrentSoftActorCriticConfig,
        int forwardDynamicsHiddenLayerDimensionSize,
        double intrinsicRewardScale,
        double intrinsicRewardClampingMinimum,
        double intrinsicRewardClampingMaximum,
        double forwardDynamicsLearningRate,
        // Inverse dynamics network parameters (predict action from s, s')
        int inverseDynamicsHiddenLayerDimensionSize,
        double inverseDynamicsLearningRate,
        double inverseLossWeight
) implements CuriosityDrivenRecurrentSoftActorCriticConfig {
}
