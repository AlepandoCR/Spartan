package org.spartan.internal.engine.config;
import org.spartan.api.engine.config.CuriosityDrivenRecurrentSoftActorCriticConfig;
import org.spartan.api.engine.config.RecurrentSoftActorCriticConfig;

public record CuriosityDrivenRecurrentSoftActorCriticConfigImpl(
        RecurrentSoftActorCriticConfig recurrentSoftActorCriticConfig,
        int forwardDynamicsHiddenLayerDimensionSize,
        double intrinsicRewardScale,
        double intrinsicRewardClampingMinimum,
        double intrinsicRewardClampingMaximum,
        double forwardDynamicsLearningRate
) implements CuriosityDrivenRecurrentSoftActorCriticConfig {
}
