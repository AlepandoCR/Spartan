package org.spartan.internal.config;
import org.spartan.api.agent.config.CuriosityDrivenRecurrentSoftActorCriticConfig;
import org.spartan.api.agent.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.agent.config.SpartanModelType;
public record CuriosityDrivenRecurrentSoftActorCriticConfigImpl(
        RecurrentSoftActorCriticConfig recurrentSoftActorCriticConfig,
        int forwardDynamicsHiddenLayerDimensionSize,
        double intrinsicRewardScale,
        double intrinsicRewardClampingMinimum,
        double intrinsicRewardClampingMaximum,
        double forwardDynamicsLearningRate
) implements CuriosityDrivenRecurrentSoftActorCriticConfig {
}
