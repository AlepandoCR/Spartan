package org.spartan.api.agent.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanAgent;
import org.spartan.api.agent.config.CuriosityDrivenRecurrentSoftActorCriticConfig;

import org.spartan.api.SpartanApi;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.context.SpartanContext;

/**
 * An exploration-focused version of RSAC.
 * <p>
 * This agent will invent its own mini-games (curiosity) to learn how the world works even if you don't give it points.
 */
public interface CuriosityDrivenRecurrentSoftActorCriticModel extends SpartanAgent<CuriosityDrivenRecurrentSoftActorCriticConfig> {
    /**
     * Helper to build this specific model type.
     *
     * @param api the API instance
     * @param config the specific Curiosity config
     * @param context the context
     * @param actions the action manager
     * @return the new model
     */
    static CuriosityDrivenRecurrentSoftActorCriticModel build(
            @NotNull SpartanApi api,
            @NotNull String identifier,
            @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions) {
        return api.createCuriosityDrivenRecurrentSoftActorCritic(
                identifier,
                config,
                context,
                actions
        );
    }
}
