package org.spartan.api.agent.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanAgent;
import org.spartan.api.agent.config.RecurrentSoftActorCriticConfig;

import org.spartan.api.SpartanApi;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.context.SpartanContext;

/**
 * A sophisticated agent that uses Recurrent Neural Networks (RNN/GRU) to remember the past.
 * <p>
 * <b>Use Cases:</b>
 * <ul>
 *   <li>Games where velocity/momentum matters (driving, flying).</li>
 *   <li>Tasks where the state is partially hidden (e.g., you can't see behind walls, but you remember someone went there).</li>
 *   <li>Continuous control (steering angle, throttle).</li>
 * </ul>
 */
public interface RecurrentSoftActorCriticModel extends SpartanAgent<RecurrentSoftActorCriticConfig> {
    /**
     * Helper to build this specific model type.
     *
     * @param api the API instance
     * @param identifier unique name for this agent
     * @param config the specific RSAC config
     * @param context the context
     * @param actions the action manager
     * @return the new model
     */
    static RecurrentSoftActorCriticModel build(SpartanApi api, String identifier, RecurrentSoftActorCriticConfig config, SpartanContext context, SpartanActionManager actions) {
        return api.createRecurrentSoftActorCritic(identifier, config, context, actions);
    }
}
