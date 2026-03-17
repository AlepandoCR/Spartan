package org.spartan.api.agent.model;

import org.spartan.api.agent.SpartanAgent;
import org.spartan.api.agent.config.DoubleDeepQNetworkConfig;

import org.spartan.api.SpartanApi;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.context.SpartanContext;

/**
 * A standard agent for Discrete Action spaces.
 * <p>
 * <b>Use Cases:</b>
 * <ul>
 *   <li>Selecting a card from a deck.</li>
 *   <li>Choosing which item to buy.</li>
 *   <li>Simple movement (Grid based: Up, Down, Left, Right).</li>
 * </ul>
 */
public interface DoubleDeepQNetworkModel extends SpartanAgent<DoubleDeepQNetworkConfig> {
    /**
     * Helper to build this specific model type.
     *
     * @param api the API instance
     * @param identifier unique name for this agent
     * @param config the specific DDQN config
     * @param context the context
     * @param actions the action manager
     * @return the new model
     */
    static DoubleDeepQNetworkModel build(SpartanApi api, String identifier, DoubleDeepQNetworkConfig config, SpartanContext context, SpartanActionManager actions) {
        return api.createDoubleDeepQNetwork(identifier, config, context, actions);
    }
}
