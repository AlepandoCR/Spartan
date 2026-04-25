package org.spartan.api.engine.model;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.SpartanApi;
import org.spartan.api.engine.SpartanAgent;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.config.ProximalPolicyOptimizationConfig;
import org.spartan.api.engine.context.SpartanContext;

/**
 * PPO Agent interface for continuous action spaces using clipped surrogate objectives.
 */
public interface ProximalPolicyOptimizationModel extends SpartanAgent<ProximalPolicyOptimizationConfig> {

    /**
     * Reads a specific action value from the actor's output.
     */
    double readActionValue(int index);

    /**
     * Reads all predicted action values (Zero-Copy recommended via buffers).
     */
    double[] readAllActionValues();

    /**
     * Helper to build the PPO model via the Spartan API.
     */
    @Contract("_, _, _, _, _ -> new")
    static @NotNull ProximalPolicyOptimizationModel build(
            @NotNull SpartanApi api,
            @NotNull String identifier,
            @NotNull ProximalPolicyOptimizationConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions) {
        return api.createProximalPolicyOptimizationModel(
                identifier,
                config,
                context,
                actions
        );
    }
}