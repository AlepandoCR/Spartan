package org.spartan.api.agent;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.config.SpartanModelConfig;

import java.lang.foreign.MemorySegment;

/**
 * Interface for decision-making ML agents that learn from rewards.
 * <p>
 * Extends {@link SpartanModel} with support for:
 * <ul>
 *   <li>Action management (interpreting predictions as game actions)</li>
 *   <li>Critic networks (for actor-critic algorithms like RSAC)</li>
 *   <li>Reward-based learning</li>
 * </ul>
 * <p>
 * Concrete implementations:
 * <ul>
 *   <li>RecurrentSoftActorCriticModel (RSAC) - continuous actions with GRU memory</li>
 *   <li>DoubleDeepQNetworkModel (DDQN) - discrete actions with experience replay</li>
 * </ul>
 *
 * @param <C> the configuration type (must be an agent config)
 */
public interface SpartanAgent<C extends SpartanModelConfig> extends SpartanModel<C> {

    /**
     * Returns the action manager for this agent.
     * The action manager interprets the raw action output buffer as game actions.
     *
     * @return the action manager (never null)
     */
    @NotNull SpartanActionManager getActionManager();

    /**
     * Returns the critic weights buffer.
     * <p>
     * For RSAC: Contains twin Q-networks (Q1, Q2) and their target networks.
     * For DDQN: May be combined with model weights.
     *
     * @return MemorySegment containing critic weights (double[])
     */
    @NotNull MemorySegment getCriticWeightsBuffer();

    /**
     * Hot-path method: applies reward and triggers native inference in one call.
     * <p>
     * This is the preferred method for agents during training. It combines:
     * <ol>
     *   <li>Context update (Zero-GC flush to off-heap)</li>
     *   <li>Clean sizes sync for variable elements</li>
     *   <li>Reward application and inference in a single native call</li>
     * </ol>
     * <p>
     * <b>Zero-GC:</b> This method performs no allocations - safe for 20+ TPS.
     *
     * @param reward the reward signal for this tick (positive = good, negative = bad)
     * @throws IllegalStateException if agent is not registered or has been closed
     * @throws org.spartan.api.exception.SpartanNativeException if native tick fails
     */
    void tick(double reward);

    /**
     * Applies a reward signal to this agent without triggering a tick.
     * <p>
     * Use this when you want to accumulate rewards before the next tick,
     * or when using the orchestrator's global tick.
     *
     * @param reward the reward value (positive = good, negative = bad)
     */
    void applyReward(double reward);

    /**
     * Returns the current cumulative reward for this episode.
     *
     * @return the accumulated reward
     */
    double getEpisodeReward();

    /**
     * Resets the episode state (reward accumulator, hidden states, etc.).
     * Called at the start of a new episode or when the agent respawns.
     */
    void resetEpisode();
}
