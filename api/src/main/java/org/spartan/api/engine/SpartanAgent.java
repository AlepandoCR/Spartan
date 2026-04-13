package org.spartan.api.engine;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.config.SpartanModelConfig;

import java.lang.foreign.MemorySegment;

/**
 * Represents an active "Decision Maker" agent.
 * <p>
 * <b>Concept:</b> Unlike a generic model which might just process data, an Agent <i>acts</i>.
 * It has:
 * <ul>
 *   <li><b>Sensors (Context):</b> To see.</li>
 *   <li><b>Brain (Model):</b> To think.</li>
 *   <li><b>Actuators (Actions):</b> To do.</li>
 * </ul>
 * This interface adds the ability to manage Actions (the output mapping).
 *
 * @param <SpartanModelConfigType> the config type
 */
public interface SpartanAgent<SpartanModelConfigType extends SpartanModelConfig> extends SpartanModel<SpartanModelConfigType> {

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
     * Gets the episode reward and atomically resets it in one call.
     * <p>
     * This is useful to avoid race conditions where the reward is read
     * and then reset, or to ensure the reward is not lost between reads.
     *
     * @return the accumulated reward for this episode
     */
    default double getAndResetEpisodeReward() {
        double reward = getEpisodeReward();
        resetEpisode();
        return reward;
    }

    /**
     * Resets the episode state (reward accumulator, hidden states, etc.).
     * Called at the start of a new episode or when the agent respawns.
     */
    void resetEpisode();
}
