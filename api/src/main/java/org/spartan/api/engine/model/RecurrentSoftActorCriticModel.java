package org.spartan.api.engine.model;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.SpartanAgent;
import org.spartan.api.engine.SpartanModel;
import org.spartan.api.engine.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.SpartanApi;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.context.SpartanContext;

/**
 * A sophisticated deep reinforcement learning agent for continuous control with temporal memory.
 * <p>
 * <b>Concept:</b> RSAC (Recurrent Soft Actor-Critic) combines three innovations:
 * <ul>
 *   <li><b>Recurrent (GRU):</b> Gated Recurrent Unit layer maintains a hidden state across time steps.</li>
 *   <li><b>Soft Actor-Critic:</b> Stochastic policy that learns both expected reward and entropy regularization.</li>
 *   <li><b>Twin Q-Critics:</b> Two critic networks (Q1, Q2) that reduce overestimation bias.</li>
 * </ul>
 * <p>
 * <b>Architecture:</b>
 * <ul>
 *   <li><b>Observation Processing:</b> Optional nested AutoEncoders for dimensionality reduction.</li>
 *   <li><b>GRU Layer:</b> observation/encoded_state -> hidden_state (temporal memory, evolves each tick).</li>
 *   <li><b>Gaussian Policy:</b> hidden_state -> action_mean and action_log_std (stochastic policy μ(s), σ(s)).</li>
 *   <li><b>Twin Critics:</b> hidden_state + sampled_action → Q1(s,a) and Q2(s,a).</li>
 *   <li><b>Output:</b> Action vector (continuous, typically normalized to [-1, 1]).</li>
 * </ul>
 * <p>
 * <b>Key Algorithm Features:</b>
 * <ul>
 *   <li><b>Entropy Bonus:</b> Policy is rewarded for being uncertain (high entropy) to encourage exploration.</li>
 *   <li><b>REMORSE Trace Buffer:</b> Temporal credit assignment via state similarity tracking.</li>
 *   <li><b>Advantage:</b> Natural exploration via action sampling from Gaussian distribution.</li>
 * </ul>
 * <p>
 * <b>Use Cases:</b>
 * <ul>
 *   <li><b>Robot Control:</b> Robotic arms, locomotion, manipulation with smooth continuous control.</li>
 *   <li><b>Game AI:</b> Driving simulators, flying games, physics-based games needing smooth movement.</li>
 *   <li><b>Partial Observability:</b> Tasks where you can't see everything but can remember past observations.</li>
 *   <li><b>Temporal Patterns:</b> Any task requiring understanding of sequences (velocity, acceleration, momentum).</li>
 * </ul>
 * <p>
 * <b>When to use RSAC vs DDQN:</b>
 * <ul>
 *   <li><b>RSAC:</b> Continuous actions (steering angle, motor speed), partial observability, temporal reasoning.</li>
 *   <li><b>DDQN:</b> Discrete actions (button presses), full observability, stateless (Markovian) tasks.</li>
 * </ul>
 * <p>
 * <b>Memory (Hidden State):</b>
 * Always call {@link #resetEpisode()} when episodes end, as the GRU maintains state between ticks.
 * The hidden state vector size is configurable ({@code hiddenStateSize} in config).
 *
 * @see SpartanAgent for inherited agent methods (tick, applyReward, getEpisodeReward, resetEpisode, etc.)
 * @see SpartanModel for inherited model lifecycle methods (register, saveModel, loadModel, decayExploration, etc.)
 */
public interface RecurrentSoftActorCriticModel extends SpartanAgent<RecurrentSoftActorCriticConfig> {
    /**
     * Reads the predicted action value for a specific action dimension.
     * <p>
     * After {@link #tick()}, the action output buffer contains the agent's continuous action predictions.
     * For continuous actions, these are typically in [-1, 1] and should be scaled to the actual action range.
     *
     * @param index the action dimension index (0 to actionDimensions - 1)
     * @return the predicted action value for the given dimension
     * @throws IndexOutOfBoundsException if index is out of range
     */
    double readActionValue(int index);

    /**
     * Reads all predicted action values.
     * <p>
     * Note: This allocates a new array - use sparingly, not in hot path.
     *
     * @return array of predicted action values (size = number of action dimensions)
     */
    double[] readAllActionValues();

    /**
     * Helper to build this specific model type.
     *
     * @param api the API instance
     * @param identifier unique name for this agent
     * @param config the specific RSAC config
     * @param context the observation context
     * @param actions the action manager
     * @return the new RSAC model
     */
    @Contract("_, _, _, _, _ -> new")
    static RecurrentSoftActorCriticModel build(
            @NotNull SpartanApi api,
            @NotNull String identifier,
            @NotNull RecurrentSoftActorCriticConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions) {
        return api.createRecurrentSoftActorCritic(
                identifier,
                config,
                context,
                actions
        );
    }
}
