package org.spartan.api.engine.model;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.SpartanAgent;
import org.spartan.api.engine.SpartanModel;
import org.spartan.api.engine.config.DoubleDeepQNetworkConfig;

import org.spartan.api.SpartanApi;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.context.SpartanContext;

/**
 * A classic deep reinforcement learning agent for discrete action spaces.
 * <p>
 * <b>Concept:</b> DDQN (Double Deep Q-Network) combines Deep Q-Learning with two improvements:
 * <ul>
 *   <li><b>Neural Networks:</b> Uses deep networks to approximate the action-value function Q(s,a).</li>
 *   <li><b>Experience Replay:</b> Stores transitions and trains on random mini-batches for stability.</li>
 *   <li><b>Double Q-Learning:</b> Uses two networks (online and target) to reduce overestimation bias.</li>
 * </ul>
 * <p>
 * <b>Architecture:</b>
 * <ul>
 *   <li><b>Online Network:</b> observation + action → dense layers → single Q-value (trained on every batch)</li>
 *   <li><b>Target Network:</b> identical but frozen, updated periodically (every N ticks)</li>
 *   <li><b>Output:</b> One Q-value per discrete action (size = number of actions)</li>
 * </ul>
 * <p>
 * <b>Use Cases:</b>
 * <ul>
 *   <li><b>Games with discrete buttons:</b> Linear games (Up, Down, Left, Right, Fire, etc.).</li>
 *   <li><b>Strategic decisions:</b> Chess moves, card selection, menu navigation.</li>
 *   <li><b>Robotic manipulation:</b> Discrete gripper actions (open/close/rotate).</li>
 *   <li><b>Any task with finite action set:</b> Where continuous control is inappropriate.</li>
 * </ul>
 * <p>
 * <b>When NOT to use DDQN:</b>
 * <ul>
 *   <li>Continuous action spaces (steering angle, joint angles, velocity) -> use RSAC instead.</li>
 *   <li>Large action spaces (>1000 actions) -> may require special exploration strategies.</li>
 *   <li>Partial observability requiring memory -> use RecurrentSoftActorCriticModel instead.</li>
 * </ul>
 *
 * @see SpartanAgent for inherited agent methods (tick, applyReward, getEpisodeReward, resetEpisode, etc.)
 * @see SpartanModel for inherited model lifecycle methods (register, saveModel, loadModel, decayExploration, etc.)
 */
public interface DoubleDeepQNetworkModel extends SpartanAgent<DoubleDeepQNetworkConfig> {
    /**
     * Reads the Q-value for a specific discrete action.
     * <p>
     * After {@link #tick()}, the action output buffer contains Q(s,a) for each action.
     * This method provides convenient access to individual Q-values.
     *
     * @param actionIndex the discrete action index (0 to actionCount - 1)
     * @return the Q-value estimate for the given action
     * @throws IndexOutOfBoundsException if actionIndex is out of range
     */
    double readQValue(int actionIndex);

    /**
     * Returns the index of the action with the highest Q-value (greedy action).
     * <p>
     * <b>Zero-GC:</b> No allocations, safe for hot path.
     *
     * @return the index of the best action according to current Q-estimates
     */
    int getBestActionIndex();

    /**
     * Returns all Q-values for every discrete action.
     * <p>
     * Note: This allocates a new array - use sparingly, not in hot path.
     *
     * @return array of Q-values (size = number of actions)
     */
    double[] readAllQValues();

    /**
     * Reads all Q-values into a pre-allocated buffer.
     * <p>
     * <b>Zero-GC:</b> No allocations. More efficient than {@link #readAllQValues()} when you have a buffer.
     *
     * @param buffer the array to fill (will use min(buffer.length, actionCount) elements)
     */
    void readQValuesIntoBuffer(double[] buffer);

    /**
     * Helper to build this specific model type.
     *
     * @param api the API instance
     * @param identifier unique name for this agent
     * @param config the specific DDQN config
     * @param context the observation context
     * @param actions the action manager
     * @return the new DDQN model
     */
    @Contract("_, _, _, _, _ -> new")
    static @NotNull DoubleDeepQNetworkModel build(
            @NotNull SpartanApi api,
            @NotNull String identifier,
            @NotNull DoubleDeepQNetworkConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions) {
        return api.createDoubleDeepQNetwork(
                identifier,
                config,
                context,
                actions
        );
    }
}
