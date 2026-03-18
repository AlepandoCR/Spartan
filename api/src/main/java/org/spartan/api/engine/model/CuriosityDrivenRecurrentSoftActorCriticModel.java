package org.spartan.api.engine.model;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.SpartanAgent;
import org.spartan.api.engine.config.CuriosityDrivenRecurrentSoftActorCriticConfig;
import org.spartan.api.engine.config.RecurrentSoftActorCriticConfig;

import org.spartan.api.SpartanApi;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.context.SpartanContext;

/**
 * An exploration-driven variant of RSAC that learns through intrinsic curiosity.
 * <p>
 * <b>Concept:</b> A Curiosity-Driven agent augments RSAC with a "world model" that predicts future states.
 * If the Forward Dynamics Network's predictions are wrong, the agent receives intrinsic reward (curiosity bonus)
 * proportional to the prediction error. This allows learning even with sparse or zero external rewards.
 * <p>
 * <b>Architecture:</b>
 * <ul>
 *   <li><b>Internal RSAC Agent:</b> Standard Recurrent Soft Actor-Critic with GRU + Gaussian Policy + Twin Q-Critics.</li>
 *   <li><b>Forward Dynamics Network:</b> Dense network that predicts next_state from (current_state, current_action).</li>
 *   <li><b>Intrinsic Reward:</b> Computed as prediction error (MSE between predicted and actual next state).</li>
 *   <li><b>Total Reward:</b> extrinsic_reward + intrinsic_reward (both fed to RSAC for training).</li>
 * </ul>
 * <p>
 * <b>Intrinsic Reward Calculation:</b>
 * <pre>
 * prediction_error = MSE(predicted_next_state, actual_next_state)
 * intrinsic_reward = min(max(prediction_error * intrinsicRewardScale, min_clamp), max_clamp)
 * total_reward = extrinsic_reward + intrinsic_reward
 * </pre>
 * <p>
 * <b>REMORSE Trace Memory:</b>
 * Tracks state similarity to avoid repeatedly visiting already-well-predicted states.
 * This prevents the agent from getting stuck in "curiosity loops" on deterministic or highly explored regions.
 * <p>
 * <b>Use Cases:</b>
 * <ul>
 *   <li><b>No External Rewards:</b> Pure exploration in environments with no reward signal (research/benchmarks).</li>
 *   <li><b>Sparse Rewards:</b> Reward only at task completion (e.g., reaching goal in maze).</li>
 *   <li><b>Dynamic/Stochastic Environments:</b> Worlds where prediction is challenging and interesting.</li>
 *   <li><b>Model Learning:</b> Simultaneously learning a world model while learning a policy.</li>
 *   <li><b>Transfer Learning:</b> World model pre-training on exploration phase, then fine-tune on downstream tasks.</li>
 * </ul>
 * <p>
 * <b>When to use Curiosity vs Standard RSAC:</b>
 * <ul>
 *   <li><b>Curiosity:</b> Few/no external rewards, need autonomous exploration, want world model.</li>
 *   <li><b>Standard RSAC:</b> Clear reward signal, well-defined objective, no need for world model.</li>
 * </ul>
 * <p>
 * <b>Configuration:</b>
 * <ul>
 *   <li>{@code intrinsicRewardScale}: How much curiosity bonus affects total reward (default 0.01).</li>
 *   <li>{@code intrinsicRewardClampingMinimum/Maximum}: Prevent extreme intrinsic rewards from noisy environments.</li>
 *   <li>{@code forwardDynamicsLearningRate}: Separate learning rate for world model vs policy.</li>
 * </ul>
 *
 * @see RecurrentSoftActorCriticModel for the underlying policy algorithm
 * @see SpartanAgent for inherited agent methods (tick, applyReward, getEpisodeReward, resetEpisode, etc.)
 * @see org.spartan.api.engine.SpartanModel for inherited model lifecycle methods (register, saveModel, loadModel, decayExploration, etc.)
 */
public interface CuriosityDrivenRecurrentSoftActorCriticModel extends SpartanAgent<CuriosityDrivenRecurrentSoftActorCriticConfig> {
    /**
     * Reads the predicted action value for a specific action dimension.
     * <p>
     * After {@link #tick()}, the action output buffer contains the agent's continuous action predictions
     * based on the combined extrinsic + intrinsic reward signal.
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
     * Returns the embedded RSAC configuration that this curiosity model wraps.
     * <p>
     * The Curiosity-Driven agent is built on top of a standard RSAC agent.
     * This method provides access to that underlying RSAC configuration.
     *
     * @return the underlying RecurrentSoftActorCritic configuration
     */
    @NotNull RecurrentSoftActorCriticConfig getEmbeddedRecurrentSoftActorCriticConfig();

    /**
     * Helper to build this specific model type.
     *
     * @param api the API instance
     * @param identifier unique name for this agent
     * @param config the specific Curiosity-Driven RSAC config
     * @param context the observation context
     * @param actions the action manager
     * @return the new Curiosity-Driven RSAC model
     */
    @Contract("_, _, _, _, _ -> new")
    static @NotNull CuriosityDrivenRecurrentSoftActorCriticModel build(
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

