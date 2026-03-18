package org.spartan.api.engine.action.type;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Range;

/**
 * Represents a single atomic capability of an Agent.
 * <p>
 * <b>Concept:</b> In Spartan's "Decentralized Scoring" architecture, acts are self-evaluating.
 * Instead of one giant function calculating points for the whole game, each Action (Jump, Shoot, Buy Item)
 * knows whether it was a good idea or not.
 * <ul>
 *   <li><b>execute (task):</b> Performs the action in the game.</li>
 *   <li><b>evaluate (award):</b> Judges the outcome of the <i>previous</i> execution.</li>
 * </ul>
 */
public interface SpartanAction {

    /**
     * Returns the unique identifier for this action.
     * Used for debugging and mapping.
     */
    @NotNull String identifier();

    /**
     * Defines the maximum physical value for this action.
     * <p>
     * The Neural Network thinks in the abstract range [-1.0, 1.0].
     * This method maps "1.0" to a concrete game value (e.g., "Set Throttle to 100%").
     *
     * @return the maximum output value
     */
    double taskMaxMagnitude();

    /**
     * Defines the minimum physical value for this action.
     * <p>
     * Maps the neural network's "-1.0" to a concrete game value (e.g., "Set Throttle to -50%").
     *
     * @return the minimum output value
     */
    double taskMinMagnitude();

    /**
     * Executes the logic.
     * <p>
     * This is where you call your action function to do whatever you want.
     * The input value is already denormalized to your range [min, max].
     *
     * @param normalizedMagnitude the value to apply
     */
    void task(double normalizedMagnitude);

    /**
     * Calculates the reward for this specific action based on its recent performance.
     * <p>
     * <b>Concept:</b> "Did doing this help?"
     * <ul>
     *   <li><b>Positive (&gt;0):</b> Good job! Do that again in this situation.</li>
     *   <li><b>Negative (&lt;0):</b> Bad idea! Avoid this next time.</li>
     *   <li><b>Zero:</b> Neutral. No strong signal.</li>
     * </ul>
     * This is summed up with all other action rewards to train the agent.
     *
     * @return reward scalar
     */
    double award();

    /**
     * <b>Internal Method (Do Not Override):</b>
     * Orchestrates the Action Lifecycle:
     * <ol>
     *   <li>Receives raw output from Neural Network [-1, 1].</li>
     *   <li>Denormalizes it to [min, max].</li>
     *   <li>Calls {@link #task(double)} to execute.</li>
     *   <li>Calls {@link #award()} to score the <i>previous</i> tick.</li>
     * </ol>
     *
     * @param modelOutput Raw model output
     * @return The reward for the previous tick
     */
    default double tick(@Range(from= -1, to= 1) double modelOutput) {
        double min = taskMinMagnitude();
        double max = taskMaxMagnitude();
        // Map [-1, 1] -> [min, max]
        // (x - (-1)) / (1 - (-1)) = (val - min) / (max - min)
        // (x + 1) / 2 = (val - min) / (max - min)
        // val = min + (x + 1) * (max - min) / 2
        double denormalized = min + (Math.max(-1.0, Math.min(1.0, modelOutput)) + 1.0) * (max - min) / 2.0;
        task(denormalized);
        return award();
    }
}
