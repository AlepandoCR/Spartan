package org.spartan.api.agent.config;

/**
 * Base configuration interface for all Spartan ML models.
 * <p>
 * Mirrors the C++ struct {@code BaseHyperparameterConfig} from
 * {@code core/src/org/spartan/internal/machinelearning/ModelHyperparameterConfig.h}
 * <p>
 * This is a sealed interface - all concrete config types must be declared as permits.
 * This enables exhaustive pattern matching and guarantees type safety when
 * serializing configs to C-compatible memory layouts.
 *
 * @see RecurrentSoftActorCriticConfig
 * @see DoubleDeepQNetworkConfig
 * @see AutoEncoderCompressorConfig
 * @see CuriosityDrivenRecurrentSoftActorCriticConfig
 * @see SpartanModelType
 */
public sealed interface SpartanModelConfig
        permits RecurrentSoftActorCriticConfig, DoubleDeepQNetworkConfig, AutoEncoderCompressorConfig,
                CuriosityDrivenRecurrentSoftActorCriticConfig {

    /**
     * Returns the model type discriminator.
     * This is written as the first field (int32_t) in the native struct
     * so C++ can identify the concrete config type.
     *
     * @return the model type enum
     */
    SpartanModelType modelType();

    /**
     * Returns the step size for the optimizer's gradient descent updates.
     * <p>
     * <b>Concept:</b> Learning Rate controls how "fast" the agent learns from new experiences.
     * Think of it as the size of the steps a hiker takes down a mountain.
     * <ul>
     *   <li><b>High Learning Rate (e.g., 0.1):</b> The agent learns quickly but might overshoot the optimal solution, becoming unstable. Ideally used early in training.</li>
     *   <li><b>Low Learning Rate (e.g., 0.0001):</b> The agent learns slowly but safely, refining its behavior with precision. Ideally used for fine-tuning.</li>
     * </ul>
     * Typical range: [1e-5, 1e-3].
     *
     * @return the learning rate
     */
    double learningRate();

    /**
     * Returns the discount factor for future rewards.
     * <p>
     * <b>Concept:</b> Gamma determines how much the agent cares about the future versus the immediate present.
     * <ul>
     *   <li><b>Gamma = 0.0:</b> The agent is short-sighted and only cares about the immediate reward from the very next action.</li>
     *   <li><b>Gamma = 0.99:</b> The agent is far-sighted and values long-term cumulative rewards nearly as much as immediate ones.</li>
     * </ul>
     * In complex games or tasks where an action now affects the outcome seconds later, a high gamma (0.90 - 0.99) is essential.
     * Range: [0.0, 1.0) (usually close to 1.0).
     *
     * @return the discount factor (gamma)
     */
    double gamma();

    /**
     * Returns the initial exploration probability for epsilon-greedy strategies.
     * <p>
     * <b>Concept:</b> Epsilon represents the chance that the agent will ignore its training and do something random to discover new possibilities.
     * <ul>
     *   <li><b>Epsilon = 1.0:</b> 100% Random. The agent explores purely by trial and error.</li>
     *   <li><b>Epsilon = 0.0:</b> 0% Random. The agent purely exploits what it already knows (inference mode).</li>
     *   <li><b>Epsilon = 0.1:</b> 10% Random. The agent mostly uses its brain but occasionally tries something new.</li>
     * </ul>
     * Range: [0.0, 1.0].
     *
     * @return the initial epsilon value
     */
    double epsilon();

    /**
     * Returns the minimum possible value for epsilon after decay.
     * <p>
     * <b>Concept:</b> Even after long training, you often want the agent to keep a tiny bit of "curiosity" or randomness
     * to prevent it from getting stuck in a loop. This sets the floor for the exploration rate.
     *
     * @return the minimum epsilon value
     */
    double epsilonMin();

    /**
     * Returns the decay rate applied to epsilon after each training step.
     * <p>
     * <b>Concept:</b> As the agent gets smarter, it needs to explore less and exploit more.
     * This factor reduces epsilon over time: {@code new_epsilon = old_epsilon * decay}.
     * <ul>
     *   <li><b>0.999:</b> Very slow decay. The agent stays curious for a long time.</li>
     *   <li><b>0.900:</b> Fast decay. The agent quickly switches to serious business.</li>
     * </ul>
     *
     * @return the epsilon decay multiplicand
     */
    double epsilonDecay();

    /**
     * Returns whether debug logging is enabled for the native engine.
     *
     * @return true to enable debug logs, false to suppress them
     */
    boolean debugLogging();

    /**
     * Returns whether the model is in training mode.
     * <p>
     * <b>Concept:</b>
     * <ul>
     *   <li><b>True:</b> The agent learns from its mistakes, updates its neural weights, and explores (uses epsilon).</li>
     *   <li><b>False:</b> The agent is "frozen". It only uses what it has already learned to make the best possible decisions (Production/Inference mode).</li>
     * </ul>
     *
     * @return true if learning is enabled, false for read-only inference
     */
    boolean isTraining();

}
