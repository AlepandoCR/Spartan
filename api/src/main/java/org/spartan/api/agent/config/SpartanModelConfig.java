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
 * @see SpartanModelType
 */
public sealed interface SpartanModelConfig
        permits RecurrentSoftActorCriticConfig, DoubleDeepQNetworkConfig, AutoEncoderCompressorConfig {

    /**
     * Returns the model type discriminator.
     * This is written as the first field (int32_t) in the native struct
     * so C++ can identify the concrete config type.
     *
     * @return the model type enum
     */
    SpartanModelType modelType();

    /**
     * Step size for gradient descent updates.
     * Typical range: [1e-5, 1e-1].
     *
     * @return the learning rate
     */
    double learningRate();

    /**
     * Discount factor for future rewards in temporal-difference learning.
     * Range: [0.0, 1.0].
     *
     * @return the discount factor (gamma)
     */
    double gamma();

    /**
     * Current exploration probability for epsilon-greedy policies.
     * Range: [0.0, 1.0].
     *
     * @return the current epsilon value
     */
    double epsilon();

    /**
     * Minimum floor for epsilon after decay.
     *
     * @return the minimum epsilon value
     */
    double epsilonMin();

    /**
     * Multiplicative decay applied to epsilon after each episode.
     * Range: [0.0, 1.0].
     *
     * @return the epsilon decay rate
     */
    double epsilonDecay();

    /**
     * Number of elements in the observation/state vector.
     *
     * @return the state size
     */
    int stateSize();

    /**
     * Number of elements in the action output vector.
     *
     * @return the action size
     */
    int actionSize();

    /**
     * Flag indicating whether the model is in training mode or inference-only.
     *
     * @return true if training, false for inference
     */
    boolean isTraining();

    // Legacy compatibility methods (deprecated, use new names)

    /** @deprecated Use {@link #learningRate()} instead */
    @Deprecated(forRemoval = true)
    default double getLearningRate() { return learningRate(); }

    /** @deprecated Use {@link #gamma()} instead */
    @Deprecated(forRemoval = true)
    default double getGamma() { return gamma(); }

    /** @deprecated Use {@link #epsilon()} instead */
    @Deprecated(forRemoval = true)
    default double getEpsilonStart() { return epsilon(); }

    /** @deprecated Use {@link #epsilonDecay()} instead */
    @Deprecated(forRemoval = true)
    default double getEpsilonDecay() { return epsilonDecay(); }

    /** @deprecated Use {@link #epsilonMin()} instead */
    @Deprecated(forRemoval = true)
    default double getEpsilonMin() { return epsilonMin(); }
}
