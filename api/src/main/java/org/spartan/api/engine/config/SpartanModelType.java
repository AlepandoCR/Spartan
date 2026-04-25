package org.spartan.api.engine.config;

import org.jetbrains.annotations.NotNull;

/**
 * Discriminator identifying which model family a config belongs to.
 * <p>
 * Mirrors the C++ enum {@code SpartanModelType} from
 * {@code core/src/org/spartan/internal/machinelearning/ModelHyperparameterConfig.h}
 * <p>
 * Java sets this as the first field of every config struct so that the
 * C++ registration logic can determine which concrete model to construct
 * without inspecting the full struct layout.
 * <p>
 * The ordinal values must match the C++ enum exactly:
 * <pre>
 * enum SpartanModelType : int32_t {
 *     SPARTAN_MODEL_TYPE_DEFAULT                     = 0,
 *     SPARTAN_MODEL_TYPE_RECURRENT_SOFT_ACTOR_CRITIC = 1,
 *     SPARTAN_MODEL_TYPE_DOUBLE_DEEP_Q_NETWORK       = 2,
 *     SPARTAN_MODEL_TYPE_AUTO_ENCODER_COMPRESSOR     = 3,
 *     SPARTAN_MODEL_TYPE_CURIOSITY_DRIVEN_RECURRENT_SOFT_ACTOR_CRITIC = 4,
 *     SPARTAN_MODEL_TYPE_MULTI_AGENT_GROUP           = 5,
 * };
 * </pre>
 */
public enum SpartanModelType {

    /**
     * Default/placeholder model type.
     */
    DEFAULT(0),

    /**
     * Recurrent Soft Actor-Critic (RSAC) model.
     * Combines GRU-based recurrence with continuous action SAC.
     */
    RECURRENT_SOFT_ACTOR_CRITIC(1),

    /**
     * Double Deep Q-Network (DDQN) model.
     * Discrete action space with target network and experience replay.
     */
    DOUBLE_DEEP_Q_NETWORK(2),

    /**
     * AutoEncoder Compressor model.
     * Representation learning for dimensionality reduction.
     */
    AUTO_ENCODER_COMPRESSOR(3),

    /**
     * Curiosity-Driven Recurrent Soft Actor-Critic model.
     * Extends RSAC with an Intrinsic Curiosity Module (ICM) that uses forward dynamics
     * prediction error as intrinsic reward to encourage exploration of novel states.
     */
    CURIOSITY_DRIVEN_RECURRENT_SOFT_ACTOR_CRITIC(4),

    /**
     * Multi-Agent group container (not a per-agent model).
     */
    MULTI_AGENT_GROUP(5),

    /**
     * Proximity policy optimization
     */
    PROXIMAL_POLICY_OPTIMIZATION(6);

    private final int nativeValue;

    SpartanModelType(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    /**
     * Returns the native C++ enum value.
     * This is written to the config struct's modelTypeIdentifier field.
     *
     * @return the int32_t value matching the C++ enum
     */
    public int getNativeValue() {
        return nativeValue;
    }

    /**
     * Converts a native value to the corresponding enum constant.
     *
     * @param nativeValue the C++ enum value
     * @return the matching SpartanModelType
     * @throws IllegalArgumentException if no matching type exists
     */
    public static @NotNull SpartanModelType fromNativeValue(int nativeValue) {
        for (SpartanModelType type : values()) {
            if (type.nativeValue == nativeValue) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown native model type: " + nativeValue);
    }
}
