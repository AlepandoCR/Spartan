package org.spartan.api.agent.config;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * Configuration for the Recurrent Soft Actor-Critic (RSAC) model.
 * <p>
 * Mirrors the C++ struct {@code RecurrentSoftActorCriticHyperparameterConfig} from
 * {@code core/src/org/spartan/internal/machinelearning/ModelHyperparameterConfig.h}
 * <p>
 * This record contains all hyperparameters required to construct and train an RSAC agent.
 * The record is immutable and thread-safe. Use {@link Builder} for convenient construction.
 *
 * @param learningRate                   Base learning rate for gradient descent
 * @param gamma                          Discount factor for future rewards [0.0, 1.0]
 * @param epsilon                        Current exploration probability [0.0, 1.0]
 * @param epsilonMin                     Minimum epsilon floor after decay
 * @param epsilonDecay                   Multiplicative decay per episode [0.0, 1.0]
 * @param stateSize                      Number of elements in observation vector
 * @param actionSize                     Number of continuous action dimensions
 * @param isTraining                     True for training mode, false for inference
 * @param hiddenStateSize                Dimensionality of GRU hidden state
 * @param recurrentLayerDepth            Number of stacked GRU layers
 * @param actorHiddenLayerNeuronCount    Neurons per actor hidden layer
 * @param actorHiddenLayerCount          Number of actor hidden layers (post-GRU)
 * @param criticHiddenLayerNeuronCount   Neurons per critic hidden layer
 * @param criticHiddenLayerCount         Number of critic hidden layers
 * @param targetSmoothingCoefficient     Soft target network update rate (tau)
 * @param entropyTemperatureAlpha        Entropy bonus weight (alpha)
 * @param firstCriticLearningRate        Learning rate for Q1 critic
 * @param secondCriticLearningRate       Learning rate for Q2 critic
 * @param policyNetworkLearningRate      Learning rate for actor (policy) network
 * @param recurrentInputFeatureCount     Input features fed into GRU layer
 * @param remorseTraceBufferCapacity     Max ticks in Remorse Trace ring buffer
 * @param remorseMinimumSimilarityThreshold  Min cosine similarity for blame assignment
 * @param encoderSlots                   Nested AutoEncoder slot descriptors (may be null/empty)
 */
public record RecurrentSoftActorCriticConfig(
        // Base config fields
        double learningRate,
        double gamma,
        double epsilon,
        double epsilonMin,
        double epsilonDecay,
        int stateSize,
        int actionSize,
        boolean isTraining,

        // RSAC-specific fields
        int hiddenStateSize,
        int recurrentLayerDepth,
        int actorHiddenLayerNeuronCount,
        int actorHiddenLayerCount,
        int criticHiddenLayerNeuronCount,
        int criticHiddenLayerCount,
        double targetSmoothingCoefficient,
        double entropyTemperatureAlpha,
        double firstCriticLearningRate,
        double secondCriticLearningRate,
        double policyNetworkLearningRate,
        int recurrentInputFeatureCount,
        int remorseTraceBufferCapacity,
        double remorseMinimumSimilarityThreshold,
        @Nullable NestedEncoderSlotDescriptor[] encoderSlots
) implements SpartanModelConfig {

    /** Maximum number of nested encoder slots (matches C++ SPARTAN_MAX_NESTED_ENCODER_SLOTS). */
    public static final int MAX_NESTED_ENCODER_SLOTS = 16;

    @Override
    public SpartanModelType modelType() {
        return SpartanModelType.RECURRENT_SOFT_ACTOR_CRITIC;
    }

    /**
     * Returns the number of active nested encoder slots.
     *
     * @return count of encoder slots, 0 if none
     */
    public int nestedEncoderCount() {
        return encoderSlots != null ? encoderSlots.length : 0;
    }

    /**
     * Gets a specific encoder slot descriptor.
     *
     * @param index slot index (0 to nestedEncoderCount - 1)
     * @return the descriptor at the given index
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public @NotNull NestedEncoderSlotDescriptor encoderSlot(int index) {
        if (encoderSlots == null || index < 0 || index >= encoderSlots.length) {
            throw new IndexOutOfBoundsException("Invalid encoder slot index: " + index);
        }
        return encoderSlots[index];
    }

    /**
     * Creates a new builder with default values.
     *
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for {@link RecurrentSoftActorCriticConfig} with sensible defaults.
     */
    public static final class Builder {
        // Base defaults
        private double learningRate = 3e-4;
        private double gamma = 0.99;
        private double epsilon = 1.0;
        private double epsilonMin = 0.01;
        private double epsilonDecay = 0.995;
        private int stateSize = 64;
        private int actionSize = 4;
        private boolean isTraining = true;

        // RSAC defaults
        private int hiddenStateSize = 128;
        private int recurrentLayerDepth = 1;
        private int actorHiddenLayerNeuronCount = 256;
        private int actorHiddenLayerCount = 2;
        private int criticHiddenLayerNeuronCount = 256;
        private int criticHiddenLayerCount = 2;
        private double targetSmoothingCoefficient = 0.005;
        private double entropyTemperatureAlpha = 0.2;
        private double firstCriticLearningRate = 3e-4;
        private double secondCriticLearningRate = 3e-4;
        private double policyNetworkLearningRate = 3e-4;
        private int recurrentInputFeatureCount = 64;
        private int remorseTraceBufferCapacity = 1000;
        private double remorseMinimumSimilarityThreshold = 0.7;
        private NestedEncoderSlotDescriptor[] encoderSlots = null;

        private Builder() {}

        public Builder learningRate(double val) { this.learningRate = val; return this; }
        public Builder gamma(double val) { this.gamma = val; return this; }
        public Builder epsilon(double val) { this.epsilon = val; return this; }
        public Builder epsilonMin(double val) { this.epsilonMin = val; return this; }
        public Builder epsilonDecay(double val) { this.epsilonDecay = val; return this; }
        public Builder stateSize(int val) { this.stateSize = val; return this; }
        public Builder actionSize(int val) { this.actionSize = val; return this; }
        public Builder isTraining(boolean val) { this.isTraining = val; return this; }
        public Builder hiddenStateSize(int val) { this.hiddenStateSize = val; return this; }
        public Builder recurrentLayerDepth(int val) { this.recurrentLayerDepth = val; return this; }
        public Builder actorHiddenLayerNeuronCount(int val) { this.actorHiddenLayerNeuronCount = val; return this; }
        public Builder actorHiddenLayerCount(int val) { this.actorHiddenLayerCount = val; return this; }
        public Builder criticHiddenLayerNeuronCount(int val) { this.criticHiddenLayerNeuronCount = val; return this; }
        public Builder criticHiddenLayerCount(int val) { this.criticHiddenLayerCount = val; return this; }
        public Builder targetSmoothingCoefficient(double val) { this.targetSmoothingCoefficient = val; return this; }
        public Builder entropyTemperatureAlpha(double val) { this.entropyTemperatureAlpha = val; return this; }
        public Builder firstCriticLearningRate(double val) { this.firstCriticLearningRate = val; return this; }
        public Builder secondCriticLearningRate(double val) { this.secondCriticLearningRate = val; return this; }
        public Builder policyNetworkLearningRate(double val) { this.policyNetworkLearningRate = val; return this; }
        public Builder recurrentInputFeatureCount(int val) { this.recurrentInputFeatureCount = val; return this; }
        public Builder remorseTraceBufferCapacity(int val) { this.remorseTraceBufferCapacity = val; return this; }
        public Builder remorseMinimumSimilarityThreshold(double val) { this.remorseMinimumSimilarityThreshold = val; return this; }
        public Builder encoderSlots(NestedEncoderSlotDescriptor[] val) { this.encoderSlots = val; return this; }

        /**
         * Builds the immutable config instance.
         *
         * @return a new RecurrentSoftActorCriticConfig
         */
        public RecurrentSoftActorCriticConfig build() {
            return new RecurrentSoftActorCriticConfig(
                    learningRate, gamma, epsilon, epsilonMin, epsilonDecay,
                    stateSize, actionSize, isTraining,
                    hiddenStateSize, recurrentLayerDepth,
                    actorHiddenLayerNeuronCount, actorHiddenLayerCount,
                    criticHiddenLayerNeuronCount, criticHiddenLayerCount,
                    targetSmoothingCoefficient, entropyTemperatureAlpha,
                    firstCriticLearningRate, secondCriticLearningRate, policyNetworkLearningRate,
                    recurrentInputFeatureCount, remorseTraceBufferCapacity,
                    remorseMinimumSimilarityThreshold, encoderSlots
            );
        }
    }
}
