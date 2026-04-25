package org.spartan.api.engine.config;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.spi.SpartanConfigRegistry;

/**
 * Configuration for the Proximal Policy Optimization (PPO) model.
 * <p>
 * <b>Concept:</b> PPO is a policy-gradient algorithm for <b>continuous</b> action spaces.
 * It learns a stochastic Gaussian policy (mean + log_std per action dimension) and a
 * separate value baseline (critic) used for Generalized Advantage Estimation (GAE).
 * The "Proximal" constraint clips the policy update ratio so training stays stable —
 * large gradient steps that would destabilize the policy are discarded.
 */
public non-sealed interface ProximalPolicyOptimizationConfig extends SpartanModelConfig {

    // ==================== Actor ====================

    /** Neurons per hidden layer in the Actor (policy) network. */
    int actorHiddenNeuronCount();

    /** Number of hidden layers in the Actor network. */
    int actorHiddenLayerCount();

    // ==================== Critic ====================

    /** Neurons per hidden layer in the Critic (value baseline) network. */
    int criticHiddenNeuronCount();

    /** Number of hidden layers in the Critic network. */
    int criticHiddenLayerCount();

    // ==================== Training loop ====================

    /**
     * Number of environment steps collected before each policy update.
     * <p>
     * Larger buffers yield lower-variance gradient estimates at the cost of
     * higher memory usage and longer rollout phases.
     */
    int trajectoryBufferCapacity();

    /** Number of full passes over the trajectory buffer per update. */
    int trainingEpochCount();

    /** Mini-batch size drawn from the trajectory buffer during each epoch. */
    int miniBatchSize();

    // ==================== PPO-specific hyperparameters ====================

    /**
     * PPO clipping range (epsilon in the paper).
     * <p>
     * Constrains the policy ratio r(θ) = π_new / π_old to [1-ε, 1+ε].
     * Typical value: 0.2.
     */
    double clipRange();

    /**
     * Discount factor used inside GAE for return computation.
     * Usually equal to {@link #gamma()} but can differ.
     */
    double gaeGamma();

    /**
     * GAE lambda — bias-variance trade-off for advantage estimation.
     * <ul>
     *   <li><b>λ = 1.0:</b> Full Monte Carlo returns (low bias, high variance).</li>
     *   <li><b>λ = 0.0:</b> One-step TD (high bias, low variance).</li>
     * </ul>
     * Typical value: 0.95.
     */
    double gaeLambda();

    /**
     * Coefficient for the entropy bonus added to the policy loss.
     * Encourages exploration by discouraging the policy from collapsing to a single action.
     */
    double entropyCoefficient();

    /**
     * Coefficient applied to the value function loss before adding to the total loss.
     * Balances how much the critic is updated relative to the actor.
     */
    double valueLossCoefficient();

    /**
     * Maximum L2 norm for gradient clipping.
     * Gradients with a norm exceeding this value are rescaled.
     * Prevents catastrophic weight updates.
     */
    double maxGradientNorm();

    @Override
    default SpartanModelType modelType() {
        return SpartanModelType.PROXIMAL_POLICY_OPTIMIZATION;
    }

    @Contract(value = " -> new", pure = true)
    static @NotNull Builder builder() {
        return new Builder();
    }

    final class Builder {
        // Base SpartanModelConfig fields
        private double  learningRate  = 3e-4;
        private double  gamma         = 0.99;
        private double  epsilon       = 0.0;   // PPO has no epsilon-greedy; kept for interface compat
        private double  epsilonMin    = 0.0;
        private double  epsilonDecay  = 1.0;
        private boolean debugLogging  = false;
        private boolean isTraining    = true;

        // Actor
        private int actorHiddenNeuronCount  = 64;
        private int actorHiddenLayerCount   = 2;

        // Critic
        private int criticHiddenNeuronCount = 64;
        private int criticHiddenLayerCount  = 2;

        // Training loop
        private int trajectoryBufferCapacity = 2048;
        private int trainingEpochCount       = 10;
        private int miniBatchSize            = 64;

        // PPO-specific
        private double clipRange            = 0.2;
        private double gaeGamma             = 0.99;
        private double gaeLambda            = 0.95;
        private double entropyCoefficient   = 0.01;
        private double valueLossCoefficient = 0.5;
        private double maxGradientNorm      = 0.5;

        private Builder() {}

        // Base fields
        public Builder learningRate(double val)  { this.learningRate  = val; return this; }
        public Builder gamma(double val)         { this.gamma         = val; return this; }
        public Builder epsilon(double val)       { this.epsilon       = val; return this; }
        public Builder epsilonMin(double val)    { this.epsilonMin    = val; return this; }
        public Builder epsilonDecay(double val)  { this.epsilonDecay  = val; return this; }
        public Builder debugLogging(boolean val) { this.debugLogging  = val; return this; }
        public Builder isTraining(boolean val)   { this.isTraining    = val; return this; }

        // Actor
        public Builder actorHiddenNeuronCount(int val)  { this.actorHiddenNeuronCount  = val; return this; }
        public Builder actorHiddenLayerCount(int val)   { this.actorHiddenLayerCount   = val; return this; }

        // Critic
        public Builder criticHiddenNeuronCount(int val) { this.criticHiddenNeuronCount = val; return this; }
        public Builder criticHiddenLayerCount(int val)  { this.criticHiddenLayerCount  = val; return this; }

        // Training loop
        public Builder trajectoryBufferCapacity(int val) { this.trajectoryBufferCapacity = val; return this; }
        public Builder trainingEpochCount(int val)       { this.trainingEpochCount       = val; return this; }
        public Builder miniBatchSize(int val)            { this.miniBatchSize            = val; return this; }

        // PPO-specific
        public Builder clipRange(double val)            { this.clipRange            = val; return this; }
        public Builder gaeGamma(double val)             { this.gaeGamma             = val; return this; }
        public Builder gaeLambda(double val)            { this.gaeLambda            = val; return this; }
        public Builder entropyCoefficient(double val)   { this.entropyCoefficient   = val; return this; }
        public Builder valueLossCoefficient(double val) { this.valueLossCoefficient = val; return this; }
        public Builder maxGradientNorm(double val)      { this.maxGradientNorm      = val; return this; }

        @Contract("-> new")
        public @NotNull ProximalPolicyOptimizationConfig build() {
            return SpartanConfigRegistry.get().createProximalPolicyOptimizationConfig(
                    learningRate, gamma, epsilon, epsilonMin, epsilonDecay,
                    debugLogging, isTraining,
                    actorHiddenNeuronCount, actorHiddenLayerCount,
                    criticHiddenNeuronCount, criticHiddenLayerCount,
                    trajectoryBufferCapacity, trainingEpochCount, miniBatchSize,
                    clipRange, gaeGamma, gaeLambda,
                    entropyCoefficient, valueLossCoefficient, maxGradientNorm
            );
        }
    }
}