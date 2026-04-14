package org.spartan.api.engine.config;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.spi.SpartanConfigRegistry;

/**
 * Configuration for the Recurrent Soft Actor-Critic (RSAC) model.
 * See {@link Builder} for construction.
 */
public non-sealed interface RecurrentSoftActorCriticConfig extends SpartanModelConfig {

    // Base config fields
    double learningRate();
    double gamma();
    double epsilon();
    double epsilonMin();
    double epsilonDecay();
    boolean isTraining();

    /**
     * Returns the size of the recurrent hidden state (memory vector).
     * <p>
     * <b>Concept:</b> This is the agent's "short-term memory".
     * Unlike standard agents that react only to the current frame, an RSAC agent keeps a memory vector
     * that evolves over time. This allows it to understand concepts like velocity, acceleration,
     * or "I just pressed jump, so I should be in the air".
     * <ul>
     *   <li><b>Large (e.g., 256):</b> Can remember complex sequences and long histories, but takes longer to train.</li>
     *   <li><b>Small (e.g., 32):</b> Faster to train, simpler memory. Good for simple tasks.</li>
     * </ul>
     *
     * @return the dimension of the hidden state vector
     */
    int hiddenStateSize();

    /**
     * Returns the number of stacked recurrent layers (GRU depth).
     * <p>
     * <b>Concept:</b> Stacking layers allows the agent to learn hierarchical time and memory patterns.
     * Layer 1 might track immediate motion, while Layer 2 tracks longer-term goals or strategy.
     * Usually 1 or 2 layers are sufficient for most game AI.
     *
     * @return the number of GRU layers
     */
    int recurrentLayerDepth();

    /**
     * Returns the number of neurons in each hidden layer of the Actor network.
     * <p>
     * <b>Concept:</b> The Actor network is the "brain" that decides what to do.
     * More neurons allow it to learn more complex decision boundaries (e.g., highly specific reactions to specific states).
     *
     * @return neuron count per actor layer
     */
    int actorHiddenLayerNeuronCount();

    /**
     * Returns the number of hidden layers in the Actor network.
     * <p>
     * <b>Concept:</b> Controls the depth of reasoning. Deeper networks can approximate more complex functions.
     *
     * @return number of actor layers
     */
    int actorHiddenLayerCount();

    /**
     * Returns the number of neurons in each hidden layer of the Critic network.
     * <p>
     * <b>Concept:</b> The Critic network is the "coach" that tells the Actor how good its actions were.
     * Usually, the Critic is slightly larger or the same size as the Actor.
     *
     * @return neuron count per critic layer
     */
    int criticHiddenLayerNeuronCount();

    /**
     * Returns the number of hidden layers in the Critic network.
     *
     * @return number of critic layers
     */
    int criticHiddenLayerCount();

    /**
     * Returns the interpolation factor (Tau) for target network updates.
     * <p>
     * <b>Concept:</b> To stabilize learning, the agent uses "Target Networks" that change slowly.
     * This coefficient controls how fast the target networks track the live networks.
     * <ul>
     *   <li><b>Small (e.g., 0.005):</b> Very stable, slow updates. Prevents the "coach" from changing its mind too often.</li>
     *   <li><b>Large (e.g., 0.1):</b> Fast updates, potential instability.</li>
     * </ul>
     *
     * @return the smoothing coefficient (tau)
     */
    double targetSmoothingCoefficient();

    /**
     * Returns the temperature parameter (Alpha) for entropy regularization.
     * <p>
     * <b>Concept:</b> In Soft Actor-Critic, the agent is rewarded not just for points, but for being unpredictable (high entropy).
     * This parameter scales that bonus.
     * <ul>
     *   <li><b>High Alpha:</b> The agent prefers to keep its options open and act randomly where possible. Robust against noise.</li>
     *   <li><b>Low Alpha:</b> The agent converges deterministically to the best solution it found.</li>
     * </ul>
     *
     * @return the temperature alpha
     */
    double entropyTemperatureAlpha();

    /**
     * Returns the target entropy for automatic alpha tuning.
     * <p>
     * <b>Concept:</b> SAC automatically adjusts alpha to maintain this target entropy level.
     * Higher entropy encourages exploration; lower entropy encourages exploitation.
     * <ul>
     *   <li><b>Default:</b> Typically -log(actionSize) for balanced exploration.</li>
     *   <li><b>Custom:</b> Tune based on problem requirements.</li>
     * </ul>
     *
     * @return the target entropy level for the policy
     */
    double targetEntropy();

    /**
     * Returns the learning rate for the alpha (entropy temperature) learner.
     * <p>
     * <b>Concept:</b> Controls how fast alpha adapts to maintain target entropy.
     * Smaller values = slower, more stable adjustments.
     * Larger values = faster adaptation to entropy changes.
     *
     * @return learning rate for alpha updates
     */
    double alphaLearningRate();

    /**
     * Returns the specific learning rate for the first Q-Critic (Twin Critics).
     * <p>
     * <b>Concept:</b> RSAC uses two critics to avoid overestimating how good an action is (Twin Delayed DDPG style).
     * Usually the same as the base learning rate.
     *
     * @return learning rate for critic 1
     */
    double firstCriticLearningRate();

    /**
     * Returns the specific learning rate for the second Q-Critic.
     *
     * @return learning rate for critic 2
     */
    double secondCriticLearningRate();

    /**
     * Returns the specific learning rate for the Policy (Actor) network.
     *
     * @return learning rate for the actor
     */
    double policyNetworkLearningRate();

    /**
     * Returns the size of the input features processed by the recurrent unit.
     * <p>
     * <b>Concept:</b> Sometimes we preprocess the raw state before feeding it to the memory unit.
     * This defines the size of that intermediate projection. Often matches {@link #hiddenStateSize()}.
     *
     * @return recurrent input dimension
     */
    int recurrentInputFeatureCount();

    /**
     * Returns the capacity of the REMORSE (Recurrent Memory Oriented Replay Storage Experience) buffer.
     * <p>
     * <b>Concept:</b> This is the agent's "long-term memory" or dream diary of past episodes.
     * It stores sequences of experience to learn from later.
     * <ul>
     *   <li><b>Capacity:</b> Number of full episodes/sequences to keep.</li>
     * </ul>
     *
     * @return the maximum number of episodes in the replay buffer
     */
    int remorseTraceBufferCapacity();

    /**
     * Returns the minimum similarity threshold for trace retention.
     * <p>
     * <b>Concept:</b> The agent tries to only remember <i>unique</i> or interesting experiences to save memory.
     * If a new episode is too similar to an old one (similarity > threshold), it might be discarded.
     *
     * @return similarity threshold [0.0, 1.0]
     */
    double remorseMinimumSimilarityThreshold();

    @Override
    default SpartanModelType modelType() {
        return SpartanModelType.RECURRENT_SOFT_ACTOR_CRITIC;
    }


    @Contract(value = " -> new", pure = true)
    static @NotNull Builder builder() {
        return new Builder();
    }

    /**
     * Builder for {@link RecurrentSoftActorCriticConfig}.
     */
    final class Builder {
        private double learningRate = 3e-4;
        private double gamma = 0.99;
        private double epsilon = 1.0;
        private double epsilonMin = 0.01;
        private double epsilonDecay = 0.995;
        private boolean debugLogging = false;
        private boolean isTraining = true;

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
        private double targetEntropy = -1.0;  // Will be computed as -log(actionSize)
        private double alphaLearningRate = 1e-4;

        private Builder() {}

        public Builder learningRate(double val) { this.learningRate = val; return this; }
        public Builder gamma(double val) { this.gamma = val; return this; }
        public Builder epsilon(double val) { this.epsilon = val; return this; }
        public Builder epsilonMin(double val) { this.epsilonMin = val; return this; }
        public Builder epsilonDecay(double val) { this.epsilonDecay = val; return this; }
        public Builder debugLogging(boolean val) { this.debugLogging = val; return this; }
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
        public Builder targetEntropy(double val) { this.targetEntropy = val; return this; }
        public Builder alphaLearningRate(double val) { this.alphaLearningRate = val; return this; }

        @Contract(" -> new")
        public @NotNull RecurrentSoftActorCriticConfig build() {
             return SpartanConfigRegistry.get().createRecurrentSoftActorCriticConfig(
                 learningRate, gamma, epsilon, epsilonMin, epsilonDecay,
                 debugLogging, isTraining,
                 hiddenStateSize, recurrentLayerDepth, actorHiddenLayerNeuronCount, actorHiddenLayerCount,
                 criticHiddenLayerNeuronCount, criticHiddenLayerCount, targetSmoothingCoefficient,
                 entropyTemperatureAlpha, firstCriticLearningRate, secondCriticLearningRate,
                 policyNetworkLearningRate, recurrentInputFeatureCount, remorseTraceBufferCapacity,
                 remorseMinimumSimilarityThreshold, targetEntropy, alphaLearningRate
             );
        }
    }
}
