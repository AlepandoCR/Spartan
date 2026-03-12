package org.spartan.api.agent.config;

/**
 * Configuration for the Double Deep Q-Network (DDQN) model.
 * <p>
 * Mirrors the C++ struct {@code DoubleDeepQNetworkHyperparameterConfig} from
 * {@code core/src/org/spartan/internal/machinelearning/ModelHyperparameterConfig.h}
 * <p>
 * DDQN is used for discrete action spaces with:
 * <ul>
 *   <li>Online and target networks to reduce overestimation bias</li>
 *   <li>Experience replay buffer for stable learning</li>
 *   <li>Epsilon-greedy exploration</li>
 * </ul>
 *
 * @param learningRate             Base learning rate for gradient descent
 * @param gamma                    Discount factor for future rewards [0.0, 1.0]
 * @param epsilon                  Current exploration probability [0.0, 1.0]
 * @param epsilonMin               Minimum epsilon floor after decay
 * @param epsilonDecay             Multiplicative decay per episode [0.0, 1.0]
 * @param stateSize                Number of elements in observation vector
 * @param actionSize               Number of discrete actions
 * @param isTraining               True for training mode, false for inference
 * @param targetNetworkSyncInterval Ticks between online -> target network sync
 * @param replayBufferCapacity     Maximum transitions in replay buffer
 * @param replayBatchSize          Transitions sampled per learning step
 * @param hiddenLayerNeuronCount   Neurons per hidden layer in Q-network
 * @param hiddenLayerCount         Number of hidden layers in Q-network
 */
public record DoubleDeepQNetworkConfig(
        // Base config fields
        double learningRate,
        double gamma,
        double epsilon,
        double epsilonMin,
        double epsilonDecay,
        int stateSize,
        int actionSize,
        boolean isTraining,

        // DDQN-specific fields
        int targetNetworkSyncInterval,
        int replayBufferCapacity,
        int replayBatchSize,
        int hiddenLayerNeuronCount,
        int hiddenLayerCount
) implements SpartanModelConfig {

    @Override
    public SpartanModelType modelType() {
        return SpartanModelType.DOUBLE_DEEP_Q_NETWORK;
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
     * Builder for {@link DoubleDeepQNetworkConfig} with sensible defaults.
     */
    public static final class Builder {
        // Base defaults
        private double learningRate = 1e-4;
        private double gamma = 0.99;
        private double epsilon = 1.0;
        private double epsilonMin = 0.01;
        private double epsilonDecay = 0.995;
        private int stateSize = 64;
        private int actionSize = 4;
        private boolean isTraining = true;

        // DDQN defaults
        private int targetNetworkSyncInterval = 1000;
        private int replayBufferCapacity = 100000;
        private int replayBatchSize = 64;
        private int hiddenLayerNeuronCount = 256;
        private int hiddenLayerCount = 2;

        private Builder() {}

        public Builder learningRate(double val) { this.learningRate = val; return this; }
        public Builder gamma(double val) { this.gamma = val; return this; }
        public Builder epsilon(double val) { this.epsilon = val; return this; }
        public Builder epsilonMin(double val) { this.epsilonMin = val; return this; }
        public Builder epsilonDecay(double val) { this.epsilonDecay = val; return this; }
        public Builder stateSize(int val) { this.stateSize = val; return this; }
        public Builder actionSize(int val) { this.actionSize = val; return this; }
        public Builder isTraining(boolean val) { this.isTraining = val; return this; }
        public Builder targetNetworkSyncInterval(int val) { this.targetNetworkSyncInterval = val; return this; }
        public Builder replayBufferCapacity(int val) { this.replayBufferCapacity = val; return this; }
        public Builder replayBatchSize(int val) { this.replayBatchSize = val; return this; }
        public Builder hiddenLayerNeuronCount(int val) { this.hiddenLayerNeuronCount = val; return this; }
        public Builder hiddenLayerCount(int val) { this.hiddenLayerCount = val; return this; }

        /**
         * Builds the immutable config instance.
         *
         * @return a new DoubleDeepQNetworkConfig
         */
        public DoubleDeepQNetworkConfig build() {
            return new DoubleDeepQNetworkConfig(
                    learningRate, gamma, epsilon, epsilonMin, epsilonDecay,
                    stateSize, actionSize, isTraining,
                    targetNetworkSyncInterval, replayBufferCapacity, replayBatchSize,
                    hiddenLayerNeuronCount, hiddenLayerCount
            );
        }
    }
}
