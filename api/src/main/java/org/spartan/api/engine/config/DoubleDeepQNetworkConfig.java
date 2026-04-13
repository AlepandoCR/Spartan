package org.spartan.api.engine.config;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.spi.SpartanConfigRegistry;

/**
 * Configuration for the Double Deep Q-Network (DDQN) model.
 * <p>
 * <b>Concept:</b> DDQN is a classic algorithm for <b>discrete</b> actions (like pressing buttons).
 * "Double" refers to using two networks to fix a math error in the original DQN where it was overly optimistic.
 */
public non-sealed interface DoubleDeepQNetworkConfig extends SpartanModelConfig {

    /**
     * Returns the frequency (in ticks) at which the Target Network is updated.
     * <p>
     * <b>Concept:</b> Q-Learning uses a fixed "target" to learn against (like shooting at a stationary target).
     * Every N ticks, we move the target to the agent's current position.
     * <ul>
     *   <li><b>Low (e.g., 100):</b> Unstable target, learning oscillates.</li>
     *   <li><b>High (e.g., 1000):</b> Stable target, consistent learning.</li>
     * </ul>
     *
     * @return ticks between target syncs
     */
    int targetNetworkSyncInterval();

    /**
     * Returns the maximum size of the Experience Replay Buffer.
     * <p>
     * <b>Concept:</b> The agent stores every move it makes in a database. Ideally, this should hold minutes or hours of gameplay.
     * If too small, the agent forgets what it learned 5 minutes ago.
     *
     * @return max experiences stored
     */
    int replayBufferCapacity();

    /**
     * Returns the number of experiences trained on per tick.
     * <p>
     * <b>Concept:</b> Every tick, the agent pulls N random memories from its replay buffer to practice.
     * <ul>
     *   <li><b>Small (32):</b> Fast performance, slower learning convergence.</li>
     *   <li><b>Large (256):</b> Slower per tick, but learns more robustly per tick.</li>
     * </ul>
     *
     * @return batch size
     */
    int replayBatchSize();

    /**
     * Returns the width of the hidden layers.
     *
     * @return neuron count per layer
     */
    int hiddenLayerNeuronCount();

    /**
     * Returns the depth of the neural network.
     *
     * @return number of hidden layers
     */
    int hiddenLayerCount();
    @Override
    default SpartanModelType modelType() {
        return SpartanModelType.DOUBLE_DEEP_Q_NETWORK;
    }


    @Contract(value = " -> new", pure = true)
    static @NotNull Builder builder() {
        return new Builder();
    }

    final class Builder {
        private double learningRate = 1e-4;
        private double gamma = 0.99;
        private double epsilon = 1.0;
        private double epsilonMin = 0.01;
        private double epsilonDecay = 0.995;
        private boolean debugLogging = false;
        private boolean isTraining = true;
        private int targetNetworkSyncInterval = 1000;
        private int replayBufferCapacity = 100000;
        private int replayBatchSize = 64;
        private int hiddenLayerNeuronCount = 256;
        private int hiddenLayerCount = 2;
        public Builder() {}
        public Builder learningRate(double val) { this.learningRate = val; return this; }
        public Builder gamma(double val) { this.gamma = val; return this; }
        public Builder epsilon(double val) { this.epsilon = val; return this; }
        public Builder epsilonMin(double val) { this.epsilonMin = val; return this; }
        public Builder epsilonDecay(double val) { this.epsilonDecay = val; return this; }
        public Builder debugLogging(boolean val) { this.debugLogging = val; return this; }
        public Builder isTraining(boolean val) { this.isTraining = val; return this; }
        public Builder targetNetworkSyncInterval(int val) { this.targetNetworkSyncInterval = val; return this; }
        public Builder replayBufferCapacity(int val) { this.replayBufferCapacity = val; return this; }
        public Builder replayBatchSize(int val) { this.replayBatchSize = val; return this; }
        public Builder hiddenLayerNeuronCount(int val) { this.hiddenLayerNeuronCount = val; return this; }
        public Builder hiddenLayerCount(int val) { this.hiddenLayerCount = val; return this; }


        @Contract("-> new")
        public @NotNull  DoubleDeepQNetworkConfig build() {
            return SpartanConfigRegistry.get().createDoubleDeepQNetworkConfig(
                learningRate, gamma, epsilon, epsilonMin, epsilonDecay,
                debugLogging, isTraining,
                targetNetworkSyncInterval, replayBufferCapacity, replayBatchSize,
                hiddenLayerNeuronCount, hiddenLayerCount
            );
        }
    }
}
