package org.spartan.internal.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanAgent;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.config.DoubleDeepQNetworkConfig;
import org.spartan.api.agent.context.SpartanContext;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Concrete implementation of Double Deep Q-Network (DDQN) model.
 * <p>
 * DDQN is used for discrete action spaces with:
 * <ul>
 *   <li>Online and target networks to reduce overestimation bias</li>
 *   <li>Experience replay buffer for stable learning</li>
 *   <li>Epsilon-greedy exploration with decay</li>
 * </ul>
 * <p>
 * Memory is managed by the parent Arena. All buffers are allocated once
 * in the constructor and reused throughout the model's lifetime.
 * <p>
 * Action output buffer contains Q-values for each discrete action.
 * The agent selects argmax(Q) during exploitation or random action during exploration.
 */
public class DoubleDeepQNetworkModel
        extends AbstractSpartanModel<DoubleDeepQNetworkConfig>
        implements SpartanAgent<DoubleDeepQNetworkConfig> {

    // DDQN doesn't use separate critic weights - combined in model weights
    // We allocate a minimal buffer for interface compatibility
    private final MemorySegment criticWeightsBuffer;
    private final int criticWeightsCount;

    // Action management
    private final SpartanActionManager actionManager;

    // Episode state
    private double episodeReward = 0.0;

    /**
     * Constructs a DDQN model with all necessary allocations.
     *
     * @param agentIdentifier unique 64-bit ID for this agent
     * @param config          DDQN configuration
     * @param context         observation context
     * @param sharedArena     Arena for memory allocation
     * @param actionManager   action interpreter for discrete actions
     */
    public DoubleDeepQNetworkModel(
            long agentIdentifier,
            @NotNull DoubleDeepQNetworkConfig config,
            @NotNull SpartanContext context,
            @NotNull Arena sharedArena,
            @NotNull SpartanActionManager actionManager
    ) {
        super(
                agentIdentifier,
                config,
                context,
                sharedArena,
                SpartanModelAllocator.calculateDDQNModelWeightCount(config),
                SpartanConfigLayout.DDQN_CONFIG_TOTAL_SIZE
        );

        this.actionManager = actionManager;

        // DDQN uses a minimal critic buffer (1 element) for interface compatibility
        // The actual Q-network is in the model weights buffer
        this.criticWeightsCount = 1;
        this.criticWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, 1);
    }

    @Override
    protected void writeConfigToSegment() {
        // Write BaseHyperparameterConfig fields
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_MODEL_TYPE_OFFSET,
                config.modelType().getNativeValue());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_LEARNING_RATE_OFFSET,
                config.learningRate());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_GAMMA_OFFSET,
                config.gamma());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_OFFSET,
                config.epsilon());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_MIN_OFFSET,
                config.epsilonMin());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_DECAY_OFFSET,
                config.epsilonDecay());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_STATE_SIZE_OFFSET,
                config.stateSize());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_ACTION_SIZE_OFFSET,
                config.actionSize());
        configSegment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_IS_TRAINING_OFFSET,
                (byte) (config.isTraining() ? 1 : 0));

        // Write DDQN-specific fields
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_TARGET_SYNC_INTERVAL_OFFSET,
                config.targetNetworkSyncInterval());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_REPLAY_BUFFER_CAPACITY_OFFSET,
                config.replayBufferCapacity());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_REPLAY_BATCH_SIZE_OFFSET,
                config.replayBatchSize());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_HIDDEN_NEURON_COUNT_OFFSET,
                config.hiddenLayerNeuronCount());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_HIDDEN_LAYER_COUNT_OFFSET,
                config.hiddenLayerCount());
    }

    @Override
    protected MemorySegment getCriticWeightsBufferInternal() {
        return criticWeightsBuffer;
    }

    @Override
    protected int getCriticWeightsCount() {
        return criticWeightsCount;
    }

    // ==================== SpartanAgent Implementation ====================

    @Override
    public @NotNull SpartanActionManager getActionManager() {
        return actionManager;
    }

    @Override
    public @NotNull MemorySegment getCriticWeightsBuffer() {
        return criticWeightsBuffer;
    }

    /**
     * Hot-path tick with reward - Zero-GC.
     * <p>
     * Combines context update, reward application, and inference in one native call.
     *
     * @param reward the reward signal for this tick
     */
    @Override
    public void tick(double reward) {
        episodeReward += reward;
        executeNativeTick(reward);
    }

    @Override
    public void applyReward(double reward) {
        episodeReward += reward;
    }

    @Override
    public double getEpisodeReward() {
        return episodeReward;
    }

    @Override
    public void resetEpisode() {
        episodeReward = 0.0;
    }

    // ==================== DDQN-Specific Methods ====================

    /**
     * Reads Q-value for a specific action.
     * <p>
     * After tick(), the action output buffer contains Q(s,a) for each action.
     *
     * @param actionIndex the discrete action index (0 to actionSize - 1)
     * @return the Q-value for the given action
     */
    public double readQValue(int actionIndex) {
        if (actionIndex < 0 || actionIndex >= config.actionSize()) {
            throw new IndexOutOfBoundsException("Action index " + actionIndex + " out of bounds");
        }
        return actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, actionIndex);
    }

    /**
     * Returns the index of the action with highest Q-value (greedy action).
     * <p>
     * Zero-GC: No allocations, safe for hot path.
     *
     * @return the index of the best action
     */
    public int getBestActionIndex() {
        int bestIndex = 0;
        double bestValue = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);

        for (int i = 1; i < config.actionSize(); i++) {
            double value = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            if (value > bestValue) {
                bestValue = value;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    /**
     * Returns all Q-values.
     * <p>
     * Note: This allocates - use sparingly, not in hot path.
     *
     * @return array of Q-values for each action
     */
    public double[] readAllQValues() {
        double[] qValues = new double[config.actionSize()];
        for (int i = 0; i < qValues.length; i++) {
            qValues[i] = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        return qValues;
    }

    /**
     * Reads all Q-values into a pre-allocated buffer (Zero-GC).
     *
     * @param buffer the buffer to fill (must be at least actionSize length)
     */
    public void readQValuesIntoBuffer(double[] buffer) {
        int count = Math.min(buffer.length, config.actionSize());
        for (int i = 0; i < count; i++) {
            buffer[i] = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
    }
}
