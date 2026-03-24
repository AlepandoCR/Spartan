package org.spartan.internal.engine.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.action.type.SpartanAction;
import org.spartan.api.engine.model.DoubleDeepQNetworkModel; // Add import
import org.spartan.api.engine.config.DoubleDeepQNetworkConfig;
import org.spartan.api.engine.context.SpartanContext;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

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
public class DoubleDeepQNetworkModelImpl
        extends AbstractSpartanModel<DoubleDeepQNetworkConfig>
        implements DoubleDeepQNetworkModel {

    // DDQN doesn't use separate critic weights - combined in model weights
    // We allocate a minimal buffer for interface compatibility
    private final MemorySegment criticWeightsBuffer;
    private final int criticWeightsCount;

    // Action management
    private final SpartanActionManager actionManager;
    private final List<SpartanAction> actions;

    // Episode state
    private double episodeReward = 0.0;

    /**
     * Constructs a DDQN model with all necessary allocations.
     *
     * @param identifier      unique string ID
     * @param agentIdentifier unique 64-bit ID for this agent
     * @param config          DDQN configuration
     * @param context         observation context
     * @param sharedArena     Arena for memory allocation
     * @param actionManager   action interpreter for discrete actions
     */
    public DoubleDeepQNetworkModelImpl(
            @NotNull String identifier,
            long agentIdentifier,
            @NotNull DoubleDeepQNetworkConfig config,
            @NotNull SpartanContext context,
            @NotNull Arena sharedArena,
            @NotNull SpartanActionManager actionManager
    ) {
        super(
                identifier,
                agentIdentifier,
                config,
                context,
                sharedArena,
                SpartanModelAllocator.calculateDDQNModelWeightCount(config, requireContextSize(context), actionManager.getActions().size()),
                SpartanConfigLayout.DDQN_CONFIG_TOTAL_SIZE_PADDED,
                actionManager.getActions().size()
        );

        this.actionManager = actionManager;
        this.actions = List.copyOf(actionManager.getActions());

        this.criticWeightsCount = 1;
        this.criticWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, 1 + SIMD_PADDING_DOUBLES);
    }

    @Override
    protected void writeConfigToSegment() {
        int stateSize = requireContextSize(context);
        MemorySegment temp = SpartanModelAllocator.writeDDQNConfig(arena, config, stateSize, actionManager.getActions().size());
        MemorySegment.copy(temp, 0, this.configSegment, 0, temp.byteSize());
    }

    @Override
    protected @NotNull MemorySegment getCriticWeightsBufferInternal() {
        return criticWeightsBuffer;
    }

    @Override
    protected int getCriticWeightsCount() {
        return criticWeightsCount;
    }


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


    /**
     * Reads Q-value for a specific action.
     * <p>
     * After tick(), the action output buffer contains Q(s,a) for each action.
     *
     * @param actionIndex the discrete action index (0 to actionSize - 1)
     * @return the Q-value for the given action
     */
    public double readQValue(int actionIndex) {
        if (actionIndex < 0 || actionIndex >= actionCount) {
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

        for (int i = 1; i < actionCount; i++) {
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
        double[] qValues = new double[actionCount];
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
    public void readQValuesIntoBuffer(double @NotNull [] buffer) {
        int count = Math.min(buffer.length, actionCount);
        for (int i = 0; i < count; i++) {
            buffer[i] = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
    }

    @Override
    protected void postTickActions() {
        int limit = Math.min(actions.size(), actionCount);
        for (int i = 0; i < limit; i++) {
            double rawOutput = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            actions.get(i).tick(rawOutput);
        }
    }
}
