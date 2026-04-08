package org.spartan.internal.engine.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.model.RecurrentSoftActorCriticModel;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.action.type.SpartanAction;
import org.spartan.api.engine.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.engine.context.SpartanContext;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * Concrete implementation of Recurrent Soft Actor-Critic (RSAC) model.
 * <p>
 * RSAC combines:
 * <ul>
 *   <li>GRU-based recurrence for temporal memory</li>
 *   <li>Soft Actor-Critic for continuous action spaces</li>
 *   <li>Twin Q-critics to reduce overestimation bias</li>
 *   <li>Optional nested AutoEncoders for representation learning</li>
 * </ul>
 * <p>
 * Memory is managed by the parent Arena. All buffers are allocated once
 * in the constructor and reused throughout the model's lifetime.
 */
public class RecurrentSoftActorCriticModelImpl
        extends AbstractSpartanModel<RecurrentSoftActorCriticConfig>
        implements RecurrentSoftActorCriticModel {

    // Critic-specific buffers
    private final MemorySegment criticWeightsBuffer;
    private final int criticWeightsCount;

    private final SpartanActionManager actionManager;
    private final List<SpartanAction> actions;

    protected double episodeReward = 0.0;
    protected double accumulatedTickReward = 0.0;

    /**
     * Constructs an RSAC model with all necessary allocations.
     *
     * @param identifier      unique string ID
     * @param agentIdentifier unique 64-bit ID for this agent
     * @param config          RSAC configuration
     * @param context         observation context
     * @param sharedArena     Arena for memory allocation
     * @param actionManager   action interpreter
     */
    public RecurrentSoftActorCriticModelImpl(
            @NotNull String identifier,
            long agentIdentifier,
            @NotNull RecurrentSoftActorCriticConfig config,
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
                SpartanModelAllocator.calculateRSACModelWeightCount(config, requireContextSize(context), actionManager.getActions().size()),
                SpartanConfigLayout.RSAC_CONFIG_TOTAL_SIZE_PADDED,
                actionManager.getActions().size()
        );

        this.actionManager = actionManager;
        this.actions = List.copyOf(actionManager.getActions());

        // Allocate critic weights buffer and cache count
        long criticWeightCountLong = SpartanModelAllocator.calculateRSACCriticWeightCount(config, requireContextSize(context), actionManager.getActions().size());
        this.criticWeightsCount = (int) criticWeightCountLong;
        this.criticWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, criticWeightCountLong + SIMD_PADDING_DOUBLES);
    }

    @Override
    protected void writeConfigToSegment() {
        int stateSize = requireContextSize(context);
        MemorySegment temp = SpartanModelAllocator.writeRSACConfig(arena, config, stateSize, actionManager.getActions().size());
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
     * Ticks the model without an external reward signal.
     * <p>
     * This can be used for internal step progression, where the reward is
     * either not applicable or already accounted for.
     */
    @Override
    public void tick() {
        double currentReward = accumulatedTickReward;
        accumulatedTickReward = 0.0;
        executeNativeTick(currentReward);
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
        accumulatedTickReward += reward;
        double currentReward = accumulatedTickReward;
        accumulatedTickReward = 0.0;
        executeNativeTick(currentReward);
    }

    @Override
    public void applyReward(double reward) {
        episodeReward += reward;
        accumulatedTickReward += reward;
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
     * Reads the predicted action values from the action output buffer.
     * <p>
     * For continuous actions, these are the raw values (typically in [-1, 1])
     * that should be scaled to the actual action range.
     *
     * @param index the action dimension index
     * @return the predicted action value
     */
    public double readActionValue(int index) {
        if (index < 0 || index >= actionCount) {
            throw new IndexOutOfBoundsException("Action index " + index + " out of bounds");
        }
        return actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    /**
     * Reads all predicted action values.
     * <p>
     * Note: This allocates a new array - use sparingly, not in hot path.
     *
     * @return array of predicted action values
     */
    public double[] readAllActionValues() {
        double[] actions = new double[actionCount];
        for (int i = 0; i < actions.length; i++) {
            actions[i] = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        return actions;
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
