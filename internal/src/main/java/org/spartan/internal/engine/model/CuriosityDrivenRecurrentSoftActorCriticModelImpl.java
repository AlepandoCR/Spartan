package org.spartan.internal.engine.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.action.type.SpartanAction;
import org.spartan.api.engine.config.SpartanModelType;
import org.spartan.api.engine.model.CuriosityDrivenRecurrentSoftActorCriticModel; // Add import
import org.spartan.api.engine.config.CuriosityDrivenRecurrentSoftActorCriticConfig;
import org.spartan.api.engine.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.engine.context.SpartanContext;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * Concrete implementation of Curiosity-Driven Recurrent Soft Actor-Critic model.
 * <p>
 * This model wraps a standard Recurrent Soft Actor-Critic agent and adds an Intrinsic
 * Curiosity Module (ICM).
 */
public class CuriosityDrivenRecurrentSoftActorCriticModelImpl
        extends AbstractSpartanModel<CuriosityDrivenRecurrentSoftActorCriticConfig>
        implements CuriosityDrivenRecurrentSoftActorCriticModel {

    // Critic-specific buffers (includes Forward Dynamics Network parameters)
    private final MemorySegment criticWeightsBuffer;
    private final int criticWeightsCount;

    private final SpartanActionManager actionManager;
    private final List<SpartanAction> actions;

    protected double episodeReward = 0.0;
    protected double accumulatedTickReward = 0.0;

    /**
     * Constructs a Curiosity-Driven RSAC model with all necessary allocations.
     *
     * @param identifier       the model identifier
     * @param agentIdentifier  the agent identifier
     * @param config           the model configuration
     * @param context          the spartan context
     * @param sharedArena      the arena for shared allocations
     * @param actionManager    the action manager
     */
    public CuriosityDrivenRecurrentSoftActorCriticModelImpl(
            @NotNull String identifier,
            long agentIdentifier,
            @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig config,
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
                SpartanModelAllocator.calculateCuriosityDrivenRecurrentSoftActorCriticModelWeightCount(
                        config,
                        requireContextSize(context),
                        actionManager.getActions().size()
                ),
                SpartanConfigLayout.CURIOSITY_RSAC_CONFIG_TOTAL_SIZE_PADDED,
                actionManager.getActions().size()
        );

        this.actionManager = actionManager;
        this.actions = List.copyOf(actionManager.getActions());


        long criticWeightCountLong = SpartanModelAllocator.calculateCuriosityDrivenRecurrentSoftActorCriticCriticWeightCount(
                config,
                requireContextSize(context),
                actionManager.getActions().size()
        );
        this.criticWeightsCount = (int) criticWeightCountLong;

        // Use ValueLayout to ensure correct element scaling
        this.criticWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, criticWeightCountLong + SIMD_PADDING_DOUBLES);
    }

    @Override
    protected void writeConfigToSegment() {
        int stateSize = requireContextSize(context);
        // Delegate to allocator to ensure consistent memory layout with C++
        MemorySegment tempConfigSegment = SpartanModelAllocator.writeCuriosityDrivenRecurrentSoftActorCriticConfig(
                arena,
                config,
                stateSize,
                actionManager.getActions().size()
        );
        this.configSegment.copyFrom(tempConfigSegment);
    }

    @Override
    protected @NotNull MemorySegment getCriticWeightsBufferInternal() { return criticWeightsBuffer; }

    @Override
    protected int getCriticWeightsCount() { return criticWeightsCount; }


    @Override public @NotNull SpartanActionManager getActionManager() { return actionManager; }
    @Override public @NotNull MemorySegment getCriticWeightsBuffer() { return criticWeightsBuffer; }

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

    @Override public double getEpisodeReward() { return episodeReward; }
    @Override public void resetEpisode() { episodeReward = 0.0; }

    /**
     * Reads a single action value from the action output buffer.
     */
    public double readActionValue(int index) {
        if (index < 0 || index >= actionCount) {
            throw new IndexOutOfBoundsException("Action index " + index + " out of bounds");
        }
        return actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    /**
     * Reads all predicted action values.
     */
    public double[] readAllActionValues() {
        double[] actions = new double[actionCount];
        for (int i = 0; i < actionCount; i++) {
            actions[i] = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        return actions;
    }

    public @NotNull RecurrentSoftActorCriticConfig getEmbeddedRecurrentSoftActorCriticConfig() {
        return config.recurrentSoftActorCriticConfig();
    }

    @Override
    protected void postTickActions() {
        int limit = Math.min(actions.size(), actionCount);
        for (int i = 0; i < limit; i++) {
            double rawOutput = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            actions.get(i).tick(rawOutput);
        }
    }

    @Override
    protected SpartanModelType getModelType() {
        return SpartanModelType.CURIOSITY_DRIVEN_RECURRENT_SOFT_ACTOR_CRITIC;
    }
}