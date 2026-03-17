package org.spartan.internal.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanAgent;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.model.CuriosityDrivenRecurrentSoftActorCriticModel; // Add import
import org.spartan.api.agent.config.CuriosityDrivenRecurrentSoftActorCriticConfig;
import org.spartan.api.agent.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.agent.context.SpartanContext;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

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
    private double episodeReward = 0.0;

    public CuriosityDrivenRecurrentSoftActorCriticModelImpl(
            @NotNull String identifier,
            long agentIdentifier,
            CuriosityDrivenRecurrentSoftActorCriticConfig config,
            SpartanContext context,
            Arena sharedArena,
            SpartanActionManager actionManager
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
                SpartanConfigLayout.CURIOSITY_RSAC_CONFIG_TOTAL_SIZE,
                actionManager.getActions().size()
        );

        this.actionManager = actionManager;


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
    protected MemorySegment getCriticWeightsBufferInternal() { return criticWeightsBuffer; }

    @Override
    protected int getCriticWeightsCount() { return criticWeightsCount; }


    @Override public @NotNull SpartanActionManager getActionManager() { return actionManager; }
    @Override public @NotNull MemorySegment getCriticWeightsBuffer() { return criticWeightsBuffer; }

    @Override
    public void tick(double extrinsicReward) {
        if (Double.isNaN(extrinsicReward)) extrinsicReward = 0.0;
        episodeReward += extrinsicReward;
        executeNativeTick(extrinsicReward);
    }

    @Override public void applyReward(double reward) { episodeReward += reward; }
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

    public RecurrentSoftActorCriticConfig getEmbeddedRecurrentSoftActorCriticConfig() {
        return config.recurrentSoftActorCriticConfig();
    }
}