package org.spartan.internal.engine.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.action.type.SpartanAction;
import org.spartan.api.engine.config.ProximalPolicyOptimizationConfig;
import org.spartan.api.engine.config.SpartanModelType;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.model.ProximalPolicyOptimizationModel;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * PPO implementation using FFM for zero-copy memory bridging with the C++ engine.
 */
public class ProximalPolicyOptimizationSpartanModelImpl
        extends AbstractSpartanModel<ProximalPolicyOptimizationConfig>
        implements ProximalPolicyOptimizationModel {

    private final SpartanActionManager actionManager;
    private final List<SpartanAction> actions;

    private final MemorySegment criticWeightsBuffer;
    private final int criticTotalCount; // Stores Weights + Biases

    protected double episodeReward = 0.0;
    protected double accumulatedTickReward = 0.0;

    public ProximalPolicyOptimizationSpartanModelImpl(
            @NotNull String identifier,
            long agentIdentifier,
            @NotNull ProximalPolicyOptimizationConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actionManager,
            @NotNull Arena sharedArena
    ) {
        super(
                identifier,
                agentIdentifier,
                config,
                context,
                sharedArena,
                computeActorTotal(config, requireContextSize(context), actionManager.getActions().size()),
                SpartanConfigLayout.PPO_CONFIG_TOTAL_SIZE_PADDED,
                actionManager.getActions().size()
        );

        int stateSize = requireContextSize(context);
        this.actionManager = actionManager;
        this.actions = List.copyOf(actionManager.getActions());

        // Sum weights and biases for the C++ spanning boundary
        int criticWeightsCount = computeCriticWeights(config, stateSize);
        int criticBiasCount = computeCriticBiases(config);
        this.criticTotalCount = criticWeightsCount + criticBiasCount;

        // Allocate Critic buffer aligned for AVX-512 SIMD processing
        this.criticWeightsBuffer = arena.allocate(
                ((long) this.criticTotalCount + SIMD_PADDING_DOUBLES) * 8L, 64);
    }

    @Override
    protected void writeConfigToSegment() {
        int stateSize = requireContextSize(context);
        MemorySegment temp = SpartanModelAllocator.writePPOConfig(
                arena, config, stateSize, actionManager.getActions().size());
        MemorySegment.copy(temp, 0, this.configSegment, 0, temp.byteSize());
    }

    @Override
    protected @NotNull MemorySegment getCriticWeightsBufferInternal() {
        return criticWeightsBuffer;
    }

    @Override
    protected int getCriticWeightsCount() {
        return criticTotalCount; // Passes the required combined parameter count
    }

    @Override
    public void tick() {
        double currentReward = accumulatedTickReward;
        accumulatedTickReward = 0.0;
        executeNativeTick(currentReward);
    }

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
    public double getAndResetEpisodeReward() {
        double reward = episodeReward;
        episodeReward = 0.0;
        return reward;
    }

    @Override
    public double getEpisodeReward() {
        return episodeReward;
    }

    @Override
    public void resetEpisode() {
        episodeReward = 0.0;
        accumulatedTickReward = 0.0;
    }

    @Override
    public double readActionValue(int index) {
        if (index < 0 || index >= actionCount) {
            throw new IndexOutOfBoundsException("PPO Action index " + index + " out of bounds");
        }
        return actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    @Override
    public double[] readAllActionValues() {
        double[] result = new double[actionCount];
        MemorySegment.copy(
                actionOutputBuffer,
                ValueLayout.JAVA_DOUBLE,
                0,
                result,
                0,
                actionCount
        );
        return result;
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
    public @NotNull SpartanActionManager getActionManager() {
        return actionManager;
    }

    @Override
    public @NotNull MemorySegment getCriticWeightsBuffer() {
        return criticWeightsBuffer;
    }

    @Override
    protected SpartanModelType getModelType() {
        return SpartanModelType.PROXIMAL_POLICY_OPTIMIZATION;
    }

    // Layer 0: stateSize -> h. Remaining (l-1) layers: h -> h.
    // Two independent output heads (mean and log_std), each h -> actionSize.
    private static int computeActorWeights(
            @NotNull ProximalPolicyOptimizationConfig config,
            int stateSize,
            int actionSize
    ) {
        int h = config.actorHiddenNeuronCount();
        int l = config.actorHiddenLayerCount();
        return (stateSize * h) + (h * h * (l - 1)) + (h * actionSize * 2);
    }

    // One bias per hidden neuron per layer.
    // Plus one bias per output neuron per head (mean + log_std).
    private static int computeActorBiases(
            @NotNull ProximalPolicyOptimizationConfig config,
            int actionSize
    ) {
        return (config.actorHiddenNeuronCount() * config.actorHiddenLayerCount()) + (actionSize * 2);
    }

    // Layer 0: stateSize -> h. Remaining (l-1) layers: h -> h.
    // Output head: h -> 1 scalar value.
    private static int computeCriticWeights(
            @NotNull ProximalPolicyOptimizationConfig config,
            int stateSize
    ) {
        int h = config.criticHiddenNeuronCount();
        int l = config.criticHiddenLayerCount();
        return (stateSize * h) + (h * h * (l - 1)) + h;
    }

    // One bias per hidden neuron per layer, plus 1 for the scalar output.
    private static int computeCriticBiases(@NotNull ProximalPolicyOptimizationConfig config) {
        return (config.criticHiddenNeuronCount() * config.criticHiddenLayerCount()) + 1;
    }

    private static long computeActorTotal(
            @NotNull ProximalPolicyOptimizationConfig config,
            int stateSize,
            int actionSize
    ) {
        return computeActorWeights(config, stateSize, actionSize)
                + computeActorBiases(config, actionSize);
    }
}