package org.spartan.internal.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanAgent;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.config.NestedEncoderSlotDescriptor;
import org.spartan.api.agent.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.agent.context.SpartanContext;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

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
public class RecurrentSoftActorCriticModel
        extends AbstractSpartanModel<RecurrentSoftActorCriticConfig>
        implements SpartanAgent<RecurrentSoftActorCriticConfig> {

    // Critic-specific buffers
    private final MemorySegment criticWeightsBuffer;
    private final int criticWeightsCount;

    // Action management
    private final SpartanActionManager actionManager;

    // Episode state
    private double episodeReward = 0.0;

    /**
     * Constructs an RSAC model with all necessary allocations.
     *
     * @param agentIdentifier unique 64-bit ID for this agent
     * @param config          RSAC configuration
     * @param context         observation context
     * @param sharedArena     Arena for memory allocation
     * @param actionManager   action interpreter
     */
    public RecurrentSoftActorCriticModel(
            long agentIdentifier,
            @NotNull RecurrentSoftActorCriticConfig config,
            @NotNull SpartanContext context,
            @NotNull Arena sharedArena,
            @NotNull SpartanActionManager actionManager
    ) {
        super(
                agentIdentifier,
                config,
                context,
                sharedArena,
                SpartanModelAllocator.calculateRSACModelWeightCount(config),
                SpartanConfigLayout.RSAC_CONFIG_TOTAL_SIZE
        );

        this.actionManager = actionManager;

        // Allocate critic weights buffer and cache count
        long criticWeightCountLong = SpartanModelAllocator.calculateRSACCriticWeightCount(config);
        this.criticWeightsCount = (int) criticWeightCountLong;
        this.criticWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, criticWeightCountLong);
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

        // Write RSAC-specific fields
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_HIDDEN_STATE_SIZE_OFFSET,
                config.hiddenStateSize());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_RECURRENT_LAYER_DEPTH_OFFSET,
                config.recurrentLayerDepth());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_ACTOR_HIDDEN_NEURON_COUNT_OFFSET,
                config.actorHiddenLayerNeuronCount());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_ACTOR_HIDDEN_LAYER_COUNT_OFFSET,
                config.actorHiddenLayerCount());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_CRITIC_HIDDEN_NEURON_COUNT_OFFSET,
                config.criticHiddenLayerNeuronCount());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_CRITIC_HIDDEN_LAYER_COUNT_OFFSET,
                config.criticHiddenLayerCount());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_TARGET_SMOOTHING_OFFSET,
                config.targetSmoothingCoefficient());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_ENTROPY_ALPHA_OFFSET,
                config.entropyTemperatureAlpha());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_FIRST_CRITIC_LR_OFFSET,
                config.firstCriticLearningRate());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_SECOND_CRITIC_LR_OFFSET,
                config.secondCriticLearningRate());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_POLICY_LR_OFFSET,
                config.policyNetworkLearningRate());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_RECURRENT_INPUT_FEATURE_COUNT_OFFSET,
                config.recurrentInputFeatureCount());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_NESTED_ENCODER_COUNT_OFFSET,
                config.nestedEncoderCount());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_REMORSE_BUFFER_CAPACITY_OFFSET,
                config.remorseTraceBufferCapacity());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_REMORSE_SIMILARITY_THRESHOLD_OFFSET,
                config.remorseMinimumSimilarityThreshold());

        // Write encoder slot descriptors
        int encoderCount = config.nestedEncoderCount();
        long slotsBase = SpartanConfigLayout.RSAC_ENCODER_SLOTS_OFFSET;

        for (int i = 0; i < encoderCount && i < SpartanConfigLayout.MAX_NESTED_ENCODER_SLOTS; i++) {
            NestedEncoderSlotDescriptor slot = config.encoderSlot(i);
            long slotOffset = slotsBase + (i * SpartanConfigLayout.SLOT_DESCRIPTOR_SIZE);

            configSegment.set(ValueLayout.JAVA_INT, slotOffset + SpartanConfigLayout.SLOT_START_INDEX_OFFSET,
                    slot.contextSliceStartIndex());
            configSegment.set(ValueLayout.JAVA_INT, slotOffset + SpartanConfigLayout.SLOT_ELEMENT_COUNT_OFFSET,
                    slot.contextSliceElementCount());
            configSegment.set(ValueLayout.JAVA_INT, slotOffset + SpartanConfigLayout.SLOT_LATENT_DIM_OFFSET,
                    slot.latentDimensionSize());
            configSegment.set(ValueLayout.JAVA_INT, slotOffset + SpartanConfigLayout.SLOT_HIDDEN_COUNT_OFFSET,
                    slot.hiddenNeuronCount());
        }
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
        // Note: GRU hidden state reset would be handled by C++ engine
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
        if (index < 0 || index >= config.actionSize()) {
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
        double[] actions = new double[config.actionSize()];
        for (int i = 0; i < actions.length; i++) {
            actions[i] = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        return actions;
    }
}
