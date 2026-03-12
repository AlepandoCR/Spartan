package org.spartan.internal.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanCompressor;
import org.spartan.api.agent.config.AutoEncoderCompressorConfig;
import org.spartan.api.agent.context.SpartanContext;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Concrete implementation of AutoEncoder Compressor model.
 * <p>
 * AutoEncoder learns to compress observations into a lower-dimensional latent representation.
 */
public class AutoEncoderCompressorModel
        extends AbstractSpartanModel<AutoEncoderCompressorConfig>
        implements SpartanCompressor<AutoEncoderCompressorConfig> {

    private final MemorySegment reconstructionBuffer;
    private volatile double reconstructionLoss = 0.0;
    private final MemorySegment criticWeightsBuffer;
    private final int criticWeightsCount;

    public AutoEncoderCompressorModel(
            long agentIdentifier,
            @NotNull AutoEncoderCompressorConfig config,
            @NotNull SpartanContext context,
            @NotNull Arena sharedArena
    ) {
        super(agentIdentifier, config, context, sharedArena,
                SpartanModelAllocator.calculateAutoEncoderModelWeightCount(config),
                SpartanConfigLayout.AE_CONFIG_TOTAL_SIZE);

        this.reconstructionBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, config.stateSize());
        this.criticWeightsCount = 1;
        this.criticWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, 1);
    }

    @Override
    protected void writeConfigToSegment() {
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_MODEL_TYPE_OFFSET,
                config.modelType().getNativeValue());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_LEARNING_RATE_OFFSET,
                config.learningRate());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_GAMMA_OFFSET, config.gamma());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_OFFSET, config.epsilon());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_MIN_OFFSET, config.epsilonMin());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_DECAY_OFFSET, config.epsilonDecay());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_STATE_SIZE_OFFSET, config.stateSize());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_ACTION_SIZE_OFFSET, config.actionSize());
        configSegment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_IS_TRAINING_OFFSET,
                (byte) (config.isTraining() ? 1 : 0));

        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.AE_LATENT_DIM_SIZE_OFFSET,
                config.latentDimensionSize());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.AE_ENCODER_HIDDEN_NEURON_OFFSET,
                config.encoderHiddenNeuronCount());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.AE_ENCODER_LAYER_COUNT_OFFSET,
                config.encoderLayerCount());
        configSegment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.AE_DECODER_LAYER_COUNT_OFFSET,
                config.decoderLayerCount());
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.AE_BOTTLENECK_REG_WEIGHT_OFFSET,
                config.bottleneckRegularisationWeight());
    }

    @Override
    protected MemorySegment getCriticWeightsBufferInternal() { return criticWeightsBuffer; }

    @Override
    protected int getCriticWeightsCount() { return criticWeightsCount; }

    @Override
    public @NotNull MemorySegment getLatentBuffer() { return actionOutputBuffer; }

    @Override
    public @NotNull MemorySegment getReconstructionBuffer() { return reconstructionBuffer; }

    @Override
    public double getReconstructionLoss() { return reconstructionLoss; }

    @Override
    public double readLatent(int index) {
        if (index < 0 || index >= config.latentDimensionSize()) {
            throw new IndexOutOfBoundsException("Latent index " + index + " out of bounds");
        }
        return actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    @Override
    public double[] readAllLatent() {
        double[] latent = new double[config.latentDimensionSize()];
        for (int i = 0; i < latent.length; i++) {
            latent[i] = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        return latent;
    }

    public void readLatentIntoBuffer(double[] buffer) {
        int count = Math.min(buffer.length, config.latentDimensionSize());
        for (int i = 0; i < count; i++) {
            buffer[i] = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
    }

    public double readReconstruction(int index) {
        if (index < 0 || index >= config.stateSize()) {
            throw new IndexOutOfBoundsException("Reconstruction index " + index + " out of bounds");
        }
        return reconstructionBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    public double[] readAllReconstruction() {
        double[] recon = new double[config.stateSize()];
        for (int i = 0; i < recon.length; i++) {
            recon[i] = reconstructionBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        return recon;
    }
}
