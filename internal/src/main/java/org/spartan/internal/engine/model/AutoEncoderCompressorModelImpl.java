package org.spartan.internal.engine.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.config.AutoEncoderCompressorConfig;
import org.spartan.api.engine.config.SpartanModelType;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.model.AutoEncoderCompressorModel;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Concrete implementation of AutoEncoder Compressor model.
 * <p>
 * AutoEncoder learns to compress observations into a lower-dimensional latent representation.
 */
public class AutoEncoderCompressorModelImpl
        extends AbstractSpartanModel<AutoEncoderCompressorConfig>
        implements AutoEncoderCompressorModel {

    private final MemorySegment reconstructionBuffer;
    private final MemorySegment latentBuffer;
    private volatile double reconstructionLoss = 0.0;
    private final MemorySegment criticWeightsBuffer;
    private final int criticWeightsCount;

    public AutoEncoderCompressorModelImpl(
            @NotNull String identifier,
            long agentIdentifier,
            @NotNull AutoEncoderCompressorConfig config,
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
                SpartanModelAllocator.calculateAutoEncoderModelWeightCount(config, context.getSize()),
                SpartanConfigLayout.AE_CONFIG_TOTAL_SIZE_PADDED,
                config.latentDimensionSize()
        );

        // SIMD padding ensures C++ AVX2 operations don't fault when reading near buffer end
        this.reconstructionBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, context.getSize() + SIMD_PADDING_DOUBLES);
        this.latentBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, config.latentDimensionSize() + SIMD_PADDING_DOUBLES);
        this.criticWeightsCount = 1;
        this.criticWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, 1 + SIMD_PADDING_DOUBLES);
    }

    @Override
    protected void writeConfigToSegment() {
        int stateSize = requireContextSize(context);
        // AutoEncoder outputs its latent vector via the action output buffer.
        MemorySegment temp = SpartanModelAllocator.writeAutoEncoderConfig(arena, config, stateSize, config.latentDimensionSize());
        this.configSegment.copyFrom(temp);
    }

    @Override
    protected @NotNull MemorySegment getCriticWeightsBufferInternal() { return criticWeightsBuffer; }

    @Override
    protected int getCriticWeightsCount() { return criticWeightsCount; }

    @Override
    public @NotNull MemorySegment getLatentBuffer() { return latentBuffer; }

    @Override
    public @NotNull MemorySegment getReconstructionBuffer() { return reconstructionBuffer; }

    @Override
    public double getReconstructionLoss() { return reconstructionLoss; }

    @Override
    public double readLatent(int index) {
        if (index < 0 || index >= config.latentDimensionSize()) {
            throw new IndexOutOfBoundsException("Latent index " + index + " out of bounds");
        }
        return latentBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    @Override
    public double[] readAllLatent() {
        double[] latent = new double[config.latentDimensionSize()];
        for (int i = 0; i < latent.length; i++) {
            latent[i] = latentBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        return latent;
    }

    public void readLatentIntoBuffer(double @NotNull [] buffer) {
        int count = Math.min(buffer.length, config.latentDimensionSize());
        for (int i = 0; i < count; i++) {
            buffer[i] = latentBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
    }

    public double readReconstruction(int index) {
        if (index < 0 || index >= context.getSize()) {
            throw new IndexOutOfBoundsException("Reconstruction index " + index + " out of bounds");
        }
        return reconstructionBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    public double[] readAllReconstruction() {
        double[] recon = new double[context.getSize()];
        for (int i = 0; i < recon.length; i++) {
            recon[i] = reconstructionBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }
        return recon;
    }

    /**
     * AutoEncoder compressor doesn't have exploration, so this is a no-op.
     */
    @Override
    public void decayExploration() {
        // AutoEncoder compressor doesn't have exploration, so this is a no-op.
    }

    @Override
    protected SpartanModelType getModelType() {
        return SpartanModelType.AUTO_ENCODER_COMPRESSOR;
    }
}
