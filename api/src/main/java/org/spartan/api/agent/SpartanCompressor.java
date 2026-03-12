package org.spartan.api.agent;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.config.SpartanModelConfig;

import java.lang.foreign.MemorySegment;

/**
 * Interface for representation-learning models that compress observations.
 * <p>
 * Compressor models learn to encode high-dimensional input data into a
 * lower-dimensional latent representation, which can then be used by
 * downstream decision-making agents.
 * <p>
 * Unlike {@link SpartanAgent}, compressors do not produce actions directly;
 * instead, they transform the observation space.
 *
 * @param <C> the configuration type (must be a compressor config)
 */
public interface SpartanCompressor<C extends SpartanModelConfig> extends SpartanModel<C> {

    /**
     * Returns the latent (compressed) representation buffer.
     * <p>
     * After each tick, C++ writes the encoded representation to this buffer.
     * The size equals {@code config.latentDimensionSize()}.
     *
     * @return MemorySegment containing latent vector (double[])
     */
    @NotNull MemorySegment getLatentBuffer();

    /**
     * Returns the reconstruction output buffer.
     * <p>
     * For AutoEncoders, this contains the decoder's reconstruction of the input.
     * The size equals {@code config.stateSize()}.
     *
     * @return MemorySegment containing reconstruction (double[])
     */
    @NotNull MemorySegment getReconstructionBuffer();

    /**
     * Returns the current reconstruction loss (MSE between input and output).
     * <p>
     * This is computed by C++ and can be used to monitor training progress.
     *
     * @return the reconstruction loss value
     */
    double getReconstructionLoss();

    /**
     * Reads a single latent dimension value.
     *
     * @param index the latent dimension index (0 to latentDimensionSize - 1)
     * @return the latent value at the given index
     */
    double readLatent(int index);

    /**
     * Reads all latent values into a new array.
     * <p>
     * Note: This allocates - use sparingly, not in hot path.
     *
     * @return array of latent values
     */
    double[] readAllLatent();
}
