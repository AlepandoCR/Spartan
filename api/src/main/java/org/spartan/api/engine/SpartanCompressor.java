package org.spartan.api.engine;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.SpartanModelConfig;

import java.lang.foreign.MemorySegment;

/**
 * Represents a passive "Data Processor" model.
 * <p>
 * <b>Concept:</b> A compressor doesn't make choices. It simplifies data.
 * Think of it as an automatic translator that turns "Raw Pixel Data" into "Concepts" (like "Wall", "Door").
 * Agents can then use these concepts to learn much faster.
 *
 * @param <SpartanModelConfigType> the config type
 */
public interface SpartanCompressor<SpartanModelConfigType extends SpartanModelConfig> extends SpartanModel<SpartanModelConfigType> {

    /**
     * Returns the memory segment containing the compressed "Latent Representation".
     * <p>
     * <b>Concept:</b> This is the "Summary" of the input.
     * If the input was an image of a cat, this buffer contains the mathematical essence of "Cat".
     *
     * @return the latent buffer
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
