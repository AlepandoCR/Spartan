package org.spartan.api.engine.model;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.SpartanCompressor;
import org.spartan.api.engine.SpartanModel;
import org.spartan.api.engine.config.AutoEncoderCompressorConfig;

import org.spartan.api.SpartanApi;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.context.SpartanContext;

/**
 * A passive utility model for dimension reduction via learned compression.
 * <p>
 * <b>Concept:</b> An AutoEncoder learns to compress high-dimensional observations into a compact latent representation.
 * The encoder compresses the input observation into the latent bottleneck.
 * The decoder reconstructs the input from the latent representation to enable loss computation.
 * <p>
 * <b>Architecture:</b>
 * <ul>
 *   <li><b>Encoder:</b> observation -> dense layers -> latent (bottleneck)</li>
 *   <li><b>Decoder:</b> latent -> dense layers -> reconstruction (same shape as input)</li>
 *   <li><b>Loss:</b> MSE between input and reconstruction</li>
 * </ul>
 * <p>
 * <b>Use Cases:</b>
 * <ul>
 *   <li>Image preprocessing: Compress 1000-dimensional pixel data to 16-dimensional "essence".</li>
 *   <li>Sensor fusion: Combine multiple context inputs into a single compact representation for downstream agents.</li>
 *   <li>Feature extraction: Learn meaningful low-dimensional features automatically without labels.</li>
 *   <li>Preprocessing for other agents: Feed the latent representation to DDQN/RSAC for faster convergence.</li>
 * </ul>
 * <p>
 * <b>Key Points:</b>
 * <ul>
 *   <li>Does NOT take actions or earn rewards - purely a data transformer.</li>
 *   <li>Must call {@link #tick()} to trigger inference (encoder + decoder).</li>
 *   <li>Always monitor {@link #getReconstructionLoss()} to track training progress.</li>
 *   <li>The latent buffer contains the compressed representation (size = latentDimensionSize).</li>
 *   <li>The reconstruction buffer contains the decoder output (size = stateSize, same as input).</li>
 * </ul>
 *
 * @see SpartanCompressor for inherited compression methods (getLatentBuffer, getReconstructionBuffer, etc.)
 * @see SpartanModel for inherited model lifecycle methods (register, saveModel, loadModel, etc.)
 */
public interface AutoEncoderCompressorModel extends SpartanCompressor<AutoEncoderCompressorConfig> {

    /**
     * Reads latent values into an existing buffer.
     * <p>
     * Copies up to buffer.length latent values into the provided array.
     * More efficient than {@link #readAllLatent()} when you have a pre-allocated buffer.
     * <p>
     * <b>Zero-GC:</b> No allocations when reusing buffer.
     *
     * @param buffer array to fill with latent values
     */
    void readLatentIntoBuffer(double[] buffer);

    /**
     * Reads a single reconstruction value for a specific state dimension.
     * <p>
     * After {@link #tick()}, the reconstruction buffer contains the decoder's output.
     *
     * @param index the state dimension index (0 to stateSize - 1)
     * @return the reconstruction value for the given dimension
     * @throws IndexOutOfBoundsException if index is out of range
     */
    double readReconstruction(int index);

    /**
     * Reads all reconstruction values into a new array.
     * <p>
     * Returns the complete reconstruction output of the decoder.
     * Note: This allocates a new array - use sparingly, not in hot path.
     *
     * @return array of reconstruction values (size = stateSize)
     */
    double[] readAllReconstruction();

    /**
     * Helper to build this specific model type.
     *
     * @param api the API instance
     * @param identifier unique name for this compressor
     * @param config the specific AutoEncoder config
     * @param context the observation context
     * @param actions the action manager
     * @return the new compressor model
     */
    @Contract("_, _, _, _, _ -> new")
    static AutoEncoderCompressorModel build(
            @NotNull SpartanApi api,
            @NotNull String identifier,
            @NotNull AutoEncoderCompressorConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions) {
        return api.createAutoEncoderCompressor(
                identifier,
                config,
                context,
                actions
        );
    }
}
