package org.spartan.api.agent.config;

/**
 * Descriptor for a nested AutoEncoder slot within an RSAC agent.
 * Mirrors C++ struct NestedEncoderSlotDescriptor (16 bytes, Standard Layout).
 * <p>
 *
 * @param contextSliceStartIndex   Zero-based index into context buffer where encoder input begins
 * @param contextSliceElementCount Number of doubles this encoder reads from context
 * @param latentDimensionSize      Dimensionality of compressed latent output vector
 * @param hiddenNeuronCount        Hidden neurons in encoder dense layer
 */
public record NestedEncoderSlotDescriptor(
        int contextSliceStartIndex,
        int contextSliceElementCount,
        int latentDimensionSize,
        int hiddenNeuronCount
) {
    /** Size in bytes of this struct in native memory (4 × int32_t = 16 bytes). */
    public static final int BYTE_SIZE = 16;

    /**
     * Validates descriptor parameters.
     */
    public NestedEncoderSlotDescriptor {
        if (contextSliceStartIndex < 0) {
            throw new IllegalArgumentException("contextSliceStartIndex must be >= 0");
        }
        if (contextSliceElementCount <= 0) {
            throw new IllegalArgumentException("contextSliceElementCount must be > 0");
        }
        if (latentDimensionSize <= 0) {
            throw new IllegalArgumentException("latentDimensionSize must be > 0");
        }
        if (hiddenNeuronCount <= 0) {
            throw new IllegalArgumentException("hiddenNeuronCount must be > 0");
        }
    }
}
