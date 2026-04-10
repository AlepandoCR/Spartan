package org.spartan.api.engine.context.element.compressed;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.SpartanApi;
import org.spartan.api.engine.config.AutoEncoderCompressorConfig;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.context.element.SpartanContextElement;
import org.spartan.api.engine.context.element.variable.SpartanVariableContextElement;
import org.spartan.api.engine.model.AutoEncoderCompressorModel;

/**
 * A VariableContextElement that wraps a compressor model internally.
 * <b>Purpose:</b> Transparently compress raw variable-size data into fixed-size latent representation.
 * <b>Key Property:</b> getSize() returns compressed dimension, enabling stable offset calculation.
 * <b>Architecture: SELF-CONTAINED</b>
 * - Manages own raw data buffer (from wrapped element)
 * - Creates internal minimal SpartanContext just for compression
 * - Creates own AutoEncoderCompressorModel instance
 * - Zero interaction with parent context needed (except the element itself)
 * <b>Lifecycle:</b>
 * <ol>
 *   <li>Create: new CompressedVariableContextElement(...)</li>
 *   <li>Use: Add to parent context like any other element</li>
 *   <li>Update: Parent context calls element.update() (which internally compresses)</li>
 *   <li>Close: Call close() to clean up compressor resources</li>
 * </ol>
 */
public class CompressedVariableContextElement implements SpartanContextElement {

    private final SpartanVariableContextElement rawElement;
    private final AutoEncoderCompressorModel compressor;
    private final SpartanContext internalContext;
    private final int compressedSize;
    private final double[] compressedBuffer;

    /**
     * Constructs the compressed element - SELF-CONTAINED.
     * <b>No need to pass SpartanContext from parent</b> - we create our own minimal context internally.
     * The element is completely independent and can be used in any context.
     * <b>Important:</b> SpartanApi must be provided via dependency injection for safety.
     *
     * @param rawElement the underlying variable-size element to compress
     * @param compressorConfig AutoEncoder configuration (defines compression ratio)
     * @param api the SpartanApi instance (injected, NOT via getInstance())
     * @throws IllegalArgumentException if any parameter is null
     */
    public CompressedVariableContextElement(
        @NotNull SpartanVariableContextElement rawElement,
        @NotNull AutoEncoderCompressorConfig compressorConfig,
        @NotNull SpartanApi api) {

        this.rawElement = rawElement;
        this.compressedSize = compressorConfig.latentDimensionSize();
        this.compressedBuffer = new double[compressedSize];

        // Create internal minimal context for the compressor to read from
        this.internalContext = SpartanContext.build(api, "internal_context_" + System.nanoTime());
        // Register only the raw element in the internal context at logical index 0
        this.internalContext.addElement(rawElement, 0);

        // Create a dummy action manager (compressor doesn't use it, but API requires it)
        var dummyActions = api.createActionManager();

        // Create compressor model - it will read from our internal context
        // The compressor becomes a registered model in the C++ engine
        this.compressor = api.createAutoEncoderCompressor(
            "compressor_" + rawElement.getIdentifier(),
            compressorConfig,
            this.internalContext,
            dummyActions
        );
        // Register compressor with the engine
        this.compressor.register();
    }

    /**
     * Returns the compressed size (latent dimension).
     * <b>KEY INSIGHT:</b> This returns COMPRESSED size, not raw size.
     * This enables parent Context to calculate stable offsets.
     * Regardless of how many raw entities, output is always compressedSize.
     *
     * @return the fixed latent dimension size
     */
    public int getSize() {
        return compressedSize;
    }

    /**
     * Returns the compressed latent representation for this tick.
     * This returns the cached array that was populated by tick().
     *
     * @return array of compressed doubles (size = compressedSize, reused every call)
     */
    @Override
    public double @NotNull [] getData() {
        return compressedBuffer;
    }

    /**
     * Prepares the raw element for the next tick.
     * Called before tick() to reset state.
     */
    @Override
    public void prepare() {
        rawElement.prepare();
    }

    /**
     * Updates the compressor with current data.
     *
     * <b>Pipeline:</b>
     * <ol>
     *   <li>Update raw element (it was prepared already)</li>
     *   <li>Update internal context with new raw data</li>
     *   <li>Run compressor inference</li>
     *   <li>Extract latent representation into cached buffer</li>
     * </ol>
     */
    @Override
    public void tick() {
        // Raw element was prepared already, just tick it
        rawElement.tick();

        // Update internal context with new raw data
        this.internalContext.update();

        // Run compressor on raw data
        this.compressor.tick();

        // Extract compressed latent representation into cached buffer
        this.compressor.readLatentIntoBuffer(compressedBuffer);
    }

    /**
     * Returns a user-friendly identifier for this compressed element.
     *
     * @return "compressed_" + rawElement's identifier
     */
    public @NotNull String getIdentifier() {
        return "compressed_" + rawElement.getIdentifier();
    }

    /**
     * Returns the internal compressor for advanced use cases.
     * <b>NOT typically needed</b> - the element handles compression transparently.
     * Use this only if you need direct access to compressor state (e.g., reconstruction loss, latent inspection).
     *
     * @return the AutoEncoderCompressorModel managing compression
     */
    public @NotNull AutoEncoderCompressorModel getInternalCompressor() {
        return compressor;
    }

    /**
     * Returns the raw element (before compression).
     * Useful for debugging or accessing raw data directly.
     *
     * @return the underlying SpartanVariableContextElement
     */
    public @NotNull SpartanVariableContextElement getRawElement() {
        return rawElement;
    }

    /**
     * Saves the internal compressor's state to a file.
     *
     * @param filePath the path to save to
     * @throws org.spartan.api.exception.SpartanPersistenceException if save fails
     */
    public void save(@NotNull java.nio.file.Path filePath) throws org.spartan.api.exception.SpartanPersistenceException {
        compressor.saveModel(filePath);
    }

    /**
     * Loads the internal compressor's state from a file.
     *
     * @param filePath the path to load from
     * @throws org.spartan.api.exception.SpartanPersistenceException if load fails
     */
    public void load(@NotNull java.nio.file.Path filePath) throws org.spartan.api.exception.SpartanPersistenceException {
        compressor.loadModel(filePath);
    }

    /**
     * Cleans up resources associated with this element.
     * <b>Must be called</b> when the element is no longer needed to free C++ resources.
     *
     * @throws Exception if compressor cleanup fails
     */
    public void close() throws Exception {
        compressor.close();
    }
}
