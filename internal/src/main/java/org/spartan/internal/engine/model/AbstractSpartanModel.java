package org.spartan.internal.engine.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.SpartanModel;
import org.spartan.api.engine.config.SpartanModelConfig;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.exception.SpartanNativeException;
import org.spartan.api.exception.SpartanPersistenceException;
import org.spartan.internal.engine.context.SpartanContextImpl;
import org.spartan.internal.bridge.SpartanNative;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;

/**
 * Abstract base for SpartanModel with FFM memory management and Zero-GC tick().
 * <p>
 * The tick() method contains NO allocations - safe for high-frequency calls.
 * Scalar parameters (agentId, counts) are now passed as primitives directly to C++.
 */
public abstract class AbstractSpartanModel<SpartanModelConfigType extends SpartanModelConfig> implements SpartanModel<SpartanModelConfigType> {

    protected static int requireContextSize(@NotNull SpartanContext context) {
        if (context.getSize() <= 0) {
            context.update();
        }
        return context.getSize();
    }

    /**
     * SIMD padding in number of doubles.
     * <p>
     * The C++ native code uses SIMD operations
     * To prevent SIMD overreads from causing access violations when buffers end at page
     * boundaries, we add significant padding to all buffer allocations.
     * Increased from 4 to 128 to provide maximum safety margin.
     */
    protected static final int SIMD_PADDING_DOUBLES = 128;

    protected final Arena arena;
    protected final long agentIdentifier;
    protected final SpartanModelConfigType config;
    protected final SpartanContext context;

    // JVM-owned buffers passed to C++ as pointers
    protected final MemorySegment configSegment;
    protected final MemorySegment modelWeightsBuffer;
    protected final MemorySegment actionOutputBuffer;

    // Cached primitive counts (no MemorySegment needed - passed directly as int/long)
    protected final int modelWeightsCount;
    protected final int actionCount;

    private volatile boolean registered = false;
    private volatile boolean closed = false;
    private int lastContextSize;
    protected final String identifier;

    protected AbstractSpartanModel(
            @NotNull String identifier,
            long agentIdentifier,
            @NotNull SpartanModelConfigType config,
            @NotNull SpartanContext context,
            @NotNull Arena sharedArena,
            long modelWeightCount,
            long configSegmentSize,
            int actionSize
    ) {
        this.identifier = identifier;
        this.arena = sharedArena;
        this.agentIdentifier = agentIdentifier;
        this.config = config;
        this.context = context;
        this.lastContextSize = requireContextSize(context);

        // Cache primitive counts
        this.modelWeightsCount = (int) modelWeightCount;
        this.actionCount = actionSize;

        this.configSegment = arena.allocate(configSegmentSize, 64);

        long modelWeightsByteSize = (modelWeightCount + SIMD_PADDING_DOUBLES) * 8L;
        this.modelWeightsBuffer = arena.allocate(modelWeightsByteSize, 64);

        long actionByteSize = (actionCount + SIMD_PADDING_DOUBLES) * 8L;
        this.actionOutputBuffer = arena.allocate(actionByteSize, 64);
    }

    @Override
    public void register() {
        if (registered || closed) throw new IllegalStateException("Invalid state for registration");

        int contextSize = requireContextSize(context);
        lastContextSize = contextSize;
        writeConfigToSegment();

        int result = SpartanNative.spartanRegisterModel(
                agentIdentifier,
                configSegment,
                getCriticWeightsBufferInternal(),
                getCriticWeightsCount(),
                modelWeightsBuffer,
                modelWeightsCount,
                ((SpartanContextImpl) context).getData(), // Cast here
                contextSize,
                actionOutputBuffer,
                actionCount
        );

        if (result != 0) {
            throw new RuntimeException("Native model registration failed: " + result);
        }

        registered = true;
    }


    @Override public void tick() { executeNativeTick(0.0); }

    protected void executeNativeTick(double reward) {
        if (!registered || closed) throw new IllegalStateException("Model not active");
        context.update();
        int currentSize = context.getSize();
        if (currentSize != lastContextSize) {
            SpartanNative.updateContextPointer(agentIdentifier, ((SpartanContextImpl) context).getData(), currentSize);
            lastContextSize = currentSize;
        }
        if (context instanceof SpartanContextImpl impl) impl.syncCleanSizes(agentIdentifier);
        int result = SpartanNative.spartanTickAgent(agentIdentifier, reward);
        if (result != 0) throw new SpartanNativeException("Native tick failed", result);
        postTickActions();
    }

    protected void postTickActions() {
        // Hook for action-based models to execute tasks after native inference.
    }

    @Override public void close() {
        if (closed) return;
        if (registered) { SpartanNative.spartanUnregisterModel(agentIdentifier); registered = false; }
        closed = true;
    }

    @Override public long getAgentIdentifier() { return agentIdentifier; }
    @Override public @NotNull String getIdentifier() { return identifier; }
    @Override public @NotNull SpartanContext getSpartanContext() { return context; }
    @Override public @NotNull SpartanModelConfigType getSpartanModelConfig() { return config; }
    @Override public @NotNull MemorySegment getActionOutputBuffer() { return actionOutputBuffer; }
    @Override public @NotNull MemorySegment getModelWeightsBuffer() { return modelWeightsBuffer; }
    @Override public boolean isRegistered() { return registered; }

    @Override
    public void saveModel(@NotNull Path filePath) throws SpartanPersistenceException {
        String pathString = filePath.toAbsolutePath().toString();
        int result = SpartanNative.spartanSaveModel(agentIdentifier, pathString);
        if (result != 0) throw new SpartanPersistenceException("Save failed: " + result);
    }

    @Override
    public void loadModel(@NotNull Path filePath) throws SpartanPersistenceException {
        String pathString = filePath.toAbsolutePath().toString();
        int result = SpartanNative.spartanLoadModel(pathString, modelWeightsBuffer, modelWeightsCount);
        if (result != 0) throw new SpartanPersistenceException("Load failed: " + result);
    }

    @Override
    public @NotNull SpartanModel<SpartanModelConfigType> copy(@NotNull String newIdentifier) {
        throw new UnsupportedOperationException("Subclasses must implement copy()");
    }

    @Override
    public void setLiveExplorationRate(double newEpsilon) {
        configSegment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_OFFSET, newEpsilon);
    }

    @Override
    public double getEpisodeReward() {
        return 0; // Overridden in classes tracking reward
    }

    @Override
    public void resetEpisode() {
        // Overridden in classes tracking reward
    }

    @Override
    public void decayExploration() {
        if (!registered || closed) return;
        SpartanNative.spartanDecayExploration(agentIdentifier);
    }

    /**
     * Unregisters from C++ and releases resources.
     * <p>
     * This method is called by the framework when the model is no longer needed.
     * Subclasses should not override this method.
     */
    protected void release() {
        if (closed) return;
        close();
        arena.close();
    }

    protected abstract void writeConfigToSegment();
    protected abstract @NotNull MemorySegment getCriticWeightsBufferInternal();
    protected abstract int getCriticWeightsCount();
}