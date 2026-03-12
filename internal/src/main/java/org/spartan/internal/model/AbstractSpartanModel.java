package org.spartan.internal.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanModel;
import org.spartan.api.agent.config.SpartanModelConfig;
import org.spartan.api.agent.context.SpartanContext;
import org.spartan.api.exception.SpartanNativeException;
import org.spartan.api.exception.SpartanPersistenceException;
import org.spartan.internal.agent.context.SpartanContextImpl;
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
public abstract class AbstractSpartanModel<C extends SpartanModelConfig> implements SpartanModel<C> {

    protected final Arena arena;
    protected final long agentIdentifier;
    protected final C config;
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

    protected AbstractSpartanModel(
            long agentIdentifier,
            @NotNull C config,
            @NotNull SpartanContext context,
            @NotNull Arena sharedArena,
            long modelWeightCount,
            long configSegmentSize
    ) {
        this.arena = sharedArena;
        this.agentIdentifier = agentIdentifier;
        this.config = config;
        this.context = context;
        this.lastContextSize = context.getSize();

        // Cache primitive counts
        this.modelWeightsCount = (int) modelWeightCount;
        this.actionCount = config.actionSize();

        // Allocate config and weight buffers (JVM-owned, C++ gets pointers)
        this.configSegment = arena.allocate(configSegmentSize, 8);
        this.modelWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, modelWeightCount);
        this.actionOutputBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, actionCount);
    }

    @Override
    public long getAgentIdentifier() {
        return agentIdentifier;
    }

    @Override
    public @NotNull SpartanContext getSpartanContext() {
        return context;
    }

    @Override
    public @NotNull C getSpartanModelConfig() {
        return config;
    }

    @Override
    public @NotNull MemorySegment getActionOutputBuffer() {
        return actionOutputBuffer;
    }

    @Override
    public @NotNull MemorySegment getModelWeightsBuffer() {
        return modelWeightsBuffer;
    }

    @Override
    public boolean isRegistered() {
        return registered;
    }

    // ==================== Tick Operations (Zero-GC) ====================

    /**
     * Zero-GC tick method. Calls native with zero reward.
     * <p>
     * Suitable for non-learning models (AutoEncoder) or when rewards
     * are applied separately via the orchestrator.
     */
    @Override
    public void tick() {
        executeNativeTick(0.0);
    }

    /**
     * Core tick implementation - Zero-GC, no allocations.
     * <p>
     * This method:
     * <ol>
     *   <li>Updates context (flushes Java data to off-heap)</li>
     *   <li>Syncs clean sizes for variable elements</li>
     *   <li>Calls native spartanTickAgent with the reward</li>
     * </ol>
     *
     * @param reward the reward signal (0.0 for non-learning ticks)
     * @throws IllegalStateException if not registered or closed
     * @throws SpartanNativeException if native call fails
     */
    protected void executeNativeTick(double reward) {
        if (!registered) {
            throw new IllegalStateException("Model not registered with native engine");
        }
        if (closed) {
            throw new IllegalStateException("Model has been closed");
        }

        // Phase 1: Update context (Zero-GC internally)
        context.update();

        // Phase 2: Handle context resize (rare)
        int currentSize = context.getSize();
        if (currentSize != lastContextSize) {
            SpartanNative.updateContextPointer(agentIdentifier, context.getData(), currentSize);
            lastContextSize = currentSize;
        }

        // Phase 3: Sync clean sizes for variable elements
        if (context instanceof SpartanContextImpl impl) {
            impl.syncCleanSizes(agentIdentifier);
        }

        // Phase 4: Call native tick with reward
        int result = SpartanNative.spartanTickAgent(agentIdentifier, reward);
        if (result != 0) {
            throw new SpartanNativeException("Native tick failed for agent " + agentIdentifier, result);
        }
    }

    @Override
    public void register() {
        if (registered) {
            throw new IllegalStateException("Model already registered");
        }
        if (closed) {
            throw new IllegalStateException("Model has been closed");
        }

        writeConfigToSegment();

        // Call native with primitives for scalars, MemorySegment for buffers
        int result = SpartanNative.spartanRegisterModel(
                agentIdentifier,                    // long - passed directly
                configSegment,                      // MemorySegment (pointer)
                getCriticWeightsBufferInternal(),   // MemorySegment (pointer)
                getCriticWeightsCount(),            // int - passed directly
                modelWeightsBuffer,                 // MemorySegment (pointer)
                modelWeightsCount,                  // int - passed directly
                context.getData(),                  // MemorySegment (pointer)
                context.getSize(),                  // int - passed directly
                actionOutputBuffer,                 // MemorySegment (pointer)
                actionCount                         // int - passed directly
        );

        if (result != 0) {
            throw new RuntimeException("Native model registration failed: " + result);
        }

        registered = true;
    }

    // ==================== Persistence ====================

    @Override
    public void saveModel(@NotNull Path filePath) throws SpartanPersistenceException {
        if (!registered) {
            throw new IllegalStateException("Model must be registered before saving");
        }
        if (closed) {
            throw new IllegalStateException("Model has been closed");
        }

        String pathString = filePath.toAbsolutePath().toString();
        int result = SpartanNative.spartanSaveModel(agentIdentifier, pathString);

        if (result != 0) {
            throw new SpartanPersistenceException(
                    "Failed to save model to: " + pathString + " (error code: " + result + ")");
        }
    }

    @Override
    public void loadModel(@NotNull Path filePath) throws SpartanPersistenceException {
        if (closed) {
            throw new IllegalStateException("Model has been closed");
        }

        String pathString = filePath.toAbsolutePath().toString();

        // Load weights into the model weights buffer
        // Note: C++ handles loading both model and critic weights if file contains both
        int result = SpartanNative.spartanLoadModel(pathString, modelWeightsBuffer, modelWeightsCount);

        if (result != 0) {
            throw new SpartanPersistenceException(
                    "Failed to load model from: " + pathString + " (error code: " + result + ")");
        }
    }

    // ==================== Exploration Management ====================

    @Override
    public void decayExploration() {
        if (!registered) {
            throw new IllegalStateException("Model must be registered before decaying exploration");
        }
        if (closed) {
            throw new IllegalStateException("Model has been closed");
        }

        SpartanNative.spartanDecayExploration(agentIdentifier);
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }

        if (registered) {
            // Pass agentIdentifier as long directly
            SpartanNative.spartanUnregisterModel(agentIdentifier);
            registered = false;
        }

        closed = true;
    }

    /** Writes config data to configSegment. Called once during registration. */
    protected abstract void writeConfigToSegment();

    /** Returns critic weights buffer (for agent models). */
    protected abstract MemorySegment getCriticWeightsBufferInternal();

    /** Returns critic weights count as primitive int. */
    protected abstract int getCriticWeightsCount();
}
