package org.spartan.api.agent;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.config.SpartanModelConfig;
import org.spartan.api.agent.context.SpartanContext;
import org.spartan.api.exception.SpartanPersistenceException;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

/**
 * Base interface for all Spartan ML models.
 * <p>
 * A SpartanModel represents a machine learning model that operates on shared memory
 * between Java and C++. Java owns all memory (via Arena), while C++ operates on
 * non-owning views (std::span).
 * <p>
 * Lifecycle:
 * <ol>
 *   <li>Construct model with config and context</li>
 *   <li>Call {@link #register()} to register with C++ engine</li>
 *   <li>Call {@link #tick()} in game loop (updates context, triggers C++ inference)</li>
 *   <li>Call {@link #close()} when done (unregisters and releases resources)</li>
 * </ol>
 *
 * @param <C> the configuration type for this model
 */
public interface SpartanModel<C extends SpartanModelConfig> extends AutoCloseable {

    /**
     * Returns the unique 64-bit identifier for this model instance.
     * Used to identify the agent in the C++ model registry.
     *
     * @return the agent identifier
     */
    long getAgentIdentifier();

    /**
     * Returns the observation context for this model.
     * The context contains elements that are flattened into a MemorySegment
     * for C++ to read during inference.
     *
     * @return the SpartanContext (never null)
     */
    @NotNull SpartanContext getSpartanContext();

    /**
     * Returns the configuration for this model.
     *
     * @return the model configuration (never null)
     */
    @NotNull C getSpartanModelConfig();

    /**
     * Returns the action output buffer where C++ writes predictions.
     * This buffer is directly accessible - no copying needed.
     *
     * @return MemorySegment containing action outputs (double[])
     */
    @NotNull MemorySegment getActionOutputBuffer();

    /**
     * Returns the model weights buffer.
     * For RSAC/DDQN, this is the actor/policy network weights.
     *
     * @return MemorySegment containing model weights (double[])
     */
    @NotNull MemorySegment getModelWeightsBuffer();

    /**
     * Hot-path method: updates context and triggers native inference.
     * <p>
     * This method performs:
     * <ol>
     *   <li>Updates the context (flushes Java data to off-heap MemorySegment)</li>
     *   <li>Syncs clean sizes for variable elements</li>
     *   <li>Calls native tick with zero reward (suitable for non-learning models)</li>
     * </ol>
     * <p>
     * For agents that need reward-based learning, use {@code agent.tick(reward)} instead.
     * <p>
     * <b>Zero-GC:</b> This method performs no allocations - safe for 20+ TPS.
     *
     * @throws IllegalStateException if model is not registered or has been closed
     * @throws org.spartan.api.exception.SpartanNativeException if native tick fails
     */
    void tick();

    /**
     * Registers this model with the C++ engine.
     * Must be called after construction, before the first tick().
     *
     * @throws IllegalStateException if already registered
     * @throws RuntimeException if native registration fails
     */
    void register();

    /**
     * Checks if this model is currently registered with the C++ engine.
     *
     * @return true if registered, false otherwise
     */
    boolean isRegistered();

    // ==================== Persistence ====================

    /**
     * Saves the model's complete state to a .spartan binary file.
     * <p>
     * This includes:
     * <ul>
     *   <li>Model weights (actor/policy network)</li>
     *   <li>Critic weights (if applicable)</li>
     *   <li>48-byte header with metadata</li>
     *   <li>CRC-32 checksum for integrity</li>
     * </ul>
     *
     * @param filePath the path to save the model (should end with .spartan)
     * @throws SpartanPersistenceException if save operation fails
     * @throws IllegalStateException if model is not registered
     */
    void saveModel(@NotNull Path filePath) throws SpartanPersistenceException;

    /**
     * Loads model weights from a .spartan binary file.
     * <p>
     * The file's CRC-32 checksum is verified before loading.
     * Weights are loaded directly into this model's weight buffers.
     *
     * @param filePath the path to load the model from
     * @throws SpartanPersistenceException if load operation fails (file not found, CRC mismatch, etc.)
     */
    void loadModel(@NotNull Path filePath) throws SpartanPersistenceException;

    // ==================== Exploration Management ====================

    /**
     * Triggers exploration rate decay for this agent.
     * <p>
     * Should be called at episode boundaries (e.g., when player respawns).
     * <ul>
     *   <li>For DDQN: Decays epsilon (epsilon-greedy exploration)</li>
     *   <li>For RSAC: Decays entropy temperature and resets remorse trace</li>
     * </ul>
     *
     * @throws IllegalStateException if model is not registered
     */
    void decayExploration();

    /**
     * Unregisters from C++ and releases resources.
     * After this call, the model cannot be used again.
     */
    @Override
    void close();
}
