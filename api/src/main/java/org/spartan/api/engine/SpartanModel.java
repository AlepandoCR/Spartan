package org.spartan.api.engine;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.SpartanModelConfig;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.exception.SpartanPersistenceException;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

/**
 * Represents an instance of a Machine Learning brain.
 * <p>
 * <b>Concept:</b> This is the bridge between the Java and the C++ Core .
 * It holds the Neural Network weights and coordinates the learning process.
 * <p>
 * <b>Lifecycle:</b>
 * <ol>
 *   <li><b>Create:</b> Define configuration and context.</li>
 *   <li><b>Register:</b> Uploads the initial brain structure to C++.</li>
 *   <li><b>Tick:</b> The main heartbeat. Observations go in, Actions come out, Learning happens.</li>
 *   <li><b>Close:</b> Frees the memory.</li>
 * </ol>
 *
 * @param <SpartanModelConfigType> the specific configuration type for this model
 */
public interface SpartanModel<SpartanModelConfigType extends SpartanModelConfig> extends AutoCloseable {

    /**
     * Returns the unique ID assigned to this agent by the Spartan Engine.
     * Useful for debugging logs or matching with saved files.
     *
     * @return the agent identifier
     */
    long getAgentIdentifier();

    /**
     * Returns the unique string identifier provided at creation.
     *
     * @return the user-defined identifier
     */
    @NotNull String getIdentifier();

    /**
     * Returns the context attached to this model.
     *
     * @return the active observation context
     */
    @NotNull SpartanContext getSpartanContext();

    /**
     * Returns the configuration used to create this model.
     *
     * @return the immutable configuration
     */

    @NotNull SpartanModelConfigType getSpartanModelConfig();

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
