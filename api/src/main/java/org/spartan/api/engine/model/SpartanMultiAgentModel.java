package org.spartan.api.engine.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.SpartanAgent;
import org.spartan.api.engine.config.SpartanModelConfig;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.exception.SpartanPersistenceException;

import java.nio.file.Path;
import java.util.Optional;

/**
 * Group of N homogeneous agents sharing a single SpartanContext.
 * Architecture:
 * - All agents receive a reference to the same SpartanContext (not a copy)
 * - Context.update() is called ONCE per tick (not N times)
 * - All agents read subspans of the same MemorySegment
 * Lifecycle:
 *  Create: new implementation instance
 *  Register: register() → Passes context to C++
 *  Add agents: addAgent() → Each gets reference to context
 * Game loop: context.updateElement(...), multiAgent.tick()
 * Close: close()
 * Auto-execution:
 * - context.update() is automatic in tick()
 * - action.task() and action.award() callbacks are automatic
 * - User only implements task() and award() in SpartanAction
 *
 * @param <ConfigType> the model configuration type
 */
public interface SpartanMultiAgentModel<ConfigType extends SpartanModelConfig>
        extends AutoCloseable {

    /**
     * Registers this multi-agent group with the C++ engine.
     * Internally:
     * - Passes context.getData() reference to C++
     * - C++ creates subspans for each agent
     * - C++ initializes the centralized critic
     * Precondition: At least one agent should be added (recommended).
     * Can be called before or after addAgent().
     *
     * @throws IllegalStateException if already registered
     * @throws RuntimeException if native registration fails
     */
    void register();

    /**
     * Checks if this group is registered with C++.
     */
    boolean isRegistered();

    /**
     * Adds a new agent to the group.
     * Critical: The agent receives a reference to the same context
     * that the MultiAgent has. No copying, no cloning.
     *
     * @param identifier    Unique string identifier for this agent
     * @param config        Configuration for this agent
     * @return              The registered agent
     * @throws IllegalArgumentException if identifier already exists
     */
    @NotNull SpartanAgent<ConfigType> addAgent(
        @NotNull String identifier,
        @NotNull ConfigType config
    );

    /**
     * Gets an agent by its numeric ID (O(1)).
     */
    @NotNull Optional<SpartanAgent<ConfigType>> getAgent(long agentId);

    /**
     * Gets an agent by its string identifier.
     * Performance: O(n) but typically fast.
     */
    @NotNull Optional<SpartanAgent<ConfigType>> getAgentByIdentifier(@NotNull String identifier);

    /**
     * Removes an agent from the group (O(1)).
     */
    boolean removeAgent(long agentId);

    /**
     * Hot-path: Executes a MARL CTDE tick.
     * Automatic flow (user does nothing manually):
     * 1. [auto] context.update()
     *    - Called once for ALL agents
     *    - All ContextElements update()
     *    - Data flattened to shared dataSegment
     * 2. [auto] SpartanNative.spartanTickMultiAgent()
     *    C++ executes:
     *    - Parallel: agent[i].processTick() for i=0..N-1
     *      (each reads contextSubspan[i], writes actionSubspan[i])
     *    - Centralized: critic.forward(sharedContext, sharedActions)
     *      (reads entire context and actions)
     * 3. [auto] Action callbacks
     *    - For each agent, for each action:
     *      - action.tick(rawOutput)
     *        → action.task(denormalized)  [USER OVERRIDE]
     *        → action.award()             [USER OVERRIDE]
     * Zero-GC Guarantee: No allocations in this method.
     *
     * @throws IllegalStateException if not registered
     * @throws org.spartan.api.exception.SpartanNativeException if C++ fails
     */
    void tick();

    /**
     * Manually applies rewards to all agents.
     * Optional: If you prefer manual reward distribution instead of
     * using action.award() callbacks.
     *
     * @param rewards   Array of N reward scalars
     * @throws IllegalArgumentException if rewards.length != agentCount()
     */
    void applyRewards(double @NotNull [] rewards);

    /**
     * Returns the SHARED context.
     * All agents see this same context instance.
     * Modifications to this context are visible to all agents.
     */
    @NotNull SpartanContext getContext();

    /**
     * Returns the number of active agents in this group.
     */
    int getAgentCount();

    /**
     * Decays exploration rate for all agents.
     * Call at: Episode boundaries, agent respawn, etc.
     */
    void decayExplorationForAll();

    /**
     * Saves all embedded submodels into a compressed .spartan file package.
     *
     * @param filePath The file path to save the package to.
     * @throws SpartanPersistenceException if an error occurs while writing.
     */
    void save(@NotNull Path filePath) throws SpartanPersistenceException;

    /**
     * Loads the submodels from a compressed multi-agent .spartan file package.
     * Matches the submodels with currently registered agents by identifier.
     *
     * @param filePath The file path to load the package from.
     * @throws SpartanPersistenceException if an error occurs while reading.
     */
    void load(@NotNull Path filePath) throws SpartanPersistenceException;
}
