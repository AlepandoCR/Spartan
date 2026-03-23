package org.spartan.api.engine.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.SpartanAgent;
import org.spartan.api.engine.config.SpartanModelConfig;
import org.spartan.api.engine.context.SpartanContext;

import java.util.Optional;

/**
 * Group of N homogeneous agents sharing a SINGLE SpartanContext.
 *
 * Architecture:
 * - All agents receive a REFERENCE to the same SpartanContext (not a copy)
 * - Context.update() is called ONCE per tick (not N times)
 * - All agents read subspans of the same MemorySegment
 *
 * Lifecycle:
 *  Create: new implementation instance
 *  Register: register() → Passes context to C++
 *  Add agents: addAgent() → Each gets REFERENCE to context
 * Game loop: context.updateElement(...), multiAgent.tick()
 * Close: close()
 *
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
     *
     * Internally:
     * - Passes context.getData() reference to C++
     * - C++ creates subspans for each agent
     * - C++ initializes the centralized critic
     *
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
     *
     * Critical: The agent receives a REFERENCE to the SAME context
     * that the MultiAgent has. No copying, no cloning.
     *
     * @param identifier    Unique string identifier for this agent
     * @param config        Configuration for this agent
     * @return              The registered agent
     * @throws IllegalArgumentException if identifier already exists
     */
    @NotNull
    SpartanAgent<ConfigType> addAgent(
        @NotNull String identifier,
        @NotNull ConfigType config
    );

    /**
     * Gets an agent by its numeric ID (O(1)).
     */
    @NotNull
    Optional<SpartanAgent<ConfigType>> getAgent(long agentId);

    /**
     * Gets an agent by its string identifier.
     * Performance: O(n) but typically fast.
     */
    @NotNull
    Optional<SpartanAgent<ConfigType>> getAgentByIdentifier(@NotNull String identifier);

    /**
     * Removes an agent from the group (O(1)).
     */
    boolean removeAgent(long agentId);

    /**
     * Hot-path: Executes a MARL CTDE tick.
     *
     * Automatic flow (user does NOTHING manually):
     *
     * 1. [AUTOMATIC] context.update()
     *    - Called once for ALL agents
     *    - All ContextElements update()
     *    - Data flattened to shared dataSegment
     *
     * 2. [AUTOMATIC] SpartanNative.spartanTickMultiAgent()
     *    C++ executes:
     *    - Parallel: agent[i].processTick() for i=0..N-1
     *      (each reads contextSubspan[i], writes actionSubspan[i])
     *    - Centralized: critic.forward(sharedContext, sharedActions)
     *      (reads ENTIRE context and actions)
     *
     * 3. [AUTOMATIC] Action callbacks
     *    - For each agent, for each action:
     *      - action.tick(rawOutput)
     *        → action.task(denormalized)  [USER OVERRIDE]
     *        → action.award()             [USER OVERRIDE]
     *
     * Zero-GC Guarantee: No allocations in this method.
     *
     * @throws IllegalStateException if not registered
     * @throws org.spartan.api.exception.SpartanNativeException if C++ fails
     */
    void tick();

    /**
     * Manually applies rewards to all agents.
     *
     * Optional: If you prefer manual reward distribution instead of
     * using action.award() callbacks.
     *
     * @param rewards   Array of N reward scalars
     * @throws IllegalArgumentException if rewards.length != agentCount()
     */
    void applyRewards(double @NotNull [] rewards);

    /**
     * Returns the SHARED context.
     *
     * All agents see this same context instance.
     * Modifications to this context are visible to all agents.
     */
    @NotNull
    SpartanContext getContext();

    /**
     * Returns the number of active agents in this group.
     */
    int getAgentCount();

    /**
     * Decays exploration rate for all agents.
     *
     * Call at: Episode boundaries, agent respawn, etc.
     */
    void decayExplorationForAll();

    /**
     * Unregisters from C++ and releases resources.
     * After this call, the group cannot be used.
     */
    @Override
    void close();
}
