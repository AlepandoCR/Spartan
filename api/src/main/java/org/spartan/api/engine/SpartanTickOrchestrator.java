package org.spartan.api.engine;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanAgent;

/**
 * Orchestrator for managing the tick lifecycle of all registered Spartan agents.
 * <p>
 * This is the main entry point for executing global ticks across all agents.
 * It handles:
 * <ul>
 *   <li>Agent registration and lifecycle management</li>
 *   <li>Reward collection from individual agents</li>
 *   <li>Context updates and synchronization with native engine</li>
 *   <li>Parallel execution of inference across all agents</li>
 * </ul>
 * <p>
 * The end-user interacts with this interface through simple Java calls:
 * <pre>
 * agent.applyReward(10.0);  // User assigns reward
 * orchestrator.tick();       // Execute global tick
 * double[] actions = agent.readAllActionValues();  // Read predictions
 * </pre>
 * <p>
 * All FFM/MemorySegment details are completely encapsulated - users never
 * see or interact with native memory directly.
 */
public interface SpartanTickOrchestrator extends AutoCloseable {

    /**
     * Registers an agent with the orchestrator.
     * <p>
     * The agent will be included in subsequent tick() calls.
     * Also registers the agent with the native C++ engine.
     *
     * @param agent the agent to register
     * @throws IllegalStateException if the agent is already registered
     */
    void registerAgent(@NotNull SpartanAgent<?> agent);

    /**
     * Unregisters an agent from the orchestrator.
     * <p>
     * The agent will no longer be included in tick() calls.
     * Also unregisters from the native C++ engine.
     *
     * @param agent the agent to unregister
     */
    void unregisterAgent(@NotNull SpartanAgent<?> agent);

    /**
     * Checks if an agent is currently registered.
     *
     * @param agentIdentifier the agent's unique identifier
     * @return true if the agent is registered
     */
    boolean isRegistered(long agentIdentifier);

    /**
     * Executes a global tick across all registered agents.
     * <p>
     * This method performs the following steps:
     * <ol>
     *   <li>Collects pending rewards from all agents</li>
     *   <li>Updates each agent's context (calls context.update())</li>
     *   <li>Syncs variable element clean sizes with native engine</li>
     *   <li>Calls native spartanTickAll with reward buffers</li>
     *   <li>Resets pending rewards on all agents</li>
     * </ol>
     * <p>
     * After tick() returns, agents can read their action outputs.
     * <p>
     * <b>Zero-GC:</b> This method performs no allocations in the hot path.
     *
     * @return 0 on success, negative value on error
     */
    int tick();

    /**
     * Executes a tick for a single specific agent.
     * <p>
     * Use this when you need to tick individual agents rather than all at once.
     * This is useful for:
     * <ul>
     *   <li>Agents that tick at different rates</li>
     *   <li>On-demand inference (e.g., when a specific event occurs)</li>
     *   <li>Testing individual agents</li>
     * </ul>
     * <p>
     * <b>Zero-GC:</b> This method performs no allocations.
     *
     * @param agent the agent to tick (must be registered)
     * @return 0 on success, negative value on error
     * @throws IllegalStateException if the agent is not registered
     */
    int tickAgent(@NotNull SpartanAgent<?> agent);

    /**
     * Returns the number of currently registered agents.
     *
     * @return active agent count
     */
    int getAgentCount();

    /**
     * Closes the orchestrator and releases all resources.
     * <p>
     * All registered agents are unregistered from the native engine.
     */
    @Override
    void close();
}
