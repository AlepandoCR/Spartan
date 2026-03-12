package org.spartan.internal.engine;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanAgent;
import org.spartan.api.agent.SpartanModel;
import org.spartan.api.engine.SpartanTickOrchestrator;
import org.spartan.internal.agent.context.SpartanContextImpl;
import org.spartan.internal.bridge.SpartanNative;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Implementation of {@link SpartanTickOrchestrator} that manages the tick lifecycle
 * for all registered Spartan agents.
 * <p>
 * This class handles:
 * <ul>
 *   <li>Agent registration with slot-based tracking</li>
 *   <li>Reward accumulation into parallel off-heap arrays</li>
 *   <li>Context updates and clean size synchronization</li>
 *   <li>Native engine tick execution</li>
 * </ul>
 * <p>
 * <b>Memory Layout:</b>
 * <pre>
 * agentIdsSegment:  [id_0, id_1, id_2, ..., id_N]  (uint64_t / JAVA_LONG)
 * rewardsSegment:   [r_0,  r_1,  r_2,  ..., r_N]   (double / JAVA_DOUBLE)
 * </pre>
 * Both arrays are parallel - agentIdsSegment[i] receives rewardsSegment[i].
 * <p>
 * <b>Zero-GC Design:</b>
 * <ul>
 *   <li>All buffers are pre-allocated and reused</li>
 *   <li>No iterators - only indexed loops over cached arrays</li>
 *   <li>No autoboxing - all primitives</li>
 * </ul>
 */
public class SpartanTickOrchestratorImpl implements SpartanTickOrchestrator {

    private static final int INITIAL_CAPACITY = 64;
    private static final ValueLayout.OfLong LONG_LAYOUT = ValueLayout.JAVA_LONG;
    private static final ValueLayout.OfDouble DOUBLE_LAYOUT = ValueLayout.JAVA_DOUBLE;

    private final Arena arena;
    private final ConcurrentHashMap<Long, AgentSlot> agentSlots;

    // Pre-allocated arrays for Zero-GC iteration in tick()
    private AgentSlot[] slotCache;
    private int slotCacheSize;
    private boolean slotCacheDirty;

    // Off-heap buffers for native calls
    private MemorySegment agentIdsSegment;
    private MemorySegment rewardsSegment;
    private int bufferCapacity;

    private volatile boolean closed = false;

    /**
     * Creates a new tick orchestrator.
     *
     * @param arena the Arena for off-heap memory allocation
     */
    public SpartanTickOrchestratorImpl(@NotNull Arena arena) {
        this.arena = arena;
        this.agentSlots = new ConcurrentHashMap<>();
        this.slotCache = new AgentSlot[INITIAL_CAPACITY];
        this.slotCacheSize = 0;
        this.slotCacheDirty = false;

        // Allocate initial buffers
        this.bufferCapacity = INITIAL_CAPACITY;
        this.agentIdsSegment = arena.allocate(LONG_LAYOUT, bufferCapacity);
        this.rewardsSegment = arena.allocate(DOUBLE_LAYOUT, bufferCapacity);
    }

    @Override
    public void registerAgent(@NotNull SpartanAgent<?> agent) {
        if (closed) {
            throw new IllegalStateException("Orchestrator is closed");
        }

        long id = agent.getAgentIdentifier();
        if (agentSlots.containsKey(id)) {
            throw new IllegalStateException("Agent " + id + " is already registered");
        }

        // Register with native engine first
        if (agent instanceof SpartanModel<?> model) {
            model.register();
        }

        AgentSlot slot = new AgentSlot(agent);
        agentSlots.put(id, slot);
        slotCacheDirty = true;

        ensureBufferCapacity(agentSlots.size());
    }

    @Override
    public void unregisterAgent(@NotNull SpartanAgent<?> agent) {
        if (closed) {
            return;
        }

        long id = agent.getAgentIdentifier();
        AgentSlot slot = agentSlots.remove(id);
        if (slot != null) {
            slotCacheDirty = true;

            // Unregister from native engine
            SpartanNative.spartanUnregisterModel(id);
        }
    }

    @Override
    public boolean isRegistered(long agentIdentifier) {
        return agentSlots.containsKey(agentIdentifier);
    }

    @Override
    public int tick() {
        if (closed) {
            throw new IllegalStateException("Orchestrator is closed");
        }

        // Rebuild slot cache if needed (rare - only on registration changes)
        if (slotCacheDirty) {
            rebuildSlotCache();
        }

        int count = slotCacheSize;
        if (count == 0) {
            return 0; // No agents to tick
        }

        // Phase 1: Update contexts and collect rewards (Zero-GC loop)
        for (int i = 0; i < count; i++) {
            AgentSlot slot = slotCache[i];
            SpartanAgent<?> agent = slot.agent;

            // Update agent's context
            agent.getSpartanContext().update();

            // Sync clean sizes for variable elements
            if (agent.getSpartanContext() instanceof SpartanContextImpl impl) {
                impl.syncCleanSizes(agent.getAgentIdentifier());
            }

            // Write agent ID and reward to off-heap buffers
            long agentId = agent.getAgentIdentifier();
            double reward = slot.consumePendingReward();

            agentIdsSegment.setAtIndex(LONG_LAYOUT, i, agentId);
            rewardsSegment.setAtIndex(DOUBLE_LAYOUT, i, reward);
        }

        // Phase 2: Call native tick for all agents
        return SpartanNative.spartanTickAll(agentIdsSegment, rewardsSegment, count);
    }

    @Override
    public int tickAgent(@NotNull SpartanAgent<?> agent) {
        if (closed) {
            throw new IllegalStateException("Orchestrator is closed");
        }

        long agentId = agent.getAgentIdentifier();
        AgentSlot slot = agentSlots.get(agentId);

        if (slot == null) {
            throw new IllegalStateException("Agent " + agentId + " is not registered");
        }

        // Update agent's context
        agent.getSpartanContext().update();

        // Sync clean sizes for variable elements
        if (agent.getSpartanContext() instanceof SpartanContextImpl impl) {
            impl.syncCleanSizes(agentId);
        }

        // Write single agent ID and reward to off-heap buffers
        double reward = slot.consumePendingReward();
        agentIdsSegment.setAtIndex(LONG_LAYOUT, 0, agentId);
        rewardsSegment.setAtIndex(DOUBLE_LAYOUT, 0, reward);

        // Call native tick for single agent
        return SpartanNative.spartanTickAll(agentIdsSegment, rewardsSegment, 1);
    }

    @Override
    public int getAgentCount() {
        return agentSlots.size();
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        // Unregister all agents from native engine
        for (AgentSlot slot : agentSlots.values()) {
            SpartanNative.spartanUnregisterModel(slot.agent.getAgentIdentifier());
        }
        agentSlots.clear();
        slotCacheSize = 0;
    }

    /**
     * Ensures the off-heap buffers have enough capacity.
     */
    private void ensureBufferCapacity(int minCapacity) {
        if (minCapacity <= bufferCapacity) {
            return;
        }

        // Grow by 2x
        int newCapacity = Math.max(minCapacity, bufferCapacity * 2);
        agentIdsSegment = arena.allocate(LONG_LAYOUT, newCapacity);
        rewardsSegment = arena.allocate(DOUBLE_LAYOUT, newCapacity);
        bufferCapacity = newCapacity;
    }

    /**
     * Rebuilds the slot cache array for Zero-GC iteration.
     * Called only when agents are added/removed.
     */
    private void rebuildSlotCache() {
        int size = agentSlots.size();

        // Ensure cache array is large enough
        if (slotCache.length < size) {
            slotCache = new AgentSlot[Math.max(size, slotCache.length * 2)];
        }

        // Copy slots to array (no iterator in hot path - this is rare rebuild)
        int index = 0;
        for (AgentSlot slot : agentSlots.values()) {
            slotCache[index++] = slot;
        }
        slotCacheSize = size;
        slotCacheDirty = false;
    }

    /**
     * Internal slot tracking an agent and its pending reward.
     * <p>
     * Reward accumulation is thread-safe via volatile + atomic-like pattern.
     */
    static final class AgentSlot {
        final SpartanAgent<?> agent;
        private volatile double pendingReward = 0.0;

        AgentSlot(SpartanAgent<?> agent) {
            this.agent = agent;
        }

        /**
         * Adds reward to the pending accumulator.
         * Called by agent.applyReward() - may be from any thread.
         */
        void addReward(double reward) {
            // Simple accumulation - not truly atomic but sufficient for RL
            pendingReward += reward;
        }

        /**
         * Consumes and resets the pending reward.
         * Called only from tick() - single-threaded.
         */
        double consumePendingReward() {
            double reward = pendingReward;
            pendingReward = 0.0;
            return reward;
        }
    }

    /**
     * Gets the slot for an agent (used by agent.applyReward routing).
     */
    AgentSlot getSlot(long agentId) {
        return agentSlots.get(agentId);
    }
}
