package org.spartan.internal.engine;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.SpartanAgent;
import org.spartan.api.engine.model.SpartanMultiAgentModel;
import org.spartan.api.engine.config.SpartanModelConfig;
import org.spartan.api.engine.config.SpartanMultiAgentGroupConfig;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.action.type.SpartanAction;
import org.spartan.api.exception.SpartanPersistenceException;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.engine.context.SpartanContextImpl;
import org.spartan.internal.engine.model.SpartanModelAllocator;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.Optional;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Implementation of SpartanMultiAgentModel.
 * Key points:
 * - Receives SHARED context from user (not a copy)
 * - Each agent gets REFERENCE to this same context
 * - tick() calls context.update() ONCE for all agents
 * - Delegates to C++ for parallelized agent inference
 */
public class SpartanMultiAgentModelImpl<ConfigType extends SpartanModelConfig>
        implements SpartanMultiAgentModel<ConfigType> {

    private final long multiAgentId;
    private final String identifier;

    // SHARED context (all agents reference this)
    private final SpartanContext sharedContext;
    private final List<SpartanAction> sharedActions;
    private final SpartanMultiAgentGroupConfig groupConfig;
    private final Arena arena;
    private final MemorySegment actionOutputBuffer; // New field

    private MemorySegment sharedContextBuffer;
    private int sharedContextSize;
    private int stateSize;
    private boolean aggregatedContextLayout;

    // Agent registry (O(1) lookup via ID)
    private final Map<Long, SpartanAgent<ConfigType>> agentById = new ConcurrentHashMap<>();

    // State tracking
    private boolean registered = false;
    private boolean closed = false;
    private final AtomicLong nextAgentId = new AtomicLong(1);

    /**
     * Constructor
     * @param identifier      Unique identifier for this group
     * @param sharedContext   The SHARED context (not copied)
     * @param groupConfig          Configuration for all agents in this group
     */
    public SpartanMultiAgentModelImpl(
        @NotNull String identifier,
        @NotNull SpartanMultiAgentGroupConfig groupConfig,
        @NotNull SpartanContext sharedContext,
        @NotNull Iterable<SpartanAction> sharedActions) {

        this.multiAgentId = System.identityHashCode(this);  // Use object identity as ID
        this.identifier = identifier;
        this.groupConfig = groupConfig;
        this.sharedContext = sharedContext;  // reference, not copy
        this.sharedActions = new ArrayList<>();
        sharedActions.forEach(this.sharedActions::add);
        this.arena = Arena.ofShared();

        // Allocate shared action buffer [actionSize * maxAgents]
        this.actionOutputBuffer = SpartanModelAllocator.allocateActionOutputBuffer(
            this.arena,
            this.sharedActions.size() * groupConfig.maxAgents()
        );
    }

    @Override
    public void register() {
        if (registered) {
            return;
        }
        if (closed) {
            throw new IllegalStateException("Multi-agent group is closed");
        }

        // Validate context
        if (!(sharedContext instanceof SpartanContextImpl contextImpl)) {
            throw new IllegalArgumentException("Context must be created via SpartanApi");
        }

        // Ensure context is updated so we have valid sizes
        if (contextImpl.getSize() <= 0) {
            contextImpl.update();
        }

        int totalContextSize = contextImpl.getSize();
        if (totalContextSize <= 0) {
            throw new IllegalStateException("Shared context size is invalid after update");
        }

        if (totalContextSize % groupConfig.maxAgents() == 0) {
            aggregatedContextLayout = true;
            stateSize = totalContextSize / groupConfig.maxAgents();
            sharedContextSize = totalContextSize;
        } else {
            aggregatedContextLayout = false;
            stateSize = totalContextSize;
            sharedContextSize = Math.multiplyExact(totalContextSize, groupConfig.maxAgents());
        }

        if (stateSize <= 0) {
            throw new IllegalStateException("Per-agent state size is invalid after update");
        }

        sharedContextBuffer = SpartanModelAllocator.allocateContextBuffer(arena, sharedContextSize);
        refreshSharedContextBuffer(contextImpl);

        int result = SpartanNative.spartanRegisterMultiAgent(
            multiAgentId,
            sharedContextBuffer,
            sharedContextSize,
            actionOutputBuffer,
            (int) (actionOutputBuffer.byteSize() / Double.BYTES), // actionBufferSize
            sharedActions.size(),                                 // actionFieldSize
            stateSize,                                            // stateSize (per agent)
            groupConfig.maxAgents()
        );

        if (result != 0) {
            throw new RuntimeException("Native failed to register multi-agent group " + multiAgentId);
        }

        registered = true;
    }

    private void refreshSharedContextBuffer(@NotNull SpartanContextImpl contextImpl) {
        int currentSize = contextImpl.getSize();
        if (aggregatedContextLayout) {
            if (currentSize != sharedContextSize) {
                throw new IllegalStateException(
                    "Shared context size changed from " + sharedContextSize + " to " + currentSize + " after registration");
            }
            long bytesToCopy = (long) currentSize * Double.BYTES;
            MemorySegment.copy(contextImpl.getData(), 0, sharedContextBuffer, 0, bytesToCopy);
            return;
        }

        if (currentSize != stateSize) {
            throw new IllegalStateException(
                "Shared context size changed from " + stateSize + " to " + currentSize + " after registration");
        }

        long bytesPerAgent = (long) stateSize * Double.BYTES;
        for (int i = 0; i < groupConfig.maxAgents(); i++) {
            long destOffset = bytesPerAgent * i;
            MemorySegment.copy(contextImpl.getData(), 0, sharedContextBuffer, destOffset, bytesPerAgent);
        }
    }

    @Override
    public boolean isRegistered() {
        return registered;
    }

    @Override
    public @NotNull SpartanAgent<ConfigType> addAgent(
            @NotNull String identifier,
            @NotNull ConfigType config
    ) {
        if (closed) {
            throw new IllegalStateException("Multi-agent group is closed");
        }
        if (agentById.values().stream().anyMatch(agent -> agent.getIdentifier().equals(identifier))) {
            throw new IllegalArgumentException("Agent with identifier '" + identifier + "' already exists");
        }
        if (agentById.size() >= groupConfig.maxAgents()) {
            throw new IllegalStateException("Multi-agent group is at max capacity: " + groupConfig.maxAgents());
        }

        long agentId = nextAgentId.getAndIncrement();

        if (stateSize <= 0) {
            sharedContext.update();
            if (sharedContext.getSize() <= 0) {
                throw new IllegalStateException("Shared context size is invalid after update");
            }
        }
        int actionCount = sharedActions.size();

        MemorySegment configSegment = SpartanModelAllocator.serialize(arena, config, stateSize, actionCount);

        SpartanModelImpl<ConfigType> agent = new SpartanModelImpl<>(
                identifier,
                agentId,
                config,
                this.sharedContext,
                this.sharedActions
        );

        int result = SpartanNative.spartanMultiAgentAddAgent(
                multiAgentId,
                agentId,
                configSegment,
                agent.getModelWeightsBuffer(),
                (int) (agent.getModelWeightsBuffer().byteSize() / Double.BYTES),
                agent.getCriticWeightsBuffer(),
                (int) (agent.getCriticWeightsBuffer().byteSize() / Double.BYTES)
        );

        if (result != 0) {
            throw new RuntimeException("Native failed to add agent to group " + multiAgentId);
        }

        agentById.put(agentId, agent);

        return agent;
    }

    @Override
    public boolean removeAgent(long agentId) {
        if (agentById.remove(agentId) != null) {
            SpartanNative.spartanMultiAgentRemoveAgent(multiAgentId, agentId);
            return true;
        }
        return false;
    }

    @Override
    public @NotNull Optional<SpartanAgent<ConfigType>> getAgent(long agentId) {
        return Optional.ofNullable(agentById.get(agentId));
    }

    @Override
    public @NotNull Optional<SpartanAgent<ConfigType>> getAgentByIdentifier(@NotNull String identifier) {
        // Linear search is acceptable as this is not a hot-path method
        for (SpartanAgent<ConfigType> agent : agentById.values()) {
            if (agent.getIdentifier().equals(identifier)) {
                return Optional.of(agent);
            }
        }
        return Optional.empty();
    }

    @Override
    public void tick() {
        if (!registered) {
            throw new IllegalStateException("Multi-agent group not registered");
        }
        if (closed) {
            throw new IllegalStateException("Multi-agent group is closed");
        }

        //  Update SHARED context (once for all agents)
        if (sharedContext instanceof SpartanContextImpl contextImpl) {
            contextImpl.update();
            refreshSharedContextBuffer(contextImpl);
        }

        // C++ executes all agents in parallel + critic evaluation
        SpartanNative.spartanTickMultiAgent(multiAgentId);
    }

    @Override
    public void applyRewards(double @NotNull [] rewards) {
        if (rewards.length != agentById.size()) {
            throw new IllegalArgumentException(
                "Rewards array size (" + rewards.length +
                ") must match agent count (" + agentById.size() + ")"
            );
        }

        int idx = 0;
        for (SpartanAgent<ConfigType> agent : agentById.values()) {
            agent.applyReward(rewards[idx++]);
        }
    }

    @Override
    public @NotNull SpartanContext getContext() {
        return sharedContext;
    }

    @Override
    public int getAgentCount() {
        return agentById.size();
    }

    @Override
    public void decayExplorationForAll() {
        for (SpartanAgent<ConfigType> agent : agentById.values()) {
            agent.decayExploration();
        }
    }

    /**
     * Unregisters from C++ and releases resources.
     * After this call, the group cannot be used.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            agentById.values().forEach(SpartanAgent::close);
            agentById.clear();
            arena.close();
        }
    }

    @Override
    public void save(@NotNull Path filePath) throws SpartanPersistenceException {
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(filePath))) {
            for (SpartanAgent<ConfigType> agent : agentById.values()) {
                ZipEntry entry = new ZipEntry(agent.getIdentifier() + ".submodel");
                zos.putNextEntry(entry);

                Path tempFile = Files.createTempFile("spartan_agent", ".tmp");
                try {
                    agent.saveModel(tempFile);
                    Files.copy(tempFile, zos);
                } finally {
                    Files.deleteIfExists(tempFile);
                    zos.closeEntry();
                }
            }
        } catch (IOException e) {
            throw new SpartanPersistenceException("Failed to save multi-agent model as zip", e);
        }
    }

    @Override
    public void load(@NotNull Path filePath) throws SpartanPersistenceException {
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(filePath))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.getName().endsWith(".submodel")) {
                    String identifier = entry.getName().substring(0, entry.getName().length() - ".submodel".length());
                    Optional<SpartanAgent<ConfigType>> agentOpt = getAgentByIdentifier(identifier);
                    if (agentOpt.isPresent()) {
                        Path tempFile = Files.createTempFile("spartan_agent", ".tmp");
                        try {
                            Files.copy(zis, tempFile, StandardCopyOption.REPLACE_EXISTING);
                            agentOpt.get().loadModel(tempFile);
                        } finally {
                            Files.deleteIfExists(tempFile);
                        }
                    }
                }
                zis.closeEntry();
            }
        } catch (IOException e) {
            throw new SpartanPersistenceException("Failed to load multi-agent model from zip", e);
        }
    }
}
