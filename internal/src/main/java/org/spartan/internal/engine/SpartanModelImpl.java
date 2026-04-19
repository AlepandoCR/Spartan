package org.spartan.internal.engine;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.SpartanAgent;
import org.spartan.api.engine.SpartanModel;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.action.type.SpartanAction;
import org.spartan.internal.engine.action.SpartanActionManagerImpl;
import org.spartan.api.engine.config.*;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.exception.SpartanPersistenceException;
import org.spartan.internal.engine.context.SpartanContextImpl;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.engine.model.SpartanConfigLayout;
import org.spartan.internal.engine.model.SpartanModelAllocator;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class SpartanModelImpl<SpartanModelConfigType extends SpartanModelConfig>
        implements SpartanAgent<SpartanModelConfigType> {

    private final Arena arena;
    private final long agentId;
    private final SpartanModelConfigType config;
    private final SpartanContextImpl context;
    private final List<SpartanAction> actions;
    private final SpartanActionManager actionManager;

    private final MemorySegment configBuffer;
    private final MemorySegment modelWeightsBuffer;
    private final MemorySegment criticWeightsBuffer;
    private final MemorySegment actionOutputBuffer;
    private final MemorySegment contextBuffer; // Reference to context's buffer

    private final int modelWeightsCount;
    private final int criticWeightsCount;
    private final int actionCount;

    private final String identifier;

    private boolean isRegistered = false;
    private boolean isClosed = false;

    private double pendingReward = 0.0;
    private double episodeReward = 0.0;
    private double accumulatedTickReward = 0.0;

    public SpartanModelImpl(@NotNull String identifier, long agentId, SpartanModelConfigType config, SpartanContext context, Iterable<SpartanAction> actions) {
        this.identifier = identifier;
        this.arena = Arena.ofShared(); // Controlled by this model instance
        this.config = config;
        this.agentId = agentId;

        if (!(context instanceof SpartanContextImpl)) {
            throw new IllegalArgumentException("SpartanContext must be created via SpartanApi");
        }
        this.context = (SpartanContextImpl) context;

        this.actions = new ArrayList<>();
        actions.forEach(this.actions::add);
        this.actionManager = new SpartanActionManagerImpl();
        for (SpartanAction action : this.actions) {
            this.actionManager.registerAction(action);
        }
        this.actionCount = this.actions.size();
        int stateSize = context.getSize();

        // Serialize Config
        this.configBuffer = SpartanModelAllocator.serialize(this.arena, config, stateSize, actionCount);

        // CRITICAL VALIDATION: Verify serialize() actually filled the buffer
        int storedSignature = configBuffer.get(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_LAYOUT_SIGNATURE_OFFSET);
        int expectedSignature = SpartanModelAllocator.getLayoutSignature();
        if (storedSignature == 0) {
            throw new IllegalStateException(
                "[CONSTRUCTOR VALIDATION FAILED] SpartanModelAllocator.serialize() did not write the signature field! " +
                "Expected signature=" + expectedSignature + " but got 0. " +
                "This indicates the config buffer was never initialized properly.");
        }
        SpartanNative.spartanLog("[Spartan-Java] [CONSTRUCTOR] Config buffer validated: signature=" + storedSignature + " (expected=" + expectedSignature + ")");

        // Allocate Buffers

        long mWeights;
        long cWeights;

        switch (config) {
            case CuriosityDrivenRecurrentSoftActorCriticConfig cConfig -> {
                mWeights = SpartanModelAllocator.calculateCuriosityDrivenRecurrentSoftActorCriticModelWeightCount(cConfig, stateSize, actionCount);
                cWeights = SpartanModelAllocator.calculateCuriosityDrivenRecurrentSoftActorCriticCriticWeightCount(cConfig, stateSize, actionCount);
            }
            case RecurrentSoftActorCriticConfig rConfig -> {
                mWeights = SpartanModelAllocator.calculateRSACModelWeightCount(rConfig, stateSize, actionCount);
                cWeights = SpartanModelAllocator.calculateRSACCriticWeightCount(rConfig, stateSize, actionCount);
            }
            case DoubleDeepQNetworkConfig dConfig -> {
                mWeights = SpartanModelAllocator.calculateDDQNModelWeightCount(dConfig, stateSize, actionCount);
                cWeights = SpartanModelAllocator.calculateDDQNCriticWeightCount(dConfig);
            }
            case AutoEncoderCompressorConfig aConfig -> {
                mWeights = SpartanModelAllocator.calculateAutoEncoderModelWeightCount(aConfig, stateSize);
                cWeights = SpartanModelAllocator.calculateAutoEncoderCriticWeightCount(aConfig);
            }
            default ->
                // Fallback or error? Assuming known types from factory.
                    throw new IllegalArgumentException("Unsupported config type: " + config.getClass());
        }

        this.modelWeightsCount = (int) mWeights;
        this.criticWeightsCount = (int) cWeights;

        this.modelWeightsBuffer = SpartanModelAllocator.allocateModelWeightsBuffer(mWeights, arena);
        this.criticWeightsBuffer = SpartanModelAllocator.allocateCriticWeightsBuffer(cWeights, arena);
        this.actionOutputBuffer = SpartanModelAllocator.allocateActionOutputBuffer(arena, actionCount);

        this.contextBuffer = this.context.getData(); // SpartanContextImpl.getData() returns MemorySegment
    }

    @Override
    public long getAgentIdentifier() {
        return agentId;
    }

    @Override
    public @NotNull String getIdentifier() {
        return identifier;
    }

    @Override
    public @NotNull SpartanContext getSpartanContext() {
        return context;
    }

    @Override
    public @NotNull SpartanModelConfigType getSpartanModelConfig() {
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
    public @NotNull SpartanActionManager getActionManager() {
        return actionManager;
    }

    @Override
    public @NotNull MemorySegment getCriticWeightsBuffer() {
        return criticWeightsBuffer;
    }

    @Override
    public void register() {
       if (isRegistered) return;



       int storedSignature = configBuffer.get(ValueLayout.JAVA_INT,
               SpartanConfigLayout.BASE_LAYOUT_SIGNATURE_OFFSET);
       int expectedSignature = SpartanModelAllocator.getLayoutSignature();

       if (storedSignature == 0) {
           throw new IllegalStateException(
               "[CRITICAL] Config buffer was never properly initialized! " +
               "Constructor should have called SpartanModelAllocator.serialize() to fill the entire buffer. " +
               "Got signature=0, expected=" + expectedSignature);
       }

       if (storedSignature != expectedSignature) {
           SpartanNative.spartanLog("[Spartan-Java] layout signature mismatch: stored=" + storedSignature + ", expected=" + expectedSignature);
       }

       SpartanNative.spartanLog("[Spartan-Java] Config buffer initialized with signature=" + storedSignature);

       int result = SpartanNative.spartanRegisterModel(
           agentId,
           configBuffer,
           criticWeightsBuffer,
           criticWeightsCount,
           modelWeightsBuffer,
           modelWeightsCount,
           contextBuffer,
           context.getSize(),
           actionOutputBuffer,
           actionCount
       );
       if (result != 0) {
           throw new RuntimeException("Native model registration failed: " + result);
       }
       isRegistered = true;
    }

    @Override
    public boolean isRegistered() {
        return isRegistered;
    }

    @Override
    public void tick(double reward) {
        episodeReward += reward;
        accumulatedTickReward += reward;
        double currentReward = accumulatedTickReward;
        accumulatedTickReward = 0.0;
        executeNativeTick(currentReward);
    }

    @Override
    public void applyReward(double reward) {
        episodeReward += reward;
        accumulatedTickReward += reward;
    }

    @Override
    public double getEpisodeReward() {
        return episodeReward;
    }

    @Override
    public void resetEpisode() {
        pendingReward = 0.0;
        episodeReward = 0.0;
    }

    @Override
    public void setLiveExplorationRate(double newEpsilon) {
        configBuffer.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_OFFSET, newEpsilon);
    }

    @Override
    public void tick() {
        double currentReward = accumulatedTickReward;
        accumulatedTickReward = 0.0;
        executeNativeTick(currentReward);
    }

    private void executeNativeTick(double currentReward) {
        if (!isRegistered) throw new IllegalStateException("Model not registered");

        context.update();

        SpartanNative.spartanTickAgent(agentId, currentReward);

        for (int i = 0; i < actions.size(); i++) {
            double rawOutput = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            actions.get(i).tick(rawOutput);
        }
    }

    @Override
    public void saveModel(@NotNull Path filePath) throws SpartanPersistenceException {
        int modelTypeId = determineModelTypeId();
        int result = SpartanNative.spartanSaveModel(agentId, filePath.toString(), modelTypeId);
        if (result != 0) throw new SpartanPersistenceException("Save failed: " + result);
    }

    @Override
    public void loadModel(@NotNull Path filePath) throws SpartanPersistenceException {
        int modelTypeId = determineModelTypeId();
        int result = SpartanNative.spartanLoadModel(agentId, filePath.toString(), modelTypeId);
        if (result != 0) throw new SpartanPersistenceException("Load failed: " + result);
    }

    /**
     * Determines the model type ID based on the config type.
     *
     * @return the model type ID matching SpartanModelType enum values
     */
    private int determineModelTypeId() {
        if (config instanceof RecurrentSoftActorCriticConfig) {
            return SpartanModelType.RECURRENT_SOFT_ACTOR_CRITIC.getNativeValue();
        } else if (config instanceof DoubleDeepQNetworkConfig) {
            return SpartanModelType.DOUBLE_DEEP_Q_NETWORK.getNativeValue();
        } else if (config instanceof AutoEncoderCompressorConfig) {
            return SpartanModelType.AUTO_ENCODER_COMPRESSOR.getNativeValue();
        } else if (config instanceof CuriosityDrivenRecurrentSoftActorCriticConfig) {
            return SpartanModelType.CURIOSITY_DRIVEN_RECURRENT_SOFT_ACTOR_CRITIC.getNativeValue();
        } else {
            throw new IllegalArgumentException("Unknown config type: " + config.getClass());
        }
    }

    @Override
    public void decayExploration() {
        SpartanNative.spartanDecayExploration(agentId);
    }

    @Override
    public @NotNull SpartanModel<SpartanModelConfigType> copy(@NotNull String newIdentifier) {
        long newAgentId = System.nanoTime(); // Generate unique ID
        SpartanModelImpl<SpartanModelConfigType> newModel = new SpartanModelImpl<>(
                newIdentifier,
                newAgentId,
                config,
                context,
                actions
        );

        // Ensure new model is ready
        MemorySegment.copy(modelWeightsBuffer, 0, newModel.getModelWeightsBuffer(), 0, modelWeightsBuffer.byteSize());
        MemorySegment.copy(criticWeightsBuffer, 0, newModel.getCriticWeightsBuffer(), 0, criticWeightsBuffer.byteSize());

        newModel.register();
        return newModel;
    }

    @Override
    public void close() {
        if (!isClosed) {
            if (isRegistered) {
                SpartanNative.spartanUnregisterModel(agentId);
            }
            arena.close();
            isClosed = true;
        }
    }
}
