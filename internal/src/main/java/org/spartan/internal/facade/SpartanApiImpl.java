package org.spartan.internal.facade;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.SpartanApi;
import org.spartan.api.engine.SpartanModel;
import org.spartan.api.engine.SpartanMultiAgentModel;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.config.*;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.model.AutoEncoderCompressorModel;
import org.spartan.api.engine.model.CuriosityDrivenRecurrentSoftActorCriticModel;
import org.spartan.api.engine.model.DoubleDeepQNetworkModel;
import org.spartan.api.engine.model.RecurrentSoftActorCriticModel;
import org.spartan.internal.engine.SpartanModelImpl;
import org.spartan.internal.engine.action.SpartanActionManagerImpl;
import org.spartan.internal.engine.context.SpartanContextImpl;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.engine.model.AutoEncoderCompressorModelImpl;
import org.spartan.internal.engine.model.CuriosityDrivenRecurrentSoftActorCriticModelImpl;
import org.spartan.internal.engine.model.DoubleDeepQNetworkModelImpl;
import org.spartan.internal.engine.model.RecurrentSoftActorCriticModelImpl;
import org.spartan.internal.engine.SpartanMultiAgentModelImpl;

import java.lang.foreign.Arena;
import java.util.concurrent.ThreadLocalRandom;

public class SpartanApiImpl implements SpartanApi {

    private final Arena arena;

    public SpartanApiImpl() {
        this.arena = Arena.ofShared();
        // Ensure native is loaded
        SpartanNative.spartanInit();
    }

    @Override
    public @NotNull SpartanContext createContext(@NotNull String identifier) {
        // Pass the shared arena. When API closes, arena closes, invalidating all contexts context.
        return new SpartanContextImpl(identifier, arena);
    }

    @Override
    public @NotNull SpartanActionManager createActionManager() {
        return new SpartanActionManagerImpl();
    }

    @Override
    public <SpartanModelConfigType extends SpartanModelConfig> @NotNull SpartanModel<SpartanModelConfigType> createModel(
            @NotNull String identifier,
            @NotNull SpartanModelConfigType config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    ) {
        return new SpartanModelImpl<>(identifier, ThreadLocalRandom.current().nextLong(), config, context, actions.getActions());
    }

    @Override
    public @NotNull RecurrentSoftActorCriticModel createRecurrentSoftActorCritic(
            @NotNull String identifier,
            @NotNull RecurrentSoftActorCriticConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    ) {
        return new RecurrentSoftActorCriticModelImpl(
                identifier,
                ThreadLocalRandom.current().nextLong(),
                config,
                context,
                arena,
                actions
        );
    }

    @Override
    public @NotNull CuriosityDrivenRecurrentSoftActorCriticModel createCuriosityDrivenRecurrentSoftActorCritic(
            @NotNull String identifier,
            @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    ) {
        return new CuriosityDrivenRecurrentSoftActorCriticModelImpl(
                identifier,
                ThreadLocalRandom.current().nextLong(),
                config,
                context,
                arena,
                actions
        );
    }

    @Override
    public @NotNull DoubleDeepQNetworkModel createDoubleDeepQNetwork(
            @NotNull String identifier,
            @NotNull DoubleDeepQNetworkConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions)
    {
        return new DoubleDeepQNetworkModelImpl(
                identifier,
                ThreadLocalRandom.current().nextLong(),
                config,
                context,
                arena,
                actions
        );
    }

    @Override
    public @NotNull AutoEncoderCompressorModel createAutoEncoderCompressor(
            @NotNull String identifier,
            @NotNull AutoEncoderCompressorConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    ) {
        return new AutoEncoderCompressorModelImpl(
                identifier,
                ThreadLocalRandom.current().nextLong(),
                config,
                context,
                arena,
                actions
        );
    }

    @Override
    public @NotNull <SpartanModelConfigType extends SpartanModelConfig> SpartanMultiAgentModel<SpartanModelConfigType> createMultiAgentModel(
            @NotNull String identifier,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    ) {
        SpartanMultiAgentGroupConfig defaultConfig = SpartanMultiAgentGroupConfig.builder().build();
        return (SpartanMultiAgentModel<SpartanModelConfigType>) createMultiAgentModel(
                identifier,
                defaultConfig,
                context,
                actions
        );
    }

    @Override
    public @NotNull SpartanMultiAgentModel<SpartanModelConfig> createMultiAgentModel(
            @NotNull String identifier,
            @NotNull SpartanMultiAgentGroupConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    ) {
        return new SpartanMultiAgentModelImpl<>(identifier, config, context, actions.getActions());
    }

    @Override
    public void close() {
        if (arena.scope().isAlive()) {
            arena.close();
        }
    }
}
