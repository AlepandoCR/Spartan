package org.spartan.internal.facade;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.SpartanApi;
import org.spartan.api.agent.SpartanModel;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.config.*;
import org.spartan.api.agent.context.SpartanContext;
import org.spartan.api.agent.model.AutoEncoderCompressorModel;
import org.spartan.api.agent.model.CuriosityDrivenRecurrentSoftActorCriticModel;
import org.spartan.api.agent.model.DoubleDeepQNetworkModel;
import org.spartan.api.agent.model.RecurrentSoftActorCriticModel;
import org.spartan.internal.agent.SpartanModelImpl;
import org.spartan.internal.agent.action.SpartanActionManagerImpl;
import org.spartan.internal.agent.context.SpartanContextImpl;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.model.AutoEncoderCompressorModelImpl;
import org.spartan.internal.model.CuriosityDrivenRecurrentSoftActorCriticModelImpl;
import org.spartan.internal.model.DoubleDeepQNetworkModelImpl;
import org.spartan.internal.model.RecurrentSoftActorCriticModelImpl;

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
    public SpartanContext createContext(@NotNull String identifier) {
        // Pass the shared arena. When API closes, arena closes, invalidating all contexts context.
        return new SpartanContextImpl(identifier, arena);
    }

    @Override
    public SpartanActionManager createActionManager() {
        return new SpartanActionManagerImpl();
    }

    @Override
    public <SpartanModelConfigType extends SpartanModelConfig> SpartanModel<SpartanModelConfigType> createModel(
            @NotNull String identifier,
            @NotNull SpartanModelConfigType config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    ) {
        return new SpartanModelImpl<>(identifier, ThreadLocalRandom.current().nextLong(), config, context, actions.getActions());
    }

    @Override
    public RecurrentSoftActorCriticModel createRecurrentSoftActorCritic(
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
    public CuriosityDrivenRecurrentSoftActorCriticModel createCuriosityDrivenRecurrentSoftActorCritic(
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
    public DoubleDeepQNetworkModel createDoubleDeepQNetwork(
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
    public AutoEncoderCompressorModel createAutoEncoderCompressor(
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
    public void close() {
        if (arena.scope().isAlive()) {
            arena.close();
        }
    }
}
