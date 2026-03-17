package org.spartan.api.agent.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanCompressor;
import org.spartan.api.agent.config.AutoEncoderCompressorConfig;

import org.spartan.api.SpartanApi;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.context.SpartanContext;

/**
 * A utility model for dimension reduction.
 * <p>
 * Does not earn rewards or make decisions. It just watches the world and learns to summarize it.
 */
public interface AutoEncoderCompressorModel extends SpartanCompressor<AutoEncoderCompressorConfig> {
    /**
     * Helper to build this specific model type.
     *
     * @param api the API instance
     * @param config the specific AutoEncoder config
     * @param context the context
     * @param actions the action manager
     * @return the new model
     */
    static AutoEncoderCompressorModel build(
            @NotNull SpartanApi api,
            @NotNull String identifier,
            @NotNull AutoEncoderCompressorConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions) {
        return api.createAutoEncoderCompressor(
                identifier,
                config,
                context,
                actions
        );
    }
}
