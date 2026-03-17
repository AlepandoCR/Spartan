package org.spartan.api;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.agent.SpartanModel;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.config.SpartanModelConfig;
import org.spartan.api.agent.context.SpartanContext;
import org.spartan.api.agent.config.*;
import org.spartan.api.agent.model.*;
import org.spartan.api.spi.SpartanCoreRegistry;

/**
 * The main entry point for the Spartan Deep Learning engine.
 * <p>
 * <b>Concept:</b> This class acts as the "Engine Instance". It manages the connection to the native C++ core
 * and handles memory allocation (the "Arena"). Everything you create (Contexts, Models, Actions) must belong to a SpartanApi instance.
 * <p>
 * <b>Usage Pattern (Try-With-Resources):</b>
 * <pre>{@code
 * try (SpartanApi api = SpartanApi.create()) {
 *     // Create context, actions, models here...
 *
 * } // Automatically closes and frees all native memory
 * }</pre>
 */
public interface SpartanApi extends AutoCloseable {

    /**
     * Creates a new instance of the Spartan Engine API.
     * Use this in a try-with-resources block to ensure proper cleanup.
     *
     * @return a new API instance
     */
    static SpartanApi create() {
        return SpartanCoreRegistry.get().createApi();
    }

    /**
     * Creates a new Observation Context.
     * <p>
     * <b>Concept:</b> The Context is the "eyes and ears" of your AI. It holds a list of sensors (Elements)
     * that read data from your game world (health, position, enemy locations).
     *
     * @param identifier a unique name for this context (useful for debugging)
     *
     * @return an empty context ready for elements
     */
    SpartanContext createContext(@NotNull String identifier);

    /**
     * Creates a new Action Manager.
     * <p>
     * <b>Concept:</b> The Action Manager is the "hands" of your AI. It holds a registry of all possible moves (Actions)
     * the agent can perform.
     *
     * @return a new, empty action manager
     */
    SpartanActionManager createActionManager();


    /**
     * Generic method to create a model from any configuration.
     * Prefer using the specific methods below for type safety.
     *
     * @param config the model configuration
     * @param context the observation context
     * @param actions the action manager
     * @param <SpartanModelConfigType> the type of configuration
     * @return the instantiated model
     */
    <SpartanModelConfigType extends SpartanModelConfig> SpartanModel<SpartanModelConfigType> createModel(
            @NotNull String identifier,
            @NotNull SpartanModelConfigType config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    );

    /**
     * Creates a Recurrent Soft Actor-Critic (RSAC) agent.
     * Best for: Continuous control (steering, movement), complex tasks requiring memory.
     *
     * @param config the RSAC configuration
     * @param context the observation context
     * @param actions the action manager
     * @return the RSAC model instance
     */
    RecurrentSoftActorCriticModel createRecurrentSoftActorCritic(
            @NotNull String identifier,
            @NotNull RecurrentSoftActorCriticConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    );

    /**
     * Creates a Curiosity-Driven RSAC agent.
     * Best for: sparse reward environments where the agent needs to explore autonomously.
     *
     * @param config the Curiosity configuration
     * @param context the observation context
     * @param actions the action manager
     * @return the Curiosity model instance
     */
    CuriosityDrivenRecurrentSoftActorCriticModel createCuriosityDrivenRecurrentSoftActorCritic(
            @NotNull String identifier,
            @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    );

    /**
     * Creates a Double Deep Q-Network (DDQN) agent.
     * Best for: Discrete control (selecting 1 item from a menu, pressing a specific button).
     *
     * @param config the DDQN configuration
     * @param context the observation context
     * @param actions the action manager
     * @return the DDQN model instance
     */
    DoubleDeepQNetworkModel createDoubleDeepQNetwork(
            @NotNull String identifier,
            @NotNull DoubleDeepQNetworkConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    );

    /**
     * Creates an AutoEncoder Compressor model.
     * Best for: Pre-processing huge inputs (like images) into small vectors for other agents.
     *
     * @param config the AutoEncoder configuration
     * @param context the observation context
     * @param actions the action manager
     * @return the AutoEncoder model instance
     */
    AutoEncoderCompressorModel createAutoEncoderCompressor(
            @NotNull String identifier,
            @NotNull AutoEncoderCompressorConfig config,
            @NotNull SpartanContext context,
            @NotNull SpartanActionManager actions
    );

    /**
     * Closes the API and frees all associated native memory (Contexts, Models, Buffers).
     * Call this when the plugin disables or the game session ends.
     */
    @Override
    void close();
}
