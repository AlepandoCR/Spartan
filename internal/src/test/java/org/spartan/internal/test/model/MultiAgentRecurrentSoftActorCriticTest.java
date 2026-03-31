package org.spartan.internal.test.model;

import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.spartan.api.SpartanApi;
import org.spartan.api.engine.SpartanAgent;
import org.spartan.api.engine.model.SpartanMultiAgentModel;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.action.type.SpartanAction;
import org.spartan.api.engine.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.engine.config.SpartanModelConfig;
import org.spartan.api.engine.config.SpartanMultiAgentGroupConfig;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.context.element.SpartanContextElement;
import org.spartan.internal.engine.config.spi.SpartanConfigFactoryServiceProviderImpl;
import org.spartan.internal.spi.SpartanApiProviderImpl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Comprehensive test suite for Multi-Agent Reinforcement Learning (MARL) with SAC agents.
 */
@DisplayName("Multi-Agent RSAC (MARL CTDE) Integration Test Suite")
public class MultiAgentRecurrentSoftActorCriticTest {

    private static SpartanApi api;

    @BeforeAll
    public static void initializeNativeEngine() {
        System.out.println("======================================================================");
        System.out.println("  MULTI-AGENT REINFORCEMENT LEARNING TEST SUITE");
        System.out.println("======================================================================");

        // Force initialization of SPI providers since we are in a unit test environment
        new SpartanApiProviderImpl();
        new SpartanConfigFactoryServiceProviderImpl();

        api = SpartanApi.create();
        System.out.println("✓ Native engine initialized\n");
    }

    @Test
    @DisplayName("Test 1: Multi-Agent Group Registration")
    public void testMultiAgentGroupRegistration() throws Exception {
        System.out.println(">>> Test 1: Multi-Agent Group Registration");

        final int numAgents = 3;
        final int stateSize = 16;
        final int actionSize = 4;

        SpartanContext sharedContext = api.createContext("shared_context");
        SpartanContextElement stateElement = new SimpleContextElement("shared_state", stateSize * numAgents);
        sharedContext.addElement(stateElement, 0);

        SpartanActionManager actionManager = createActionManager(actionSize);

        SpartanMultiAgentGroupConfig groupConfig = SpartanMultiAgentGroupConfig.builder()
                .maxAgents(numAgents)
                .build();

        SpartanMultiAgentModel<SpartanModelConfig> multiAgent =
                api.createMultiAgentModel("multi_agent_group_1", groupConfig, sharedContext, actionManager);

        multiAgent.register();
        assert multiAgent.isRegistered() : "Multi-agent group failed to register";

        System.out.println("✓ Multi-agent group registered successfully");
        System.out.println("  - Group ID: multi_agent_group_1");
        System.out.println("  - Max Agents: " + numAgents);
        System.out.println("  - State Size per Agent: " + stateSize);
        System.out.println("  - Action Size: " + actionSize);
        System.out.println("  - Shared Context Total Size: " + (stateSize * numAgents) + "\n");

        multiAgent.close();
    }

    @Test
    @DisplayName("Test 2: Dynamic Agent Addition & Removal")
    public void testDynamicAgentManagement() throws Exception {
        System.out.println(">>> Test 2: Dynamic Agent Addition & Removal");

        final int numAgents = 3;
        final int stateSize = 16;
        final int actionSize = 4;

        SpartanContext sharedContext = api.createContext("dynamic_context");
        SpartanContextElement stateElement = new SimpleContextElement("state", stateSize * numAgents);
        sharedContext.addElement(stateElement, 0);

        SpartanActionManager actionManager = createActionManager(actionSize);

        SpartanMultiAgentGroupConfig groupConfig = SpartanMultiAgentGroupConfig.builder()
                .maxAgents(numAgents)
                .build();

        SpartanMultiAgentModel<SpartanModelConfig> multiAgent =
                api.createMultiAgentModel("multi_agent_dynamic", groupConfig, sharedContext, actionManager);

        multiAgent.register();

        System.out.println("  Adding agents...");
        List<SpartanAgent<SpartanModelConfig>> agents = new ArrayList<>();

        for (int i = 0; i < numAgents; i++) {
            RecurrentSoftActorCriticConfig config = buildRsacConfig();
            SpartanAgent<SpartanModelConfig> agent = multiAgent.addAgent("agent_" + i, config);
            agents.add(agent);
            System.out.println("    ✓ Added agent_" + i);
        }

        assert multiAgent.getAgentCount() == numAgents;

        System.out.println("  Removing agent_1...");
        boolean removed = multiAgent.removeAgent(agents.get(1).getAgentIdentifier());
        assert removed;
        assert multiAgent.getAgentCount() == numAgents - 1;

        multiAgent.close();
        System.out.println("✓ Dynamic add/remove OK\n");
    }

    @Test
    @DisplayName("Test 3: Parallel MARL Tick with Reward Distribution")
    public void testParallelMarlTick() throws Exception {
        System.out.println(">>> Test 3: Parallel MARL Tick with Reward Distribution");

        final int numAgents = 3;
        final int stateSize = 16;
        final int actionSize = 4;
        final int ticks = 5;

        SpartanContext sharedContext = api.createContext("tick_context");
        SimpleContextElement stateElement = new SimpleContextElement("state", stateSize * numAgents);
        sharedContext.addElement(stateElement, 0);

        SpartanActionManager actionManager = createActionManager(actionSize);

        SpartanMultiAgentGroupConfig groupConfig = SpartanMultiAgentGroupConfig.builder()
                .maxAgents(numAgents)
                .build();

        SpartanMultiAgentModel<SpartanModelConfig> multiAgent =
                api.createMultiAgentModel("tick_test", groupConfig, sharedContext, actionManager);

        multiAgent.register();

        for (int i = 0; i < numAgents; i++) {
            RecurrentSoftActorCriticConfig config = buildRsacConfig();
            multiAgent.addAgent("agent_" + i, config);
        }

        System.out.println("  Running " + ticks + " ticks...");
        long startTime = System.nanoTime();

        for (int tick = 0; tick < ticks; tick++) {
            // Write dummy state
            for (int agent = 0; agent < numAgents; agent++) {
                for (int state = 0; state < stateSize; state++) {
                    stateElement.data[agent * stateSize + state] = Math.sin(tick * 0.1 + agent);
                }
            }

            multiAgent.tick();

            double[] rewards = new double[numAgents];
            Arrays.fill(rewards, Math.cos(tick * 0.1) * 0.5);
            multiAgent.applyRewards(rewards);
        }

        long elapsedTime = System.nanoTime() - startTime;
        System.out.println("  ✓ Completed in " + (elapsedTime / 1_000_000.0) + " ms\n");

        multiAgent.close();
    }

    private static SpartanActionManager createActionManager(int actionSize) {
        SpartanActionManager manager = api.createActionManager();
        for (int i = 0; i < actionSize; i++) {
            final int idx = i;
            manager.registerAction(new SpartanAction() {
                @Override public @NotNull String identifier() { return "action_" + idx; }
                @Override public double taskMaxMagnitude() { return 1.0; }
                @Override public double taskMinMagnitude() { return -1.0; }
                @Override public void task(double normalizedMagnitude) { }
                @Override public double award() { return 0.0; }
            });
        }
        return manager;
    }

    private static RecurrentSoftActorCriticConfig buildRsacConfig() {
        return RecurrentSoftActorCriticConfig.builder()
                .learningRate(0.001)
                .gamma(0.99)
                .epsilon(0.1)
                .epsilonMin(0.01)
                .epsilonDecay(0.995)
                .hiddenStateSize(32)
                .recurrentLayerDepth(1)
                .actorHiddenLayerNeuronCount(64)
                .actorHiddenLayerCount(2)
                .criticHiddenLayerNeuronCount(256)
                .criticHiddenLayerCount(2)
                .build();
    }

    private static final class SimpleContextElement implements SpartanContextElement {
        private final String id;
        private final double[] data;

        private SimpleContextElement(String id, int size) {
            this.id = id;
            this.data = new double[size];
        }

        @Override
        public double @NotNull [] getData() {
            return data;
        }

        @Override
        public int getSize() {
            return data.length;
        }

        @Override
        public void prepare() {
        }

        @Override
        public void tick() {
        }

        @Override
        public @NotNull String getIdentifier() {
            return id;
        }
    }
}

