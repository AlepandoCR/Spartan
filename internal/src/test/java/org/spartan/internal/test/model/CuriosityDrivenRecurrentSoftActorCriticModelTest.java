package org.spartan.internal.test.model;

import org.junit.jupiter.api.*;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.action.type.SpartanAction;
import org.spartan.api.agent.config.CuriosityDrivenRecurrentSoftActorCriticConfig;
import org.spartan.api.agent.context.SpartanContext;
import org.spartan.api.agent.context.element.SpartanContextElement;
import org.spartan.internal.agent.context.SpartanContextImpl;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.model.CuriosityDrivenRecurrentSoftActorCriticModelImpl;
import org.spartan.internal.model.SpartanModelAllocator;
import org.spartan.internal.config.spi.SpartanConfigFactoryServiceProviderImpl;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Curiosity-Driven Recurrent Soft Actor-Critic model.
 */
@DisplayName("Curiosity-Driven RSAC Model - Comprehensive Integration Test")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class CuriosityDrivenRecurrentSoftActorCriticModelTest {

    private static final long BASE_AGENT_IDENTIFIER = 0x0CCA_1051_0001L;
    private static final int OBSERVATION_SIZE = 16;
    private static final int CONTINUOUS_ACTION_SIZE = 4;
    private static final int HIDDEN_STATE_SIZE = 64;
    private static final int FORWARD_DYNAMICS_HIDDEN_SIZE = 128;
    private static final int SIMULATION_TICK_COUNT = 50;

    private static boolean nativeEngineInitialized = false;
    private static long agentCounter = 0;
    private MockObservationElement lastObservationElement;

    @BeforeAll
    static void initializeNativeEngine() {
        new SpartanConfigFactoryServiceProviderImpl();
        try {
            SpartanNative.spartanInit();
            nativeEngineInitialized = true;
            System.out.println("\n" + "=".repeat(70));
            System.out.println("  CURIOSITY-DRIVEN RSAC MODEL TEST SUITE");
            System.out.println("=".repeat(70));
        } catch (Exception exception) {
            System.err.println("Failed to initialize native engine: " + exception.getMessage());
        }
    }

    @AfterAll
    static void cleanupResources() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("  CURIOSITY-DRIVEN RSAC TEST SUITE COMPLETED");
        System.out.println("=".repeat(70) + "\n");
    }

    private static synchronized long getNextAgentIdentifier() {
        return BASE_AGENT_IDENTIFIER + (++agentCounter * 0x10000);
    }

    @Test
    @Order(1)
    @DisplayName("1. Configuration and Curiosity Parameters Validation")
    void testConfigurationAndCuriosityParametersValidation() {
        System.out.println("\n-> Test 1: Configuration & Memory Layout");
        System.out.println("-".repeat(60));

        CuriosityDrivenRecurrentSoftActorCriticConfig configuration =
                CuriosityDrivenRecurrentSoftActorCriticConfig.builder()
                        .hiddenStateSize(HIDDEN_STATE_SIZE)
                        .forwardDynamicsHiddenLayerDimensionSize(FORWARD_DYNAMICS_HIDDEN_SIZE)
                        .intrinsicRewardScale(0.01)
                        .intrinsicRewardClampingMinimum(-1.0)
                        .intrinsicRewardClampingMaximum(1.0)
                        .forwardDynamicsLearningRate(3e-4)
                        .debugLogging(false)
                        .build();

        System.out.println("  Curiosity Module Parameters:");
        System.out.println("    - Forward Dynamics Hidden Size: " + configuration.forwardDynamicsHiddenLayerDimensionSize());
        System.out.println("    - Intrinsic Reward Scale: " + configuration.intrinsicRewardScale());
        System.out.println("    - Clamping Range: [" + configuration.intrinsicRewardClampingMinimum() +
                ", " + configuration.intrinsicRewardClampingMaximum() + "]");

        assertEquals(FORWARD_DYNAMICS_HIDDEN_SIZE, configuration.forwardDynamicsHiddenLayerDimensionSize());
        assertEquals(0.01, configuration.intrinsicRewardScale(), 1e-9);
        System.out.println("  [OK] Configuration validated successfully");
    }

    @Test
    @Order(2)
    @DisplayName("2. Weight Buffer Allocation with Forward Dynamics Network")
    void testWeightBufferAllocationWithForwardDynamicsNetwork() {
        System.out.println("\n-> Test 2: Weight Buffer Allocation");
        System.out.println("-".repeat(60));

        CuriosityDrivenRecurrentSoftActorCriticConfig configuration =
                CuriosityDrivenRecurrentSoftActorCriticConfig.builder()
                        .hiddenStateSize(HIDDEN_STATE_SIZE)
                        .forwardDynamicsHiddenLayerDimensionSize(FORWARD_DYNAMICS_HIDDEN_SIZE)
                        .recurrentInputFeatureCount(OBSERVATION_SIZE)
                        .build();

        long modelWeightCount = SpartanModelAllocator
                .calculateCuriosityDrivenRecurrentSoftActorCriticModelWeightCount(configuration, OBSERVATION_SIZE, CONTINUOUS_ACTION_SIZE);
        long criticWeightCount = SpartanModelAllocator
                .calculateCuriosityDrivenRecurrentSoftActorCriticCriticWeightCount(configuration, OBSERVATION_SIZE, CONTINUOUS_ACTION_SIZE);

        System.out.println("  Model Weights: " + modelWeightCount);
        System.out.println("  Critic Weights (includes Forward Dynamics): " + criticWeightCount);

        // Calculate expected Forward Dynamics weights
        int stateSize = OBSERVATION_SIZE;
        int actionSize = CONTINUOUS_ACTION_SIZE;
        int hiddenSize = FORWARD_DYNAMICS_HIDDEN_SIZE;
        long inputToHidden = (long) (stateSize + actionSize) * hiddenSize;
        long hiddenToOutput = (long) hiddenSize * stateSize;
        long biases = (long) hiddenSize + stateSize;
        long forwardDynamicsRawTotal = inputToHidden + hiddenToOutput + biases;
        long forwardDynamicsTotal = (forwardDynamicsRawTotal + 7) & ~7; // SIMD padding
        System.out.println("  Forward Dynamics Weights (calculated + padded): " + forwardDynamicsTotal);
        System.out.println("    - Input->Hidden: " + inputToHidden);
        System.out.println("    - Hidden->Output: " + hiddenToOutput);
        System.out.println("    - Biases: " + biases);

        assertTrue(modelWeightCount > 0);
        assertTrue(criticWeightCount > 0);
        System.out.println("  [OK] Weight allocation validation passed");
    }

    @Test
    @Order(3)
    @DisplayName("3. Full Lifecycle: Register -> 50 Ticks -> Unregister")
    void testFullLifecycleWithFiftyTicks() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 3: Full Lifecycle (50 Ticks)");
        System.out.println("-".repeat(60));

        try (Arena testArena = Arena.ofShared()) {
            long agentIdentifier = getNextAgentIdentifier();
            SpartanContext observationContext = createMockObservationContext(testArena);
            observationContext.update();

            CuriosityDrivenRecurrentSoftActorCriticConfig configuration =
                    CuriosityDrivenRecurrentSoftActorCriticConfig.builder()
                            .hiddenStateSize(HIDDEN_STATE_SIZE)
                            .recurrentInputFeatureCount(OBSERVATION_SIZE)
                            .forwardDynamicsHiddenLayerDimensionSize(FORWARD_DYNAMICS_HIDDEN_SIZE)
                            .actorHiddenLayerNeuronCount(HIDDEN_STATE_SIZE) // Explicitly set actor hidden size to match hidden state size
                            .debugLogging(false)
                            .isTraining(true)
                            .build();

            SpartanActionManager actionManager = createMockActionManager();
            CuriosityDrivenRecurrentSoftActorCriticModelImpl model =
                    new CuriosityDrivenRecurrentSoftActorCriticModelImpl(
                            "curiosity-rsac-test-model",
                            agentIdentifier,
                            configuration,
                            observationContext,
                            testArena,
                            actionManager
                    );

            try {
                // DEBUG: print buffer metadata before registering
                long debugModelWeightCount = SpartanModelAllocator.calculateCuriosityDrivenRecurrentSoftActorCriticModelWeightCount(configuration, OBSERVATION_SIZE, CONTINUOUS_ACTION_SIZE);
                long debugCriticWeightCount = SpartanModelAllocator.calculateCuriosityDrivenRecurrentSoftActorCriticCriticWeightCount(configuration, OBSERVATION_SIZE, CONTINUOUS_ACTION_SIZE);
                System.out.println("  [DEBUG] Model weights count (int): " + debugModelWeightCount);
                System.out.println("  [DEBUG] Critic weights count (int): " + debugCriticWeightCount);
                System.out.println("  [DEBUG] Action count: " + actionManager.getActions().size());
                System.out.println("  [DEBUG] Context size: " + observationContext.getSize());
                System.out.println("  [DEBUG] Model weights buffer address: " + model.getModelWeightsBuffer().address());
                System.out.println("  [DEBUG] Critic weights buffer address: " + model.getCriticWeightsBuffer().address());
                System.out.println("  [DEBUG] Action output buffer address: " + model.getActionOutputBuffer().address());

                System.out.println("  Registering model...");
                model.register();
                assertTrue(model.isRegistered());
                System.out.println("  [OK] Model registered");

                System.out.println("\n  Running " + SIMULATION_TICK_COUNT + " ticks...");
                Random randomGenerator = new Random(42);
                long totalExecutionTime = 0;
                long minTickTime = Long.MAX_VALUE;
                long maxTickTime = Long.MIN_VALUE;

                for (int tickIndex = 0; tickIndex < SIMULATION_TICK_COUNT; tickIndex++) {
                    double[] observationData = lastObservationElement.observationData;
                    for (int observationIndex = 0; observationIndex < OBSERVATION_SIZE; observationIndex++) {
                        observationData[observationIndex] = randomGenerator.nextDouble() * 2.0 - 1.0;
                    }
                    // CRITICAL: First tick (index 0) must have zero reward to bootstrap
                    // without attempting to compute gradients from invalid previous state.
                    double rewardSignal = (tickIndex == 0) ? 0.0 : (randomGenerator.nextDouble() > 0.5 ? 1.0 : -0.5);
                    long tickStartTime = System.nanoTime();
                    model.tick(rewardSignal);
                    long tickDuration = System.nanoTime() - tickStartTime;

                    totalExecutionTime += tickDuration;
                    if (tickDuration < minTickTime) minTickTime = tickDuration;
                    if (tickDuration > maxTickTime) maxTickTime = tickDuration;
                }

                System.out.printf("  Average tick time: %.2f us%n", totalExecutionTime / 1000.0 / SIMULATION_TICK_COUNT);
                System.out.printf("  Min tick time:     %.2f us%n", minTickTime / 1000.0);
                System.out.printf("  Max tick time:     %.2f us%n", maxTickTime / 1000.0);
                System.out.println("  [OK] " + SIMULATION_TICK_COUNT + " ticks completed");
            } finally {
                model.close();
                assertFalse(model.isRegistered());
                System.out.println("  [OK] Model unregistered");
            }
        }
    }

    @Test
    @Order(4)
    @DisplayName("4. Zero-Copy Memory Verification")
    void testZeroCopyMemoryVerification() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 4: Zero-Copy Memory Verification");
        System.out.println("-".repeat(60));

        try (Arena testArena = Arena.ofShared()) {
            long agentIdentifier = getNextAgentIdentifier();
            SpartanContext observationContext = createMockObservationContext(testArena);
            observationContext.update();

            CuriosityDrivenRecurrentSoftActorCriticConfig configuration =
                    CuriosityDrivenRecurrentSoftActorCriticConfig.builder()
                            .hiddenStateSize(HIDDEN_STATE_SIZE)
                            .recurrentInputFeatureCount(OBSERVATION_SIZE)
                            .forwardDynamicsHiddenLayerDimensionSize(FORWARD_DYNAMICS_HIDDEN_SIZE)
                            .actorHiddenLayerNeuronCount(HIDDEN_STATE_SIZE) // Explicitly set actor hidden size to match hidden state size
                            .debugLogging(false)
                            .isTraining(true)
                            .build();

            SpartanActionManager actionManager = createMockActionManager();
            try (CuriosityDrivenRecurrentSoftActorCriticModelImpl model = new CuriosityDrivenRecurrentSoftActorCriticModelImpl(
                    "curiosity-rsac-zerocopy",
                    agentIdentifier,
                    configuration,
                    observationContext,
                    testArena,
                    actionManager
            )) {
                // DEBUG: print buffer metadata
                long zerocopyModelWeightCount = SpartanModelAllocator.calculateCuriosityDrivenRecurrentSoftActorCriticModelWeightCount(configuration, OBSERVATION_SIZE, CONTINUOUS_ACTION_SIZE);
                long zerocopyCriticWeightCount = SpartanModelAllocator.calculateCuriosityDrivenRecurrentSoftActorCriticCriticWeightCount(configuration, OBSERVATION_SIZE, CONTINUOUS_ACTION_SIZE);
                System.out.println("  [DEBUG] (ZeroCopy) Model weights count: " + zerocopyModelWeightCount);
                System.out.println("  [DEBUG] (ZeroCopy) Critic weights count: " + zerocopyCriticWeightCount);
                System.out.println("  [DEBUG] (ZeroCopy) Action count: " + actionManager.getActions().size());
                System.out.println("  [DEBUG] (ZeroCopy) Context size: " + observationContext.getSize());
                System.out.println("  [DEBUG] (ZeroCopy) Model weights buffer addr: " + model.getModelWeightsBuffer().address());
                System.out.println("  [DEBUG] (ZeroCopy) Critic weights buffer addr: " + model.getCriticWeightsBuffer().address());
                System.out.println("  [DEBUG] (ZeroCopy) Action output buffer addr: " + model.getActionOutputBuffer().address());

                model.register();

                // Write test values
                double[] observationData = lastObservationElement.observationData;
                for (int observationIndex = 0; observationIndex < OBSERVATION_SIZE; observationIndex++) {
                    observationData[observationIndex] = observationIndex * 0.1;
                }

                // Tick 0: Bootstrap the environment. No previous action exists.
                model.tick(0.0);

                // Tick 1: Standard step with extrinsic reward.
                model.tick(1.0);

                // Read action outputs
                double[] actionOutputs = model.readAllActionValues();
                for (int actionIndex = 0; actionIndex < actionManager.getActions().size(); actionIndex++) {
                    assertFalse(Double.isNaN(actionOutputs[actionIndex]));
                    assertFalse(Double.isInfinite(actionOutputs[actionIndex]));
                }
            } finally {
                System.out.println("  [OK] Zero-Copy verification completed");
            }
        }
    }

    @Test
    @Order(5)
    @DisplayName("5. Episode Reset and Exploration Decay")
    void testEpisodeResetAndExplorationDecay() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 5: Episode Reset and Exploration Decay");
        System.out.println("-".repeat(60));

        try (Arena testArena = Arena.ofShared()) {
            long agentIdentifier = getNextAgentIdentifier();
            SpartanContext observationContext = createMockObservationContext(testArena);
            observationContext.update();

            CuriosityDrivenRecurrentSoftActorCriticConfig configuration =
                    CuriosityDrivenRecurrentSoftActorCriticConfig.builder()
                            .hiddenStateSize(HIDDEN_STATE_SIZE)
                            .recurrentInputFeatureCount(OBSERVATION_SIZE)
                            .forwardDynamicsHiddenLayerDimensionSize(FORWARD_DYNAMICS_HIDDEN_SIZE)
                            .actorHiddenLayerNeuronCount(HIDDEN_STATE_SIZE) // Explicitly set actor hidden size to match hidden state size
                            .debugLogging(false)
                            .isTraining(true)
                            .build();

            SpartanActionManager actionManager = createMockActionManager();
            try (CuriosityDrivenRecurrentSoftActorCriticModelImpl model = new CuriosityDrivenRecurrentSoftActorCriticModelImpl(
                    "curiosity-rsac-episode-reset",
                    agentIdentifier,
                    configuration,
                    observationContext,
                    testArena,
                    actionManager
            )) {
                model.register();

                model.tick(0.0);

                for (int tickIndex = 0; tickIndex < 15; tickIndex++) {
                    model.tick(2.0);
                }
                assertEquals(30.0, model.getEpisodeReward(), 0.001);
                System.out.println("  Episode reward after 15 ticks: " + model.getEpisodeReward());

                // Reset episode
                model.resetEpisode();
                assertEquals(0.0, model.getEpisodeReward(), 0.001);
                System.out.println("  Episode reward after reset: " + model.getEpisodeReward());

                // Decay exploration (delegates to internal RSAC agent)
                assertDoesNotThrow(model::decayExploration);
                System.out.println("  [OK] Exploration decay executed");
            } finally {
                System.out.println("  [OK] Episode reset and exploration decay test completed");
            }
        }
    }

    // ==================== Helper Methods ====================

    private SpartanContext createMockObservationContext(Arena arena) {
        SpartanContextImpl context = new SpartanContextImpl("curiosity-rsac-test-context", arena);
        lastObservationElement = new MockObservationElement();
        context.addElement(lastObservationElement, 0);
        return context;
    }

    private SpartanActionManager createMockActionManager() {
        MockActionManager manager = new MockActionManager();
        for (int i = 0; i < CONTINUOUS_ACTION_SIZE; i++) {
            manager.registerAction(new MockAction("mock_action_" + i));
        }
        return manager;
    }

    static class MockActionManager implements SpartanActionManager {
        private final java.util.List<SpartanAction> actions = new java.util.ArrayList<>();

        @Override
        public SpartanActionManager registerAction(SpartanAction action) {
            actions.add(action);
            return this;
        }

        @Override
        public java.util.List<SpartanAction> getActions() {
            return java.util.Collections.unmodifiableList(actions);
        }

        @Override
        public <SpartanActionType extends SpartanAction> java.util.List<SpartanActionType> getActionsByType(Class<SpartanActionType> actionClass) {
            java.util.List<SpartanActionType> matches = new java.util.ArrayList<>();
            for (SpartanAction action : actions) {
                if (actionClass.isInstance(action)) {
                    matches.add(actionClass.cast(action));
                }
            }
            return matches;
        }

        @Override
        public List<SpartanAction> getActionsByIdentifier(String identifier) {
            java.util.List<SpartanAction> matches = new java.util.ArrayList<>();
            for (SpartanAction action : actions) {
                if (identifier.equals(action.identifier())) {
                    matches.add(action);
                }
            }
            return matches;
        }
    }

    record MockAction(String identifier) implements SpartanAction {

        @Override
            public double taskMaxMagnitude() {
                return 1.0;
            }

            @Override
            public double taskMinMagnitude() {
                return -1.0;
            }

            @Override
            public void task(double normalizedMagnitude) {
            }

            @Override
            public double award() {
                return 0.0;
            }
        }

    static class MockObservationElement implements SpartanContextElement {
        private final double[] observationData = new double[OBSERVATION_SIZE];

        @Override
        public void tick() {}

        @Override
        public void prepare() {}

        @Override
        public double[] getData() {
            return observationData;
        }

        @Override
        public int getSize() {
            return OBSERVATION_SIZE;
        }

        @Override
        public String getIdentifier() {
            return "mock_curiosity_observation_element";
        }
    }
}
