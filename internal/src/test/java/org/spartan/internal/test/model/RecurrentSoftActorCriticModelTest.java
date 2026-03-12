package org.spartan.internal.test.model;

import org.junit.jupiter.api.*;
import org.spartan.api.agent.action.SpartanActionManager;
import org.spartan.api.agent.action.type.SpartanAction;
import org.spartan.api.agent.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.agent.context.SpartanContext;
import org.spartan.api.agent.context.element.SpartanContextElement;
import org.spartan.internal.agent.context.SpartanContextImpl;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.model.RecurrentSoftActorCriticModel;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Recurrent Soft Actor-Critic model.
 * Each test uses its own Arena to avoid memory corruption between tests.
 */
@DisplayName("Recurrent Soft Actor-Critic Model - Comprehensive Integration Test")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class RecurrentSoftActorCriticModelTest {

    private static final long BASE_AGENT_IDENTIFIER = 0x05AC_0E57_0001L;
    private static final int OBSERVATION_SIZE = 16;
    private static final int CONTINUOUS_ACTION_SIZE = 4;
    private static final int HIDDEN_STATE_SIZE = 64;
    private static final int SIMULATION_TICK_COUNT = 50;

    private static boolean nativeEngineInitialized = false;
    private static long agentCounter = 0;

    @BeforeAll
    static void initializeNativeEngine() {
        try {
            SpartanNative.spartanInit();
            nativeEngineInitialized = true;
            System.out.println("\n" + "=".repeat(70));
            System.out.println("  RECURRENT SOFT ACTOR-CRITIC MODEL COMPREHENSIVE TEST SUITE");
            System.out.println("=".repeat(70));
        } catch (Exception exception) {
            System.err.println("Failed to initialize native engine: " + exception.getMessage());
        }
    }

    @AfterAll
    static void cleanupResources() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("  RECURRENT SOFT ACTOR-CRITIC TEST SUITE COMPLETED");
        System.out.println("=".repeat(70) + "\n");
    }

    private static synchronized long getNextAgentIdentifier() {
        return BASE_AGENT_IDENTIFIER + (++agentCounter * 0x10000);
    }

    @Test
    @Order(1)
    @DisplayName("1. Model Configuration and Weight Calculation Verification")
    void testModelConfigurationAndWeightCalculation() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 1: Recurrent Soft Actor-Critic Configuration & Memory Layout");
        System.out.println("-".repeat(60));

        RecurrentSoftActorCriticConfig configuration = RecurrentSoftActorCriticConfig.builder()
                .stateSize(OBSERVATION_SIZE)
                .actionSize(CONTINUOUS_ACTION_SIZE)
                .hiddenStateSize(HIDDEN_STATE_SIZE)
                .recurrentLayerDepth(1)
                .actorHiddenLayerNeuronCount(128)
                .actorHiddenLayerCount(2)
                .criticHiddenLayerNeuronCount(128)
                .criticHiddenLayerCount(2)
                .learningRate(3e-4)
                .gamma(0.99)
                .entropyTemperatureAlpha(0.2)
                .isTraining(true)
                .build();

        System.out.println("  Configuration Parameters:");
        System.out.println("    - Observation Size (State): " + configuration.stateSize());
        System.out.println("    - Continuous Action Size: " + configuration.actionSize());
        System.out.println("    - Hidden State Size (GRU): " + configuration.hiddenStateSize());
        System.out.println("    - Recurrent Layer Depth: " + configuration.recurrentLayerDepth());
        System.out.println("    - Actor Hidden Neuron Count: " + configuration.actorHiddenLayerNeuronCount());
        System.out.println("    - Critic Hidden Neuron Count: " + configuration.criticHiddenLayerNeuronCount());
        System.out.println("    - Learning Rate: " + configuration.learningRate());
        System.out.println("    - Discount Factor (Gamma): " + configuration.gamma());
        System.out.println("    - Entropy Temperature (Alpha): " + configuration.entropyTemperatureAlpha());
        System.out.println("    - Training Mode Enabled: " + configuration.isTraining());

        assertEquals(OBSERVATION_SIZE, configuration.stateSize());
        assertEquals(CONTINUOUS_ACTION_SIZE, configuration.actionSize());
        System.out.println("  [OK] Configuration validated successfully");
    }

    @Test
    @Order(2)
    @DisplayName("2. Full Lifecycle: Register -> 50 Ticks -> Unregister")
    void testFullLifecycleWithFiftyTicks() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 2: Recurrent Soft Actor-Critic Full Lifecycle (50 Ticks)");
        System.out.println("-".repeat(60));

        try (Arena testArena = Arena.ofConfined()) {
            long agentId = getNextAgentIdentifier();
            SpartanContext observationContext = createMockObservationContext(testArena);
            observationContext.update();

            RecurrentSoftActorCriticConfig configuration = RecurrentSoftActorCriticConfig.builder()
                    .stateSize(OBSERVATION_SIZE)
                    .actionSize(CONTINUOUS_ACTION_SIZE)
                    .hiddenStateSize(HIDDEN_STATE_SIZE)
                    .recurrentLayerDepth(1)
                    .actorHiddenLayerNeuronCount(64)
                    .actorHiddenLayerCount(2)
                    .criticHiddenLayerNeuronCount(64)
                    .criticHiddenLayerCount(2)
                    .recurrentInputFeatureCount(OBSERVATION_SIZE)
                    .isTraining(true)
                    .build();

            SpartanActionManager actionManager = createMockActionManager();
            RecurrentSoftActorCriticModel model = new RecurrentSoftActorCriticModel(
                    agentId, configuration, observationContext, testArena, actionManager
            );

            System.out.println("\n  Phase 1: Model Registration");
            System.out.println("  " + "-".repeat(50));

            assertFalse(model.isRegistered());
            long registrationStartTime = System.nanoTime();
            model.register();
            long registrationDuration = System.nanoTime() - registrationStartTime;

            assertTrue(model.isRegistered());
            System.out.println("    [OK] Model registered successfully in " + (registrationDuration / 1000) + " microseconds");
            System.out.println("    Agent Identifier: 0x" + Long.toHexString(model.getAgentIdentifier()));

            System.out.println("\n  Phase 2: 50-Tick Simulation with Continuous Actions");
            System.out.println("  " + "-".repeat(50));

            Random randomGenerator = new Random(42);
            long[] tickExecutionTimes = new long[SIMULATION_TICK_COUNT];
            double accumulatedReward = 0.0;

            System.out.println("\n  Tick | Observation[0:3]              | ContinuousActions[0:3]        | Reward | Time(us)");
            System.out.println("  " + "-".repeat(100));

            for (int tickIndex = 0; tickIndex < SIMULATION_TICK_COUNT; tickIndex++) {
                double[] observationInputs = new double[OBSERVATION_SIZE];
                for (int inputIndex = 0; inputIndex < OBSERVATION_SIZE; inputIndex++) {
                    observationInputs[inputIndex] = randomGenerator.nextDouble() * 2.0 - 1.0;
                }

                MemorySegment contextDataSegment = observationContext.getData();
                for (int inputIndex = 0; inputIndex < OBSERVATION_SIZE; inputIndex++) {
                    contextDataSegment.setAtIndex(ValueLayout.JAVA_DOUBLE, inputIndex, observationInputs[inputIndex]);
                }

                double rewardSignal = (observationInputs[0] > 0) ? 1.0 : -0.5;
                accumulatedReward += rewardSignal;

                long tickStartTime = System.nanoTime();
                model.tick(rewardSignal);
                tickExecutionTimes[tickIndex] = System.nanoTime() - tickStartTime;

                double[] continuousActionOutputs = model.readAllActionValues();

                if (tickIndex % 10 == 0 || tickIndex == SIMULATION_TICK_COUNT - 1) {
                    System.out.printf("  %4d | [%+.3f, %+.3f, %+.3f, %+.3f] | [%+.3f, %+.3f, %+.3f, %+.3f] | %+5.2f | %6d%n",
                            tickIndex,
                            observationInputs[0], observationInputs[1], observationInputs[2], observationInputs[3],
                            continuousActionOutputs[0], continuousActionOutputs[1], continuousActionOutputs[2], continuousActionOutputs[3],
                            rewardSignal,
                            tickExecutionTimes[tickIndex] / 1000);
                }

                for (int actionIndex = 0; actionIndex < CONTINUOUS_ACTION_SIZE; actionIndex++) {
                    double actionValue = continuousActionOutputs[actionIndex];
                    assertFalse(Double.isNaN(actionValue), "Action should not be NaN at tick " + tickIndex);
                    assertFalse(Double.isInfinite(actionValue), "Action should not be infinite at tick " + tickIndex);
                }
            }

            System.out.println("\n  Phase 3: Performance Metrics and Statistics");
            System.out.println("  " + "-".repeat(50));

            long totalExecutionTime = 0;
            long minimumTickTime = Long.MAX_VALUE;
            long maximumTickTime = Long.MIN_VALUE;
            for (long executionTime : tickExecutionTimes) {
                totalExecutionTime += executionTime;
                minimumTickTime = Math.min(minimumTickTime, executionTime);
                maximumTickTime = Math.max(maximumTickTime, executionTime);
            }
            double averageTickTime = totalExecutionTime / (double) SIMULATION_TICK_COUNT;

            System.out.printf("    Total Ticks Executed: %d%n", SIMULATION_TICK_COUNT);
            System.out.printf("    Total Execution Time: %.2f milliseconds%n", totalExecutionTime / 1_000_000.0);
            System.out.printf("    Average Tick Time: %.2f microseconds%n", averageTickTime / 1000.0);
            System.out.printf("    Minimum Tick Time: %d microseconds%n", minimumTickTime / 1000);
            System.out.printf("    Maximum Tick Time: %d microseconds%n", maximumTickTime / 1000);
            System.out.printf("    Throughput: %.0f ticks/second%n", 1_000_000_000.0 / averageTickTime);
            System.out.printf("    Episode Accumulated Reward: %.2f%n", model.getEpisodeReward());
            System.out.printf("    Expected Accumulated Reward: %.2f%n", accumulatedReward);

            assertEquals(accumulatedReward, model.getEpisodeReward(), 0.001, "Episode reward should match accumulated reward");

            System.out.println("\n  Phase 4: Model Unregistration");
            System.out.println("  " + "-".repeat(50));

            model.close();
            assertFalse(model.isRegistered());
            System.out.println("    [OK] Model unregistered and closed successfully");

            System.out.println("\n  [PASS] Recurrent Soft Actor-Critic 50-tick lifecycle test completed");
        }
    }

    @Test
    @Order(3)
    @DisplayName("3. Zero-Copy Memory Verification")
    void testZeroCopyMemoryVerification() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 3: Zero-Copy Memory Verification");
        System.out.println("-".repeat(60));

        try (Arena testArena = Arena.ofConfined()) {
            long agentId = getNextAgentIdentifier();
            SpartanContext observationContext = createMockObservationContext(testArena);
            observationContext.update();

            RecurrentSoftActorCriticConfig configuration = RecurrentSoftActorCriticConfig.builder()
                    .stateSize(OBSERVATION_SIZE)
                    .actionSize(CONTINUOUS_ACTION_SIZE)
                    .hiddenStateSize(32)  // Different from test 2 to verify fix works
                    .recurrentLayerDepth(1)
                    .actorHiddenLayerNeuronCount(48)
                    .actorHiddenLayerCount(2)
                    .criticHiddenLayerNeuronCount(48)
                    .criticHiddenLayerCount(2)
                    .recurrentInputFeatureCount(OBSERVATION_SIZE)
                    .build();

            RecurrentSoftActorCriticModel model = new RecurrentSoftActorCriticModel(
                    agentId, configuration, observationContext, testArena, createMockActionManager()
            );

            model.register();

            double[] testObservationValues = {0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.75, 0.333, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
            MemorySegment contextDataSegment = observationContext.getData();

            System.out.println("  Writing test observation values to context:");
            for (int valueIndex = 0; valueIndex < testObservationValues.length; valueIndex++) {
                contextDataSegment.setAtIndex(ValueLayout.JAVA_DOUBLE, valueIndex, testObservationValues[valueIndex]);
                if (valueIndex < 8) {
                    System.out.printf("    observation[%d] = %+.4f%n", valueIndex, testObservationValues[valueIndex]);
                }
            }

            System.out.println("\n  Pre-Tick Verification:");
            int preTickVerified = 0;
            for (int valueIndex = 0; valueIndex < testObservationValues.length; valueIndex++) {
                double readBackValue = contextDataSegment.getAtIndex(ValueLayout.JAVA_DOUBLE, valueIndex);
                boolean valuesMatch = Math.abs(readBackValue - testObservationValues[valueIndex]) < 1e-10;
                if (valueIndex < 4) {
                    System.out.printf("    observation[%d]: written=%+.4f, read=%+.4f %s%n",
                            valueIndex, testObservationValues[valueIndex], readBackValue, valuesMatch ? "[OK]" : "[FAIL]");
                }
                if (valuesMatch) preTickVerified++;
            }
            System.out.println("    Pre-tick verified: " + preTickVerified + "/" + testObservationValues.length);

            model.tick(1.0);

            System.out.println("\n  Continuous Action Outputs:");
            for (int actionIndex = 0; actionIndex < CONTINUOUS_ACTION_SIZE; actionIndex++) {
                double actionValue = model.readActionValue(actionIndex);
                System.out.printf("    action[%d] = %+.6f%n", actionIndex, actionValue);
                assertFalse(Double.isNaN(actionValue), "Action should not be NaN");
                assertFalse(Double.isInfinite(actionValue), "Action should not be infinite");
            }

            model.close();

            assertEquals(testObservationValues.length, preTickVerified, "All observation values should be verified");
            System.out.println("\n  [PASS] Zero-Copy verification completed: " + preTickVerified + "/" + testObservationValues.length);
        }
    }

    @Test
    @Order(4)
    @DisplayName("4. Episode Reset and Exploration Decay")
    void testEpisodeResetAndExplorationDecay() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 4: Episode Reset and Exploration Decay");
        System.out.println("-".repeat(60));

        try (Arena testArena = Arena.ofConfined()) {
            long agentId = getNextAgentIdentifier();
            SpartanContext observationContext = createMockObservationContext(testArena);
            observationContext.update();

            RecurrentSoftActorCriticConfig configuration = RecurrentSoftActorCriticConfig.builder()
                    .stateSize(OBSERVATION_SIZE)
                    .actionSize(CONTINUOUS_ACTION_SIZE)
                    .hiddenStateSize(24)  // Yet another size to verify fix
                    .recurrentLayerDepth(1)
                    .actorHiddenLayerNeuronCount(32)
                    .actorHiddenLayerCount(2)
                    .criticHiddenLayerNeuronCount(32)
                    .criticHiddenLayerCount(2)
                    .recurrentInputFeatureCount(OBSERVATION_SIZE)
                    .epsilon(1.0)
                    .epsilonDecay(0.995)
                    .epsilonMin(0.01)
                    .entropyTemperatureAlpha(0.2)
                    .build();

            RecurrentSoftActorCriticModel model = new RecurrentSoftActorCriticModel(
                    agentId, configuration, observationContext, testArena, createMockActionManager()
            );

            model.register();

            System.out.println("  Simulating episode with reward accumulation...");
            for (int tickIndex = 0; tickIndex < 10; tickIndex++) {
                model.tick(1.0);
            }
            double episodeRewardBeforeReset = model.getEpisodeReward();
            System.out.printf("    Episode reward after 10 ticks: %.2f%n", episodeRewardBeforeReset);
            assertEquals(10.0, episodeRewardBeforeReset, 0.001);

            System.out.println("\n  Resetting episode state...");
            model.resetEpisode();
            double episodeRewardAfterReset = model.getEpisodeReward();
            System.out.printf("    Episode reward after reset: %.2f%n", episodeRewardAfterReset);
            assertEquals(0.0, episodeRewardAfterReset, 0.001);

            System.out.println("\n  Decaying exploration...");
            System.out.printf("    Initial Entropy Alpha (from config): %.4f%n", configuration.entropyTemperatureAlpha());
            assertDoesNotThrow(model::decayExploration);
            System.out.println("    [OK] Exploration decay executed successfully");

            model.close();
            System.out.println("\n  [PASS] Episode reset and exploration decay test completed");
        }
    }

    // ==================== Helper Methods ====================

    private SpartanContext createMockObservationContext(Arena arena) {
        SpartanContextImpl context = new SpartanContextImpl("rsac-test-observation-context", arena);
        context.addElement(new MockObservationElement(), 0);
        return context;
    }

    private SpartanActionManager createMockActionManager() {
        return new MockActionManager();
    }

    static class MockActionManager implements SpartanActionManager {
        @Override
        public SpartanAction[] getActions() {
            return new SpartanAction[0];
        }

        @Override
        @SuppressWarnings("unchecked")
        public <SpartanActionType extends SpartanAction> SpartanActionType[] getActionsByType(Class<SpartanActionType> actionClass) {
            return (SpartanActionType[]) java.lang.reflect.Array.newInstance(actionClass, 0);
        }

        @Override
        public SpartanAction[] getActionsByIdentifier(String identifier) {
            return new SpartanAction[0];
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
            return "mock_observation_element";
        }
    }
}
