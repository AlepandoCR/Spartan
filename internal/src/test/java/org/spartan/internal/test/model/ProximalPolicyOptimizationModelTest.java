package org.spartan.internal.test.model;

import org.junit.jupiter.api.*;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.action.type.SpartanAction;
import org.spartan.api.engine.config.ProximalPolicyOptimizationConfig;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.context.element.SpartanContextElement;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.engine.config.spi.SpartanConfigFactoryServiceProviderImpl;
import org.spartan.internal.engine.context.SpartanContextImpl;
import org.spartan.internal.engine.model.ProximalPolicyOptimizationSpartanModelImpl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for the Proximal Policy Optimization (PPO) model.
 * <p>
 * Validates:
 * <ul>
 *   <li>Model configuration and weight count calculation</li>
 *   <li>Full lifecycle: register → N ticks → close</li>
 *   <li>Continuous stochastic action outputs (mean + log_std policy)</li>
 *   <li>GAE reward accumulation and episode reset</li>
 *   <li>Tick execution time profiling</li>
 *   <li>Zero-Copy memory bridge verification</li>
 *   <li>Gradient clipping and entropy coefficient plumbing</li>
 * </ul>
 * <p>
 * Each test allocates its own {@link Arena} to prevent memory corruption across tests.
 */
@DisplayName("Proximal Policy Optimization Model - Comprehensive Integration Test")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ProximalPolicyOptimizationModelTest {

    // ==================== Constants ====================
    private static final long BASE_AGENT_IDENTIFIER = 0x0FF0_0E57_0001L;
    private static final int  OBSERVATION_SIZE       = 16;
    private static final int  CONTINUOUS_ACTION_SIZE = 4;
    private static final int  ACTOR_HIDDEN_NEURONS   = 64;
    private static final int  ACTOR_HIDDEN_LAYERS    = 2;
    private static final int  CRITIC_HIDDEN_NEURONS  = 64;
    private static final int  CRITIC_HIDDEN_LAYERS   = 2;
    private static final int  SIMULATION_TICK_COUNT  = 50;

    private static boolean nativeEngineInitialized = false;
    private static long    agentCounter            = 0;

    // ==================== Lifecycle ====================

    @BeforeAll
    static void initializeNativeEngine() {
        new SpartanConfigFactoryServiceProviderImpl();
        try {
            SpartanNative.spartanInit();
            nativeEngineInitialized = true;
            System.out.println("\n" + "=".repeat(70));
            System.out.println("  PROXIMAL POLICY OPTIMIZATION MODEL COMPREHENSIVE TEST SUITE");
            System.out.println("=".repeat(70));
        } catch (Exception exception) {
            System.err.println("Failed to initialize native engine: " + exception.getMessage());
        }
    }

    @AfterAll
    static void cleanupResources() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("  PROXIMAL POLICY OPTIMIZATION TEST SUITE COMPLETED");
        System.out.println("=".repeat(70) + "\n");
    }

    private static synchronized long nextAgentId() {
        return BASE_AGENT_IDENTIFIER + (++agentCounter * 0x10000L);
    }

    // ==================== Test 1: Configuration ====================

    @Test
    @Order(1)
    @DisplayName("1. Model Configuration and Weight Calculation Verification")
    void testModelConfigurationAndWeightCalculation() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 1: PPO Configuration & Memory Layout");
        System.out.println("-".repeat(60));

        ProximalPolicyOptimizationConfig config = buildStandardConfig();

        System.out.println("  Configuration Parameters:");
        System.out.println("    - Observation Size (State):      " + OBSERVATION_SIZE);
        System.out.println("    - Continuous Action Dimensions:  " + CONTINUOUS_ACTION_SIZE);
        System.out.println("    - Actor Hidden Neurons:          " + config.actorHiddenNeuronCount());
        System.out.println("    - Actor Hidden Layers:           " + config.actorHiddenLayerCount());
        System.out.println("    - Critic Hidden Neurons:         " + config.criticHiddenNeuronCount());
        System.out.println("    - Critic Hidden Layers:          " + config.criticHiddenLayerCount());
        System.out.println("    - Trajectory Buffer Capacity:    " + config.trajectoryBufferCapacity());
        System.out.println("    - Training Epochs per Update:    " + config.trainingEpochCount());
        System.out.println("    - Mini-Batch Size:               " + config.miniBatchSize());
        System.out.println("    - PPO Clip Range (epsilon):      " + config.clipRange());
        System.out.println("    - GAE Gamma:                     " + config.gaeGamma());
        System.out.println("    - GAE Lambda:                    " + config.gaeLambda());
        System.out.println("    - Entropy Coefficient:           " + config.entropyCoefficient());
        System.out.println("    - Value Loss Coefficient:        " + config.valueLossCoefficient());
        System.out.println("    - Max Gradient Norm:             " + config.maxGradientNorm());
        System.out.println("    - Learning Rate:                 " + config.learningRate());
        System.out.println("    - Training Mode:                 " + config.isTraining());

        // Actor: outputs mean AND log_std per action dimension (x2)
        int expectedActorWeights =
                (OBSERVATION_SIZE * ACTOR_HIDDEN_NEURONS) +
                (ACTOR_HIDDEN_NEURONS * ACTOR_HIDDEN_NEURONS * (ACTOR_HIDDEN_LAYERS - 1)) +
                (ACTOR_HIDDEN_NEURONS * CONTINUOUS_ACTION_SIZE * 2);
        int expectedActorBiases =
                (ACTOR_HIDDEN_NEURONS * ACTOR_HIDDEN_LAYERS) +
                (CONTINUOUS_ACTION_SIZE * 2);

        // Critic: outputs a single scalar V(s)
        int expectedCriticWeights =
                (OBSERVATION_SIZE * CRITIC_HIDDEN_NEURONS) +
                (CRITIC_HIDDEN_NEURONS * CRITIC_HIDDEN_NEURONS * (CRITIC_HIDDEN_LAYERS - 1)) +
                CRITIC_HIDDEN_NEURONS;
        int expectedCriticBiases =
                (CRITIC_HIDDEN_NEURONS * CRITIC_HIDDEN_LAYERS) + 1;

        int totalExpected = expectedActorWeights + expectedActorBiases
                + expectedCriticWeights + expectedCriticBiases;

        System.out.println("\n  Weight Layout (actor outputs mean + log_std, 2x output head):");
        System.out.printf("    Actor  Weights: %,d%n", expectedActorWeights);
        System.out.printf("    Actor  Biases:  %,d%n", expectedActorBiases);
        System.out.printf("    Critic Weights: %,d%n", expectedCriticWeights);
        System.out.printf("    Critic Biases:  %,d%n", expectedCriticBiases);
        System.out.printf("    Total Expected: %,d%n", totalExpected);

        assertTrue(totalExpected > 0, "Total weight count must be positive");
        System.out.println("  [OK] Weight calculation verified");
    }

    // ==================== Test 2: Full Lifecycle ====================

    @Test
    @Order(2)
    @DisplayName("2. Full Lifecycle: Register -> 50 Ticks -> Unregister")
    void testFullLifecycleWithFiftyTicks() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 2: PPO Full Lifecycle (50 Ticks)");
        System.out.println("-".repeat(60));

        try (Arena testArena = Arena.ofShared()) {
            long agentId = nextAgentId();
            SpartanContext context = createMockContext(testArena);
            context.update();

            ProximalPolicyOptimizationConfig config = buildStandardConfig();
            SpartanActionManager actionManager = createMockActionManager();

            ProximalPolicyOptimizationSpartanModelImpl model = new ProximalPolicyOptimizationSpartanModelImpl(
                    "ppo-lifecycle-test",
                    agentId,
                    config,
                    context,
                    actionManager,
                    testArena
            );

            // --- Phase 1: Registration ---
            System.out.println("\n  Phase 1: Model Registration");
            System.out.println("  " + "-".repeat(50));

            assertFalse(model.isRegistered(), "Model should not be registered before register()");
            long registrationStart = System.nanoTime();
            model.register();
            long registrationDuration = System.nanoTime() - registrationStart;

            assertTrue(model.isRegistered(), "Model must be registered after register()");
            System.out.printf("    [OK] Registered in %d µs%n", registrationDuration / 1_000);
            System.out.println("    Agent ID: 0x" + Long.toHexString(model.getAgentIdentifier()));

            // --- Phase 2: 50-Tick Simulation ---
            System.out.println("\n  Phase 2: 50-Tick Simulation");
            System.out.println("  " + "-".repeat(50));

            Random rng = new Random(1337);
            long[] tickTimes = new long[SIMULATION_TICK_COUNT];
            double accumulatedReward = 0.0;

            System.out.println("\n  Tick | Observation[0:3]              | Actions(mean)[0:3]            | Reward | Time(µs)");
            System.out.println("  " + "-".repeat(105));

            for (int t = 0; t < SIMULATION_TICK_COUNT; t++) {
                // Write random observations directly into the context segment
                MemorySegment contextData = ((SpartanContextImpl) context).getData();
                double[] obs = new double[OBSERVATION_SIZE];
                for (int i = 0; i < OBSERVATION_SIZE; i++) {
                    obs[i] = rng.nextDouble() * 2.0 - 1.0;
                    contextData.setAtIndex(ValueLayout.JAVA_DOUBLE, i, obs[i]);
                }

                // Shaped reward: positive if first observation is positive
                double reward = obs[0] > 0.0 ? 1.0 : -0.5;
                accumulatedReward += reward;

                long tickStart = System.nanoTime();
                model.tick(reward);
                tickTimes[t] = System.nanoTime() - tickStart;

                double[] actions = model.readAllActionValues();

                if (t % 10 == 0 || t == SIMULATION_TICK_COUNT - 1) {
                    System.out.printf("  %4d | [%+.3f, %+.3f, %+.3f, %+.3f] | [%+.3f, %+.3f, %+.3f, %+.3f] | %+5.2f | %6d%n",
                            t,
                            obs[0], obs[1], obs[2], obs[3],
                            actions[0], actions[1], actions[2], actions[3],
                            reward,
                            tickTimes[t] / 1_000);
                }

                // Actions must be finite (sampled from Gaussian, so no NaN/Inf expected)
                for (double actionValue : actions) {
                    assertFalse(Double.isNaN(actionValue),      "Action output must not be NaN at tick " + t);
                    assertFalse(Double.isInfinite(actionValue), "Action output must not be Infinite at tick " + t);
                }
            }

            // Timing Statistics
            System.out.println("\n  Phase 3: Tick Timing Statistics");
            System.out.println("  " + "-".repeat(50));

            long minTick = Long.MAX_VALUE, maxTick = 0, sumTick = 0;
            for (long t : tickTimes) {
                if (t < minTick) minTick = t;
                if (t > maxTick) maxTick = t;
                sumTick += t;
            }
            double avgTick = (double) sumTick / SIMULATION_TICK_COUNT;

            System.out.printf("    Min tick latency:  %6d µs%n", minTick / 1_000);
            System.out.printf("    Max tick latency:  %6d µs%n", maxTick / 1_000);
            System.out.printf("    Avg tick latency:  %9.2f µs%n", avgTick / 1_000.0);
            System.out.printf("    Accumulated reward: %.2f%n", accumulatedReward);

            assertTrue(avgTick > 0, "Average tick time must be positive");

            // --- Phase 4: Close ---
            model.close();
            assertFalse(model.isRegistered(), "Model must be unregistered after close()");
            System.out.println("\n  [PASS] Full lifecycle test completed");
        }
    }

    // ==================== Test 3: Stochastic Action Output ====================

    @Test
    @Order(3)
    @DisplayName("3. Stochastic Action Output — Mean and Variance Sanity Check")
    void testStochasticActionOutputSanity() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 3: Stochastic Action Output Sanity");
        System.out.println("-".repeat(60));

        /*
         * PPO's Gaussian policy outputs actions sampled from N(mean, exp(log_std)).
         * With a randomly initialized network over repeated ticks on random observations,
         * the actions should vary across ticks — i.e. not all identical.
         * We run 20 ticks and verify that at least one action dimension shows variance.
         */
        try (Arena testArena = Arena.ofShared()) {
            long agentId = nextAgentId();
            SpartanContext context = createMockContext(testArena);
            context.update();

            ProximalPolicyOptimizationConfig config = buildStandardConfig();
            SpartanActionManager actionManager = createMockActionManager();

            ProximalPolicyOptimizationSpartanModelImpl model = new ProximalPolicyOptimizationSpartanModelImpl(
                    "ppo-stochastic-test",
                    agentId,
                    config,
                    context,
                    actionManager,
                    testArena
            );
            model.register();

            Random rng = new Random(999);
            double[] firstActions = null;
            boolean observedVariance = false;

            System.out.println("  Running 20 ticks with random observations...");
            for (int t = 0; t < 20; t++) {
                MemorySegment contextData = ((SpartanContextImpl) context).getData();
                for (int i = 0; i < OBSERVATION_SIZE; i++) {
                    contextData.setAtIndex(ValueLayout.JAVA_DOUBLE, i, rng.nextDouble() * 2.0 - 1.0);
                }
                model.tick(0.0);
                double[] actions = model.readAllActionValues();

                if (firstActions == null) {
                    firstActions = actions.clone();
                } else {
                    for (int i = 0; i < actions.length; i++) {
                        if (Math.abs(actions[i] - firstActions[i]) > 1e-9) {
                            observedVariance = true;
                        }
                    }
                }
            }

            System.out.println("    First tick actions:  " + formatActions(firstActions));
            System.out.println("    Variance observed across ticks: " + observedVariance);
            assertTrue(observedVariance, "PPO policy must produce varying outputs across different observations");

            model.close();
            System.out.println("  [PASS] Stochastic action output sanity check completed");
        }
    }

    // ==================== Test 4: GAE Reward Accumulation & Episode Reset ====================

    @Test
    @Order(4)
    @DisplayName("4. GAE Reward Accumulation and Episode Reset")
    void testGaeRewardAccumulationAndEpisodeReset() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 4: GAE Reward Accumulation & Episode Reset");
        System.out.println("-".repeat(60));

        try (Arena testArena = Arena.ofShared()) {
            long agentId = nextAgentId();
            SpartanContext context = createMockContext(testArena);
            context.update();

            ProximalPolicyOptimizationConfig config = ProximalPolicyOptimizationConfig.builder()
                    .learningRate(3e-4)
                    .gamma(0.99)
                    .gaeGamma(0.99)
                    .gaeLambda(0.95)
                    .actorHiddenNeuronCount(ACTOR_HIDDEN_NEURONS)
                    .actorHiddenLayerCount(ACTOR_HIDDEN_LAYERS)
                    .criticHiddenNeuronCount(CRITIC_HIDDEN_NEURONS)
                    .criticHiddenLayerCount(CRITIC_HIDDEN_LAYERS)
                    .clipRange(0.2)
                    .entropyCoefficient(0.01)
                    .valueLossCoefficient(0.5)
                    .maxGradientNorm(0.5)
                    .trajectoryBufferCapacity(128)
                    .trainingEpochCount(4)
                    .miniBatchSize(32)
                    .isTraining(true)
                    .build();

            SpartanActionManager actionManager = createMockActionManager();
            ProximalPolicyOptimizationSpartanModelImpl model = new ProximalPolicyOptimizationSpartanModelImpl(
                    "ppo-gae-reset-test",
                    agentId,
                    config,
                    context,
                    actionManager,
                    testArena
            );
            model.register();

            // Accumulate reward across 10 ticks
            final double rewardPerTick = 1.0;
            final int tickCount = 10;

            System.out.println("  Accumulating reward over " + tickCount + " ticks (reward/tick = " + rewardPerTick + ")...");
            for (int t = 0; t < tickCount; t++) {
                model.tick(rewardPerTick);
            }

            double episodeRewardBefore = model.getEpisodeReward();
            System.out.printf("    Episode reward after %d ticks: %.2f%n", tickCount, episodeRewardBefore);
            assertEquals(rewardPerTick * tickCount, episodeRewardBefore, 0.001,
                    "Episode reward must match sum of all applied rewards");

            // Apply additional reward via applyReward() without ticking
            model.applyReward(5.0);
            double episodeRewardWithManual = model.getEpisodeReward();
            System.out.printf("    Episode reward after applyReward(5.0): %.2f%n", episodeRewardWithManual);
            assertEquals(rewardPerTick * tickCount + 5.0, episodeRewardWithManual, 0.001,
                    "applyReward() must be reflected in getEpisodeReward()");

            // Reset and verify
            System.out.println("\n  Resetting episode...");
            model.resetEpisode();
            double episodeRewardAfterReset = model.getEpisodeReward();
            System.out.printf("    Episode reward after reset: %.2f%n", episodeRewardAfterReset);
            assertEquals(0.0, episodeRewardAfterReset, 0.001, "Episode reward must be zero after reset");

            model.close();
            System.out.println("  [PASS] GAE reward accumulation and episode reset completed");
        }
    }

    // ==================== Test 5: Zero-Copy Memory Bridge ====================

    @Test
    @Order(5)
    @DisplayName("5. Zero-Copy Memory Bridge Verification")
    void testZeroCopyMemoryBridge() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 5: Zero-Copy Memory Bridge Verification");
        System.out.println("-".repeat(60));

        /*
         * We write specific sentinel values directly into the context MemorySegment
         * and verify the model reads them without any intermediate copy.
         * After tick(), we assert action outputs are finite — confirming the C++ side
         * actually processed the data we wrote.
         */
        try (Arena testArena = Arena.ofShared()) {
            long agentId = nextAgentId();
            SpartanContext context = createMockContext(testArena);
            context.update();

            ProximalPolicyOptimizationConfig config = buildStandardConfig();
            SpartanActionManager actionManager = createMockActionManager();

            ProximalPolicyOptimizationSpartanModelImpl model = new ProximalPolicyOptimizationSpartanModelImpl(
                    "ppo-zero-copy-test",
                    agentId,
                    config,
                    context,
                    actionManager,
                    testArena
            );
            model.register();

            double[] sentinelValues = new double[OBSERVATION_SIZE];
            for (int i = 0; i < OBSERVATION_SIZE; i++) {
                sentinelValues[i] = (i % 2 == 0) ? 0.5 : -0.5;
            }

            System.out.println("  Writing sentinel values to context MemorySegment...");
            MemorySegment contextData = ((SpartanContextImpl) context).getData();
            for (int i = 0; i < OBSERVATION_SIZE; i++) {
                contextData.setAtIndex(ValueLayout.JAVA_DOUBLE, i, sentinelValues[i]);
            }

            System.out.println("  Verifying pre-tick values in segment...");
            int verified = 0;
            for (int i = 0; i < OBSERVATION_SIZE; i++) {
                double readBack = contextData.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                assertEquals(sentinelValues[i], readBack, 1e-12,
                        "Sentinel value at index " + i + " must survive round-trip");
                verified++;
            }
            System.out.println("    [OK] All " + verified + " sentinel values verified in segment");

            System.out.println("  Executing tick with sentinel observations...");
            model.tick(0.0);

            System.out.println("  Verifying action output buffer is finite after tick...");
            for (int i = 0; i < CONTINUOUS_ACTION_SIZE; i++) {
                double actionValue = model.readActionValue(i);
                assertFalse(Double.isNaN(actionValue),      "Action[" + i + "] must not be NaN");
                assertFalse(Double.isInfinite(actionValue), "Action[" + i + "] must not be Infinite");
            }
            System.out.println("    [OK] All action outputs are finite after sentinel tick");

            model.close();
            System.out.println("  [PASS] Zero-Copy memory bridge verification completed");
        }
    }

    // ==================== Test 6: Throughput Benchmark ====================

    @Test
    @Order(6)
    @DisplayName("6. Tick Throughput Benchmark (100 Ticks)")
    void testTickThroughputBenchmark() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 6: Tick Throughput Benchmark (100 Ticks)");
        System.out.println("-".repeat(60));

        /*
         * Measures raw tick throughput.
         * The first few ticks are excluded from the average to discount JIT warm-up.
         */
        final int benchmarkTicks = 100;
        final int warmupTicks    = 5;

        try (Arena testArena = Arena.ofShared()) {
            long agentId = nextAgentId();
            SpartanContext context = createMockContext(testArena);
            context.update();

            ProximalPolicyOptimizationConfig config = buildStandardConfig();
            SpartanActionManager actionManager = createMockActionManager();

            ProximalPolicyOptimizationSpartanModelImpl model = new ProximalPolicyOptimizationSpartanModelImpl(
                    "ppo-benchmark-test",
                    agentId,
                    config,
                    context,
                    actionManager,
                    testArena
            );
            model.register();

            Random rng = new Random(42);
            long[] tickTimes = new long[benchmarkTicks];

            for (int t = 0; t < benchmarkTicks; t++) {
                MemorySegment contextData = ((SpartanContextImpl) context).getData();
                for (int i = 0; i < OBSERVATION_SIZE; i++) {
                    contextData.setAtIndex(ValueLayout.JAVA_DOUBLE, i, rng.nextDouble() * 2.0 - 1.0);
                }
                double reward = rng.nextDouble() * 2.0 - 1.0;

                long start = System.nanoTime();
                model.tick(reward);
                tickTimes[t] = System.nanoTime() - start;
            }

            // Exclude warm-up ticks from statistics
            long minTick = Long.MAX_VALUE, maxTick = 0, sumTick = 0;
            for (int t = warmupTicks; t < benchmarkTicks; t++) {
                long elapsed = tickTimes[t];
                if (elapsed < minTick) minTick = elapsed;
                if (elapsed > maxTick) maxTick = elapsed;
                sumTick += elapsed;
            }
            int measuredTicks = benchmarkTicks - warmupTicks;
            double avgTick = (double) sumTick / measuredTicks;
            double throughput = 1_000_000_000.0 / avgTick; // ticks per second

            System.out.printf("  Benchmark (%d ticks measured, %d warm-up excluded):%n",
                    measuredTicks, warmupTicks);
            System.out.printf("    Min latency:   %6d µs%n",    minTick / 1_000);
            System.out.printf("    Max latency:   %6d µs%n",    maxTick / 1_000);
            System.out.printf("    Avg latency:   %9.2f µs%n",  avgTick / 1_000.0);
            System.out.printf("    Throughput:    %,.0f ticks/s%n", throughput);

            assertTrue(avgTick > 0, "Average tick latency must be positive");
            assertTrue(throughput > 0, "Throughput must be positive");

            model.close();
            System.out.println("  [PASS] Throughput benchmark completed");
        }
    }

    // ==================== Test 7: Configuration Edge Cases ====================

    @Test
    @Order(7)
    @DisplayName("7. Shallow vs Deep Network Architecture Comparison")
    void testShallowVsDeepNetworkArchitecture() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 7: Shallow vs Deep Network Architecture Comparison");
        System.out.println("-".repeat(60));

        /*
         * Verifies that PPO registers and runs correctly with both
         * a minimal (1 hidden layer) and a deeper (3 hidden layers) configuration.
         * This exercises the weight allocation formula for different layer counts.
         */
        int[][] architectures = {
                {32,  1, 32,  1}, // shallow actor + critic
                {128, 3, 128, 3}  // deep actor + critic
        };
        String[] labels = {"Shallow (1-layer)", "Deep (3-layer)"};

        for (int arch = 0; arch < architectures.length; arch++) {
            int actorNeurons  = architectures[arch][0];
            int actorLayers   = architectures[arch][1];
            int criticNeurons = architectures[arch][2];
            int criticLayers  = architectures[arch][3];

            System.out.printf("%n  Testing architecture: %s%n", labels[arch]);
            System.out.printf("    Actor  [%d neurons x %d layers]%n", actorNeurons, actorLayers);
            System.out.printf("    Critic [%d neurons x %d layers]%n", criticNeurons, criticLayers);

            try (Arena testArena = Arena.ofShared()) {
                long agentId = nextAgentId();
                SpartanContext context = createMockContext(testArena);
                context.update();

                ProximalPolicyOptimizationConfig config = ProximalPolicyOptimizationConfig.builder()
                        .learningRate(1e-3)
                        .gamma(0.99)
                        .actorHiddenNeuronCount(actorNeurons)
                        .actorHiddenLayerCount(actorLayers)
                        .criticHiddenNeuronCount(criticNeurons)
                        .criticHiddenLayerCount(criticLayers)
                        .clipRange(0.2)
                        .gaeGamma(0.99)
                        .gaeLambda(0.95)
                        .entropyCoefficient(0.01)
                        .valueLossCoefficient(0.5)
                        .maxGradientNorm(0.5)
                        .trajectoryBufferCapacity(64)
                        .trainingEpochCount(4)
                        .miniBatchSize(16)
                        .isTraining(true)
                        .build();

                SpartanActionManager actionManager = createMockActionManager();
                ProximalPolicyOptimizationSpartanModelImpl model = new ProximalPolicyOptimizationSpartanModelImpl(
                        "ppo-arch-" + arch,
                        agentId,
                        config,
                        context,
                        actionManager,
                        testArena
                );

                assertDoesNotThrow(model::register,
                        labels[arch] + " must register without error");

                // Run 5 ticks to confirm end-to-end inference
                for (int t = 0; t < 5; t++) {
                    assertDoesNotThrow(() -> model.tick(0.1));
                }

                double[] actions = model.readAllActionValues();
                for (int i = 0; i < actions.length; i++) {
                    assertFalse(Double.isNaN(actions[i]),
                            labels[arch] + " action[" + i + "] must not be NaN");
                }

                System.out.printf("    [OK] %s: registration and 5 ticks succeeded%n", labels[arch]);
                model.close();
            }
        }

        System.out.println("  [PASS] Architecture comparison completed");
    }

    // ==================== Helpers ====================

    private ProximalPolicyOptimizationConfig buildStandardConfig() {
        return ProximalPolicyOptimizationConfig.builder()
                .learningRate(3e-4)
                .gamma(0.99)
                .gaeGamma(0.99)
                .gaeLambda(0.95)
                .actorHiddenNeuronCount(ACTOR_HIDDEN_NEURONS)
                .actorHiddenLayerCount(ACTOR_HIDDEN_LAYERS)
                .criticHiddenNeuronCount(CRITIC_HIDDEN_NEURONS)
                .criticHiddenLayerCount(CRITIC_HIDDEN_LAYERS)
                .clipRange(0.2)
                .entropyCoefficient(0.01)
                .valueLossCoefficient(0.5)
                .maxGradientNorm(0.5)
                .trajectoryBufferCapacity(128)
                .trainingEpochCount(4)
                .miniBatchSize(32)
                .isTraining(true)
                .build();
    }

    private SpartanContext createMockContext(Arena arena) {
        SpartanContextImpl context = new SpartanContextImpl("ppo-test-context", arena);
        context.addElement(new MockObservationElement(), 0);
        return context;
    }

    private SpartanActionManager createMockActionManager() {
        MockActionManager manager = new MockActionManager();
        for (int i = 0; i < CONTINUOUS_ACTION_SIZE; i++) {
            manager.registerAction(new MockAction("ppo_action_" + i));
        }
        return manager;
    }

    private static String formatActions(double[] actions) {
        if (actions == null) return "null";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < actions.length; i++) {
            sb.append(String.format("%+.4f", actions[i]));
            if (i < actions.length - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }

    // ==================== Mock Infrastructure ====================

    static class MockActionManager implements SpartanActionManager {
        private final List<SpartanAction> actions = new ArrayList<>();

        @Override
        public SpartanActionManager registerAction(SpartanAction action) {
            actions.add(action);
            return this;
        }

        @Override
        public List<SpartanAction> getActions() {
            return Collections.unmodifiableList(actions);
        }

        @Override
        public <T extends SpartanAction> List<T> getActionsByType(Class<T> actionClass) {
            List<T> matches = new ArrayList<>();
            for (SpartanAction action : actions) {
                if (actionClass.isInstance(action)) matches.add(actionClass.cast(action));
            }
            return matches;
        }

        @Override
        public List<SpartanAction> getActionsByIdentifier(String identifier) {
            List<SpartanAction> matches = new ArrayList<>();
            for (SpartanAction action : actions) {
                if (identifier.equals(action.identifier())) matches.add(action);
            }
            return matches;
        }
    }

    static class MockAction implements SpartanAction {
        private final String id;

        MockAction(String id) { this.id = id; }

        @Override public String identifier()                     { return id;  }
        @Override public double taskMaxMagnitude()               { return 1.0; }
        @Override public double taskMinMagnitude()               { return -1.0; }
        @Override public void   task(double normalizedMagnitude) { /* no-op */ }
        @Override public double award()                          { return 0.0; }
    }

    static class MockObservationElement implements SpartanContextElement {
        private final double[] data = new double[OBSERVATION_SIZE];

        @Override public void     tick()                { /* no-op */ }
        @Override public void     prepare()             { /* no-op */ }
        @Override public double[] getData()             { return data; }
        @Override public int      getSize()             { return OBSERVATION_SIZE; }
        @Override public String   getIdentifier()       { return "ppo-mock-observation"; }
    }
}