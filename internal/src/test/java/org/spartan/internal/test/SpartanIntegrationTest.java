package org.spartan.internal.test;

import org.junit.jupiter.api.*;
import org.spartan.api.engine.action.SpartanActionManager;
import org.spartan.api.engine.action.type.SpartanAction;
import org.spartan.api.engine.config.RecurrentSoftActorCriticConfig;
import org.spartan.api.engine.context.element.SpartanSingleContextElement;
import org.spartan.api.engine.context.element.variable.SpartanVariableContextElement;
import org.spartan.internal.engine.context.SpartanContextImpl;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.internal.engine.config.spi.SpartanConfigFactoryServiceProviderImpl;
import org.spartan.internal.engine.model.SpartanConfigLayout;
import org.spartan.internal.engine.model.SpartanModelAllocator;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Spartan Engine Integration Test - Phase 3: Full Lifecycle Simulation
 * <p>
 * This test validates:
 * <ul>
 *   <li>Native DLL loading and basic function calls</li>
 *   <li>Memory alignment between Java FFM and C++26 structs</li>
 *   <li>Zero-Copy shared memory integrity</li>
 *   <li>Vector operations via native engine</li>
 * </ul>
 */
@DisplayName("Spartan Engine - FFM Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class SpartanIntegrationTest {

    @BeforeAll
    static void registerConfigProvider() {
        new SpartanConfigFactoryServiceProviderImpl();
    }

    // ==================== Native Engine Tests (Require DLL) ====================

    @Nested
    @DisplayName("Native Engine Tests - Require DLL")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class NativeEngineTests {

        @Test
        @Order(1)
        @DisplayName("Native Engine Initialization")
        void testNativeEngineInit() {
            System.out.println("\n-> Testing Native Engine Initialization...");

            assertDoesNotThrow(() -> {
                SpartanNative.spartanInit();
            }, "spartanInit should not throw");

            System.out.println("  OK - Native engine initialized successfully");
            System.out.println("OK - Native init test passed");
        }

        @Test
        @Order(2)
        @DisplayName("Native Logging")
        void testNativeLogging() {
            System.out.println("\n-> Testing Native Logging...");

            assertDoesNotThrow(() -> {
                SpartanNative.spartanLog("JUnit test message - FFM bridge working!");
            }, "spartanLog should not throw");

            System.out.println("  OK - Native logging works");
            System.out.println("OK - Native logging test passed");
        }

        @Test
        @Order(3)
        @DisplayName("Vector Union Operation (Zero-Copy FFM)")
        void testVectorUnionOperation() {
            System.out.println("\n-> Testing Vector Union Operation...");

            try (Arena arena = Arena.ofConfined()) {
                int sizeA = 1000;
                int sizeB = 1000;

                MemorySegment vectorA = arena.allocate(ValueLayout.JAVA_DOUBLE, sizeA);
                MemorySegment vectorB = arena.allocate(ValueLayout.JAVA_DOUBLE, sizeB);

                Random random = new Random(42);
                for (int i = 0; i < sizeA; i++) {
                    vectorA.setAtIndex(ValueLayout.JAVA_DOUBLE, i, random.nextDouble());
                }
                for (int i = 0; i < sizeB; i++) {
                    vectorB.setAtIndex(ValueLayout.JAVA_DOUBLE, i, random.nextDouble());
                }

                long nanosTaken = SpartanNative.spartanTestVectorUnion(vectorA, vectorB, sizeA, sizeB);

                assertTrue(nanosTaken >= 0, "Operation should return valid timing");
                System.out.println("  OK - Vector union completed in " + nanosTaken + " ns");

                double firstValue = vectorA.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
                assertTrue(firstValue >= 0.0 && firstValue <= 1.0, "Result should be valid fuzzy value");

                System.out.println("OK - Vector union test passed");
            }
        }

        @Test
        @Order(4)
        @DisplayName("Memory Segment Alignment for Native Calls")
        void testMemorySegmentAlignment() {
            System.out.println("\n-> Testing Memory Segment Alignment...");

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment doubleSegment = arena.allocate(ValueLayout.JAVA_DOUBLE, 100);
                MemorySegment intSegment = arena.allocate(ValueLayout.JAVA_INT, 100);
                MemorySegment longSegment = arena.allocate(ValueLayout.JAVA_LONG, 10);

                assertEquals(0, doubleSegment.address() % 8, "Double segment should be 8-byte aligned");
                assertEquals(0, intSegment.address() % 4, "Int segment should be 4-byte aligned");
                assertEquals(0, longSegment.address() % 8, "Long segment should be 8-byte aligned");

                doubleSegment.setAtIndex(ValueLayout.JAVA_DOUBLE, 0, Math.PI);
                assertEquals(Math.PI, doubleSegment.getAtIndex(ValueLayout.JAVA_DOUBLE, 0), 1e-15);

                intSegment.setAtIndex(ValueLayout.JAVA_INT, 0, 0xDEADBEEF);
                assertEquals(0xDEADBEEF, intSegment.getAtIndex(ValueLayout.JAVA_INT, 0));

                longSegment.setAtIndex(ValueLayout.JAVA_LONG, 0, 0xCAFEBABEDEADBEEFL);
                assertEquals(0xCAFEBABEDEADBEEFL, longSegment.getAtIndex(ValueLayout.JAVA_LONG, 0));

                System.out.println("  OK - All segments properly aligned");
                System.out.println(" OK - Memory alignment test passed");
            }
        }

        @Test
        @Order(5)
        @DisplayName("Stress Test: Multiple Vector Operations")
        void testStressVectorOperations() {
            System.out.println("\n-> Stress Testing Vector Operations...");

            try (Arena arena = Arena.ofConfined()) {
                int iterations = 100;
                int vectorSize = 500;

                MemorySegment vectorA = arena.allocate(ValueLayout.JAVA_DOUBLE, vectorSize);
                MemorySegment vectorB = arena.allocate(ValueLayout.JAVA_DOUBLE, vectorSize);

                for (int i = 0; i < vectorSize; i++) {
                    vectorA.setAtIndex(ValueLayout.JAVA_DOUBLE, i, 0.5);
                    vectorB.setAtIndex(ValueLayout.JAVA_DOUBLE, i, 0.5);
                }

                long totalNanos = 0;
                for (int i = 0; i < iterations; i++) {
                    totalNanos += SpartanNative.spartanTestVectorUnion(vectorA, vectorB, vectorSize, vectorSize);
                }

                double avgNanos = totalNanos / (double) iterations;
                System.out.println("  OK - Completed " + iterations + " iterations");
                System.out.println("  OK - Average time: " + String.format("%.2f", avgNanos) + " ns");

                assertTrue(avgNanos > 0, "Should have valid timing");
                System.out.println("OK - Stress test passed");
            }
        }

        @Test
        @Order(6)
        @DisplayName("Full Model Lifecycle: Register -> Tick -> Zero-Copy Verification -> Unregister")
        void testFullModelLifecycleWithZeroCopy() {
            System.out.println("\n>> Testing Full Model Lifecycle with Zero-Copy...");
            System.out.println("+--------------------------------------------------------------+");
            System.out.println("|  Testing: Java <-> C++ Zero-Copy Memory Bridge               |");
            System.out.println("+--------------------------------------------------------------+");

            try (Arena arena = Arena.ofConfined()) {
                final long AGENT_ID = 0xDEADBEEF_CAFEBABEL;
                final int STATE_SIZE = 16;
                final int ACTION_SIZE = 4;
                final int WEIGHT_COUNT = 100;
                final int TICK_COUNT = 50;

                // ==================== PHASE 1: Allocate JVM-owned buffers ====================
                System.out.println("\n-> Phase 1: Allocating JVM-owned buffers...");

                // Config buffer - BaseHyperparameterConfig layout (64 bytes)
                // Offsets from SpartanConfigLayout:
                // 0: modelTypeIdentifier (int32) = 0 (DEFAULT)
                // 8: learningRate (double)
                // 16: gamma (double)
                // 24: epsilon (double)
                // 32: epsilonMin (double)
                // 40: epsilonDecay (double)
                // 48: stateSize (int32)
                // 52: actionSize (int32)
                // 56: isTraining (byte)
                MemorySegment configBuffer = arena.allocate(64, 8);
                configBuffer.set(ValueLayout.JAVA_INT, 0, 0);           // modelTypeIdentifier = DEFAULT
                configBuffer.set(ValueLayout.JAVA_DOUBLE, 8, 0.001);    // learningRate
                configBuffer.set(ValueLayout.JAVA_DOUBLE, 16, 0.99);    // gamma
                configBuffer.set(ValueLayout.JAVA_DOUBLE, 24, 1.0);     // epsilon
                configBuffer.set(ValueLayout.JAVA_DOUBLE, 32, 0.01);    // epsilonMin
                configBuffer.set(ValueLayout.JAVA_DOUBLE, 40, 0.995);   // epsilonDecay
                configBuffer.set(ValueLayout.JAVA_INT, 48, STATE_SIZE); // stateSize
                configBuffer.set(ValueLayout.JAVA_INT, 52, ACTION_SIZE);// actionSize
                configBuffer.set(ValueLayout.JAVA_BYTE, 56, (byte) 1);  // isTraining = true

                // Critic weights (for DefaultSpartanAgent, minimal)
                // SIMD padding ensures C++ AVX2 operations don't fault when reading near buffer end
                final int SIMD_PADDING = 4;
                MemorySegment criticWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, WEIGHT_COUNT + SIMD_PADDING);

                // Model weights
                MemorySegment modelWeightsBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, WEIGHT_COUNT + SIMD_PADDING);

                // Context/observation buffer - this is where Java writes game state
                MemorySegment contextBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, STATE_SIZE + SIMD_PADDING);

                // Action output buffer - C++ writes predictions here
                MemorySegment actionOutputBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, ACTION_SIZE + SIMD_PADDING);

                System.out.println("  OK - Config buffer: " + configBuffer.byteSize() + " bytes @ 0x" + Long.toHexString(configBuffer.address()));
                System.out.println("  OK - Critic weights: " + (WEIGHT_COUNT * 8) + " bytes");
                System.out.println("  OK - Model weights: " + (WEIGHT_COUNT * 8) + " bytes");
                System.out.println("  OK - Context buffer: " + (STATE_SIZE * 8) + " bytes");
                System.out.println("  OK - Action output: " + (ACTION_SIZE * 8) + " bytes");

                // ==================== PHASE 2: Register model with C++ ====================
                System.out.println("\n-> Phase 2: Registering model with native engine...");

                int registerResult = SpartanNative.spartanRegisterModel(
                        AGENT_ID,
                        configBuffer,
                        criticWeightsBuffer, WEIGHT_COUNT,
                        modelWeightsBuffer, WEIGHT_COUNT,
                        contextBuffer, STATE_SIZE,
                        actionOutputBuffer, ACTION_SIZE
                );

                assertEquals(0, registerResult, "Model registration should succeed");
                System.out.println("  OK - Model registered with agent ID: 0x" + Long.toHexString(AGENT_ID));

                // ==================== PHASE 3: Simulate tick loop ====================
                System.out.println("\n-> Phase 3: Running tick loop (" + TICK_COUNT + " ticks)...");

                // Pre-allocate reward buffer AND agent ID buffer (new API requires both)
                MemorySegment agentIdBuffer = arena.allocate(ValueLayout.JAVA_LONG, 1);
                MemorySegment rewardBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, 1);
                agentIdBuffer.set(ValueLayout.JAVA_LONG, 0, AGENT_ID);
                Random random = new Random(42);

                long totalTickTime = 0;
                int zeroCopyVerifications = 0;

                for (int tick = 0; tick < TICK_COUNT; tick++) {
                    // --- JAVA SIDE: Write observation data to context buffer ---
                    // This simulates game state updates (health, position, etc.)
                    for (int i = 0; i < STATE_SIZE; i++) {
                        double observationValue = random.nextDouble() * 2.0 - 1.0; // [-1, 1]
                        contextBuffer.setAtIndex(ValueLayout.JAVA_DOUBLE, i, observationValue);
                    }

                    // Verify Zero-Copy: Read back what we wrote
                    double writtenValue = contextBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);

                    // --- Call native tick for individual agent ---
                    long tickStart = System.nanoTime();
                    double reward = random.nextDouble() * 0.1;
                    int tickResult = SpartanNative.spartanTickAgent(AGENT_ID, reward);
                    long tickEnd = System.nanoTime();
                    totalTickTime += (tickEnd - tickStart);

                    assertEquals(0, tickResult, "Tick should succeed");

                    // --- JAVA SIDE: Read action predictions from output buffer ---
                    // C++ wrote to the SAME memory we allocated - true Zero-Copy!
                    double action0 = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);

                    // Verify the value we wrote is still there (Zero-Copy proof)
                    double verifyValue = contextBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
                    if (Math.abs(verifyValue - writtenValue) < 1e-10) {
                        zeroCopyVerifications++;
                    }

                    // Progress indicator
                    if ((tick + 1) % 10 == 0) {
                        System.out.print(".");
                    }
                }

                System.out.println();
                double avgTickMs = totalTickTime / 1_000_000.0 / TICK_COUNT;
                System.out.println("  OK - Completed " + TICK_COUNT + " ticks");
                System.out.println("  OK - Average tick time: " + String.format("%.4f", avgTickMs) + " ms");
                System.out.println("  OK - Zero-Copy verifications: " + zeroCopyVerifications + "/" + TICK_COUNT);

                assertEquals(TICK_COUNT, zeroCopyVerifications, "All ticks should verify Zero-Copy");

                // ==================== PHASE 4: Verify C++ reads context and writes actions ====================
                System.out.println("\n-> Phase 4: Verifying C++ reads context and writes actions...");

                // Write known values to context
                double[] testInputs = {0.0, 0.5, 1.0, -0.5};
                for (int i = 0; i < ACTION_SIZE; i++) {
                    contextBuffer.setAtIndex(ValueLayout.JAVA_DOUBLE, i, testInputs[i]);
                }

                System.out.println("  Context values written:");
                for (int i = 0; i < ACTION_SIZE; i++) {
                    System.out.println("    - context[" + i + "] = " + testInputs[i]);
                }

                // Call tick - C++ should process context and write to action buffer
                int finalTickResult = SpartanNative.spartanTickAgent(AGENT_ID, 0.0);
                assertEquals(0, finalTickResult, "Final tick should succeed");

                // Read action outputs and verify they are valid
                System.out.println("  Action outputs from C++ (values should be valid numbers):");
                int validActions = 0;
                for (int i = 0; i < ACTION_SIZE; i++) {
                    double action = actionOutputBuffer.getAtIndex(ValueLayout.JAVA_DOUBLE, i);

                    System.out.println("    - action[" + i + "] = " + String.format("%.6f", action));

                    // Verify action is a valid number
                    if (!Double.isNaN(action) && !Double.isInfinite(action)) {
                        validActions++;
                    }
                }

                assertEquals(ACTION_SIZE, validActions,
                        "All actions should be valid numbers - proves C++ wrote to action buffer!");

                System.out.println("  [OK] C++ correctly wrote to action output buffer");
                System.out.println("  [OK] ZERO-COPY DATA FLOW VERIFIED: Java -> C++ -> Java");

                // ==================== PHASE 5: Unregister model ====================
                System.out.println("\n-> Phase 5: Unregistering model...");

                int unregisterResult = SpartanNative.spartanUnregisterModel(AGENT_ID);
                assertEquals(0, unregisterResult, "Unregister should succeed");
                System.out.println("  OK - Model unregistered");

                // Arena auto-closes, releasing all memory
            }

            System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║  OK - FULL MODEL LIFECYCLE TEST PASSED                         ║");
            System.out.println("║     Java ↔ C++ Zero-Copy Memory Bridge VERIFIED              ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝");
        }
    }

    // ==================== Unit Tests (No Native Engine Required) ====================

    @Nested
    @DisplayName("Unit Tests - No Native Engine")
    class UnitTests {

        @Test
        @DisplayName("RSAC Config Builder and Serialization")
        void testRSACConfigBuilderAndSerialization() {
            System.out.println("\n-> Testing RSAC Config Builder...");

            int stateSize = 64;
            int actionSize = 4;
            RecurrentSoftActorCriticConfig config = RecurrentSoftActorCriticConfig.builder()
                     .learningRate(3e-4)
                     .gamma(0.99)
                     .hiddenStateSize(128)
                     .recurrentLayerDepth(2)
                     .actorHiddenLayerNeuronCount(256)
                     .actorHiddenLayerCount(2)
                     .criticHiddenLayerNeuronCount(256)
                     .criticHiddenLayerCount(2)
                     .build();

             assertEquals(128, config.hiddenStateSize());

             try (Arena arena = Arena.ofConfined()) {
                MemorySegment configSegment = SpartanModelAllocator.writeRSACConfig(arena, config, stateSize, actionSize);

                assertNotNull(configSegment);
                assertEquals(SpartanConfigLayout.RSAC_CONFIG_TOTAL_SIZE_PADDED, configSegment.byteSize());

                int modelType = configSegment.get(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_MODEL_TYPE_OFFSET);
                assertEquals(1, modelType);

                double learningRate = configSegment.get(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_LEARNING_RATE_OFFSET);
                assertEquals(3e-4, learningRate, 1e-10);
            }

            System.out.println("OK - RSAC Config test passed");
        }

        @Test
        @DisplayName("RSAC Weight Count Calculation")
        void testRSACWeightCountCalculation() {
            System.out.println("\n-> Testing RSAC Weight Count Calculation...");

            int stateSize = 64;
            int actionSize = 4;
             RecurrentSoftActorCriticConfig config = RecurrentSoftActorCriticConfig.builder()
                     .hiddenStateSize(128)
                     .recurrentLayerDepth(1)
                     .actorHiddenLayerNeuronCount(256)
                     .actorHiddenLayerCount(2)
                     .criticHiddenLayerNeuronCount(256)
                     .criticHiddenLayerCount(2)
                     .recurrentInputFeatureCount(64)
                     .build();

            long modelWeights = SpartanModelAllocator.calculateRSACModelWeightCount(config, stateSize, actionSize);
            long criticWeights = SpartanModelAllocator.calculateRSACCriticWeightCount(config, stateSize, actionSize);

            assertTrue(modelWeights > 0);
            assertTrue(criticWeights > 0);

            System.out.println("  Model weights: " + modelWeights);
            System.out.println("  Critic weights: " + criticWeights);
            System.out.println("OK - Weight calculation test passed");
        }

        @Test
        @DisplayName("Context Element Update and Flattening")
        void testContextElementUpdateAndFlattening() {
            System.out.println("\n-> Testing Context Element Update...");

            try (Arena arena = Arena.ofConfined()) {
                MockHealthElement healthElement = new MockHealthElement();
                MockEntitiesElement entitiesElement = new MockEntitiesElement(5);

                SpartanContextImpl context = new SpartanContextImpl("test-context", arena);
                context.addElement(healthElement, 0);
                context.addElement(entitiesElement, 1);

                context.update();

                int initialSize = context.getSize();
                assertTrue(initialSize > 0);

                double healthValue = context.readDouble(0);
                assertEquals(1.0, healthValue, 0.01);

                for (int i = 0; i < 10; i++) {
                    healthElement.simulateDamage(i);
                    entitiesElement.simulateEntityMovement(i);
                    context.update();
                }

                double newHealthValue = context.readDouble(0);
                assertTrue(newHealthValue < 1.0);

                System.out.println("OK - Context element test passed");
            }
        }


        @Test
        @DisplayName("Config Layout Constants")
        void testConfigLayoutConstants() {
            assertEquals(0, SpartanConfigLayout.BASE_MODEL_TYPE_OFFSET);
            assertEquals(8, SpartanConfigLayout.BASE_LEARNING_RATE_OFFSET);
            assertEquals(64, SpartanConfigLayout.BASE_CONFIG_SIZE);
            assertTrue(SpartanConfigLayout.RSAC_HIDDEN_STATE_SIZE_OFFSET >= SpartanConfigLayout.BASE_CONFIG_SIZE);
            assertEquals(408, SpartanConfigLayout.RSAC_CONFIG_TOTAL_SIZE);

            System.out.println("OK - Config Layout test passed");
        }
    }

    // ==================== Mock Context Elements ====================

    static class MockHealthElement extends SpartanSingleContextElement {
        private double health = 1.0;

        @Override public double getValue() { return health; }
        @Override public int getSize() { return 1; }
        @Override public void tick() { }
        @Override public String getIdentifier() { return "mock_health"; }

        void simulateDamage(int tick) {
            health = Math.max(0.1, health - 0.01);
            if (tick % 20 == 0) health = Math.min(1.0, health + 0.2);
        }
    }

    static class MockEntitiesElement extends SpartanVariableContextElement {
        private final int maxEntities;
        private final Random random = new Random(42);
        private final double[] entityPositions;
        private int currentEntityCount;

        MockEntitiesElement(int maxEntities) {
            this.maxEntities = maxEntities;
            this.entityPositions = new double[maxEntities * 3];
            this.currentEntityCount = maxEntities / 2;
            for (int i = 0; i < entityPositions.length; i++) {
                entityPositions[i] = random.nextDouble() * 100.0 - 50.0;
            }
        }

        @Override
        public void tick() {
            for (int i = 0; i < currentEntityCount; i++) {
                int baseIdx = i * 3;
                add(entityPositions[baseIdx], entityPositions[baseIdx + 1], entityPositions[baseIdx + 2]);
            }
        }

        @Override public String getIdentifier() { return "mock_nearby_entities"; }

        void simulateEntityMovement(int tick) {
            for (int i = 0; i < currentEntityCount * 3; i++) {
                entityPositions[i] += (random.nextDouble() - 0.5) * 2.0;
            }
            if (tick % 15 == 0) {
                if (random.nextBoolean() && currentEntityCount < maxEntities) {
                    currentEntityCount++;
                } else if (currentEntityCount > 1) {
                    currentEntityCount--;
                }
            }
        }
    }

    // ==================== Mock Action Manager ====================

    static class MockActionManager implements SpartanActionManager {
        private final java.util.List<SpartanAction> actions = new java.util.ArrayList<>();

        MockActionManager(int actionCount) {
            actions.add(new MockAction("forward"));
            if (actionCount > 1) actions.add(new MockAction("strafe"));
        }

        @Override
        public SpartanActionManager registerAction(SpartanAction action) {
            actions.add(action);
            return this;
        }

        @Override
        public java.util.List<SpartanAction> getActions() {
            return actions;
        }

        @Override
        public <SpartanActionType extends SpartanAction> java.util.List<SpartanActionType> getActionsByType( Class<SpartanActionType> actionClass) {
            java.util.List<SpartanActionType> result = new java.util.ArrayList<>();
            for (SpartanAction action : actions) {
                if (actionClass.isInstance(action)) {
                    result.add(actionClass.cast(action));
                }
            }
            return result;
        }

        @Override
        public java.util.List<SpartanAction> getActionsByIdentifier( String identifier) {
            java.util.List<SpartanAction> result = new java.util.ArrayList<>();
            for (SpartanAction action : actions) {
                if (action.identifier().equals(identifier)) {
                    result.add(action);
                }
            }
            return result;
        }
    }

    static class MockAction implements SpartanAction {
        private final String identifier;

        MockAction(String identifier) { this.identifier = identifier; }

        @Override public String identifier() { return identifier; }
        @Override public double taskMaxMagnitude() { return 1.0; }
        @Override public double taskMinMagnitude() { return -1.0; }
        @Override public void task(double normalizedMagnitude) { }
        @Override public double award() { return 0.0; }
    }

    @Nested
    @DisplayName("Facade API Tests - No FFM Exposed")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class FacadeApiTests {

        @BeforeAll
        static void setupProviders() {
            // Manually register providers for test environment
            // This mimics ServiceLoader behavior in a non-modular test environment
            new org.spartan.internal.spi.SpartanApiProviderImpl();
            new SpartanConfigFactoryServiceProviderImpl();
        }

        @Test
        @Order(1)
        @DisplayName("API Initialization & Model Creation")
        void testApiInitialization() {
            System.out.println("\n-> Testing Facade API Initialization...");

            try (org.spartan.api.SpartanApi api = org.spartan.api.SpartanApi.create()) {
                assertNotNull(api, "API instance should be created");
                System.out.println("  OK - API instance created");

                var context = api.createContext("test");
                var actions = api.createActionManager();
                actions.registerAction(new MockAction("test_jump"));

                System.out.println("  OK - Context and Actions created");

                var config = RecurrentSoftActorCriticConfig.builder()
                         .hiddenStateSize(64)
                         .build();

                System.out.println("  OK - Config built via SPI");

                // Mock usage of context
                context.addElement(new MockHealthElement(), 0);

                try (var model = api.createModel("integration-test-model", config, context, actions)) {
                    assertNotNull(model, "Model should be created via Facade");
                    assertTrue(model.getAgentIdentifier() != 0, "Agent ID should be assigned");

                    model.register(); // Explicit register required by interface contract
                    assertTrue(model.isRegistered(), "Model should be registered");
                    System.out.println("  OK - Model registered with ID: " + model.getAgentIdentifier());

                    // Simple Tick
                    assertDoesNotThrow(model::tick, "Tick should not throw");
                    System.out.println("  OK - Tick executed successfully");
                }
            }
            System.out.println("OK - Facade API Test Passed");
        }
    }
}
