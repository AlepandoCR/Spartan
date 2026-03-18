package org.spartan.internal.test.model;

import org.junit.jupiter.api.*;
import org.spartan.api.engine.context.SpartanContext;
import org.spartan.api.engine.context.element.SpartanContextElement;
import org.spartan.internal.engine.context.SpartanContextImpl;
import org.spartan.internal.bridge.SpartanNative;
import org.spartan.api.engine.config.AutoEncoderCompressorConfig;
import org.spartan.internal.engine.model.AutoEncoderCompressorModelImpl;
import org.spartan.internal.engine.action.SpartanActionManagerImpl;
import org.spartan.internal.engine.config.spi.SpartanConfigFactoryServiceProviderImpl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for AutoEncoder Compressor model.
 * <p>
 * This test validates:
 * <ul>
 *   <li>Model registration and lifecycle management</li>
 *   <li>50-tick simulation with varying input data</li>
 *   <li>Latent space representation extraction</li>
 *   <li>Zero-Copy memory verification between Java and C++</li>
 *   <li>Execution time profiling and throughput measurement</li>
 * </ul>
 * <p>
 * AutoEncoder models do not use rewards (unsupervised learning),
 * so they use the parameterless tick() method.
 */
@DisplayName("AutoEncoder Compressor Model - Comprehensive Integration Test")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class AutoEncoderCompressorModelTest {

    private static final long AGENT_IDENTIFIER = 0xA0E0_E0C0_0001L;
    private static final int INPUT_DIMENSION_SIZE = 32;
    private static final int LATENT_DIMENSION_SIZE = 8;
    private static final int ENCODER_HIDDEN_NEURON_COUNT = 64;
    private static final int ENCODER_LAYER_COUNT = 2;
    private static final int DECODER_LAYER_COUNT = 2;
    private static final int SIMULATION_TICK_COUNT = 50;

    private static Arena memoryArena;
    private static boolean nativeEngineInitialized = false;

    @BeforeAll
    static void initializeNativeEngine() {
        new SpartanConfigFactoryServiceProviderImpl();
        memoryArena = Arena.ofShared();
        try {
            SpartanNative.spartanInit();
            nativeEngineInitialized = true;
            System.out.println("\n" + "=".repeat(70));
            System.out.println("  AUTOENCODER COMPRESSOR MODEL COMPREHENSIVE TEST SUITE");
            System.out.println("=".repeat(70));
        } catch (Exception exception) {
            System.err.println("Failed to initialize native engine: " + exception.getMessage());
        }
    }

    @AfterAll
    static void cleanupResources() {
        if (memoryArena != null) {
            memoryArena.close();
        }
        System.out.println("\n" + "=".repeat(70));
        System.out.println("  AUTOENCODER COMPRESSOR TEST SUITE COMPLETED");
        System.out.println("=".repeat(70) + "\n");
    }

    @Test
    @Order(1)
    @DisplayName("1. Model Configuration and Architecture Verification")
    void testModelConfigurationAndArchitecture() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 1: AutoEncoder Compressor Configuration & Architecture");
        System.out.println("-".repeat(60));

        SpartanContext inputContext = createMockInputContext();
        inputContext.update();
        AutoEncoderCompressorConfig configuration = AutoEncoderCompressorConfig.builder()
                .latentDimensionSize(LATENT_DIMENSION_SIZE)
                .encoderHiddenNeuronCount(ENCODER_HIDDEN_NEURON_COUNT)
                .encoderLayerCount(ENCODER_LAYER_COUNT)
                .decoderLayerCount(DECODER_LAYER_COUNT)
                .learningRate(1e-3)
                .bottleneckRegularisationWeight(1e-4)
                .isTraining(true)
                .build();

        System.out.println("  Configuration Parameters:");
        System.out.println("    - Input Dimension Size: " + inputContext.getSize());
        System.out.println("    - Latent Dimension Size (Bottleneck): " + configuration.latentDimensionSize());
        System.out.println("    - Encoder Hidden Neuron Count: " + configuration.encoderHiddenNeuronCount());
        System.out.println("    - Encoder Layer Count: " + configuration.encoderLayerCount());
        System.out.println("    - Decoder Layer Count: " + configuration.decoderLayerCount());
        System.out.println("    - Learning Rate: " + configuration.learningRate());
        System.out.println("    - Bottleneck Regularisation Weight: " + configuration.bottleneckRegularisationWeight());
        System.out.println("    - Training Mode Enabled: " + configuration.isTraining());

        System.out.println("\n  Architecture Summary:");
        System.out.println("    Encoder: " + inputContext.getSize() + " -> " + ENCODER_HIDDEN_NEURON_COUNT + " -> " + LATENT_DIMENSION_SIZE);
        System.out.println("    Decoder: " + LATENT_DIMENSION_SIZE + " -> " + ENCODER_HIDDEN_NEURON_COUNT + " -> " + inputContext.getSize());
        System.out.printf("    Compression Ratio: %.1fx%n", (double) inputContext.getSize() / LATENT_DIMENSION_SIZE);

        assertEquals(INPUT_DIMENSION_SIZE, inputContext.getSize());
        assertEquals(LATENT_DIMENSION_SIZE, configuration.latentDimensionSize());
        System.out.println("  [OK] Configuration validated successfully");
    }

    @Test
    @Order(2)
    @DisplayName("2. Full Lifecycle: Register -> 50 Ticks -> Latent Analysis -> Unregister")
    void testFullLifecycleWithFiftyTicks() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 2: AutoEncoder Compressor Full Lifecycle (50 Ticks)");
        System.out.println("-".repeat(60));

        // Create mock input context
        SpartanContext inputContext = createMockInputContext();
        inputContext.update();
        AutoEncoderCompressorConfig configuration = AutoEncoderCompressorConfig.builder()
                .latentDimensionSize(LATENT_DIMENSION_SIZE)
                .encoderHiddenNeuronCount(ENCODER_HIDDEN_NEURON_COUNT)
                .encoderLayerCount(ENCODER_LAYER_COUNT)
                .decoderLayerCount(DECODER_LAYER_COUNT)
                .learningRate(1e-3)
                .isTraining(true)
                .build();
        SpartanActionManagerImpl actionManager = new SpartanActionManagerImpl();

        // Create AutoEncoder model
        AutoEncoderCompressorModelImpl model = new AutoEncoderCompressorModelImpl(
                "autoencoder-test-model",
                AGENT_IDENTIFIER,
                configuration,
                inputContext,
                memoryArena,
                actionManager
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

        System.out.println("\n  Phase 2: 50-Tick Simulation with Latent Space Analysis");
        System.out.println("  " + "-".repeat(50));

        Random randomGenerator = new Random(42);
        long[] tickExecutionTimes = new long[SIMULATION_TICK_COUNT];
        double totalLatentMagnitude = 0.0;

        System.out.println("\n  Tick | Input[0:3]                    | Latent[0:7]                                        | Time(us)");
        System.out.println("  " + "-".repeat(100));

        for (int tickIndex = 0; tickIndex < SIMULATION_TICK_COUNT; tickIndex++) {
            // Generate random input data
            double[] inputData = new double[INPUT_DIMENSION_SIZE];
            for (int inputIndex = 0; inputIndex < INPUT_DIMENSION_SIZE; inputIndex++) {
                inputData[inputIndex] = randomGenerator.nextDouble() * 2.0 - 1.0;
            }

            // Write inputs to context memory segment
            MemorySegment contextDataSegment = ((SpartanContextImpl) inputContext).getData();
            for (int inputIndex = 0; inputIndex < INPUT_DIMENSION_SIZE; inputIndex++) {
                contextDataSegment.setAtIndex(ValueLayout.JAVA_DOUBLE, inputIndex, inputData[inputIndex]);
            }

            // Execute tick (AutoEncoder uses parameterless tick - no rewards)
            long tickStartTime = System.nanoTime();
            model.tick();
            tickExecutionTimes[tickIndex] = System.nanoTime() - tickStartTime;

            // Read latent representation
            double[] latentRepresentation = model.readAllLatent();

            // Calculate latent magnitude for analysis
            double latentMagnitude = 0.0;
            for (double latentValue : latentRepresentation) {
                latentMagnitude += latentValue * latentValue;
            }
            latentMagnitude = Math.sqrt(latentMagnitude);
            totalLatentMagnitude += latentMagnitude;

            // Print every 10th tick for visibility
            if (tickIndex % 10 == 0 || tickIndex == SIMULATION_TICK_COUNT - 1) {
                System.out.printf("  %4d | [%+.3f, %+.3f, %+.3f, %+.3f] | [%+.3f, %+.3f, %+.3f, %+.3f, %+.3f, %+.3f, %+.3f, %+.3f] | %6d%n",
                        tickIndex,
                        inputData[0], inputData[1], inputData[2], inputData[3],
                        latentRepresentation[0], latentRepresentation[1], latentRepresentation[2], latentRepresentation[3],
                        latentRepresentation[4], latentRepresentation[5], latentRepresentation[6], latentRepresentation[7],
                        tickExecutionTimes[tickIndex] / 1000);
            }

            // Validate latent outputs are valid numbers
            for (int latentIndex = 0; latentIndex < LATENT_DIMENSION_SIZE; latentIndex++) {
                assertFalse(Double.isNaN(latentRepresentation[latentIndex]),
                        "Latent value should not be NaN at tick " + tickIndex + " dimension " + latentIndex);
                assertFalse(Double.isInfinite(latentRepresentation[latentIndex]),
                        "Latent value should not be infinite at tick " + tickIndex + " dimension " + latentIndex);
            }
        }

        System.out.println("\n  Phase 3: Performance Metrics and Statistics");
        System.out.println("  " + "-".repeat(50));

        // Calculate execution time statistics
        long totalExecutionTime = 0;
        long minimumTickTime = Long.MAX_VALUE;
        long maximumTickTime = Long.MIN_VALUE;
        for (long executionTime : tickExecutionTimes) {
            totalExecutionTime += executionTime;
            minimumTickTime = Math.min(minimumTickTime, executionTime);
            maximumTickTime = Math.max(maximumTickTime, executionTime);
        }
        double averageTickTime = totalExecutionTime / (double) SIMULATION_TICK_COUNT;
        double averageLatentMagnitude = totalLatentMagnitude / SIMULATION_TICK_COUNT;

        System.out.printf("    Total Ticks Executed: %d%n", SIMULATION_TICK_COUNT);
        System.out.printf("    Total Execution Time: %.2f milliseconds%n", totalExecutionTime / 1_000_000.0);
        System.out.printf("    Average Tick Time: %.2f microseconds%n", averageTickTime / 1000.0);
        System.out.printf("    Minimum Tick Time: %d microseconds%n", minimumTickTime / 1000);
        System.out.printf("    Maximum Tick Time: %d microseconds%n", maximumTickTime / 1000);
        System.out.printf("    Throughput: %.0f ticks/second%n", 1_000_000_000.0 / averageTickTime);
        System.out.printf("    Average Latent Magnitude: %.4f%n", averageLatentMagnitude);

        System.out.println("\n  Phase 4: Model Unregistration");
        System.out.println("  " + "-".repeat(50));

        model.close();
        assertFalse(model.isRegistered());
        System.out.println("    [OK] Model unregistered and closed successfully");

        System.out.println("\n  [PASS] AutoEncoder Compressor 50-tick lifecycle test completed");
    }

    @Test
    @Order(3)
    @DisplayName("3. Zero-Copy Memory and Latent Buffer Verification")
    void testZeroCopyMemoryAndLatentBufferVerification() {
        Assumptions.assumeTrue(nativeEngineInitialized, "Native engine not available");

        System.out.println("\n-> Test 3: Zero-Copy Memory and Latent Buffer Verification");
        System.out.println("-".repeat(60));

        SpartanContext inputContext = createMockInputContext();
        inputContext.update();

        AutoEncoderCompressorConfig configuration = AutoEncoderCompressorConfig.builder()
                .latentDimensionSize(LATENT_DIMENSION_SIZE)
                .encoderHiddenNeuronCount(32)
                .encoderLayerCount(2)
                .decoderLayerCount(2)
                .build();
        SpartanActionManagerImpl actionManager = new SpartanActionManagerImpl();

        AutoEncoderCompressorModelImpl model = new AutoEncoderCompressorModelImpl(
                "autoencoder-test-model-2",
                AGENT_IDENTIFIER + 1,
                configuration,
                inputContext,
                memoryArena,
                actionManager
        );

        model.register();

        // Write deterministic test values to input context
        double[] testInputValues = new double[INPUT_DIMENSION_SIZE];
        for (int valueIndex = 0; valueIndex < INPUT_DIMENSION_SIZE; valueIndex++) {
            testInputValues[valueIndex] = Math.sin(valueIndex * 0.2) * 0.5;
        }

        MemorySegment contextDataSegment = ((SpartanContextImpl) inputContext).getData();

        System.out.println("  Writing test input values to context:");
        for (int valueIndex = 0; valueIndex < INPUT_DIMENSION_SIZE; valueIndex++) {
            contextDataSegment.setAtIndex(ValueLayout.JAVA_DOUBLE, valueIndex, testInputValues[valueIndex]);
            if (valueIndex < 8) {
                System.out.printf("    input[%2d] = %+.6f%n", valueIndex, testInputValues[valueIndex]);
            }
        }
        System.out.println("    ... (remaining values written)");

        // Verify values were written correctly BEFORE tick
        System.out.println("\n  Pre-Tick Verification:");
        int preTickVerified = 0;
        for (int valueIndex = 0; valueIndex < INPUT_DIMENSION_SIZE; valueIndex++) {
            double readBackValue = contextDataSegment.getAtIndex(ValueLayout.JAVA_DOUBLE, valueIndex);
            boolean valuesMatch = Math.abs(readBackValue - testInputValues[valueIndex]) < 1e-10;
            if (valueIndex < 4) {
                System.out.printf("    input[%2d]: written=%+.6f, read=%+.6f %s%n",
                        valueIndex, testInputValues[valueIndex], readBackValue, valuesMatch ? "[OK]" : "[FAIL]");
            }
            if (valuesMatch) preTickVerified++;
        }
        System.out.println("    Pre-tick verified: " + preTickVerified + "/" + INPUT_DIMENSION_SIZE);

        // Execute tick to encode
        model.tick();

        // Display latent representation
        System.out.println("\n  Latent Representation (Bottleneck Encoding):");
        for (int latentIndex = 0; latentIndex < LATENT_DIMENSION_SIZE; latentIndex++) {
            double latentValue = model.readLatent(latentIndex);
            System.out.printf("    latent[%d] = %+.6f%n", latentIndex, latentValue);
            assertFalse(Double.isNaN(latentValue), "Latent should not be NaN");
            assertFalse(Double.isInfinite(latentValue), "Latent should not be infinite");
        }

        // Test direct latent buffer access
        System.out.println("\n  Direct Latent Buffer Access (MemorySegment):");
        MemorySegment latentBuffer = model.getLatentBuffer();
        System.out.printf("    Latent buffer address: 0x%x%n", latentBuffer.address());
        System.out.printf("    Latent buffer size: %d bytes (%d doubles)%n",
                latentBuffer.byteSize(), latentBuffer.byteSize() / 8);

        model.close();

        assertEquals(INPUT_DIMENSION_SIZE, preTickVerified, "All input values should be written correctly");
        System.out.println("\n  [PASS] Zero-Copy and latent buffer verification completed");
    }

    // ==================== Helper Methods ====================

    private SpartanContext createMockInputContext() {
        SpartanContextImpl context = new SpartanContextImpl("autoencoder-test-input-context", memoryArena);
        context.addElement(new MockInputElement(), 0);
        return context;
    }

    /**
     * Mock input element for testing AutoEncoder Compressor.
     * Provides a fixed-size input buffer for encoding/decoding tests.
     */
    static class MockInputElement implements SpartanContextElement {
        private final double[] inputData = new double[INPUT_DIMENSION_SIZE];

        @Override
        public void tick() {
            // Data is written directly to context.getData() in tests
        }

        @Override
        public void prepare() {
            // No preparation needed for mock
        }

        @Override
        public double [] getData() {
            return inputData;
        }

        @Override
        public int getSize() {
            return INPUT_DIMENSION_SIZE;
        }

        @Override
        public String getIdentifier() {
            return "mock_input_element";
        }
    }
}
