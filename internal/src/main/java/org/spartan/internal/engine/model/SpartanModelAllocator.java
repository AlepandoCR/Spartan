package org.spartan.internal.engine.model;

import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Utility class responsible for allocating C-compatible memory buffers
 * required by the Spartan native engine for model weights and parameters.
 * <p>
 * All allocations use a strict 64-byte alignment to support AVX-512 (8 doubles per SIMD register).
 * This provides optimal performance on Zen 5 processors (256-bit L1 cache lines) and prevents
 * over-read violations during vectorized tensor operations.
 * <p>
 * Element padding ensures all weight buffers contain a multiple of 8 doubles, allowing
 * SIMD kernels to process the entire buffer without special case handling at boundaries.
 */
public final class SpartanModelAllocator {

    /**
     * Alignment for AVX-512 and optimal cache line utilization.
     * 64 bytes = 8 doubles = 512 bits.
     */
    private static final long SIMD_ALIGNMENT_BYTES = 64;

    /**
     * Extra padding elements to prevent AVX-512 over-read for unaligned tail accesses.
     * Increased to 1024 to provide robust safety margin against strided access patterns.
     */
    private static final long SAFETY_PADDING_ELEMENTS = 1024L;

    private SpartanModelAllocator() {}

    /**
     * Pads a double element count to the nearest multiple of 8.
     * <p>
     * AVX-512 processes 8 doubles per instruction (_mm512_load_pd).
     * This ensures that all weight buffers can be safely processed by vectorized kernels
     * without remainder loops or boundary violations.
     * <p>
     * Guard: Input validation to prevent integer overflows during SIMD padding calculations.
     *
     * @param elementCount the number of double elements
     * @return the padded element count (always a multiple of 8)
     * @throws IllegalArgumentException if elementCount is negative
     */
    private static long simdPadElementCount(long elementCount) {
        if (elementCount < 0L) {
            throw new IllegalArgumentException("Element count cannot be negative: " + elementCount);
        }
        // Pad to nearest multiple of 8
        return ((elementCount + 7L) >> 3) << 3;
    }

    /**
     * Pads a byte count to the nearest multiple of 64.
     * <p>
     * Ensures that memory allocations are always sized to full cache lines (512 bits / 64 bytes),
     * enabling safe AVX-512 read operations without page boundary crossings.
     *
     * @param byteCount raw number of bytes
     * @return padded byte count (always a multiple of 64)
     */
    private static long simdPadBytes(long byteCount) {
        if (byteCount < 0L) {
            throw new IllegalArgumentException("Byte count cannot be negative: " + byteCount);
        }
        return (byteCount + 63L) & ~63L;
    }

    /**
     * Pads a raw element count to the nearest multiple of 8.
     * <p>
     * This is a lightweight helper for padding counts that are known to be safe (e.g., already validated).
     *
     * @param elementCount the raw element count
     * @return the padded element count (multiple of 8)
     */
    private static long simdPad(long elementCount) {
        return (elementCount + 7) & ~7;
    }

    /**
     * Allocates a buffer for model weights with explicit count.
     * The buffer is padded to a multiple of 8 elements and aligned to 64 bytes.
     *
     * @param modelWeightCount total number of double weights
     * @param arena the Arena for allocation
     * @return allocated weight buffer
     */
    public static MemorySegment allocateModelWeightsBuffer(long modelWeightCount, Arena arena) {
        return allocateDoubles(arena, modelWeightCount);
    }

    /**
     * Allocates a buffer for critic weights with explicit count.
     * The buffer is padded to a multiple of 8 elements and aligned to 64 bytes.
     *
     * @param criticWeightCount total number of double weights for critic
     * @param arena the Arena for allocation
     * @return allocated critic weight buffer
     */
    public static MemorySegment allocateCriticWeightsBuffer(long criticWeightCount, Arena arena) {
        return allocateDoubles(arena, criticWeightCount);
    }

    /**
     * Allocates a buffer for action outputs.
     * Padded to support SIMD operations.
     *
     * @param arena the Arena for allocation
     * @param actionSize number of action dimensions
     * @return allocated action buffer with 64-byte alignment
     */
    public static @NotNull MemorySegment allocateActionOutputBuffer(@NotNull Arena arena, int actionSize) {
        return allocateDoubles(arena, actionSize);
    }

    /**
     * Allocates a buffer for context data (doubles).
     * Padded to support SIMD operations.
     *
     * @param arena the Arena for allocation
     * @param contextSize number of context elements
     * @return allocated context buffer with 64-byte alignment
     */
    public static @NotNull MemorySegment allocateContextBuffer(@NotNull Arena arena, int contextSize) {
        return allocateDoubles(arena, contextSize);
    }

    /**
     * Allocates a generic buffer of doubles with SIMD padding and alignment.
     *
     * @param arena the Arena for allocation
     * @param count the number of double elements
     * @return allocated buffer with 64-byte alignment
     */
    public static @NotNull MemorySegment allocateDoubles(@NotNull Arena arena, long count) {
        long paddedCount = simdPad(count);
        return arena.allocate(paddedCount * Double.BYTES, 64);
    }


    // ==================== Topology Calculations ====================

    // ==================== RSAC Config Serialization ====================

    /**
     * Allocates and writes an RSAC config to a MemorySegment with C-compatible layout.
     * <p>
     * The resulting segment can be passed directly to C++ as a void* pointer.
     * The layout matches {@code RecurrentSoftActorCriticHyperparameterConfig} exactly.
     * Allocated with 64-byte alignment.
     *
     * @param arena  the Arena for allocation
     * @param config the Java config record
     * @return MemorySegment containing the serialized config (aligned to 64 bytes)
     */
    public static @NotNull MemorySegment writeRSACConfig(
            @NotNull Arena arena,
            @NotNull RecurrentSoftActorCriticConfig config,
            int stateSize,
            int actionSize
    ) {
        // Allocate with 64-byte alignment for AVX-512 safety
        // Ensure size is padded to 64-byte boundary
        MemorySegment segment = arena.allocate(simdPadBytes(SpartanConfigLayout.RSAC_CONFIG_TOTAL_SIZE), SIMD_ALIGNMENT_BYTES);

        // Write BaseHyperparameterConfig fields
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_MODEL_TYPE_OFFSET,
                config.modelType().getNativeValue());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_LEARNING_RATE_OFFSET,
                config.learningRate());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_GAMMA_OFFSET,
                config.gamma());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_OFFSET,
                config.epsilon());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_MIN_OFFSET,
                config.epsilonMin());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_DECAY_OFFSET,
                config.epsilonDecay());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_STATE_SIZE_OFFSET,
                stateSize);
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_ACTION_SIZE_OFFSET,
                actionSize);
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_IS_TRAINING_OFFSET,
                (byte) (config.isTraining() ? 1 : 0));
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_DEBUG_LOGGING_OFFSET,
                (byte) (config.debugLogging() ? 1 : 0));

        // Write RSAC-specific fields
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_HIDDEN_STATE_SIZE_OFFSET,
                config.hiddenStateSize());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_RECURRENT_LAYER_DEPTH_OFFSET,
                config.recurrentLayerDepth());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_ACTOR_HIDDEN_NEURON_COUNT_OFFSET,
                config.actorHiddenLayerNeuronCount());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_ACTOR_HIDDEN_LAYER_COUNT_OFFSET,
                config.actorHiddenLayerCount());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_CRITIC_HIDDEN_NEURON_COUNT_OFFSET,
                config.criticHiddenLayerNeuronCount());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_CRITIC_HIDDEN_LAYER_COUNT_OFFSET,
                config.criticHiddenLayerCount());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_TARGET_SMOOTHING_OFFSET,
                config.targetSmoothingCoefficient());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_ENTROPY_ALPHA_OFFSET,
                config.entropyTemperatureAlpha());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_FIRST_CRITIC_LR_OFFSET,
                config.firstCriticLearningRate());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_SECOND_CRITIC_LR_OFFSET,
                config.secondCriticLearningRate());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_POLICY_LR_OFFSET,
                config.policyNetworkLearningRate());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_TARGET_ENTROPY_OFFSET,
                config.targetEntropy());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_ALPHA_LEARNING_RATE_OFFSET,
                config.alphaLearningRate());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_RECURRENT_INPUT_FEATURE_COUNT_OFFSET,
                config.recurrentInputFeatureCount());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_NESTED_ENCODER_COUNT_OFFSET,
                0); // Nested encoders no longer supported - always 0
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_REMORSE_BUFFER_CAPACITY_OFFSET,
                config.remorseTraceBufferCapacity());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_REMORSE_SIMILARITY_THRESHOLD_OFFSET,
                config.remorseMinimumSimilarityThreshold());

        return segment;
    }

    /**
     * Calculates the total weight count for an RSAC model's actor network.
     *
     * Includes:
     * 1. Policy Network (Actor): Multi-layer MLP with GRU input.
     *
     * Note: Nested AutoEncoders are no longer supported.
     *
     * @param config the RSAC configuration
     * @return total number of double weights for the actor (SIMD-padded)
     */
    public static long calculateRSACModelWeightCount(
            @NotNull RecurrentSoftActorCriticConfig config,
            int stateSize,
            int actionSize
    ) {
        // Convert to long to prevent integer overflow
        long gruHiddenSize = config.hiddenStateSize();
        long actorHiddenSize = config.actorHiddenLayerNeuronCount();
        long actorLayerCount = config.actorHiddenLayerCount();

        // Policy Network (Actor) Calculation
        // Input layer: GRU Hidden State -> First Actor Hidden Layer
        long policyLayer1Weights = actorHiddenSize * gruHiddenSize;

        // Intermediate Hidden Layers: (actorLayerCount - 1) * hidden * hidden
        long additionalHiddenLayersCount = Math.max(0L, actorLayerCount - 1L);
        long intermediateWeights = additionalHiddenLayersCount * actorHiddenSize * actorHiddenSize;

        // Output layers: Final Hidden -> Mean and Final Hidden -> LogStd
        // These are parallel dense layers mapping to action space
        long policyMeanWeights = (long) actionSize * actorHiddenSize;
        long policyLogStdWeights = (long) actionSize * actorHiddenSize;

        // Biases: Hidden neurons per layer + Output neurons (Mean + LogStd)
        long actorBiases = (actorLayerCount * actorHiddenSize) + ((long) actionSize * 2);

        // Individual SIMD Padding for the Actor block
        long actorTotalRaw = policyLayer1Weights + intermediateWeights +
                policyMeanWeights + policyLogStdWeights + actorBiases;
        long actorTotalPadded = simdPadElementCount(actorTotalRaw);

        // Final result: Actor weights only (nested encoders no longer supported)
        return actorTotalPadded + SAFETY_PADDING_ELEMENTS;
    }

    /**
     * Calculates the total weight count for RSAC critic buffer.
     * <p>
     * The critic buffer contains:
     * <ul>
     *   <li>GRU: 3 gates × (input + hidden) × hidden weights + biases + hidden state</li>
     *   <li>Twin Q-Critics (Q1, Q2): each takes (GRU_output + action) → hidden → scalar Q-value</li>
     * </ul>
     * <p>
     *
     * @param config the RSAC configuration
     * @return total number of double weights for GRU + critics (SIMD-padded)
     * @throws IllegalArgumentException if configuration parameters would cause arithmetic overflow
     */
    public static long calculateRSACCriticWeightCount(
            @NotNull RecurrentSoftActorCriticConfig config,
            int stateSize,
            int actionSize
    ) {
        long gruHiddenSize = config.hiddenStateSize();
        long gruInputSize = config.recurrentInputFeatureCount() > 0 ? config.recurrentInputFeatureCount() : stateSize;
        long gruConcatSize = gruHiddenSize + gruInputSize;

        // 1. GRU Components
        long gruGateWeights = 3L * gruHiddenSize * gruConcatSize;
        long gruGateBiases = 3L * gruHiddenSize;
        long gruHiddenState = gruHiddenSize;
        long gruTotal = simdPadElementCount(gruGateWeights + gruGateBiases + gruHiddenState);

        // 2. Twin Critic Components (Multi-Layer)
        long criticInput = gruHiddenSize + actionSize;
        long criticHidden = config.criticHiddenLayerNeuronCount();
        long criticLayers = config.criticHiddenLayerCount();

        // Weights: Input->L1 + (L1->Ln) + Ln->Output
        long weightCountPerNetwork = (criticInput * criticHidden)
                + (criticHidden * criticHidden * (criticLayers - 1))
                + criticHidden;

        // Biases: Hidden Layers + Output Scalar
        long biasCountPerNetwork = (criticHidden * criticLayers) + 1L;

        long singleCriticTotal = simdPadElementCount(weightCountPerNetwork + biasCountPerNetwork);
        long twinCriticsTotal = singleCriticTotal * 2L;

        return gruTotal + twinCriticsTotal + SAFETY_PADDING_ELEMENTS;
    }

    /**
     * Calculates the weight count for an AutoEncoder (encoder + decoder).
     * <p>
     * Validates dimensions and ensures all intermediate calculations use long type
     * to prevent integer overflow.
     *
     * @param inputSize      input/output dimensionality
     * @param hiddenNeurons  neurons in hidden layer
     * @param latentSize     bottleneck dimensionality
     * @return total number of weights
     * @throws IllegalArgumentException if dimensions are invalid or would cause overflow
     */
    public static long calculateAutoEncoderWeightCount(int inputSize, int hiddenNeurons, int latentSize) {
        // Convert to long to prevent integer overflow

        // Validate dimensions
        if ((long) inputSize <= 0L || (long) hiddenNeurons <= 0L || (long) latentSize <= 0L) {
            throw new IllegalArgumentException(
                    "Invalid AutoEncoder dimensions: input=" + (long) inputSize + ", hidden=" + (long) hiddenNeurons + ", latent=" + (long) latentSize);
        }

        // Encoder: input -> hidden -> latent
        long total = getTotalWeight(inputSize, hiddenNeurons, latentSize);

        // Validate total is positive
        if (total <= 0L) {
            throw new IllegalArgumentException(
                    "AutoEncoder weight calculation resulted in non-positive value: " + total);
        }

        // Add safety padding
        return total + SAFETY_PADDING_ELEMENTS;
    }

    private static long getTotalWeight(long inputSize, long hiddenNeurons, long latentSize) {
        long encoderInputLayer = inputSize * hiddenNeurons;
        long encoderLatentLayer = hiddenNeurons * latentSize;
        long encoderBiases = hiddenNeurons + latentSize;

        // Decoder: latent -> hidden -> output
        long decoderHiddenLayer = latentSize * hiddenNeurons;
        long decoderOutputLayer = hiddenNeurons * inputSize;
        long decoderBiases = hiddenNeurons + inputSize;

        // Sum all weights
        return encoderInputLayer + encoderLatentLayer + encoderBiases
                + decoderHiddenLayer + decoderOutputLayer + decoderBiases;
    }

    // ==================== DDQN Weight Calculations ====================

    /**
     * Calculates the total weight count for a DDQN model.
     * <p>
     * DDQN uses two networks: online and target (identical structure).
     * Each network is an MLP: state -> hidden layers -> Q-values per action.
     * <p>
     * Each network is SIMD-padded individually before multiplication to 2 to ensure
     * safe vectorized memory access for both online and target networks.
     *
     * @param config the DDQN configuration
     * @return total number of double weights for both networks (SIMD-padded)
     * @throws IllegalArgumentException if configuration parameters would cause arithmetic overflow
     */
    public static long calculateDDQNModelWeightCount(
            @NotNull DoubleDeepQNetworkConfig config,
            int stateSize,
            int actionSize
    ) {
        // Convert to long to prevent integer overflow
        long hiddenNeurons = config.hiddenLayerNeuronCount();
        long hiddenLayers = config.hiddenLayerCount();

        // Validate dimensions
        if ((long) stateSize <= 0L || hiddenNeurons <= 0L || hiddenLayers <= 0L || (long) actionSize <= 0L) {
            throw new IllegalArgumentException(
                    "Invalid DDQN dimensions: state=" + (long) stateSize + ", hidden=" + hiddenNeurons
                            + ", layers=" + hiddenLayers + ", actions=" + (long) actionSize);
        }

        long inputLayer = (long) stateSize * hiddenNeurons;
        long additionalHiddenLayers = Math.max(0L, hiddenLayers - 1L);
        long hiddenLayersWeights = additionalHiddenLayers * hiddenNeurons * hiddenNeurons;
        long outputLayer = hiddenNeurons * (long) actionSize;
        long biases = hiddenLayers * hiddenNeurons + (long) actionSize;

        long singleNetworkRawWeights = inputLayer + hiddenLayersWeights + outputLayer + biases;

        // Validate positive
        if (singleNetworkRawWeights <= 0L) {
            throw new IllegalArgumentException(
                    "Single network weight calculation resulted in non-positive value: " + singleNetworkRawWeights);
        }

        // SIMD-pad single network
        long singleNetworkWeights = simdPadElementCount(singleNetworkRawWeights);

        // Online network + Target network
        if (singleNetworkWeights > Long.MAX_VALUE / 2L) {
            throw new IllegalArgumentException(
                    "Total DDQN weight count would overflow: single=" + singleNetworkWeights);
        }

        // Add safety padding
        return singleNetworkWeights * 2L + SAFETY_PADDING_ELEMENTS;
    }

    /**
     * DDQN doesn't have separate critic weights - uses same buffer as model.
     * Returns 0 as placeholder for interface compatibility.
     */
    public static long calculateDDQNCriticWeightCount(@NotNull DoubleDeepQNetworkConfig config) {
        // DDQN combines value estimation in the Q-network itself
        // No separate critic buffer needed
        return 0;
    }

    // ==================== DDQN Config Serialization ====================

    /**
     * Allocates and writes a DDQN config to a MemorySegment with C-compatible layout.
     * <p>
     * Layout matches {@code DoubleDeepQNetworkHyperparameterConfig} exactly.
     * Allocated with 64-byte alignment.
     *
     * @param arena  the Arena for allocation
     * @param config the Java config record
     * @return MemorySegment containing the serialized config (aligned to 64 bytes)
     */
    public static @NotNull MemorySegment writeDDQNConfig(
            @NotNull Arena arena,
            @NotNull DoubleDeepQNetworkConfig config,
            int stateSize,
            int actionSize
    ) {
        MemorySegment segment = arena.allocate(simdPadBytes(SpartanConfigLayout.DDQN_CONFIG_TOTAL_SIZE), SIMD_ALIGNMENT_BYTES);

        // Write BaseHyperparameterConfig fields
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_MODEL_TYPE_OFFSET,
                config.modelType().getNativeValue());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_LEARNING_RATE_OFFSET,
                config.learningRate());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_GAMMA_OFFSET,
                config.gamma());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_OFFSET,
                config.epsilon());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_MIN_OFFSET,
                config.epsilonMin());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_DECAY_OFFSET,
                config.epsilonDecay());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_STATE_SIZE_OFFSET,
                stateSize);
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_ACTION_SIZE_OFFSET,
                actionSize);
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_IS_TRAINING_OFFSET,
                (byte) (config.isTraining() ? 1 : 0));
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_DEBUG_LOGGING_OFFSET,
                (byte) (config.debugLogging() ? 1 : 0));

        // Write DDQN-specific fields
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_TARGET_SYNC_INTERVAL_OFFSET,
                config.targetNetworkSyncInterval());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_REPLAY_BUFFER_CAPACITY_OFFSET,
                config.replayBufferCapacity());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_REPLAY_BATCH_SIZE_OFFSET,
                config.replayBatchSize());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_HIDDEN_NEURON_COUNT_OFFSET,
                config.hiddenLayerNeuronCount());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.DDQN_HIDDEN_LAYER_COUNT_OFFSET,
                config.hiddenLayerCount());

        return segment;
    }

    // ==================== AutoEncoder Weight Calculations ====================

    /**
     * Calculates the total weight count for an AutoEncoder model.
     * <p>
     * CRITICAL: This must match EXACTLY what C++ expects in constructAutoEncoderCompressorModel.
     * The AutoEncoder uses a 2-layer architecture for both encoder and decoder:
     * <pre>
     * Encoder Layer 1: stateSize -> hiddenSize (weights + biases)
     * Encoder Layer 2: hiddenSize -> latentSize (weights + biases)
     * Decoder Layer 1: latentSize -> hiddenSize (weights + biases)
     * Decoder Layer 2: hiddenSize -> stateSize (weights + biases)
     * Latent Buffer: latentSize doubles
     *
     * Layout: [encoderWeights | encoderBiases | decoderWeights | decoderBiases | latentBuffer]
     * Each sub-section is SIMD-padded individually.
     * </pre>
     *
     * @param config the AutoEncoder configuration
     * @return total number of double weights (SIMD-padded)
     */
    public static long calculateAutoEncoderModelWeightCount(
            @NotNull AutoEncoderCompressorConfig config,
            int stateSize
    ) {
        int hiddenSize = config.encoderHiddenNeuronCount();
        int latentSize = config.latentDimensionSize();

        // Encoder: 2 layers (input->hidden, hidden->latent)
        long encoderLayer1Weights = (long) hiddenSize * stateSize;
        long encoderLayer2Weights = (long) latentSize * hiddenSize;
        long encoderWeightCount = simdPadElementCount(encoderLayer1Weights + encoderLayer2Weights);
        long encoderBiasCount = simdPadElementCount((long) hiddenSize + latentSize);

        // Decoder: 2 layers (latent->hidden, hidden->output)
        long decoderLayer1Weights = (long) hiddenSize * latentSize;
        long decoderLayer2Weights = (long) stateSize * hiddenSize;
        long decoderWeightCount = simdPadElementCount(decoderLayer1Weights + decoderLayer2Weights);
        long decoderBiasCount = simdPadElementCount((long) hiddenSize + stateSize);

        // Latent buffer space
        long latentBufferCount = simdPadElementCount(latentSize);

        return encoderWeightCount + encoderBiasCount
                + decoderWeightCount + decoderBiasCount
                + latentBufferCount;
    }

    /**
     * AutoEncoder doesn't have separate critic weights.
     * Returns 0 as placeholder for interface compatibility.
     */
    public static long calculateAutoEncoderCriticWeightCount(@NotNull AutoEncoderCompressorConfig config) {
        return 0;
    }

    // ==================== AutoEncoder Config Serialization ====================

    /**
     * Allocates and writes an AutoEncoder config to a MemorySegment with C-compatible layout.
     * <p>
     * Layout matches {@code AutoEncoderCompressorHyperparameterConfig} exactly.
     * Allocated with 64-byte alignment.
     *
     * @param arena  the Arena for allocation
     * @param config the Java config record
     * @return MemorySegment containing the serialized config (aligned to 64 bytes)
     */
    public static @NotNull MemorySegment writeAutoEncoderConfig(
            @NotNull Arena arena,
            @NotNull AutoEncoderCompressorConfig config,
            int stateSize,
            int actionSize
    ) {
        MemorySegment segment = arena.allocate(simdPadBytes(SpartanConfigLayout.AE_CONFIG_TOTAL_SIZE), SIMD_ALIGNMENT_BYTES);

        // Write BaseHyperparameterConfig fields
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_MODEL_TYPE_OFFSET,
                config.modelType().getNativeValue());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_LEARNING_RATE_OFFSET,
                config.learningRate());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_GAMMA_OFFSET,
                config.gamma());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_OFFSET,
                config.epsilon());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_MIN_OFFSET,
                config.epsilonMin());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_DECAY_OFFSET,
                config.epsilonDecay());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_STATE_SIZE_OFFSET,
                stateSize);
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_ACTION_SIZE_OFFSET,
                actionSize);
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_IS_TRAINING_OFFSET,
                (byte) (config.isTraining() ? 1 : 0));
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_DEBUG_LOGGING_OFFSET,
                (byte) (config.debugLogging() ? 1 : 0));

        // Write AutoEncoder-specific fields
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.AE_LATENT_DIM_SIZE_OFFSET,
                config.latentDimensionSize());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.AE_ENCODER_HIDDEN_NEURON_OFFSET,
                config.encoderHiddenNeuronCount());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.AE_ENCODER_LAYER_COUNT_OFFSET,
                config.encoderLayerCount());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.AE_DECODER_LAYER_COUNT_OFFSET,
                config.decoderLayerCount());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.AE_BOTTLENECK_REG_WEIGHT_OFFSET,
                config.bottleneckRegularisationWeight());

        return segment;
    }

    /**
     * Allocates and writes a Multi-Agent Group config to a MemorySegment with C-compatible layout.
     *
     * @param arena  the Arena for allocation
     * @param config the Java config record
     * @return MemorySegment containing the serialized config (aligned to 64 bytes)
     */
    public static @NotNull MemorySegment writeMultiAgentGroupConfig(
            @NotNull Arena arena,
            @NotNull SpartanMultiAgentGroupConfig config
    ) {
        MemorySegment segment = arena.allocate(
                simdPadBytes(SpartanConfigLayout.MULTI_AGENT_CONFIG_TOTAL_SIZE),
                SIMD_ALIGNMENT_BYTES);

        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_MODEL_TYPE_OFFSET,
                config.modelType().getNativeValue());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_LEARNING_RATE_OFFSET,
                config.learningRate());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_GAMMA_OFFSET,
                config.gamma());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_OFFSET,
                config.epsilon());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_MIN_OFFSET,
                config.epsilonMin());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.BASE_EPSILON_DECAY_OFFSET,
                config.epsilonDecay());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_STATE_SIZE_OFFSET, 0);
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_ACTION_SIZE_OFFSET, 0);
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_IS_TRAINING_OFFSET,
                (byte) (config.isTraining() ? 1 : 0));
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_DEBUG_LOGGING_OFFSET,
                (byte) (config.debugLogging() ? 1 : 0));

        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.MULTI_AGENT_MAX_AGENTS_OFFSET,
                config.maxAgents());

        return segment;
    }

    // ==================== Curiosity-Driven RSAC Weight Calculations ====================

    /**
     * Calculates the total weight count for the model (actor) weights in a Curiosity-Driven RSAC.
     * <p>
     * The Curiosity-Driven RSAC model weights are identical to the base RSAC model weights.
     * The Forward Dynamics network parameters are stored in the critic buffer.
     *
     * @param config the Curiosity-Driven RSAC configuration
     * @return total number of double weights for the model (actor + nested encoders)
     */
    public static long calculateCuriosityDrivenRecurrentSoftActorCriticModelWeightCount(
            @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig config,
            int stateSize,
            int actionSize
    ) {
        // Model weights are identical to base RSAC - delegate to existing method
        return calculateRSACModelWeightCount(config.recurrentSoftActorCriticConfig(), stateSize, actionSize);
    }

    /**
     * Calculates the total weight count for the critic buffer in a Curiosity-Driven RSAC.
     * <p>
     * The critic buffer contains:
     * <ul>
     *   <li>Base RSAC critic weights (GRU + twin Q-networks)</li>
     *   <li>Forward Dynamics Network parameters (appended and SIMD-padded)</li>
     * </ul>
     * <p>
     * Forward Dynamics Network architecture:
     * <pre>
     * Input: state + action (concatenated)
     * Hidden: single dense layer with forwardDynamicsHiddenLayerDimensionSize neurons
     * Output: predicted next state (stateSize)
     *
     * Weights: (inputSize * hiddenSize) + (hiddenSize * stateSize)
     * Biases: hiddenSize + stateSize
     *
     * Each module is SIMD-padded individually to ensure safe memory access.
     * </pre>
     *
     * @param config the Curiosity-Driven RSAC configuration
     * @return total number of double weights for the critic buffer (RSAC critics + forward dynamics, SIMD-padded)
     * @throws IllegalArgumentException if configuration parameters would cause arithmetic overflow
     */
    public static long calculateCuriosityDrivenRecurrentSoftActorCriticCriticWeightCount(
            @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig config,
            int stateSize,
            int actionSize
    ) {

        // Start with base RSAC critic weights (GRU + twin critics)
        long baseRsacCriticWeightsPadded = calculateRSACCriticWeightCount(
                config.recurrentSoftActorCriticConfig(),
                stateSize,
                actionSize);
        // Don't remove safety padding - it's already included in the total
        long baseRsacWeightsRaw = baseRsacCriticWeightsPadded;

        if (baseRsacWeightsRaw < 0L) {
            throw new IllegalArgumentException("Base RSAC critic weights calculation returned invalid value");
        }

        // forward dynamics network parameters
        long stateSizeLong = (long) stateSize;
        long actionSizeLong = (long) actionSize;
        long hiddenSize = config.forwardDynamicsHiddenLayerDimensionSize();

        if (stateSizeLong <= 0L || actionSizeLong < 0L || hiddenSize <= 0L) {
            throw new IllegalArgumentException(
                    "Invalid Forward Dynamics dimensions: state=" + stateSizeLong
                            + ", action=" + actionSizeLong + ", hidden=" + hiddenSize);
        }

        // structure of forward dynamics weights and biases (before padding):
        // Forward Dynamics: input (state + action) -> hidden -> output (state)
        long inputHiddenWeights = (stateSizeLong + actionSizeLong) * hiddenSize;
        long hiddenOutputWeights = hiddenSize * stateSizeLong;
        long biases = hiddenSize + stateSizeLong;
        long forwardDynamicsTotalRaw = inputHiddenWeights + hiddenOutputWeights + biases;

        // apply SIMD padding to the forward dynamics block
        long forwardDynamicsTotalPadded = simdPadElementCount(forwardDynamicsTotalRaw);

        // sum weights: base critics  + forward dynamics (con padding)
        long totalWeights = baseRsacWeightsRaw + forwardDynamicsTotalPadded;

        if (totalWeights < 0L) {
            throw new IllegalArgumentException("Total critic weight count would overflow: " + totalWeights);
        }

        return totalWeights;
    }

    // ==================== Curiosity-Driven RSAC Config Serialization ====================

    /**
     * Allocates and writes a Curiosity-Driven RSAC config to a MemorySegment with C-compatible layout.
     * <p>
     * Memory Layout (Strict C++ Standard Layout compliance):
     * <pre>
     * Offset 0:    RecurrentSoftActorCriticHyperparameterConfig (408 bytes)
     *              - Includes BaseHyperparameterConfig at offsets 0-64
     *              - Includes RSAC-specific fields at offsets 64-152
     *              - Includes Encoder Slots (5 * 16 = 80 bytes) at 152-232
     *              - Includes padding to reach 408 bytes
     * Offset 408:  forwardDynamicsHiddenLayerDimensionSize (int32_t)
     * Offset 412:  [4 bytes padding for alignment]
     * Offset 416:  intrinsicRewardScale (double)
     * Offset 424:  intrinsicRewardClampingMinimum (double)
     * Offset 432:  intrinsicRewardClampingMaximum (double)
     * Offset 440:  forwardDynamicsLearningRate (double)
     * Total: 448 bytes
     * </pre>
     *
     * @param arena  the Arena for allocation
     * @param config the Java config record
     * @return MemorySegment containing the serialized config (aligned to 64 bytes)
     */
    public static @NotNull MemorySegment writeCuriosityDrivenRecurrentSoftActorCriticConfig(
            @NotNull Arena arena,
            @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig config,
            int stateSize,
            int actionSize
    ) {

        // Allocate with 64-byte alignment for AVX-512 safety
        MemorySegment segment = arena.allocate(simdPadBytes(SpartanConfigLayout.CURIOSITY_RSAC_CONFIG_TOTAL_SIZE_PADDED), SIMD_ALIGNMENT_BYTES);

        // Use a ConfiningArena to ensure temporary RSAC segment stays valid during the copy operation
        try (Arena temporaryArena = Arena.ofConfined()) {
            // Serialize the RSAC config into a temporary segment
            // This generates 408 bytes
            MemorySegment rsacSegment = writeRSACConfig(temporaryArena, config.recurrentSoftActorCriticConfig(), stateSize, actionSize);

            // Copy the RSAC segment to the beginning of our shared-arena segment
            MemorySegment.copy(rsacSegment, 0, segment, 0, SpartanConfigLayout.RSAC_CONFIG_TOTAL_SIZE);
        }
        // Confined arena is closed here; temporary rsacSegment is deallocated

        // CRITICAL: Overwrite model ID to 4 (CURIOSITY_DRIVEN_RSAC)
        // The RSAC serializer wrote ID 3 (RSAC), but we need 4.
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_MODEL_TYPE_OFFSET,
                config.modelType().getNativeValue());

        // Write Curiosity-specific fields (offsets start at RSAC_CONFIG_TOTAL_SIZE)
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.CURIOSITY_RSAC_FORWARD_DYNAMICS_HIDDEN_SIZE_OFFSET,
                config.forwardDynamicsHiddenLayerDimensionSize());

        // Implicit padding (4 bytes) at offset 412 is zero-initialized by arena allocation

        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.CURIOSITY_RSAC_INTRINSIC_REWARD_SCALE_OFFSET,
                config.intrinsicRewardScale());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.CURIOSITY_RSAC_INTRINSIC_REWARD_CLAMPING_MINIMUM_OFFSET,
                config.intrinsicRewardClampingMinimum());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.CURIOSITY_RSAC_INTRINSIC_REWARD_CLAMPING_MAXIMUM_OFFSET,
                config.intrinsicRewardClampingMaximum());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.CURIOSITY_RSAC_FORWARD_DYNAMICS_LEARNING_RATE_OFFSET,
                config.forwardDynamicsLearningRate());

        return segment;
    }

    /**
     * Serializes any SpartanModelConfig into a C-compatible MemorySegment.
     * Dispatches to the specific write method based on the config type.
     *
     * @param arena  the Arena for allocation
     * @param config the configuration object
     * @return the serialized MemorySegment
     * @throws IllegalArgumentException if the config type is unknown
     */
    public static MemorySegment serialize(Arena arena, SpartanModelConfig config, int stateSize, int actionSize) {
        if (config instanceof CuriosityDrivenRecurrentSoftActorCriticConfig curiosity) {
            return writeCuriosityDrivenRecurrentSoftActorCriticConfig(arena, curiosity, stateSize, actionSize);
        } else if (config instanceof RecurrentSoftActorCriticConfig rsac) {
            return writeRSACConfig(arena, rsac, stateSize, actionSize);
        } else if (config instanceof DoubleDeepQNetworkConfig ddqn) {
            return writeDDQNConfig(arena, ddqn, stateSize, actionSize);
        } else if (config instanceof AutoEncoderCompressorConfig ae) {
            return writeAutoEncoderConfig(arena, ae, stateSize, actionSize);
        } else if (config instanceof SpartanMultiAgentGroupConfig groupConfig) {
            return writeMultiAgentGroupConfig(arena, groupConfig);
        }
        throw new IllegalArgumentException("Unknown config type: " + config.getClass().getName());
    }
}









