package org.spartan.internal.model;

import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;
import org.spartan.api.agent.config.AutoEncoderCompressorConfig;
import org.spartan.api.agent.config.DoubleDeepQNetworkConfig;
import org.spartan.api.agent.config.NestedEncoderSlotDescriptor;
import org.spartan.api.agent.config.RecurrentSoftActorCriticConfig;
import org.spartan.internal.util.SpartanMemoryUtil;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Utility class responsible for allocating C-compatible memory buffers
 * required by the Spartan native engine for model weights and parameters.
 * <p>
 * All allocations are performed using Project Panama's Arena API to ensure
 * proper alignment and lifecycle management. Buffers are contiguous and
 * suitable for direct access by C++26 code via FFM.
 * <p>
 * Memory Layout Strategy:
 * <ul>
 *   <li>All weight buffers use JAVA_DOUBLE (8 bytes) with natural alignment</li>
 *   <li>Count/size parameters use JAVA_INT (4 bytes)</li>
 *   <li>Buffers are zero-initialized by the Arena</li>
 * </ul>
 */
public final class SpartanModelAllocator {

    private static final ValueLayout.OfDouble DOUBLE_LAYOUT = ValueLayout.JAVA_DOUBLE;
    private static final ValueLayout.OfInt INT_LAYOUT = ValueLayout.JAVA_INT;

    private SpartanModelAllocator() {} // Utility class

    // RSAC (Recurrent Soft Actor-Critic)

    /**
     * Allocates weight buffers for an RSAC (Recurrent Soft Actor-Critic) model.
     * <p>
     * RSAC Architecture Memory Layout:
     * <pre>
     * Actor Network:
     *   - Input Layer:  contextSize × hiddenSize weights
     *   - Hidden Layers: (numHiddenLayers - 1) × hiddenSize × hiddenSize weights
     *   - Output Layer: hiddenSize × actionSize weights
     *   - Biases: numHiddenLayers × hiddenSize + actionSize
     *   - GRU Cell: 3 × hiddenSize × hiddenSize (for recurrent state)
     *
     * Total = inputWeights + hiddenWeights + outputWeights + biases + gruWeights
     * </pre>
     *
     * @param arena the Arena for allocation (controls memory lifecycle)
     * @param contextSize number of input features (from SpartanContext)
     * @param hiddenSize neurons per hidden layer
     * @param numHiddenLayers number of hidden layers
     * @param actionSize number of output actions
     * @return allocated weight buffer with proper alignment
     */
    public static MemorySegment allocateRSACWeights(
            @NotNull Arena arena,
            int contextSize,
            int hiddenSize,
            int numHiddenLayers,
            int actionSize
    ) {
        // Calculate total weight count for actor network
        long inputWeights = (long) contextSize * hiddenSize;
        long hiddenWeights = (long) (numHiddenLayers - 1) * hiddenSize * hiddenSize;
        long outputWeights = (long) hiddenSize * actionSize;
        long biases = (long) numHiddenLayers * hiddenSize + actionSize;

        // GRU cell: 3 gates (reset, update, new) each with hiddenSize × hiddenSize
        long gruWeights = 3L * hiddenSize * hiddenSize;

        long totalWeights = inputWeights + hiddenWeights + outputWeights + biases + gruWeights;

        return arena.allocate(DOUBLE_LAYOUT, totalWeights);
    }

    /**
     * Allocates critic weight buffers for RSAC (twin Q-networks).
     * <p>
     * RSAC uses twin critics to reduce overestimation bias.
     * Each critic takes (context + action) as input.
     *
     * @param arena the Arena for allocation
     * @param contextSize number of context features
     * @param actionSize number of action dimensions
     * @param hiddenSize neurons per hidden layer
     * @param numHiddenLayers number of hidden layers
     * @return allocated critic weight buffer (for both Q1 and Q2)
     */
    public static MemorySegment allocateRSACCriticWeights(
            @NotNull Arena arena,
            int contextSize,
            int actionSize,
            int hiddenSize,
            int numHiddenLayers
    ) {
        int criticInputSize = contextSize + actionSize;

        // Single critic network
        long inputWeights = (long) criticInputSize * hiddenSize;
        long hiddenWeights = (long) (numHiddenLayers - 1) * hiddenSize * hiddenSize;
        long outputWeights = hiddenSize; // Q-value is scalar
        long biases = (long) numHiddenLayers * hiddenSize + 1;

        long singleCriticWeights = inputWeights + hiddenWeights + outputWeights + biases;

        // Twin critics (Q1 and Q2)
        long totalWeights = singleCriticWeights * 2;

        return arena.allocate(DOUBLE_LAYOUT, totalWeights);
    }

    //  DDQN (Double Deep Q-Network)

    /**
     * Allocates weight buffers for a DDQN (Double Deep Q-Network) model.
     * <p>
     * DDQN Architecture Memory Layout:
     * <pre>
     * Main Network + Target Network (identical structure):
     *   - Input Layer:  contextSize × hiddenSize weights
     *   - Hidden Layers: (numHiddenLayers - 1) × hiddenSize × hiddenSize weights
     *   - Output Layer: hiddenSize × numActions weights (Q-value per action)
     *   - Biases: numHiddenLayers × hiddenSize + numActions
     *
     * Total = 2 × (inputWeights + hiddenWeights + outputWeights + biases)
     * </pre>
     *
     * @param arena the Arena for allocation
     * @param contextSize number of input features
     * @param hiddenSize neurons per hidden layer
     * @param numHiddenLayers number of hidden layers
     * @param numActions number of discrete actions
     * @return allocated weight buffer for both main and target networks
     */
    public static MemorySegment allocateDDQNWeights(
            @NotNull Arena arena,
            int contextSize,
            int hiddenSize,
            int numHiddenLayers,
            int numActions
    ) {
        // Single network weight count
        long inputWeights = (long) contextSize * hiddenSize;
        long hiddenWeights = (long) (numHiddenLayers - 1) * hiddenSize * hiddenSize;
        long outputWeights = (long) hiddenSize * numActions;
        long biases = (long) numHiddenLayers * hiddenSize + numActions;

        long singleNetworkWeights = inputWeights + hiddenWeights + outputWeights + biases;

        // Main network + Target network
        long totalWeights = singleNetworkWeights * 2;

        return arena.allocate(DOUBLE_LAYOUT, totalWeights);
    }

    // PPO (Proximal Policy Optimization)

    /**
     * Allocates weight buffers for a PPO model.
     * <p>
     * PPO Architecture Memory Layout:
     * <pre>
     * Actor Network (Policy):
     *   - Input → Hidden → Output (action logits or mean/std for continuous)
     *
     * Critic Network (Value):
     *   - Input → Hidden → Scalar value output
     * </pre>
     *
     * @param arena the Arena for allocation
     * @param contextSize number of input features
     * @param hiddenSize neurons per hidden layer
     * @param numHiddenLayers number of hidden layers
     * @param actionSize number of actions (discrete) or action dimensions (continuous)
     * @param continuous true if continuous action space (allocates mean + std outputs)
     * @return allocated weight buffer for actor + critic
     */
    public static MemorySegment allocatePPOWeights(
            @NotNull Arena arena,
            int contextSize,
            int hiddenSize,
            int numHiddenLayers,
            int actionSize,
            boolean continuous
    ) {
        // Actor network
        long actorInputWeights = (long) contextSize * hiddenSize;
        long actorHiddenWeights = (long) (numHiddenLayers - 1) * hiddenSize * hiddenSize;
        // Continuous: outputs mean and log_std for each action dimension
        int actorOutputSize = continuous ? actionSize * 2 : actionSize;
        long actorOutputWeights = (long) hiddenSize * actorOutputSize;
        long actorBiases = (long) numHiddenLayers * hiddenSize + actorOutputSize;

        long actorWeights = actorInputWeights + actorHiddenWeights + actorOutputWeights + actorBiases;

        // Critic network (same structure, scalar output)
        long criticInputWeights = (long) contextSize * hiddenSize;
        long criticHiddenWeights = (long) (numHiddenLayers - 1) * hiddenSize * hiddenSize;
        long criticOutputWeights = hiddenSize; // Single value output
        long criticBiases = (long) numHiddenLayers * hiddenSize + 1;

        long criticWeights = criticInputWeights + criticHiddenWeights + criticOutputWeights + criticBiases;

        return arena.allocate(DOUBLE_LAYOUT, actorWeights + criticWeights);
    }

    //  Generic Allocation
    /**
     * Allocates a generic weight buffer with explicit count.
     * Use when you have pre-calculated the exact weight count.
     *
     * @param arena the Arena for allocation
     * @param weightCount total number of double weights
     * @return allocated weight buffer
     */
    public static MemorySegment allocateWeights(@NotNull Arena arena, long weightCount) {
        return arena.allocate(DOUBLE_LAYOUT, weightCount);
    }

    /**
     * Allocates a buffer for context data (doubles).
     *
     * @param arena the Arena for allocation
     * @param contextSize number of context elements
     * @return allocated context buffer
     */
    public static MemorySegment allocateContextBuffer(@NotNull Arena arena, int contextSize) {
        return arena.allocate(DOUBLE_LAYOUT, contextSize);
    }

    /**
     * Allocates a buffer for action outputs.
     *
     * @param arena the Arena for allocation
     * @param actionSize number of action dimensions
     * @return allocated action buffer
     */
    public static MemorySegment allocateActionBuffer(@NotNull Arena arena, int actionSize) {
        return arena.allocate(DOUBLE_LAYOUT, actionSize);
    }

    /**
     * Allocates a segment for a single int value (e.g., count parameters).
     *
     * @param arena the Arena for allocation
     * @param value initial value
     * @return allocated segment
     */
    public static @NonNull MemorySegment allocateIntScalar(@NotNull Arena arena, int value) {
        MemorySegment segment = arena.allocate(INT_LAYOUT);
        segment.set(INT_LAYOUT, 0, value);
        return segment;
    }

    /**
     * Allocates a segment for a single long value (e.g., agent identifiers).
     *
     * @param arena the Arena for allocation
     * @param value initial value
     * @return allocated segment
     */
    public static @NonNull MemorySegment allocateLongScalar(@NotNull Arena arena, long value) {
        return SpartanMemoryUtil.allocateLong(arena, value);
    }

    // ==================== Topology Calculations ====================

    /**
     * Calculates the total weight count for a fully-connected MLP.
     *
     * @param inputSize input layer size
     * @param hiddenSize hidden layer size
     * @param numHiddenLayers number of hidden layers
     * @param outputSize output layer size
     * @return total number of weights (including biases)
     */
    public static long calculateMLPWeightCount(int inputSize, int hiddenSize, int numHiddenLayers, int outputSize) {
        long inputWeights = (long) inputSize * hiddenSize;
        long hiddenWeights = (long) (numHiddenLayers - 1) * hiddenSize * hiddenSize;
        long outputWeights = (long) hiddenSize * outputSize;
        long biases = (long) numHiddenLayers * hiddenSize + outputSize;

        return inputWeights + hiddenWeights + outputWeights + biases;
    }

    // ==================== RSAC Config Serialization ====================

    /**
     * Allocates and writes an RSAC config to a MemorySegment with C-compatible layout.
     * <p>
     * The resulting segment can be passed directly to C++ as a void* pointer.
     * The layout matches {@code RecurrentSoftActorCriticHyperparameterConfig} exactly.
     *
     * @param arena  the Arena for allocation
     * @param config the Java config record
     * @return MemorySegment containing the serialized config (408 bytes)
     */
    public static @NonNull MemorySegment writeRSACConfig(
            @NotNull Arena arena,
            @NotNull RecurrentSoftActorCriticConfig config
    ) {
        // Allocate with 8-byte alignment for double fields
        MemorySegment segment = arena.allocate(SpartanConfigLayout.RSAC_CONFIG_TOTAL_SIZE, 8);

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
                config.stateSize());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_ACTION_SIZE_OFFSET,
                config.actionSize());
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_IS_TRAINING_OFFSET,
                (byte) (config.isTraining() ? 1 : 0));

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
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_RECURRENT_INPUT_FEATURE_COUNT_OFFSET,
                config.recurrentInputFeatureCount());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_NESTED_ENCODER_COUNT_OFFSET,
                config.nestedEncoderCount());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.RSAC_REMORSE_BUFFER_CAPACITY_OFFSET,
                config.remorseTraceBufferCapacity());
        segment.set(ValueLayout.JAVA_DOUBLE, SpartanConfigLayout.RSAC_REMORSE_SIMILARITY_THRESHOLD_OFFSET,
                config.remorseMinimumSimilarityThreshold());

        // Write encoder slot descriptors (up to MAX_NESTED_ENCODER_SLOTS)
        int encoderCount = config.nestedEncoderCount();
        long slotsBase = SpartanConfigLayout.RSAC_ENCODER_SLOTS_OFFSET;

        for (int i = 0; i < encoderCount && i < SpartanConfigLayout.MAX_NESTED_ENCODER_SLOTS; i++) {
            NestedEncoderSlotDescriptor slot = config.encoderSlot(i);
            long slotOffset = slotsBase + (i * SpartanConfigLayout.SLOT_DESCRIPTOR_SIZE);

            segment.set(ValueLayout.JAVA_INT, slotOffset + SpartanConfigLayout.SLOT_START_INDEX_OFFSET,
                    slot.contextSliceStartIndex());
            segment.set(ValueLayout.JAVA_INT, slotOffset + SpartanConfigLayout.SLOT_ELEMENT_COUNT_OFFSET,
                    slot.contextSliceElementCount());
            segment.set(ValueLayout.JAVA_INT, slotOffset + SpartanConfigLayout.SLOT_LATENT_DIM_OFFSET,
                    slot.latentDimensionSize());
            segment.set(ValueLayout.JAVA_INT, slotOffset + SpartanConfigLayout.SLOT_HIDDEN_COUNT_OFFSET,
                    slot.hiddenNeuronCount());
        }

        // Remaining slots (if any) are left as zero-initialized by Arena

        return segment;
    }

    /**
     * Calculates the total weight count for an RSAC model's actor network.
     * <p>
     * Includes:
     * <ul>
     *   <li>Actor MLP: GRU_output → actor hidden layers → output (mean + log_std for each action)</li>
     *   <li>Nested AutoEncoders: encoder + decoder weights for each slot</li>
     * </ul>
     * <p>
     * NOTE: GRU weights are stored in the CRITIC buffer, not here. See {@link #calculateRSACCriticWeightCount}.
     *
     * @param config the RSAC configuration
     * @return total number of double weights for the model (actor + nested encoders)
     */
    public static long calculateRSACModelWeightCount(@NotNull RecurrentSoftActorCriticConfig config) {
        // Actor Network (post-GRU)
        // Input: GRU hidden state output (hiddenStateSize)
        // First layer: gruHiddenSize -> actorHiddenLayerNeuronCount
        int gruHiddenSize = config.hiddenStateSize();
        int actorHiddenSize = config.actorHiddenLayerNeuronCount();
        int actionSize = config.actionSize();

        // Policy layer 1: GRU output -> actor hidden
        long policyLayer1Weights = (long) actorHiddenSize * gruHiddenSize;

        // Additional hidden layers (if actorHiddenLayerCount > 1)
        long actorHiddenLayers = (long) Math.max(0, config.actorHiddenLayerCount() - 1)
                * actorHiddenSize * actorHiddenSize;

        // Output layers: mean + log_std for continuous actions
        long policyMeanWeights = (long) actionSize * actorHiddenSize;
        long policyLogStdWeights = (long) actionSize * actorHiddenSize;

        // Biases: actor hidden layers + mean output + log_std output
        long actorBiases = (long) config.actorHiddenLayerCount() * actorHiddenSize
                + actionSize + actionSize;

        long actorTotal = policyLayer1Weights + actorHiddenLayers
                + policyMeanWeights + policyLogStdWeights + actorBiases;

        // Nested AutoEncoders (if any)
        long nestedEncoderTotal = 0;
        int encoderCount = config.nestedEncoderCount();
        for (int i = 0; i < encoderCount; i++) {
            NestedEncoderSlotDescriptor slot = config.encoderSlot(i);
            nestedEncoderTotal += calculateAutoEncoderWeightCount(
                    slot.contextSliceElementCount(),
                    slot.hiddenNeuronCount(),
                    slot.latentDimensionSize()
            );
        }

        return actorTotal + nestedEncoderTotal;
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
     * NOTE: Target networks are handled separately by C++ (soft updates from online networks).
     *
     * @param config the RSAC configuration
     * @return total number of double weights for GRU + critics
     */
    public static long calculateRSACCriticWeightCount(@NotNull RecurrentSoftActorCriticConfig config) {
        int gruHiddenSize = config.hiddenStateSize();
        int gruInputSize = config.recurrentInputFeatureCount() > 0
                ? config.recurrentInputFeatureCount()
                : config.stateSize();
        int gruConcatSize = gruHiddenSize + gruInputSize;

        // GRU: 3 gates (reset, update, candidate)
        long gruGateWeights = 3L * gruHiddenSize * gruConcatSize;
        long gruGateBiases = 3L * gruHiddenSize;
        long gruHiddenState = gruHiddenSize;
        long gruTotal = gruGateWeights + gruGateBiases + gruHiddenState;

        // Critic input: GRU hidden state + action vector
        int criticInput = gruHiddenSize + config.actionSize();
        int criticHiddenSize = config.criticHiddenLayerNeuronCount();

        // Single critic network: input -> hidden -> scalar Q-value
        // Weights: (input × hidden) + (hidden × 1 for output)
        // Biases: hidden + 1 (for output)
        long criticWeightsPerNetwork = (long) criticHiddenSize * criticInput + criticHiddenSize;
        long criticBiasesPerNetwork = criticHiddenSize + 1L;
        long singleCriticTotal = criticWeightsPerNetwork + criticBiasesPerNetwork;

        // Twin critics: Q1 + Q2
        long twinCriticsTotal = singleCriticTotal * 2;

        return gruTotal + twinCriticsTotal;
    }

    /**
     * Calculates the weight count for an AutoEncoder (encoder + decoder).
     *
     * @param inputSize      input/output dimensionality
     * @param hiddenNeurons  neurons in hidden layer
     * @param latentSize     bottleneck dimensionality
     * @return total number of weights
     */
    public static long calculateAutoEncoderWeightCount(int inputSize, int hiddenNeurons, int latentSize) {
        // Encoder: input -> hidden -> latent
        long encoderInputLayer = (long) inputSize * hiddenNeurons;
        long encoderLatentLayer = (long) hiddenNeurons * latentSize;
        long encoderBiases = hiddenNeurons + latentSize;

        // Decoder: latent -> hidden -> output
        long decoderHiddenLayer = (long) latentSize * hiddenNeurons;
        long decoderOutputLayer = (long) hiddenNeurons * inputSize;
        long decoderBiases = hiddenNeurons + inputSize;

        return encoderInputLayer + encoderLatentLayer + encoderBiases
                + decoderHiddenLayer + decoderOutputLayer + decoderBiases;
    }

    // ==================== DDQN Weight Calculations ====================

    /**
     * Calculates the total weight count for a DDQN model.
     * <p>
     * DDQN uses two networks: online and target (identical structure).
     * Each network is an MLP: state -> hidden layers -> Q-values per action.
     *
     * @param config the DDQN configuration
     * @return total number of double weights for both networks
     */
    public static long calculateDDQNModelWeightCount(@NotNull DoubleDeepQNetworkConfig config) {
        // Single Q-network: MLP from state to Q-values
        long inputLayer = (long) config.stateSize() * config.hiddenLayerNeuronCount();
        long hiddenLayers = (long) Math.max(0, config.hiddenLayerCount() - 1)
                * config.hiddenLayerNeuronCount()
                * config.hiddenLayerNeuronCount();
        long outputLayer = (long) config.hiddenLayerNeuronCount() * config.actionSize();
        long biases = (long) config.hiddenLayerCount() * config.hiddenLayerNeuronCount()
                + config.actionSize();

        long singleNetworkWeights = inputLayer + hiddenLayers + outputLayer + biases;

        // Online network + Target network
        return singleNetworkWeights * 2;
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
     *
     * @param arena  the Arena for allocation
     * @param config the Java config record
     * @return MemorySegment containing the serialized config (88 bytes)
     */
    public static @NonNull MemorySegment writeDDQNConfig(
            @NotNull Arena arena,
            @NotNull DoubleDeepQNetworkConfig config
    ) {
        MemorySegment segment = arena.allocate(SpartanConfigLayout.DDQN_CONFIG_TOTAL_SIZE, 8);

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
                config.stateSize());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_ACTION_SIZE_OFFSET,
                config.actionSize());
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_IS_TRAINING_OFFSET,
                (byte) (config.isTraining() ? 1 : 0));

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
     * Calculates the total weight count for an AutoEncoder Compressor model.
     * <p>
     * Architecture: input -> encoder hidden layers -> latent -> decoder hidden layers -> output
     *
     * @param config the AutoEncoder configuration
     * @return total number of double weights
     */
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
     * </pre>
     *
     * @param config the AutoEncoder configuration
     * @return total number of double weights
     */
    public static long calculateAutoEncoderModelWeightCount(@NotNull AutoEncoderCompressorConfig config) {
        int stateSize = config.stateSize();
        int hiddenSize = config.encoderHiddenNeuronCount();
        int latentSize = config.latentDimensionSize();

        // Encoder: 2 layers (input->hidden, hidden->latent)
        long encoderLayer1Weights = (long) hiddenSize * stateSize;
        long encoderLayer2Weights = (long) latentSize * hiddenSize;
        long encoderWeightCount = encoderLayer1Weights + encoderLayer2Weights;
        long encoderBiasCount = (long) hiddenSize + latentSize;

        // Decoder: 2 layers (latent->hidden, hidden->output)
        long decoderLayer1Weights = (long) hiddenSize * latentSize;
        long decoderLayer2Weights = (long) stateSize * hiddenSize;
        long decoderWeightCount = decoderLayer1Weights + decoderLayer2Weights;
        long decoderBiasCount = (long) hiddenSize + stateSize;

        // Latent buffer space
        long latentBufferCount = latentSize;

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
     *
     * @param arena  the Arena for allocation
     * @param config the Java config record
     * @return MemorySegment containing the serialized config (88 bytes)
     */
    public static @NonNull MemorySegment writeAutoEncoderConfig(
            @NotNull Arena arena,
            @NotNull AutoEncoderCompressorConfig config
    ) {
        MemorySegment segment = arena.allocate(SpartanConfigLayout.AE_CONFIG_TOTAL_SIZE, 8);

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
                config.stateSize());
        segment.set(ValueLayout.JAVA_INT, SpartanConfigLayout.BASE_ACTION_SIZE_OFFSET,
                config.actionSize());
        segment.set(ValueLayout.JAVA_BYTE, SpartanConfigLayout.BASE_IS_TRAINING_OFFSET,
                (byte) (config.isTraining() ? 1 : 0));

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
}
