package org.spartan.internal.model;

import org.jetbrains.annotations.NotNull;
import org.jspecify.annotations.NonNull;
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
}
