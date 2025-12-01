package dev.alepando.spartan.ai.dqn

import ModelStore
import dev.alepando.spartan.ai.dqn.models.ModelType
import dev.alepando.spartan.ai.input.QAction
import dev.alepando.spartan.database.data.DqnDto
import org.bukkit.Bukkit

/**
 * Implements a two-layer Feed-Forward Neural Network (MLP) to approximate the Action-Value function `Q(s, a)`.
 *
 * **Network Architecture**
 *
 * Input Layer -> Hidden Layer (64 neurons, ReLU) -> Output Layer (Linear)
 *
 * **Mathematical Formulation**
 * * **Hidden Projection:** `h = σ(W_h · s + b_h)`
 * * **Output Projection:** `Q(s, ·) = W_out · h + b_out`
 *
 * > **Note:** The output layer uses an implicit Identity activation (Linear) to allow unbounded regression values for Q.
 *
 * @see Activation
 */
class QNetwork(
    /** Input feature vector size (must match observation size). */
    val neuronsIn: Int,
    /** Action space available to this network. */
    val actions: List<QAction>,

    val modelType: ModelType,
) {
    private val neuronsOut = actions.size

    // Internal Layers using the optimized DenseLayer implementation
    // Hidden Layer
    val hidden = DenseLayer(neuronsIn, neuronsOut, activation = Activation.RELU)
    // Output Layer: Maps hidden features to Q-values for each action
    val output = DenseLayer(neuronsIn, neuronsOut, activation = Activation.LINEAR)

    /**
     * Computes Q(s,·) for the given state using a forward pass.
     *
     * @param state Normalized observation vector `s`.
     * @return Array of Q-values where index `i` corresponds to `actions[i]`.
     */
    fun predict(state: DoubleArray): DoubleArray {
        // Forward propagation: Input -> Hidden (ReLU) -> Output (Linear)
        val hiddenOut = hidden.forward(state)
        return output.forward(hiddenOut)
    }

    /**
     * Selects the action with the maximum predicted Q-value (Greedy Policy).
     * `π(s) = argmax_a Q(s, a)`
     *
     * @param state Normalized observation vector.
     * @return The `QAction` instance with the highest expected reward.
     */
    fun getBestAction(state: DoubleArray): QAction {
        val q = predict(state)
        // Find index of max value safely
        val idx = q.indices.maxByOrNull { q[it] } ?: 0
        return actions[idx]
    }

    /**
     * Resolves the index of a specific action object.
     * Useful for mapping training data actions back to neuron indices.
     *
     * @param action The `QAction` instance to locate.
     * @return The index of the action in the output layer, or -1 if not found.
     */
    fun getActionIndex(action: QAction): Int = actions.indexOf(action)

    /**
     * Creates a deep copy of this network.
     * Essential for creating the **Target Network** in DQN to stabilize training.
     */
    fun copy(): QNetwork {
        val network = QNetwork(neuronsIn, actions, modelType)
        // Manually clone weights to ensure deep copy (avoid reference sharing)
        network.hidden.weights = hidden.weights.map { it.clone() }.toTypedArray()
        network.hidden.biases = hidden.biases.clone()
        network.output.weights = output.weights.map { it.clone() }.toTypedArray()
        network.output.biases = output.biases.clone()
        return network
    }

    /**
     * Persists parameters and performance metrics using the provided storage abstraction.
     * Checks for numerical stability (Finite values) before saving to prevent corruption.
     *
     * @param modelType Identifier key used in storage (e.g., hash).
     * @param store The persistence layer.
     * @param performance Scalar metric (e.g., average reward) to track model quality.
     */
    fun save(store: ModelStore, performance: Double) {
        // Safety Check: Ensure no NaN or Infinite weights exist
        val hiddenWeightsFinite = hidden.weights.all { row -> row.all { it.isFinite() } }
        val hiddenBiasesFinite = hidden.biases.all { it.isFinite() }
        val outputWeightsFinite = output.weights.all { row -> row.all { it.isFinite() } }
        val outputBiasesFinite = output.biases.all { it.isFinite() }

        if (!hiddenWeightsFinite || !hiddenBiasesFinite || !outputWeightsFinite || !outputBiasesFinite) {
            Bukkit.getLogger().warning("Attempted to save a model with non-finite weights or biases. Aborting save.")
            return
        }

        val dto = toDto(performance)

        store.save(dto)
    }

    private fun toDto(
        performance: Double
    ): DqnDto {
        val dto = DqnDto(
            hash = modelType.hash,
            inputSize = neuronsIn,
            hiddenWeights = hidden.weights.map { it.toList() },
            hiddenBiases = hidden.biases.toList(),
            outputWeights = output.weights.map { it.toList() },
            outputBiases = output.biases.toList(),
            performance = performance
        )
        return dto
    }

    companion object {
        /**
         * Reconstructs a network from storage and binds it to the provided action space.
         *
         * @param modelType Identifier key used in storage.
         * @param store Persistence abstraction.
         * @param actions Action list this network will serve (must match saved topology).
         * @return The loaded `QNetwork` with restored weights, or `null` if not found.
         */
        fun load(modelType: ModelType, store: ModelStore, actions: List<QAction>): QNetwork? {
            val snapshot = store.load(modelType.hash) ?: return null

            // Reconstruct topology
            val network = QNetwork(snapshot.inputSize, actions, modelType)

            // Restore weights and biases
            // Note: Assumes standard List<List<Double>> to Array<DoubleArray> conversion
            network.hidden.weights = snapshot.hiddenWeights.map { it.toDoubleArray() }.toTypedArray()
            network.hidden.biases = snapshot.hiddenBiases.toDoubleArray()
            network.output.weights = snapshot.outputWeights.map { it.toDoubleArray() }.toTypedArray()
            network.output.biases = snapshot.outputBiases.toDoubleArray()

            return network
        }
    }
}