package dev.alepando.spartan.ai.dqn.training

import dev.alepando.spartan.ai.dqn.Activation
import dev.alepando.spartan.ai.dqn.QNetwork

/**
 * Implements **Mini-Batch Stochastic Gradient Descent (SGD)**.
 *
 * Unlike naive SGD which updates weights after every sample, this optimizer accumulates
 * gradients over a batch of transitions and performs a single averaged update. This provides
 * a more accurate estimate of the gradient direction and allows for parallelization.
 *
 * **Update Rule (Averaged)**
 * `W ← W - α · (1/N) Σ ∇L_i`
 *
 * **Backpropagation Chain Rule Derivation**
 *
 * 1. **Output Layer (Linear):**
 * * Error term `δ_out`: `(Q(s, a) - target)`
 * * Gradient `∂L/∂W_out`: `δ_out · h_j`
 *
 * 2. **Hidden Layer:**
 * * Propagated Error `δ_h`: `(Σ δ_out · W_out) · f'(z_h)`
 * * Gradient `∂L/∂W_h`: `δ_h · s_j`
 *
 * **Stability Features**
 * * **Gradient Clipping:** Gradients are clamped to `[-clipThreshold, clipThreshold]` per sample
 * to prevent the "Exploding Gradient" problem.
 */
class Optimizer(
    private val network: QNetwork,
    private val learningRate: Double,
    private val clipThreshold: Double = 1.0
) {
    // Accumulators for Mini-Batch averaging (Zeroed at initialization)
    private val accOutputWeights = Array(network.output.neuronsOut) { DoubleArray(network.output.neuronsIn) }
    private val accOutputBiases = DoubleArray(network.output.neuronsOut)

    private val accHiddenWeights = Array(network.hidden.neuronsOut) { DoubleArray(network.hidden.neuronsIn) }
    private val accHiddenBiases = DoubleArray(network.hidden.neuronsOut)

    /**
     * Calculates gradients for a single sample and adds them to the accumulators.
     * **Does NOT modify network weights yet.**
     *
     * @param state The normalized input observation vector `s`.
     * @param actionIndex The specific output neuron index `a`.
     * @param target The calculated TD-Target (Label).
     */
    fun accumulateGradients(state: DoubleArray, actionIndex: Int, target: Double) {
        val qValues = network.predict(state)

        // Calculate TD Error (Clipped for stability)
        val error = qValues[actionIndex] - target.coerceIn(-1.0, 1.0)

        val outputLayer = network.output
        val hiddenLayer = network.hidden

        // -------------------------------------------------------
        // Backward Pass: Output Layer Gradients
        // -------------------------------------------------------
        val outputGradients = DoubleArray(outputLayer.neuronsOut)
        outputGradients[actionIndex] = error.coerceIn(-clipThreshold, clipThreshold)

        // Accumulate Output Gradients (dL/dW and dL/db)
        for (i in 0 until outputLayer.neuronsOut) {
            val grad = outputGradients[i]
            if (grad != 0.0) { // Optimization: skip if gradient is zero
                for (j in 0 until outputLayer.neuronsIn) {
                    accOutputWeights[i][j] += grad * hiddenLayer.lastOutput[j]
                }
                accOutputBiases[i] += grad
            }
        }

        // -------------------------------------------------------
        // Backward Pass: Hidden Layer Gradients
        // -------------------------------------------------------
        val hiddenGradients = DoubleArray(hiddenLayer.neuronsOut)

        // Propagate error backwards: δ_h = (W_out^T · δ_out) ⊙ f'(z)
        for (k in 0 until hiddenLayer.neuronsOut) {
            var sum = 0.0
            for (i in 0 until outputLayer.neuronsOut) {
                sum += outputLayer.weights[i][k] * outputGradients[i]
            }

            val deriv = when (hiddenLayer.activation) {
                Activation.RELU -> if (hiddenLayer.lastOutput[k] > 0) 1.0 else 0.0
                Activation.SIGMOID -> hiddenLayer.activation.derivative(hiddenLayer.lastOutput[k])
                Activation.TANH -> hiddenLayer.activation.derivative(hiddenLayer.lastOutput[k])
                Activation.LINEAR -> 1.0
            }

            hiddenGradients[k] = (sum * deriv).coerceIn(-clipThreshold, clipThreshold)
        }

        // Accumulate Hidden Gradients
        for (k in 0 until hiddenLayer.neuronsOut) {
            val grad = hiddenGradients[k]
            if (grad != 0.0) {
                for (j in 0 until hiddenLayer.neuronsIn) {
                    accHiddenWeights[k][j] += grad * state[j]
                }
                accHiddenBiases[k] += grad
            }
        }
    }

    /**
     * Applies the average of accumulated gradients to the network weights and resets accumulators.
     *
     * **Formula:** `W_new = W_old - LearningRate * (SumGrads / BatchSize)`
     *
     * @param batchSize The number of samples accumulated (divisor for averaging).
     */
    fun applyGradients(batchSize: Int) {
        if (batchSize == 0) return
        val outputLayer = network.output
        val hiddenLayer = network.hidden
        val scale = learningRate / batchSize // Pre-calculate scaling factor

        // Apply Output Layer Updates
        for (i in 0 until outputLayer.neuronsOut) {
            for (j in 0 until outputLayer.neuronsIn) {
                outputLayer.weights[i][j] -= accOutputWeights[i][j] * scale
                accOutputWeights[i][j] = 0.0 // Reset immediately
            }
            outputLayer.biases[i] -= accOutputBiases[i] * scale
            accOutputBiases[i] = 0.0 // Reset immediately
        }

        // Apply Hidden Layer Updates
        for (k in 0 until hiddenLayer.neuronsOut) {
            for (j in 0 until hiddenLayer.neuronsIn) {
                hiddenLayer.weights[k][j] -= accHiddenWeights[k][j] * scale
                accHiddenWeights[k][j] = 0.0 // Reset immediately
            }
            hiddenLayer.biases[k] -= accHiddenBiases[k] * scale
            accHiddenBiases[k] = 0.0 // Reset immediately
        }
    }
}