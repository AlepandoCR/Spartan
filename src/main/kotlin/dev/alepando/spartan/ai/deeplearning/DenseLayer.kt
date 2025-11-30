package dev.alepando.spartan.ai.deeplearning

import kotlin.math.sqrt
import java.util.Random

/**
 * Represents a **Dense (Fully-Connected) Layer** implementing the transformation `y = f(W · x + b)`.
 * ... (KDoc preservado) ...
 */
class DenseLayer(
    val inputSize: Int,
    val outputSize: Int,
    val activation: Activation = Activation.RELU,
    private val random: Random = Random()
) {
    var weights: Array<DoubleArray>
    var biases: DoubleArray
    var lastOutput = DoubleArray(outputSize)

    init {
        // Advanced Initialization Logic
        // We define the scale (standard deviation) based on the activation function logic.
        val scale = if (activation == Activation.RELU) {
            sqrt(2.0 / inputSize) // He Initialization (Optimal for ReLU)
        } else {
            sqrt(1.0 / inputSize) // Xavier Initialization (Optimal for Sigmoid/Tanh/Linear)
        }

        weights = Array(outputSize) {
            DoubleArray(inputSize) {
                // Generate Gaussian random number (Mean=0, StdDev=scale)
                // nextGaussian() returns mean 0.0, sigma 1.0. Multiply by scale to adjust.
                random.nextGaussian() * scale
            }
        }
        biases = DoubleArray(outputSize) { 0.0 }
    }

    /**
     * Computes the forward pass of the layer.
     * ...
     */
    fun forward(input: DoubleArray): DoubleArray {
        if (input.size != inputSize) throw IllegalArgumentException("Input size mismatch")

        val output = DoubleArray(outputSize)

        // Matrix-Vector Multiplication: z = Wx + b
        // Optimized loop structure for cache locality
        for (i in 0 until outputSize) {
            var z = biases[i]
            val row = weights[i] // Local reference optimization
            for (j in 0 until inputSize) {
                z += row[j] * input[j]
            }
            output[i] = activation.fn(z)
        }

        lastOutput = output
        return output
    }

    /**
     * Creates a deep copy of the layer.
     * Note: We create a new Random() for the clone to avoid state coupling,
     * but strictly speaking, clones are for inference/target, not re-initialization.
     */
    fun clone(): DenseLayer {
        val newLayer = DenseLayer(inputSize, outputSize, activation, Random()) // New independent random
        for (i in 0 until outputSize) {
            newLayer.weights[i] = this.weights[i].clone()
        }
        newLayer.biases = this.biases.clone()
        return newLayer
    }
}