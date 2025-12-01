package dev.alepando.spartan.ai.dqn

import kotlin.math.sqrt
import java.util.Random

/**
 * Represents a **Dense (Fully-Connected) Layer** implementing the transformation `y = f(W · x + b)`.
 * @property neuronsIn Number of input neurons.
 * @property neuronsOut Number of output neurons.
 * @property activation Activation function applied to the output.
 */
class DenseLayer(
    val neuronsIn: Int,
    val neuronsOut: Int,
    val activation: Activation,
    private val random: Random = Random()
) {
    var weights: Array<DoubleArray>
    var biases: DoubleArray
    var lastOutput = DoubleArray(neuronsOut)

    init {
        // Advanced Initialization Logic
        // We define the scale (standard deviation) based on the activation function logic.
        val scale = if (activation == Activation.RELU) {
            sqrt(2.0 / neuronsIn) // He Initialization (Optimal for ReLU)
        } else {
            sqrt(1.0 / neuronsIn) // Xavier Initialization (Optimal for Sigmoid/Tanh/Linear)
        }

        weights = Array(neuronsOut) {
            DoubleArray(neuronsIn) {
                // Generate Gaussian random number (Mean=0, StdDev=scale)
                // nextGaussian() returns mean 0.0, sigma 1.0. Multiply by scale to adjust.
                random.nextGaussian() * scale
            }
        }
        biases = DoubleArray(neuronsOut) { 0.0 }
    }

    /**
     * Computes the forward pass of the layer.
     * ...
     */
    fun forward(input: DoubleArray): DoubleArray {
        if (input.size != neuronsIn) throw IllegalArgumentException("Input size mismatch")

        val output = DoubleArray(neuronsOut)

        // Matrix-Vector Multiplication: z = Wx + b
        // Optimized loop structure for cache locality
        for (i in 0 until neuronsOut) {
            var z = biases[i]
            val row = weights[i] // Local reference optimization
            for (j in 0 until neuronsIn) {
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
        val newLayer = DenseLayer(neuronsIn, neuronsOut, activation, Random()) // New independent random
        for (i in 0 until neuronsOut) {
            newLayer.weights[i] = this.weights[i].clone()
        }
        newLayer.biases = this.biases.clone()
        return newLayer
    }
}