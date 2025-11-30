package dev.alepando.spartan.ai.deeplearning

import kotlin.math.exp
import kotlin.math.tanh

/**
 * Enumeration of activation functions and their first derivatives.
 *
 * **Mathematical Convention for Derivatives:**
 * To optimize performance during backpropagation, the `derivative` functions are designed to accept
 * the **output value** `y = f(x)` (post-activation), NOT the input `x` (pre-activation).
 *
 * This leverages algebraic properties (e.g., Sigmoid derivative is `y * (1 - y)`) to avoid
 * re-computing expensive exponentials.
 */
enum class Activation(
    /** The forward function `f(x)`. */
    val fn: (Double) -> Double,
    /** The derivative `f'(x)` expressed as a function of `y`. */
    val derivative: (Double) -> Double
) {
    /**
     * Identity function: `f(x) = x`.
     * Used for the output layer in regression (DQN).
     */
    LINEAR(
        fn = { x -> x },
        derivative = { _ -> 1.0 }
    ),

    /**
     * Rectified Linear Unit: `f(x) = max(0, x)`.
     * The gold standard for hidden layers.
     */
    RELU(
        fn = { x -> if (x > 0) x else 0.0 },
        derivative = { y -> if (y > 0) 1.0 else 0.0 }
    ),

    /**
     * Sigmoid function: `f(x) = 1 / (1 + e^-x)`.
     * Maps input to (0, 1). Rarely used in modern hidden layers due to vanishing gradients.
     */
    SIGMOID(
        fn = { x -> 1.0 / (1.0 + exp(-x)) },
        derivative = { y -> y * (1.0 - y) }
    ),

    /**
     * Hyperbolic Tangent: `f(x) = tanh(x)`.
     * Maps input to (-1, 1). Zero-centered, often better than Sigmoid.
     */
    TANH(
        fn = { x -> tanh(x) },
        derivative = { y -> 1.0 - (y * y) }
    );
}