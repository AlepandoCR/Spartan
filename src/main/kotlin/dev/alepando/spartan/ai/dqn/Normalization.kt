package dev.alepando.spartan.ai.dqn

/** Input normalization utilities. */
object Normalization {

    /**
     * Scales a value to the [0, 1] range based on min/max bounds.
     * Useful for inputs like Health, Hunger, Distance.
     */
    fun normalize(value: Double, min: Double, max: Double): Double {
        if (max - min == 0.0) return 0.0
        val result = (value - min) / (max - min)
        return result.coerceIn(0.0, 1.0)
    }

    /**
     * Scales a value to [-1, 1]. Better for Coordinates and Angles (Yaw/Pitch).
     * Neural Networks (Tanh/ReLU) often prefer inputs centered around 0.
     */
    fun normalizeSym(value: Double, min: Double, max: Double): Double {
        return normalize(value, min, max) * 2.0 - 1.0
    }

}