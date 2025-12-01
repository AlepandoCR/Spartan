package dev.alepando.spartan.ai.context.features

import dev.alepando.spartan.ai.context.GameContext

/**
 * Contract for extracting one or more numeric features from a game snapshot.
 *
 * Implementations must be pure functions of the provided [GameContext] and
 * return a fixed-size [DoubleArray].
 *
 * - Input: [GameContext] describing the current world and entities.
 * - Output: [DoubleArray] of features; length must always equal [size].
 * - Error modes: Implementations should be defensive and return
 *   normalized defaults (e.g., 0.0) when data is missing.
 */
interface FeatureExtractor<C: GameContext> {
    /** Number of features this extractor produces. Must be constant. */
    val size: Int

    /** Extract the feature vector for the given [context]. Length must be [size]. */
    fun extract(context: C): DoubleArray
}

