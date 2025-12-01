package dev.alepando.spartan.ai.context.features

import dev.alepando.spartan.ai.context.GameContext

/**
 * Combines multiple [FeatureExtractor]s into a single vector.
 */
class CompositeFeatureExtractor<C: GameContext>(
    private val extractors: List<FeatureExtractor<C>>
) : FeatureExtractor<C> {
    override val size: Int = extractors.sumOf { it.size }
    override fun extract(context: C): DoubleArray {
        val parts = extractors.flatMap { it.extract(context).asIterable() }
        return parts.toDoubleArray()
    }
}