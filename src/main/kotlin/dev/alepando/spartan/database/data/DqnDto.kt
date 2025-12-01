package dev.alepando.spartan.database.data

/** Serialized parameters and metadata for a DQN model. */
data class DqnDto(
    val hash: String,
    val inputSize: Int,
    val hiddenWeights: List<List<Double>>,
    val hiddenBiases: List<Double>,
    val outputWeights: List<List<Double>>,
    val outputBiases: List<Double>,
    val performance: Double = 0.0
)
