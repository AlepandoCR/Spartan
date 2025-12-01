import dev.alepando.spartan.database.data.DqnDto

/** Persistence abstraction for model parameters and metadata. */
interface ModelStore {
    /** Snapshot of model parameters. */
    data class ModelSnapshot(
        val inputSize: Int,
        val hiddenWeights: List<List<Double>>,
        val hiddenBiases: List<Double>,
        val outputWeights: List<List<Double>>,
        val outputBiases: List<Double>,
        val performance: Double
    )

    /** Saves [modelDto]. */
    fun save(modelDto: DqnDto)

    /** Loads a snapshot for [hash], or returns null if not found. */
    fun load(hash: String): ModelSnapshot?
}