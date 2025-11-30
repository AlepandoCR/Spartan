package dev.alepando.spartan.database.store

import dev.alepando.spartan.database.data.DQNDto
import dev.alepando.spartan.database.table.types.DqnModelTable

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

    /** Saves [snapshot] under the identifier [hash]. */
    fun save(hash: String, snapshot: ModelSnapshot)

    /** Loads a snapshot for [hash], or returns null if not found. */
    fun load(hash: String): ModelSnapshot?
}

/** ModelStore implementation backed by MySQL via DqnModelTable. */
class ModelStoreImp(private val table: DqnModelTable) : ModelStore {
    /** Saves a snapshot under a given hash identifier. */
    override fun save(hash: String, snapshot: ModelStore.ModelSnapshot) {
        table.insertOrUpdate(
            DQNDto(
                hash = hash,
                inputSize = snapshot.inputSize,
                hiddenWeights = snapshot.hiddenWeights,
                hiddenBiases = snapshot.hiddenBiases,
                outputWeights = snapshot.outputWeights,
                outputBiases = snapshot.outputBiases,
                performance = snapshot.performance
            )
        )
    }

    /** Loads a snapshot by its hash identifier, or null if not present. */
    override fun load(hash: String): ModelStore.ModelSnapshot? {
        val entry: DQNDto = table.findBy("hash", hash) ?: return null
        return ModelStore.ModelSnapshot(
            inputSize = entry.inputSize,
            hiddenWeights = entry.hiddenWeights,
            hiddenBiases = entry.hiddenBiases,
            outputWeights = entry.outputWeights,
            outputBiases = entry.outputBiases,
            performance = entry.performance
        )
    }
}
