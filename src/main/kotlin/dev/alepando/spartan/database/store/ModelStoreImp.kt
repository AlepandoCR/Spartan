package dev.alepando.spartan.database.store

import ModelStore
import dev.alepando.spartan.database.data.DqnDto
import dev.alepando.spartan.database.table.types.DqnModelTable



/** ModelStore implementation backed by MySQL via DqnModelTable. */
class ModelStoreImp(private val table: DqnModelTable) : ModelStore {
    /** Saves a snapshot under a given hash identifier. */
    override fun save(modelDto: DqnDto) {
        table.insertOrUpdate(modelDto)
    }

    /** Loads a snapshot by its hash identifier, or null if not present. */
    override fun load(hash: String): ModelStore.ModelSnapshot? {
        val entry: DqnDto = table.findBy("hash", hash) ?: return null
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
