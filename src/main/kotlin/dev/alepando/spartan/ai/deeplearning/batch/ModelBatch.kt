package dev.alepando.spartan.ai.deeplearning.batch

import dev.alepando.spartan.ai.deeplearning.QNetwork
import dev.alepando.spartan.ai.deeplearning.models.ModelType
import dev.alepando.spartan.ai.input.actions.set.ActionSet
import dev.alepando.spartan.database.store.ModelStore
import dev.alepando.spartan.util.async

/**
 * Manages multiple Q-network instances and tracks their scores.
 * Supports loading/saving via ModelStore.
 */
class ModelBatch<T : ModelType>(
    private val modelType: T,
    private val store: ModelStore,
    private val actionSet: ActionSet,
    private val inputSize: Int
) {
    private val models = mutableListOf<QNetwork>()
    private val scores = mutableListOf<Double>()

    /** Loads an existing model or creates a new one bound to the current actions. */
    fun getModel(): QNetwork {
        val actions = actionSet.get()
        val model = QNetwork.load(modelType, store, actions) ?: QNetwork(inputSize, actions)
        models.add(model)
        scores.add(0.0)
        return model
    }

    /** Updates the tracked score for the given model asynchronously. */
    fun updateModel(model: QNetwork, score: Double) {
        async {
            val index = models.indexOf(model)
            if (index != -1) {
                scores[index] = score
            }
        }
    }

    /** Saves the best-scoring model to the store. */
    fun saveBestModel() {
        if (models.isNotEmpty()) {
            val bestModelIndex = scores.indices.maxByOrNull { scores[it] } ?: 0
            val bestModel = models[bestModelIndex]
            val bestScore = scores[bestModelIndex]
            bestModel.save(modelType, store, bestScore)
        }
    }
}
