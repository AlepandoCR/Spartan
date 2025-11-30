package dev.alepando.spartan.ai.deeplearning.models

/** Identifiers for model variants used by Spartan. */
enum class ModelType(val hash: String) {
    /** Global shared model used by all agents regardless of role. */
    GLOBAL("global_spartan_dqn"),
}