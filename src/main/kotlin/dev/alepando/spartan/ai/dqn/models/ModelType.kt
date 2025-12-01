package dev.alepando.spartan.ai.dqn.models

/** Identifiers for model variants used by Spartan. */
enum class ModelType(val hash: String) {
    CRITIC_1("critic_1"),
    CRITIC_2("critic_2"),
}