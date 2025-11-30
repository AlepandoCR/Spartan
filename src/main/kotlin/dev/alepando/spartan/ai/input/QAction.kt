package dev.alepando.spartan.ai.input

import dev.alepando.spartan.ai.context.GameContext

/** Abstract action an agent can perform. */
abstract class QAction {
    /** Executes the action using the given [context]. */
    fun execute(context: GameContext) { task(context) }

    /** Implementation of the action behavior. */
    protected abstract fun task(context: GameContext)

    /** Returns the immediate outcome used as a training signal. */
    abstract fun outcome(context: GameContext): Double
}
