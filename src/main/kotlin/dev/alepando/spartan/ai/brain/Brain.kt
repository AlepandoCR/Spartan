package dev.alepando.spartan.ai.brain

import dev.alepando.spartan.ai.input.QAction
import dev.alepando.spartan.ai.input.actions.set.ActionSet
import dev.alepando.spartan.ai.context.GameContext

/** Base class for agent brains managing action registration and decision ticks. */
abstract class Brain {
    /** Registered actions available for selection. */
    protected val actions = mutableListOf<QAction>()

    /** Registers a single [action] into the action space. */
    fun registerAction(action: QAction) { actions.add(action) }

    /** Registers all actions from the provided [actionSet]. */
    fun registerSet(actionSet: ActionSet) { actions.addAll(actionSet.get()) }

    /** Advances one decision step using the provided [context]. */
    abstract fun tick(context: GameContext)
}
