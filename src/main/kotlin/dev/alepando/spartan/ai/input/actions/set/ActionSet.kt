package dev.alepando.spartan.ai.input.actions.set

import dev.alepando.spartan.ai.input.QAction

/** Set of actions available to an agent. */
interface ActionSet {
    /** @return the list of actions. */
    fun get(): List<QAction>

    /** @return the number of actions. */
    fun size(): Int {
        return get().size
    }
}