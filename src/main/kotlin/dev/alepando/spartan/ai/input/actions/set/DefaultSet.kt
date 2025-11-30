package dev.alepando.spartan.ai.input.actions.set

import dev.alepando.spartan.ai.input.QAction

/** Default action set placeholder (intentionally empty). */
object DefaultSet : ActionSet {
    override fun get(): List<QAction> = emptyList()
}