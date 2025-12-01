package dev.alepando.spartan.ai.context

/**
 * Environment context passed to brains and actions.
 * Implementations must provide a normalized observation and a terminal flag.
 */
interface GameContext {

    /** Returns the normalized observation vector for the current timestep. */
    fun observation(): DoubleArray

    /** Returns true if the current episode should terminate. */
    fun isTerminal(): Boolean = false

}
