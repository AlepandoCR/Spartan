package dev.alepando.spartan.ai.dqn.training

import dev.alepando.spartan.ai.input.QAction

/**
 * Experience tuple used for training DQN.
 */
data class Transition(
    val state: DoubleArray,
    val action: QAction,
    val reward: Double,
    val nextState: DoubleArray,
    val done: Boolean
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Transition

        if (reward != other.reward) return false
        if (done != other.done) return false
        if (!state.contentEquals(other.state)) return false
        if (action != other.action) return false
        if (!nextState.contentEquals(other.nextState)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = reward.hashCode()
        result = 31 * result + done.hashCode()
        result = 31 * result + state.contentHashCode()
        result = 31 * result + action.hashCode()
        result = 31 * result + nextState.contentHashCode()
        return result
    }
}
