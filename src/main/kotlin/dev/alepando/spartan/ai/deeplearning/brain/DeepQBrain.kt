package dev.alepando.spartan.ai.deeplearning.brain

import dev.alepando.spartan.ai.brain.Brain
import dev.alepando.spartan.ai.context.GameContext
import dev.alepando.spartan.ai.deeplearning.QNetwork
import dev.alepando.spartan.ai.deeplearning.batch.ModelBatch
import dev.alepando.spartan.ai.deeplearning.buffer.ReplayBuffer
import dev.alepando.spartan.ai.deeplearning.models.ModelType
import dev.alepando.spartan.ai.deeplearning.training.DqnTrainer
import dev.alepando.spartan.ai.deeplearning.training.Transition
import dev.alepando.spartan.ai.input.QAction
import dev.alepando.spartan.ai.input.actions.set.ActionSet
import kotlin.random.Random

/**
 * The **Deep Q-Learning Agent (DQN)** controller.
 *
 * Connects the Environment (Minecraft/AIContext) with the Neural Network.
 * Handles the "Agent-Environment Loop": Observe -> Act -> Wait -> Reward -> Learn.
 *
 * **Hyperparameters:**
 * * [gamma]: Discount factor for future rewards (0.99 = long-term focus).
 * * [epsilon]: Exploration rate (1.0 = 100% random, 0.0 = 100% greedy).
 * * [decisionFrequency]: How often (in ticks) the AI selects a new action.
 * Value of 4 (approx 200ms) allows physics to resolve before next decision.
 */
class DeepQBrain(
    private val modelBatch: ModelBatch<out ModelType>,
    actionSet: ActionSet,
    private val gamma: Double = 0.99,
    private val batchSize: Int = 32,
    learningRate: Double = 0.001,
    private var epsilon: Double = 1.0,
    private val epsilonDecay: Double = 0.995,
    private val epsilonMin: Double = 0.05,
    private val decisionFrequency: Int = 4
) : Brain() {

    private val buffer = ReplayBuffer(capacity = 10000)
    private val model: QNetwork
    private val trainer: DqnTrainer

    // State persistence variables to handle the time gap between ticks
    private var lastState: DoubleArray? = null
    private var lastAction: QAction? = null
    private var ticksAlive = 0

    init {
        registerSet(actionSet)
        // Ensure the ModelBatch provides a compatible QNetwork
        model = modelBatch.getModel()
        trainer = DqnTrainer(model, buffer, gamma, batchSize, learningRate)
    }

    override fun tick(context: GameContext) {
        ticksAlive++

        // Observation (State t)
        val currentState = context.observation()
        val isTerminal = context.isTerminal()

        // Learning Step (Delayed Reward Association)
        // We link the action taken in the PREVIOUS decision tick with the CURRENT state/reward
        if (lastState != null && lastAction != null) {
            // Calculate reward based on the outcome of the previous action
            val reward = lastAction!!.outcome(context)

            // Store experience: (s_t-1, a_t-1, r, s_t, done)
            buffer.add(Transition(
                state = lastState!!,
                action = lastAction!!,
                reward = reward,
                nextState = currentState,
                done = isTerminal
            ))

            // Train the network (Mini-Batch SGD)
            // We train every decision tick to keep performance stable
            if (ticksAlive % decisionFrequency == 0) {
                trainer.train(epsilon)
            }
        }

        // Reset if terminal
        if (isTerminal) {
            lastState = null
            lastAction = null
            // Update global model stats periodically
            modelBatch.updateModel(model, buffer.averageReward())
            return
        }

        // Action Selection (Frame Skip Logic)
        // Only make a new decision every X ticks to allow physics to happen
        if (ticksAlive % decisionFrequency == 0) {
            val action = selectAction(currentState)

            action.execute(context)

            // Update persistence for the next cycle
            lastState = currentState
            lastAction = action

            // Decay exploration rate
            if (epsilon > epsilonMin) {
                epsilon *= epsilonDecay
            }
        } else {
            //TBD: Maintain last action or do nothing
        }
    }

    /** Epsilon-Greedy Strategy */
    private fun selectAction(state: DoubleArray): QAction {
        // Exploration: Choose random action
        if (Random.nextDouble() < epsilon) {
            return actions.random()
        }
        // Exploitation: Choose best action from Neural Net
        return model.getBestAction(state)
    }
}