package dev.alepando.spartan.ai.dqn.training

import dev.alepando.spartan.ai.dqn.QNetwork
import dev.alepando.spartan.ai.dqn.buffer.ReplayBuffer

/**
 * Orchestrates the **Double Deep Q-Learning (DDQN)** training loop.
 *
 * This trainer implements an advanced variation of DQN designed to reduce **Maximization Bias**
 * by decoupling the action selection from the value estimation.
 *
 * **Core Components**
 * * **Online Network (`model`):** Selects the best action `a*` for the next state (parameters `θ`).
 * * **Target Network (`targetModel`):** Evaluates the Q-value of that selected action (parameters `θ⁻`).
 * * **Replay Buffer:** Stores transitions to break temporal correlations.
 *
 * **Objective Function (Double DQN)**
 * Minimizes the Bellman error using the following target formulation:
 *
 * `Y = r + γ · Q(s', argmax Q(s', a; θ); θ⁻)`
 *
 * Unlike standard DQN, which uses `max Q(s', a; θ⁻)`, this approach prevents the overestimation
 * of action values common in stochastic environments.
 */
class DqnTrainer(
    private val model: QNetwork,
    private val replayBuffer: ReplayBuffer,
    private val gamma: Double,
    private val samplesPerTrainingStep: Int,
    learningRate: Double,
    private val targetUpdateFrequency: Int = 100
) {
    private val targetModel = model.copy()
    private var trainSteps = 0
    private var totalLoss = 0.0
    private val optimizer = Optimizer(model, learningRate, 1.0)

    /**
     * Executes a single training step using **Double DQN** logic.
     *
     * **Process Flow**
     * 1. **Sampling:** Draws a random batch of transitions.
     * 2. **Action Selection (Online):** The Online network determines which action is "best" in `s'`.
     * `a* = argmax_a Q(s', a; θ)`
     * 3. **Value Evaluation (Target):** The Target network calculates the value of `a*`.
     * `value = Q_target(s', a*)`
     * 4. **Backpropagation:** Updates weights based on the computed TD Error.
     *
     * @param epsilon Current exploration rate used only for logging/monitoring purposes.
     */
    fun train(epsilon: Double) {
        if (replayBuffer.size() < samplesPerTrainingStep) return
        var batchLoss = 0.0
        val batch = replayBuffer.sample(samplesPerTrainingStep)

        for (transition in batch) {
            //Prediction for current state (to compute error later)
            val qValues = model.predict(transition.state)

            //Double DQN Logic for Target Calculation
            val futureQ: Double
            if (transition.done) {
                futureQ = 0.0
            } else {
                // Select best action using Online Network (θ)
                val qNextOnline = model.predict(transition.nextState)
                val bestActionIdx = qNextOnline.indices.maxByOrNull { qNextOnline[it] } ?: 0

                // Evaluate that specific action using Target Network (θ⁻)
                val qNextTarget = targetModel.predict(transition.nextState)
                val doubleQValue = qNextTarget[bestActionIdx]

                futureQ = if (doubleQValue.isFinite()) doubleQValue else 0.0
            }

            //Compute Target (Bellman Equation)
            val target = (transition.reward + gamma * futureQ).coerceIn(-1.0, 1.0)

            //Calculate Error & Update
            val actionIndex = model.getActionIndex(transition.action)
            if (actionIndex == -1) continue

            val error = qValues[actionIndex] - target
            batchLoss += error * error

            optimizer.accumulateGradients(transition.state, actionIndex, target)
        }

        //Apply accumulated gradients after processing the batch
        optimizer.applyGradients(batch.size)


        // Logging & Sync Logic
        totalLoss += batchLoss / samplesPerTrainingStep
        trainSteps++

        if (trainSteps % targetUpdateFrequency == 0) {
            updateTargetModel()
            val avgLoss = totalLoss / targetUpdateFrequency
            val avgReward = replayBuffer.averageReward()
            println("Steps: $trainSteps, Epsilon: $epsilon, Avg Loss: $avgLoss, Avg Reward: $avgReward")
            totalLoss = 0.0
            trainSteps = 0
        }
    }

    /**
     * Synchronizes the Target Network weights with the Online Network (`θ⁻ ← θ`).
     *
     * **Stability Mechanism**
     * This prevents the "Moving Target" problem. By keeping the target function constant
     * for a period (`targetUpdateFrequency`), the optimization landscape remains stable.
     */
    private fun updateTargetModel() {
        targetModel.hidden.weights = model.hidden.weights.clone()
        targetModel.hidden.biases = model.hidden.biases.clone()
        targetModel.output.weights = model.output.weights.clone()
        targetModel.output.biases = model.output.biases.clone()
    }
}