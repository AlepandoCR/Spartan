package org.spartan.api.engine.config;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.spartan.api.engine.config.spi.SpartanConfigRegistry;

/**
 * Configuration for the Curiosity-Driven Recurrent Soft Actor-Critic model.
 * <p>
 * <b>Concept:</b> Standard agents only learn when they get points (Extrinsic Reward).
 * A Curiosity-Driven agent creates its own points (Intrinsic Reward) by exploring things it doesn't understand.
 * It does this by trying to predict the future. If its prediction is wrong, it gets "curious" and visits that state again to learn.
 */
public non-sealed interface CuriosityDrivenRecurrentSoftActorCriticConfig extends SpartanModelConfig {

    /**
     * Returns the base RSAC configuration.
     * The curiosity module is wrapped <i>around</i> a standard RSAC agent.
     *
     * @return the underlying policy configuration
     */
    RecurrentSoftActorCriticConfig recurrentSoftActorCriticConfig();

    /**
     * Returns the size of the hidden layers in the Forward Dynamics Model.
     * <p>
     * <b>Concept:</b> The "Forward Dynamics" network tries to predict the next state given current state + action.
     * This parameter controls how smart this predictor is.
     *
     * @return neuron count for dynamics model
     */
    int forwardDynamicsHiddenLayerDimensionSize();

    /**
     * Returns the scaling factor for intrinsic rewards (curiosity).
     * <p>
     * <b>Concept:</b> How important is curiosity compared to the main game goal?
     * <ul>
     *   <li><b>High (e.g., 1.0):</b> The agent cares more about exploring than winning.</li>
     *   <li><b>Low (e.g., 0.01):</b> The agent uses curiosity mainly to get out of stuck spots, but focuses on the goal.</li>
     * </ul>
     *
     * @return curiosity scale
     */
    double intrinsicRewardScale();

    /**
     * Returns the minimum clamp value for intrinsic rewards.
     * <p>
     * <b>Concept:</b> Prevents the internal curiosity signal from becoming too negative (which shouldn't happen usually, but for stability).
     *
     * @return minimum intrinsic reward
     */
    double intrinsicRewardClampingMinimum();

    /**
     * Returns the maximum clamp value for intrinsic rewards.
     * <p>
     * <b>Concept:</b> Prevents the agent from becoming obsessed with "TV Screen scenarios" (static noise) where prediction error is infinite.
     * Caps the maximum "surprise" it can feel.
     *
     * @return maximum intrinsic reward
     */
    double intrinsicRewardClampingMaximum();

    /**
     * Returns the learning rate specifically for the world model (Forward Dynamics).
     * <p>
     * <b>Concept:</b> The world model needs to learn faster or slower than the actor/critic.
     * Usually similar to the base learning rate.
     *
     * @return dynamics learning rate
     */
    double forwardDynamicsLearningRate();
    @Override
    default SpartanModelType modelType() {
        return SpartanModelType.CURIOSITY_DRIVEN_RECURRENT_SOFT_ACTOR_CRITIC;
    }
    default double learningRate() { return recurrentSoftActorCriticConfig().learningRate(); }
    default double gamma() { return recurrentSoftActorCriticConfig().gamma(); }
    default double epsilon() { return recurrentSoftActorCriticConfig().epsilon(); }
    default double epsilonMin() { return recurrentSoftActorCriticConfig().epsilonMin(); }
    default double epsilonDecay() { return recurrentSoftActorCriticConfig().epsilonDecay(); }
    default boolean isTraining() { return recurrentSoftActorCriticConfig().isTraining(); }


    default boolean debugLogging() { return recurrentSoftActorCriticConfig().debugLogging(); }

    @Contract(value = " -> new", pure = true)
    static @NotNull Builder builder() {
        return new Builder();
    }


    final class Builder {
        private double learningRate = 3e-4;
        private double gamma = 0.99;
        private double epsilon = 1.0;
        private double epsilonMin = 0.01;
        private double epsilonDecay = 0.995;
        private boolean isTraining = true;
        private boolean debugLogging = false;
        private int hiddenStateSize = 128;
        private int recurrentLayerDepth = 1;
        private int actorHiddenLayerNeuronCount = 256;
        private int actorHiddenLayerCount = 2;
        private int criticHiddenLayerNeuronCount = 256;
        private int criticHiddenLayerCount = 2;
        private double targetSmoothingCoefficient = 0.005;
        private double entropyTemperatureAlpha = 0.2;
        private double firstCriticLearningRate = 3e-4;
        private double secondCriticLearningRate = 3e-4;
        private double policyNetworkLearningRate = 3e-4;
        private int recurrentInputFeatureCount = 64;
        private int remorseTraceBufferCapacity = 1000;
        private double remorseMinimumSimilarityThreshold = 0.7;
        private int forwardDynamicsHiddenLayerDimensionSize = 128;
        private double intrinsicRewardScale = 0.01;
        private double intrinsicRewardClampingMinimum = -1.0;
        private double intrinsicRewardClampingMaximum = 1.0;
        private double forwardDynamicsLearningRate = 3e-4;
        private RecurrentSoftActorCriticConfig recurrentSoftActorCriticConfig = null;
        public Builder() {}
        public Builder learningRate(double val) { this.learningRate = val; return this; }
        public Builder gamma(double val) { this.gamma = val; return this; }
        public Builder epsilon(double val) { this.epsilon = val; return this; }
        public Builder epsilonMin(double val) { this.epsilonMin = val; return this; }
        public Builder epsilonDecay(double val) { this.epsilonDecay = val; return this; }
        public Builder isTraining(boolean val) { this.isTraining = val; return this; }
        public Builder debugLogging(boolean val) { this.debugLogging = val; return this; }
        public Builder hiddenStateSize(int val) { this.hiddenStateSize = val; return this; }
        public Builder recurrentLayerDepth(int val) { this.recurrentLayerDepth = val; return this; }
        public Builder actorHiddenLayerNeuronCount(int val) { this.actorHiddenLayerNeuronCount = val; return this; }
        public Builder actorHiddenLayerCount(int val) { this.actorHiddenLayerCount = val; return this; }
        public Builder criticHiddenLayerNeuronCount(int val) { this.criticHiddenLayerNeuronCount = val; return this; }
        public Builder criticHiddenLayerCount(int val) { this.criticHiddenLayerCount = val; return this; }
        public Builder targetSmoothingCoefficient(double val) { this.targetSmoothingCoefficient = val; return this; }
        public Builder entropyTemperatureAlpha(double val) { this.entropyTemperatureAlpha = val; return this; }
        public Builder firstCriticLearningRate(double val) { this.firstCriticLearningRate = val; return this; }
        public Builder secondCriticLearningRate(double val) { this.secondCriticLearningRate = val; return this; }
        public Builder policyNetworkLearningRate(double val) { this.policyNetworkLearningRate = val; return this; }
        public Builder recurrentInputFeatureCount(int val) { this.recurrentInputFeatureCount = val; return this; }
        public Builder remorseTraceBufferCapacity(int val) { this.remorseTraceBufferCapacity = val; return this; }
        public Builder remorseMinimumSimilarityThreshold(double val) { this.remorseMinimumSimilarityThreshold = val; return this; }
        public Builder recurrentSoftActorCriticConfig(RecurrentSoftActorCriticConfig config) {
            this.recurrentSoftActorCriticConfig = config;
            return this;
        }
        public Builder forwardDynamicsHiddenLayerDimensionSize(int val) { this.forwardDynamicsHiddenLayerDimensionSize = val; return this; }
        public Builder intrinsicRewardScale(double val) { this.intrinsicRewardScale = val; return this; }
        public Builder intrinsicRewardClampingMinimum(double val) { this.intrinsicRewardClampingMinimum = val; return this; }
        public Builder intrinsicRewardClampingMaximum(double val) { this.intrinsicRewardClampingMaximum = val; return this; }
        public Builder forwardDynamicsLearningRate(double val) { this.forwardDynamicsLearningRate = val; return this; }

        public @NotNull CuriosityDrivenRecurrentSoftActorCriticConfig build() {
             RecurrentSoftActorCriticConfig rsac = this.recurrentSoftActorCriticConfig;
             if (rsac == null) {
                 rsac = RecurrentSoftActorCriticConfig.builder()
                        .learningRate(learningRate)
                        .gamma(gamma)
                        .epsilon(epsilon)
                        .epsilonMin(epsilonMin)
                        .epsilonDecay(epsilonDecay)
                        .debugLogging(debugLogging)
                        .isTraining(isTraining)
                        .hiddenStateSize(hiddenStateSize)
                        .recurrentLayerDepth(recurrentLayerDepth)
                        .actorHiddenLayerNeuronCount(actorHiddenLayerNeuronCount)
                        .actorHiddenLayerCount(actorHiddenLayerCount)
                        .criticHiddenLayerNeuronCount(criticHiddenLayerNeuronCount)
                        .criticHiddenLayerCount(criticHiddenLayerCount)
                        .targetSmoothingCoefficient(targetSmoothingCoefficient)
                        .entropyTemperatureAlpha(entropyTemperatureAlpha)
                        .firstCriticLearningRate(firstCriticLearningRate)
                        .secondCriticLearningRate(secondCriticLearningRate)
                        .policyNetworkLearningRate(policyNetworkLearningRate)
                        .recurrentInputFeatureCount(recurrentInputFeatureCount)
                        .remorseTraceBufferCapacity(remorseTraceBufferCapacity)
                        .remorseMinimumSimilarityThreshold(remorseMinimumSimilarityThreshold)
                        .build();
             }
            return SpartanConfigRegistry.get().createCuriosityDrivenRecurrentSoftActorCriticConfig(
                rsac,
                forwardDynamicsHiddenLayerDimensionSize, intrinsicRewardScale,
                intrinsicRewardClampingMinimum, intrinsicRewardClampingMaximum,
                forwardDynamicsLearningRate
            );
        }
    }
}
