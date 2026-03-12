package org.spartan.api.agent.config;

/**
 * Configuration for the AutoEncoder Compressor model.
 * Mirrors the C++ struct AutoEncoderCompressorHyperparameterConfig.
 */
public record AutoEncoderCompressorConfig(
        double learningRate,
        double gamma,
        double epsilon,
        double epsilonMin,
        double epsilonDecay,
        int stateSize,
        int actionSize,
        boolean isTraining,
        int latentDimensionSize,
        int encoderHiddenNeuronCount,
        int encoderLayerCount,
        int decoderLayerCount,
        double bottleneckRegularisationWeight
) implements SpartanModelConfig {

    @Override
    public SpartanModelType modelType() {
        return SpartanModelType.AUTO_ENCODER_COMPRESSOR;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private double learningRate = 1e-3;
        private double gamma = 0.0;
        private double epsilon = 0.0;
        private double epsilonMin = 0.0;
        private double epsilonDecay = 0.0;
        private int stateSize = 64;
        private int actionSize = 16;
        private boolean isTraining = true;
        private int latentDimensionSize = 16;
        private int encoderHiddenNeuronCount = 128;
        private int encoderLayerCount = 2;
        private int decoderLayerCount = 2;
        private double bottleneckRegularisationWeight = 1e-4;

        private Builder() {}

        public Builder learningRate(double val) { this.learningRate = val; return this; }
        public Builder gamma(double val) { this.gamma = val; return this; }
        public Builder epsilon(double val) { this.epsilon = val; return this; }
        public Builder epsilonMin(double val) { this.epsilonMin = val; return this; }
        public Builder epsilonDecay(double val) { this.epsilonDecay = val; return this; }
        public Builder stateSize(int val) { this.stateSize = val; return this; }
        public Builder actionSize(int val) { this.actionSize = val; return this; }
        public Builder isTraining(boolean val) { this.isTraining = val; return this; }
        public Builder latentDimensionSize(int val) {
            this.latentDimensionSize = val;
            this.actionSize = val;
            return this;
        }
        public Builder encoderHiddenNeuronCount(int val) { this.encoderHiddenNeuronCount = val; return this; }
        public Builder encoderLayerCount(int val) { this.encoderLayerCount = val; return this; }
        public Builder decoderLayerCount(int val) { this.decoderLayerCount = val; return this; }
        public Builder bottleneckRegularisationWeight(double val) { this.bottleneckRegularisationWeight = val; return this; }

        public AutoEncoderCompressorConfig build() {
            return new AutoEncoderCompressorConfig(
                    learningRate, gamma, epsilon, epsilonMin, epsilonDecay,
                    stateSize, actionSize, isTraining,
                    latentDimensionSize, encoderHiddenNeuronCount,
                    encoderLayerCount, decoderLayerCount,
                    bottleneckRegularisationWeight
            );
        }
    }
}
