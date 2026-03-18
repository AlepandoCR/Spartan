package org.spartan.api.engine.config;

import org.spartan.api.engine.config.spi.SpartanConfigRegistry;

/**
 * Configuration for the AutoEncoder Compressor model.
 * <p>
 * <b>Concept:</b> An AutoEncoder is an AI that learns to "zip" data.
 * It takes a large input (e.g., 1000 pixels), squeezes it into a tiny vector (Latent Dimension),
 * and then tries to "unzip" it back to the original. If it succeeds, the tiny vector contains the <i>essence</i> of the image.
 */
public non-sealed interface AutoEncoderCompressorConfig extends SpartanModelConfig {

    /**
     * Returns the size of the compressed representation.
     * <p>
     * <b>Concept:</b> This is the size of the "bottleneck".
     * <ul>
     *   <li><b>Too Small:</b> Essential details are lost (blurry reconstruction).</li>
     *   <li><b>Too Large:</b> No useful compression happens.</li>
     * </ul>
     * This vector becomes the input for other processing agents.
     *
     * @return bottleneck size
     */
    int latentDimensionSize();

    /**
     * Returns the width of the encoder/decoder hidden layers.
     *
     * @return neuron count per layer
     */
    int encoderHiddenNeuronCount();

    /**
     * Returns the depth of the Encoder half.
     *
     * @return number of encoder layers
     */
    int encoderLayerCount();

    /**
     * Returns the depth of the Decoder half.
     *
     * @return number of decoder layers
     */
    int decoderLayerCount();

    /**
     * Returns the weight of the regularization term.
     * <p>
     * <b>Concept:</b> Prevents the encoder from cheating or overfitting.
     * Applies a small force to keep the latent values small and centered.
     *
     * @return regularization weight
     */
    double bottleneckRegularisationWeight();

    @Override
    default SpartanModelType modelType() {
        return SpartanModelType.AUTO_ENCODER_COMPRESSOR;
    }

    static Builder builder() {
        return new Builder();
    }

    final class Builder {
        private double learningRate = 1e-3;
        private double gamma = 0.0;
        private double epsilon = 0.0;
        private double epsilonMin = 0.0;
        private double epsilonDecay = 0.0;
        private boolean debugLogging = false;
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
        public Builder debugLogging(boolean val) { this.debugLogging = val; return this; }
        public Builder isTraining(boolean val) { this.isTraining = val; return this; }
        public Builder latentDimensionSize(int val) {
            this.latentDimensionSize = val;
            return this;
        }
        public Builder encoderHiddenNeuronCount(int val) { this.encoderHiddenNeuronCount = val; return this; }
        public Builder encoderLayerCount(int val) { this.encoderLayerCount = val; return this; }
        public Builder decoderLayerCount(int val) { this.decoderLayerCount = val; return this; }
        public Builder bottleneckRegularisationWeight(double val) { this.bottleneckRegularisationWeight = val; return this; }


        public AutoEncoderCompressorConfig build() {
            return SpartanConfigRegistry.get().createAutoEncoderCompressorConfig(
                learningRate, gamma, epsilon, epsilonMin, epsilonDecay,
                debugLogging, isTraining,
                latentDimensionSize, encoderHiddenNeuronCount, encoderLayerCount,
                decoderLayerCount, bottleneckRegularisationWeight
            );
        }
    }
}
