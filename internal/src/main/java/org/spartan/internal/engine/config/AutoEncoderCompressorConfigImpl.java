package org.spartan.internal.engine.config;

import org.spartan.api.engine.config.AutoEncoderCompressorConfig;

public record AutoEncoderCompressorConfigImpl(
        double learningRate,
        double gamma,
        double epsilon,
        double epsilonMin,
        double epsilonDecay,
        boolean debugLogging,
        boolean isTraining,
        int latentDimensionSize,
        int encoderHiddenNeuronCount,
        int encoderLayerCount,
        int decoderLayerCount,
        double bottleneckRegularisationWeight
) implements AutoEncoderCompressorConfig {

}
