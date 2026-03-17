package org.spartan.internal.config;

import org.spartan.api.agent.config.AutoEncoderCompressorConfig;
import org.spartan.api.agent.config.SpartanModelType;

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
