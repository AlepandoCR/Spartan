package org.spartan.internal.engine.config;
import org.spartan.api.engine.config.DoubleDeepQNetworkConfig;

public record DoubleDeepQNetworkConfigImpl(
        double learningRate,
        double gamma,
        double epsilon,
        double epsilonMin,
        double epsilonDecay,
        boolean debugLogging,
        boolean isTraining,
        int targetNetworkSyncInterval,
        int replayBufferCapacity,
        int replayBatchSize,
        int hiddenLayerNeuronCount,
        int hiddenLayerCount
) implements DoubleDeepQNetworkConfig {
}
