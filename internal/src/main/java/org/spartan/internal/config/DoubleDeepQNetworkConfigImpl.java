package org.spartan.internal.config;
import org.spartan.api.agent.config.DoubleDeepQNetworkConfig;
import org.spartan.api.agent.config.SpartanModelType;
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
