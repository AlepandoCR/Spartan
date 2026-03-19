package org.spartan.internal.engine.config;

import org.spartan.api.engine.config.SpartanMultiAgentGroupConfig;

public record SpartanMultiAgentGroupConfigImpl(
        double learningRate,
        double gamma,
        double epsilon,
        double epsilonMin,
        double epsilonDecay,
        boolean debugLogging,
        boolean isTraining,
        int maxAgents
) implements SpartanMultiAgentGroupConfig {
}

