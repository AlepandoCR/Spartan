package org.spartan.api.engine.config;

import org.jetbrains.annotations.Contract;
import org.jspecify.annotations.NonNull;
import org.spartan.api.engine.config.spi.SpartanConfigRegistry;

/**
 * Configuration for a Multi-Agent group container.
 *
 * This config applies to the group-level container (capacity, base settings),
 * not to the per-agent model configs passed to addAgent().
 */
public non-sealed interface SpartanMultiAgentGroupConfig extends SpartanModelConfig {

    /**
     * Maximum number of agents allowed in the group.
     */
    int maxAgents();

    @Override
    default SpartanModelType modelType() {
        return SpartanModelType.MULTI_AGENT_GROUP;
    }

    @Contract(value = " -> new", pure = true)
    static @NonNull Builder builder() {
        return new Builder();
    }

    final class Builder {
        private double learningRate = 1e-4;
        private double gamma = 0.99;
        private double epsilon = 1.0;
        private double epsilonMin = 0.01;
        private double epsilonDecay = 0.995;
        private boolean debugLogging = false;
        private boolean isTraining = true;
        private int maxAgents = 1024;

        public Builder() {}

        public Builder learningRate(double val) { this.learningRate = val; return this; }
        public Builder gamma(double val) { this.gamma = val; return this; }
        public Builder epsilon(double val) { this.epsilon = val; return this; }
        public Builder epsilonMin(double val) { this.epsilonMin = val; return this; }
        public Builder epsilonDecay(double val) { this.epsilonDecay = val; return this; }
        public Builder debugLogging(boolean val) { this.debugLogging = val; return this; }
        public Builder isTraining(boolean val) { this.isTraining = val; return this; }
        public Builder maxAgents(int val) { this.maxAgents = val; return this; }

        public SpartanMultiAgentGroupConfig build() {
            return SpartanConfigRegistry.get().createMultiAgentGroupConfig(
                learningRate,
                gamma,
                epsilon,
                epsilonMin,
                epsilonDecay,
                debugLogging,
                isTraining,
                maxAgents
            );
        }
    }
}

