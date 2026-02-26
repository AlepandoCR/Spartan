package org.spartan.api.agent.config;

/**
 * Mirror to {@code core/src/org/spartan/core/machinelearning/ModelHyperparameterConfig.h}
 */
public interface SpartanModelConfig {

    double getLearningRate();

    double getGamma();

    double getEpsilonStart();

    double getEpsilonDecay();

    double getEpsilonMin();

    boolean isTraining();
}
