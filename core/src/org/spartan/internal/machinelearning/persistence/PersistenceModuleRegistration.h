#pragma once

namespace org::spartan::internal::machinelearning::persistence {

    /**
     * @brief Initializes all persistence modules at library startup.
     *
     * This function is automatically called when the shared library loads
     * via a static initializer in PersistenceModuleRegistration.cpp.
     *
     * Ensures that all model type persistence modules (RSAC, DDQN, AutoEncoder,
     * Curiosity RSAC) are registered in the ModelPersistenceRegistry before
     * the first save/load operation occurs.
     */
    void initializePersistenceModules();

}

