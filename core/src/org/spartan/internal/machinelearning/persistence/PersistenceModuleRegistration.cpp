//
// Created by Alepando on 12/4/2026.
// Centralizes initialization of all model persistence modules.
//

#include "PersistenceModuleRegistration.h"
#include "RsacPersistenceModule.h"
#include "DdqnPersistenceModule.h"
#include "AutoEncoderPersistenceModule.h"
#include "CuriosityRsacPersistenceModule.h"
#include "../../logging/SpartanLogger.h"

namespace org::spartan::internal::machinelearning::persistence {

    void initializePersistenceModules() {
        logging::SpartanLogger::debug("[PersistenceModuleRegistration] Initializing persistence modules...");

        // Call static initialization on each module
        // This avoids any recursion issues and ensures modules are registered
        RsacPersistenceModule::initializeAndRegister();
        DdqnPersistenceModule::initializeAndRegister();
        AutoEncoderPersistenceModule::initializeAndRegister();
        CuriosityRsacPersistenceModule::initializeAndRegister();

        logging::SpartanLogger::debug("[PersistenceModuleRegistration] All persistence modules registered successfully");
    }

    /**
     * Static initializer to auto-register modules at library load time.
     */
    struct PersistenceModuleAutoInitializer {
        PersistenceModuleAutoInitializer() {
            initializePersistenceModules();
        }
    };

    // This static variable ensures modules are initialized when the library loads
    static const PersistenceModuleAutoInitializer moduleAutoInitializer;

}




