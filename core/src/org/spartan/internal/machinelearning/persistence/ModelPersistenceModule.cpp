#include "ModelPersistenceModule.h"
#include "../../logging/SpartanLogger.h"
#include <string>

namespace org::spartan::internal::machinelearning::persistence {

    ModelPersistenceRegistry& ModelPersistenceRegistry::getInstance() {
        static ModelPersistenceRegistry instance;
        return instance;
    }

    void ModelPersistenceRegistry::registerModule(std::unique_ptr<ModelPersistenceModule> module) {
        if (!module) {
            logging::SpartanLogger::error("ModelPersistenceRegistry: Attempted to register null module");
            return;
        }

        const uint32_t modelTypeId = module->modelTypeId();

        if (modules_.find(modelTypeId) != modules_.end()) {
            logging::SpartanLogger::warn("ModelPersistenceRegistry: Module for type " +
                                       std::to_string(modelTypeId) + " already registered, replacing");
        }

        modules_[modelTypeId] = std::move(module);
        logging::SpartanLogger::debug("ModelPersistenceRegistry: Registered module for type " +
                                    std::to_string(modelTypeId));
    }

    ModelPersistenceModule* ModelPersistenceRegistry::getModule(uint32_t modelTypeId) {
        auto it = modules_.find(modelTypeId);
        if (it == modules_.end()) {
            logging::SpartanLogger::error("ModelPersistenceRegistry: No module registered for type " +
                                        std::to_string(modelTypeId));
            return nullptr;
        }
        return it->second.get();
    }

}
