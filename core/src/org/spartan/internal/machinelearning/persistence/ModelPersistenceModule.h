#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <map>

namespace org::spartan::internal::machinelearning {
    class SpartanModel;
}

namespace org::spartan::internal::machinelearning::persistence {

    // Model type identifiers
    constexpr uint32_t MODEL_TYPE_RSAC = 1;
    constexpr uint32_t MODEL_TYPE_DDQN = 2;
    constexpr uint32_t MODEL_TYPE_AUTOENCODER = 3;
    constexpr uint32_t MODEL_TYPE_CURIOSITY_RSAC = 4;

    /**
     * @class ModelPersistenceModule
     * @brief Abstract interface for model-type-specific persistence.
     *
     * Each model type (RSAC, DDQN, AutoEncoder, etc.) implements this
     * to provide serialization and deserialization of its weights.
     */
    class ModelPersistenceModule {
    public:
        virtual ~ModelPersistenceModule() = default;

        /**
         * Serializes a model's weights into a flat vector.
         *
         * @param model Pointer to the SpartanModel to serialize
         * @return Vector of doubles representing all weights in deterministic order
         */
        virtual std::vector<double> serialize(const SpartanModel* model) = 0;

        /**
         * Deserializes weights from a flat vector into a model.
         *
         * @param model Pointer to the SpartanModel to restore into
         * @param weights Vector of doubles (must match expected size)
         * @return true if successful, false if size/type mismatch
         */
        virtual bool deserialize(SpartanModel* model,
                                 const std::vector<double>& weights) = 0;

        /**
         * Checks if this module can handle the given model type.
         *
         * @param modelTypeIdentifier The model type ID from .spartan header
         * @return true if this module handles this type
         */
        [[nodiscard]] virtual bool canHandle(uint32_t modelTypeIdentifier) const = 0;

        /**
         * Returns the model type ID this module is responsible for.
         *
         * @return Model type identifier (e.g., MODEL_TYPE_RSAC)
         */
        [[nodiscard]] virtual uint32_t modelTypeId() const = 0;
    };

    /**
     * @class ModelPersistenceRegistry
     * @brief Singleton registry/factory for model persistence modules.
     *
     * Manages all registered model type persistence modules and routes
     * save/load operations to the appropriate handler.
     */
    class ModelPersistenceRegistry {
    public:
        /**
         * Gets the singleton instance.
         */
        static ModelPersistenceRegistry& getInstance();

        /**
         * Registers a new model type persistence module.
         *
         * @param module Unique pointer to the module implementation
         */
        void registerModule(std::unique_ptr<ModelPersistenceModule> module);

        /**
         * Gets the module responsible for a model type.
         *
         * @param modelTypeId The model type identifier
         * @return Pointer to the module, or nullptr if not registered
         */
        ModelPersistenceModule* getModule(uint32_t modelTypeId);

        ModelPersistenceRegistry(const ModelPersistenceRegistry&) = delete;
        ModelPersistenceRegistry& operator=(const ModelPersistenceRegistry&) = delete;

    private:
        ModelPersistenceRegistry() = default;
        ~ModelPersistenceRegistry() = default;


        std::map<uint32_t, std::unique_ptr<ModelPersistenceModule>> modules_;
    };

}
