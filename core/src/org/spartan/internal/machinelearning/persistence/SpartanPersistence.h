//
// Created by Alepando on 10/3/2026.
//

#pragma once

#include <cstdint>
#include <span>

// Forward declaration
namespace org::spartan::internal::machinelearning {
    class SpartanModel;
}

/**
 * @file SpartanPersistence.h
 * @brief Binary persistence format for saving and loading Spartan models.
 *
 * Defines the hierarchical .spartan file format that serializes a complete
 * model (main network plus all nested sub-models) into a single cross-platform
 * binary file.
 *
 * The format uses Standard Layout structs for the header and table of contents
 * entries so that Java (via ByteBuffer.order(LITTLE_ENDIAN)) and .NET (via
 * BinaryReader) can parse the same file without C++ involvement.
 *
 * All numeric values are stored in little-endian IEEE 754 format.
 * A trailing CRC-32 checksum protects against file corruption.
 */
namespace org::spartan::internal::machinelearning::persistence {

    /**
     * @brief Magic bytes identifying a .spartan file: 'S' 'P' 'R' 'T'.
     */
    constexpr uint8_t SPARTAN_FILE_MAGIC_BYTES[4] = { 0x53, 0x50, 0x52, 0x54 };

    /** @brief Current format version. */
    constexpr uint16_t SPARTAN_FILE_FORMAT_VERSION = 1;

    /**
     * @enum SpartanModelTypeIdentifier
     * @brief Enum identifying the top-level model type stored in the file.
     */
    enum SpartanModelTypeIdentifier : uint32_t {
        MODEL_TYPE_RECURRENT_SOFT_ACTOR_CRITIC           = 1,
        MODEL_TYPE_DOUBLE_DEEP_Q_NETWORK                 = 2,
        MODEL_TYPE_AUTO_ENCODER_COMPRESSOR               = 3,
        MODEL_TYPE_CURIOSITY_DRIVEN_RECURRENT_SOFT_ACTOR_CRITIC = 4,
    };

    /**
     * @enum SubModelRole
     * @brief Enum identifying the role of each sub-model in the table of contents.
     */
    enum SubModelRole : uint32_t {
        SUB_MODEL_GATED_RECURRENT_UNIT    = 0,
        SUB_MODEL_GAUSSIAN_POLICY         = 1,
        SUB_MODEL_Q_CRITIC_FIRST          = 2,
        SUB_MODEL_Q_CRITIC_SECOND         = 3,
        SUB_MODEL_NESTED_ENCODER          = 4,
        SUB_MODEL_TARGET_NETWORK          = 5,
    };

    /**
     * @enum ActivationFunctionType
     * @brief Enum for the activation function used in a sub-model's layers.
     */
    enum ActivationFunctionType : uint32_t {
        ACTIVATION_RELU       = 0,
        ACTIVATION_LEAKY_RELU = 1,
        ACTIVATION_TANH       = 2,
        ACTIVATION_SIGMOID    = 3,
    };

    /**
     * @struct SpartanFileHeader
     * @brief The first 48 bytes of every .spartan file. Standard Layout, no padding.
     */
    struct SpartanFileHeader {
        /** @brief Magic bytes: 0x53 0x50 0x52 0x54 ("SPRT"). */
        uint8_t magicBytes[4];

        /** @brief Format version number. */
        uint16_t formatVersion;

        /** @brief Reserved flags for future use (compression, encryption). */
        uint16_t formatFlags;

        /** @brief Top-level model type (SpartanModelTypeIdentifier). */
        uint32_t modelTypeIdentifier;

        /** @brief Total number of sub-model entries in the table of contents. */
        uint32_t subModelCount;

        /** @brief Total number of raw double-precision weight values in the file. */
        uint64_t totalWeightCount;

        /** @brief Byte offset from file start to the first weight byte. */
        uint64_t weightBlobByteOffset;

        /** @brief Total byte size of the concatenated weight blob section. */
        uint64_t weightBlobTotalByteSize;

        /** @brief Reserved for future expansion. */
        uint64_t reservedPadding;
    };
    static_assert(sizeof(SpartanFileHeader) == 48, "SpartanFileHeader must be exactly 48 bytes");

    /**
     * @struct SubModelTopologyEntry
     * @brief One entry in the table of contents describing a sub-model. 64 bytes.
     */
    struct SubModelTopologyEntry {
        /** @brief Role of this sub-model (SubModelRole). */
        uint32_t subModelRole;

        /** @brief Zero-based index within its role group. */
        uint32_t subModelIndex;

        /** @brief Number of input features. */
        int32_t inputDimensionSize;

        /** @brief Number of output features. */
        int32_t outputDimensionSize;

        /** @brief Number of hidden neurons per layer. */
        int32_t hiddenNeuronCount;

        /** @brief Number of hidden layers. */
        int32_t hiddenLayerCount;

        /** @brief Activation function type (ActivationFunctionType). */
        uint32_t activationFunctionType;

        /** @brief Padding for alignment. */
        uint32_t reservedPadding;

        /** @brief Byte offset relative to weightBlobByteOffset for weights. */
        uint64_t weightByteOffsetRelative;

        /** @brief Number of doubles in the weight array. */
        uint64_t weightElementCount;

        /** @brief Byte offset relative to weightBlobByteOffset for biases. */
        uint64_t biasesByteOffsetRelative;

        /** @brief Number of doubles in the bias array. */
        uint64_t biasElementCount;
    };
    static_assert(sizeof(SubModelTopologyEntry) == 64, "SubModelTopologyEntry must be exactly 64 bytes");

    /**
     * @brief Writes a model's header and table of contents to a .spartan file.
     *
     * This function writes the 48-byte header, followed by the table of contents
     * entries, followed by the raw weight blob, followed by a CRC-32 checksum.
     *
     * @param filePath                Null-terminated path to the output file.
     * @param modelTypeIdentifier     The top-level model type enum value.
     * @param topologyEntries         Read-only span of sub-model topology entries.
     * @param weightBlob              Read-only span of all concatenated weight doubles.
     * @return True on success, false on I/O failure.
     */
    bool saveModel(const char* filePath,
                   uint32_t modelTypeIdentifier,
                   std::span<const SubModelTopologyEntry> topologyEntries,
                   std::span<const double> weightBlob);

    /**
     * @brief Reads and validates a .spartan file header.
     *
     * @param filePath      Null-terminated path to the input file.
     * @param outHeader     Receives the parsed file header.
     * @return True on success and valid magic bytes, false otherwise.
     */
    bool loadHeader(const char* filePath,
                    SpartanFileHeader& outHeader);

    /**
     * @brief Loads the weight blob from a .spartan file into a pre-allocated buffer.
     *
     * @param filePath              Null-terminated path to the input file.
     * @param header                Previously loaded header for offset/size info.
     * @param targetWeightBuffer    Writable span where weights will be copied.
     * @return True on success and CRC-32 match, false otherwise.
     */
    bool loadWeights(const char* filePath,
                     const SpartanFileHeader& header,
                     std::span<double> targetWeightBuffer);

    /**
     * @brief Saves a model using its specialized persistence module.
     *
     * This function dispatches to the appropriate ModelPersistenceModule
     * based on the model type, serializes the model, and writes it to disk.
     *
     * @param filePath              Null-terminated path to output .spartan file.
     * @param model                 Pointer to the SpartanModel to save.
     * @param modelTypeIdentifier   The model type (MODEL_TYPE_RSAC, etc.)
     * @return True on success, false on failure.
     */
    bool saveModelWithModule(const char* filePath,
                            const org::spartan::internal::machinelearning::SpartanModel* model,
                            uint32_t modelTypeIdentifier);

    /**
     * @brief Loads a model using its specialized persistence module.
     *
     * This function reads the file, dispatches to the appropriate
     * ModelPersistenceModule, and deserializes the model.
     *
     * @param filePath              Null-terminated path to input .spartan file.
     * @param model                 Pointer to the SpartanModel to restore into.
     * @param modelTypeIdentifier   Expected model type (must match file header).
     * @return True on success, false on failure.
     */
    bool loadModelWithModule(const char* filePath,
                            org::spartan::internal::machinelearning::SpartanModel* model,
                            uint32_t modelTypeIdentifier);

}

