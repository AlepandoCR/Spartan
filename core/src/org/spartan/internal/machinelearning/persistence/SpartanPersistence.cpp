//
// Created by Alepando on 10/3/2026.
//

#include "SpartanPersistence.h"

#include <fstream>
#include <cstring>
#include <vector>

namespace org::spartan::internal::machinelearning::persistence {

    /**
     * Static 256-entry lookup table for CRC-32/ISO-HDLC (polynomial 0xEDB88320).
     *
     * Using a pre-computed table reduces the cost per byte from 8 conditional
     * branches (64 cycles worst-case) to a single table lookup plus XOR and shift
     * (approximately 4 cycles per byte). The table is generated at compile time
     * and lives in the read-only data segment.
     */
    static constexpr uint32_t generateCrc32TableEntry(const uint32_t index) {
        uint32_t remainder = index;
        for (int bitIndex = 0; bitIndex < 8; ++bitIndex) {
            remainder = (remainder & 1)
                ? (remainder >> 1) ^ 0xEDB88320
                : (remainder >> 1);
        }
        return remainder;
    }

    static constexpr auto generateCrc32LookupTable() {
        struct Crc32LookupTable { uint32_t entries[256]; };
        Crc32LookupTable table{};
        for (uint32_t byteValue = 0; byteValue < 256; ++byteValue) {
            table.entries[byteValue] = generateCrc32TableEntry(byteValue);
        }
        return table;
    }

    /** @brief Compile-time generated CRC-32 lookup table (256 entries, read-only). */
    static constexpr auto CRC32_LOOKUP_TABLE = generateCrc32LookupTable();

    /**
     * Computes a CRC-32 checksum over an arbitrary byte buffer using the lookup table.
     *
     * @param dataPointer  Pointer to the byte buffer.
     * @param byteLength   Number of bytes to process.
     * @return The finalised CRC-32 value.
     */
    static uint32_t computeCrc32(const uint8_t* dataPointer, const size_t byteLength) {
        uint32_t crcAccumulator = 0xFFFFFFFF;
        for (size_t byteIndex = 0; byteIndex < byteLength; ++byteIndex) {
            const auto lookupIndex = static_cast<uint8_t>(crcAccumulator ^ dataPointer[byteIndex]);
            crcAccumulator = (crcAccumulator >> 8) ^ CRC32_LOOKUP_TABLE.entries[lookupIndex];
        }
        return ~crcAccumulator;
    }

    /**
     * Accumulates a CRC-32 checksum over a byte buffer without finalizing.
     *
     * This variant accepts and returns the raw (non-inverted) accumulator so that
     * multiple data segments can be checksummed sequentially without re-reading.
     *
     * @param crcAccumulator The running CRC accumulator (pass 0xFFFFFFFF for the first call).
     * @param dataPointer    Pointer to the byte buffer.
     * @param byteLength     Number of bytes to process.
     * @return The updated (non-finalised) CRC accumulator.
     */
    static inline uint32_t accumulateCrc32(
            uint32_t crcAccumulator,
            const uint8_t* dataPointer,
            const size_t byteLength) {
        for (size_t byteIndex = 0; byteIndex < byteLength; ++byteIndex) {
            const auto lookupIndex = static_cast<uint8_t>(crcAccumulator ^ dataPointer[byteIndex]);
            crcAccumulator = (crcAccumulator >> 8) ^ CRC32_LOOKUP_TABLE.entries[lookupIndex];
        }
        return crcAccumulator;
    }

    bool saveModel(
            const char* filePath,
            const uint32_t modelTypeIdentifier,
            const std::span<const SubModelTopologyEntry> topologyEntries,
            const std::span<const double> weightBlob) {

        std::ofstream outputStream(filePath, std::ios::binary | std::ios::trunc);
        if (!outputStream.is_open()) {
            return false;
        }

        // Build the file header
        const uint64_t tableOfContentsByteSize =
            topologyEntries.size() * sizeof(SubModelTopologyEntry);
        const uint64_t weightBlobByteSize =
            weightBlob.size() * sizeof(double);

        SpartanFileHeader header{};
        std::memcpy(header.magicBytes, SPARTAN_FILE_MAGIC_BYTES, 4);
        header.formatVersion = SPARTAN_FILE_FORMAT_VERSION;
        header.formatFlags = 0;
        header.modelTypeIdentifier = modelTypeIdentifier;
        header.subModelCount = static_cast<uint32_t>(topologyEntries.size());
        header.totalWeightCount = weightBlob.size();
        header.weightBlobByteOffset = sizeof(SpartanFileHeader) + tableOfContentsByteSize;
        header.weightBlobTotalByteSize = weightBlobByteSize;
        header.reservedPadding = 0;

        // Write the 48-byte header
        outputStream.write(
            reinterpret_cast<const char*>(&header), sizeof(SpartanFileHeader));

        // Write the table of contents entries
        outputStream.write(
            reinterpret_cast<const char*>(topologyEntries.data()),
            static_cast<std::streamsize>(tableOfContentsByteSize));

        // Write the raw weight blob
        outputStream.write(
            reinterpret_cast<const char*>(weightBlob.data()),
            static_cast<std::streamsize>(weightBlobByteSize));

        // Compute CRC-32 by accumulating across all three segments using the lookup table.
        // Each call processes its segment in a single sequential pass.
        uint32_t runningCrc = 0xFFFFFFFF;
        runningCrc = accumulateCrc32(runningCrc,
            reinterpret_cast<const uint8_t*>(&header),
            sizeof(SpartanFileHeader));
        runningCrc = accumulateCrc32(runningCrc,
            reinterpret_cast<const uint8_t*>(topologyEntries.data()),
            tableOfContentsByteSize);
        runningCrc = accumulateCrc32(runningCrc,
            reinterpret_cast<const uint8_t*>(weightBlob.data()),
            weightBlobByteSize);
        runningCrc = ~runningCrc;

        // Write the 4-byte CRC-32 checksum
        outputStream.write(reinterpret_cast<const char*>(&runningCrc), sizeof(uint32_t));

        return outputStream.good();
    }

    bool loadHeader(const char* filePath, SpartanFileHeader& outHeader) {
        std::ifstream inputStream(filePath, std::ios::binary);
        if (!inputStream.is_open()) {
            return false;
        }

        inputStream.read(reinterpret_cast<char*>(&outHeader), sizeof(SpartanFileHeader));
        if (!inputStream.good()) {
            return false;
        }

        // Validate magic bytes
        if (std::memcmp(outHeader.magicBytes, SPARTAN_FILE_MAGIC_BYTES, 4) != 0) {
            return false;
        }

        // Validate format version
        if (outHeader.formatVersion > SPARTAN_FILE_FORMAT_VERSION) {
            return false;
        }

        return true;
    }

    bool loadWeights(
            const char* filePath,
            const SpartanFileHeader& header,
            const std::span<double> targetWeightBuffer) {

        std::ifstream inputStream(filePath, std::ios::binary);
        if (!inputStream.is_open()) {
            return false;
        }

        // Validate that the target buffer can hold all weights
        if (targetWeightBuffer.size() < header.totalWeightCount) {
            return false;
        }

        // Seek to the weight blob offset
        inputStream.seekg(static_cast<std::streamoff>(header.weightBlobByteOffset));
        if (!inputStream.good()) {
            return false;
        }

        // Read the weight data directly into the target buffer
        inputStream.read(
            reinterpret_cast<char*>(targetWeightBuffer.data()),
            static_cast<std::streamsize>(header.weightBlobTotalByteSize));

        if (!inputStream.good()) {
            return false;
        }

        // Read and validate the CRC-32 checksum
        uint32_t storedChecksum = 0;
        inputStream.read(reinterpret_cast<char*>(&storedChecksum), sizeof(uint32_t));

        // Rewind and compute CRC over the full payload (header + TOC + weights)
        const size_t totalPayloadSize =
            sizeof(SpartanFileHeader) +
            (header.subModelCount * sizeof(SubModelTopologyEntry)) +
            header.weightBlobTotalByteSize;

        std::vector<uint8_t> fullPayload(totalPayloadSize);
        inputStream.seekg(0);
        inputStream.read(reinterpret_cast<char*>(fullPayload.data()),
                         static_cast<std::streamsize>(totalPayloadSize));

        const uint32_t computedChecksum = computeCrc32(fullPayload.data(), totalPayloadSize);

        return computedChecksum == storedChecksum;
    }

}





