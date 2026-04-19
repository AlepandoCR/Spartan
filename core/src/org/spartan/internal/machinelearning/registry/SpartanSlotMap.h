//
// Created by Alepando on 19/3/2026.
//

#pragma once

#include <vector>
#include <cassert>
#include <cstdint>
#include <algorithm>

namespace org::spartan::internal::machinelearning {

/**
 * @class SpartanSlotMap
 * @brief O(1) insert, lookup, erase with cache-friendly iteration.
 *
 * Layout:
 * - Dense array: Contiguous storage [value_0, value_1, ...]
 * - Sparse array: ID → dense index mapping
 * - Generation: Versioning to detect use-after-free
 * - Free-list: Recycled indices for erase()
 *
 * Usage:
 *   SpartanSlotMap<SpartanAgent> registry;
 *   registry.insert(agentId, std::move(agent));
 *   auto* agent = registry.get(agentId);
 *   registry.erase(agentId);
 */
template<typename T>
class SpartanSlotMap {
private:
    static constexpr uint64_t INVALID_GEN = 0;
    static constexpr uint32_t INVALID_IDX = UINT32_MAX;

    struct Slot {
        T value;
        uint64_t generation;
    };

    // Dense storage (cache-friendly iteration)
    std::vector<Slot> dense_;

    // Sparse mapping: ID → dense index
    // Allocated large enough to handle any 64-bit ID
    std::vector<uint32_t> sparse_;

    // Generation tracking for each sparse slot
    std::vector<uint64_t> sparseGen_;

    // Free-list for recycled indices
    std::vector<uint32_t> freeList_;

    // Counter for generation versioning
    uint64_t generation_ = 1;

public:
    SpartanSlotMap() = default;
    ~SpartanSlotMap() = default;

    SpartanSlotMap(const SpartanSlotMap&) = delete;
    SpartanSlotMap& operator=(const SpartanSlotMap&) = delete;

    SpartanSlotMap(SpartanSlotMap&&) noexcept = default;
    SpartanSlotMap& operator=(SpartanSlotMap&&) noexcept = default;

    /**
     * Insert a new value with the given ID (O(1) amortized).
     * If ID already exists, does nothing.
     */
    void insert(uint64_t id, T&& value) {
        // Ensure sparse array is large enough
        if (id >= sparse_.size()) {
            sparse_.resize(id + 1, INVALID_IDX);
            sparseGen_.resize(id + 1, INVALID_GEN);
        }

        // Already exists
        if (sparse_[id] != INVALID_IDX && sparseGen_[id] == generation_) {
            return;
        }

        uint32_t denseIdx;
        if (!freeList_.empty()) {
            // Reuse recycled index
            denseIdx = freeList_.back();
            freeList_.pop_back();
        } else {
            // Allocate new index
            denseIdx = static_cast<uint32_t>(dense_.size());
            dense_.emplace_back();
        }

        dense_[denseIdx].value = std::move(value);
        dense_[denseIdx].generation = generation_;

        sparse_[id] = denseIdx;
        sparseGen_[id] = generation_;
    }

    /**
     * Get pointer to value (O(1), returns nullptr if not found).
     */
    T* get(uint64_t id) {
        if (id >= sparse_.size()) {
            return nullptr;
        }

        uint32_t denseIdx = sparse_[id];
        if (denseIdx == INVALID_IDX) {
            return nullptr;
        }

        if (sparseGen_[id] != generation_ || denseIdx >= dense_.size()) {
            return nullptr;
        }

        if (dense_[denseIdx].generation != generation_) {
            return nullptr;
        }

        return &dense_[denseIdx].value;
    }

    /**
     * Get const pointer to value (O(1), returns nullptr if not found).
     */
    const T* get(uint64_t id) const {
        if (id >= sparse_.size()) {
            return nullptr;
        }

        uint32_t denseIdx = sparse_[id];
        if (denseIdx == INVALID_IDX) {
            return nullptr;
        }

        if (sparseGen_[id] != generation_ || denseIdx >= dense_.size()) {
            return nullptr;
        }

        if (dense_[denseIdx].generation != generation_) {
            return nullptr;
        }

        return &dense_[denseIdx].value;
    }

    /**
     * Erase value by ID (O(1)).
     */
    void erase(uint64_t id) {
        if (id >= sparse_.size()) {
            return;
        }

        uint32_t denseIdx = sparse_[id];
        if (denseIdx == INVALID_IDX) {
            return;
        }

        // Mark as free
        sparse_[id] = INVALID_IDX;
        dense_[denseIdx].value = T{};
        dense_[denseIdx].generation = INVALID_GEN;
        freeList_.push_back(denseIdx);
    }

    /**
     * Check if ID exists (O(1)).
     */
    bool contains(uint64_t id) const {
        return get(id) != nullptr;
    }

    /**
     * Iterate over all active values (O(n) where n = size).
     */
    template<typename Func>
    void forEach(Func callback) {
        for (auto& slot : dense_) {
            if (slot.generation == generation_) {
                callback(slot.value);
            }
        }
    }

    /**
     * Iterate over all active values (const version).
     */
    template<typename Func>
    void forEach(Func callback) const {
        for (const auto& slot : dense_) {
            if (slot.generation == generation_) {
                callback(slot.value);
            }
        }
    }

    [[nodiscard]] size_t denseSize() const {
        return dense_.size();
    }

    T* getDenseIfActive(size_t index) {
        if (index >= dense_.size()) {
            return nullptr;
        }
        auto& slot = dense_[index];
        if (slot.generation != generation_) {
            return nullptr;
        }
        return &slot.value;
    }

    const T* getDenseIfActive(size_t index) const {
        if (index >= dense_.size()) {
            return nullptr;
        }
        const auto& slot = dense_[index];
        if (slot.generation != generation_) {
            return nullptr;
        }
        return &slot.value;
    }

    [[nodiscard]] size_t size() const {
        size_t count = 0;
        for (const auto& slot : dense_) {
            if (slot.generation == generation_) {
                ++count;
            }
        }
        return count;
    }

    /**
     * Clear all entries.
     */
    void clear() {
        dense_.clear();
        sparse_.assign(sparse_.size(), INVALID_IDX);
        sparseGen_.assign(sparseGen_.size(), INVALID_GEN);
        freeList_.clear();
        generation_++;
    }
};

}  // namespace org::spartan::internal::machinelearning

