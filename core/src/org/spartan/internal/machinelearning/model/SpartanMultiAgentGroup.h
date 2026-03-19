//
// Created by Alepando on 19/3/2026.
//

#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <memory>

#include "../registry/SpartanSlotMap.h"

namespace org::spartan::internal::machinelearning {

class SpartanAgent;
class SpartanCritic;

/**
 * @class SpartanMultiAgentGroup
 * @brief Container for N homogeneous agents sharing one context.
 *
 * Architecture:
 * - sharedContext: [Agent_0_State | Agent_1_State | ... | Agent_N-1_State]
 *   All agents READ from this SAME MemorySegment (different subspans)
 *
 * - sharedActions: [Agent_0_Actions | Agent_1_Actions | ... | Agent_N-1_Actions]
 *   All agents WRITE to this SAME MemorySegment (different subspans)
 *
 * - agentSlotMap: O(1) lookup by agent ID
 *   Stores unique_ptr<SpartanAgent>
 *
 * - globalCritic: Evaluates JointContext + JointAction
 *   Receives FULL context and action buffers
 */
class SpartanMultiAgentGroup {
private:
    // Shared buffers (Java-owned, stored as references)
    std::span<const double> sharedContext_;    // [stateSize * N]
    std::span<double> sharedActions_;          // [actionSize * N]

    // Agent storage (O(1) lookup)
    SpartanSlotMap<std::unique_ptr<SpartanAgent>> agentSlotMap_;

    // Subspans for each agent (cached for performance)
    std::vector<std::span<const double>> contextSubspans_;
    std::vector<std::span<double>> actionSubspans_;

    // Configuration
    int32_t stateSize_;
    int32_t actionSize_;
    uint64_t groupId_;

    // Optional: Global critic for CTDE
    std::unique_ptr<SpartanCritic> globalCritic_;

    // Agent count (for indexing subspans)
    int32_t activeAgentCount_ = 0;

public:
    /**
     * Constructor.
     *
     * @param groupId               Unique identifier for this group
     * @param sharedContext         Reference to Java-owned context buffer [stateSize * N]
     * @param sharedActions         Reference to Java-owned action buffer [actionSize * N]
     * @param stateSize             Size of context per agent
     * @param actionSize            Size of actions per agent
     * @param maxAgents             Pre-allocated capacity
     */
    SpartanMultiAgentGroup(
        uint64_t groupId,
        std::span<const double> sharedContext,
        std::span<double> sharedActions,
        int32_t stateSize,
        int32_t actionSize,
        int32_t maxAgents = 1024);

    ~SpartanMultiAgentGroup();

    SpartanMultiAgentGroup(const SpartanMultiAgentGroup&) = delete;
    SpartanMultiAgentGroup& operator=(const SpartanMultiAgentGroup&) = delete;

    SpartanMultiAgentGroup(SpartanMultiAgentGroup&&) noexcept;
    SpartanMultiAgentGroup& operator=(SpartanMultiAgentGroup&&) noexcept;

    /**
     * Add an agent to the group (O(1)).
     *
     * @param agentId                Unique 64-bit agent identifier
     * @param agent                  Unique pointer to SpartanAgent
     */
    void addAgent(uint64_t agentId, std::unique_ptr<SpartanAgent> agent);

    /**
     * Get agent by ID (O(1)).
     */
    SpartanAgent* getAgent(uint64_t agentId);

    /**
     * Remove agent by ID (O(1)).
     */
    void removeAgent(uint64_t agentId);

    /**
     * Execute MARL CTDE tick.
     *
     * Flow:
     * 1. Parallelizable: Each agent.processTick()
     *    - Reads: contextSubspan[i]
     *    - Writes: actionSubspan[i]
     *
     * 2. Serializable: Global critic evaluation
     *    - Reads: FULL sharedContext + sharedActions
     *    - Computes: Q_global, gradients
     *
     * 3. Epilogue: Reward distribution
     */
    void processTick();

    /**
     * Apply rewards to all agents.
     *
     * @param rewards  Array of N reward signals
     */
    void applyRewardsToAll(std::span<const double> rewards);

    /**
     * Get number of active agents.
     */
    [[nodiscard]] int32_t getAgentCount() const;

    /**
     * Get group ID.
     */
    [[nodiscard]] uint64_t getGroupId() const;

    /**
     * Set global critic (optional).
     */
    void setGlobalCritic(std::unique_ptr<SpartanCritic> critic);

    /**
     * Get shared context buffer (for debugging).
     */
    [[nodiscard]] std::span<const double> getSharedContext() const;

    /**
     * Get shared action buffer (for debugging).
     */
    [[nodiscard]] std::span<double> getSharedActions();
};

}  // namespace org::spartan::internal::machinelearning

