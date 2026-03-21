//
// Created by Alepando on 19/3/2026.
//

#include "SpartanMultiAgentGroup.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "SpartanAgent.h"
#include "SpartanCritic.h"

namespace org::spartan::internal::machinelearning {

SpartanMultiAgentGroup::SpartanMultiAgentGroup(
    const uint64_t groupId,
    const std::span<const double> sharedContext,
    const std::span<double> sharedActions,
    const int32_t stateSize,
    const int32_t actionSize,
    const int32_t maxAgents)
    : sharedContext_(sharedContext),
      sharedActions_(sharedActions),
      stateSize_(stateSize),
      actionSize_(actionSize),
      groupId_(groupId) {
    contextSubspans_.reserve(maxAgents);
    actionSubspans_.reserve(maxAgents);
}

SpartanMultiAgentGroup::~SpartanMultiAgentGroup() = default;

SpartanMultiAgentGroup::SpartanMultiAgentGroup(SpartanMultiAgentGroup&&) noexcept = default;
SpartanMultiAgentGroup& SpartanMultiAgentGroup::operator=(SpartanMultiAgentGroup&&) noexcept = default;

void SpartanMultiAgentGroup::addAgent(const uint64_t agentId, std::unique_ptr<SpartanAgent> agent) {
    const int32_t agentIdx = activeAgentCount_;

    auto contextSubspan = sharedContext_.subspan(
        agentIdx * stateSize_,
        stateSize_);
    auto actionSubspan = sharedActions_.subspan(
        agentIdx * actionSize_,
        actionSize_);

    contextSubspans_.push_back(contextSubspan);
    actionSubspans_.push_back(actionSubspan);

    // Retrieve the config pointer from the agent before rebind clears it (if we passed nullptr)
    // Actually, we can just pass agent->getOpaqueHyperparameterConfig() back into rebind.
    // Casting away const-ness is necessary because rebind accepts void* (generic).
    void* configPtr = const_cast<void*>(agent->getOpaqueHyperparameterConfig());

    agent->rebind(agentId, configPtr, agent->getModelWeightsMutable(), contextSubspan, actionSubspan);

    agentSlotMap_.insert(agentId, std::move(agent));

    activeAgentCount_++;
}

SpartanAgent* SpartanMultiAgentGroup::getAgent(const uint64_t agentId) {
    auto ptr = agentSlotMap_.get(agentId);
    return ptr ? ptr->get() : nullptr;
}

void SpartanMultiAgentGroup::removeAgent(const uint64_t agentId) {
    auto* agent = getAgent(agentId);
    if (agent) {
        agent->unbind();
        agentSlotMap_.erase(agentId);
    }
}

void SpartanMultiAgentGroup::processTick() {
    const auto denseSize = agentSlotMap_.denseSize();
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int32_t i = 0; i < static_cast<int32_t>(denseSize); ++i) {
        auto* agentPtr = agentSlotMap_.getDenseIfActive(static_cast<size_t>(i));
        if (agentPtr && *agentPtr) {
            (*agentPtr)->processTick();
        }
    }

    if (globalCritic_) {
        globalCritic_->processTick();
    }
}

void SpartanMultiAgentGroup::applyRewardsToAll(const std::span<const double> rewards) {
    int idx = 0;
    agentSlotMap_.forEach([&](std::unique_ptr<SpartanAgent>& agentPtr) {
        if (agentPtr && idx < static_cast<int>(rewards.size())) {
            agentPtr->applyReward(rewards[idx++]);
        }
    });
}

int32_t SpartanMultiAgentGroup::getAgentCount() const {
    return static_cast<int32_t>(agentSlotMap_.size());
}

uint64_t SpartanMultiAgentGroup::getGroupId() const {
    return groupId_;
}

void SpartanMultiAgentGroup::setGlobalCritic(std::unique_ptr<SpartanCritic> critic) {
    globalCritic_ = std::move(critic);
}

std::span<const double> SpartanMultiAgentGroup::getSharedContext() const {
    return sharedContext_;
}

std::span<double> SpartanMultiAgentGroup::getSharedActions() {
    return sharedActions_;
}

}
