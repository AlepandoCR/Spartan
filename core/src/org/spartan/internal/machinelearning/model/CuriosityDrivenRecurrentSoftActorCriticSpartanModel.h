//
// Created by Alepando on 12/3/2026.
//

#pragma once

#include <memory>
#include <span>
#include <vector>
#include <cstdint>

#include "SpartanAgent.h"
#include "RecurrentSoftActorCriticSpartanModel.h"
#include "../ModelHyperparameterConfig.h"

namespace org::spartan::internal::machinelearning {

    // Alias for a unique_ptr that uses a custom C-style function for deletion
    using AlignedMemoryDeleter = void(*)(void*);

    class CuriosityDrivenRecurrentSoftActorCriticSpartanModel final : public SpartanAgent {
    public:
        CuriosityDrivenRecurrentSoftActorCriticSpartanModel(
                uint64_t agentIdentifier,
                void* opaqueHyperparameterConfig,
                std::span<double> modelWeights,
                std::span<const double> contextBuffer,
                std::span<double> actionOutputBuffer,
                std::span<double> recurrentSoftActorCriticCriticWeights,
                std::span<double> forwardDynamicsWeights,
                std::span<double> forwardDynamicsBiases,
                std::unique_ptr<RecurrentSoftActorCriticSpartanModel> internalRecurrentSoftActorCriticModel);

        ~CuriosityDrivenRecurrentSoftActorCriticSpartanModel() override = default;

        // Rule of 5: Move semantics are now automatically safe thanks to unique_ptr
        CuriosityDrivenRecurrentSoftActorCriticSpartanModel(CuriosityDrivenRecurrentSoftActorCriticSpartanModel&&) noexcept = default;
        CuriosityDrivenRecurrentSoftActorCriticSpartanModel& operator=(CuriosityDrivenRecurrentSoftActorCriticSpartanModel&&) noexcept = default;
        CuriosityDrivenRecurrentSoftActorCriticSpartanModel(const CuriosityDrivenRecurrentSoftActorCriticSpartanModel&) = delete;
        CuriosityDrivenRecurrentSoftActorCriticSpartanModel& operator=(const CuriosityDrivenRecurrentSoftActorCriticSpartanModel&) = delete;

        void processTick() override;
        void applyReward(double extrinsicReward) override;
        void decayExploration() override;

        [[nodiscard]] std::span<const double> getCriticWeights() const noexcept override;

        [[nodiscard]] std::span<double> getCriticWeightsMutable() noexcept override {
            return const_cast<double*>(criticWeightsSpan_.data()) ?
                std::span<double>(const_cast<double*>(criticWeightsSpan_.data()), criticWeightsSpan_.size()) : std::span<double>();
        }

    private:
        // Store a local copy of the Curiosity config to avoid dereferencing Java memory
        CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig localConfig_{};

        [[nodiscard]] const CuriosityDrivenRecurrentSoftActorCriticHyperparameterConfig* typedConfig() const noexcept {
            return &localConfig_;
        }

        void runForwardDynamicsNetworkInference();
        void trainForwardDynamicsNetwork(double predictionError);

        std::unique_ptr<RecurrentSoftActorCriticSpartanModel> internalRecurrentSoftActorCriticModel_;

        // Non-owning span over the full critic (target) weight buffer for persistence
        std::span<const double> criticWeightsSpan_;

        std::span<double> forwardDynamicsWeights_;
        std::span<double> forwardDynamicsBiases_;

        // =========================================================
        // STRICTLY ALIGNED MEMORY BLOCK FOR AVX2 (Safe move semantics)
        // =========================================================
        std::unique_ptr<void, AlignedMemoryDeleter> alignedScratchpadMemory_;

        // Mathematical buffers mapped directly to the aligned block
        std::span<double> previousStateBuffer_;
        std::span<double> previousActionBuffer_;
        std::span<double> predictedNextStateBuffer_;
        std::span<double> forwardNetworkInputBuffer_;
        std::span<double> forwardNetworkHiddenBuffer_;
        std::span<double> forwardNetworkOutputGradient_;
        std::span<double> forwardDynamicsHiddenActivationGradients_;
        std::span<double> forwardNetworkInputGradientDummy_;
        std::span<double> forwardDynamicsWeightGradients_;
        std::span<double> forwardDynamicsBiasGradients_;
        std::span<double> forwardWeightsFirstMoment_;
        std::span<double> forwardWeightsSecondMoment_;
        std::span<double> forwardBiasesFirstMoment_;
        std::span<double> forwardBiasesSecondMoment_;

        double lastIntrinsicReward_ = 0.0;
        bool hasValidPreviousTick_ = false;
        int32_t adamTimeStep_ = 0;

        mutable std::vector<double> fullCriticSaveBuffer_;
    };

}