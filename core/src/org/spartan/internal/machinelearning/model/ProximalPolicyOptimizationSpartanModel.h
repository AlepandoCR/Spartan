#pragma once

#include <span>
#include <vector>
#include <cstdint>

#include "SpartanAgent.h"
#include "../ModelHyperparameterConfig.h"
#include "../network/ProximalPolicyOptimizationActorNetwork.h"
#include "../network/ProximalPolicyOptimizationCriticNetwork.h"
#include "../trace/GeneralizedAdvantageBuffer.h"
#include "internal/math/tensor/SpartanTensorMath.h"

namespace org::spartan::internal::machinelearning {

    using namespace org::spartan::internal::math::tensor;
    using namespace network;
    using namespace trace;

    class ProximalPolicyOptimizationSpartanModel final : public SpartanAgent {
    public:
        ProximalPolicyOptimizationSpartanModel(
                uint64_t agentIdentifier,
                void* opaqueHyperparameterConfig,
                std::span<double> modelWeights,
                std::span<const double> contextBuffer,
                std::span<double> actionOutputBuffer,
                std::span<double> actorNetworkWeights,
                std::span<double> actorNetworkBiases,
                std::span<double> criticNetworkWeights,
                std::span<double> criticNetworkBiases);

        ~ProximalPolicyOptimizationSpartanModel() override = default;

        ProximalPolicyOptimizationSpartanModel(ProximalPolicyOptimizationSpartanModel&&) noexcept = default;
        ProximalPolicyOptimizationSpartanModel& operator=(ProximalPolicyOptimizationSpartanModel&&) noexcept = default;

        void processTick() override;
        void applyReward(double rewardSignal) override;
        void decayExploration() override;

        [[nodiscard]] std::span<const double> getCriticWeights() const noexcept override;
        [[nodiscard]] std::span<double> getCriticWeightsMutable() noexcept override;

        [[nodiscard]] std::span<const double> getActorWeights() const noexcept;
        [[nodiscard]] std::span<double> getActorWeightsMutable() const noexcept;

        [[nodiscard]] std::span<const double> getActorBiases() const noexcept;
        [[nodiscard]] std::span<double> getActorBiasesMutable() const noexcept;

        [[nodiscard]] std::span<const double> getCriticNetworkWeights() const noexcept;
        [[nodiscard]] std::span<double> getCriticNetworkWeightsMutable() const noexcept;

        [[nodiscard]] std::span<const double> getCriticBiases() const noexcept;
        [[nodiscard]] std::span<double> getCriticBiasesMutable() const noexcept;

        [[nodiscard]] static int32_t getDebugScalarCount() noexcept;
        [[nodiscard]] int32_t copyDebugScalars(std::span<double> outputBuffer) const noexcept;

    private:
        [[nodiscard]] const ProximalPolicyOptimizationHyperparameterConfig* typedConfig() const noexcept {
            return static_cast<const ProximalPolicyOptimizationHyperparameterConfig*>(
                opaqueHyperparameterConfig_);
        }

        void executeTrainingUpdate();

        ProximalPolicyOptimizationActorNetwork actorNetwork_;
        ProximalPolicyOptimizationCriticNetwork criticNetwork_;
        GeneralizedAdvantageBuffer trajectoryBuffer_;

        int32_t trainingStepCounter_ = 0;
        int32_t ticksSinceLastUpdate_ = 0;
        double currentExplorationRate_ = 1.0;

        std::vector<double> previousStateSnapshot_;
        bool hasPreviousState_ = false;

        std::vector<double> scratchpadA_;
        std::vector<double> scratchpadB_;

        std::vector<double> actorMeansBuffer_;
        std::vector<double> actorLogStdDevsBuffer_;
        double criticValueBuffer_ = 0.0;

        std::vector<double> logProbsNew_;
        std::vector<double> logProbsOld_;
        std::vector<double> probabilityRatios_;
        std::vector<double> tdErrors_;
        std::vector<double> advantages_;
        std::vector<double> surrogateLosses_;
        std::vector<double> valueLosses_;

        std::vector<double> adamActorWeightMomentum_;
        std::vector<double> adamActorWeightVelocity_;
        std::vector<double> adamActorBiasMomentum_;
        std::vector<double> adamActorBiasVelocity_;
        std::vector<double> adamCriticWeightMomentum_;
        std::vector<double> adamCriticWeightVelocity_;
        std::vector<double> adamCriticBiasMomentum_;
        std::vector<double> adamCriticBiasVelocity_;

        std::span<double> criticWeightsSpan_;
        std::span<double> actorNetworkWeights_;
        std::span<double> actorNetworkBiases_;
        std::span<double> criticNetworkWeights_;
        std::span<double> criticNetworkBiases_;
    };

}