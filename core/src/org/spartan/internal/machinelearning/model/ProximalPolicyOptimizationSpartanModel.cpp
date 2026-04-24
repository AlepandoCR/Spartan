#include "ProximalPolicyOptimizationSpartanModel.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace org::spartan::internal::machinelearning {

    ProximalPolicyOptimizationSpartanModel::ProximalPolicyOptimizationSpartanModel(
            const uint64_t agentIdentifier,
            void* opaqueHyperparameterConfig,
            const std::span<double> modelWeights,
            const std::span<const double> contextBuffer,
            const std::span<double> actionOutputBuffer,
            const std::span<double> actorNetworkWeights,
            const std::span<double> actorNetworkBiases,
            const std::span<double> criticNetworkWeights,
            const std::span<double> criticNetworkBiases)
        : SpartanAgent(
            agentIdentifier,
            opaqueHyperparameterConfig,
            modelWeights,
            contextBuffer,
            actionOutputBuffer
            ),
          actorNetwork_(actorNetworkWeights, actorNetworkBiases),
          criticWeightsSpan_(criticNetworkWeights),
          actorNetworkWeights_(actorNetworkWeights),
          criticNetworkWeights_(criticNetworkWeights) {

        auto* config = typedConfig();

        criticNetwork_.initialize(criticNetworkWeights, criticNetworkBiases);

        trajectoryBuffer_.initialize(
            config->baseConfig.stateSize,
            config->baseConfig.actionSize,
            config->trajectoryBufferCapacity);

        const int maxHiddenSize = std::max(
            config->actorHiddenNeuronCount,
            config->criticHiddenNeuronCount
            );
        const int maxInputSize = config->baseConfig.stateSize;

        scratchpadA_.resize(std::max(maxHiddenSize, maxInputSize));
        scratchpadB_.resize(std::max(maxHiddenSize, maxInputSize));

        actorMeansBuffer_.resize(config->baseConfig.actionSize);
        actorLogStdDevsBuffer_.resize(config->baseConfig.actionSize);

        logProbsNew_.resize(config->miniBatchSize);
        logProbsOld_.resize(config->miniBatchSize);
        probabilityRatios_.resize(config->miniBatchSize);
        tdErrors_.resize(config->trajectoryBufferCapacity);
        advantages_.resize(config->trajectoryBufferCapacity);
        surrogateLosses_.resize(config->miniBatchSize);
        valueLosses_.resize(config->miniBatchSize);

        previousStateSnapshot_.resize(config->baseConfig.stateSize);

        adamActorWeightMomentum_.resize(actorNetworkWeights.size());
        adamActorWeightVelocity_.resize(actorNetworkWeights.size());
        adamActorBiasMomentum_.resize(actorNetworkBiases.size());
        adamActorBiasVelocity_.resize(actorNetworkBiases.size());
        adamCriticWeightMomentum_.resize(criticNetworkWeights.size());
        adamCriticWeightVelocity_.resize(criticNetworkWeights.size());
        adamCriticBiasMomentum_.resize(criticNetworkBiases.size());
        adamCriticBiasVelocity_.resize(criticNetworkBiases.size());

        currentExplorationRate_ = config->baseConfig.epsilon;
    }

    void ProximalPolicyOptimizationSpartanModel::processTick() {
        auto* config = typedConfig();

        criticValueBuffer_ = criticNetwork_.computeValueImpl(
            contextBuffer_,
            config,
            scratchpadA_,
            scratchpadB_);

        actorNetwork_.computePolicyOutputImpl(
            contextBuffer_,
            config,
            actorMeansBuffer_,
            actorLogStdDevsBuffer_,
            scratchpadA_,
            scratchpadB_);

        TensorOps::applyGaussianNoise(
            actorMeansBuffer_,
            actorLogStdDevsBuffer_,
            actionOutputBuffer_,
            agentIdentifier_);

        std::vector<double> logProbBuffer(1);
        TensorOps::computeGaussianLogProbabilities(
            actionOutputBuffer_,
            actorMeansBuffer_,
            actorLogStdDevsBuffer_,
            config->baseConfig.actionSize,
            1,
            logProbBuffer);

        previousStateSnapshot_.assign(
            contextBuffer_.begin(),
            contextBuffer_.end());

        trajectoryBuffer_.recordTransition(
            previousStateSnapshot_,
            actionOutputBuffer_,
            logProbBuffer[0],
            criticValueBuffer_);

        hasPreviousState_ = true;
    }

    void ProximalPolicyOptimizationSpartanModel::applyReward(const double rewardSignal) {
        auto* config = typedConfig();

        if (!hasPreviousState_) return;

        const double nextValue = criticNetwork_.computeValueImpl(
            contextBuffer_, config, scratchpadA_, scratchpadB_);

        trajectoryBuffer_.finalizeStep(rewardSignal, nextValue, false);

        ++ticksSinceLastUpdate_;

        if (trajectoryBuffer_.isFull() || ticksSinceLastUpdate_ >= config->trajectoryBufferCapacity) {
            executeTrainingUpdate();
            trajectoryBuffer_.reset();
            ticksSinceLastUpdate_ = 0;
        }
    }

    void ProximalPolicyOptimizationSpartanModel::decayExploration() {
        auto* config = typedConfig();
        currentExplorationRate_ *= config->baseConfig.epsilonDecay;
        currentExplorationRate_ = std::max(currentExplorationRate_, config->baseConfig.epsilonMin);
    }

    void ProximalPolicyOptimizationSpartanModel::executeTrainingUpdate() {
        auto* config = typedConfig();

        const auto rewards     = trajectoryBuffer_.getRewards();
        const auto values      = trajectoryBuffer_.getValues();
        const auto logProbsOld = trajectoryBuffer_.getLogProbsOld();
        const auto states      = trajectoryBuffer_.getStates();
        const auto actions     = trajectoryBuffer_.getActions();

        const size_t trajectorySize = trajectoryBuffer_.size();
        if (trajectorySize == 0) return;

        const int stateSize  = config->baseConfig.stateSize;
        const int actionSize = config->baseConfig.actionSize;
        const int aHidden    = config->actorHiddenNeuronCount;
        const int aLayers    = config->actorHiddenLayerCount;
        const int cHidden    = config->criticHiddenNeuronCount;
        const int cLayers    = config->criticHiddenLayerCount;

        //  Bootstrap next-values
        std::vector<double> nextValues(trajectorySize);
        std::copy(values.begin() + 1, values.end(), nextValues.begin());
        nextValues[trajectorySize - 1] = 0.0;

        //  GAE + advantage normalisation
        TensorOps::computeTDErrors(
            std::span(rewards.data(), trajectorySize),
            std::span(values.data(), trajectorySize),
            std::span<const double>(nextValues.data(), trajectorySize),
            config->gaeGamma,
            std::span(tdErrors_.data(), trajectorySize));

        TensorOps::computeGeneralizedAdvantages(
            std::span<const double>(tdErrors_.data(), trajectorySize),
            config->gaeGamma,
            config->gaeLambda,
            std::span(advantages_.data(), trajectorySize));

        {
            const double meanAdv =
                std::accumulate(advantages_.begin(), advantages_.begin() + trajectorySize, 0.0)
                / static_cast<double>(trajectorySize);
            double varAdv = 0.0;
            for (size_t i = 0; i < trajectorySize; ++i) {
                const double d = advantages_[i] - meanAdv;
                varAdv += d * d;
            }
            varAdv /= static_cast<double>(trajectorySize);
            const double stdAdv = std::sqrt(varAdv + 1e-8);
            for (size_t i = 0; i < trajectorySize; ++i)
                advantages_[i] = (advantages_[i] - meanAdv) / stdAdv;
        }

        // compute actor/critic trunk weight counts once
        // Actor trunk layers: input->h0, h0->h1, ...; then 2 output heads
        size_t actorTrunkWeightCount = 0;
        size_t actorTrunkBiasCount   = 0;
        for (int l = 0; l < aLayers; ++l) {
            const int inS = (l == 0) ? stateSize : aHidden;
            actorTrunkWeightCount += aHidden * inS;
            actorTrunkBiasCount   += aHidden;
        }
        // Each output head: aHidden -> actionSize
        const size_t actorHeadWeightCount = static_cast<size_t>(actionSize * aHidden);
        const size_t actorHeadBiasCount   = static_cast<size_t>(actionSize);

        // Critic trunk layers; then 1-neuron output head
        size_t criticTrunkWeightCount = 0;
        size_t criticTrunkBiasCount   = 0;
        for (int l = 0; l < cLayers; ++l) {
            const int inS = (l == 0) ? stateSize : cHidden;
            criticTrunkWeightCount += cHidden * inS;
            criticTrunkBiasCount   += cHidden;
        }
        // Output head: cHidden -> 1
        const size_t criticHeadWeightCount = static_cast<size_t>(cHidden);

        // Bias spans (mutable, JVM-owned memory)
        std::span<double> actorBiases  = actorNetwork_.getPolicyBiases();
        std::span<double> criticBiases = criticNetwork_.getBiasesMutable();

        // per-sample activation storage for backprop
        // postActs slot 0 holds the input copy, then one slot per trunk layer
        std::vector<double> actorPreActs(aLayers * aHidden);
        std::vector<double> actorPostActs(stateSize + aLayers * aHidden);
        std::vector<double> criticPreActs(cLayers * cHidden);
        std::vector<double> criticPostActs(stateSize + cLayers * cHidden);

        // training loop
        for (int epoch = 0; epoch < config->trainingEpochCount; ++epoch) {
            for (size_t batchStart = 0; batchStart < trajectorySize; batchStart += config->miniBatchSize) {
                const size_t batchEnd  = std::min(batchStart + static_cast<size_t>(config->miniBatchSize), trajectorySize);
                const size_t batchSize = batchEnd - batchStart;

                // Gradient accumulators — zeroed each mini-batch
                std::vector actorWeightGrads(actorNetworkWeights_.size(), 0.0);
                std::vector actorBiasGrads(actorBiases.size(), 0.0);
                std::vector criticWeightGrads(criticNetworkWeights_.size(), 0.0);
                std::vector criticBiasGrads(criticBiases.size(), 0.0);

                // Batch-level forward pass (means/log-stds for all samples)
                std::vector<double> batchActorMeans(batchSize * actionSize);
                std::vector<double> batchActorLogStds(batchSize * actionSize);

                for (size_t i = batchStart; i < batchEnd; ++i) {
                    const size_t bi = i - batchStart;
                    const auto state = states.subspan(i * stateSize, stateSize);
                    actorNetwork_.computePolicyOutputImpl(
                        state, config,
                        actorMeansBuffer_, actorLogStdDevsBuffer_,
                        scratchpadA_, scratchpadB_);
                    std::ranges::copy(actorMeansBuffer_,
                                      batchActorMeans.data() + bi * actionSize);
                    std::ranges::copy(actorLogStdDevsBuffer_,
                                      batchActorLogStds.data() + bi * actionSize);
                }

                // Compute log-probs, ratios, surrogate losses for the whole batch
                std::vector<double> batchLogProbsNew(batchSize);
                TensorOps::computeGaussianLogProbabilities(
                    std::span(actions.data() + batchStart * actionSize, batchSize * actionSize),
                    std::span<const double>(batchActorMeans.data(), batchActorMeans.size()),
                    std::span<const double>(batchActorLogStds.data(), batchActorLogStds.size()),
                    actionSize, static_cast<int>(batchSize), batchLogProbsNew);

                std::vector batchLogProbsOldVec(
                    logProbsOld.data() + batchStart,
                    logProbsOld.data() + batchEnd);

                std::vector<double> batchRatios(batchSize);
                TensorOps::computeProbabilityRatios(
                    std::span<const double>(batchLogProbsNew.data(), batchSize),
                    std::span<const double>(batchLogProbsOldVec.data(), batchSize),
                    std::span(batchRatios.data(), batchSize));

                std::vector<double> batchSurrogateLosses(batchSize);
                TensorOps::computeClippedSurrogateLoss(
                    std::span<const double>(batchRatios.data(), batchSize),
                    std::span<const double>(advantages_.data() + batchStart, batchSize),
                    config->clipRange,
                    std::span(batchSurrogateLosses.data(), batchSize));

                // Per-sample backprop
                for (size_t bi = 0; bi < batchSize; ++bi) {
                    const size_t i = batchStart + bi;
                    const auto state = states.subspan(i * stateSize, stateSize);
                    const double* means   = batchActorMeans.data()   + bi * actionSize;
                    const double* logStds = batchActorLogStds.data() + bi * actionSize;
                    const double* act     = actions.data() + i * actionSize;


                    // actor trunk forward with saved activations
                    // postActs[0..stateSize-1]  = input copy
                    // postActs[stateSize..]     = post-LeakyReLU per layer
                    // preActs[0..]              = pre-LeakyReLU per layer
                    {
                        std::ranges::copy(state, actorPostActs.begin());

                        size_t wOff = 0, bOff = 0;
                        size_t preOff = 0, postOff = stateSize;

                        for (int l = 0; l < aLayers; ++l) {
                            const int inS = (l == 0) ? stateSize : aHidden;
                            std::span<const double> layerIn(actorPostActs.data() + postOff - inS, inS);
                            std::span<const double> lw = actorNetworkWeights_.subspan(wOff, aHidden * inS);
                            std::span<const double> lb = actorBiases.subspan(bOff, aHidden);
                            std::span preOut(actorPreActs.data() + preOff, aHidden);
                            std::span postOut(actorPostActs.data() + postOff, aHidden);

                            TensorOps::denseForwardPass(layerIn, lw, lb, preOut);
                            std::ranges::copy(preOut, postOut.begin());
                            TensorOps::applyLeakyReLU(postOut, 0.01);

                            wOff    += aHidden * inS;
                            bOff    += aHidden;
                            preOff  += aHidden;
                            postOff += aHidden;
                        }
                    }

                    std::span<const double> trunkOut(
                        actorPostActs.data() + actorPostActs.size() - aHidden, aHidden);

                    // Compute per-sample gradients for both heads

                    // surrogate gradient w.r.t. log_prob:
                    //   L_clip = -min(r*A, clip(r)*A)   (already negated by computeClippedSurrogateLoss)
                    //   dL/d(log_prob) = -A*r  when unclipped, 0 when clipped
                    //   averaged over batchSize
                    const double r   = batchRatios[bi];
                    const double adv = advantages_[i];
                    const double clipMin = 1.0 - config->clipRange;
                    const double clipMax = 1.0 + config->clipRange;
                    const double dL_dLogProb = (r > clipMin && r < clipMax)
                        ? (-adv * r) / static_cast<double>(batchSize)
                        : 0.0;

                    std::vector<double> dL_dMean(actionSize);
                    std::vector<double> dL_dLogStd(actionSize);

                    for (int d = 0; d < actionSize; ++d) {
                        const double sigma  = std::exp(logStds[d]);
                        const double sigma2 = sigma * sigma;
                        const double diff   = act[d] - means[d];

                        // chain rule through Gaussian log-prob:
                        //   d(log_prob)/d(mu)    =  (a - mu) / sigma^2
                        //   d(log_prob)/d(logSig) = (a - mu)^2 / sigma^2 - 1
                        dL_dMean[d]   = dL_dLogProb * (diff / sigma2);

                        // entropy term: H = sum(log_std + 0.5*log(2*pi*e))
                        //   d(-beta*H)/d(log_std) = -beta / batchSize
                        dL_dLogStd[d] = dL_dLogProb * (diff * diff / sigma2 - 1.0)
                                      - config->entropyCoefficient / static_cast<double>(batchSize);
                    }

                    // Backprop through mean head (linear, no activation)
                    {
                        const size_t mwOff = actorTrunkWeightCount;
                        const size_t mbOff = actorTrunkBiasCount;
                        std::span<const double> hw = actorNetworkWeights_.subspan(mwOff, actorHeadWeightCount);
                        auto wg = std::span(actorWeightGrads.data() + mwOff, actorHeadWeightCount);
                        auto bg = std::span(actorBiasGrads.data()   + mbOff, actorHeadBiasCount);

                        std::vector trunkGradMean(aHidden, 0.0);
                        TensorOps::denseBackwardPass(
                            trunkOut,
                            std::span<const double>(dL_dMean.data(), actionSize),
                            hw,
                            wg,
                            std::span(trunkGradMean.data(), aHidden));

                        for (int d = 0; d < actionSize; ++d) bg[d] += dL_dMean[d];

                        //Backprop through log-std head
                        const size_t swOff = mwOff + actorHeadWeightCount;
                        const size_t sbOff = mbOff + actorHeadBiasCount;
                        std::span<const double> shw = actorNetworkWeights_.subspan(swOff, actorHeadWeightCount);
                        auto swg = std::span(actorWeightGrads.data() + swOff, actorHeadWeightCount);
                        auto sbg = std::span(actorBiasGrads.data()   + sbOff, actorHeadBiasCount);

                        std::vector trunkGradStd(aHidden, 0.0);
                        TensorOps::denseBackwardPass(
                            trunkOut,
                            std::span<const double>(dL_dLogStd.data(), actionSize),
                            shw, swg,
                            std::span(trunkGradStd.data(), aHidden));

                        for (int d = 0; d < actionSize; ++d) sbg[d] += dL_dLogStd[d];

                        // Sum gradients from both heads; propagate through trunk backwards
                        std::vector<double> trunkGrad(aHidden);
                        for (int h = 0; h < aHidden; ++h)
                            trunkGrad[h] = trunkGradMean[h] + trunkGradStd[h];

                        //  Trunk backprop
                        size_t wOff  = actorTrunkWeightCount;
                        size_t bOff  = actorTrunkBiasCount;
                        size_t preOff  = aLayers * aHidden;
                        size_t postOff = actorPostActs.size();

                        std::vector<double> currentGrad = trunkGrad;

                        for (int l = aLayers - 1; l >= 0; --l) {
                            const int inS  = (l == 0) ? stateSize : aHidden;
                            const int outS = aHidden;

                            wOff    -= outS * inS;
                            bOff    -= outS;
                            preOff  -= outS;
                            postOff -= outS;

                            // LeakyReLU backward
                            std::span<const double> preAct(actorPreActs.data() + preOff, outS);
                            for (int j = 0; j < outS; ++j)
                                currentGrad[j] *= (preAct[j] > 0.0) ? 1.0 : 0.01;

                            // Bias grad
                            std::span bgSlice(actorBiasGrads.data() + bOff, outS);
                            for (int j = 0; j < outS; ++j) bgSlice[j] += currentGrad[j];

                            // Weight grad + input grad
                            const size_t prevPostOff = postOff - inS;
                            std::span<const double> layerIn(actorPostActs.data() + prevPostOff, inS);
                            std::span<const double> lw = actorNetworkWeights_.subspan(wOff, outS * inS);
                            auto weight_gradients = std::span(actorWeightGrads.data() + wOff, outS * inS);

                            std::vector inputGrad(inS, 0.0);
                            TensorOps::denseBackwardPass(
                                layerIn,
                                std::span<const double>(currentGrad.data(), outS),
                                lw, weight_gradients,
                                std::span(inputGrad.data(), inS));

                            currentGrad.assign(inputGrad.begin(), inputGrad.end());
                        }
                    }

                    // backdrop
                    {
                        constexpr size_t criticHeadBiasCount = 1;
                        // Critic trunk forward saving activations
                        std::ranges::copy(state, criticPostActs.begin());

                        size_t wOff = 0, bOff = 0;
                        size_t preOff = 0, postOff = stateSize;

                        for (int l = 0; l < cLayers; ++l) {
                            const int inS = (l == 0) ? stateSize : cHidden;
                            std::span<const double> layerIn(criticPostActs.data() + postOff - inS, inS);
                            std::span<const double> lw = criticNetworkWeights_.subspan(wOff, cHidden * inS);
                            std::span<const double> lb = criticBiases.subspan(bOff, cHidden);
                            std::span preOut(criticPreActs.data() + preOff, cHidden);
                            std::span postOut(criticPostActs.data() + postOff, cHidden);

                            TensorOps::denseForwardPass(layerIn, lw, lb, preOut);
                            std::ranges::copy(preOut, postOut.begin());
                            TensorOps::applyLeakyReLU(postOut, 0.01);

                            wOff    += cHidden * inS;
                            bOff    += cHidden;
                            preOff  += cHidden;
                            postOff += cHidden;
                        }

                        std::span<const double> criticTrunkOut(
                            criticPostActs.data() + criticPostActs.size() - cHidden, cHidden);

                        // Output head forward (linear, 1 neuron)
                        double vNew = 0.0;
                        std::span vNewSpan(&vNew, 1);
                        std::span<const double> headW = criticNetworkWeights_.subspan(criticTrunkWeightCount, criticHeadWeightCount);
                        std::span<const double> headB = criticBiases.subspan(criticTrunkBiasCount, criticHeadBiasCount);
                        TensorOps::denseForwardPass(criticTrunkOut, headW, headB, vNewSpan);

                        // dL/dV = 2*(V_new - (A + V_old)) / batchSize
                        const double returnTarget = advantages_[i] + values[i];
                        const double dL_dVNew = 2.0 * (vNew - returnTarget) / static_cast<double>(batchSize);

                        // Output head backward
                        auto hwg = std::span(criticWeightGrads.data() + criticTrunkWeightCount, criticHeadWeightCount);
                        auto hbg = std::span(criticBiasGrads.data()   + criticTrunkBiasCount,   criticHeadBiasCount);
                        hbg[0] += dL_dVNew;

                        std::vector trunkGrad(cHidden, 0.0);
                        TensorOps::denseBackwardPass(
                            criticTrunkOut,
                            std::span(&dL_dVNew, 1),
                            headW, hwg,
                            std::span(trunkGrad.data(), cHidden));

                        // Trunk backprop
                        std::vector<double> currentGrad = trunkGrad;

                        size_t twOff   = criticTrunkWeightCount;
                        size_t tbOff   = criticTrunkBiasCount;
                        size_t tPreOff = cLayers * cHidden;
                        size_t tPostOff = criticPostActs.size();

                        for (int l = cLayers - 1; l >= 0; --l) {
                            const int inS  = (l == 0) ? stateSize : cHidden;
                            const int outS = cHidden;

                            twOff   -= outS * inS;
                            tbOff   -= outS;
                            tPreOff -= outS;
                            tPostOff -= outS;

                            std::span<const double> preAct(criticPreActs.data() + tPreOff, outS);
                            for (int j = 0; j < outS; ++j)
                                currentGrad[j] *= (preAct[j] > 0.0) ? 1.0 : 0.01;

                            std::span<double> bgSlice(criticBiasGrads.data() + tbOff, outS);
                            for (int j = 0; j < outS; ++j) bgSlice[j] += currentGrad[j];

                            const size_t prevPostOff = tPostOff - inS;
                            std::span<const double> layerIn(criticPostActs.data() + prevPostOff, inS);
                            std::span<const double> lw = criticNetworkWeights_.subspan(twOff, outS * inS);
                            std::span<double>       wg = std::span<double>(criticWeightGrads.data() + twOff, outS * inS);

                            std::vector<double> inputGrad(inS, 0.0);
                            TensorOps::denseBackwardPass(
                                layerIn,
                                std::span<const double>(currentGrad.data(), outS),
                                lw, wg,
                                std::span<double>(inputGrad.data(), inS));

                            currentGrad.assign(inputGrad.begin(), inputGrad.end());
                        }
                    }
                }  // end per-sample loop

                // adam step for actor and critic weights/biases
                TensorOps::applyAdamUpdate(
                    actorNetworkWeights_,
                    std::span<const double>(actorWeightGrads.data(), actorNetworkWeights_.size()),
                    adamActorWeightMomentum_,
                    adamActorWeightVelocity_,
                    config->baseConfig.learningRate,
                    0.9, 0.999, 1e-8,
                    trainingStepCounter_);

                TensorOps::applyAdamUpdate(
                    actorBiases,
                    std::span<const double>(actorBiasGrads.data(), actorBiases.size()),
                    adamActorBiasMomentum_,
                    adamActorBiasVelocity_,
                    config->baseConfig.learningRate,
                    0.9, 0.999, 1e-8,
                    trainingStepCounter_);

                TensorOps::applyAdamUpdate(
                    criticNetworkWeights_,
                    std::span<const double>(criticWeightGrads.data(), criticNetworkWeights_.size()),
                    adamCriticWeightMomentum_,
                    adamCriticWeightVelocity_,
                    config->baseConfig.learningRate,
                    0.9, 0.999, 1e-8,
                    trainingStepCounter_);

                TensorOps::applyAdamUpdate(
                    criticBiases,
                    std::span<const double>(criticBiasGrads.data(), criticBiases.size()),
                    adamCriticBiasMomentum_,
                    adamCriticBiasVelocity_,
                    config->baseConfig.learningRate,
                    0.9, 0.999, 1e-8,
                    trainingStepCounter_);

                ++trainingStepCounter_;
            }
        }
    }

    std::span<const double> ProximalPolicyOptimizationSpartanModel::getCriticWeights() const noexcept {
        return criticWeightsSpan_;
    }

    std::span<double> ProximalPolicyOptimizationSpartanModel::getCriticWeightsMutable() noexcept {
        return const_cast<double*>(criticWeightsSpan_.data()) ?
            std::span(const_cast<double*>(criticWeightsSpan_.data()), criticWeightsSpan_.size()) :
            std::span<double>();
    }

}