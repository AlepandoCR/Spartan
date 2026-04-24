#pragma once

#include <span>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace org::spartan::internal::machinelearning::trace {

    class GeneralizedAdvantageBuffer {
    public:
        GeneralizedAdvantageBuffer() = default;

        void initialize(
            const int stateSize,
            const int actionSize,
            const int capacity
            ) {
            stateSize_ = stateSize;
            actionSize_ = actionSize;
            capacity_ = capacity;

            states_.resize(capacity * stateSize);
            actions_.resize(capacity * actionSize);
            rewards_.resize(capacity);
            logProbsOld_.resize(capacity);
            values_.resize(capacity);
            advantages_.resize(capacity);

            size_ = 0;
        }

        void recordTransition(
                const std::span<const double> state,
                const std::span<const double> action,
                const double logProbOld,
                const double valueOld) {

            if (size_ >= capacity_) return;

            const size_t idx = size_;
            std::ranges::copy(state,
                              states_.begin() + static_cast<std::ptrdiff_t>(idx * stateSize_)
                              );
            std::ranges::copy(action,
                              actions_.begin() + static_cast<std::ptrdiff_t>(idx * actionSize_)
                              );

            logProbsOld_[idx] = logProbOld;
            values_[idx] = valueOld;
            ++size_;
        }

        void finalizeStep(
            const double reward,
            const double nextValue,
            const bool isEpisodeDone
            ) {
            if (size_ == 0 || size_ > capacity_) return;

            const size_t idx = size_ - 1;
            rewards_[idx] = reward;

            if (isEpisodeDone) {
                advantages_[idx] = reward - values_[idx];
            } else {
                advantages_[idx] = reward + nextValue - values_[idx];
            }
        }

        std::span<const double> getStates() const {
            return std::span(states_.data(), size_ * stateSize_);
        }

        std::span<const double> getActions() const {
            return std::span(actions_.data(), size_ * actionSize_);
        }

        std::span<const double> getRewards() const {
            return std::span(rewards_.data(), size_);
        }

        std::span<const double> getLogProbsOld() const {
            return std::span(logProbsOld_.data(), size_);
        }

        std::span<const double> getValues() const {
            return std::span(values_.data(), size_);
        }

        std::span<const double> getAdvantages() const {
            return std::span(advantages_.data(), size_);
        }

        bool isFull() const {
            return size_ >= capacity_;
        }

        size_t size() const {
            return size_;
        }

        void reset() {
            size_ = 0;
        }

    private:
        std::vector<double> states_;
        std::vector<double> actions_;
        std::vector<double> rewards_;
        std::vector<double> logProbsOld_;
        std::vector<double> values_;
        std::vector<double> advantages_;

        int stateSize_ = 0;
        int actionSize_ = 0;
        size_t capacity_ = 0;
        size_t size_ = 0;
    };

}


