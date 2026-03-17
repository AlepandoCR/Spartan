//
// Created by Alepando on 25/2/2026.
//

#include "SpartanLogger.h"

#include <print>

namespace org::spartan::internal::logging {

    std::atomic<bool> SpartanLogger::debugEnabled_{true};

    void SpartanLogger::setDebugEnabled(const bool enabled) {
        debugEnabled_.store(enabled, std::memory_order_relaxed);
    }

    bool SpartanLogger::isDebugEnabled() {
        return debugEnabled_.load(std::memory_order_relaxed);
    }

    void SpartanLogger::info(const std::string_view message) {
        std::println("[Spartan-Core] >> {}", message);
    }

    void SpartanLogger::error(const std::string_view message) {
        std::println("[Spartan-Core] [ERROR] >> {}", message);
    }

    void SpartanLogger::debug(const std::string_view message) {
        if (!debugEnabled_.load(std::memory_order_relaxed)) {
            return;
        }
        std::println("[Spartan-Core] [DEBUG] >> {}", message);
    }

    void SpartanLogger::warn(const std::string_view message) {
        std::println("[Spartan-Core] [WARN] >> {}", message);
    }



} // namespace org::spartan::core::logging
