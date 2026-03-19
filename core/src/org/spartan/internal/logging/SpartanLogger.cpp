//
// Created by Alepando on 25/2/2026.
//

#include "SpartanLogger.h"

#include <cstdio>

namespace org::spartan::internal::logging {

    std::atomic<bool> SpartanLogger::debugEnabled_{true};

    void SpartanLogger::setDebugEnabled(const bool enabled) {
        debugEnabled_.store(enabled, std::memory_order_relaxed);
    }

    bool SpartanLogger::isDebugEnabled() {
        return debugEnabled_.load(std::memory_order_relaxed);
    }

    void SpartanLogger::info(const std::string_view message) {
        std::fprintf(stderr, "[Spartan-Core] >> %.*s\n",
                     static_cast<int>(message.size()), message.data());
    }

    void SpartanLogger::error(const std::string_view message) {
        std::fprintf(stderr, "[Spartan-Core] [ERROR] >> %.*s\n",
                     static_cast<int>(message.size()), message.data());
    }

    void SpartanLogger::debug(const std::string_view message) {
        if (!debugEnabled_.load(std::memory_order_relaxed)) {
            return;
        }
        std::fprintf(stderr, "[Spartan-Core] [DEBUG] >> %.*s\n",
                     static_cast<int>(message.size()), message.data());
    }

    void SpartanLogger::warn(const std::string_view message) {
        std::fprintf(stderr, "[Spartan-Core] [WARN] >> %.*s\n",
                     static_cast<int>(message.size()), message.data());
    }



} // namespace org::spartan::core::logging
