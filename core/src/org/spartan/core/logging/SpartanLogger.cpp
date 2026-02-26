//
// Created by Alepando on 25/2/2026.
//

#include "SpartanLogger.h"

#include <print>

namespace org::spartan::core::logging {

    void SpartanLogger::info(const std::string_view message) {
        std::println("[Spartan-Core] >> {}", message);
    }

    void SpartanLogger::error(const std::string_view message) {
        std::println("[Spartan-Core] [ERROR] >> {}", message);
    }

} // namespace org::spartan::core::logging

