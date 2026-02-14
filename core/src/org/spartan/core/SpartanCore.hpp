#pragma once
#include <string_view>
#include <print>

namespace org::spartan::core {

    class SpartanEngine {

    public:
        /**
         * Logs a message to the console using C++26 std::print.
         * * @param message The string view to log.
         * @contract Require(message.length() > 0) - Message must not be empty.
         */
        void log(std::string_view message) {
            std::println("[Spartan-Core] >> {}", message);
        }

    };

}