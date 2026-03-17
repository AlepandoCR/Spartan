//
// Created by Alepando on 25/2/2026.
//

#pragma once

#include <string_view>
#include <atomic>

/**
 * @file SpartanLogger.h
 * @brief Centralized logging facility for the Spartan native engine.
 *
 * Every log message in the engine must go through this class so that
 * output format, destination, and filtering can be changed in a single
 * place without touching callers.
 *
 */
namespace org::spartan::internal::logging {

    /**
     * @class SpartanLogger
     * @brief Stateless, header-declared logger with a single static entry point.
     *
     * All formatting and prefixing is handled here.  Callers simply provide
     * the raw message text.
     */
    class SpartanLogger {
    public:
        /**
         * @brief Logs an informational message with the engine prefix.
         *
         * @param message The message body (UTF-8).  The "[Spartan-Core]"
         *                prefix is prepended automatically.
         */
        static void info(std::string_view message);

        /**
         * @brief Logs an error message with a distinguishable prefix.
         *
         * @param message The error description (UTF-8).
         */
        static void error(std::string_view message);

        /**
         * @brief Logs a debug message with a distinguishable prefix.
         *
         * @param message The debug information (UTF-8).
         */
        static void debug(std::string_view message);

        /**
         * @brief  warns a message with a distinguishable prefix.
         * @param message The warning information (UTF-8).
         */
        static void warn(std::string_view message);

        static void setDebugEnabled(bool enabled);
        static bool isDebugEnabled();
    private:
        static std::atomic<bool> debugEnabled_;
    };

} // namespace org::spartan::core::logging

