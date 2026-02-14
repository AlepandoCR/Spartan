#include "SpartanCore.hpp"

using namespace org::spartan::core;

// Global instance of the engine to maintain state (DQN memory, etc.)
static SpartanEngine engine;

extern "C" {

    /**
     * Native function exported to Java.
     * Receives a C-String (pointer) and passes it to the C++ Core.
     * @param msg A null-terminated C string (UTF-8).
     */
    __declspec(dllexport) void spartan_log(const char* msg) {
        if (msg == nullptr) {
            engine.log("[Error] Received null message pointer.");
            return;
        }
        engine.log(msg);
    }

    /**
     * Initializes the native backend.
     * Called once when the plugin enables.
     */
    __declspec(dllexport) void spartan_init() {
        std::println("[Spartan-Core] Detected C++ Spartan Core...");
    }
}