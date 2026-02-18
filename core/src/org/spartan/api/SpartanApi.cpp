#include "../core/SpartanCore.hpp"

using namespace org::spartan::core;

static SpartanEngine engine;

extern "C" {

    //TODO add contracts once c++26 is supported by all major compilers
    /**
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
     *  Check for bridged connecting to the engine. This can be used for health checks.
     */
    __declspec(dllexport) void spartan_init() {
        std::println("[Spartan-Core] Detected C++ Spartan Core...");
    }
}