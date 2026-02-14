cmake_minimum_required(VERSION 3.28)
project(spartan_core CXX)

# --- Compiler Configuration ---

if(MSVC)
    # MSVC (Visual Studio) Configuration
    add_compile_options(/std:c++latest /utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)

    # Note: MSVC still struggles with the ':' syntax in contracts as of Feb 2026.
    # Keep using the Macros (Require/Ensure) defined in SpartanCore.hpp.

elseif(MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # MinGW (GCC) Configuration

    # 1. Enforce C++26 Standard
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # 2. Enable C++26 Contracts (Experimental)
    # This flag is required to recognize [[pre:]] and [[post:]] attributes
    add_compile_options(-fcontracts)

    # 3. Static Linking (CRITICAL for Minecraft)
    # This bakes the C++ runtime into your DLL so Java doesn't crash looking for them.
    add_link_options(-static-libgcc -static-libstdc++ -static)

    # 4. Optimizations for AI/DQN (Optional but recommended)
    # add_compile_options(-O3 -march=native)

    message(STATUS "Configuring for MinGW with C++26 and Contracts support.")
endif()

# --- Source Definitions ---

set(SRC_PATH src/org/spartan/core)

add_library(spartan_core SHARED
        ${SRC_PATH}/SpartanAPI.cpp
        ${SRC_PATH}/SpartanCore.hpp
)

# --- Include Directories ---

target_include_directories(spartan_core PRIVATE ${SRC_PATH})

# --- Export Configuration ---

set_target_properties(spartan_core PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)